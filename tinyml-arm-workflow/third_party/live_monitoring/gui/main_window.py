# gui/main_window.py (FIXED - hapus duplikasi, tampilkan semua output)
"""
Main window implementation for Sleep Stage Development Studio.
"""

import re
from typing import Optional, List
from PySide6.QtWidgets import (
    QMainWindow, QStatusBar, QWidget,
    QVBoxLayout, QLabel, QHBoxLayout
)
from PySide6.QtCore import Qt, Slot

from gui.toolbar import ToolBar
from gui.dashboard import DashboardPanel
from serial_port.serial_manager import SerialManager
from serial_port.serial_worker import SerialWorker
from serial_port.serial_parser import Packet
from serial_port.serial_protocol import Protocol, PacketType
from models.feature_model import FeatureModel
from models.prediction_model import PredictionModel
from logger.session_manager import SessionManager


class MainWindow(QMainWindow):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.serial_manager = SerialManager()
        self.session_manager = SessionManager(base_dir="recordings")
        self.serial_worker: Optional[SerialWorker] = None
        
        self._setup_window_properties()
        self._create_toolbar()
        self._create_dashboard()
        self._create_status_bar()
        self._connect_signals()
        
        self._update_connection_state(False)
        self._packet_counts = {pkt_type: 0 for pkt_type in PacketType}
    
    def _setup_window_properties(self) -> None:
        self.setWindowTitle("Sleep Stage Development Studio")
        self.setMinimumSize(1024, 768)
        self.resize(1400, 900)
        screen = self.screen().availableGeometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )
    
    def _create_toolbar(self) -> None:
        self.toolbar = ToolBar(self.serial_manager)
        self.addToolBar(self.toolbar)
    
    def _create_dashboard(self) -> None:
        self.dashboard = DashboardPanel()
        self.setCentralWidget(self.dashboard)
    
    def _create_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        
        self.status_indicator = QLabel("Disconnected")
        self.status_indicator.setObjectName("statusIndicator")
        self.status_indicator.setStyleSheet(
            "background-color: #e0e0e0; color: #505050; padding: 2px 8px; "
            "border-radius: 3px; font-weight: bold;"
        )
        
        self.port_label = QLabel("No port selected")
        self.port_label.setStyleSheet("padding: 2px 8px; color: #707070;")
        
        self.accel_counter = QLabel("ACCEL: 0")
        self.accel_counter.setStyleSheet("padding: 2px 8px; color: #e74c3c;")
        
        self.ppg_counter = QLabel("PPG: 0")
        self.ppg_counter.setStyleSheet("padding: 2px 8px; color: #27ae60;")
        
        self.feature_counter = QLabel("FEATURE: 0")
        self.feature_counter.setStyleSheet("padding: 2px 8px; color: #8e44ad;")
        
        self.total_counter = QLabel("Total: 0")
        self.total_counter.setStyleSheet("padding: 2px 8px; color: #2c3e50; font-weight: bold;")
        
        self.status_bar.addWidget(self.status_indicator)
        self.status_bar.addWidget(self.port_label)
        self.status_bar.addWidget(self.accel_counter)
        self.status_bar.addWidget(self.ppg_counter)
        self.status_bar.addWidget(self.feature_counter)
        self.status_bar.addWidget(self.total_counter)
        self.status_bar.addPermanentWidget(QLabel("Sleep Stage Development Studio v4.0"))
        
        self.setStatusBar(self.status_bar)
    
    def _connect_signals(self) -> None:
        self.toolbar.connection_requested.connect(self._handle_connection_request)
        self.toolbar.disconnection_requested.connect(self._handle_disconnection)
        self.toolbar.message_log.connect(self.dashboard.console.append)
        self.session_manager.recording_started.connect(self._on_recording_started)
        self.session_manager.recording_stopped.connect(self._on_recording_stopped)
        self.session_manager.duration_updated.connect(self._on_duration_updated)
        self.session_manager.error_occurred.connect(self._on_recording_error)
        self.toolbar.recording_started.connect(self._start_recording)
        self.toolbar.recording_stopped.connect(self._stop_recording)
    
    # ========== DATA RECEIVED (raw line dari UART) ==========
    @Slot(str)
    def _on_data_received(self, line: str) -> None:
        """
        Handle raw UART line - log & tampilkan di console.

        Tampilkan semua seperti picocom, TAPI kecualikan ACCEL/PPG agar
        console tidak spam. Fitur (F=), prediksi (SCORES/PREDICTION),
        status, dan log lain tetap tampil.
        """
        # Log raw line ke file recording (semua tetap tercatat di file)
        self.session_manager.log_raw_line(line)
        
        # Parse & routre data peak detection dari debug line board.
        # Board mengirim debug human-readable:
        #   RR[i]=<rr>ms (HR=<hr>bpm)
        #   PEAK_DEBUG: peaks=N, rr=M
        self._parse_peak_debug_line(line)
        
        # Identifikasi jenis packet untuk filtering
        packet_type = Protocol.identify_packet(line)
        
        # Jangan tampilkan ACCEL/PPG di console (spam)
        if packet_type in (PacketType.ACCEL, PacketType.PPG):
            return
        
        # Tampilkan sisanya seperti di picocom
        self.dashboard.console.append(line)

    def _parse_peak_debug_line(self, line: str) -> None:
        """Parses peak-detection debug lines from board into the peak tab."""
        try:
            # Format: RR[i]=<rr>ms (HR=<hr>bpm)
            m = re.search(r'RR\[\d+\]=(\d+)ms\s*\(HR=(\d+)bpm\)', line)
            if m:
                rr_ms = float(m.group(1))
                hr = float(m.group(2))
                self.dashboard.add_rr_interval(rr_ms)
                self.dashboard.add_hr(hr)
                return
        except (ValueError, AttributeError):
            pass
    
    # ========== PACKET RECEIVED ==========
    @Slot(object)
    def _on_packet_received(self, packet: Packet) -> None:
        """Route parsed packet to appropriate handler."""
        self._packet_counts[packet.packet_type] = self._packet_counts.get(packet.packet_type, 0) + 1
        self._update_counters()
        
        if packet.packet_type == PacketType.ACCEL:
            self._handle_accel_packet(packet)
        elif packet.packet_type == PacketType.PPG:
            self._handle_ppg_packet(packet)
        elif packet.packet_type == PacketType.FEATURE:
            self._handle_feature_packet(packet)
        elif packet.packet_type == PacketType.PRED:
            self._handle_prediction_packet(packet)
    
    # ========== PACKET HANDLERS ==========
    def _handle_accel_packet(self, packet: Packet) -> None:
        try:
            if len(packet.payload) >= 3:
                x = float(packet.payload[0])
                y = float(packet.payload[1])
                z = float(packet.payload[2])
                self.dashboard.accel_plot.update(x, y, z)
                self.session_manager.log_accel(x, y, z)
        except (ValueError, IndexError):
            pass
    
    def _handle_ppg_packet(self, packet: Packet) -> None:
        try:
            if len(packet.payload) >= 1:
                value = float(packet.payload[0])
                self.dashboard.ppg_plot.update(value)
                self.dashboard.add_ppg_sample(value)  # Peak analyzer
                self.session_manager.log_ppg(value)
        except (ValueError, IndexError):
            pass
    
    def _handle_peak_packet(self, packet: Packet) -> None:
        """Handle PEAK packet."""
        try:
            if len(packet.payload) >= 1:
                value = float(packet.payload[0])
                self.dashboard.add_peak(value)
        except (ValueError, IndexError):
            pass
    
    def _handle_thresh_packet(self, packet: Packet) -> None:
        """Handle THRESH packet."""
        try:
            if len(packet.payload) >= 1:
                value = float(packet.payload[0])
                self.dashboard.add_threshold(value)
        except (ValueError, IndexError):
            pass
    
    def _handle_rr_packet(self, packet: Packet) -> None:
        """Handle RR packet."""
        try:
            if len(packet.payload) >= 1:
                rr_ms = float(packet.payload[0])
                self.dashboard.add_rr_interval(rr_ms)
        except (ValueError, IndexError):
            pass
    
    def _handle_hr_packet(self, packet: Packet) -> None:
        """Handle HR packet."""
        try:
            if len(packet.payload) >= 1:
                hr = float(packet.payload[0])
                self.dashboard.add_hr(hr)
        except (ValueError, IndexError):
            pass
    
    def _handle_feature_packet(self, packet: Packet) -> None:
        """Handle FEATURE/F= packet - update feature cards & log."""
        try:
            payload = list(packet.payload)
            
            # Strip "F=" prefix if present
            if len(payload) > 0 and '=' in payload[0]:
                payload[0] = payload[0].split('=', 1)[-1]
            
            features = FeatureModel.from_uart(payload)
            
            if features is not None:
                epoch_num = self._packet_counts.get(PacketType.FEATURE, 0)
                features.epoch_index = epoch_num
                self.dashboard.update_features(features)
                self.session_manager.log_feature(payload)
        except Exception as e:
            self.dashboard.console.append(f"⚠️ Feature error: {e}")
    
    def _handle_prediction_packet(self, packet: Packet) -> None:
        """
        Handle PRED packet to update prediction card & confidence chart.

        Board sends both:
          SCORES=score0,score1,score2,score3
          PREDICTION=stage (stage_name)
        """
        try:
            if len(packet.payload) >= 1:
                first = packet.payload[0].strip()
                # SCORES=0.1,0.2,... route the 4 class-confidence values.
                if ',' in first or all(self._is_float(x.strip().split('=', 1)[-1])
                                       for x in packet.payload):
                    scores_raw = [x.strip().split('=', 1)[-1] for x in packet.payload]
                    scores = [float(v) for v in scores_raw if v]
                    if len(scores) >= 4:
                        # Board mengirim raw logits (bisa >1 / negatif).
                        # Softmax agar jadi probabilitas valid (jumlah 1.0),
                        # sehingga tampil 45%, bukan 450%.
                        probs = self._softmax(scores[:4])
                        best_idx = scores.index(max(scores[:4]))
                        best_name = PredictionModel.STAGE_NAMES.get(best_idx, f"Stage{best_idx}")
                        best_conf = max(probs)
                        self.dashboard.confidence_chart.update_scores(probs)
                        self.dashboard.prediction_card.update_prediction(
                            best_name, best_conf,
                            PredictionModel.STAGE_COLORS.get(best_idx, "#3498db"))
                        self.dashboard.console.append(
                            f"\U0001F3AF Prediction: {best_idx} "
                            f"({best_name})")
                        # Persist to prediction.csv (this was missing -> file stayed empty).
                        self.session_manager.log_prediction(best_idx, best_conf, best_name)
                else:
                    # PREDICTION=2 (Deep Sleep) -> stage number.
                    try:
                        stage = int(first.split('=', 1)[-1].split()[0])
                    except (ValueError, IndexError):
                        stage_num_txt = first.split('=', 1)[-1]
                        parts = stage_num_txt.split()
                        stage = int(parts[0]) if parts and parts[0].lstrip('-').isdigit() else -1
                    if stage >= 0:
                        stage_name = PredictionModel.STAGE_NAMES.get(stage, '?')
                        self.dashboard.console.append(
                            f"\U0001F3AF Prediction: {stage} ({stage_name})")
                        # Persist to prediction.csv.
                        # PREDICTION=stage carries no confidence, so log 0.0.
                        self.session_manager.log_prediction(stage, 0.0, stage_name)
        except (ValueError, IndexError):
            pass

    @staticmethod
    def _is_float(text: str) -> bool:
        """Return True if text can be parsed as a float."""
        try:
            float(text)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _softmax(scores: List[float]) -> List[float]:
        """
        Convert raw logits to normalized probabilities (sum to 1.0).

        The board sends dequantized logits (can be >1 or negative).
        Softmax maps them to [0,1] so the confidence chart/card shows
        valid percentages (e.g. 45%, not 450%).
        """
        import math
        raw = list(scores)
        if not raw:
            return []
        # Numerical stability: subtract max before exp.
        m = max(raw)
        exps = [math.exp(v - m) for v in raw]
        total = sum(exps)
        if total <= 0.0:
            # Fallback: uniform distribution.
            return [1.0 / len(raw)] * len(raw)
        return [e / total for e in exps]
    
    # ========== CONNECTION ==========
    @Slot(str, int)
    def _handle_connection_request(self, port_name: str, baud_rate: int) -> None:
        if self.serial_worker and self.serial_worker.isRunning():
            self._handle_disconnection()
        
        try:
            self.serial_worker = SerialWorker(port_name, baud_rate)
            self.serial_worker.data_received.connect(self._on_data_received)
            self.serial_worker.packet_received.connect(self._on_packet_received)
            self.serial_worker.error_occurred.connect(self._on_worker_error)
            self.serial_worker.connection_status.connect(self._update_connection_state)
            self.serial_worker.finished.connect(self._on_worker_finished)
            
            self._packet_counts = {pkt_type: 0 for pkt_type in PacketType}
            self._update_counters()
            self.dashboard.clear_all()
            self.serial_worker.start()
        except Exception as e:
            self.dashboard.console.append(f"❌ Failed: {e}")
            self._update_connection_state(False)
    
    @Slot()
    def _handle_disconnection(self) -> None:
        if self.serial_worker:
            try:
                self.serial_worker.data_received.disconnect(self._on_data_received)
                self.serial_worker.packet_received.disconnect(self._on_packet_received)
                self.serial_worker.error_occurred.disconnect(self._on_worker_error)
                self.serial_worker.connection_status.disconnect(self._update_connection_state)
                self.serial_worker.finished.disconnect(self._on_worker_finished)
            except Exception:
                pass
            self.serial_worker.stop()
            self.serial_worker = None
        self._update_connection_state(False)
    
    @Slot(str)
    def _on_worker_error(self, error_message: str) -> None:
        self.dashboard.console.append(f"❌ {error_message}")
        if "Serial" in error_message or "port" in error_message.lower():
            self._update_connection_state(False)
    
    @Slot()
    def _on_worker_finished(self) -> None:
        self.dashboard.console.append("Worker finished")
        self._update_connection_state(False)
    
    # ========== UI UPDATES ==========
    def _update_counters(self) -> None:
        self.accel_counter.setText(f"ACCEL: {self._packet_counts.get(PacketType.ACCEL, 0)}")
        self.ppg_counter.setText(f"PPG: {self._packet_counts.get(PacketType.PPG, 0)}")
        self.feature_counter.setText(f"FEATURE: {self._packet_counts.get(PacketType.FEATURE, 0)}")
        total = sum(self._packet_counts.values())
        self.total_counter.setText(f"Total: {total}")
    
    def _update_connection_state(self, connected: bool) -> None:
        if connected:
            self.status_indicator.setText("Connected")
            self.status_indicator.setStyleSheet(
                "background-color: #4caf50; color: white; padding: 2px 8px; "
                "border-radius: 3px; font-weight: bold;"
            )
            if self.serial_worker:
                self.port_label.setText(f"Port: {self.serial_worker.port_name} @ {self.serial_worker.baud_rate}")
        else:
            self.status_indicator.setText("Disconnected")
            self.status_indicator.setStyleSheet(
                "background-color: #e0e0e0; color: #505050; padding: 2px 8px; "
                "border-radius: 3px; font-weight: bold;"
            )
            self.port_label.setText("No port selected")
        self.toolbar.update_connection_state(connected)
    
    def closeEvent(self, event) -> None:
        if self.serial_worker and self.serial_worker.isRunning():
            self._handle_disconnection()
        event.accept()
    
    def _start_recording(self) -> None:
        """Start recording session."""
        port = self.serial_worker.port_name if self.serial_worker else ""
        baud = self.serial_worker.baud_rate if self.serial_worker else 921600
        
        path = self.session_manager.start_recording(
            serial_port=port,
            baudrate=baud
        )
        
        if path:
            self.dashboard.console.append(f"📁 Recording started: {path}")
            
            # Kirim waktu mulai (unix timestamp) ke board agar time_of_night benar.
            # Firmware memparse command "START=<unix>" dari UART (bukan hardcode).
            if self.serial_worker:
                unix_now = int(__import__('time').time())
                self.serial_worker.send_command(f"START={unix_now}")
                self.dashboard.console.append(
                    f"🕒 Sent START={unix_now} to board")
        else:
            self.dashboard.console.append("❌ Failed to start recording")
    
    @Slot()
    def _stop_recording(self) -> None:
        """Stop recording session."""
        self.session_manager.stop_recording()
    
    @Slot(str)
    def _on_recording_started(self, path: str) -> None:
        """Handle recording started."""
        self.toolbar.update_recording_state(True, "00:00:00")
        self.dashboard.console.append(f"📁 Session: {path}")
    
    @Slot(str)
    def _on_recording_stopped(self, path: str) -> None:
        """Handle recording stopped."""
        self.toolbar.update_recording_state(False)
        self.dashboard.console.append(f"💾 Recording saved: {path}")
        self.dashboard.console.append(f"📊 Packets: {self.session_manager.state.packet_count}")
        self.dashboard.console.append(f"⏱ Duration: {self.session_manager.state.formatted_duration}")
    
    @Slot(float)
    def _on_duration_updated(self, elapsed: float) -> None:
        """Update duration display."""
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.toolbar.update_recording_state(True, duration)
    
    @Slot(str)
    def _on_recording_error(self, error: str) -> None:
        """Handle recording errors."""
        self.dashboard.console.append(f"❌ Recording error: {error}")

