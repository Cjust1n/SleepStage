# logger/session_manager.py
"""
Session manager for recording sessions.
Creates timestamped folders, manages metadata, and coordinates logging.
"""

from typing import Optional
import os
import json
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from models.recording_state import RecordingState, RecordingStatus, SessionMetadata
from logger.csv_logger import CsvLogger
from logger.file_writer import FileWriter


class SessionManager(QObject):
    """
    Manages recording sessions with timestamped folders.
    
    Creates session directories, coordinates CSV and raw logging,
    and maintains session metadata.
    
    Signals:
        recording_started: Emitted when recording begins
        recording_stopped: Emitted when recording ends
        duration_updated: Emitted with elapsed seconds
        packet_counted: Emitted with total packet count
        error_occurred: Emitted on errors
    
    Attributes:
        state: Current recording state
        csv_logger: CsvLogger instance
        raw_logger: FileWriter for raw UART log
        base_recordings_dir: Root directory for recordings
    """
    
    recording_started = Signal(str)      # session_path
    recording_stopped = Signal(str)      # session_path
    duration_updated = Signal(float)     # elapsed_seconds
    packet_counted = Signal(int)         # total_packets
    error_occurred = Signal(str)
    
    def __init__(
        self,
        base_dir: str = "recordings",
        parent: Optional[QObject] = None
    ) -> None:
        """
        Initialize session manager.
        
        Args:
            base_dir: Base directory for all recordings
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self.state = RecordingState()
        self.csv_logger = CsvLogger()
        self.raw_logger = FileWriter(flush_interval=1000, max_buffer_size=100)
        self.base_recordings_dir = base_dir
        
        # Create base directory
        os.makedirs(self.base_recordings_dir, exist_ok=True)
        
        # Timer for duration updates
        from PySide6.QtCore import QTimer
        self._duration_timer = QTimer(self)
        self._duration_timer.timeout.connect(self._update_duration)
        self._duration_timer.setInterval(1000)  # Every second
    
    def start_recording(
        self,
        serial_port: str = "",
        baudrate: int = 921600,
        board_name: str = "Grove Vision AI V2",
        firmware_version: str = "0.9.4"
    ) -> Optional[str]:
        """
        Start a new recording session.
        
        Creates timestamped folder and initializes all log files.
        
        Args:
            serial_port: Connected serial port
            baudrate: Baud rate used
            board_name: Name of the device
            firmware_version: Firmware version
        
        Returns:
            Session path if successful, None otherwise
        """
        if self.state.is_recording:
            self.error_occurred.emit("Already recording")
            return None
        
        try:
            # Create session folder
            now = datetime.now()
            session_id = now.strftime("%Y-%m-%d_%H-%M-%S")
            session_path = os.path.join(self.base_recordings_dir, session_id)
            os.makedirs(session_path, exist_ok=True)
            
            # Initialize CSV logger
            if not self.csv_logger.start_session(session_path):
                self.error_occurred.emit("Failed to create CSV files")
                return None
            
            # Initialize raw UART logger
            raw_path = os.path.join(session_path, "uart.log")
            if not self.raw_logger.open(raw_path):
                self.error_occurred.emit("Failed to create UART log")
                self.csv_logger.stop_session()
                return None
            
            # Update state
            self.state.status = RecordingStatus.RECORDING
            self.state.session_id = session_id
            self.state.start_time = now
            self.state.elapsed_seconds = 0.0
            self.state.packet_count = 0
            self.state.session_path = session_path
            
            # Set metadata
            self.state.metadata = SessionMetadata(
                session_id=session_id,
                start_time=now,
                serial_port=serial_port,
                baudrate=baudrate,
                board_name=board_name,
                firmware_version=firmware_version,
                sample_counts={}
            )
            
            # Start duration timer
            self._duration_timer.start()
            
            # Save initial metadata
            self._save_metadata()
            
            self.recording_started.emit(session_path)
            return session_path
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start recording: {str(e)}")
            return None
    
    def stop_recording(self) -> None:
        """Stop current recording session."""
        if not self.state.is_recording:
            return
        
        # Update metadata
        self.state.metadata.end_time = datetime.now()
        self.state.metadata.duration_seconds = self.state.elapsed_seconds
        self.state.metadata.sample_counts = self._get_sample_counts()
        
        # Save final metadata
        self._save_metadata()
        
        # Stop timer
        self._duration_timer.stop()
        
        # Close all files
        self.csv_logger.stop_session()
        self.raw_logger.close()
        
        # Update state
        self.state.status = RecordingStatus.STOPPED
        
        self.recording_stopped.emit(self.state.session_path)
    
    def log_raw_line(self, line: str) -> None:
        """
        Log a raw UART line.
        
        Args:
            line: Raw line from UART
        """
        if self.state.is_recording:
            self.raw_logger.write_line(line)
            self.state.packet_count += 1
            self.packet_counted.emit(self.state.packet_count)
    
    def log_accel(self, x: float, y: float, z: float) -> None:
        """Log accelerometer sample."""
        if self.state.is_recording:
            self.csv_logger.log_accel(x, y, z)
            self.state.packet_count += 1
    
    def log_ppg(self, value: float) -> None:
        """Log PPG sample."""
        if self.state.is_recording:
            self.csv_logger.log_ppg(value)
            self.state.packet_count += 1
    
    def log_feature(self, features: list) -> None:
        """Log feature vector."""
        if self.state.is_recording:
            self.csv_logger.log_feature(features)
            self.state.packet_count += 1
    
    def log_prediction(self, stage: int, confidence: float, stage_name: str = "") -> None:
        """Log prediction."""
        if self.state.is_recording:
            self.csv_logger.log_prediction(stage, confidence, stage_name)
            self.state.packet_count += 1
    
    def open_recording_folder(self) -> None:
        """Open the current recordings directory in file explorer."""
        import subprocess
        import sys
        
        path = self.state.session_path or self.base_recordings_dir
        
        if sys.platform == 'win32':
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', path])
        else:
            subprocess.run(['xdg-open', path])
    
    def get_disk_usage(self) -> str:
        """
        Get disk usage for recordings directory.
        
        Returns:
            Human-readable size string
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.base_recordings_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            # Format size
            if total_size < 1024:
                return f"{total_size} B"
            elif total_size < 1024 * 1024:
                return f"{total_size / 1024:.1f} KB"
            elif total_size < 1024 * 1024 * 1024:
                return f"{total_size / (1024 * 1024):.1f} MB"
            else:
                return f"{total_size / (1024 * 1024 * 1024):.2f} GB"
        except Exception:
            return "N/A"
    
    def _update_duration(self) -> None:
        """Update elapsed duration."""
        if self.state.is_recording and self.state.start_time:
            self.state.elapsed_seconds = (
                datetime.now() - self.state.start_time
            ).total_seconds()
            self.duration_updated.emit(self.state.elapsed_seconds)
    
    def _save_metadata(self) -> None:
        """Save session metadata to JSON file."""
        if not self.state.session_path:
            return
        
        metadata_path = os.path.join(self.state.session_path, "session.json")
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.state.metadata.to_dict(), f, indent=2)
        except Exception as e:
            self.error_occurred.emit(f"Failed to save metadata: {str(e)}")
    
    def _get_sample_counts(self) -> dict:
        """Get current sample counts from CSV logger."""
        return {
            'accel': self.state.metadata.sample_counts.get('accel', 0),
            'ppg': self.state.metadata.sample_counts.get('ppg', 0),
            'feature': self.state.metadata.sample_counts.get('feature', 0),
            'prediction': self.state.metadata.sample_counts.get('prediction', 0),
            'total': self.state.packet_count
        }