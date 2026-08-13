# gui/toolbar.py (FULL UPDATED VERSION)
"""
Toolbar implementation for Sleep Stage Development Studio.
Provides serial port selection, connection controls, and status feedback.
"""

from typing import Optional, List
from PySide6.QtWidgets import (
    QToolBar, QComboBox, QPushButton, QLabel, QWidget,
    QHBoxLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QAction

from serial_port.serial_manager import SerialManager


class ToolBar(QToolBar):
    """
    Application toolbar for serial connection management.
    
    Provides interface for selecting serial ports, configuring baud rates,
    and managing serial connections with visual feedback.
    
    Signals:
        connection_requested: Emitted when user requests connection (port_name, baud_rate)
        disconnection_requested: Emitted when user requests disconnection
        port_selected: Emitted when a port is selected (str)
        message_log: Emitted for logging messages to console (str)
    """
    
    # Signals
    connection_requested = Signal(str, int)  # port_name, baud_rate
    disconnection_requested = Signal()
    port_selected = Signal(str)
    message_log = Signal(str)
    recording_started = Signal()
    recording_stopped = Signal()
    open_folder_requested = Signal()
    
    def __init__(self, serial_manager: SerialManager, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the toolbar with serial manager reference.
        
        Args:
            serial_manager: Instance of SerialManager for port operations
            parent: Parent widget (usually the main window)
        """
        super().__init__("Connection Toolbar", parent)
        
        self.serial_manager = serial_manager
        self._is_connected = False
        
        self._setup_toolbar()
        self._create_widgets()
        self._connect_signals()
        
        # Initial port refresh
        self._refresh_ports()
    
    def _setup_toolbar(self) -> None:
        """Configure toolbar appearance and behavior."""
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(QSize(20, 20))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    
    def _create_widgets(self) -> None:
        """Create and add all toolbar widgets."""
        # Serial Port section
        self.addWidget(QLabel("Serial Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(200)
        self.port_combo.setToolTip("Select serial port for connection")
        self.addWidget(self.port_combo)
        
        self.addSeparator()
        
        # Baud Rate section
        self.addWidget(QLabel("Baud Rate:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems([
            "9600", "19200", "38400", "57600", 
            "115200", "230400", "460800", "921600"
        ])
        # Set default baud rate
        default_baud_index = self.baud_combo.findText("921600")
        if default_baud_index >= 0:
            self.baud_combo.setCurrentIndex(default_baud_index)
        self.baud_combo.setToolTip("Select baud rate for serial communication")
        self.addWidget(self.baud_combo)
        
        self.addSeparator()
        
        # Action buttons
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setToolTip("Refresh available serial ports")
        self.refresh_btn.setMaximumWidth(120)
        self.addWidget(self.refresh_btn)
        
        self.connect_btn = QPushButton("🔌 Connect")
        self.connect_btn.setToolTip("Connect to selected serial port")
        self.connect_btn.setMaximumWidth(120)
        self.connect_btn.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; border-radius: 4px; "
            "padding: 4px 12px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("🔌 Disconnect")
        self.disconnect_btn.setToolTip("Disconnect from serial port")
        self.disconnect_btn.setMaximumWidth(120)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; border-radius: 4px; "
            "padding: 4px 12px; }"
            "QPushButton:hover { background-color: #da190b; }"
            "QPushButton:disabled { background-color: #cccccc; color: #666666; }"
        )
        self.addWidget(self.disconnect_btn)

        self.addSeparator()
        
        # Recording section
        self.record_btn = QPushButton("⏺ Start Recording")
        self.record_btn.setToolTip("Start recording session")
        self.record_btn.setMaximumWidth(140)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; color: white; border-radius: 4px;
                padding: 4px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.addWidget(self.record_btn)
        
        self.stop_record_btn = QPushButton("⏹ Stop")
        self.stop_record_btn.setToolTip("Stop recording session")
        self.stop_record_btn.setMaximumWidth(80)
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6; color: white; border-radius: 4px;
                padding: 4px 12px;
            }
            QPushButton:hover { background-color: #7f8c8d; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.addWidget(self.stop_record_btn)
        
        # Recording status indicator
        self.recording_indicator = QLabel("⭘")
        self.recording_indicator.setStyleSheet(
            "color: #e74c3c; font-size: 16px; padding: 0px 4px;"
        )
        self.recording_indicator.setToolTip("Recording status")
        self.recording_indicator.hide()
        self.addWidget(self.recording_indicator)
        
        self.recording_time_label = QLabel("")
        self.recording_time_label.setStyleSheet(
            "color: #2c3e50; font-weight: bold; padding: 0px 4px;"
        )
        self.recording_time_label.hide()
        self.addWidget(self.recording_time_label)
    
    def _connect_signals(self) -> None:
        """Connect widget signals to appropriate slots."""
        self.refresh_btn.clicked.connect(self._refresh_ports)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        self.port_combo.currentTextChanged.connect(self.port_selected.emit)
        self.record_btn.clicked.connect(self.recording_started.emit)
        self.stop_record_btn.clicked.connect(self.recording_stopped.emit)
    
    def _refresh_ports(self) -> None:
        """
        Refresh the list of available serial ports.
        """
        current_port = self.port_combo.currentText()
        
        self.port_combo.clear()
        ports = self.serial_manager.list_ports()
        
        if ports:
            self.port_combo.addItems(ports)
            self.port_combo.setEnabled(True)
            
            if current_port and current_port in ports:
                index = self.port_combo.findText(current_port)
                self.port_combo.setCurrentIndex(index)
            
            self.message_log.emit(f"Found {len(ports)} serial port(s)")
            
            port_details = self.serial_manager.get_port_details()
            for detail in port_details:
                if detail['device'] in ports:
                    self.message_log.emit(
                        f"  • {detail['device']} - {detail['description']}"
                    )
        else:
            self.port_combo.addItem("No serial ports available")
            self.port_combo.setEnabled(False)
            self.message_log.emit("No serial ports found")
    
    def _on_connect_clicked(self) -> None:
        """Handle connect button click."""
        port = self.port_combo.currentText()
        baud = int(self.baud_combo.currentText())
        
        if not port or port == "No serial ports available":
            self.message_log.emit("❌ Error: No valid serial port selected")
            return
        
        self.message_log.emit(f"Connecting to {port} at {baud} baud...")
        self.connection_requested.emit(port, baud)
    
    def _on_disconnect_clicked(self) -> None:
        """Handle disconnect button click."""
        self.message_log.emit("Disconnecting...")
        self.disconnection_requested.emit()
    
    def update_connection_state(self, connected: bool) -> None:
        """
        Update UI elements based on connection state.
        
        Args:
            connected: True if connected, False otherwise
        """
        self._is_connected = connected
        
        self.port_combo.setEnabled(not connected)
        self.baud_combo.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        
    def update_recording_state(self, is_recording: bool, duration: str = "") -> None:
        """Update recording UI elements."""
        self.record_btn.setEnabled(not is_recording)
        self.stop_record_btn.setEnabled(is_recording)
        
        if is_recording:
            self.recording_indicator.setText("●")
            self.recording_indicator.show()
            self.recording_time_label.setText(duration)
            self.recording_time_label.show()
        else:
            self.recording_indicator.hide()
            self.recording_time_label.hide()