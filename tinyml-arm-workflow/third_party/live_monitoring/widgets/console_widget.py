# widgets/console_widget.py
"""
Console widget for displaying UART output and debug messages.
Provides a scrollable text display with timestamp and message filtering.
"""

from typing import Optional
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QHBoxLayout,
    QPushButton, QLabel, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QTextCursor


class ConsoleWidget(QWidget):
    """
    Console widget for displaying serial output and debug messages.
    
    Provides a professional terminal-like interface with timestamp,
    auto-scrolling, and clear functionality.
    
    Signals:
        clear_requested: Emitted when user clicks clear button
    
    Attributes:
        text_display: Main text edit widget for console output
        auto_scroll: Toggle for automatic scrolling
    """
    
    clear_requested = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the console widget with text display and controls.
        
        Args:
            parent: Parent widget (usually the splitter)
        """
        super().__init__(parent)
        self._auto_scroll = True
        self._max_lines = 1000  # Maximum lines to keep
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Configure the console layout and widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        title_label = QLabel("UART Console")
        title_label.setFont(QFont("Consolas", 11, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMaximumWidth(60)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #dfe4e6;
            }
        """)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.clear_btn)
        
        layout.addLayout(header_layout)
        
        # Console text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Consolas", 10))
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        layout.addWidget(self.text_display)
        
        # Connect signals
        self.clear_btn.clicked.connect(self.clear)
    
    @Slot(str)
    def append(self, text: str) -> None:
        """
        Append text to the console with timestamp.
        
        Automatically handles line limits and scrolling based on user preferences.
        
        Args:
            text: Text to append to the console
        """
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_text = f"[{timestamp}] {text}"
        
        # Check line count and remove old lines if necessary
        document = self.text_display.document()
        if document.blockCount() > self._max_lines:
            # Remove first block
            cursor = QTextCursor(document.firstBlock())
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # Remove newline
        
        # Append new text
        self.text_display.append(formatted_text)
        
        # Auto-scroll to bottom
        if self._auto_scroll:
            scrollbar = self.text_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def clear(self) -> None:
        """Clear all text from the console."""
        self.text_display.clear()
        self.clear_requested.emit()
        self.append("Console cleared")
    
    def set_auto_scroll(self, enabled: bool) -> None:
        """
        Enable or disable automatic scrolling.
        
        Args:
            enabled: True to enable auto-scroll, False to disable
        """
        self._auto_scroll = enabled
    
    def set_max_lines(self, max_lines: int) -> None:
        """
        Set maximum number of lines to keep in the console.
        
        Args:
            max_lines: Maximum line count (minimum 100)
        """
        self._max_lines = max(max_lines, 100)