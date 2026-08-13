# widgets/buffer_progress.py
"""
Buffer progress indicator showing GRU sequence buffer fill status.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class BufferProgress(QWidget):
    """
    Progress indicator for GRU sequence buffer.
    
    Shows how many epochs have been collected out of the required 30.
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # Title
        title = QLabel("Sequence Buffer")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; border: none;")
        layout.addWidget(title)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(30)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m epochs")
        self.progress_bar.setFont(QFont("Segoe UI", 9))
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #f0f0f0;
                text-align: center;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Collecting data...")
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setStyleSheet("color: #7f8c8d; border: none;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setStyleSheet("""
            BufferProgress {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
    
    def update_progress(self, filled: int, total: int = 30) -> None:
        """
        Update buffer progress.
        
        Args:
            filled: Number of collected epochs
            total: Total epochs needed
        """
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(min(filled, total))
        
        if filled >= total:
            self.status_label.setText("✓ Buffer full - Inference active")
            self.status_label.setStyleSheet("color: #27ae60; border: none;")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    background-color: #f0f0f0;
                    text-align: center;
                    height: 24px;
                }
                QProgressBar::chunk {
                    background-color: #27ae60;
                    border-radius: 5px;
                }
            """)
        else:
            remaining = total - filled
            self.status_label.setText(f"Collecting... ({remaining} epochs remaining)")
            self.status_label.setStyleSheet("color: #7f8c8d; border: none;")
    
    def clear(self) -> None:
        """Reset progress."""
        self.progress_bar.setValue(0)
        self.status_label.setText("Collecting data...")
        self.status_label.setStyleSheet("color: #7f8c8d; border: none;")