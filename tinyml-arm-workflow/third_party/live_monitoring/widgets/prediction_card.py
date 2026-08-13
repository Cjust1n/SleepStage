# widgets/prediction_card.py (COMPACT)
"""Compact prediction card."""

from typing import Optional
from datetime import datetime
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class PredictionCard(QWidget):
    """Compact prediction card."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignCenter)
        
        self.setStyleSheet("""
            PredictionCard {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        
        title = QLabel("Prediction")
        title.setFont(QFont("Segoe UI", 8))
        title.setStyleSheet("color: #7f8c8d; border: none;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.stage_label = QLabel("---")
        self.stage_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.stage_label.setStyleSheet("color: #95a5a6; border: none;")
        self.stage_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stage_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Segoe UI", 12))
        self.confidence_label.setStyleSheet("color: #2c3e50; border: none;")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
    
    def update_prediction(self, stage_name: str, confidence: float, color: str = "#3498db") -> None:
        self.stage_label.setText(stage_name.upper())
        self.stage_label.setStyleSheet(f"color: {color}; border: none;")
        self.confidence_label.setText(f"{confidence:.0%}")
    
    def clear(self) -> None:
        self.stage_label.setText("---")
        self.stage_label.setStyleSheet("color: #95a5a6; border: none;")
        self.confidence_label.setText("")