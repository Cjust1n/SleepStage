# widgets/feature_card.py (UPDATE)
"""
Compact feature card widget.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class FeatureCard(QWidget):
    """Compact feature card showing name, value, and unit."""
    
    def __init__(
        self,
        feature_name: str,
        unit: str = "",
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._setup_ui(feature_name, unit)
    
    def _setup_ui(self, name: str, unit: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(1)
        layout.setAlignment(Qt.AlignCenter)
        
        self.setStyleSheet("""
            FeatureCard {
                background-color: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 6px;
            }
        """)
        
        # Feature name (small)
        self.name_label = QLabel(name)
        self.name_label.setFont(QFont("Segoe UI", 7))
        self.name_label.setStyleSheet("color: #95a5a6; border: none;")
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)
        
        # Value (medium-bold)
        self.value_label = QLabel("---")
        self.value_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.value_label.setStyleSheet("color: #2c3e50; border: none;")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        
        # Unit (small)
        if unit:
            self.unit_label = QLabel(unit)
            self.unit_label.setFont(QFont("Segoe UI", 7))
            self.unit_label.setStyleSheet("color: #bdc3c7; border: none;")
            self.unit_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.unit_label)
        
        self.setMinimumSize(70, 55)
        self.setMaximumHeight(65)
    
    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
    
    def clear(self) -> None:
        self.value_label.setText("---")