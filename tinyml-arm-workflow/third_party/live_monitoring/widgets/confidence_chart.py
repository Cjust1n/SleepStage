# widgets/confidence_chart.py
"""
Horizontal bar chart showing confidence scores for all sleep stages.
"""

from typing import Optional, List, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor


class ConfidenceBar(QWidget):
    """Single confidence bar for one sleep stage."""
    
    def __init__(self, name: str, color: tuple, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._name = name
        self._color = QColor(*color)
        self._value = 0.0
        self.setFixedHeight(28)
    
    def set_value(self, value: float) -> None:
        """Set confidence value (0-1)."""
        self._value = max(0.0, min(1.0, value))
        self.update()
    
    def paintEvent(self, event) -> None:
        """Custom paint for the confidence bar."""
        from PySide6.QtGui import QPainter, QPen, QBrush
        from PySide6.QtCore import QRectF
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(245, 245, 250))
        
        # Bar
        bar_width = int(self.width() * self._value)
        bar_rect = self.rect()
        bar_rect.setWidth(bar_width)
        painter.fillRect(bar_rect, self._color)
        
        # Text
        painter.setPen(QColor(44, 62, 80))
        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)
        
        text = f"{self._name}  {self._value:.0%}"
        painter.drawText(self.rect().adjusted(8, 0, -8, 0), Qt.AlignVCenter, text)
        
        painter.end()


class ConfidenceChart(QWidget):
    """
    Horizontal bar chart for all class confidence scores.
    """
    
    CLASSES = [
        ("Wake", (243, 156, 18)),
        ("REM", (155, 89, 182)),
        ("Light", (52, 152, 219)),
        ("Deep", (46, 204, 113)),
    ]
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._bars: List[ConfidenceBar] = []
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Title
        title = QLabel("Confidence Scores")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; border: none;")
        layout.addWidget(title)
        
        # Bars
        for name, color in self.CLASSES:
            bar = ConfidenceBar(name, color)
            self._bars.append(bar)
            layout.addWidget(bar)
        
        layout.addStretch()
        
        self.setStyleSheet("""
            ConfidenceChart {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
    
    def update_scores(self, scores: List[float]) -> None:
        """
        Update all confidence bars.
        
        Args:
            scores: List of 4 confidence values [wake, rem, light, deep]
        """
        for i, bar in enumerate(self._bars):
            if i < len(scores):
                bar.set_value(scores[i])
    
    def clear(self) -> None:
        """Reset all bars."""
        for bar in self._bars:
            bar.set_value(0.0)