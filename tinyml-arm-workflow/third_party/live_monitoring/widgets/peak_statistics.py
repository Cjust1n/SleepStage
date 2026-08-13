# widgets/peak_statistics.py
"""
Peak detection statistics panel.
Shows real-time metrics for algorithm debugging.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from models.peak_model import PeakStatistics


class StatCard(QFrame):
    """Single statistic card."""
    
    def __init__(self, label: str, value: str = "---", color: str = "#2c3e50", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 6px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignCenter)
        
        self.label = QLabel(label)
        self.label.setFont(QFont("Segoe UI", 7))
        self.label.setStyleSheet("color: #7f8c8d; border: none;")
        self.label.setAlignment(Qt.AlignCenter)
        
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.value_label.setStyleSheet(f"color: {color}; border: none;")
        self.value_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.label)
        layout.addWidget(self.value_label)
        
        self.setMinimumSize(90, 55)
        self.setMaximumHeight(60)
    
    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class PeakStatisticsPanel(QWidget):
    """
    Real-time peak detection statistics dashboard.
    
    Displays: Current HR, Avg HR, Peaks, RR Count,
    Threshold, Refractory State, Peak Amplitude
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # Title
        title = QLabel("Peak Detector Statistics")
        title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)
        
        # Grid of stat cards (4 columns x 2 rows)
        grid = QGridLayout()
        grid.setSpacing(4)
        
        self.cards = {}
        
        stats_config = [
            ("Current HR", "--- bpm", "#e74c3c", 0, 0),
            ("Avg HR", "--- bpm", "#e67e22", 0, 1),
            ("Detected Peaks", "0", "#2c3e50", 0, 2),
            ("RR Count", "0", "#2c3e50", 0, 3),
            ("Threshold", "---", "#27ae60", 1, 0),
            ("Refractory", "---", "#8e44ad", 1, 1),
            ("Peak Amplitude", "---", "#2980b9", 1, 2),
            ("Last RR", "--- ms", "#2c3e50", 1, 3),
        ]
        
        for label, value, color, row, col in stats_config:
            card = StatCard(label, value, color)
            self.cards[label] = card
            grid.addWidget(card, row, col)
        
        layout.addLayout(grid)
    
    def update_statistics(self, stats: PeakStatistics) -> None:
        """Update all statistics cards."""
        self.cards["Current HR"].set_value(f"{stats.current_hr:.0f} bpm")
        self.cards["Avg HR"].set_value(f"{stats.avg_hr:.0f} bpm")
        self.cards["Detected Peaks"].set_value(str(stats.total_peaks))
        self.cards["RR Count"].set_value(str(stats.rr_count))
        self.cards["Threshold"].set_value(f"{stats.threshold_value:.0f}")
        self.cards["Refractory"].set_value("Active" if stats.refractory_active else "Ready")
        self.cards["Peak Amplitude"].set_value(f"{stats.peak_amplitude:.0f}")
        self.cards["Last RR"].set_value(f"{stats.last_rr_ms:.0f} ms")
    
    def clear(self) -> None:
        """Reset all cards."""
        for card in self.cards.values():
            card.set_value("---")