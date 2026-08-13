# widgets/heart_rate_plot.py
"""
Heart Rate trend plot with rolling history.
"""

from typing import Optional
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont

import pyqtgraph as pg


class HeartRatePlot(QWidget):
    """
    Heart Rate vs Time plot.
    
    Shows rolling HR history with normal range highlighting.
    Normal: 60-100 bpm (green zone)
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        
        title = QLabel("Heart Rate")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.setLabel('left', 'HR (bpm)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.setYRange(40, 120)
        
        # Normal HR zone
        self.normal_region = pg.LinearRegionItem(
            values=[60, 100],
            orientation='horizontal',
            brush=pg.mkBrush(46, 204, 113, 30),
            pen=pg.mkPen(None)
        )
        self.plot_widget.addItem(self.normal_region)
        
        # HR curve (red line)
        self.hr_curve = self.plot_widget.plot(
            [], [],
            pen=pg.mkPen(color=(231, 76, 60), width=2),
            name='Heart Rate'
        )
        
        layout.addWidget(self.plot_widget)
    
    def update_data(self, times: np.ndarray, hr_values: np.ndarray) -> None:
        """Update HR plot."""
        if len(times) > 0:
            self.hr_curve.setData(times, hr_values)
            self.plot_widget.autoRange()
    
    def clear(self) -> None:
        self.hr_curve.clear()