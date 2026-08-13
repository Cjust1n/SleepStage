# widgets/rr_interval_plot.py
"""
RR Interval plot showing beat-to-beat intervals.
Highlights abnormal RR values for arrhythmia detection.
"""

from typing import Optional, Tuple
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont

import pyqtgraph as pg


class RRIntervalPlot(QWidget):
    """
    RR Interval visualization.
    
    X-axis: Beat number
    Y-axis: RR interval (ms)
    
    Normal range: 600-1000 ms (green)
    Abnormal: <600 or >1000 ms (red)
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        
        title = QLabel("RR Intervals")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        layout.addWidget(title)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.setLabel('left', 'RR (ms)')
        self.plot_widget.setLabel('bottom', 'Beat #')
        
        # Normal range shading
        self.normal_region = pg.LinearRegionItem(
            values=[600, 1000],
            orientation='horizontal',
            brush=pg.mkBrush(46, 204, 113, 30),
            pen=pg.mkPen(None)
        )
        self.plot_widget.addItem(self.normal_region)
        
        # RR intervals (bar chart style)
        self.rr_curve = self.plot_widget.plot(
            [], [],
            pen=None,
            symbol='o',
            symbolSize=6,
            symbolBrush=(52, 152, 219),
            symbolPen=(41, 128, 185),
            name='RR Intervals'
        )
        
        layout.addWidget(self.plot_widget)
    
    def update_data(self, indices: np.ndarray, rr_values: np.ndarray) -> None:
        """Update RR interval plot."""
        if len(indices) > 0:
            self.rr_curve.setData(indices, rr_values)
            self.plot_widget.autoRange()
    
    def clear(self) -> None:
        self.rr_curve.clear()