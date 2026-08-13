# widgets/ppg_peak_plot.py
"""
PPG waveform plot with peak markers and adaptive threshold overlay.
Professional debugging visualization for peak detection algorithm.
"""

from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QFont

import pyqtgraph as pg


class PPGPeakPlot(QWidget):
    """
    PPG waveform with detected peaks and threshold visualization.
    
    Displays:
    - Raw PPG signal (continuous blue line)
    - Detected peaks (red circle markers)
    - Adaptive threshold (dashed green line)
    
    All on the same graph for easy algorithm debugging.
    """
    
    def __init__(
        self,
        history_duration: float = 10.0,
        sample_rate: float = 50.0,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        
        self._sample_rate = sample_rate
        self._history_size = int(history_duration * sample_rate)
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Create plot with PPG, peaks, and threshold."""
        from PySide6.QtWidgets import QVBoxLayout, QLabel
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        
        # Title
        title = QLabel("PPG + Peak Detection")
        title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 2px 0px;")
        layout.addWidget(title)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAntialiasing(True)
        self.plot_widget.setLabel('left', 'PPG Amplitude')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # Professional axis styling
        axis_pen = pg.mkPen(color='#34495e', width=1)
        self.plot_widget.getAxis('left').setPen(axis_pen)
        self.plot_widget.getAxis('bottom').setPen(axis_pen)
        
        # Legend
        self.legend = self.plot_widget.addLegend(
            offset=(10, 10),
            brush=pg.mkBrush(255, 255, 255, 200),
            pen=pg.mkPen(200, 200, 200),
            labelTextSize='9pt'
        )
        
        # PPG curve (blue line)
        self.ppg_curve = self.plot_widget.plot(
            [], [],
            pen=pg.mkPen(color=(52, 152, 219), width=1.5),
            name='PPG Signal'
        )
        
        # Peak markers (red circles)
        self.peak_scatter = self.plot_widget.plot(
            [], [],
            pen=None,
            symbol='o',
            symbolSize=8,
            symbolBrush=(231, 76, 60),
            symbolPen=(192, 57, 43),
            name='Detected Peaks'
        )
        
        # Threshold (dashed green line)
        self.threshold_curve = self.plot_widget.plot(
            [], [],
            pen=pg.mkPen(color=(46, 204, 113), width=1.5, style=pg.QtCore.Qt.DashLine),
            name='Threshold'
        )
        
        layout.addWidget(self.plot_widget)
    
    def update_ppg(self, times: np.ndarray, values: np.ndarray) -> None:
        """Update PPG waveform."""
        if len(times) > 0:
            self.ppg_curve.setData(times, values)
    
    def update_peaks(self, times: np.ndarray, values: np.ndarray) -> None:
        """Update peak markers."""
        if len(times) > 0:
            self.peak_scatter.setData(times, values)
    
    def update_threshold(self, times: np.ndarray, values: np.ndarray) -> None:
        """Update threshold line."""
        if len(times) > 0 and len(values) > 0:
            self.threshold_curve.setData(times, values)
    
    def clear(self) -> None:
        """Clear all data."""
        self.ppg_curve.clear()
        self.peak_scatter.clear()
        self.threshold_curve.clear()
    
    def auto_range(self) -> None:
        """Auto-range Y axis."""
        self.plot_widget.enableAutoRange(axis='y')