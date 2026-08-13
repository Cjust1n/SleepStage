# widgets/feature_history_plot.py
"""
Mini trend graph for feature history visualization.
Small sparkline-style plot showing feature evolution over epochs.
"""

from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont

import pyqtgraph as pg


class FeatureHistoryPlot(QWidget):
    """
    Small trend graph showing feature history over epochs.
    
    Compact sparkline-style plot for monitoring feature trends.
    Designed to be placed alongside feature cards.
    
    Attributes:
        plot_widget: Internal PyQtGraph PlotWidget
        curve: Plot curve for the feature trend
        title_label: Feature name label
    """
    
    def __init__(
        self,
        title: str = "Trend",
        color: Tuple[int, int, int] = (52, 152, 219),
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize trend graph.
        
        Args:
            title: Graph title
            color: RGB color tuple for the trend line
            parent: Parent widget
        """
        super().__init__(parent)
        
        self._title = title
        self._color = color
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Create plot layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # Title
        self.title_label = QLabel(self._title)
        self.title_label.setFont(QFont("Segoe UI", 8))
        self.title_label.setStyleSheet("color: #7f8c8d; padding: 0px;")
        layout.addWidget(self.title_label)
        
        # Mini plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMaximumHeight(80)
        self.plot_widget.setMinimumHeight(60)
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        self.plot_widget.setAntialiasing(True)
        
        # Hide axis labels for compact display
        self.plot_widget.getAxis('left').setStyle(showValues=False)
        self.plot_widget.getAxis('bottom').setStyle(showValues=False)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#d0d0d0', width=1))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#d0d0d0', width=1))
        
        # Create curve
        pen = pg.mkPen(color=self._color, width=1.5)
        self.curve = self.plot_widget.plot([], [], pen=pen)
        
        layout.addWidget(self.plot_widget)
        
        # Styling
        self.setStyleSheet("""
            FeatureHistoryPlot {
                background-color: #ffffff;
                border: 1px solid #e8e8e8;
                border-radius: 4px;
            }
        """)
    
    def update_data(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Update trend data.
        
        Args:
            x_data: Epoch numbers (x-axis)
            y_data: Feature values (y-axis)
        """
        if len(x_data) > 0 and len(y_data) > 0:
            self.curve.setData(x_data, y_data)
            self.plot_widget.autoRange()
    
    def clear(self) -> None:
        """Clear trend data."""
        self.curve.clear()