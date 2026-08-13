# widgets/plot_base.py
"""
Base plot widget providing common functionality for all visualization widgets.
Reusable foundation for PyQtGraph-based plots with consistent styling.
"""

from typing import Optional, Tuple
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont

import pyqtgraph as pg


class PlotBase(QWidget):
    """
    Reusable base class for PyQtGraph plot widgets.
    
    Provides a consistent interface for creating real-time plots
    with configurable titles, labels, grid, and auto-scrolling.
    
    Attributes:
        plot_widget: Internal PyQtGraph PlotWidget
        title_label: Widget title display
        curves: Dictionary of plot curves keyed by name
    """
    
    def __init__(
        self,
        title: str = "Plot",
        y_label: str = "Value",
        x_label: str = "Time (s)",
        history_size: int = 500,
        show_legend: bool = False,
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize the base plot widget.
        
        Args:
            title: Plot title displayed above the graph
            y_label: Label for the Y axis
            x_label: Label for the X axis
            history_size: Number of data points to display (rolling window)
            show_legend: Whether to show the plot legend
            parent: Parent widget
        """
        super().__init__(parent)
        
        self._history_size = history_size
        
        # Store curves
        self.curves: dict = {}
        
        # Setup UI
        self._setup_ui(title, y_label, x_label, show_legend)
    
    def _setup_ui(self, title: str, y_label: str, x_label: str, show_legend: bool) -> None:
        """Create and arrange the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Title label
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.title_label.setStyleSheet("color: #2c3e50; padding: 2px 0px;")
        layout.addWidget(self.title_label)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # White background
        layout.addWidget(self.plot_widget)
        
        # Configure plot
        self.plot_widget.setLabel('left', y_label)
        self.plot_widget.setLabel('bottom', x_label)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAntialiasing(True)
        
        # Professional axis styling
        axis_pen = pg.mkPen(color='#34495e', width=1)
        self.plot_widget.getAxis('left').setPen(axis_pen)
        self.plot_widget.getAxis('bottom').setPen(axis_pen)
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#2c3e50'))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#2c3e50'))
        
        # Font sizes
        font = QFont("Segoe UI", 9)
        self.plot_widget.getAxis('left').setTickFont(font)
        self.plot_widget.getAxis('bottom').setTickFont(font)
        
        # Legend
        if show_legend:
            self.legend = self.plot_widget.addLegend(
                offset=(10, 10),
                brush=pg.mkBrush(255, 255, 255, 200),
                pen=pg.mkPen(200, 200, 200),
                labelTextSize='10pt'
            )
    
    def add_curve(
        self,
        name: str,
        color: Tuple[int, int, int],
        width: float = 1.5
    ) -> pg.PlotDataItem:
        """
        Add a new curve to the plot.
        
        Args:
            name: Curve name (used as key and legend label)
            color: RGB color tuple (0-255)
            width: Line width
        
        Returns:
            PlotDataItem for the new curve
        """
        if name in self.curves:
            raise ValueError(f"Curve '{name}' already exists")
        
        pen = pg.mkPen(color=color, width=width)
        curve = self.plot_widget.plot([], [], pen=pen, name=name)
        self.curves[name] = curve
        return curve
    
    def update_data(self, curve_name: str, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Update data for a specific curve.
        
        Args:
            curve_name: Name of the curve to update
            x_data: X-axis data array
            y_data: Y-axis data array
        """
        if curve_name in self.curves:
            self.curves[curve_name].setData(x_data, y_data)
    
    def set_y_range(self, min_val: float, max_val: float) -> None:
        """Set fixed Y-axis range."""
        self.plot_widget.setYRange(min_val, max_val)
    
    def auto_range_y(self) -> None:
        """Enable automatic Y-axis ranging."""
        self.plot_widget.enableAutoRange(axis='y')
    
    def clear(self) -> None:
        """Clear all data from all curves."""
        for curve in self.curves.values():
            curve.clear()
    
    @property
    def history_size(self) -> int:
        """Get the history size."""
        return self._history_size