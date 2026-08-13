# widgets/ppg_plot.py
"""
PPG (Photoplethysmography) visualization widget.
Displays real-time PPG signal with rolling history.
"""

from typing import Optional
import numpy as np

from PySide6.QtWidgets import QWidget
from widgets.plot_base import PlotBase
from models.circular_buffer import CircularBuffer


class PPGPlot(PlotBase):
    """
    Real-time PPG signal plot.
    
    Displays a single curve showing photoplethysmography data
    with automatic scaling and rolling history.
    
    Expected UART format: PPG,value
    Example: PPG,51234
    
    Attributes:
        time_buffer: Circular buffer for timestamps
        signal_buffer: Circular buffer for PPG values
    """
    
    SIGNAL_COLOR = (39, 174, 96)  # Medical green
    
    def __init__(
        self,
        history_duration: float = 10.0,
        sample_rate: float = 25.0,
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize the PPG plot.
        
        Args:
            history_duration: Duration of visible history in seconds
            sample_rate: Expected sample rate in Hz
            parent: Parent widget
        """
        buffer_size = int(history_duration * sample_rate * 1.2)
        
        super().__init__(
            title="PPG Signal",
            y_label="Amplitude",
            x_label="Time (s)",
            history_size=buffer_size,
            show_legend=False,
            parent=parent
        )
        
        # Create data buffers
        self.time_buffer = CircularBuffer(buffer_size)
        self.signal_buffer = CircularBuffer(buffer_size)
        
        # Time tracking
        self._start_time: Optional[float] = None
        self._sample_count: int = 0
        self._sample_rate = sample_rate
        
        # Create curve
        self.add_curve('PPG', self.SIGNAL_COLOR, width=1.5)
        
        # Auto-range for PPG
        self.auto_range_y()
    
    def update(self, value: float, timestamp: Optional[float] = None) -> None:
        """
        Update PPG data with new sample.
        
        Args:
            value: PPG sensor reading
            timestamp: Optional timestamp
        """
        if self._start_time is None:
            self._start_time = timestamp if timestamp else 0.0
        
        if timestamp:
            rel_time = timestamp - self._start_time
        else:
            rel_time = self._sample_count / self._sample_rate
        
        self.time_buffer.append(rel_time)
        self.signal_buffer.append(value)
        
        self.update_data('PPG', self.time_buffer.get(), self.signal_buffer.get())
        
        self._sample_count += 1
    
    def clear(self) -> None:
        """Clear all PPG data."""
        super().clear()
        self.time_buffer.clear()
        self.signal_buffer.clear()
        self._start_time = None
        self._sample_count = 0