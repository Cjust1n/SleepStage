# widgets/accel_plot.py
"""
Accelerometer visualization widget for Sleep Stage Development Studio.
Displays real-time 3-axis accelerometer data with rolling history.
"""

from typing import Optional
import numpy as np

from PySide6.QtWidgets import QWidget
from widgets.plot_base import PlotBase
from models.circular_buffer import CircularBuffer


class AccelPlot(PlotBase):
    """
    Real-time accelerometer plot showing X, Y, and Z axes.
    
    Displays three colored curves on a single graph with legend.
    Uses circular buffers for efficient data storage.
    
    Expected UART format: ACCEL,x,y,z
    Example: ACCEL,0.02,-0.01,0.98
    
    Attributes:
        time_buffer: Circular buffer for timestamps
        x_buffer: Circular buffer for X-axis data
        y_buffer: Circular buffer for Y-axis data
        z_buffer: Circular buffer for Z-axis data
    """
    
    # Color scheme for accelerometer axes
    COLORS = {
        'X': (231, 76, 60),   # Red
        'Y': (46, 204, 113),  # Green
        'Z': (52, 152, 219),  # Blue
    }
    
    def __init__(
        self,
        history_duration: float = 10.0,  # seconds
        sample_rate: float = 50.0,       # Hz
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize the accelerometer plot.
        
        Args:
            history_duration: Duration of visible history in seconds
            sample_rate: Expected sample rate in Hz
            parent: Parent widget
        """
        buffer_size = int(history_duration * sample_rate * 1.2)
        
        super().__init__(
            title="Accelerometer",
            y_label="Acceleration (g)",
            x_label="Time (s)",
            history_size=buffer_size,
            show_legend=True,
            parent=parent
        )
        
        # Create data buffers
        self.time_buffer = CircularBuffer(buffer_size)
        self.x_buffer = CircularBuffer(buffer_size)
        self.y_buffer = CircularBuffer(buffer_size)
        self.z_buffer = CircularBuffer(buffer_size)
        
        # Time tracking
        self._start_time: Optional[float] = None
        self._sample_count: int = 0
        self._sample_rate = sample_rate
        
        # Create curves
        self.add_curve('X', self.COLORS['X'], width=1.5)
        self.add_curve('Y', self.COLORS['Y'], width=1.5)
        self.add_curve('Z', self.COLORS['Z'], width=1.5)
        
        # Set default Y range for accelerometer
        self.set_y_range(-2.0, 2.0)
    
    def update(self, x: float, y: float, z: float, timestamp: Optional[float] = None) -> None:
        """
        Update accelerometer data with new sample.
        
        Args:
            x: X-axis acceleration in g
            y: Y-axis acceleration in g
            z: Z-axis acceleration in g
            timestamp: Optional timestamp (auto-generated if None)
        """
        if self._start_time is None:
            self._start_time = timestamp if timestamp else 0.0
        
        if timestamp:
            rel_time = timestamp - self._start_time
        else:
            rel_time = self._sample_count / self._sample_rate
        
        self.time_buffer.append(rel_time)
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        self.z_buffer.append(z)
        
        time_data = self.time_buffer.get()
        self.update_data('X', time_data, self.x_buffer.get())
        self.update_data('Y', time_data, self.y_buffer.get())
        self.update_data('Z', time_data, self.z_buffer.get())
        
        self._sample_count += 1
    
    def clear(self) -> None:
        """Clear all accelerometer data."""
        super().clear()
        self.time_buffer.clear()
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.z_buffer.clear()
        self._start_time = None
        self._sample_count = 0