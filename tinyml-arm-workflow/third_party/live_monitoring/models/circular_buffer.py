# models/circular_buffer.py
"""
Circular buffer implementation for efficient real-time data storage.
Uses NumPy for fast array operations with constant memory usage.
"""

from typing import Optional, Any
import numpy as np


class CircularBuffer:
    """
    Fixed-size circular buffer for time series data.
    
    Provides O(1) append operations with automatic overwriting
    of oldest data when capacity is reached. Uses NumPy arrays
    for efficient numerical operations.
    
    Attributes:
        _buffer: Internal NumPy array storing the data
        _capacity: Maximum number of elements
        _index: Current write position
        _size: Current number of valid elements
        _full: Whether buffer has wrapped around
    
    Example:
        >>> buf = CircularBuffer(100)
        >>> buf.append(1.5)
        >>> buf.append(2.3)
        >>> data = buf.get()  # Returns array of valid data
    """
    
    def __init__(self, capacity: int, dtype: np.dtype = np.float64) -> None:
        """
        Initialize circular buffer with specified capacity.
        
        Args:
            capacity: Maximum number of elements to store
            dtype: NumPy data type for the buffer (default: float64)
        
        Raises:
            ValueError: If capacity is less than 1
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        
        self._buffer: np.ndarray = np.zeros(capacity, dtype=dtype)
        self._capacity: int = capacity
        self._index: int = 0
        self._size: int = 0
        self._full: bool = False
    
    def append(self, value: Any) -> None:
        """
        Append a single value to the buffer.
        
        If buffer is full, oldest value is overwritten.
        Automatically converts value to buffer's dtype.
        
        Args:
            value: Value to append (must be convertible to buffer dtype)
        """
        self._buffer[self._index] = value
        self._index = (self._index + 1) % self._capacity
        
        if not self._full:
            self._size += 1
            if self._size >= self._capacity:
                self._full = True
    
    def get(self) -> np.ndarray:
        """
        Get all valid data from the buffer in chronological order.
        
        Returns:
            NumPy array of valid data elements, oldest first
        
        Example:
            >>> buf = CircularBuffer(5)
            >>> buf.append(1); buf.append(2); buf.append(3)
            >>> buf.get()
            array([1., 2., 3.])
        """
        if self._size == 0:
            return np.array([], dtype=self._buffer.dtype)
        
        if not self._full:
            return self._buffer[:self._size].copy()
        else:
            # Return data in order: index to end, then start to index
            return np.concatenate([
                self._buffer[self._index:],
                self._buffer[:self._index]
            ])
    
    def clear(self) -> None:
        """Clear all data from the buffer."""
        self._buffer.fill(0)
        self._index = 0
        self._size = 0
        self._full = False
    
    @property
    def capacity(self) -> int:
        """Get the buffer capacity."""
        return self._capacity
    
    @property
    def size(self) -> int:
        """Get the current number of valid elements."""
        return self._size
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._size == 0
    
    def __len__(self) -> int:
        """Get the current number of elements."""
        return self._size
    
    def __repr__(self) -> str:
        """String representation showing capacity and size."""
        return f"CircularBuffer(capacity={self._capacity}, size={self._size})"