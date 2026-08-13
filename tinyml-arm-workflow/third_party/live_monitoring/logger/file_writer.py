# logger/file_writer.py
"""
Buffered file writer for recording UART data.
Provides efficient writing with periodic flushing to prevent GUI lag.
"""

from typing import Optional, List
import os
from PySide6.QtCore import QObject, QTimer, Signal


class FileWriter(QObject):
    """
    Buffered file writer with automatic periodic flushing.
    
    Accumulates lines in memory buffer and flushes to disk
    periodically to avoid I/O overhead on every packet.
    
    Signals:
        error_occurred: Emitted when write error occurs
        bytes_written: Emitted with total bytes written
    
    Attributes:
        filepath: Path to the output file
        _buffer: List of buffered lines
        _file_handle: Open file handle
        _total_bytes: Total bytes written
        _flush_interval: Milliseconds between flushes
        _max_buffer_size: Maximum buffer lines before forced flush
    """
    
    error_occurred = Signal(str)
    
    def __init__(
        self,
        flush_interval: int = 1000,  # 1 second
        max_buffer_size: int = 500,
        parent: Optional[QObject] = None
    ) -> None:
        """
        Initialize file writer.
        
        Args:
            flush_interval: Milliseconds between auto-flushes
            max_buffer_size: Max lines before forced flush
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self.filepath: str = ""
        self._buffer: List[str] = []
        self._file_handle = None
        self._total_bytes: int = 0
        self._flush_interval = flush_interval
        self._max_buffer_size = max_buffer_size
        self._closed: bool = True
        
        # Setup flush timer
        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self._flush)
    
    def open(self, filepath: str, headers: Optional[str] = None) -> bool:
        """
        Open file for writing.
        
        Args:
            filepath: Path to output file
            headers: Optional header line to write first
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Open file
            self._file_handle = open(filepath, 'w', encoding='utf-8')
            self.filepath = filepath
            self._closed = False
            self._total_bytes = 0
            self._buffer.clear()
            
            # Write header
            if headers:
                self._file_handle.write(headers + '\n')
                self._total_bytes += len(headers) + 1
            
            # Start flush timer
            self._flush_timer.start(self._flush_interval)
            
            return True
            
        except (OSError, IOError) as e:
            self.error_occurred.emit(f"Failed to open {filepath}: {str(e)}")
            self._file_handle = None
            self._closed = True
            return False
    
    def write_line(self, line: str) -> None:
        """
        Write a line to the buffer.
        
        Args:
            line: Line to write (without newline)
        """
        if self._closed or not self._file_handle:
            return
        
        self._buffer.append(line)
        
        # Force flush if buffer is full
        if len(self._buffer) >= self._max_buffer_size:
            self._flush()
    
    def _flush(self) -> None:
        """Flush buffer to disk."""
        if self._closed or not self._file_handle or not self._buffer:
            return
        
        try:
            # Join buffer and write
            data = '\n'.join(self._buffer) + '\n'
            self._file_handle.write(data)
            self._file_handle.flush()
            
            self._total_bytes += len(data)
            self._buffer.clear()
            
        except (OSError, IOError) as e:
            self.error_occurred.emit(f"Write error: {str(e)}")
    
    def close(self) -> None:
        """Close file and flush remaining data."""
        if self._closed:
            return
        
        self._flush_timer.stop()
        self._flush()  # Final flush
        
        try:
            if self._file_handle:
                self._file_handle.close()
        except Exception:
            pass
        
        self._file_handle = None
        self._closed = True
        self._buffer.clear()
    
    @property
    def bytes_written(self) -> int:
        """Get total bytes written to disk."""
        return self._total_bytes
    
    @property
    def buffer_size(self) -> int:
        """Get current buffer line count."""
        return len(self._buffer)
    
    @property
    def is_open(self) -> bool:
        """Check if file is open."""
        return not self._closed and self._file_handle is not None
    
    def __del__(self) -> None:
        """Ensure file is closed on destruction."""
        self.close()