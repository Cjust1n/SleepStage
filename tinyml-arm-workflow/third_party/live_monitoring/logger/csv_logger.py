# logger/csv_logger.py
"""
CSV logger for structured sensor data.
Manages separate CSV files for accelerometer, PPG, and features.
"""

from typing import Optional, Dict
import os
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from logger.file_writer import FileWriter


class CsvLogger(QObject):
    """
    Manages multiple CSV log files for different packet types.
    
    Creates and maintains separate CSV files with appropriate headers
    for each data type received from the device.
    
    Signals:
        error_occurred: Emitted on write errors
    
    Attributes:
        writers: Dictionary of FileWriter instances by data type
        session_path: Current session directory path
        start_timestamp: Recording start time for relative timestamps
    """
    
    error_occurred = Signal(str)
    
    # CSV headers for each data type
    HEADERS = {
        'accel': 'timestamp,ax,ay,az',
        'ppg': 'timestamp,ppg_raw',
'feature': (
            'timestamp,relative_position,sd2,sin_time_of_night,'
            'rolling_mean_hr,rmssd,time_of_night,energy,rolling_hr_range,'
            'acceleration_jerk,rolling_mean_acc,rms,lf,rolling_std_acc,'
            'zero_crossing,hr_slope,hf,lf_hf,hr_delta'
        ),
        'prediction': 'timestamp,stage,confidence,stage_name',
    }
    
    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Initialize CSV logger."""
        super().__init__(parent)
        
        self.writers: Dict[str, FileWriter] = {}
        self.session_path: str = ""
        self.start_timestamp: Optional[datetime] = None
    
    def start_session(self, session_path: str) -> bool:
        """
        Create CSV files for a new recording session.
        
        Args:
            session_path: Path to session directory
        
        Returns:
            True if all files created successfully
        """
        self.session_path = session_path
        self.start_timestamp = datetime.now()
        
        success = True
        
        for data_type, headers in self.HEADERS.items():
            filepath = os.path.join(session_path, f"{data_type}.csv")
            writer = FileWriter(flush_interval=2000, max_buffer_size=200)
            
            if writer.open(filepath, headers):
                self.writers[data_type] = writer
            else:
                success = False
        
        return success
    
    def log_accel(self, x: float, y: float, z: float) -> None:
        """
        Log accelerometer sample.
        
        Args:
            x: X-axis acceleration (g)
            y: Y-axis acceleration (g)
            z: Z-axis acceleration (g)
        """
        if 'accel' in self.writers:
            timestamp = self._relative_timestamp()
            line = f"{timestamp:.3f},{x:.4f},{y:.4f},{z:.4f}"
            self.writers['accel'].write_line(line)
    
    def log_ppg(self, value: float) -> None:
        """
        Log PPG sample.
        
        Args:
            value: PPG sensor reading
        """
        if 'ppg' in self.writers:
            timestamp = self._relative_timestamp()
            line = f"{timestamp:.3f},{value:.1f}"
            self.writers['ppg'].write_line(line)
    
    def log_feature(self, features: list) -> None:
        """
        Log feature vector.
        
        Args:
            features: List of 18 feature values
        """
        if 'feature' in self.writers and len(features) >= 18:
            timestamp = self._relative_timestamp()
            values = ','.join(str(f) for f in features[:18])
            line = f"{timestamp:.3f},{values}"
            self.writers['feature'].write_line(line)
    
    def log_prediction(self, stage: int, confidence: float, stage_name: str = "") -> None:
        """
        Log sleep stage prediction.
        
        Args:
            stage: Predicted stage (0-3)
            confidence: Prediction confidence (0-1)
            stage_name: Human-readable stage name
        """
        if 'prediction' in self.writers:
            timestamp = self._relative_timestamp()
            line = f"{timestamp:.3f},{stage},{confidence:.3f},{stage_name}"
            self.writers['prediction'].write_line(line)
    
    def stop_session(self) -> None:
        """Close all CSV files."""
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
        self.session_path = ""
        self.start_timestamp = None
    
    def _relative_timestamp(self) -> float:
        """
        Get relative timestamp in seconds since recording start.
        
        Returns:
            Seconds elapsed since start_timestamp
        """
        if self.start_timestamp is None:
            return 0.0
        return (datetime.now() - self.start_timestamp).total_seconds()
    
    @property
    def is_recording(self) -> bool:
        """Check if CSV logging is active."""
        return len(self.writers) > 0