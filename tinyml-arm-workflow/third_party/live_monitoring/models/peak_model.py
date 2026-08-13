# models/peak_model.py
"""
Peak detection data models for real-time analysis.
Stores PPG peaks, RR intervals, heart rate, and threshold data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class PeakData:
    """
    Single detected peak information.
    
    Attributes:
        timestamp: Relative time in seconds
        value: PPG value at peak
        index: Sample index in buffer
        accepted: Whether peak was accepted (not rejected/refractory)
    """
    timestamp: float = 0.0
    value: float = 0.0
    index: int = 0
    accepted: bool = True


@dataclass
class RRInterval:
    """
    Single RR interval measurement.
    
    Attributes:
        timestamp: Time of measurement
        rr_ms: RR interval in milliseconds
        hr_bpm: Instantaneous heart rate
    """
    timestamp: float = 0.0
    rr_ms: float = 0.0
    hr_bpm: float = 0.0


@dataclass
class PeakStatistics:
    """
    Real-time peak detection statistics.
    
    Attributes:
        current_hr: Current heart rate (bpm)
        avg_hr: Average heart rate (bpm)
        total_peaks: Total peaks detected
        accepted_peaks: Accepted peaks count
        rejected_peaks: Rejected peaks count
        rr_count: Number of RR intervals computed
        threshold_value: Current adaptive threshold
        refractory_active: Whether in refractory period
        peak_amplitude: Average peak amplitude
        last_rr_ms: Most recent RR interval
    """
    current_hr: float = 0.0
    avg_hr: float = 0.0
    total_peaks: int = 0
    accepted_peaks: int = 0
    rejected_peaks: int = 0
    rr_count: int = 0
    threshold_value: float = 0.0
    refractory_active: bool = False
    peak_amplitude: float = 0.0
    last_rr_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_hr": round(self.current_hr, 1),
            "avg_hr": round(self.avg_hr, 1),
            "total_peaks": self.total_peaks,
            "accepted_peaks": self.accepted_peaks,
            "rejected_peaks": self.rejected_peaks,
            "rr_count": self.rr_count,
            "threshold_value": round(self.threshold_value, 1),
            "refractory_active": self.refractory_active,
            "peak_amplitude": round(self.peak_amplitude, 1),
            "last_rr_ms": round(self.last_rr_ms, 1),
        }
    
    def clear(self) -> None:
        """Reset all statistics."""
        self.current_hr = 0.0
        self.avg_hr = 0.0
        self.total_peaks = 0
        self.accepted_peaks = 0
        self.rejected_peaks = 0
        self.rr_count = 0
        self.threshold_value = 0.0
        self.refractory_active = False
        self.peak_amplitude = 0.0
        self.last_rr_ms = 0.0


@dataclass
class PeakModel:
    """
    Complete peak detection state from UART data.
    
    Parses PEAK, THRESH, RR, HR packets and maintains
    rolling history for visualization.
    """
    
    # Current values
    latest_ppg_value: float = 0.0
    latest_peak_value: float = 0.0
    latest_threshold: float = 0.0
    latest_rr_ms: float = 0.0
    latest_hr: float = 0.0
    
    # History
    ppg_values: List[float] = field(default_factory=list)
    ppg_times: List[float] = field(default_factory=list)
    peaks: List[PeakData] = field(default_factory=list)
    thresholds: List[float] = field(default_factory=list)
    rr_intervals: List[RRInterval] = field(default_factory=list)
    heart_rates: List[tuple] = field(default_factory=list)  # (time, hr)
    
    # Statistics
    statistics: PeakStatistics = field(default_factory=PeakStatistics)
    
    # Configuration
    MAX_HISTORY: int = 500  # 10 seconds @ 50Hz
    
    def add_ppg(self, value: float, timestamp: float) -> None:
        """Add raw PPG sample."""
        self.latest_ppg_value = value
        self.ppg_values.append(value)
        self.ppg_times.append(timestamp)
        
        # Trim history
        if len(self.ppg_values) > self.MAX_HISTORY:
            self.ppg_values = self.ppg_values[-self.MAX_HISTORY:]
            self.ppg_times = self.ppg_times[-self.MAX_HISTORY:]
    
    def add_peak(self, value: float, timestamp: float, accepted: bool = True) -> None:
        """Add detected peak."""
        self.latest_peak_value = value
        peak = PeakData(timestamp=timestamp, value=value, accepted=accepted)
        self.peaks.append(peak)
        
        # Update statistics
        self.statistics.total_peaks += 1
        if accepted:
            self.statistics.accepted_peaks += 1
        else:
            self.statistics.rejected_peaks += 1
        
        # Trim peaks (keep last 50)
        if len(self.peaks) > 50:
            self.peaks = self.peaks[-50:]
    
    def add_threshold(self, value: float) -> None:
        """Add threshold value."""
        self.latest_threshold = value
        self.statistics.threshold_value = value
        self.thresholds.append(value)
        
        if len(self.thresholds) > self.MAX_HISTORY:
            self.thresholds = self.thresholds[-self.MAX_HISTORY:]
    
    def add_rr(self, rr_ms: float, timestamp: float) -> None:
        """Add RR interval."""
        self.latest_rr_ms = rr_ms
        hr = 60000.0 / rr_ms if rr_ms > 0 else 0.0
        
        rr = RRInterval(timestamp=timestamp, rr_ms=rr_ms, hr_bpm=hr)
        self.rr_intervals.append(rr)
        
        # Update statistics
        self.statistics.rr_count += 1
        self.statistics.last_rr_ms = rr_ms
        self.statistics.current_hr = hr
        
        # Update average HR
        if self.rr_intervals:
            hrs = [r.hr_bpm for r in self.rr_intervals[-10:]]
            self.statistics.avg_hr = sum(hrs) / len(hrs)
        
        # Trim
        if len(self.rr_intervals) > 100:
            self.rr_intervals = self.rr_intervals[-100:]
    
    def add_hr(self, hr: float, timestamp: float) -> None:
        """Add heart rate value."""
        self.latest_hr = hr
        self.heart_rates.append((timestamp, hr))
        self.statistics.current_hr = hr
        
        if len(self.heart_rates) > 300:  # 5 minutes @ 1Hz
            self.heart_rates = self.heart_rates[-300:]
    
    @classmethod
    def from_uart_peak(cls, payload: List[str], timestamp: float) -> Optional[Dict]:
        """Parse PEAK packet."""
        try:
            if len(payload) >= 1:
                return {
                    "type": "peak",
                    "value": float(payload[0]),
                    "timestamp": timestamp
                }
        except (ValueError, IndexError):
            pass
        return None
    
    @classmethod
    def from_uart_thresh(cls, payload: List[str]) -> Optional[float]:
        """Parse THRESH packet."""
        try:
            if len(payload) >= 1:
                return float(payload[0])
        except (ValueError, IndexError):
            pass
        return None
    
    @classmethod
    def from_uart_rr(cls, payload: List[str], timestamp: float) -> Optional[float]:
        """Parse RR packet."""
        try:
            if len(payload) >= 1:
                return float(payload[0])
        except (ValueError, IndexError):
            pass
        return None
    
    @classmethod
    def from_uart_hr(cls, payload: List[str], timestamp: float) -> Optional[float]:
        """Parse HR packet."""
        try:
            if len(payload) >= 1:
                return float(payload[0])
        except (ValueError, IndexError):
            pass
        return None
    
    def get_ppg_arrays(self) -> tuple:
        """Get PPG data as numpy arrays."""
        return (
            np.array(self.ppg_times),
            np.array(self.ppg_values)
        )
    
    def get_peak_arrays(self) -> tuple:
        """Get peak data as numpy arrays."""
        if not self.peaks:
            return np.array([]), np.array([])
        times = [p.timestamp for p in self.peaks]
        values = [p.value for p in self.peaks]
        return np.array(times), np.array(values)
    
    def get_rr_arrays(self) -> tuple:
        """Get RR interval data."""
        if not self.rr_intervals:
            return np.array([]), np.array([])
        indices = np.arange(len(self.rr_intervals))
        values = np.array([r.rr_ms for r in self.rr_intervals])
        return indices, values
    
    def get_hr_arrays(self) -> tuple:
        """Get heart rate data."""
        if not self.heart_rates:
            return np.array([]), np.array([])
        times = np.array([h[0] for h in self.heart_rates])
        values = np.array([h[1] for h in self.heart_rates])
        return times, values
    
    def clear(self) -> None:
        """Clear all data."""
        self.ppg_values.clear()
        self.ppg_times.clear()
        self.peaks.clear()
        self.thresholds.clear()
        self.rr_intervals.clear()
        self.heart_rates.clear()
        self.statistics.clear()