# models/feature_model.py
"""
Feature model for Sleep Stage Development Studio.
Stores all 18 extracted features from a single epoch using dataclass.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class FeatureModel:
    """
    Data model for one epoch of extracted sleep features.
    
    Contains all 18 features extracted by the firmware every 30-second epoch.
    Features are in the exact order received via UART (matching the model's
    [1, 30, 18] input ordering defined in
    SleepStage/configs/train_selected_features.yaml).
    
    Attributes:
        relative_position: Position within recording (0.0 to 1.0)
        sd2: Poincaré plot SD2 (HRV nonlinear)
        sin_time_of_night: sin(2*pi * time_of_night/24)
        rolling_mean_hr: Rolling mean of HR trend window
        rmssd: Root mean square of successive RR differences (ms)
        time_of_night: Time in hours since recording start
        energy: Movement energy
        rolling_hr_range: Rolling max-min of HR trend window
        acceleration_jerk: mean(|diff(mag)|) of accelerometer
        rolling_mean_acc: Rolling mean of accel mean history
        rms: RMS of accelerometer magnitude
        lf: Low frequency power (HRV frequency domain)
        rolling_std_acc: Rolling std of accel mean history
        zero_crossing: Number of zero crossings in accelerometer
        hr_slope: Linear fit slope of HR trend window
        hf: High frequency power (HRV frequency domain)
        lf_hf: LF/HF ratio
        hr_delta: Change in mean HR vs previous epoch
        epoch_index: Epoch number (set externally)
    """
    
    # Feature values in UART order (18 features)
    relative_position: float = 0.0
    sd2: float = 0.0
    sin_time_of_night: float = 0.0
    rolling_mean_hr: float = 0.0
    rmssd: float = 0.0
    time_of_night: float = 0.0
    energy: float = 0.0
    rolling_hr_range: float = 0.0
    acceleration_jerk: float = 0.0
    rolling_mean_acc: float = 0.0
    rms: float = 0.0
    lf: float = 0.0
    rolling_std_acc: float = 0.0
    zero_crossing: float = 0.0
    hr_slope: float = 0.0
    hf: float = 0.0
    lf_hf: float = 0.0
    hr_delta: float = 0.0
    
    # Metadata
    epoch_index: int = 0
    
    # Feature metadata: (name, unit, decimal_places, is_integer)
    FEATURE_META: Dict[int, tuple] = field(default_factory=lambda: {
        0:  ("Relative Position", "", 6, False),
        1:  ("SD2", "ms", 1, False),
        2:  ("Sin Time of Night", "", 6, False),
        3:  ("Rolling Mean HR", "bpm", 1, False),
        4:  ("RMSSD", "ms", 1, False),
        5:  ("Time of Night", "h", 2, False),
        6:  ("Energy", "", 1, False),
        7:  ("Rolling HR Range", "bpm", 1, False),
        8:  ("Accel Jerk", "", 4, False),
        9:  ("Rolling Mean Acc", "g", 4, False),
        10: ("RMS", "g", 4, False),
        11: ("LF", "ms²", 1, False),
        12: ("Rolling Std Acc", "g", 4, False),
        13: ("Zero Crossing", "", 0, True),
        14: ("HR Slope", "", 4, False),
        15: ("HF", "ms²", 1, False),
        16: ("LF/HF", "", 2, False),
        17: ("HR Delta", "bpm", 1, False),
    })
    
    @classmethod
    def from_uart(cls, payload: List[str]) -> Optional['FeatureModel']:
        """
        Create FeatureModel from UART FEATURE packet payload.
        
        Args:
            payload: List of string values from FEATURE packet
        
        Returns:
            FeatureModel instance or None if parsing fails
        
        Example:
            >>> payload = ["0.003128", "37.44", "-0.31", "72.5", ...]
            >>> model = FeatureModel.from_uart(payload)
        """
        if len(payload) < 18:
            return None
        
        try:
            values = [float(v.strip()) for v in payload[:18]]
            
            return cls(
                relative_position=values[0],
                sd2=values[1],
                sin_time_of_night=values[2],
                rolling_mean_hr=values[3],
                rmssd=values[4],
                time_of_night=values[5],
                energy=values[6],
                rolling_hr_range=values[7],
                acceleration_jerk=values[8],
                rolling_mean_acc=values[9],
                rms=values[10],
                lf=values[11],
                rolling_std_acc=values[12],
                zero_crossing=values[13],
                hr_slope=values[14],
                hf=values[15],
                lf_hf=values[16],
                hr_delta=values[17],
            )
        except (ValueError, IndexError):
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert features to dictionary.
        
        Returns:
            Dictionary with feature names as keys
        """
        return {
            "relative_position": self.relative_position,
            "sd2": self.sd2,
            "sin_time_of_night": self.sin_time_of_night,
            "rolling_mean_hr": self.rolling_mean_hr,
            "rmssd": self.rmssd,
            "time_of_night": self.time_of_night,
            "energy": self.energy,
            "rolling_hr_range": self.rolling_hr_range,
            "acceleration_jerk": self.acceleration_jerk,
            "rolling_mean_acc": self.rolling_mean_acc,
            "rms": self.rms,
            "lf": self.lf,
            "rolling_std_acc": self.rolling_std_acc,
            "zero_crossing": self.zero_crossing,
            "hr_slope": self.hr_slope,
            "hf": self.hf,
            "lf_hf": self.lf_hf,
            "hr_delta": self.hr_delta,
            "epoch": self.epoch_index,
        }
    
    def get_value(self, index: int) -> float:
        """
        Get feature value by index (0-17).
        
        Args:
            index: Feature index (0-based)
        
        Returns:
            Feature value as float
        """
        values = [
            self.relative_position, self.sd2, self.sin_time_of_night,
            self.rolling_mean_hr, self.rmssd, self.time_of_night,
            self.energy, self.rolling_hr_range, self.acceleration_jerk,
            self.rolling_mean_acc, self.rms, self.lf,
            self.rolling_std_acc, self.zero_crossing, self.hr_slope,
            self.hf, self.lf_hf, self.hr_delta
        ]
        return values[index]
    
    def get_formatted_value(self, index: int) -> str:
        """
        Get human-readable formatted feature value.
        
        Args:
            index: Feature index (0-17)
        
        Returns:
            Formatted string
        """
        name, unit, decimals, is_int = self.FEATURE_META.get(index, ("Unknown", "", 1, False))
        value = self.get_value(index)
        
        # Special formatting for time_of_night
        if index == 5:
            hours = int(value)
            minutes = int((value - hours) * 60)
            return f"{hours:02d}:{minutes:02d}"
        
        if is_int:
            return f"{int(value)}"
        else:
            return f"{value:.{decimals}f}"
    
    def get_display_name(self, index: int) -> str:
        """
        Get human-readable feature name.
        
        Args:
            index: Feature index (0-17)
        
        Returns:
            Display name string
        """
        name, _, _, _ = self.FEATURE_META.get(index, ("Unknown", "", 1, False))
        return name
    
    def get_unit(self, index: int) -> str:
        """
        Get feature unit.
        
        Args:
            index: Feature index (0-17)
        
        Returns:
            Unit string (empty string if no unit)
        """
        _, unit, _, _ = self.FEATURE_META.get(index, ("", "", 1, False))
        return unit
    
    def clear(self) -> None:
        """Reset all feature values to zero."""
        self.relative_position = 0.0
        self.sd2 = 0.0
        self.sin_time_of_night = 0.0
        self.rolling_mean_hr = 0.0
        self.rmssd = 0.0
        self.time_of_night = 0.0
        self.energy = 0.0
        self.rolling_hr_range = 0.0
        self.acceleration_jerk = 0.0
        self.rolling_mean_acc = 0.0
        self.rms = 0.0
        self.lf = 0.0
        self.rolling_std_acc = 0.0
        self.zero_crossing = 0.0
        self.hr_slope = 0.0
        self.hf = 0.0
        self.lf_hf = 0.0
        self.hr_delta = 0.0
        self.epoch_index = 0
