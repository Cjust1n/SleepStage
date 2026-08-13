# models/feature_history.py
"""
Feature history storage for Sleep Stage Development Studio.
Maintains rolling history of feature values for trend visualization.
"""

from typing import List, Optional
import numpy as np
from models.circular_buffer import CircularBuffer
from models.feature_model import FeatureModel


class FeatureHistory:
    """
    Stores rolling history for all 18 features.
    
    Maintains the last N epochs of feature data for trend graphs
    and statistical analysis. Uses circular buffers for each feature.
    
    Attributes:
        max_epochs: Maximum number of epochs to store
        feature_buffers: List of CircularBuffer, one per feature
        epoch_indices: CircularBuffer for epoch numbers
    """
    
    NUM_FEATURES = 18
    
    def __init__(self, max_epochs: int = 100) -> None:
        """
        Initialize feature history storage.
        
        Args:
            max_epochs: Maximum number of epochs to retain
        """
        self.max_epochs = max_epochs
        self.feature_buffers: List[CircularBuffer] = [
            CircularBuffer(max_epochs) for _ in range(self.NUM_FEATURES)
        ]
        self.epoch_indices = CircularBuffer(max_epochs)
    
    def append(self, features: FeatureModel) -> None:
        """
        Append a new epoch of features to history.
        
        Args:
            features: FeatureModel instance with current epoch data
        """
        self.epoch_indices.append(features.epoch_index)
        
        for i in range(self.NUM_FEATURES):
            self.feature_buffers[i].append(features.get_value(i))
    
    def get_history(self, feature_index: int) -> np.ndarray:
        """
        Get history for a specific feature.
        
        Args:
            feature_index: Feature index (0-17)
        
        Returns:
            NumPy array of feature values in chronological order
        """
        if 0 <= feature_index < self.NUM_FEATURES:
            return self.feature_buffers[feature_index].get()
        return np.array([])
    
    def get_epochs(self) -> np.ndarray:
        """
        Get epoch indices.
        
        Returns:
            NumPy array of epoch numbers
        """
        return self.epoch_indices.get()
    
    def get_latest(self) -> Optional[FeatureModel]:
        """
        Get the most recent feature values as FeatureModel.
        
        Returns:
            FeatureModel or None if no data
        """
        if self.epoch_indices.is_empty:
            return None
        
        model = FeatureModel()
        for i in range(self.NUM_FEATURES):
            buf = self.feature_buffers[i]
            if not buf.is_empty:
                # Get last value
                data = buf.get()
                setattr(model, self._get_attr_name(i), float(data[-1]))
        
        return model
    
    def clear(self) -> None:
        """Clear all feature history."""
        self.epoch_indices.clear()
        for buf in self.feature_buffers:
            buf.clear()
    
    @property
    def epoch_count(self) -> int:
        """Get number of stored epochs."""
        return self.epoch_indices.size
    
    @staticmethod
    def _get_attr_name(index: int) -> str:
        """Map feature index to FeatureModel attribute name."""
        names = [
            "relative_position", "sd2", "sin_time_of_night", "rolling_mean_hr",
            "rmssd", "time_of_night", "energy", "rolling_hr_range",
            "acceleration_jerk", "rolling_mean_acc", "rms", "lf",
            "rolling_std_acc", "zero_crossing", "hr_slope",
            "hf", "lf_hf", "hr_delta"
        ]
        return names[index] if 0 <= index < len(names) else ""
