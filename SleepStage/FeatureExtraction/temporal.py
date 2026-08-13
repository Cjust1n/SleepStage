"""
temporal.py

Temporal features.

Author: Justin
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

LOCAL_ZONE = ZoneInfo("America/New_York")

class TemporalFeatureExtractor:

    def __init__(self, epoch_length=30):

        self.epoch_length = epoch_length

    def extract(
        self,
        epoch,
        total_epochs,
        rec_start,
    ):
        """
        Parameters
        ----------
        epoch : int

        total_epochs : int

        rec_start : unix timestamp
        """

        relative_position = (
            epoch / (total_epochs - 1)
            if total_epochs > 1 else 0
        )
        current_time = (
            rec_start +
            epoch * self.epoch_length
        )

        dt = datetime.fromtimestamp(
            current_time,
            tz=LOCAL_ZONE,
        )

        hour = dt.hour

        minute = dt.minute

        time_of_night = hour + minute / 60
        theta = 2.0 * np.pi * (time_of_night / 24.0)

        return {
            "time_of_night": time_of_night,
            "sin_time_of_night": float(np.sin(theta)),
            "cos_time_of_night": float(np.cos(theta)),
            "relative_position": relative_position,
        }
    @staticmethod
    def feature_names():

        return [
            "time_of_night",
            "sin_time_of_night",
            "cos_time_of_night",
            "relative_position",
        ]


    def to_vector(self, feature_dict):

        return np.array(
            [feature_dict[name]
             for name in self.feature_names()],
            dtype=np.float32,
        )
