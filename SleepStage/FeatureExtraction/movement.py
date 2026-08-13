"""
movement.py

Extract movement features from one epoch.

Author: Justin
"""

import numpy as np
from .utils import (
    magnitude,
    rms,
    signal_energy,
    zero_crossing,
    movement_count,
)


class MovementFeatureExtractor:

    def __init__(self):
        self._history_mean_acc: list[float] = []
        self._history_std_acc: list[float] = []

    def extract(self, motion_epoch):

        x = motion_epoch["x"].to_numpy(dtype=np.float32)

        y = motion_epoch["y"].to_numpy(dtype=np.float32)

        z = motion_epoch["z"].to_numpy(dtype=np.float32)

        mag = np.sqrt(x*x + y*y + z*z)

        rms = np.sqrt(np.mean(mag**2))

        energy = np.sum(mag**2)

        variance = np.var(mag)

        mean = np.mean(mag)

        std = np.std(mag)

        movement_count = np.sum(np.abs(np.diff(mag)) > 0.05)
        movement_ratio = float(np.mean(np.abs(mag - mean) > 0.05)) if len(mag) else 0.0
        acceleration_jerk = float(np.mean(np.abs(np.diff(mag)))) if len(mag) > 1 else 0.0

        centered = mag - mean

        zero_crossing = np.sum(
            centered[:-1] * centered[1:] < 0
        )

        rolling_mean_acc = float(np.mean(self._history_mean_acc[-2:] + [mean])) if self._history_mean_acc else float(mean)
        rolling_std_acc = float(np.std(self._history_mean_acc[-2:] + [mean])) if self._history_mean_acc else 0.0

        features = {

            "mean_acc": mean,

            "std_acc": std,
            "rolling_mean_acc": rolling_mean_acc,
            "rolling_std_acc": rolling_std_acc,

            "variance": variance,

            "energy": energy,

            "rms": rms,

            "movement_count": movement_count,
            "movement_ratio": movement_ratio,
            "acceleration_jerk": acceleration_jerk,

            "zero_crossing": zero_crossing,
        }

        self._history_mean_acc.append(float(mean))
        self._history_std_acc.append(float(std))
        if len(self._history_mean_acc) > 3:
            self._history_mean_acc = self._history_mean_acc[-3:]
        if len(self._history_std_acc) > 3:
            self._history_std_acc = self._history_std_acc[-3:]

        return features

    @staticmethod
    def feature_names():

        return [
            "mean_acc",
            "std_acc",
            "rolling_mean_acc",
            "rolling_std_acc",
            "variance",
            "energy",
            "rms",
            "movement_count",
            "movement_ratio",
            "acceleration_jerk",
            "zero_crossing",
        ]


    def to_vector(self, feature_dict):

        return np.array(
            [feature_dict[name]
             for name in self.feature_names()],
            dtype=np.float32,
        )
