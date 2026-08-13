"""
cleaning.py

Epoch Quality Control.

Author : Justin
"""

import numpy as np
import pandas as pd

from config import *


class EpochCleaner:

    def __init__(self):

        self.reset()

    def reset(self):

        self.total = 0

        self.valid = 0

        self.drop_motion = 0

        self.drop_hr = 0

        self.drop_label = 0

        self.interpolated = 0

    def clean(
        self,
        motion_epochs,
        hr_epochs,
        labels,
    ):

        clean_motion = {}
        clean_hr = {}
        clean_labels = {}

        self.reset()

        epochs = sorted(labels.keys())

        for epoch in epochs:

            self.total += 1

            motion = motion_epochs.get(epoch)

            hr = hr_epochs.get(epoch)

            label = labels[epoch]

            # --------------------------
            # Unknown label
            # --------------------------

            if DROP_UNKNOWN_LABEL:

                if label == UNKNOWN_LABEL:

                    self.drop_label += 1

                    continue

            # --------------------------
            # Motion
            # --------------------------

            if motion is None:

                self.drop_motion += 1

                continue

            if len(motion) < MIN_MOTION_SAMPLES:

                self.drop_motion += 1

                continue

            # --------------------------
            # HR
            # --------------------------

            if hr is None:

                self.drop_hr += 1

                continue

            if len(hr) < MIN_HR_SAMPLES:

                self.drop_hr += 1

                continue

            # --------------------------
            # HR interpolation (optional)
            # --------------------------

            if INTERPOLATE_HR:

                if len(hr) < EXPECTED_HR_SAMPLES:

                    hr = self.interpolate_hr(hr)

                    self.interpolated += 1

            clean_motion[epoch] = motion

            clean_hr[epoch] = hr

            clean_labels[epoch] = label

            self.valid += 1

        return clean_motion, clean_hr, clean_labels

    def interpolate_hr(self, hr):

        hr = hr.copy()

        full_time = np.linspace(
            hr.Timestamp.min(),
            hr.Timestamp.max(),
            EXPECTED_HR_SAMPLES,
        )

        new_hr = np.interp(
            full_time,
            hr.Timestamp,
            hr.HR,
        )

        return pd.DataFrame({

            "Timestamp": full_time,

            "HR": new_hr,

            "epoch": hr.epoch.iloc[0],

        })

    def summary(self):

        print()

        print("="*60)

        print("Cleaning Report")

        print("="*60)

        print(f"Total Epoch          : {self.total}")

        print(f"Valid Epoch          : {self.valid}")

        print()

        print(f"Dropped Motion       : {self.drop_motion}")

        print(f"Dropped HR           : {self.drop_hr}")

        print(f"Dropped Label        : {self.drop_label}")

        print()

        print(f"Interpolated HR      : {self.interpolated}")

        print()

        print(
            "Remaining Epoch (%) :",
            round(
                100*self.valid/self.total,
                2
            )
        )