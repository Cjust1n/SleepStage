"""
sanity_check.py

Dataset quality checker.

Author: Justin
"""

from pathlib import Path
import numpy as np
import pandas as pd


class DatasetChecker:

    def __init__(self):
        pass

    def estimate_sampling_rate(self, timestamps):

        timestamps = np.asarray(timestamps)

        dt = np.diff(timestamps)

        dt = dt[dt > 0]

        if len(dt) == 0:
            return np.nan

        return 1.0 / np.mean(dt)

    def estimate_duration(self, timestamps):

        timestamps = np.asarray(timestamps)

        if len(timestamps) == 0:
            return 0

        return timestamps[-1] - timestamps[0]

    def check_recording(self, night):

        motion = night.motion
        hr = night.hr

        motion_fs = self.estimate_sampling_rate(
            motion["Timestamp"]
        )

        hr_fs = self.estimate_sampling_rate(
            hr["Timestamp"]
        )

        motion_duration = self.estimate_duration(
            motion["Timestamp"]
        )

        hr_duration = self.estimate_duration(
            hr["Timestamp"]
        )

        label_epochs = len(night.expert_label)

        expected_duration = label_epochs * 30

        return {
            "Subject": night.subject,
            "Night": night.night,

            "Motion Samples": len(motion),
            "HR Samples": len(hr),

            "Motion Fs": motion_fs,
            "HR Fs": hr_fs,

            "Motion Duration (s)": motion_duration,
            "HR Duration (s)": hr_duration,

            "Motion Duration (h)": motion_duration / 3600,
            "HR Duration (h)": hr_duration / 3600,

            "Epoch Labels": label_epochs,

            "Expected Duration (s)": expected_duration,

            "Motion Missing (s)": expected_duration - motion_duration,

            "HR Missing (s)": expected_duration - hr_duration,
        }

    def check_dataset(self, dataset):

        rows = []

        for night in dataset:

            rows.append(
                self.check_recording(night)
            )

        df = pd.DataFrame(rows)

        return df

    def print_summary(self, df):

        pd.set_option("display.max_columns", None)

        print()

        print("=" * 100)

        print(df)

        print("=" * 100)

        print()

        print("Average Motion Fs :",
              round(df["Motion Fs"].mean(), 2))

        print("Average HR Fs :",
              round(df["HR Fs"].mean(), 3))

        print()

        print("Average Motion Duration :",
              round(df["Motion Duration (h)"].mean(), 2),
              "hours")

        print("Average HR Duration :",
              round(df["HR Duration (h)"].mean(), 2),
              "hours")