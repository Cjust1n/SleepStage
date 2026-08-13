"""
align.py

Align physiological signals into 30-second epochs.

Author: Justin
"""

import numpy as np

from config import EPOCH_LENGTH


class EpochAligner:

    def __init__(self, epoch_length=EPOCH_LENGTH):
        self.epoch_length = epoch_length

    def compute_epoch(self, timestamps, rec_start):
        """
        Compute epoch index for every timestamp.

        Parameters
        ----------
        timestamps : array-like
            Unix timestamps.

        rec_start : float
            Recording start time (Unix timestamp).

        Returns
        -------
        numpy.ndarray
            Epoch indices.
        """

        timestamps = np.asarray(timestamps, dtype=np.float64)

        epoch = np.floor(
            (timestamps - rec_start) / self.epoch_length
        ).astype(np.int32)

        return epoch

    def align_motion(self, motion_df, rec_start):
        """
        Add epoch column to accelerometer dataframe.
        """

        motion = motion_df.copy()

        motion["epoch"] = self.compute_epoch(
            motion["Timestamp"].values,
            rec_start,
        )

        # Remove samples before recording start
        motion = motion[motion["epoch"] >= 0]

        motion.reset_index(drop=True, inplace=True)

        return motion

    def align_hr(self, hr_df, rec_start):
        """
        Add epoch column to HR dataframe.
        """

        hr = hr_df.copy()

        hr["epoch"] = self.compute_epoch(
            hr["Timestamp"].values,
            rec_start,
        )

        # Remove samples before recording start
        hr = hr[hr["epoch"] >= 0]

        hr.reset_index(drop=True, inplace=True)

        return hr

    def align_night(self, night):
        """
        Align one NightData object.

        Returns
        -------
        motion : pandas.DataFrame
        hr : pandas.DataFrame
        """

        motion = self.align_motion(
            night.motion,
            night.rec_start,
        )

        hr = self.align_hr(
            night.hr,
            night.rec_start,
        )

        return motion, hr