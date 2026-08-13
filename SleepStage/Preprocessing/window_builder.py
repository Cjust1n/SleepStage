"""
window_builder.py

Build HR windows from timestamp-based history.

Author: Justin
"""

import pandas as pd

from config import HRV_WINDOW_MINUTES


class WindowBuilder:

    def __init__(self, window_minutes=HRV_WINDOW_MINUTES):
        self.window_minutes = window_minutes

    def get_hr_window(self, hr_df, current_timestamp):
        """
        Build a rolling HR window from the current timestamp and previous duration.

        Parameters
        ----------
        hr_df : pandas.DataFrame
            Aligned HR samples.
        current_timestamp : float
            Unix timestamp representing the current time.

        Returns
        -------
        pandas.DataFrame or None
        """
        if hr_df is None or current_timestamp is None:
            return None

        start_time = current_timestamp - self.window_minutes * 60
        mask = (
            hr_df["Timestamp"] >= start_time
            ) & (
            hr_df["Timestamp"] <= current_timestamp
        )

        window = hr_df.loc[mask].copy()
        if window.empty:
            return None

        return window.reset_index(drop=True)
