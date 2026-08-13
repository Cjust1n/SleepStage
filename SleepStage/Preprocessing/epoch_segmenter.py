"""
epoch_segmenter.py

Split aligned signals into epoch-by-epoch segments.

Author: Justin
"""

import pandas as pd


class EpochSegmenter:

    def __init__(self):
        pass

    def segment(self, df):
        """
        Split dataframe by epoch.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        dict
            {
                epoch_number : dataframe
            }
        """

        epochs = {}

        grouped = df.groupby("epoch")

        for epoch, data in grouped:

            epochs[int(epoch)] = data.reset_index(drop=True)

        return epochs

    def segment_motion(self, motion):

        return self.segment(motion)

    def segment_hr(self, hr):

        return self.segment(hr)

    def segment_night(self, motion, hr):

        motion_epochs = self.segment_motion(motion)

        hr_epochs = self.segment_hr(hr)

        return motion_epochs, hr_epochs