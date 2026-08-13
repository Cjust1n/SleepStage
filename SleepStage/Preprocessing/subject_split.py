"""
subject_split.py

Split sequence data by subject for train/validation/test partitions.

Author: Justin
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


class SubjectSplitter:

    def __init__(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
    ):
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: pd.DataFrame,
        return_metadata: bool = False,
    ):
        if len(X) != len(y) or len(X) != len(metadata):
            raise ValueError("X, y, and metadata must have the same length")

        if "Subject" not in metadata.columns:
            raise ValueError("metadata must contain a Subject column")

        metadata = metadata.reset_index(drop=True).copy()

        def select(subjects):
            return metadata[metadata["Subject"].isin(subjects)].index.to_numpy()

        train_idx = select(self.train_subjects)
        val_idx = select(self.val_subjects)
        test_idx = select(self.test_subjects)

        if return_metadata:
            return (
                (
                    X[train_idx],
                    y[train_idx],
                    metadata.iloc[train_idx].reset_index(drop=True),
                ),
                (
                    X[val_idx],
                    y[val_idx],
                    metadata.iloc[val_idx].reset_index(drop=True),
                ),
                (
                    X[test_idx],
                    y[test_idx],
                    metadata.iloc[test_idx].reset_index(drop=True),
                ),
            )

        return (
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            (X[test_idx], y[test_idx]),
        )