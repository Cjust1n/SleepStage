"""
sequence_builder.py

Build sequential samples from feature vectors and metadata.

Author: Justin
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class SequenceBuilder:

    def __init__(
        self,
        sequence_length: int = 30,
        step: int = 1,
        require_contiguous: bool = True,
        sort_columns: Sequence[str] = ("Subject", "Night", "Epoch"),
        group_columns: Sequence[str] = ("Subject", "Night"),
        epoch_column: str = "Epoch",
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if step < 1:
            raise ValueError("step must be >= 1")

        self.sequence_length = sequence_length
        self.step = step
        self.require_contiguous = require_contiguous
        self.sort_columns = list(sort_columns)
        self.group_columns = list(group_columns)
        self.epoch_column = epoch_column

    def build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Convert feature rows into overlapping sequences.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Labels of shape (n_samples,).
        metadata : pd.DataFrame
            Metadata for each row. Must contain sort/group columns.

        Returns
        -------
        np.ndarray
            X sequences with shape (n_sequences, sequence_length, n_features).
        np.ndarray
            Labels for each sequence.
        pd.DataFrame
            Metadata for each sequence.
        """
        if len(X) != len(y) or len(X) != len(metadata):
            raise ValueError("X, y, and metadata must have the same length")

        metadata = metadata.reset_index(drop=True).copy()

        for column in self.sort_columns:
            if column not in metadata.columns:
                raise ValueError(f"Missing required column '{column}' in metadata")

        sorted_idx = metadata.sort_values(self.sort_columns).index.to_numpy()
        X_sorted = np.asarray(X)[sorted_idx]
        y_sorted = np.asarray(y)[sorted_idx]
        metadata_sorted = metadata.loc[sorted_idx].reset_index(drop=True)

        sequences: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        sequence_meta: List[dict] = []

        group_keys = list(self.group_columns)
        grouped = metadata_sorted.groupby(group_keys, sort=False)

        for _, group in grouped:
            group_idx = group.index.to_numpy()
            epochs = group[self.epoch_column].to_numpy(dtype=np.int64)
            n_samples = len(group_idx)

            if n_samples < self.sequence_length:
                continue

            for start in range(0, n_samples - self.sequence_length + 1, self.step):
                window_epochs = epochs[start : start + self.sequence_length]
                if self.require_contiguous:
                    if not np.all(np.diff(window_epochs) == 1):
                        continue

                window_idx = group_idx[start : start + self.sequence_length]
                sequences.append(X_sorted[window_idx])
                labels.append(y_sorted[window_idx[-1]])

                sequence_meta.append({
                    "Subject": group["Subject"].iloc[start],
                    "Night": group["Night"].iloc[start],
                    "start_epoch": int(window_epochs[0]),
                    "end_epoch": int(window_epochs[-1]),
                })

        if not sequences:
            return (
                np.zeros((0, self.sequence_length, X.shape[1]), dtype=X.dtype),
                np.zeros((0,), dtype=y.dtype),
                pd.DataFrame(
                    columns=[*self.group_columns, "start_epoch", "end_epoch"],
                ),
            )

        X_seq = np.stack(sequences, axis=0)
        y_seq = np.asarray(labels, dtype=y_sorted.dtype)
        meta_seq = pd.DataFrame(sequence_meta)

        return X_seq, y_seq, meta_seq
