"""sequence_builder.py

Thin wrapper to build sequences using the existing Preprocessing/sequence_builder.py
implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from Preprocessing.sequence_builder import SequenceBuilder as _SequenceBuilder


class SequenceCreator:
    def __init__(
        self,
        sequence_length: int,
        step: int = 1,
        require_contiguous: bool = True,
    ):
        self._builder = _SequenceBuilder(
            sequence_length=sequence_length,
            step=step,
            require_contiguous=require_contiguous,
        )

    def build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        return self._builder.build_sequences(X=X, y=y, metadata=metadata)

