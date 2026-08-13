"""split_subject.py

Wrapper around Preprocessing/subject_split.py.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from Preprocessing.subject_split import SubjectSplitter


class SubjectWiseSplitter:
    def __init__(
        self,
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
    ):
        self._splitter = SubjectSplitter(
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
        )

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: pd.DataFrame,
        return_metadata: bool = False,
    ):
        return self._splitter.split(
            X=X,
            y=y,
            metadata=metadata,
            return_metadata=return_metadata,
        )
