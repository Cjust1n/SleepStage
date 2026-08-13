"""scaler.py

Wrapper around Preprocessing/normalizer.py to provide a consistent interface
for training/evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from Preprocessing.normalizer import FeatureNormalizer


class Scaler:
    def __init__(self):
        self._norm = FeatureNormalizer()

    def fit(self, X: np.ndarray) -> "Scaler":
        self._norm.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._norm.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._norm.fit_transform(X)

    def save(self, path: str | Path) -> None:
        self._norm.save(path)

    @property
    def mean(self) -> Optional[np.ndarray]:
        return self._norm.mean

    @property
    def std(self) -> Optional[np.ndarray]:
        return self._norm.std

