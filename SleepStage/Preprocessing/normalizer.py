"""
normalizer.py

Feature normalization using Z-score.

Normalized value:

    x_norm = (x - mean) / std

Author : Justin
"""

import json
from pathlib import Path

import numpy as np


class FeatureNormalizer:

    def __init__(self):

        self.mean = None
        self.std = None

    # -------------------------------------------------------
    # Fit
    # -------------------------------------------------------

    def fit(self, X):

        X = np.asarray(X, dtype=np.float32)

        self.mean = np.mean(X, axis=0)

        self.std = np.std(X, axis=0)

        # Hindari divide-by-zero
        self.std[self.std < 1e-8] = 1.0

        return self

    # -------------------------------------------------------
    # Transform
    # -------------------------------------------------------

    def transform(self, X):

        if self.mean is None:
            raise RuntimeError("Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        return (X - self.mean) / self.std

    # -------------------------------------------------------
    # Fit + Transform
    # -------------------------------------------------------

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)

    # -------------------------------------------------------
    # Inverse Transform
    # -------------------------------------------------------

    def inverse_transform(self, X):

        if self.mean is None:
            raise RuntimeError("Call fit() first.")

        X = np.asarray(X, dtype=np.float32)

        return X * self.std + self.mean

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    def save(self, filename):

        filename = Path(filename)

        data = {

            "mean": self.mean.tolist(),

            "std": self.std.tolist(),

        }

        with open(filename, "w") as f:

            json.dump(data, f, indent=4)

    # -------------------------------------------------------
    # Load
    # -------------------------------------------------------

    def load(self, filename):

        filename = Path(filename)

        with open(filename, "r") as f:

            data = json.load(f)

        self.mean = np.asarray(
            data["mean"],
            dtype=np.float32,
        )

        self.std = np.asarray(
            data["std"],
            dtype=np.float32,
        )

        return self

    # -------------------------------------------------------

    def summary(self):

        if self.mean is None:

            print("Normalizer belum di-fit.")

            return

        print("=" * 60)

        print("Feature Normalizer")

        print("=" * 60)

        print("Jumlah Feature :", len(self.mean))

        print()

        for i in range(len(self.mean)):

            print(
                f"{i:02d}  Mean = {self.mean[i]:12.4f}"
                f"   Std = {self.std[i]:12.4f}"
            )