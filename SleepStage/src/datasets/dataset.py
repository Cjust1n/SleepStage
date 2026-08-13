"""dataset.py

Dataset loader for the SleepStage processed arrays.

Expected files in `processed/`:
- X.npy or X_normalized.npy: float32, shape (n_samples, n_features)
- y.npy: int, shape (n_samples,)
- metadata.csv: must contain Subject, Night, Epoch

Note: sequences are created separately by the sequence builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ProcessedDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def _load_npy_array(path: Path) -> np.ndarray:
    """Load a .npy array robustly.

    This project historically produced some corrupted/serialized variants.
    We attempt the default load first, then retry allow_pickle=True.
    """

    try:
        return np.load(path)
    except Exception:
        return np.load(path, allow_pickle=True)


def load_processed_dataset(processed_dir: str | Path) -> ProcessedDataset:
    processed_dir = Path(processed_dir)

    x_candidates = [
        processed_dir / "X_normalized.npy",
        processed_dir / "X.npy",
    ]

    x = None
    last_x_err: Exception | None = None
    for cand in x_candidates:
        if not cand.exists():
            continue

        print(f"Trying X file: {cand.resolve()}")

        try:
            x = np.load(cand)
            print(f"Loaded {cand.name} shape={x.shape}")
            break
        except Exception as e:
            print(e)
            last_x_err = e
            try:
                x = np.load(cand, allow_pickle=True)
                print(f"Loaded {cand.name} (pickle) shape={x.shape}")
                break
            except Exception as e2:
                last_x_err = e2


    if x is None:
        tried = ", ".join(str(p) for p in x_candidates)
        raise RuntimeError(
            f"Failed to load X array. Tried: {tried}. Last error: {last_x_err}"
        )

    y_path = processed_dir / "y.npy"
    meta_path = processed_dir / "metadata.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing file: {meta_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    # Load y; if it is object/pickled, coerce to int.
    try:
        y = np.load(y_path)
    except Exception:
        y = np.load(y_path, allow_pickle=True)

    y = np.asarray(y)
    if y.dtype == object:
        y = y.astype(np.int64)

    if y.ndim != 1:
        raise ValueError(
            f"Expected y to be 1D array, got shape={y.shape}, dtype={y.dtype}"
        )

    metadata = pd.read_csv(meta_path)

    print("=" * 60)
    print("processed_dir =", processed_dir.resolve())
    print("X shape =", x.shape)
    print("y shape =", y.shape)
    print("metadata =", len(metadata))
    print("=" * 60)

    print("=" * 60)
    print("processed_dir =", processed_dir.resolve())
    print("X shape =", x.shape)
    print("y shape =", y.shape)
    print("metadata =", len(metadata))
    print("=" * 60)

    # Use explicit shapes to avoid any confusion about object/ndarray views.
    x_len = x.shape[0]
    y_len = y.shape[0]
    meta_len = len(metadata)

    print("Length check: x_len=", x_len, "y_len=", y_len, "meta_len=", meta_len)

    if x_len != y_len or x_len != meta_len:
        raise ValueError(
            f"Length mismatch: X={x_len}, y={y_len}, metadata={meta_len}"
        )




    # Ensure expected columns
    for col in ("Subject", "Night", "Epoch"):
        if col not in metadata.columns:
            raise ValueError(f"metadata.csv missing required column: {col}")

    return ProcessedDataset(X=x, y=y, metadata=metadata)


def infer_num_classes(y: np.ndarray) -> int:
    y = np.asarray(y)
    uniq = np.unique(y)
    return int(uniq.max() + 1)

