from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from Dataset import DatasetBuilder


def test_build_dataset_normalizes_single_feature_vectors(monkeypatch):
    builder = DatasetBuilder()

    nights = [
        SimpleNamespace(subject="S1", night=1),
        SimpleNamespace(subject="S1", night=2),
    ]

    def fake_build_night(night):
        if night.night == 1:
            return (
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([0], dtype=np.int32),
                [{"Subject": "S1", "Night": 1, "Epoch": 0}],
            )

        return (
            np.array([[3.0, 4.0]], dtype=np.float32),
            np.array([1], dtype=np.int32),
            [{"Subject": "S1", "Night": 2, "Epoch": 0}],
        )

    monkeypatch.setattr(builder, "build_night", fake_build_night)

    X, y, meta = builder.build_dataset(nights)

    assert X.shape == (2, 2)
    assert y.shape == (2,)
    assert list(meta["Night"]) == [1, 2]
