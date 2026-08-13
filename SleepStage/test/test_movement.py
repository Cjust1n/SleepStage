from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Preprocessing.align import EpochAligner
from Preprocessing.epoch_segmenter import EpochSegmenter

from FeatureExtraction.movement import MovementFeatureExtractor


def test_sample_entropy_is_removed_from_motion_features():
    extractor = MovementFeatureExtractor()
    motion_epoch = {
        "x": pd.Series([0.0, 0.2, -0.1, 0.3, -0.2, 0.4], dtype=np.float32),
        "y": pd.Series([0.0, 0.1, 0.0, 0.2, 0.0, 0.3], dtype=np.float32),
        "z": pd.Series([0.0, 0.0, 0.1, 0.0, 0.2, 0.0], dtype=np.float32),
    }

    features = extractor.extract(motion_epoch)

    assert "sample_entropy" not in extractor.feature_names()
    assert "sample_entropy" not in features


def main():

    dataset_path = PROJECT_ROOT / "Dataset" / "raw"

    parser = SleepDatasetParser(dataset_path)

    dataset = parser.load_all(subject_filter="Bidslab00")

    night = dataset[0]

    aligner = EpochAligner()

    motion, hr = aligner.align_night(night)

    segmenter = EpochSegmenter()

    motion_epochs, _ = segmenter.segment_night(
        motion,
        hr,
    )

    extractor = MovementFeatureExtractor()

    feature = extractor.extract(
        motion_epochs[0]
    )

    print("="*60)

    print("Movement Features")

    print("="*60)

    for k, v in feature.items():

        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()