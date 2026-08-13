from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Preprocessing.align import EpochAligner
from Preprocessing.epoch_segmenter import EpochSegmenter

from FeatureExtraction.hrv import HRVFeatureExtractor


def test_compute_frequency_domain_handles_insufficient_data():
    extractor = HRVFeatureExtractor()

    lf, hf, lf_hf = extractor.compute_frequency_domain(
        np.array([0.0, 1.0]),
        np.array([900.0, 1000.0]),
    )

    assert (lf, hf, lf_hf) == (0.0, 0.0, 0.0)


def test_compute_frequency_domain_returns_finite_log_features():
    extractor = HRVFeatureExtractor()

    lf, hf, lf_hf = extractor.compute_frequency_domain(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([900.0, 950.0, 920.0, 930.0, 910.0]),
    )

    assert np.isfinite(lf)
    assert np.isfinite(hf)
    assert np.isfinite(lf_hf)
    assert lf >= 0.0
    assert hf >= 0.0
    assert lf_hf >= 0.0


def main():

    dataset_path = PROJECT_ROOT / "Dataset" / "raw"

    parser = SleepDatasetParser(dataset_path)

    dataset = parser.load_all(subject_filter="Bidslab00")

    night = dataset[0]

    aligner = EpochAligner()

    motion, hr = aligner.align_night(night)

    segmenter = EpochSegmenter()

    _, hr_epochs = segmenter.segment_night(
        motion,
        hr,
    )

    extractor = HRVFeatureExtractor()

    feature = extractor.extract(
        hr_epochs[0]
    )

    print("="*60)

    print("HRV Features")

    print("="*60)

    for k, v in feature.items():

        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()