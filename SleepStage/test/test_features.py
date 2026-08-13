from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Preprocessing.align import EpochAligner
from Preprocessing.epoch_segmenter import EpochSegmenter

from FeatureExtraction import (
    HRVFeatureExtractor,
    MovementFeatureExtractor,
    TemporalFeatureExtractor,
)

dataset_path = PROJECT_ROOT / "Dataset" / "raw"

parser = SleepDatasetParser(dataset_path)

dataset = parser.load_all(subject_filter="Bidslab00")

night = dataset[0]

aligner = EpochAligner()

motion, hr = aligner.align_night(night)

segmenter = EpochSegmenter()

motion_epochs, hr_epochs = segmenter.segment_night(
    motion,
    hr,
)

motion_extractor = MovementFeatureExtractor()

hrv_extractor = HRVFeatureExtractor()

temporal_extractor = TemporalFeatureExtractor()

m = motion_extractor.extract(
    motion_epochs[0]
)

h = hrv_extractor.extract(
    hr_epochs[0]
)

t = temporal_extractor.extract(
    epoch=0,
    total_epochs=len(night.expert_label),
    rec_start=night.rec_start,
)

vector = np.concatenate([

    hrv_extractor.to_vector(h),

    motion_extractor.to_vector(m),

    temporal_extractor.to_vector(t),

])

print("="*60)

print("Feature Vector")

print("="*60)

print(vector)

print()

print("Shape :", vector.shape)