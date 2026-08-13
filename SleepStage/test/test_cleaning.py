from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Preprocessing.align import EpochAligner
from Preprocessing.epoch_segmenter import EpochSegmenter
from Preprocessing.cleaning import EpochCleaner

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

labels = {
    i: int(label)
    for i, label in enumerate(night.expert_label)
}

cleaner = EpochCleaner()

motion_clean, hr_clean, labels_clean = cleaner.clean(
    motion_epochs,
    hr_epochs,
    labels,
)

cleaner.summary()