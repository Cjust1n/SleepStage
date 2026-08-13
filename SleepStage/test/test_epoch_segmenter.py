from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Preprocessing.align import EpochAligner
from Preprocessing.epoch_segmenter import EpochSegmenter


def main():

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

    print("="*60)
    print("Motion Epoch 0")
    print("="*60)

    print(motion_epochs[0].head())

    print()

    print("Samples :", len(motion_epochs[0]))

    print()

    print("="*60)
    print("HR Epoch 0")
    print("="*60)

    print(hr_epochs[0].head())

    print()

    print("Samples :", len(hr_epochs[0]))

    print()

    print("Total Motion Epoch :", len(motion_epochs))

    print("Total HR Epoch :", len(hr_epochs))


if __name__ == "__main__":
    main()