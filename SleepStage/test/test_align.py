from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Preprocessing.align import EpochAligner


def main():

    dataset_path = PROJECT_ROOT / "Dataset" / "raw"

    parser = SleepDatasetParser(dataset_path)

    dataset = parser.load_all(subject_filter="Bidslab00")

    aligner = EpochAligner()

    night = dataset[0]

    motion, hr = aligner.align_night(night)

    print("=" * 60)
    print("Motion")
    print("=" * 60)
    print(motion.head())

    print()
    print("Epoch range :", motion["epoch"].min(), "->", motion["epoch"].max())
    print("Unique epochs :", motion["epoch"].nunique())

    print()

    print("=" * 60)
    print("Heart Rate")
    print("=" * 60)
    print(hr.head())

    print()
    print("Epoch range :", hr["epoch"].min(), "->", hr["epoch"].max())
    print("Unique epochs :", hr["epoch"].nunique())


if __name__ == "__main__":
    main()