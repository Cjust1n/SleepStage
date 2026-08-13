"""
Test parser.py

Usage:
python tests/test_parser.py
"""

from pathlib import Path
import sys

# Tambahkan root project ke PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser


def main():

    dataset_path = PROJECT_ROOT / "Dataset" / "raw"

    parser = SleepDatasetParser(dataset_path)

    dataset = parser.load_all()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Total recordings : {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset kosong!")
        return

    sample = dataset[0]

    print()
    print("Subject :", sample.subject)
    print("Night   :", sample.night)

    print()
    print("Motion shape :", sample.motion.shape)
    print(sample.motion.head())

    print()
    print("HR shape :", sample.hr.shape)
    print(sample.hr.head())

    print()
    print("recStart")
    print(sample.rec_start)

    print()
    print("Expert label length :", len(sample.expert_label))
    print("First 20 labels :", sample.expert_label[:20])

    print()
    print("Dreem label length :", len(sample.dreem_label))
    print("First 20 labels :", sample.dreem_label[:20])

    print()
    print("=" * 60)
    print("Parser berhasil membaca dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()