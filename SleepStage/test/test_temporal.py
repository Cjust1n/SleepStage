from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser

from FeatureExtraction.temporal import TemporalFeatureExtractor


def main():

    dataset_path = PROJECT_ROOT / "Dataset" / "raw"

    parser = SleepDatasetParser(dataset_path)

    dataset = parser.load_all(subject_filter="Bidslab00")

    night = dataset[0]

    extractor = TemporalFeatureExtractor()

    feature = extractor.extract(
        epoch=0,
        total_epochs=len(night.expert_label),
        rec_start=night.rec_start,
    )

    print("="*60)

    print("Temporal Features")

    print("="*60)

    for k, v in feature.items():

        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()