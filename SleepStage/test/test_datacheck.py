from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Preprocessing.datacheck import DatasetChecker


def main():

    dataset_path = PROJECT_ROOT / "Dataset" / "raw"

    parser = SleepDatasetParser(dataset_path)

    dataset = parser.load_all(
        subject_filter="Bidslab00"
    )

    checker = DatasetChecker()

    result = checker.check_dataset(dataset)

    checker.print_summary(result)

    report_path = PROJECT_ROOT / "Dataset" / "dataset_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        report_path,
        index=False,
    )

    print()

    print("Saved report to Dataset/dataset_report.csv")


if __name__ == "__main__":
    main()