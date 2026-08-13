"""Build the processed feature dataset for all available Bidslab subjects.
run: python build_all_processed_dataset.py --output-subdir processed_15_test --intermediate-dir batches_test
test: python build_all_processed_dataset.py --max-nights 1 --batch-size 1 --output-subdir processed_15_test --intermediate-dir batches_test
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Dataset.dataset_stream_builder import StreamingDatasetBuilder


CURRENT_SELECTED_FEATURES = [
    "mean_ibi",
    "mean_hr",
    "sdnn",
    "rmssd",
    "sdsd",
    "nn50",
    "pnn50",
    "lf",
    "hf",
    "lf_hf",
    "sd1",
    "sd2",
    "mean_acc",
    "std_acc",
    "variance",
    "energy",
    "rms",
    "movement_count",
    "movement_ratio",
    "acceleration_jerk",
    "zero_crossing",
    "time_of_night",
    "sin_time_of_night",
    "cos_time_of_night",
    "relative_position",
    "hr_delta",
    "rolling_mean_hr",
    "rolling_std_hr",
    "hr_slope",
    "rolling_hr_range",
    "rolling_mean_acc",
    "rolling_std_acc",
]


def main():
    dataset_path = Path(
        "/media/christopher-justin/UBUNTU 20_0/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    )

    import argparse

    parser_args = argparse.ArgumentParser()
    parser_args.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Path to the raw dataset root containing Bidslab subjects.",
    )
    parser_args.add_argument(
        "--feature-selection-file",
        type=str,
        default=None,
        help=(
            "Optional path to selected_features.txt. If omitted, the script "
            "writes and uses outputs/selected_features_current.txt."
        ),
    )
    parser_args.add_argument(
        "--output-subdir",
        type=str,
        default="processed_15",
        help="Output subfolder name under repo root (e.g. processed_15).",
    )
    parser_args.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of nights buffered before flushing to disk (reduces RAM usage).",
    )
    parser_args.add_argument(
        "--intermediate-dir",
        type=str,
        default="batches",
        help="Subfolder name inside processed/ for per-night intermediate chunks.",
    )
    parser_args.add_argument(
        "--max-nights",
        type=int,
        default=None,
        help=(
            "Limit how many nights are processed. Default: process all nights. "
            "(debug / reduce RAM/compute)"
        ),
    )
    args = parser_args.parse_args()

    if args.dataset_root is not None:
        dataset_path = Path(args.dataset_root)
    else:
        dataset_path = Path(
            "/media/christopher-justin/UBUNTU 20_0/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
        )
        if not dataset_path.exists():
            candidate = Path.home() / "Downloads" / "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
            if candidate.exists():
                dataset_path = candidate

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_path}. Pass --dataset-root explicitly."
        )

    parser = SleepDatasetParser(dataset_path)

    output_dir = PROJECT_ROOT / args.output_subdir

    def iter_nights(max_nights: int):

        count = 0
        for subject in parser.get_subjects():
            print(f"\n========== {subject.name} ==========")
            for night in parser.get_nights(subject):
                if count >= max_nights:
                    return
                print(f"Loading {subject.name} Night {night.name}")
                try:
                    yield parser.load_night(night)
                    count += 1
                except Exception:
                    continue

    builder = StreamingDatasetBuilder()

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    selected_features_path = outputs_dir / "selected_features_current.txt"
    if args.feature_selection_file is not None:
        selected_features_path = Path(args.feature_selection_file)
        selected_features_path.parent.mkdir(parents=True, exist_ok=True)
        if not selected_features_path.exists():
            selected_features_path.write_text("\n".join(CURRENT_SELECTED_FEATURES) + "\n")
    else:
        selected_features_path.write_text("\n".join(CURRENT_SELECTED_FEATURES) + "\n")

    max_nights = args.max_nights
    if max_nights is None:
        # Process all nights
        max_nights = 10**18

    info = builder.build_and_save_batched(
        iter_nights(int(max_nights)),
        output_dir,
        batch_size=args.batch_size,
        intermediate_dir_name=args.intermediate_dir,
        selected_features=CURRENT_SELECTED_FEATURES,
    )


    print("\nProcessed dataset saved to:", output_dir)
    print("X shape:", info["X_shape"])
    print("y shape:", info["y_shape"])
    print("Rows:", info["rows"])
    print("Chunks:", info["chunks"])
    print("Feature selection file:", selected_features_path)


if __name__ == "__main__":
    main()
