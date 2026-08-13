from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Preprocessing.parser import SleepDatasetParser
from Dataset.dataset_builder import DatasetBuilder
from Preprocessing.sequence_builder import SequenceBuilder


def main():
    dataset_path = PROJECT_ROOT / "Dataset" / "raw"

    parser = SleepDatasetParser(dataset_path)
    dataset = parser.load_all()

    builder = DatasetBuilder()
    X, y, metadata = builder.build_dataset(dataset)

    sequence_builder = SequenceBuilder(sequence_length=10)
    X_seq, y_seq, metadata_seq = sequence_builder.build_sequences(
        X,
        y,
        metadata,
    )

    print("X_seq.shape", X_seq.shape)
    print("y_seq.shape", y_seq.shape)
    print("metadata_seq.shape", metadata_seq.shape)
    if len(X_seq) > 0:
        print("X_seq[0]", X_seq[0].shape)
        print("y_seq[0]", y_seq[0])
        print("metadata_seq.iloc[0]", metadata_seq.iloc[0].to_dict())
        print("metadata_seq.iloc[-1]", metadata_seq.iloc[-1].to_dict())

    print()
    print("Sequence counts per Subject/Night:")
    print(metadata_seq.groupby(["Subject", "Night"]).size())

    unique, counts = np.unique(y_seq, return_counts=True)
    print()
    print("Label Distribution")
    for u, c in zip(unique, counts):
        print(u, c)


if __name__ == "__main__":
    main()
