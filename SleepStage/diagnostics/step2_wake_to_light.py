from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.datasets.dataset import load_processed_dataset
from src.preprocessing.sequence_builder import SequenceCreator
from src.preprocessing.split_subject import SubjectWiseSplitter


CLASS_ID_TO_STAGE = {
    0: "Wake",
    1: "N1 & N2",
    2: "N3",
    3: "REM",
}


def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="processed")
    ap.add_argument("--saved_model_keras", type=str, default="outputs/saved_model.keras")
    ap.add_argument("--sequence_length", type=int, default=30)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--require_contiguous", type=int, default=1)

    ap.add_argument("--test_subjects", type=str, required=True, help="comma-separated")
    ap.add_argument("--val_subjects", type=str, default=None, help="comma-separated optional")
    ap.add_argument("--val_size_subjects", type=int, default=1)

    ap.add_argument("--sample_per_case", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def parse_subject_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    ds = load_processed_dataset(args.processed_dir)

    seq_creator = SequenceCreator(
        sequence_length=args.sequence_length,
        step=args.step,
        require_contiguous=bool(args.require_contiguous),
    )

    X_seq, y_seq, metadata_seq = seq_creator.build_sequences(ds.X, ds.y, ds.metadata)
    if X_seq.shape[0] == 0:
        raise RuntimeError("No sequences created; check sequence_length/contiguity/step")

    test_subjects = parse_subject_list(args.test_subjects)
    if not test_subjects:
        raise ValueError("--test_subjects cannot be empty")

    val_subjects = parse_subject_list(args.val_subjects)

    splitter = SubjectWiseSplitter(
        train_subjects=[],
        val_subjects=[],
        test_subjects=[],
    )

    # We need deterministic partitions exactly like train.py does.
    subjects = sorted(metadata_seq["Subject"].astype(str).unique().tolist())

    if val_subjects is not None:
        train_subjects = [s for s in subjects if s not in set(val_subjects + test_subjects)]
        val_subjects_resolved = val_subjects
        test_subjects_resolved = test_subjects
    else:
        remaining = [s for s in subjects if s not in set(test_subjects)]
        val_subjects_resolved = remaining[: args.val_size_subjects]
        train_subjects = [s for s in remaining if s not in set(val_subjects_resolved)]
        test_subjects_resolved = test_subjects

    splitter = SubjectWiseSplitter(
        train_subjects=train_subjects,
        val_subjects=val_subjects_resolved,
        test_subjects=test_subjects_resolved,
    )

    (_, _y_train_dummy) = (None, None)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splitter.split(
        X_seq, y_seq, metadata_seq
    )

    print("Split shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("  X_test :", X_test.shape, "y_test :", y_test.shape)

    # Load trained keras model
    model_path = Path(args.saved_model_keras)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    # Some saved models contain custom layers (e.g. _TCNBlock). Import them so Keras
    # can deserialize correctly.
    custom_objects = {}
    try:
        from src.models.rnn import _TCNBlock  # noqa: F401
        custom_objects['_TCNBlock'] = _TCNBlock
    except Exception:
        # If the model isn't using that custom layer, it's safe to ignore.
        pass

    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        custom_objects=custom_objects if custom_objects else None,
    )


    # Get logits/probabilities
    logits = model.predict(X_test, verbose=0)
    logits = np.asarray(logits)

    probs = softmax_np(logits, axis=1)
    y_pred = np.argmax(probs, axis=1)

    wake_id = 0
    light_id = 1  # N1 & N2

    wake_mask = y_test == wake_id
    pred_light_mask = y_pred == light_id
    case_mask = wake_mask & pred_light_mask

    case_indices = np.where(case_mask)[0]
    print(f"Wake(true) -> Light(pred) cases found: {len(case_indices)}")

    if len(case_indices) == 0:
        print("No cases found. Nothing to plot; stopping.")
        return

    # Pick subset deterministically
    rng = np.random.default_rng(args.seed)
    if len(case_indices) > args.sample_per_case:
        picked = rng.choice(case_indices, size=args.sample_per_case, replace=False)
    else:
        picked = case_indices

    picked = np.sort(picked)

    for k, idx in enumerate(picked, start=1):
        true_id = int(y_test[idx])
        pred_id = int(y_pred[idx])
        p = probs[idx]

        # Metadata alignment: current splitter.split returns X_test derived from metadata_seq,
        # but SubjectWiseSplitter wrapper doesn't return meta. So we just print logits/probs.
        # If you want epoch range, we’ll need splitter to return metadata for test indices.

        print("=" * 80)
        print(f"Case {k}: test_index={idx}")
        print(f"  True : {CLASS_ID_TO_STAGE[true_id]} (id={true_id})")
        print(f"  Pred : {CLASS_ID_TO_STAGE[pred_id]} (id={pred_id})")
        print("  Softmax:")
        for cid in range(len(CLASS_ID_TO_STAGE)):
            print(f"    {CLASS_ID_TO_STAGE[cid]:8s} (id={cid}): {float(p[cid]):.4f}")

        # Quick feature-sequence sanity: show per-timestep mean of features
        seq = X_test[idx]
        print("  Sequence summary:")
        print("    shape:", seq.shape, "mean_timestep(first/last):", float(seq.mean(axis=1)[0]), float(seq.mean(axis=1)[-1]))

    print("=" * 80)


if __name__ == "__main__":
    main()

