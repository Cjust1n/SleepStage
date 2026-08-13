"""Export a golden_header.h for INT8 inference.

This version targets row-wise processed features from ``X.npy`` / ``y.npy``.
It filters the dataset by subject, fits the scaler on train rows, quantizes
the selected test samples with the TFLite interpreter's INT8 params, and
writes a C header that is easy to use in an FVP Corstone 300 simulation.

Default behavior:
- ``test_subjects = ["Bidslab01"]``
- take the first ``sample_count = 10`` rows from that subject
- emit ``golden_input[10][22]`` and ``golden_output[10]``
- ``golden_output`` is the per-sample argmax class prediction

Usage example:
    python -m src.evaluation.export_golden_header \
        --processed_dir processed \
        --tflite_model outputs/checkpoints/gru_int8.tflite \
        --test_subjects Bidslab01 \
        --sample_count 10 \
        --output_header golden_header.h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import tensorflow as tf

from src.datasets.dataset import load_processed_dataset
from src.preprocessing.sequence_builder import SequenceCreator
from src.preprocessing.split_subject import SubjectWiseSplitter
from src.preprocessing.scaler import Scaler


def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def _parse_subject_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def scale_like_train_2d(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Scaler]:
    """Fit scaler on train only, then transform val/test row-wise."""
    if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            "scale_like_train_2d expects 2D arrays shaped [samples, features]."
        )

    scaler = Scaler()

    scaler.fit(X_train.astype(np.float32))

    X_train_scaled = scaler.transform(X_train.astype(np.float32))
    X_val_scaled = scaler.transform(X_val.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.astype(np.float32))

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def scale_like_train_3d(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Scaler]:
    """Fit scaler on train only, then transform sequence tensors."""
    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "scale_like_train_3d expects 3D arrays shaped [samples, time, features]."
        )

    n_train, t, f = X_train.shape
    scaler = Scaler()

    X_train_2d = X_train.reshape(-1, f)
    scaler.fit(X_train_2d)

    X_val_2d = X_val.reshape(-1, f)
    X_test_2d = X_test.reshape(-1, f)

    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, t, f)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape[0], t, f)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape[0], t, f)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def _parse_subject_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _split_rows_by_subject(
    X: np.ndarray,
    y: np.ndarray,
    metadata,
    test_subjects: list[str],
    val_subjects: list[str] | None,
    val_size_subjects: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    subjects = sorted(metadata["Subject"].astype(str).unique().tolist())

    if val_subjects is not None:
        train_subjects = [s for s in subjects if s not in set(val_subjects + test_subjects)]
    else:
        remaining = [s for s in subjects if s not in set(test_subjects)]
        val_subjects = remaining[:val_size_subjects]
        train_subjects = [s for s in remaining if s not in set(val_subjects)]

    splitter = SubjectWiseSplitter(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    return splitter.split(X, y, metadata)


def _split_sequences_by_subject(
    X: np.ndarray,
    y: np.ndarray,
    metadata,
    sequence_length: int,
    step: int,
    require_contiguous: bool,
    test_subjects: list[str],
    val_subjects: list[str] | None,
    val_size_subjects: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    seq_creator = SequenceCreator(
        sequence_length=sequence_length,
        step=step,
        require_contiguous=require_contiguous,
    )
    X_seq, y_seq, metadata_seq = seq_creator.build_sequences(X, y, metadata)
    if len(X_seq) == 0:
        raise RuntimeError("No sequences created; check sequence_length/contiguity/step")

    subjects = sorted(metadata_seq["Subject"].astype(str).unique().tolist())

    if val_subjects is not None:
        train_subjects = [s for s in subjects if s not in set(val_subjects + test_subjects)]
    else:
        remaining = [s for s in subjects if s not in set(test_subjects)]
        val_subjects = remaining[:val_size_subjects]
        train_subjects = [s for s in remaining if s not in set(val_subjects)]

    splitter = SubjectWiseSplitter(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    return splitter.split(X_seq, y_seq, metadata_seq)


def quant_params_from_interpreter(interp: tf.lite.Interpreter) -> Tuple[float, int, float, int]:
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    in_scale, in_zero = in_detail.get("quantization")
    out_scale, out_zero = out_detail.get("quantization")

    if in_scale is None or out_scale is None:
        raise RuntimeError(
            "Interpreter quantization params are missing. Ensure model is fully quantized INT8."
        )

    return float(in_scale), int(in_zero), float(out_scale), int(out_zero)


def run_int8_batch(
    interp: tf.lite.Interpreter,
    x_int8: np.ndarray,
) -> np.ndarray:
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    x_int8 = np.asarray(x_int8, dtype=np.int8)

    if x_int8.ndim < 2:
        raise ValueError(
            f"Expected x_int8 to have at least 2 dimensions, got {x_int8.shape}."
        )

    input_shape = tuple(int(dim) for dim in in_detail["shape"])
    if len(input_shape) != x_int8.ndim:
        raise ValueError(
            "This generator expects the TFLite input rank to match the sample rank. "
            f"Got {input_shape}."
        )

    if input_shape[1:] != tuple(int(dim) for dim in x_int8.shape[1:]):
        raise ValueError(
            f"Sample dimension mismatch: model expects {input_shape[1:]}, got {x_int8.shape[1:]}."
        )

    if input_shape != tuple(x_int8.shape):
        interp.resize_tensor_input(in_detail["index"], x_int8.shape)
        interp.allocate_tensors()
        in_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]

    interp.set_tensor(in_detail["index"], x_int8)
    interp.invoke()
    out_q = interp.get_tensor(out_detail["index"])  # likely int8[1,4] or int8[4]

    return np.asarray(out_q)


def _format_1d_int8(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    body = ", ".join(str(int(x)) for x in flat.tolist())
    return f"const int8_t {name}[{flat.size}] = {{\n    {body}\n}};\n"


def _format_2d_int8(name: str, values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {name}, got shape {arr.shape}")

    rows: list[str] = [f"const int8_t {name}[{arr.shape[0]}][{arr.shape[1]}] = {{"]
    for row_index, row in enumerate(arr):
        suffix = "," if row_index < arr.shape[0] - 1 else ""
        rows.append("    { " + ", ".join(str(int(x)) for x in row.tolist()) + f" }}{suffix}")
    rows.append("};")
    return "\n".join(rows) + "\n"


def _format_3d_int8(name: str, values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int8)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array for {name}, got shape {arr.shape}")

    rows: list[str] = [f"const int8_t {name}[{arr.shape[0]}][{arr.shape[1]}][{arr.shape[2]}] = {{"]
    for sample_index, sample in enumerate(arr):
        sample_suffix = "," if sample_index < arr.shape[0] - 1 else ""
        rows.append("    {")
        for row_index, row in enumerate(sample):
            row_suffix = "," if row_index < sample.shape[0] - 1 else ""
            rows.append("        { " + ", ".join(str(int(x)) for x in row.tolist()) + f" }}{row_suffix}")
        rows.append(f"    }}{sample_suffix}")
    rows.append("};")
    return "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--processed_dir", type=str, default="processed")
    ap.add_argument("--tflite_model", type=str, required=True)
    ap.add_argument("--sample_count", type=int, default=10)
    ap.add_argument("--sequence_length", type=int, default=30)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--require_contiguous", type=int, default=1)

    ap.add_argument("--test_subjects", type=str, required=True, help="comma-separated")
    ap.add_argument("--val_subjects", type=str, default=None, help="comma-separated (optional)")
    ap.add_argument("--val_size_subjects", type=int, default=1)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--output_header", type=str, default="golden_header.h")

    args = ap.parse_args()

    _set_seed(args.seed)

    val_subjects = _parse_subject_list(args.val_subjects)
    test_subjects = _parse_subject_list(args.test_subjects) or []
    if not test_subjects:
        raise ValueError("--test_subjects cannot be empty")

    ds = load_processed_dataset(args.processed_dir)

    # Load INT8 interpreter
    tflite_path = Path(args.tflite_model)
    if not tflite_path.exists():
        raise FileNotFoundError(str(tflite_path))

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()

    input_rank = len(tuple(int(dim) for dim in interp.get_input_details()[0]["shape"]))

    in_scale, in_zero, out_scale, out_zero = quant_params_from_interpreter(interp)

    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")

    if input_rank == 2:
        (_X_train, _y_train), (_X_val, _y_val), (X_test, y_test) = _split_rows_by_subject(
            ds.X,
            ds.y,
            ds.metadata,
            test_subjects=test_subjects,
            val_subjects=val_subjects,
            val_size_subjects=args.val_size_subjects,
        )

        X_train_s, X_val_s, X_test_s, scaler = scale_like_train_2d(_X_train, _X_val, X_test)

        if X_test_s.shape[0] == 0:
            raise RuntimeError("No test rows selected for the requested subject split.")
        if args.sample_count > X_test_s.shape[0]:
            raise IndexError(
                f"sample_count={args.sample_count} is larger than available test rows "
                f"({X_test_s.shape[0]})."
            )

        x_f = X_test_s[: args.sample_count]
        y_true = np.asarray(y_test[: args.sample_count], dtype=np.int8)
        x_q = np.round(x_f / in_scale + in_zero).astype(np.int8)
        out_q = run_int8_batch(interp, x_q)

    elif input_rank == 3:
        require_contiguous = bool(args.require_contiguous)
        (_X_train, _y_train), (_X_val, _y_val), (X_test, y_test) = _split_sequences_by_subject(
            ds.X,
            ds.y,
            ds.metadata,
            sequence_length=args.sequence_length,
            step=args.step,
            require_contiguous=require_contiguous,
            test_subjects=test_subjects,
            val_subjects=val_subjects,
            val_size_subjects=args.val_size_subjects,
        )

        X_train_s, X_val_s, X_test_s, scaler = scale_like_train_3d(_X_train, _X_val, X_test)

        if X_test_s.shape[0] == 0:
            raise RuntimeError("No test sequences selected for the requested subject split.")
        if args.sample_count > X_test_s.shape[0]:
            raise IndexError(
                f"sample_count={args.sample_count} is larger than available test sequences "
                f"({X_test_s.shape[0]})."
            )

        x_f = X_test_s[: args.sample_count]
        y_true = np.asarray(y_test[: args.sample_count], dtype=np.int8)
        x_q = np.round(x_f / in_scale + in_zero).astype(np.int8)
        out_q = run_int8_batch(interp, x_q)

    else:
        raise ValueError(f"Unsupported TFLite input rank: {input_rank}")

    out_q = np.asarray(out_q)
    if out_q.ndim == 1:
        out_q = out_q[np.newaxis, ...]
    if out_q.ndim != 2:
        raise RuntimeError(f"Expected 2D output logits, got shape {out_q.shape}")
    if out_q.shape[0] != x_q.shape[0]:
        raise RuntimeError(
            f"Output batch size mismatch: got {out_q.shape}, expected first dim {x_q.shape[0]}"
        )

    golden_input = x_q.astype(np.int8)
    golden_logits = out_q.astype(np.int8)
    golden_output = np.argmax(golden_logits, axis=1).astype(np.int8)

    if input_rank == 2:
        expected_input_shape = (args.sample_count, x_f.shape[1])
    else:
        expected_input_shape = (args.sample_count, x_f.shape[1], x_f.shape[2])

    if golden_input.shape != expected_input_shape:
        raise RuntimeError(
            f"golden_input shape mismatch: got {golden_input.shape}, expected {expected_input_shape}"
        )
    if golden_output.shape != (args.sample_count,):
        raise RuntimeError(
            f"golden_output shape mismatch: got {golden_output.shape}, expected ({args.sample_count},)"
        )

    # Build header content
    header_path = Path(args.output_header)
    if not header_path.is_absolute():
        # treat relative paths as relative to SleepStage project root
        header_path = Path(__file__).resolve().parent.parent.parent / args.output_header

    header = (
        "#ifndef GOLDEN_DATA_H\n"
        "#define GOLDEN_DATA_H\n\n"
        "#include <stdint.h>\n\n"
        f"{_format_2d_int8('golden_input', golden_input) if input_rank == 2 else _format_3d_int8('golden_input', golden_input)}\n"
        f"{_format_1d_int8('golden_output', golden_output)}\n"
        f"{_format_2d_int8('golden_logits', golden_logits)}\n"
        f"// meta: input_shape={[int(x) for x in golden_input.shape]}, output_shape={[int(x) for x in golden_output.shape]}\n"
        f"// logits_shape={[int(x) for x in golden_logits.shape]}\n"
        f"// input_quant: scale={in_scale}, zero_point={in_zero}\n"
        f"// output_quant: scale={out_scale}, zero_point={out_zero}\n\n"
        "#endif\n"
    )

    header_path.write_text(header)

    # Also write a small JSON debug file next to header
    debug_path = header_path.with_suffix(".json")
    payload: Dict[str, Any] = {
        "tflite_model": str(tflite_path),
        "test_subjects": test_subjects,
        "val_subjects": val_subjects,
        "sample_count": args.sample_count,
        "input_rank": input_rank,
        "X_test_s_len": int(X_test_s.shape[0]),
        "input_shape": [int(x) for x in golden_input.shape],
        "output_shape": [int(x) for x in golden_output.shape],
        "logits_shape": [int(x) for x in golden_logits.shape],
        "golden_input": golden_input.tolist(),
        "golden_output": golden_output.tolist(),
        "golden_logits": golden_logits.tolist(),
        "y_true": y_true.tolist(),
        "predicted_class": golden_output.tolist(),
        "input_quant": {"scale": in_scale, "zero_point": in_zero},
        "output_quant": {"scale": out_scale, "zero_point": out_zero},
        "scaler_path_note": "scaler.fit happens inside export_golden_header.py",
    }
    debug_path.write_text(json.dumps(payload, indent=2))

    print("Exported:", str(header_path))
    print("Debug:", str(debug_path))


if __name__ == "__main__":
    main()

