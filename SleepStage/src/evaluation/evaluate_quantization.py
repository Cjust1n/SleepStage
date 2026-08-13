"""evaluate_quantization.py

Compare FLOAT32 Keras model vs INT8 TFLite model on the same test set.

Usage (example):
  python -m src.evaluation.evaluate_quantization \
    --keras_model outputs/saved_model.keras \
    --tflite_model outputs/checkpoints/simple_rnn_int8.tflite \
    --processed_dir processed_15_all \
    --sequence_length 30

Notes:
- This script reuses existing evaluation logic where possible.
- It quantizes inputs to int8 using the TFLite interpreter's input quantization params.
- It dequantizes output logits using the output quantization params before computing metrics.

"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from src.datasets.dataset import load_processed_dataset
from src.preprocessing.sequence_builder import SequenceCreator
from src.preprocessing.split_subject import SubjectWiseSplitter
from src.preprocessing.scaler import Scaler
from src.preprocessing.feature_subset import (
    FeatureSubsetConfig,
    infer_feature_selection_from_dim,
    load_feature_names,
    resolve_feature_selection,
    select_features_from_X,
)



def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def build_test_split(
    processed_dir: str | Path,
    sequence_length: int,
    step: int,
    require_contiguous: bool,
    feature_group_mode: str = "all",
    use_feature_subset: bool = False,
    feature_subset_features: tuple[str, ...] | None = None,
    val_subjects: list[str] | None = None,
    test_subjects: list[str] | None = None,
    val_size_subjects: int = 1,
    test_size_subjects: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_test_s, y_test, metadata_seq_for_debug)."""

    ds = load_processed_dataset(processed_dir)
    feature_names = load_feature_names(processed_dir)

    seq_creator = SequenceCreator(
        sequence_length=sequence_length,
        step=step,
        require_contiguous=require_contiguous,
    )
    X_seq, y_seq, metadata_seq = seq_creator.build_sequences(ds.X, ds.y, ds.metadata)

    feature_group_mode = str(feature_group_mode).strip().lower()
    if feature_group_mode != "all":
        selection = resolve_feature_selection(feature_names, feature_preset=feature_group_mode, explicit_features=None)
        X_seq = select_features_from_X(
            X_seq,
            feature_names=feature_names,
            selected_features=selection.selected_features,
        )
    elif use_feature_subset:
        selection = resolve_feature_selection(feature_names, feature_preset="all", explicit_features=feature_subset_features if feature_subset_features is not None else FeatureSubsetConfig().feature_subset)
        X_seq = select_features_from_X(
            X_seq,
            feature_names=feature_names,
            selected_features=selection.selected_features,
        )

    if len(X_seq) == 0:
        raise RuntimeError("No sequences created; check sequence_length/contiguity")

    # Resolve deterministic subject split compatible with train.py.
    subjects = sorted(metadata_seq["Subject"].astype(str).unique().tolist())

    if test_subjects is not None and val_subjects is not None:
        train_subjects = [s for s in subjects if s not in set(val_subjects + test_subjects)]
    else:
        val_subjects = subjects[:val_size_subjects]
        test_subjects = subjects[val_size_subjects : val_size_subjects + test_size_subjects]
        train_subjects = [s for s in subjects if s not in set(val_subjects + test_subjects)]

    splitter = SubjectWiseSplitter(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    (_X_train, _y_train), (_X_val, _y_val), (X_test, y_test) = splitter.split(
        X_seq, y_seq, metadata_seq
    )

    return X_test, y_test, metadata_seq


def _load_saved_feature_selection(model_path: str | Path) -> dict[str, Any] | None:
    """Load persisted feature-selection metadata near a saved model, if present."""
    model_path = Path(model_path)
    candidates = [
        model_path.parent / "feature_selection.json",
        model_path.parent / "run_summary.json",
    ]

    for cand in candidates:
        if not cand.exists():
            continue
        try:
            payload = json.loads(cand.read_text())
        except Exception:
            continue

        if "feature_selection" in payload and isinstance(payload["feature_selection"], dict):
            return payload["feature_selection"]
        if "selected_features" in payload:
            return payload

    return None


def _resolve_selection_for_model(
    *,
    feature_names: list[str],
    expected_feature_dim: int,
    model_path: str | Path,
    feature_group_mode: str,
    use_feature_subset: bool,
    feature_subset_features: tuple[str, ...] | None,
) -> tuple[str, tuple[str, ...]]:
    mode = str(feature_group_mode).strip().lower()

    if mode != "all":
        try:
            selection = resolve_feature_selection(
                feature_names,
                feature_preset=mode,
                explicit_features=None,
            )
            return selection.mode, selection.selected_features
        except KeyError:
            pass

    if use_feature_subset:
        explicit = (
            feature_subset_features
            if feature_subset_features is not None
            else FeatureSubsetConfig().feature_subset
        )
        if explicit is not None:
            try:
                selection = resolve_feature_selection(
                    feature_names,
                    feature_preset="all",
                    explicit_features=explicit,
                )
                return selection.mode, selection.selected_features
            except KeyError:
                pass

    saved = _load_saved_feature_selection(model_path)
    if saved is not None:
        selected = tuple(saved.get("selected_features", []))
        if selected:
            try:
                selection = resolve_feature_selection(
                    feature_names,
                    feature_preset="all",
                    explicit_features=selected,
                )
                if len(selection.selected_features) == int(expected_feature_dim):
                    return selection.mode, selection.selected_features
            except KeyError:
                pass

    inferred = infer_feature_selection_from_dim(feature_names, expected_feature_dim)
    if inferred is not None:
        return inferred.mode, inferred.selected_features

    if int(expected_feature_dim) == len(feature_names):
        selection = resolve_feature_selection(
            feature_names,
            feature_preset="all",
            explicit_features=None,
        )
        return selection.mode, selection.selected_features

    raise ValueError(
        "Could not resolve feature subset for model input dimension "
        f"{expected_feature_dim}. Provide --feature_group_mode, "
        "--use_feature_subset, or save feature_selection.json alongside the model."
    )


def scale_like_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Scaler]:
    """Fit scaler on train only, then transform val/test."""
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


def run_int8_inference(
    interp: tf.lite.Interpreter,
    X_test_s: np.ndarray,
    input_scale: float,
    input_zero_point: int,
) -> np.ndarray:
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    # TFLite expects int8 input; quantize using scale/zero_point.
    # X_test_s is float32 scaled inputs.
    Xq = np.round(X_test_s / input_scale + input_zero_point).astype(np.int8)

    # Ensure shape matches model input.
    expected_shape = tuple(in_detail["shape"])
    # Many models use [1, T, F] even if batch is dynamic.
    # We'll set batch dimension dynamically by resizing if needed.

    preds = []
    for i in range(Xq.shape[0]):
        sample = Xq[i : i + 1]

        if tuple(in_detail["shape"]) != tuple(sample.shape):
            # Resize input tensor to match batch size.
            interp.resize_tensor_input(in_detail["index"], sample.shape)
            interp.allocate_tensors()
            in_detail = interp.get_input_details()[0]
            out_detail = interp.get_output_details()[0]

        interp.set_tensor(in_detail["index"], sample)
        interp.invoke()
        out = interp.get_tensor(out_detail["index"])  # quantized output logits
        preds.append(out[0])

    return np.array(preds)


def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "cohen_kappa": float(kappa),
        "mcc": float(mcc),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras_model", type=str, required=True)
    parser.add_argument("--tflite_model", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, default="processed_15_all")

    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--require_contiguous", type=int, default=1)
    parser.add_argument("--feature_group_mode", type=str, default="all")
    parser.add_argument("--use_feature_subset", action="store_true")
    parser.add_argument(
        "--feature_subset_features",
        type=str,
        default=None,
        help="comma-separated list of features to select when use_feature_subset is enabled",
    )

    # If you want exact match with train.py, pass explicit subject lists.
    parser.add_argument("--val_subjects", type=str, default=None, help="comma-separated")
    parser.add_argument("--test_subjects", type=str, default=None, help="comma-separated")
    parser.add_argument("--val_size_subjects", type=int, default=1)
    parser.add_argument("--test_size_subjects", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        type=str,
        default="quantization_comparison.json",
        help="Where to save comparison metrics (if successful)",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    require_contiguous = bool(args.require_contiguous)
    feature_group_mode = str(args.feature_group_mode).strip().lower()
    feature_subset_features = None
    if args.feature_subset_features:
        feature_subset_features = tuple(
            s.strip() for s in args.feature_subset_features.split(",") if s.strip()
        )

    val_subjects = None
    test_subjects = None
    if args.val_subjects:
        val_subjects = [s.strip() for s in args.val_subjects.split(",") if s.strip()]
    if args.test_subjects:
        test_subjects = [s.strip() for s in args.test_subjects.split(",") if s.strip()]

    # Load FLOAT32 model early so we can resolve the expected feature dimension
    # before selecting features and scaling the test split.
    from src.models.rnn import build_gru_model  # noqa: F401

    keras_model = tf.keras.models.load_model(args.keras_model, compile=False)
    input_shape = keras_model.input_shape
    expected_feature_dim = int(input_shape[-1]) if input_shape and input_shape[-1] is not None else None
    if expected_feature_dim is None:
        raise RuntimeError("Could not determine model input feature dimension from keras_model.input_shape")

    # Build and scale the SAME test set as training pipeline.
    ds = load_processed_dataset(args.processed_dir)
    feature_names = load_feature_names(args.processed_dir)
    seq_creator = SequenceCreator(
        sequence_length=args.sequence_length,
        step=args.step,
        require_contiguous=require_contiguous,
    )
    X_seq, y_seq, metadata_seq = seq_creator.build_sequences(ds.X, ds.y, ds.metadata)

    selection_mode, selected_features = _resolve_selection_for_model(
        feature_names=feature_names,
        expected_feature_dim=expected_feature_dim,
        model_path=args.keras_model,
        feature_group_mode=feature_group_mode,
        use_feature_subset=args.use_feature_subset,
        feature_subset_features=feature_subset_features,
    )
    if len(selected_features) != expected_feature_dim:
        raise ValueError(
            f"Resolved {len(selected_features)} selected features for mode '{selection_mode}', "
            f"but model expects {expected_feature_dim} features."
        )
    if len(selected_features) != len(feature_names):
        X_seq = select_features_from_X(
            X_seq,
            feature_names=feature_names,
            selected_features=selected_features,
        )


    subjects = sorted(metadata_seq["Subject"].astype(str).unique().tolist())
    if val_subjects is not None and test_subjects is not None:
        train_subjects = [s for s in subjects if s not in set(val_subjects + test_subjects)]
    else:
        val_subjects = subjects[: args.val_size_subjects]
        test_subjects = subjects[
            args.val_size_subjects : args.val_size_subjects + args.test_size_subjects
        ]
        train_subjects = [s for s in subjects if s not in set(val_subjects + test_subjects)]

    splitter = SubjectWiseSplitter(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splitter.split(
        X_seq, y_seq, metadata_seq
    )

    _X_train_s, _X_val_s, X_test_s, scaler = scale_like_train(X_train, X_val, X_test)

    # Ensure deterministic inference.
    probs_float = keras_model.predict(X_test_s, verbose=0)
    float_logits = probs_float.astype(np.float32)


    # Load INT8 TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()

    in_scale, in_zero, out_scale, out_zero = quant_params_from_interpreter(interpreter)

    # Run INT8 inference (quantize input + collect quantized output logits)
    out_int8_logits_q = run_int8_inference(
        interpreter,
        X_test_s=X_test_s,
        input_scale=in_scale,
        input_zero_point=in_zero,
    )

    # Dequantize output logits back to float32 before argmax
    out_float_logits = (out_int8_logits_q.astype(np.float32) - out_zero) * out_scale

    print("Unique labels:", np.unique(y_test))
    print("Counts:", np.bincount(y_test))
    print("=" * 50)
    print("Unique y_test labels:", np.unique(y_test))
    print("Support:", np.bincount(y_test))

    print("Float logits shape:", float_logits.shape)
    print("INT8 logits shape:", out_float_logits.shape)

    print("Model output shape:", keras_model.output_shape)
    print("=" * 50)
    # Metrics
    float_metrics = metrics_from_logits(float_logits, y_test)
    int8_metrics = metrics_from_logits(out_float_logits, y_test)

    # Agreement / difference
    y_pred_float = float_metrics.pop("y_pred")
    y_pred_int8 = int8_metrics.pop("y_pred")

    diff = (y_pred_float != y_pred_int8)
    diff_pct = float(np.mean(diff) * 100.0)
    agreement_pct = float(100.0 - diff_pct)

    print("================ FLOAT32 ================")
    print(f"Accuracy: {float_metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy: {float_metrics['balanced_accuracy']:.6f}")
    print(f"Macro F1: {float_metrics['f1_macro']:.6f}")
    print(f"Weighted F1: {float_metrics['f1_weighted']:.6f}")
    print(f"Kappa: {float_metrics['cohen_kappa']:.6f}")
    print(f"MCC: {float_metrics['mcc']:.6f}")
    print()

    print("================ INT8 ===================")
    print(f"Accuracy: {int8_metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy: {int8_metrics['balanced_accuracy']:.6f}")
    print(f"Macro F1: {int8_metrics['f1_macro']:.6f}")
    print(f"Weighted F1: {int8_metrics['f1_weighted']:.6f}")
    print(f"Kappa: {int8_metrics['cohen_kappa']:.6f}")
    print(f"MCC: {int8_metrics['mcc']:.6f}")
    print()

    print("Prediction Agreement:")
    print(f"  Agreement: {agreement_pct:.2f}%")
    print("Prediction Difference:")
    print(f"  INT8 != FLOAT32 on: {diff_pct:.2f}% samples")
    print()

    # Confusion matrices + classification reports
    cm_float = float_metrics["confusion_matrix"]
    cm_int8 = int8_metrics["confusion_matrix"]

    print("--- Confusion Matrix (FLOAT32) ---")
    print(cm_float)
    print("--- Confusion Matrix (INT8) ---")
    print(cm_int8)
    print()

    print("--- Classification Report (FLOAT32) ---")
    print(classification_report(y_test, y_pred_float, zero_division=0))
    print("--- Classification Report (INT8) ---")
    print(classification_report(y_test, y_pred_int8, zero_division=0))
    print()

    # Summary: absolute differences
    diff_report = {
        "accuracy_abs_diff": abs(float_metrics["accuracy"] - int8_metrics["accuracy"]),
        "balanced_accuracy_abs_diff": abs(
            float_metrics["balanced_accuracy"] - int8_metrics["balanced_accuracy"]
        ),
        "f1_macro_abs_diff": abs(float_metrics["f1_macro"] - int8_metrics["f1_macro"]),
        "f1_weighted_abs_diff": abs(
            float_metrics["f1_weighted"] - int8_metrics["f1_weighted"]
        ),
        "kappa_abs_diff": abs(float_metrics["cohen_kappa"] - int8_metrics["cohen_kappa"]),
        "mcc_abs_diff": abs(float_metrics["mcc"] - int8_metrics["mcc"]),
        "prediction_difference_pct": diff_pct,
    }

    print("=== Deployment suitability summary (Ethos-U55) ===")
    print("Absolute metric differences (FLOAT32 vs INT8):")
    for k, v in diff_report.items():
        if k.endswith("pct"):
            continue
        print(f"- {k}: {v:.6f}")
    print(f"- prediction_difference_pct: {diff_report['prediction_difference_pct']:.2f}%")

    # Save JSON (optional)
    out_json_path = Path(args.output_json)
    out_payload = {
        "keras_model": str(args.keras_model),
        "tflite_model": str(args.tflite_model),
        "tag": {
            "sequence_length": args.sequence_length,
            "step": args.step,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        },
        "float32": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in float_metrics.items()
            if k != "confusion_matrix"
        },
        "float32_confusion_matrix": cm_float.tolist(),
        "int8": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in int8_metrics.items()
            if k != "confusion_matrix"
        },
        "int8_confusion_matrix": cm_int8.tolist(),
        "prediction_agreement_pct": agreement_pct,
        "prediction_difference_pct": diff_pct,
        "metric_abs_differences": diff_report,
    }

    try:
        out_json_path.write_text(json.dumps(out_payload, indent=2))
        print(f"Saved comparison metrics to: {out_json_path}")
    except Exception as e:
        print(f"WARNING: failed to save {out_json_path}: {e}")


if __name__ == "__main__":
    main()
