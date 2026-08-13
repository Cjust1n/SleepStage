#!/usr/bin/env python3
"""
visualize_sleep_from_features.py

Offline sleep-stage visualization from a recorded feature.csv using the
deployed INT8 TFLite model (gru_int8.tflite).

Pipeline (matches the training/eval pipeline in SleepStage/):
  1. Read feature.csv (18 features per epoch, board order).
  2. Z-score normalize each feature using outputs/scaler.json (fit on train).
  3. Build sliding windows of 30 epochs (matches the model's [1,30,18] input).
  4. Run the INT8 TFLite model (auto-quantize input, dequantize output logits).
  5. Render a hypnogram + confidence stackplot and save prediction.csv.

Stage mapping (board enum):
  0 = Wake, 1 = Light Sleep, 2 = Deep Sleep, 3 = REM

Usage:
  python visualize_sleep_from_features.py \
      --features recordings/2026-08-08_23-00-40/feature.csv \
      --model tinyml-arm-workflow/model/gru_int8.tflite \
      --scaler SleepStage/outputs/scaler.json \
      --out_dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Optional: tensorflow-lite runtime. Falls back to tflite_runtime if available.
try:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter
except Exception:
    try:
        import tflite_runtime.interpreter as tflite
        _Interpreter = tflite.Interpreter
    except Exception:
        _Interpreter = None


STAGE_NAMES = {0: "Wake", 1: "Light Sleep", 2: "Deep Sleep", 3: "REM"}
STAGE_COLORS = {0: "#f39c12", 1: "#3498db", 2: "#2ecc71", 3: "#9b59b6"}

# 18-feature columns in the exact order the model expects (matches
# models/feature_model.py and the board F= print order).
FEATURE_COLS = [
    "relative_position", "sd2", "sin_time_of_night", "rolling_mean_hr",
    "rmssd", "time_of_night", "energy", "rolling_hr_range",
    "acceleration_jerk", "rolling_mean_acc", "rms", "lf",
    "rolling_std_acc", "zero_crossing", "hr_slope", "hf", "lf_hf", "hr_delta",
]


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mean/std arrays from scaler.json (z-score params)."""
    data = json.loads(Path(path).read_text())
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def build_windows(features: np.ndarray, seq_len: int = 30, step: int = 1):
    """Return list of (window_start_epoch, window) for sliding windows."""
    n = features.shape[0]
    windows = []
    for start in range(0, n - seq_len + 1, step):
        windows.append((start, features[start : start + seq_len]))
    return windows


def run_model(interpreter, scaled_seq: np.ndarray) -> np.ndarray:
    """
    Run one scaled [30,18] sequence through the INT8 TFLite model.
    Returns dequantized float logits (length 4).
    """
    in_d = interpreter.get_input_details()[0]
    out_d = interpreter.get_output_details()[0]

    in_scale, in_zero = in_d.get("quantization") or (1.0, 0)
    if in_scale is None:
        in_scale, in_zero = 1.0, 0

    # Quantize scaled float input to int8.
    batch = scaled_seq.astype(np.float32)[None, ...]  # [1,30,18]
    Xq = np.round(batch / in_scale + in_zero).astype(np.int8)

    # Resize input tensor if needed.
    if tuple(in_d["shape"]) != tuple(Xq.shape):
        interpreter.resize_tensor_input(in_d["index"], Xq.shape)
        interpreter.allocate_tensors()
        in_d = interpreter.get_input_details()[0]
        out_d = interpreter.get_output_details()[0]

    interpreter.set_tensor(in_d["index"], Xq)
    interpreter.invoke()
    out_q = interpreter.get_tensor(out_d["index"])  # quantized logits

    out_scale, out_zero = out_d.get("quantization") or (1.0, 0)
    if out_scale is None:
        out_scale, out_zero = 1.0, 0
    logits = (out_q.astype(np.float32) - out_zero) * out_scale
    return logits[0]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Path to feature.csv")
    parser.add_argument("--model", required=True, help="Path to INT8 .tflite")
    parser.add_argument("--scaler", required=True, help="Path to scaler.json")
    parser.add_argument("--seq_len", type=int, default=30, help="Window length")
    parser.add_argument("--step", type=int, default=1, help="Window stride")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _Interpreter is None:
        raise SystemExit("No TFLite runtime found. Install tensorflow or tflite-runtime.")

    # 1. Load features
    df = pd.read_csv(args.features)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"feature.csv missing columns: {missing}")
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    print(f"Loaded {X.shape[0]} epochs x {X.shape[1]} features from {args.features}")

    # 2. Z-score normalize
    mean, std = load_scaler(args.scaler)
    if len(mean) != X.shape[1]:
        raise SystemExit(
            f"Scaler dim ({len(mean)}) != feature dim ({X.shape[1]}). "
            "scaler.json and feature.csv must use the SAME 18-feature ordering."
        )
    Xs = (X - mean) / std

    # 3. Build windows
    windows = build_windows(Xs, seq_len=args.seq_len, step=args.step)
    if not windows:
        raise SystemExit(
            f"Need at least {args.seq_len} epochs; feature.csv has {X.shape[0]}."
        )
    print(f"Built {len(windows)} sliding windows of length {args.seq_len}")

    # 4. Run model
    interpreter = _Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    preds = []      # (start_epoch, end_epoch, stage, confidence, probs)
    for start, win in windows:
        logits = run_model(interpreter, win)
        probs = softmax(logits)
        stage = int(np.argmax(probs))
        conf = float(probs[stage])
        preds.append((start, start + args.seq_len - 1, stage, conf, probs))

    # 5. Save results
    records = []
    for start, end, stage, conf, probs in preds:
        records.append({
            "window_start_epoch": start,
            "window_end_epoch": end,
            "epoch_center": (start + end) / 2.0,
            "stage": stage,
            "stage_name": STAGE_NAMES.get(stage, "?"),
            "confidence": round(conf, 4),
            "p_wake": round(float(probs[0]), 4),
            "p_light": round(float(probs[1]), 4),
            "p_deep": round(float(probs[2]), 4),
            "p_rem": round(float(probs[3]), 4),
        })
    result_df = pd.DataFrame(records)
    pred_csv = out_dir / "sleep_prediction.csv"
    result_df.to_csv(pred_csv, index=False)
    print(f"Saved predictions -> {pred_csv}")

    # 6. Visualize
    if HAVE_MPL:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 7), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )

        # Hypnogram
        centers = [p["epoch_center"] for p in records]
        stages = [p["stage"] for p in records]
        ax1.plot(centers, stages, drawstyle="steps-mid", color="#2c3e50", lw=1.6)
        ax1.scatter(centers, stages, c=[STAGE_COLORS.get(s, "#95a5a6") for s in stages],
                    s=22, zorder=3)
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels([STAGE_NAMES[0], STAGE_NAMES[1], STAGE_NAMES[2], STAGE_NAMES[3]])
        ax1.set_ylabel("Sleep Stage")
        ax1.set_title("Predicted Sleep Hypnogram (from feature.csv via INT8 TFLite)")
        ax1.grid(alpha=0.3)

        # Confidence stackplot
        labels = ["Wake", "Light", "Deep", "REM"]
        probs_arr = np.array([[p["p_wake"], p["p_light"], p["p_deep"], p["p_rem"]]
                              for p in records])
        ax2.stackplot(centers, probs_arr.T, labels=labels,
                      colors=[STAGE_COLORS[0], STAGE_COLORS[1], STAGE_COLORS[2], STAGE_COLORS[3]],
                      alpha=0.85)
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Epoch (center)")
        ax2.legend(loc="upper right", ncol=4, fontsize=8)
        ax2.set_ylim(0, 1)

        fig.tight_layout()
        png = out_dir / "sleep_hypnogram.png"
        plt.savefig(png, dpi=150)
        print(f"Saved plot -> {png}")

    # Summary
    from collections import Counter
    stage_counts = Counter(p["stage"] for p in records)
    print("\n=== Stage distribution (per 30-epoch window) ===")
    for s in sorted(stage_counts):
        print(f"  {STAGE_NAMES.get(s, s):<12}: {stage_counts[s]}")


if __name__ == "__main__":
    main()
