"""inference_runner.py

Batch TFLite inference runner for INT8 sleep-stage inputs.

This utility reads a binary file containing sequential INT8 samples with shape
``(sequence_length, feature_dim)``, runs inference sample-by-sample, writes the
predicted class indices to ``predictions.bin``, and prints aggregate timing
statistics.

The runner is intentionally split into reusable helpers so the same core logic
can be reused in a desktop TFLite workflow or ported to an embedded runner.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class RunnerConfig:
    tflite_model: Path
    input_bin: Path
    predictions_bin: Path
    sequence_length: int = 30
    feature_dim: int = 22
    cpu_frequency_hz: int | None = None
    npu_frequency_hz: int | None = None


@dataclass
class BatchRunResult:
    num_samples: int
    average_inference_time_s: float
    average_cpu_cycles: float | None
    average_npu_cycles: float | None
    predictions: np.ndarray


def load_interpreter(tflite_model: Path) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model))
    interpreter.allocate_tensors()
    return interpreter


def get_tensor_details(interpreter: tf.lite.Interpreter) -> tuple[dict, dict]:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    return input_details, output_details


def load_input_samples(
    input_bin: Path,
    sequence_length: int,
    feature_dim: int,
) -> np.ndarray:
    if not input_bin.exists():
        raise FileNotFoundError(f"Missing input file: {input_bin}")

    raw = np.fromfile(input_bin, dtype=np.int8)
    sample_size = sequence_length * feature_dim
    if sample_size <= 0:
        raise ValueError("sequence_length and feature_dim must be positive")

    if raw.size % sample_size != 0:
        raise ValueError(
            "input.bin size is not divisible by the sample size. "
            f"Got {raw.size} bytes, expected a multiple of {sample_size}."
        )

    num_samples = raw.size // sample_size
    if num_samples == 0:
        raise ValueError("input.bin does not contain any complete samples")

    return raw.reshape(num_samples, sequence_length, feature_dim)


def infer_sample(
    interpreter: tf.lite.Interpreter,
    sample: np.ndarray,
    input_index: int,
    output_index: int,
) -> np.ndarray:
    interpreter.set_tensor(input_index, sample)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)


def _estimate_cycles(
    average_time_s: float,
    clock_frequency_hz: int | None,
) -> float | None:
    if clock_frequency_hz is None or clock_frequency_hz <= 0:
        return None
    return average_time_s * float(clock_frequency_hz)


def run_batch_inference(
    interpreter: tf.lite.Interpreter,
    samples: np.ndarray,
    cpu_frequency_hz: int | None = None,
    npu_frequency_hz: int | None = None,
) -> BatchRunResult:
    input_details, output_details = get_tensor_details(interpreter)
    input_index = int(input_details["index"])
    output_index = int(output_details["index"])

    expected_shape = tuple(int(dim) for dim in input_details["shape"])
    if len(expected_shape) != 3:
        raise ValueError(
            "Expected a 3D input tensor shape [batch, sequence_length, feature_dim], "
            f"got {expected_shape}"
        )

    sample_shape = tuple(samples.shape[1:])
    if sample_shape != tuple(expected_shape[1:]):
        raise ValueError(
            "Input sample shape does not match the model input tensor. "
            f"Model expects {expected_shape[1:]}, but input.bin contains {sample_shape}."
        )

    predictions = np.empty(samples.shape[0], dtype=np.uint8)
    total_time_s = 0.0

    # Resize once to batch size 1 so the runtime always receives one sample at a time.
    if expected_shape[0] != 1:
        interpreter.resize_tensor_input(input_index, (1, *sample_shape))
        interpreter.allocate_tensors()
        input_details, output_details = get_tensor_details(interpreter)
        input_index = int(input_details["index"])
        output_index = int(output_details["index"])

    for sample_index, sample in enumerate(samples):
        batched_sample = np.asarray(sample, dtype=np.int8)[np.newaxis, :, :]

        start_time = time.perf_counter()
        logits = infer_sample(interpreter, batched_sample, input_index, output_index)
        total_time_s += time.perf_counter() - start_time

        predictions[sample_index] = np.uint8(int(np.argmax(logits[0], axis=-1)))

    average_time_s = total_time_s / float(samples.shape[0])
    average_cpu_cycles = _estimate_cycles(average_time_s, cpu_frequency_hz)
    average_npu_cycles = _estimate_cycles(average_time_s, npu_frequency_hz)

    return BatchRunResult(
        num_samples=int(samples.shape[0]),
        average_inference_time_s=average_time_s,
        average_cpu_cycles=average_cpu_cycles,
        average_npu_cycles=average_npu_cycles,
        predictions=predictions,
    )


def save_predictions(predictions: np.ndarray, predictions_bin: Path) -> None:
    predictions_bin.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(predictions, dtype=np.uint8).tofile(predictions_bin)


def print_summary(result: BatchRunResult) -> None:
    print(f"Processed samples: {result.num_samples}")
    print(f"Average inference time: {result.average_inference_time_s * 1000.0:.6f} ms")

    if result.average_cpu_cycles is not None:
        print(f"Average CPU cycles: {result.average_cpu_cycles:.2f}")
    else:
        print("Average CPU cycles: N/A (provide --cpu_frequency_hz to estimate)")

    if result.average_npu_cycles is not None:
        print(f"Average NPU cycles: {result.average_npu_cycles:.2f}")
    else:
        print("Average NPU cycles: N/A (provide --npu_frequency_hz to estimate)")


def parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Batch TFLite inference runner")
    parser.add_argument("--tflite_model", type=Path, required=True)
    parser.add_argument("--input_bin", type=Path, default=Path("input.bin"))
    parser.add_argument("--predictions_bin", type=Path, default=Path("predictions.bin"))
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--feature_dim", type=int, default=22)
    parser.add_argument(
        "--cpu_frequency_hz",
        type=int,
        default=None,
        help="Optional CPU clock frequency used to estimate CPU cycles from wall time.",
    )
    parser.add_argument(
        "--npu_frequency_hz",
        type=int,
        default=None,
        help="Optional NPU clock frequency used to estimate NPU cycles from wall time.",
    )
    args = parser.parse_args()

    return RunnerConfig(
        tflite_model=args.tflite_model,
        input_bin=args.input_bin,
        predictions_bin=args.predictions_bin,
        sequence_length=args.sequence_length,
        feature_dim=args.feature_dim,
        cpu_frequency_hz=args.cpu_frequency_hz,
        npu_frequency_hz=args.npu_frequency_hz,
    )


def main() -> None:
    config = parse_args()

    interpreter = load_interpreter(config.tflite_model)
    samples = load_input_samples(
        config.input_bin,
        sequence_length=config.sequence_length,
        feature_dim=config.feature_dim,
    )

    result = run_batch_inference(
        interpreter,
        samples,
        cpu_frequency_hz=config.cpu_frequency_hz,
        npu_frequency_hz=config.npu_frequency_hz,
    )
    save_predictions(result.predictions, config.predictions_bin)
    print_summary(result)


if __name__ == "__main__":
    main()