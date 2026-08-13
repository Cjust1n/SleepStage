"""
quantize_tflite.py

Convert a TensorFlow SavedModel to fully INT8 TFLite.
Compatible with TensorFlow 2.16+ / Keras 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf


def quantize_savedmodel_to_int8(
    saved_model_dir: str | Path,
    tflite_path: str | Path,
    representative_data: np.ndarray,
    representative_steps: int = 200,
) -> None:
    saved_model_dir = Path(saved_model_dir)
    tflite_path = Path(tflite_path)

    tflite_path.parent.mkdir(parents=True, exist_ok=True)

    if representative_data.ndim != 3:
        raise ValueError(
            "Expected representative_data shape (N,T,F). "
            "In your pipeline, pass X_train_s (sequence data) not processed/X_normalized.npy. "
            f"Got shape {representative_data.shape}" 
        )


    steps = min(representative_steps, representative_data.shape[0])

    def representative_dataset() -> Iterator[list[np.ndarray]]:
        for i in range(steps):
            yield [representative_data[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(saved_model_dir)
    )

    # Enable optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset
    converter.representative_dataset = representative_dataset

    # -------- IMPORTANT FIXES / INT8-FULL STRATEGY (FAIL-FAST) --------
    # Tujuan:
    # - Jika model tidak bisa direduksi menjadi TFLite builtins-only (no TF Select ops),
    #   maka converter harus FAIL agar kita tahu kuantisasi INT8 benar-benar tidak tercapai.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Fail fast untuk custom ops
    converter.allow_custom_ops = False

    # Resource variables: must be True for Keras 3 exported SavedModels.
    # Keras 3 uses resource variables internally; disabling this causes
    # "Variable constant folding is failed" during TFLite conversion.
    converter.experimental_enable_resource_variables = True

    # Boundary input/output harus int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # tf.lite.constants tidak tersedia di beberapa versi TF, jadi jangan set ini.

    # ------------------------------------------------------------------
    # Pre-conversion diagnostic: inspect the SavedModel for ops that are
    # known to break builtins-only TFLite conversion (e.g. TensorList ops
    # from tf.while_loop / TensorArray in unrolled=False RNN layers).
    # ------------------------------------------------------------------
    def _diagnose_saved_model(saved_model_dir: Path) -> None:
        """Scan the SavedModel graph for ops that commonly break TFLite conversion."""
        try:
            from tensorflow.python.tools import saved_model_utils
            from tensorflow.core.framework import graph_pb2

            tag_set = "serve"
            graph_def = saved_model_utils.get_meta_graph_def(
                str(saved_model_dir), tag_set
            ).graph_def

            problematic_ops = {
                "TensorListReserve",
                "TensorListStack",
                "TensorListFromTensor",
                "TensorListSetItem",
                "TensorListGetItem",
                "TensorListLength",
                "TensorListPushBack",
                "TensorListPopBack",
                "While",
                "StatelessWhile",
                "TensorArrayV2",
                "TensorArrayV3",
                "TensorArrayWriteV2",
                "TensorArrayWriteV3",
                "TensorArrayReadV2",
                "TensorArrayReadV3",
                "TensorArraySizeV2",
                "TensorArraySizeV3",
                "TensorArrayGatherV2",
                "TensorArrayGatherV3",
                "TensorArrayScatterV2",
                "TensorArrayScatterV3",
                "TensorArrayCloseV2",
                "TensorArrayCloseV3",
            }

            op_counts: dict[str, int] = {}
            for node in graph_def.node:
                op = node.op
                op_counts[op] = op_counts.get(op, 0) + 1

            print("\n=== SavedModel Op Diagnostic ===")
            print(f"Total nodes: {len(graph_def.node)}")
            print(f"Unique ops: {len(op_counts)}")

            # Show all ops sorted by count (most frequent first)
            for op, count in sorted(op_counts.items(), key=lambda kv: -kv[1]):
                marker = "  <-- PROBLEMATIC" if op in problematic_ops else ""
                print(f"  {op}: {count}{marker}")

            problematic_found = [op for op in op_counts if op in problematic_ops]
            if problematic_found:
                print(
                    f"\n[WARNING] Found {len(problematic_found)} problematic op type(s) "
                    f"for builtins-only TFLite conversion: {sorted(problematic_found)}"
                )
                print(
                    "These typically come from tf.while_loop / TensorArray usage "
                    "(e.g. RNN layers with unroll=False)."
                )
                print(
                    "Fix: set unroll=True on RNN layers, or use SELECT_TF_OPS "
                    "(not deployable on Ethos-U55/TFLM)."
                )
            else:
                print("\n[OK] No problematic ops found. Builtins-only conversion should work.")
            print("=" * 40)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"[Diagnostic] Could not inspect SavedModel: {e}")

    _diagnose_saved_model(saved_model_dir)

    import traceback

    def _try_convert(tag: str) -> bytes:
        try:
            tflite_model = converter.convert()
            return tflite_model
        except Exception as e:
            # Lempar dengan traceback lengkap agar bisa di-diagnosis
            tb = traceback.format_exc()
            raise RuntimeError(
                f"[TFLITE CONVERT FAILED] tag={tag}\n"
                f"Original exception type: {type(e).__name__}\n"
                f"Original exception message: {e}\n\n"
                f"Full traceback:\n{tb}"
            ) from e
        

    tflite_model = _try_convert("builtins_only")
    tflite_path.write_bytes(tflite_model)
    print(f"INT8 model saved to {tflite_path} (builtins_only)")

    