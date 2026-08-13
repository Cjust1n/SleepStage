#!/usr/bin/env python3
"""
MRE: SimpleRNN → TFLite Converter Investigation for TF 2.16.1

Tests:
1. Baseline: SimpleRNN + TFLiteConverter.from_keras_model() → ops list
2. from_saved_model() vs from_keras_model()
3. Converter flags to force UNIDIRECTIONAL_SEQUENCE_RNN
4. Model architecture variants (unroll, implementation, no dropout, etc.)
5. Quantization vs no quantization
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")
print(f"Python version: {sys.version}")

SEQUENCE_LENGTH = 30
FEATURE_DIM = 22
NUM_CLASSES = 4
HIDDEN_UNITS = 64


def build_simple_rnn(dropout=0.0, unroll=False, implementation=2, use_bias=True):
    """Build a SimpleRNN model with configurable params."""
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name="x")
    x = tf.keras.layers.SimpleRNN(
        HIDDEN_UNITS,
        activation="tanh",
        return_sequences=False,
        dropout=dropout,
        recurrent_dropout=0.0,
        unroll=unroll,
        implementation=implementation,
        use_bias=use_bias,
        name="simple_rnn",
    )(inputs)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=None, name="logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="simple_rnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def examine_tflite_ops(tflite_path: str):
    """Extract op codes from a TFLite model."""
    import numpy as np
    # Use flatbuffers directly via TFLite schema
    # Alternative: use tflite interpreter to list ops
    try:
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        
        # Get op details via metadata if available
        # Otherwise use flatbuffers inspection
        from tensorflow.lite.python import schema_py_generated as schema
        
        with open(tflite_path, "rb") as f:
            buf = f.read()
        
        model_obj = schema.Model.GetRootAsModel(buf, 0)
        subgraphs_len = model_obj.SubgraphsLength()
        
        all_ops = []
        for sg_idx in range(subgraphs_len):
            subgraph = model_obj.Subgraphs(sg_idx)
            ops_len = subgraph.OperatorsLength()
            for op_idx in range(ops_len):
                op = subgraph.Operators(op_idx)
                opcode_idx = op.OpcodeIndex()
                opcode = model_obj.OperatorCodes(opcode_idx)
                builtin_code = opcode.BuiltinCode()
                custom_code = opcode.CustomCode()
                if custom_code and custom_code.decode("utf-8", errors="replace"):
                    all_ops.append(f"FLEX/CUSTOM:{custom_code.decode('utf-8', errors='replace')}")
                else:
                    # Try to get builtin name
                    try:
                        builtin_name = tf.lite.OpSet.TFLITE_BUILTINS.name_for_builtin_code(builtin_code)
                        if not builtin_name:
                            builtin_name = f"BUILTIN_CODE_{builtin_code}"
                        all_ops.append(builtin_name)
                    except:
                        all_ops.append(f"BUILTIN_CODE_{builtin_code}")
        
        # Also get input/output details
        in_detail = interp.get_input_details()[0]
        out_detail = interp.get_output_details()[0]
        in_scale, in_zero = in_detail.get("quantization", (None, None))
        out_scale, out_zero = out_detail.get("quantization", (None, None))
        
        return {
            "ops": all_ops,
            "unique_ops": sorted(set(all_ops)),
            "input_shape": in_detail["shape"].tolist(),
            "input_dtype": str(in_detail["dtype"]),
            "output_shape": out_detail["shape"].tolist(),
            "output_dtype": str(out_detail["dtype"]),
            "input_quant": {"scale": in_scale, "zero_point": in_zero},
            "output_quant": {"scale": out_scale, "zero_point": out_zero},
            "op_count": len(all_ops),
        }
    except Exception as e:
        return {"error": str(e)}


def tflite_detail_dump(tflite_path: str):
    """Fallback: use tflite interpreter op profiling to identify ops."""
    try:
        interp = tf.lite.Interpreter(
            model_path=tflite_path,
            experimental_preserve_all_tensors=True,
        )
        interp.allocate_tensors()
        
        # Get the TFLite model as a flatbuffer string
        model_str = interp._get_legacy_model()
        
        # Parse flatbuffer to get op codes
        from tensorflow.lite.python import schema_py_generated as schema_fb
        
        model_obj = schema_fb.Model.GetRootAsModel(model_str, 0)
        
        ops = []
        for i in range(model_obj.OperatorCodesLength()):
            oc = model_obj.OperatorCodes(i)
            code = oc.BuiltinCode()
            custom = oc.CustomCode()
            version = oc.Version()
            if custom and custom.decode("utf-8", errors="replace"):
                ops.append(f"CUSTOM:{custom.decode('utf-8')}(v{version})")
            else:
                ops.append(f"BUILTIN:{code}(v{version})")
        
        return {"opcodes": ops, "num_opcodes": len(ops)}
    except Exception as e:
        return {"error_detail": str(e)}


def test_converter_flag(
    name: str,
    converter_flags: dict,
    model: tf.keras.Model,
    save_dir: str,
    quantize: bool = False,
    use_savedmodel: bool = False,
):
    """Test a converter flag configuration."""
    os.makedirs(save_dir, exist_ok=True)
    
    tflite_path = os.path.join(save_dir, f"{name}.tflite")
    
    if use_savedmodel:
        sm_path = os.path.join(save_dir, f"{name}_savedmodel")
        model.export(sm_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(sm_path)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply all flags
    for k, v in converter_flags.items():
        if hasattr(converter, k):
            setattr(converter, k, v)
        elif k == "_experimental_lower_tensor_list_ops":
            try:
                converter._experimental_lower_tensor_list_ops = v
            except:
                pass
        elif k == "experimental_enable_resource_variables":
            try:
                converter.experimental_enable_resource_variables = v
            except:
                pass
        elif k == "_experimental_disable_per_channel":
            try:
                converter._experimental_disable_per_channel = v
            except:
                pass
        elif k == "experimental_new_converter":
            try:
                converter.experimental_new_converter = v
            except:
                pass
        elif k == "_experimental_default_to_single_batch_in_tensor_list_ops":
            try:
                converter._experimental_default_to_single_batch_in_tensor_list_ops = v
            except:
                pass
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.allow_custom_ops = False
        converter.experimental_enable_resource_variables = False
        
        # Representative dataset
        def representative_dataset():
            rng = np.random.default_rng(42)
            for _ in range(100):
                yield [rng.normal(0, 1, (1, SEQUENCE_LENGTH, FEATURE_DIM)).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
    
    try:
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        result = examine_tflite_ops(tflite_path)
        result["converter_flags"] = {k: str(v) for k, v in converter_flags.items()}
        result["quantize"] = quantize
        result["use_savedmodel"] = use_savedmodel
        result["status"] = "SUCCESS"
        result["file_size"] = os.path.getsize(tflite_path)
        return result
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
            "converter_flags": {k: str(v) for k, v in converter_flags.items()},
            "quantize": quantize,
            "use_savedmodel": use_savedmodel,
        }


def main():
    BASE_DIR = "/tmp/mre_investigation"
    
    # ================================================================
    # TEST 1: Baseline - SimpleRNN with default settings
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Baseline SimpleRNN (dropout=0.0, unroll=False, impl=2)")
    print("=" * 80)
    
    model = build_simple_rnn(dropout=0.0, unroll=False, implementation=2)
    result = test_converter_flag(
        "test1_baseline", {}, model, f"{BASE_DIR}/test1"
    )
    print(json.dumps(result, indent=2, default=str))
    
    # ================================================================
    # TEST 2: Baseline with quantization
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Baseline + INT8 Quantization")
    print("=" * 80)
    
    model = build_simple_rnn(dropout=0.0, unroll=False, implementation=2)
    result = test_converter_flag(
        "test2_quantized", {}, model, f"{BASE_DIR}/test2", quantize=True
    )
    print(json.dumps(result, indent=2, default=str))
    
    # ================================================================
    # TEST 3: from_saved_model vs from_keras_model
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 3a: from_keras_model()")
    print("=" * 80)
    
    model_3 = build_simple_rnn(dropout=0.0, unroll=False, implementation=2)
    result_keras = test_converter_flag(
        "test3a_from_keras", {}, model_3, f"{BASE_DIR}/test3", use_savedmodel=False
    )
    print(json.dumps(result_keras, indent=2, default=str))
    
    print("\n" + "=" * 80)
    print("TEST 3b: from_saved_model()")
    print("=" * 80)
    
    model_3b = build_simple_rnn(dropout=0.0, unroll=False, implementation=2)
    result_sm = test_converter_flag(
        "test3b_from_savedmodel", {}, model_3b, f"{BASE_DIR}/test3", use_savedmodel=True
    )
    print(json.dumps(result_sm, indent=2, default=str))
    
    # ================================================================
    # TEST 4: Converter flags to force UNIDIRECTIONAL_SEQUENCE_RNN
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Converter flags exploration")
    print("=" * 80)
    
    flag_configs = [
        {"name": "4a_lower_tensor_list", "flags": {"_experimental_lower_tensor_list_ops": True}},
        {"name": "4b_resource_vars_false", "flags": {"experimental_enable_resource_variables": False}},
        {"name": "4c_resource_vars_true", "flags": {"experimental_enable_resource_variables": True}},
        {"name": "4d_disable_per_channel", "flags": {"_experimental_disable_per_channel": True}},
        {"name": "4e_new_converter", "flags": {"experimental_new_converter": True}},
        {"name": "4f_mlir_converter", "flags": {"experimental_enable_mlir_converter": True}},
        {"name": "4g_single_batch_tensorlist", "flags": {"_experimental_default_to_single_batch_in_tensor_list_ops": True}},
        {"name": "4h_all_fixes", "flags": {
            "_experimental_lower_tensor_list_ops": True,
            "experimental_enable_resource_variables": False,
            "_experimental_default_to_single_batch_in_tensor_list_ops": True,
            "experimental_new_converter": True,
        }},
    ]
    
    for cfg in flag_configs:
        print(f"\n--- {cfg['name']} ---")
        model_f = build_simple_rnn(dropout=0.0, unroll=False, implementation=2)
        result = test_converter_flag(
            cfg["name"], cfg["flags"], model_f, f"{BASE_DIR}/test4"
        )
        ops_summary = result.get("unique_ops", result.get("ops", []))
        has_flex = any("FLEX" in str(op) or "SELECT_TF" in str(op) for op in ops_summary)
        has_while = any("WHILE" in str(op) for op in ops_summary)
        has_uni_rnn = any("UNIDIRECTIONAL_SEQUENCE_RNN" in str(op) for op in ops_summary)
        print(f"  Status: {result.get('status')}")
        print(f"  Has FLEX: {has_flex}, Has WHILE: {has_while}, Has UNI_RNN: {has_uni_rnn}")
        print(f"  Ops: {ops_summary}")
        if result.get("status") == "FAILED":
            print(f"  Error: {result.get('error')}")
    
    # ================================================================
    # TEST 5: Model architecture variants
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Model architecture variants")
    print("=" * 80)
    
    arch_configs = [
        {"name": "5a_unroll_true", "kwargs": {"unroll": True}},
        {"name": "5b_implementation_1", "kwargs": {"implementation": 1}},
        {"name": "5c_no_bias", "kwargs": {"use_bias": False}},
        {"name": "5d_dropout_0.2", "kwargs": {"dropout": 0.2}},
        {"name": "5e_unroll_no_bias", "kwargs": {"unroll": True, "use_bias": False}},
        {"name": "5f_unroll_impl1", "kwargs": {"unroll": True, "implementation": 1}},
    ]
    
    for cfg in arch_configs:
        print(f"\n--- {cfg['name']} ---")
        model_a = build_simple_rnn(**cfg["kwargs"])
        result = test_converter_flag(
            cfg["name"], {}, model_a, f"{BASE_DIR}/test5"
        )
        ops_summary = result.get("unique_ops", result.get("ops", []))
        has_flex = any("FLEX" in str(op) or "SELECT_TF" in str(op) for op in ops_summary)
        has_while = any("WHILE" in str(op) for op in ops_summary)
        has_uni_rnn = any("UNIDIRECTIONAL_SEQUENCE_RNN" in str(op) for op in ops_summary)
        print(f"  Status: {result.get('status')}")
        print(f"  Has FLEX: {has_flex}, Has WHILE: {has_while}, Has UNI_RNN: {has_uni_rnn}")
        print(f"  Ops: {ops_summary}")
        if result.get("status") == "FAILED":
            print(f"  Error: {result.get('error')}")
    
    # ================================================================
    # TEST 6: from_keras_model vs from_savedmodel + quantized
    # ================================================================
    print("\n" + "=" * 80)
    print("TEST 6a: quantized from_keras_model() — all fixes")
    print("=" * 80)
    
    model_6a = build_simple_rnn(dropout=0.0, unroll=True, implementation=1)
    result_6a = test_converter_flag(
        "test6a_keras_quant",
        {
            "_experimental_lower_tensor_list_ops": True,
            "experimental_enable_resource_variables": False,
        },
        model_6a,
        f"{BASE_DIR}/test6",
        quantize=True,
        use_savedmodel=False,
    )
    ops_summary = result_6a.get("unique_ops", result_6a.get("ops", []))
    has_flex = any("FLEX" in str(op) or "SELECT_TF" in str(op) for op in ops_summary)
    has_while = any("WHILE" in str(op) for op in ops_summary)
    has_uni_rnn = any("UNIDIRECTIONAL_SEQUENCE_RNN" in str(op) for op in ops_summary)
    print(f"  Status: {result_6a.get('status')}")
    print(f"  Has FLEX: {has_flex}, Has WHILE: {has_while}, Has UNI_RNN: {has_uni_rnn}")
    print(f"  Ops: {ops_summary}")
    print(f"  Input: {result_6a.get('input_shape')} {result_6a.get('input_dtype')}")
    
    print("\n" + "=" * 80)
    print("TEST 6b: quantized from_saved_model() — all fixes")
    print("=" * 80)
    
    model_6b = build_simple_rnn(dropout=0.0, unroll=True, implementation=1)
    result_6b = test_converter_flag(
        "test6b_savedmodel_quant",
        {
            "_experimental_lower_tensor_list_ops": True,
            "experimental_enable_resource_variables": False,
        },
        model_6b,
        f"{BASE_DIR}/test6",
        quantize=True,
        use_savedmodel=True,
    )
    ops_summary = result_6b.get("unique_ops", result_6b.get("ops", []))
    has_flex = any("FLEX" in str(op) or "SELECT_TF" in str(op) for op in ops_summary)
    has_while = any("WHILE" in str(op) for op in ops_summary)
    has_uni_rnn = any("UNIDIRECTIONAL_SEQUENCE_RNN" in str(op) for op in ops_summary)
    print(f"  Status: {result_6b.get('status')}")
    print(f"  Has FLEX: {has_flex}, Has WHILE: {has_while}, Has UNI_RNN: {has_uni_rnn}")
    print(f"  Ops: {ops_summary}")
    print(f"  Input: {result_6b.get('input_shape')} {result_6b.get('input_dtype')}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Collect all results
    all_results = {}
    for test_dir in ["test1", "test2", "test3", "test4", "test5", "test6"]:
        for f in os.listdir(f"{BASE_DIR}/{test_dir}"):
            if f.endswith(".json"):
                pass  # Could save results as JSON
    
    print("Investigation complete. Results saved to /tmp/mre_investigation/")


if __name__ == "__main__":
    main()

