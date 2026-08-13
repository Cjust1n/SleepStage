import numpy as np
import tensorflow as tf

from src.training.train import (
    load_config,
    create_sequences_and_split,
    _scale_sequence_data,
)

MODEL_PATH = "outputs/checkpoints/gru_int8.tflite"
CONFIG_PATH = "configs/train_selected_features.yaml"

NUM_SAMPLES = 100

###########################################################################
# Load dataset using EXACTLY the same preprocessing as training
###########################################################################

cfg = load_config(CONFIG_PATH)

(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    metadata,
    _
) = create_sequences_and_split(cfg)

X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
    X_train,
    X_val,
    X_test,
)

###########################################################################
# Load TFLite model
###########################################################################

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input shape :", input_details["shape"])
print("Input dtype :", input_details["dtype"])
print("Quantization:", input_details["quantization"])

assert tuple(X_test_s.shape[1:]) == tuple(input_details["shape"][1:])

###########################################################################
# Quantization parameters
###########################################################################

scale, zero_point = input_details["quantization"]

if scale == 0:
    raise RuntimeError("Input tensor is not quantized!")

###########################################################################
# Prepare samples
###########################################################################

golden_input = []
golden_output = []

for sample in X_test_s[:NUM_SAMPLES]:

    sample_q = np.round(sample / scale + zero_point)
    sample_q = np.clip(sample_q, -128, 127).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        sample_q[np.newaxis, :, :]
    )

    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])

    pred = int(np.argmax(output))

    golden_input.append(sample_q)
    golden_output.append(pred)

golden_input = np.asarray(golden_input, dtype=np.int8)
golden_output = np.asarray(golden_output, dtype=np.uint8)

###########################################################################
# Statistics
###########################################################################

print("\nGenerated samples:", len(golden_input))
print("Prediction counts :", np.bincount(golden_output))

###########################################################################
# Write Header
###########################################################################

HEADER_NAME = "golden_reference.h"

with open(HEADER_NAME, "w") as f:

    f.write("#ifndef GOLDEN_REFERENCE_H\n")
    f.write("#define GOLDEN_REFERENCE_H\n\n")

    f.write("#include <stdint.h>\n\n")

    f.write(f"constexpr int NUM_SAMPLES = {NUM_SAMPLES};\n")
    f.write("constexpr int SEQ_LEN = 30;\n")
    f.write("constexpr int NUM_FEATURES = 16;\n\n")

    #######################################################################
    # Input
    #######################################################################

    f.write(
        "const int8_t golden_input[NUM_SAMPLES][SEQ_LEN][NUM_FEATURES] = {\n"
    )

    for sample in golden_input:

        f.write("    {\n")

        for timestep in sample:

            values = ", ".join(map(str, timestep))

            f.write(f"        {{{values}}},\n")

        f.write("    },\n")

    f.write("};\n\n")

    #######################################################################
    # Output
    #######################################################################

    values = ", ".join(map(str, golden_output))

    f.write(
        f"const uint8_t golden_output[NUM_SAMPLES] = {{{values}}};\n\n"
    )

    #######################################################################
    # Ground truth (optional)
    #######################################################################

    gt = ", ".join(map(str, y_test[:NUM_SAMPLES]))

    f.write(
        f"const uint8_t ground_truth[NUM_SAMPLES] = {{{gt}}};\n\n"
    )

    f.write("#endif\n")

print("\nGenerated:", HEADER_NAME)