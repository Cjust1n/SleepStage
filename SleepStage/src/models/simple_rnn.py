"""simple_rnn.py

Pure SimpleRNN model for sleep stage sequence classification.

This model uses tf.keras.layers.SimpleRNN (not GRU/LSTM) because:
- SimpleRNN maps to TFLite builtin op UNIDIRECTIONAL_SEQUENCE_RNN
- No recurrent_dropout (avoids TensorList/Flex ops during TFLite conversion)
- No Bidirectional wrapper (avoids Flex ops)
- Fully quantizable to INT8 with builtins-only TFLite

Jika kuantisasi INT8 gagal, kemungkinan karena:
- recurrent_dropout > 0 (opsional dropout hanya di input, bukan recurrent)
- Bidirectional wrapper
- Custom layers yang tidak terdaftar
"""

from __future__ import annotations

import tensorflow as tf


def build_simple_rnn_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    hidden_units: int = 64,
    dropout: float = 0.0,
    l2_weight_decay: float = 1e-4,
) -> tf.keras.Model:
    """Build a pure SimpleRNN classifier that is quantizable to INT8 TFLite.

    Args:
        sequence_length: Number of timesteps per sequence.
        feature_dim: Number of features per timestep.
        num_classes: Number of target sleep stage classes.
        hidden_units: Number of units in the SimpleRNN layer.
        dropout: Dropout rate (applied to INPUT only, NOT recurrent).
            Set to 0.0 for best TFLite compatibility.
        l2_weight_decay: L2 regularisation strength.
    """

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        name="x",
    )

    # SimpleRNN — pure RNN builtin for TFLite.
    # Penting: recurrent_dropout=0.0 (TIDAK digunakan) karena dapat
    # menghasilkan TensorList ops yang mencegah kuantisasi builtins-only.
    #
    # unroll=True: statically unrolls the RNN computation (sequence_length steps).
    # This eliminates tf.while_loop / TensorArray / TensorListReserve ops from the
    # exported SavedModel, which are the root cause of the TFLite conversion failure:
    #   "'tf.TensorListReserve' op requires element_shape to be static during TF Lite
    #    transformation pass"
    # With unroll=True the graph contains only UNIDIRECTIONAL_SEQUENCE_RNN builtin ops
    # (or fully unrolled FC ops), keeping the model fully INT8-quantizable with
    # builtins-only TFLite (required for Ethos-U55 / TFLM deployment).
    x = tf.keras.layers.SimpleRNN(
        hidden_units,
        activation="tanh",
        return_sequences=False,
        dropout=dropout if dropout > 0.0 else 0.0,
        recurrent_dropout=0.0,  # HARUS 0 — recurrent_dropout causes Flex/TensorList ops
        use_bias=True,
        unroll=True,  # CRITICAL: eliminates TensorList/while_loop ops for TFLite builtins-only conversion
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        recurrent_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="simple_rnn",
    )(inputs)

    # Optional dropout after RNN (only during training, not in TFLite graph).
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(dropout, name="rnn_dropout")(x)

    # Output logits (no softmax — handled by SparseCategoricalCrossentropy).
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="logits",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="simple_rnn_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
