"""lstm.py

Baseline LSTM model for sequence classification.

We use an LSTM + dense head.
"""

from __future__ import annotations

import tensorflow as tf


def build_lstm_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    hidden_units: int = 64,
    dropout: float = 0.2,
    bidirectional: bool = False,
    l2_weight_decay: float = 1e-4,
) -> tf.keras.Model:

    inputs = tf.keras.Input(shape=(sequence_length, feature_dim), name="x")

    lstm_layer = tf.keras.layers.LSTM(
        hidden_units,
        return_sequences=False,
        dropout=dropout,
        recurrent_dropout=dropout,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        recurrent_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
    )


    x = inputs
    if bidirectional:
        x = tf.keras.layers.Bidirectional(lstm_layer)(x)
    else:
        x = lstm_layer(x)


    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="logits",
    )(x)


    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model

