"""cnn.py

1D Convolutional Neural Network (CNN) for sleep stage sequence classification.

The model applies multiple Conv1D blocks with increasing dilation rates to
capture temporal patterns at various scales, followed by global pooling and
a dense classifier head.

This is a CNN-only alternative to the RNN/TCN-LSTM models.
"""

from __future__ import annotations

import tensorflow as tf


def build_cnn_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    filters: int = 64,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    dropout: float = 0.3,
    l2_weight_decay: float = 1e-4,
) -> tf.keras.Model:
    """Build a 1D CNN classifier for sleep stage sequence data.

    Args:
        sequence_length: Number of timesteps per sequence.
        feature_dim: Number of features per timestep.
        num_classes: Number of target sleep stage classes.
        filters: Number of convolution filters in each Conv1D block.
        kernel_size: Temporal kernel size.
        dilations: Dilation rates for stacked Conv1D layers. More dilations
            give larger receptive field without increasing parameters.
        dropout: Dropout probability after each Conv1D block.
        l2_weight_decay: L2 regularisation strength.
    """

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        name="x",
    )

    # Stem projection: align input channels to filter count.
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding="same",
        activation=None,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="stem_proj",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_act")(x)

    # Stacked Conv1D blocks with increasing dilation rates.
    for i, dilation_rate in enumerate(dilations, start=1):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation=None,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            name=f"conv_block_{i}_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"conv_block_{i}_bn")(x)
        x = tf.keras.layers.Activation("relu", name=f"conv_block_{i}_act")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"conv_block_{i}_dropout")(x)

    # Global pooling to reduce time dimension to a fixed-size vector.
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)

    # Optional intermediate dense layer for more representational capacity.
    x = tf.keras.layers.Dense(
        filters,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="dense_hidden",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)

    # Output logits (no softmax — handled by SparseCategoricalCrossentropy).
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="logits",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model

