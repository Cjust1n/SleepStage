"""TCN.py

Sequence classifier used by the training pipeline.

IMPORT (Ethos-U55 / Vela / TFLite builtins-only INT8):
The original GRU implementation generated TensorList/Flex ops during
TFLite conversion. To ensure builtins-only conversion, we implement a
TCN-style (dilated causal Conv1D) sequence model instead.

The function is intentionally named `build_gru_model` to avoid changing
training code, but it no longer uses a GRU layer.
"""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class _TCNBlock(tf.keras.layers.Layer):

    """A simple dilated causal Conv1D block with residual connection."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout: float,
        activation: str = "relu",
        l2_weight_decay: float = 1e-4,
        name: str | None = None,

        trainable: bool = True,
        dtype=None,
        **kwargs,
    ) -> None:


        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.activation = activation

        self.conv1 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation=None,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            name="conv1",
        )

        self.act1 = tf.keras.layers.Activation(activation, name="act1")
        self.drop1 = tf.keras.layers.Dropout(dropout, name="drop1")


        self.conv2 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            activation=None,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            name="conv2",
        )



        # 1x1 projection for residual when channel dims differ
        self.res_proj = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=None,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            name="res_proj",
        )

        self.act_out = tf.keras.layers.Activation(activation, name="act_out")

    def call(self, x, training=None):
        residual = x

        y = self.conv1(x)
        y = self.act1(y)
        y = self.drop1(y, training=training)

        y = self.conv2(y)


        # residual projection (safe even when same channels)
        residual = self.res_proj(residual)
        out = self.act_out(residual + y)
        return out


def build_tcn_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    hidden_units: int = 64,
    dropout: float = 0.2,
    bidirectional: bool = False,
    l2_weight_decay: float = 1e-4,
) -> tf.keras.Model:

    """Build a builtins-only-friendly sequence model.

    We keep the original signature (including bidirectional) for compatibility,
    but `bidirectional` is ignored because the causal TCN is inherently
    directionally consistent.
    """

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        name="x",
    )

    # Keep channels aligned with `hidden_units`.
    x = tf.keras.layers.Conv1D(
        filters=hidden_units,
        kernel_size=1,
        padding="same",
        activation=None,
        use_bias=True,
        name="stem_proj",
    )(inputs)

    # TCN depth: enough receptive field for sequence_length=30.
    # receptive_field ~= 1 + 2*(k-1)*sum(dilations) for 2 convs/block.
    kernel_size = 3
    dilations = [1, 2, 4]

    for i, d in enumerate(dilations):
        x = _TCNBlock(
            filters=hidden_units,
            kernel_size=kernel_size,
            dilation_rate=d,
            dropout=dropout,
            name=f"tcn_block_{i}_d{d}",
        )(x)

    # Global pooling to convert time dimension -> fixed vector.
    x = tf.keras.layers.GlobalMaxPooling1D(name="gap")(x)


    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="logits",
    )(x)



    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tcn_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


