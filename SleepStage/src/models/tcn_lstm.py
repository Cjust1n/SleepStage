"""tcn_lstm.py

Hybrid TCN -> LSTM sequence classifier.

The model keeps the temporal axis throughout the TCN stack, then passes the
resulting sequence to an LSTM for long-range dependency modeling.
"""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


def _temporal_conv(
    use_separable_conv: bool,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    name: str,
) -> tf.keras.layers.Layer:
    """Create a causal temporal convolution for one residual branch.

    SeparableConv1D does not accept `padding="causal"`, so causal behavior is
    implemented explicitly with left zero padding before the convolution.
    """

    if use_separable_conv:
        pad_width = dilation_rate * (kernel_size - 1)
        return tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding1D(
                    padding=(pad_width, 0),
                    name=f"{name}_pad",
                ),
                tf.keras.layers.SeparableConv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    padding="valid",
                    activation="tanh",
                    use_bias=True,
                    name=name,
                ),
            ],
            name=f"{name}_separable_causal",
        )

    return tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="tanh",
        use_bias=True,
        name=name,
    )


def _tcn_residual_block(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float,
    use_separable_conv: bool,
    block_index: int,
) -> tf.Tensor:
    """Apply one causal temporal residual block.

    The residual path is projected only when the channel dimension differs from
    the block width so the sequence length remains unchanged.
    """

    residual = x

    y = _temporal_conv(
        use_separable_conv=use_separable_conv,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        name=f"tcn_block_{block_index}_conv1",
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f"tcn_block_{block_index}_bn1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"tcn_block_{block_index}_dropout")(y)
    y = _temporal_conv(
        use_separable_conv=use_separable_conv,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        name=f"tcn_block_{block_index}_conv2",
    )(y)

    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=None,
            use_bias=True,
            name=f"tcn_block_{block_index}_residual_proj",
        )(residual)

    return tf.keras.layers.Add(name=f"tcn_block_{block_index}_add")([residual, y])


def build_tcn_lstm_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    hidden_units: int = 64,
    filters: int = 64,
    kernel_size: int = 3,
    dilations: Sequence[int] = (1, 2, 4, 8),
    dropout: float = 0.2,
    use_separable_conv: bool = False,
) -> tf.keras.Model:
    """Build a hybrid TCN -> LSTM classifier.

    Args:
        sequence_length: Number of timesteps per sequence.
        feature_dim: Number of features per timestep.
        num_classes: Number of target classes.
        hidden_units: Units in the final LSTM layer.
        filters: Number of convolution filters in each TCN block.
        kernel_size: Temporal kernel size used by the TCN blocks.
        dilations: Dilation rates for the residual TCN stack.
        dropout: Dropout probability used in the TCN stack and dense head.
        use_separable_conv: When True, replace Conv1D with SeparableConv1D in
            the TCN blocks for a more deployment-friendly temporal extractor.
    """

    inputs = tf.keras.Input(shape=(sequence_length, feature_dim), name="x")

    # Project features to the TCN width before the residual stack.
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding="same",
        activation=None,
        use_bias=True,
        name="stem_projection",
    )(inputs)

    for block_index, dilation_rate in enumerate(dilations, start=1):
        x = _tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=dropout,
            use_separable_conv=use_separable_conv,
            block_index=block_index,
        )

    x = tf.keras.layers.LSTM(
        hidden_units,
        activation="tanh",
        recurrent_activation="softmax",
        return_sequences=False,
        name="lstm",
    )(x)

    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tcn_lstm_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model