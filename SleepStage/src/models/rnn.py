"""rnn.py

Builtins-only residual TCN for sleep stage classification.

The legacy function name ``build_gru_model`` is preserved for compatibility
with the training pipeline, but the implementation is a lightweight residual
TCN intended for fully quantized TFLite INT8 deployment on Ethos-U55.
"""

from __future__ import annotations

import tensorflow as tf


def _residual_block(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float,
    l2_weight_decay: float,
    name: str,
) -> tf.Tensor:
    """Residual TCN block using only builtins-friendly layers.

    The block follows the classic TCN pattern:
    Conv1D -> ReLU -> Dropout -> Conv1D -> Add(residual) -> ReLU
    """

    residual = x

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation=None,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name=f"{name}_conv1",
    )(x)
    y = tf.keras.layers.ReLU(name=f"{name}_relu1")(y)
    # Use SpatialDropout1D so entire feature maps are dropped consistently across
    # the temporal axis. For Conv1D/TCN blocks this usually regularizes better
    # than element-wise Dropout because it prevents the block from over-relying
    # on a few channels while preserving local temporal structure.
    y = tf.keras.layers.SpatialDropout1D(dropout, name=f"{name}_drop1")(y)

    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation=None,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name=f"{name}_conv2",
    )(y)

    # Only project the shortcut when the channel count changes.
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=None,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
            name=f"{name}_res_proj",
        )(residual)

    x = tf.keras.layers.Add(name=f"{name}_add")([residual, y])
    x = tf.keras.layers.ReLU(name=f"{name}_out_relu")(x)
    return x


def build_gru_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    hidden_units: int = 64,
    dropout: float = 0.2,
    bidirectional: bool = False,
    l2_weight_decay: float = 1e-4,
    kernel_size: int = 3,
    use_global_skip_connections: bool = True,
) -> tf.keras.Model:
    """Build a lightweight residual TCN classifier.

Args:
        sequence_length: Number of timesteps in the input sequence.
        feature_dim: Number of features per timestep.
        num_classes: Number of target classes.
        hidden_units: Channel width (number of filters per residual block).
        dropout: Dropout probability inside residual blocks.
        bidirectional: Kept for API compatibility and intentionally ignored.
        l2_weight_decay: L2 regularisation strength for convolution and dense
            kernels. A slightly stronger value than the previous default is
            typically more helpful here because the validation curve shows
            classic overfitting after the early epochs.
        kernel_size: Temporal convolution kernel size.
use_global_skip_connections: When True, concatenate intermediate block
            outputs before pooling.
    """

    del bidirectional

    # Use the requested width directly (no cap), so the caller controls the
    # model capacity via hidden_units.
    filters = int(hidden_units)

    inputs = tf.keras.Input(shape=(sequence_length, feature_dim), name="x")

    # Stem projection aligns the 18 engineered features to the TCN channel space.
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding="same",
        activation=None,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="stem_proj",
    )(inputs)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)

    # Four dilation levels give a wider receptive field while keeping the model
    # small and fully causal: [1, 2, 4, 8].
    block_outputs = []
    for i, dilation_rate in enumerate((1, 2, 4, 8), start=1):
        x = _residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=dropout,
            l2_weight_decay=l2_weight_decay,
            name=f"tcn_block_{i}_d{dilation_rate}",
        )
        block_outputs.append(x)

    # Global skip concatenation preserves features from multiple temporal scales
    # without changing the recurrent-free, builtins-only design.
    if use_global_skip_connections and len(block_outputs) > 1:
        x = tf.keras.layers.Concatenate(axis=-1, name="global_skip_concat")(block_outputs)

    # Pooling choice:
    # - GAP captures average temporal evidence.
    # - GMP captures the strongest stage-specific activation.
    # Their concatenation is a common lightweight compromise for sequence
    # classification and remains TFLite/Ethos-U55 friendly.
    # gap = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    # gmp = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)
    # x = tf.keras.layers.Concatenate(name="pool_concat")([gap, gmp])
    x = tf.keras.layers.GlobalMaxPooling1D(name="gmp")(x)

    # Slightly richer classifier head improves separability without a major
    # parameter increase and keeps ReLU-only activations.
    x = tf.keras.layers.Dense(
        filters // 2,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="dense_hidden",
    )(x)
    x = tf.keras.layers.ReLU(name="dense_hidden_relu")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight_decay),
        name="logits",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="residual_tcn_classifier",
    )
    model.compile(
        # Adam remains a good fit for this small, dense feature sequence model.
        # The actual training loop may override the learning rate, but we keep
        # a conservative default here for standalone compatibility.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
