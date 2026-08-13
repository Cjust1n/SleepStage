"""multi_tcn.py

Multi-branch TCN classifier for modality-aware sleep staging.

The model splits input features into modality-specific branches:
- Movement branch
- HRV branch
- HR branch
- Temporal branch

Each branch has its own capacity, then the branch embeddings are fused and
classified. The TCN core remains builtins-friendly for TFLite INT8 export.
"""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf


MOVEMENT_FEATURES = (
    "mean_acc",
    "std_acc",
    "variance",
    "energy",
    "rms",
    "movement_count",
    "movement_ratio",
    "acceleration_jerk",
    "zero_crossing",
    "rolling_mean_acc",
    "rolling_std_acc",
)

HRV_FEATURES = (
    "mean_ibi",
    "sdnn",
    "rmssd",
    "sdsd",
    "nn50",
    "pnn50",
    "lf",
    "hf",
    "lf_hf",
    "sd1",
    "sd2",
)

HR_FEATURES = (
    "mean_hr",
    "hr_delta",
    "rolling_mean_hr",
    "rolling_std_hr",
    "hr_slope",
    "rolling_hr_range",
)

TEMPORAL_FEATURES = (
    "relative_position",
    "sin_time_of_night",
    "cos_time_of_night",
)


def _conv_block(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    residual = x
    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        activation=None,
        use_bias=True,
        name=f"{name}_conv1",
    )(x)
    y = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = tf.keras.layers.ReLU(name=f"{name}_relu1")(y)
    y = tf.keras.layers.Dropout(dropout, name=f"{name}_drop1")(y)
    y = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        activation=None,
        use_bias=True,
        name=f"{name}_conv2",
    )(y)
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            activation=None,
            use_bias=True,
            name=f"{name}_res_proj",
        )(residual)
    y = tf.keras.layers.Add(name=f"{name}_add")([residual, y])
    return tf.keras.layers.ReLU(name=f"{name}_out_relu")(y)


def _branch_encoder(
    inputs: tf.Tensor,
    feature_names: Sequence[str],
    selected_names: Sequence[str],
    *,
    filters: int,
    kernel_size: int,
    dropout: float,
    branch_name: str,
) -> tf.Tensor:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    available_names = [name for name in selected_names if name in name_to_idx]
    missing = [name for name in selected_names if name not in name_to_idx]
    if not available_names:
        raise KeyError(
            f"No available feature(s) for {branch_name} branch. "
            f"Expected one of: {', '.join(selected_names)}"
        )

    idxs = [name_to_idx[name] for name in available_names]
    x = tf.keras.layers.Lambda(
        lambda t, ids=idxs: tf.gather(t, ids, axis=-1),
        name=f"{branch_name}_select",
    )(inputs)

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        padding="same",
        activation=None,
        use_bias=True,
        name=f"{branch_name}_stem",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{branch_name}_stem_bn")(x)
    x = tf.keras.layers.ReLU(name=f"{branch_name}_stem_relu")(x)

    x = _conv_block(
        x,
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        name=f"{branch_name}_block1",
    )
    x = _conv_block(
        x,
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        name=f"{branch_name}_block2",
    )
    return tf.keras.layers.GlobalAveragePooling1D(name=f"{branch_name}_gap")(x)


def build_multi_tcn_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    *,
    feature_names: Sequence[str],
    dropout: float = 0.2,
    kernel_size: int = 3,
    branch_filters: dict[str, int] | None = None,
) -> tf.keras.Model:
    """Build a modality-aware multi-branch TCN classifier."""

    del feature_dim

    branch_filters = branch_filters or {
        "movement": 48,
        "hrv": 48,
        "hr": 32,
        "temporal": 16,
    }

    inputs = tf.keras.Input(shape=(sequence_length, len(feature_names)), name="x")

    branches = []
    branch_specs = [
        ("movement", MOVEMENT_FEATURES, branch_filters["movement"]),
        ("hrv", HRV_FEATURES, branch_filters["hrv"]),
        ("hr", HR_FEATURES, branch_filters["hr"]),
        ("temporal", TEMPORAL_FEATURES, branch_filters["temporal"]),
    ]

    for branch_name, branch_features, branch_width in branch_specs:
        available = [name for name in branch_features if name in feature_names]
        if not available:
            continue
        branches.append(
            _branch_encoder(
                inputs,
                feature_names,
                available,
                filters=branch_width,
                kernel_size=kernel_size,
                dropout=dropout,
                branch_name=branch_name,
            )
        )

    if not branches:
        raise ValueError(
            "multi_tcn requires at least one supported feature to build a branch."
        )

    x = tf.keras.layers.Concatenate(name="branch_concat")(branches)
    x = tf.keras.layers.Dense(64, activation="relu", name="fusion_dense")(x)
    x = tf.keras.layers.Dropout(dropout, name="fusion_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="multi_tcn_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
