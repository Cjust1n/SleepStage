import tensorflow as tf


def build_gru_real_model(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    hidden_units: int = 64,
    dropout: float = 0.0,
    bidirectional: bool = False,
    l2_weight_decay: float = 0.0,
):
    if bidirectional:
        raise ValueError(
            "bidirectional=True is disabled for Ethos-U55 deployment."
        )

    inputs = tf.keras.Input(
        shape=(sequence_length, feature_dim),
        name="input",
        dtype=tf.float32,
    )

    x = tf.keras.layers.GRU(
        units=hidden_units,

        # IMPORTANT
        activation="tanh",
        recurrent_activation="sigmoid",

        # Only need final hidden state
        return_sequences=False,

        # Important for TFLite conversion
        return_state=False,

        # Avoid dynamic TensorList / While graph
        unroll=True,

        # Keep standard GRU formulation
        reset_after=True,

        dropout=0.0,
        recurrent_dropout=0.0,

        kernel_regularizer=(
            tf.keras.regularizers.l2(l2_weight_decay)
            if l2_weight_decay > 0
            else None
        ),
        recurrent_regularizer=(
            tf.keras.regularizers.l2(l2_weight_decay)
            if l2_weight_decay > 0
            else None
        ),

        name="gru",
    )(inputs)

    if dropout > 0.0:
        x = tf.keras.layers.Dropout(
            dropout,
            name="dropout",
        )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="sleepstage_gru_ethosu",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model