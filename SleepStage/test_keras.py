import tensorflow as tf

model = tf.keras.models.load_model(
    "outputs/saved_model.keras",
    compile=False,
)

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)