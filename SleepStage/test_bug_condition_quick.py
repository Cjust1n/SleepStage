#!/usr/bin/env python
"""Quick test to verify bug condition is detectable."""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import (
    TrainConfig,
    create_sequences_and_split,
    _scale_sequence_data,
    build_model,
)

# Set GPU memory
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

# Create buggy configuration
cfg = TrainConfig(
    processed_dir="processed",
    outputs_dir="outputs_quick",
    use_feature_subset=True,
    sequence_length=30,
    model_type="tcn_lstm",
    dilations=(1, 2, 4, 8),
    patience=5,
    epochs=5,
    batch_size=64,
    learning_rate=1e-3,
    use_undersampling=True,
    use_class_weight=True,
    val_size_subjects=1,
    test_size_subjects=1,
    seed=42,
)

print("Creating sequences...")
(X_train, y_train, X_val, y_val, X_test, y_test, _, _) = create_sequences_and_split(cfg)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

X_train_s, X_val_s, X_test_s, _ = _scale_sequence_data(X_train, X_val, X_test)

num_classes = int(np.unique(y_train).max() + 1)
model = build_model(cfg, X_train.shape[1], X_train.shape[2], num_classes)

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

print("Training...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=cfg.patience, 
        min_delta=1e-4, restore_best_weights=True, verbose=0
    ),
]

history = model.fit(
    X_train_s, y_train,
    validation_data=(X_val_s, y_val),
    batch_size=cfg.batch_size,
    epochs=cfg.epochs,
    callbacks=callbacks,
    verbose=1,
    class_weight=class_weight,
)

# Print results
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
gap = train_acc - val_acc

print(f"\nResults:")
print(f"  Train accuracy: {train_acc:.4f}")
print(f"  Val accuracy: {val_acc:.4f}")
print(f"  Train-val gap: {gap:.4f}")
print(f"  Epochs trained: {len(history.history['accuracy'])}")

# Check bug
bug_exists = val_acc < 0.66 or gap > 0.02
print(f"\nBug status: {'BUG EXISTS (val < 0.66 or gap > 0.02)' if bug_exists else 'Bug may be fixed'}")
