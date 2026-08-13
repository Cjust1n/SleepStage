#!/usr/bin/env python
"""Quick bug condition exploration script.

This script demonstrates the overfitting bug by training with buggy configurations
and measuring if the expected behavior holds.

Expected on UNFIXED code:
- validation_accuracy < 0.66 (BUG)
- train_accuracy - val_accuracy > 0.02 (BUG)
- rem_accuracy < 0.40 (BUG)

After fix, all assertions should pass.
"""

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
from src.training.evaluate import evaluate_model


def run_bug_condition_test():
    """Run training with bug-triggering configuration."""
    
    # Create buggy configuration
    cfg = TrainConfig(
        processed_dir="processed",
        outputs_dir="outputs_bug_exploration",
        use_feature_subset=True,
        sequence_length=30,  # BUG: too short
        step=1,
        require_contiguous=True,
        model_type="tcn_lstm",
        hidden_units=64,
        filters=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),  # BUG: limited receptive field
        dropout=0.3,
        use_separable_conv=False,
        batch_size=64,
        epochs=20,  # Reduced for faster exploration
        learning_rate=1e-3,
        patience=5,  # BUG: too aggressive
        val_size_subjects=1,
        test_size_subjects=1,
        seed=42,
        use_class_weight=True,
        use_undersampling=True,  # BUG: aggressive undersampling
        undersample_seed=42,
        use_prebuilt_sequences=False,
        quantize_int8=False,
    )
    
    print("=" * 70)
    print("BUG CONDITION EXPLORATION TEST")
    print("=" * 70)
    print("\nConfiguration (all bug factors present):")
    print(f"  - use_undersampling: {cfg.use_undersampling}")
    print(f"  - sequence_length: {cfg.sequence_length}")
    print(f"  - dilations: {cfg.dilations}")
    print(f"  - patience: {cfg.patience}")
    
    # Create sequences and split
    print("\nCreating sequences and splits...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        metadata_seq,
        subjects,
    ) = create_sequences_and_split(cfg)
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    num_classes = int(np.unique(y_train).max() + 1)
    sequence_length = X_train.shape[1]
    feature_dim = X_train.shape[2]
    
    # Scale data
    print("\nScaling data...")
    X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
        X_train, X_val, X_test
    )
    
    # Build model
    print("\nBuilding TCN-LSTM model...")
    model = build_model(cfg, sequence_length, feature_dim, num_classes)
    
    # Compute class weights
    print("\nComputing class weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    
    for c, w in class_weight.items():
        print(f"  Class {c}: {w:.4f}")
    
    # Train model
    print("\nTraining model with bug-triggering configuration...")
    print("(This should show overfitting if the bug exists)")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="outputs_bug_exploration/best.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(1, cfg.patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    history = model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=2,
        class_weight=class_weight,
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    eval_result = evaluate_model(
        model,
        X_test_s,
        y_test,
        output_dir=Path("outputs_bug_exploration"),
    )
    
    # Extract metrics
    val_accuracy = eval_result.accuracy
    train_accuracy = history.history["accuracy"][-1]
    train_val_gap = train_accuracy - val_accuracy
    rem_accuracy = eval_result.recall_per_class[3] if len(eval_result.recall_per_class) > 3 else 0.0
    
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print(f"Train Accuracy (final): {train_accuracy:.4f}")
    print(f"Train-Val Gap: {train_val_gap:.4f}")
    print(f"Epochs Trained: {len(history.history['accuracy'])}")
    print(f"Recall per class: {eval_result.recall_per_class}")
    print(f"REM Classification Accuracy (class 3): {rem_accuracy:.4f}")
    
    # Show training history to visualize overfitting
    print("\nTraining History (last 5 epochs):")
    hist_len = len(history.history["accuracy"])
    start = max(0, hist_len - 5)
    for i in range(start, hist_len):
        acc = history.history["accuracy"][i]
        val_acc = history.history["val_accuracy"][i]
        loss = history.history["loss"][i]
        val_loss = history.history["val_loss"][i]
        print(f"  Epoch {i+1}: acc={acc:.4f}, val_acc={val_acc:.4f}, loss={loss:.4f}, val_loss={val_loss:.4f}")
    
    # Check bug condition indicators
    print("\n" + "=" * 70)
    print("BUG CONDITION ANALYSIS")
    print("=" * 70)
    
    print("\nExpected Behavior (after fix):")
    print("  ✓ validation_accuracy >= 0.66")
    print("  ✓ train_accuracy - val_accuracy < 0.02")
    print("  ✓ rem_accuracy >= 0.40")
    
    print("\nActual Results:")
    
    bug_found = False
    
    if val_accuracy >= 0.66:
        print(f"  ✓ validation_accuracy: {val_accuracy:.4f} >= 0.66 ✓")
    else:
        print(f"  ✗ validation_accuracy: {val_accuracy:.4f} < 0.66 (BUG DETECTED)")
        bug_found = True
    
    if train_val_gap < 0.02:
        print(f"  ✓ train_accuracy - val_accuracy: {train_val_gap:.4f} < 0.02 ✓")
    else:
        print(f"  ✗ train_accuracy - val_accuracy: {train_val_gap:.4f} >= 0.02 (BUG DETECTED)")
        bug_found = True
    
    if rem_accuracy >= 0.40:
        print(f"  ✓ rem_accuracy: {rem_accuracy:.4f} >= 0.40 ✓")
    else:
        print(f"  ✗ rem_accuracy: {rem_accuracy:.4f} < 0.40 (BUG DETECTED)")
        bug_found = True
    
    print("\n" + "=" * 70)
    if bug_found:
        print("✗ BUG CONDITION EXPLORATION TEST: FAILED (as expected)")
        print("  This failure confirms the overfitting bug exists on unfixed code.")
        print("  The test demonstrates the bug by failing when:")
        print("    - validation accuracy drops below 0.66")
        print("    - training-validation gap exceeds 0.02")
        print("    - REM classification accuracy falls below 0.40")
    else:
        print("✓ BUG CONDITION EXPLORATION TEST: PASSED")
        print("  This suggests the bug may already be fixed or test conditions")
        print("  are not sufficient to trigger it.")
    print("=" * 70)
    
    return bug_found


if __name__ == "__main__":
    # Set GPU memory to avoid allocation issues
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Warning: {e}")
    
    run_bug_condition_test()
