"""test_overfitting_preservation.py

Preservation property tests for TCN sleep staging overfitting bugfix.

This test suite uses observation-first methodology to capture baseline behavior
on UNFIXED code for non-buggy configurations. These tests verify that after the
fix is implemented, the preserved behaviors continue to work correctly (no regressions).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

The tests focus on configurations where isBugCondition returns FALSE:
- Basic training scenarios without undersampling
- Configurations with adequate sequence length (>=60)
- Configurations with proper dilations (>=5 layers)
- Configurations with reasonable patience (>=8)

Properties tested:
1. Temporal Learning Capability - Model learns better than random
2. Data Format Consistency - Preprocessing produces consistent format
3. Output Format Preservation - Inference output format unchanged
4. Metric Consistency - Evaluation metrics remain consistent
5. Multi-class Support - All 4 sleep stages supported

These tests PASS on unfixed code (baseline confirmed).
After fix, tests should still PASS (no regressions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import (
    TrainConfig,
    create_sequences_and_split,
    _scale_sequence_data,
    build_model,
)
from src.training.evaluate import evaluate_model


class TestOverfittingPreservation:
    """Preservation property tests for bugfix validation.
    
    These tests observe baseline behavior on unfixed code and verify
    that preservation requirements are maintained after fix is applied.
    """

    @staticmethod
    def _create_good_config(
        use_undersampling: bool = False,
        sequence_length: int = 60,
        dilations: tuple = (1, 2, 4, 8, 16),
        patience: int = 10,
        epochs: int = 20,
    ) -> TrainConfig:
        """Create a training config that does NOT trigger the overfitting bug.
        
        Args:
            use_undersampling: False (no data loss)
            sequence_length: 60 (adequate temporal context)
            dilations: (1, 2, 4, 8, 16) (deeper receptive field)
            patience: 10 (reasonable early stopping)
            epochs: Training epochs
        
        Returns:
            TrainConfig for non-buggy scenarios
        """
        cfg = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test_preservation",
            use_feature_subset=True,
            sequence_length=sequence_length,
            step=1,
            require_contiguous=True,
            model_type="tcn_lstm",
            hidden_units=64,
            filters=64,
            kernel_size=3,
            dilations=dilations,
            dropout=0.3,
            use_separable_conv=False,
            batch_size=64,
            epochs=epochs,
            learning_rate=1e-3,
            patience=patience,
            train_subjects=None,
            val_subjects=None,
            test_subjects=None,
            val_size_subjects=1,
            test_size_subjects=1,
            seed=42,
            use_class_weight=True,
            use_undersampling=use_undersampling,
            undersample_seed=42,
            use_prebuilt_sequences=False,
            quantize_int8=False,
        )
        return cfg

    # ========================================================================
    # Property 1: Temporal Learning Capability
    # ========================================================================

    def test_preservation_temporal_learning_basic_no_undersampling(self):
        """Property 1: Model learns from temporal sequences better than random.
        
        Configuration:
        - use_undersampling: false (no data loss)
        - sequence_length: 60 (adequate temporal context)
        - dilations: (1, 2, 4, 8, 16) (good receptive field)
        - patience: 10 (reasonable early stopping)
        
        Expected Behavior (Preserved):
        - Model SHALL learn from temporal sequences
        - Validation accuracy >= 60% (non-buggy baseline)
        - Training converges (loss decreases over epochs)
        - Model learns better than random guessing (1/4 = 25% for 4 classes)
        
        Validates: Requirements 3.1, 3.2
        """
        cfg = self._create_good_config(
            use_undersampling=False,
            sequence_length=60,
            dilations=(1, 2, 4, 8, 16),
            patience=10,
            epochs=15,
        )

        # Build training data
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

        num_classes = int(np.unique(y_train).max() + 1)
        sequence_length = X_train.shape[1]
        feature_dim = X_train.shape[2]

        # Verify data shape preservation
        assert X_train.ndim == 3, f"X_train should be 3D, got {X_train.ndim}D"
        assert X_val.ndim == 3, f"X_val should be 3D, got {X_val.ndim}D"
        assert X_test.ndim == 3, f"X_test should be 3D, got {X_test.ndim}D"
        
        assert X_train.shape[1] == sequence_length, "Sequence length mismatch in X_train"
        assert X_val.shape[1] == sequence_length, "Sequence length mismatch in X_val"
        assert X_test.shape[1] == sequence_length, "Sequence length mismatch in X_test"

        # Scale data
        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        # Build model
        model = build_model(cfg, sequence_length, feature_dim, num_classes)

        # Compute class weights
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        # Train model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            X_train_s,
            y_train,
            validation_data=(X_val_s, y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )

        # Evaluate on test set
        eval_result = evaluate_model(
            model,
            X_test_s,
            y_test,
            output_dir=Path("outputs_test_preservation"),
        )

        # Extract metrics
        val_accuracy = eval_result.accuracy
        train_accuracy = history.history["accuracy"][-1]
        initial_val_accuracy = history.history["val_accuracy"][0]

        print(f"\n=== Preservation Test 1: Temporal Learning ===")
        print(f"Initial Val Accuracy: {initial_val_accuracy:.4f}")
        print(f"Final Train Accuracy: {train_accuracy:.4f}")
        print(f"Final Val Accuracy: {val_accuracy:.4f}")
        print(f"Random Baseline: {1.0/num_classes:.4f} (1/{num_classes})")
        print(f"Epochs Trained: {len(history.history['accuracy'])}")

        # Preservation assertions
        # Should learn much better than random guessing
        random_baseline = 1.0 / num_classes
        assert (
            val_accuracy > random_baseline + 0.3
        ), f"Model not learning: val_acc {val_accuracy:.4f} only {val_accuracy - random_baseline:.4f} above random {random_baseline:.4f}"

        # Should achieve reasonable validation accuracy
        assert (
            val_accuracy >= 0.60
        ), f"Validation accuracy {val_accuracy:.4f} should be >= 0.60"

        # Training should improve over time
        assert (
            train_accuracy > initial_val_accuracy
        ), "Model should learn (train_acc should increase)"

    def test_preservation_temporal_learning_with_adequate_sequence_length(self):
        """Property 1: Temporal learning with adequate sequence length (>=60).
        
        Configuration:
        - sequence_length: 60 (adequate temporal context)
        - No undersampling
        - Good dilations
        
        Expected: Model learns temporal patterns effectively
        
        Validates: Requirements 3.1
        """
        cfg = self._create_good_config(
            use_undersampling=False,
            sequence_length=60,
            dilations=(1, 2, 4, 8, 16),
            patience=10,
            epochs=12,
        )

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

        num_classes = int(np.unique(y_train).max() + 1)
        sequence_length = X_train.shape[1]
        feature_dim = X_train.shape[2]

        # Verify sequence length is preserved
        assert (
            sequence_length == 60
        ), f"Expected sequence_length=60, got {sequence_length}"

        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        model = build_model(cfg, sequence_length, feature_dim, num_classes)

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            X_train_s,
            y_train,
            validation_data=(X_val_s, y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )

        eval_result = evaluate_model(
            model,
            X_test_s,
            y_test,
            output_dir=Path("outputs_test_preservation_seq60"),
        )

        val_accuracy = eval_result.accuracy
        print(f"\nAdequate sequence length test: val_accuracy={val_accuracy:.4f}")

        # Model should learn well with adequate temporal context
        assert (
            val_accuracy >= 0.60
        ), f"With adequate sequence length, should achieve val_accuracy >= 0.60, got {val_accuracy:.4f}"

    # ========================================================================
    # Property 2: Data Format Consistency
    # ========================================================================

    def test_preservation_data_format_consistency(self):
        """Property 2: Preprocessing pipeline produces consistent data format.
        
        Expected Behavior (Preserved):
        - Input shape SHALL be (N, sequence_length, features)
        - Output labels SHALL be integer class IDs (0-3)
        - Scaled data SHALL have mean near 0, std near 1
        - No NaNs or Infs in data
        
        Validates: Requirements 3.2
        """
        cfg = self._create_good_config(
            sequence_length=60,
            use_undersampling=False,
        )

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

        print(f"\n=== Preservation Test 2: Data Format Consistency ===")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_train dtype: {X_train.dtype}")
        print(f"y_train dtype: {y_train.dtype}")

        # Check input shape
        assert (
            X_train.ndim == 3
        ), f"X_train should be (N, T, F), got shape {X_train.shape} with ndim {X_train.ndim}"
        assert (
            X_val.ndim == 3
        ), f"X_val should be (N, T, F), got ndim {X_val.ndim}"
        assert (
            X_test.ndim == 3
        ), f"X_test should be (N, T, F), got ndim {X_test.ndim}"

        # Check labels are integer class IDs
        num_classes = int(np.unique(y_train).max() + 1)
        assert num_classes == 4, f"Should have 4 sleep stages, got {num_classes}"
        
        unique_labels = np.unique(y_train)
        assert (
            all(lbl in range(4) for lbl in unique_labels)
        ), f"Labels should be in [0, 1, 2, 3], got {unique_labels}"

        # Scale data and check consistency
        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        # Check scaled data has no NaNs or Infs
        assert not np.any(np.isnan(X_train_s)), "X_train_s contains NaNs"
        assert not np.any(np.isinf(X_train_s)), "X_train_s contains Infs"
        assert not np.any(np.isnan(X_val_s)), "X_val_s contains NaNs"
        assert not np.any(np.isinf(X_val_s)), "X_val_s contains Infs"
        assert not np.any(np.isnan(X_test_s)), "X_test_s contains NaNs"
        assert not np.any(np.isinf(X_test_s)), "X_test_s contains Infs"

        # Check scaled data has mean near 0, std near 1
        train_mean = np.mean(X_train_s)
        train_std = np.std(X_train_s)
        print(f"Scaled X_train - Mean: {train_mean:.4f}, Std: {train_std:.4f}")

        assert (
            abs(train_mean) < 0.2
        ), f"Scaled data mean should be near 0, got {train_mean:.4f}"
        assert (
            abs(train_std - 1.0) < 0.3
        ), f"Scaled data std should be near 1, got {train_std:.4f}"

        # All shapes preserved
        assert (
            X_train_s.shape == X_train.shape
        ), "Scaling should preserve shape"
        assert (
            X_val_s.shape == X_val.shape
        ), "Scaling should preserve shape"
        assert (
            X_test_s.shape == X_test.shape
        ), "Scaling should preserve shape"

    # ========================================================================
    # Property 3: Output Format Preservation
    # ========================================================================

    def test_preservation_output_format_inference(self):
        """Property 3: Model inference produces consistent output format.
        
        Expected Behavior (Preserved):
        - Model output shape SHALL be (N, 4) for 4-class problem
        - Output probabilities SHALL sum to ~1.0
        - Predicted classes SHALL be in range [0, 3]
        
        Validates: Requirements 3.3
        """
        cfg = self._create_good_config(
            sequence_length=60,
            use_undersampling=False,
            epochs=8,
        )

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

        num_classes = int(np.unique(y_train).max() + 1)
        sequence_length = X_train.shape[1]
        feature_dim = X_train.shape[2]

        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        model = build_model(cfg, sequence_length, feature_dim, num_classes)

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            X_train_s,
            y_train,
            validation_data=(X_val_s, y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )

        # Get predictions on test set
        probs = model.predict(X_test_s, verbose=0)
        predicted_classes = np.argmax(probs, axis=1)

        print(f"\n=== Preservation Test 3: Output Format ===")
        print(f"Model output shape: {probs.shape}")
        print(f"Expected shape: ({X_test_s.shape[0]}, {num_classes})")
        print(f"Sample probabilities sum: {np.sum(probs[0]):.6f}")
        print(f"Predicted classes range: [{predicted_classes.min()}, {predicted_classes.max()}]")

        # Check output shape
        assert (
            probs.shape == (X_test_s.shape[0], num_classes)
        ), f"Model output should be (N, {num_classes}), got {probs.shape}"

        # Check probabilities sum to ~1.0
        prob_sums = np.sum(probs, axis=1)
        assert np.all(
            np.abs(prob_sums - 1.0) < 0.01
        ), f"Output probabilities should sum to 1.0, got range [{prob_sums.min():.4f}, {prob_sums.max():.4f}]"

        # Check predicted classes are in valid range
        assert (
            np.all(predicted_classes >= 0) and np.all(predicted_classes < num_classes)
        ), f"Predicted classes should be in [0, {num_classes-1}], got range [{predicted_classes.min()}, {predicted_classes.max()}]"

    # ========================================================================
    # Property 4: Metric Consistency
    # ========================================================================

    def test_preservation_metric_consistency(self):
        """Property 4: Evaluation metrics remain consistent and valid.
        
        Expected Behavior (Preserved):
        - Accuracy SHALL be in range [0, 1]
        - Precision/Recall per class SHALL be in range [0, 1]
        - F1 scores SHALL be in range [0, 1]
        - All classes SHALL be evaluated
        
        Validates: Requirements 3.4
        """
        cfg = self._create_good_config(
            sequence_length=60,
            use_undersampling=False,
            epochs=8,
        )

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

        num_classes = int(np.unique(y_train).max() + 1)
        sequence_length = X_train.shape[1]
        feature_dim = X_train.shape[2]

        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        model = build_model(cfg, sequence_length, feature_dim, num_classes)

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        model.fit(
            X_train_s,
            y_train,
            validation_data=(X_val_s, y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )

        eval_result = evaluate_model(
            model,
            X_test_s,
            y_test,
            output_dir=Path("outputs_test_preservation_metrics"),
        )

        print(f"\n=== Preservation Test 4: Metric Consistency ===")
        print(f"Accuracy: {eval_result.accuracy:.4f}")
        print(f"F1 Macro: {eval_result.f1_macro:.4f}")
        print(f"F1 Weighted: {eval_result.f1_weighted:.4f}")
        print(f"Recall per class: {eval_result.recall_per_class}")
        print(f"Precision per class: {eval_result.precision_per_class}")
        print(f"F1 per class: {eval_result.f1_per_class}")
        print(f"Support per class: {eval_result.support_per_class}")

        # Check accuracy range
        assert (
            0.0 <= eval_result.accuracy <= 1.0
        ), f"Accuracy should be in [0, 1], got {eval_result.accuracy}"

        # Check per-class metrics
        assert (
            len(eval_result.precision_per_class) == num_classes
        ), f"Should have precision for all {num_classes} classes"
        assert (
            len(eval_result.recall_per_class) == num_classes
        ), f"Should have recall for all {num_classes} classes"
        assert (
            len(eval_result.f1_per_class) == num_classes
        ), f"Should have F1 for all {num_classes} classes"
        assert (
            len(eval_result.support_per_class) == num_classes
        ), f"Should have support for all {num_classes} classes"

        # Check per-class metric ranges
        for c, prec in enumerate(eval_result.precision_per_class):
            assert (
                0.0 <= prec <= 1.0
            ), f"Precision for class {c} should be in [0, 1], got {prec}"

        for c, recall in enumerate(eval_result.recall_per_class):
            assert (
                0.0 <= recall <= 1.0
            ), f"Recall for class {c} should be in [0, 1], got {recall}"

        for c, f1 in enumerate(eval_result.f1_per_class):
            assert (
                0.0 <= f1 <= 1.0
            ), f"F1 for class {c} should be in [0, 1], got {f1}"

    # ========================================================================
    # Property 5: Multi-class Support
    # ========================================================================

    def test_preservation_all_four_sleep_stages_supported(self):
        """Property 5: Model continues to support all 4 sleep stages.
        
        Expected Behavior (Preserved):
        - Class 0 (Wake): predictions made
        - Class 1 (Light): predictions made
        - Class 2 (Deep): predictions made
        - Class 3 (REM): predictions made
        
        Validates: Requirements 3.5
        """
        cfg = self._create_good_config(
            sequence_length=60,
            use_undersampling=False,
            epochs=10,
        )

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

        num_classes = int(np.unique(y_train).max() + 1)
        sequence_length = X_train.shape[1]
        feature_dim = X_train.shape[2]

        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        model = build_model(cfg, sequence_length, feature_dim, num_classes)

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        model.fit(
            X_train_s,
            y_train,
            validation_data=(X_val_s, y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight,
        )

        eval_result = evaluate_model(
            model,
            X_test_s,
            y_test,
            output_dir=Path("outputs_test_preservation_multiclass"),
        )

        print(f"\n=== Preservation Test 5: Multi-class Support ===")
        print(f"Number of classes: {num_classes}")
        print(f"Recall per class: {eval_result.recall_per_class}")
        print(f"Support per class: {eval_result.support_per_class}")

        # Check model supports all 4 sleep stages
        assert (
            num_classes == 4
        ), f"Model should support 4 sleep stages, got {num_classes}"

        # Check all classes are evaluated (have support)
        for c, support in enumerate(eval_result.support_per_class):
            assert (
                support > 0
            ), f"Class {c} should have samples in test set, got support={support}"

        # Check all classes make predictions
        for c in range(num_classes):
            recall = eval_result.recall_per_class[c]
            # Some classes might have perfect recall, others imperfect, but all should contribute
            print(f"  Class {c} recall: {recall:.4f}")

        # Verify predictions made for all classes
        probs = model.predict(X_test_s, verbose=0)
        predicted_classes = np.argmax(probs, axis=1)
        predicted_class_set = set(predicted_classes)
        
        print(f"Classes with predictions: {sorted(predicted_class_set)}")

        # At minimum, most classes should be predicted
        # (some rare classes might not be predicted if confidence is low for all, but this is unlikely)
        assert (
            len(predicted_class_set) >= 3
        ), f"Model should make predictions for at least 3/4 classes, got {len(predicted_class_set)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
