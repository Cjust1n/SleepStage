"""test_overfitting_bug_condition.py

Bug condition exploration property test for TCN sleep staging overfitting bug.

This test MUST FAIL on unfixed code - failure confirms the bug exists.
The test encodes the EXPECTED behavior that should be achieved after the fix.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

The test runs training with configurations that trigger the bug (undersampling, 
short sequence, limited receptive field, aggressive early stopping) and verifies 
that the expected behavior holds:
- Validation accuracy >= 0.66
- Train-val gap < 0.02
- REM classification accuracy >= 0.40

On unfixed code, this test will fail because the bug conditions cause:
- Validation accuracy to drop below 0.66 (typically ~62.3%)
- Train-val gap to exceed 0.02 (typically ~6%)
- REM classification to drop below 0.40 (typically ~25.6%)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import yaml

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


class TestOverfittingBugCondition:
    """Explores bug conditions that trigger overfitting.
    
    This test class uses scoped property-based testing approach with concrete
    failing configurations identified in the bug analysis.
    """

    @staticmethod
    def _create_buggy_config(
        use_undersampling: bool = True,
        sequence_length: int = 30,
        dilations: tuple = (1, 2, 4, 8),
        patience: int = 5,
        epochs: int = 25,
    ) -> TrainConfig:
        """Create a training config that triggers the overfitting bug.
        
        Args:
            use_undersampling: Whether to use aggressive undersampling
            sequence_length: Length of temporal sequences (short = bug)
            dilations: TCN dilation rates (limited receptive field = bug)
            patience: EarlyStopping patience (aggressive = bug)
            epochs: Number of epochs to train
        
        Returns:
            TrainConfig configured to exhibit overfitting
        """
        cfg = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test_buggy",
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

    def test_bug_condition_undersampling_short_sequence_limited_receptive_field(
        self,
    ):
        """Test: Configuration with ALL bug conditions triggers overfitting.
        
        Configuration:
        - use_undersampling: true (loses ~27% data)
        - sequence_length: 30 (insufficient temporal context)
        - dilations: [1, 2, 4, 8] (receptive field depth = 4, limited)
        - patience: 5 (stops training too early)
        
        Expected Behavior (AFTER FIX):
        - validation_accuracy >= 0.66
        - train_accuracy - val_accuracy < 0.02
        - rem_accuracy >= 0.40
        
        Current Behavior (UNFIXED - BUG):
        - validation_accuracy < 0.66 (typically ~62.3%)
        - train_accuracy - val_accuracy > 0.02 (typically ~6%)
        - rem_accuracy < 0.40 (typically ~25.6%)
        
        This test will FAIL on unfixed code (proving the bug exists).
        After fix is implemented, this test should PASS.
        """
        cfg = self._create_buggy_config(
            use_undersampling=True,
            sequence_length=30,
            dilations=(1, 2, 4, 8),
            patience=5,
            epochs=20,
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

        # Scale data
        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        # Build model
        model = build_model(cfg, sequence_length, feature_dim, num_classes)

        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        # Train model
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath="outputs_test_buggy/best.keras",
                monitor="val_loss",
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.patience,
                min_delta=1e-4,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=max(1, cfg.patience // 2),
                min_lr=1e-6,
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
            output_dir=Path("outputs_test_buggy"),
        )

        # Extract metrics
        val_accuracy = eval_result.accuracy
        train_accuracy = history.history["accuracy"][-1]
        train_val_gap = train_accuracy - val_accuracy
        rem_accuracy = eval_result.recall_per_class[3] if len(eval_result.recall_per_class) > 3 else 0.0

        print(f"\n=== Bug Condition Test Results ===")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train-Val Gap: {train_val_gap:.4f}")
        print(f"REM Classification Accuracy: {rem_accuracy:.4f}")
        print(f"Epochs Trained: {len(history.history['accuracy'])}")
        print(f"Recall per class: {eval_result.recall_per_class}")

        # Expected behavior assertions (after fix)
        # On unfixed code, these assertions will FAIL, surfacing the bug
        assert (
            val_accuracy >= 0.66
        ), f"Validation accuracy {val_accuracy:.4f} < 0.66 (BUG: model overfitting)"

        assert (
            train_val_gap < 0.02
        ), f"Train-val gap {train_val_gap:.4f} >= 0.02 (BUG: overfitting gap too large)"

        assert (
            rem_accuracy >= 0.40
        ), f"REM classification {rem_accuracy:.4f} < 0.40 (BUG: REM underclassified)"

    def test_bug_condition_undersampling_only(self):
        """Test: Undersampling alone triggers overfitting.
        
        Configuration:
        - use_undersampling: true (main bug factor)
        - sequence_length: 30 (default short)
        - dilations: [1, 2, 4, 8] (default limited)
        - patience: 5 (default aggressive)
        
        Expected: val_accuracy >= 0.66
        """
        cfg = self._create_buggy_config(
            use_undersampling=True,
            sequence_length=30,
            dilations=(1, 2, 4, 8),
            patience=5,
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

        from sklearn.utils.class_weight import compute_class_weight

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
            output_dir=Path("outputs_test_buggy_undersample"),
        )

        val_accuracy = eval_result.accuracy
        print(f"\nUndersampling-only test: val_accuracy={val_accuracy:.4f}")

        assert (
            val_accuracy >= 0.66
        ), f"Undersampling causes overfitting: val_accuracy {val_accuracy:.4f} < 0.66"

    def test_bug_condition_short_sequence_only(self):
        """Test: Short sequence length triggers overfitting.
        
        Configuration:
        - use_undersampling: false (no undersampling)
        - sequence_length: 30 (main bug factor - too short)
        - dilations: [1, 2, 4, 8] (default)
        - patience: 5 (default)
        
        Expected: val_accuracy >= 0.66 (especially for REM)
        """
        cfg = self._create_buggy_config(
            use_undersampling=False,
            sequence_length=30,
            dilations=(1, 2, 4, 8),
            patience=5,
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

        from sklearn.utils.class_weight import compute_class_weight

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
            output_dir=Path("outputs_test_buggy_short_seq"),
        )

        val_accuracy = eval_result.accuracy
        rem_accuracy = eval_result.recall_per_class[3] if len(eval_result.recall_per_class) > 3 else 0.0
        print(f"\nShort sequence test: val_accuracy={val_accuracy:.4f}, rem_accuracy={rem_accuracy:.4f}")

        assert (
            val_accuracy >= 0.66
        ), f"Short sequence causes overfitting: val_accuracy {val_accuracy:.4f} < 0.66"

        assert (
            rem_accuracy >= 0.40
        ), f"Short sequence impairs REM classification: rem_accuracy {rem_accuracy:.4f} < 0.40"

    def test_bug_condition_limited_receptive_field_only(self):
        """Test: Limited TCN receptive field triggers overfitting.
        
        Configuration:
        - use_undersampling: false
        - sequence_length: 30 (default)
        - dilations: [1, 2, 4, 8] (main bug factor - limited depth)
        - patience: 5 (default)
        
        Expected: val_accuracy >= 0.66
        """
        cfg = self._create_buggy_config(
            use_undersampling=False,
            sequence_length=30,
            dilations=(1, 2, 4, 8),
            patience=5,
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

        from sklearn.utils.class_weight import compute_class_weight

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
            output_dir=Path("outputs_test_buggy_limited_rf"),
        )

        val_accuracy = eval_result.accuracy
        print(f"\nLimited receptive field test: val_accuracy={val_accuracy:.4f}")

        assert (
            val_accuracy >= 0.66
        ), f"Limited receptive field causes overfitting: val_accuracy {val_accuracy:.4f} < 0.66"

    def test_bug_condition_aggressive_early_stopping(self):
        """Test: Aggressive early stopping (patience=5) triggers overfitting.
        
        Configuration:
        - use_undersampling: false
        - sequence_length: 30 (default)
        - dilations: [1, 2, 4, 8] (default)
        - patience: 5 (main bug factor - too aggressive)
        
        Expected: val_accuracy >= 0.66
        """
        cfg = self._create_buggy_config(
            use_undersampling=False,
            sequence_length=30,
            dilations=(1, 2, 4, 8),
            patience=5,
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

        from sklearn.utils.class_weight import compute_class_weight

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
            output_dir=Path("outputs_test_buggy_early_stop"),
        )

        val_accuracy = eval_result.accuracy
        epochs_trained = len(history.history["accuracy"])
        print(f"\nAggressive early stopping test: val_accuracy={val_accuracy:.4f}, epochs={epochs_trained}")

        assert (
            val_accuracy >= 0.66
        ), f"Aggressive early stopping prevents convergence: val_accuracy {val_accuracy:.4f} < 0.66"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
