"""test_overfitting_bug_minimal.py

Minimal bug condition exploration test that verifies configuration creation
and data preparation work correctly for testing the overfitting bug.

This test verifies that the bug conditions can be properly configured and
set up for exploration without requiring full training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import (
    TrainConfig,
    create_sequences_and_split,
    _scale_sequence_data,
    build_model,
)


class TestOverfittingBugSetup:
    """Validates that bug condition configurations are properly set up."""

    def test_buggy_config_with_undersampling_created_correctly(self):
        """Verify that buggy configuration with undersampling is created."""
        cfg = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test",
            sequence_length=30,
            model_type="tcn_lstm",
            dilations=(1, 2, 4, 8),
            patience=5,
            use_undersampling=True,
            use_class_weight=True,
            seed=42,
        )

        assert cfg.sequence_length == 30
        assert cfg.use_undersampling is True
        assert cfg.dilations == (1, 2, 4, 8)
        assert cfg.patience == 5
        print(f"✓ Buggy config created: undersampling={cfg.use_undersampling}")

    def test_sequence_and_split_creation_with_buggy_config(self):
        """Test that sequences and splits can be created with bug configuration."""
        cfg = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test",
            sequence_length=30,
            model_type="tcn_lstm",
            dilations=(1, 2, 4, 8),
            patience=5,
            use_undersampling=True,
            use_class_weight=True,
            seed=42,
            val_size_subjects=1,
            test_size_subjects=1,
        )

        # This should complete without error
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

        # Verify shapes are correct
        assert len(X_train) > 0, "X_train should not be empty"
        assert X_train.ndim == 3, "X_train should be 3D (N, T, F)"
        assert X_train.shape[1] == 30, "Sequence length should be 30"
        assert X_val.ndim == 3, "X_val should be 3D"
        assert X_test.ndim == 3, "X_test should be 3D"

        print(f"✓ Sequences created successfully")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  X_test: {X_test.shape}")

    def test_data_scaling_works_with_buggy_sequences(self):
        """Test that data scaling works correctly."""
        cfg = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test",
            sequence_length=30,
            model_type="tcn_lstm",
            dilations=(1, 2, 4, 8),
            patience=5,
            use_undersampling=True,
            use_class_weight=True,
            seed=42,
            val_size_subjects=1,
            test_size_subjects=1,
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

        X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
            X_train, X_val, X_test
        )

        # Verify scaled data is reasonable
        assert not np.isnan(X_train_s).any(), "Scaled data should not contain NaNs"
        assert not np.isinf(X_train_s).any(), "Scaled data should not contain Infs"

        # Check that data is actually scaled (mean near 0, std near 1)
        mean = np.mean(X_train_s)
        std = np.std(X_train_s)
        assert abs(mean) < 0.1, f"Scaled data mean should be near 0, got {mean}"
        assert 0.8 < std < 1.2, f"Scaled data std should be near 1, got {std}"

        print(f"✓ Data scaling successful")
        print(f"  Mean: {mean:.4f}, Std: {std:.4f}")

    def test_model_builds_with_buggy_config(self):
        """Test that model can be built with bug configuration."""
        cfg = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test",
            sequence_length=30,
            model_type="tcn_lstm",
            hidden_units=64,
            filters=64,
            kernel_size=3,
            dilations=(1, 2, 4, 8),
            dropout=0.3,
            use_separable_conv=False,
            patience=5,
            use_undersampling=True,
            use_class_weight=True,
            seed=42,
            val_size_subjects=1,
            test_size_subjects=1,
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

        # Build model
        model = build_model(cfg, sequence_length, feature_dim, num_classes)

        # Verify model properties
        assert model is not None, "Model should be created"
        assert len(model.layers) > 0, "Model should have layers"
        print(f"✓ Model built successfully")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Number of layers: {len(model.layers)}")

    def test_class_weight_calculation(self):
        """Test class weight computation for buggy data."""
        cfg = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test",
            sequence_length=30,
            model_type="tcn_lstm",
            patience=5,
            use_undersampling=True,
            use_class_weight=True,
            seed=42,
            val_size_subjects=1,
            test_size_subjects=1,
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

        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        # Verify class weights are reasonable
        assert len(class_weight) > 0, "Class weights should be computed"
        assert all(w > 0 for w in class_weight.values()), "All weights should be positive"

        print(f"✓ Class weights computed successfully")
        for c, w in sorted(class_weight.items()):
            print(f"  Class {c}: {w:.4f}")

    def test_undersampling_reduces_data(self):
        """Verify that undersampling actually reduces training data."""
        # Config without undersampling
        cfg_no_undersample = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test",
            sequence_length=30,
            model_type="tcn_lstm",
            patience=5,
            use_undersampling=False,
            use_class_weight=True,
            seed=42,
            val_size_subjects=1,
            test_size_subjects=1,
        )

        # Config with undersampling
        cfg_undersample = TrainConfig(
            processed_dir="processed",
            outputs_dir="outputs_test",
            sequence_length=30,
            model_type="tcn_lstm",
            patience=5,
            use_undersampling=True,
            use_class_weight=True,
            seed=42,
            val_size_subjects=1,
            test_size_subjects=1,
        )

        # Get data for both configs
        (
            X_train_no_us,
            y_train_no_us,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = create_sequences_and_split(cfg_no_undersample)

        (
            X_train_us,
            y_train_us,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = create_sequences_and_split(cfg_undersample)

        # Verify undersampling reduces data
        assert len(X_train_us) < len(X_train_no_us), (
            f"Undersampling should reduce training data: "
            f"without={len(X_train_no_us)}, with={len(X_train_us)}"
        )

        reduction_ratio = len(X_train_us) / len(X_train_no_us)
        print(f"✓ Undersampling confirmed to reduce data")
        print(f"  Without undersampling: {len(X_train_no_us)} samples")
        print(f"  With undersampling: {len(X_train_us)} samples")
        print(f"  Reduction ratio: {reduction_ratio:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
