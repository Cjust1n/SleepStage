"""train.py

End-to-end training pipeline:
Raw processed arrays -> create sequences -> subject-wise split -> scaling -> train RNN/LSTM -> evaluate.

command :python -m src.training.train --config configs/baseline_undersample.yaml
Then optional INT8 TFLite export+quantization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

from pathlib import Path
from typing import Any, Dict, Tuple

from sklearn.utils.class_weight import compute_class_weight


import numpy as np
import yaml
import tensorflow as tf


from src.datasets.dataset import load_processed_dataset, infer_num_classes
from src.preprocessing.scaler import Scaler
from src.preprocessing.sequence_builder import SequenceCreator
from src.preprocessing.feature_subset import (
    FeatureSubsetConfig,
    load_feature_names,
    infer_feature_selection_from_dim,
    resolve_feature_selection,
    select_features_from_X,
)

from src.preprocessing.split_subject import SubjectWiseSplitter
from src.models.rnn import build_gru_model
#from src.models.lstm import build_lstm_model
from src.models.gru_real import build_gru_real_model
from src.models.tcn_lstm import build_tcn_lstm_model
from src.models.tcn import build_tcn_model
from src.models.cnn import build_cnn_model
from src.models.simple_rnn import build_simple_rnn_model
from src.models.multi_tcn import build_multi_tcn_model
from src.training.evaluate import evaluate_model, save_eval_outputs


@dataclass
class TrainConfig:
    processed_dir: str = "processed_15_all"
    outputs_dir: str = "outputs"

    # Runtime feature subsetting (select only specific feature columns)
    # based on processed/feature_names.json.
    # Default: OFF (use all features).
    use_feature_subset: bool = False
    feature_group_mode: str = "all"
    feature_preset: str = "all"
    feature_subset: FeatureSubsetConfig = field(default_factory=FeatureSubsetConfig)




# PSO weight scaling (class_weight) search
    use_pso_weight: bool = False
    pso_particles: int = 10
    pso_max_iters: int = 30
    pso_target_accuracy: float = 1.0
    pso_epochs_search: int = 5


    sequence_length: int = 30
    step: int = 1
    require_contiguous: bool = True

    model_type: str = "gru"  # gru|lstm|tcn_lstm
    hidden_units: int = 64
    dropout: float = 0.3
    bidirectional: bool = False
    filters: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    use_separable_conv: bool = False
    # Weight regularization (L2) strength applied to conv/dense kernels.
    l2_weight_decay: float = 1e-4

    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    patience: int = 5

    # Subject split
    train_subjects: list[str] | None = None
    val_subjects: list[str] | None = None
    test_subjects: list[str] | None = None
    test_size_subjects: int = 1
    val_size_subjects: int = 1

    # Seed
    seed: int = 42

    use_class_weight: bool = True

    # Undersampling (optional)
    # Jika aktif: hanya split training yang di-undersample agar jumlah sample per kelas sama
    # (diambil sebanyak kelas paling sedikit).
    use_undersampling: bool = False
    undersample_seed: int | None = None

    # Catatan: undersampling hanya memodifikasi X_train/y_train; X_val/X_test tetap asli.

    # Noise-based oversampling (optional)
    # Jika aktif (use_undersampling=false, use_noise_oversampling=true):
    # Setelah subject-wise split, X_train/y_train di-augment dengan duplicate + noise
    # Gaussian per-feature (std = noise_oversample_std * feature_std) sehingga
    # jumlah sample per kelas sama dengan kelas mayoritas.
    # Hanya train yang di-augment; val/test tetap asli.
    # Dilakukan SEBELUM scaling agar scaler mempelajari distribusi yang seimbang.
    use_noise_oversampling: bool = True
    noise_oversample_std: float = 0.02

    # If true: do not call SequenceCreator.build_sequences().
    # Use ds.X/ds.y/ds.metadata directly from processed/ as prebuilt samples.
    # Useful when preprocessing/sequence building is too slow to re-run.
    use_prebuilt_sequences: bool = False


    # Quantization (optional)
    use_reduce_lr_on_plateau: bool = True

    quantize_int8: bool = True
    representative_steps: int = 200




def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def _normalize_feature_subset_config(cfg: TrainConfig) -> FeatureSubsetConfig:
    """Return a FeatureSubsetConfig whose processed_dir matches TrainConfig."""
    feature_subset = cfg.feature_subset
    if isinstance(feature_subset, dict):
        feature_subset = FeatureSubsetConfig(**feature_subset)

    return FeatureSubsetConfig(
        processed_dir=cfg.processed_dir,
        feature_preset=str(getattr(cfg, "feature_preset", getattr(cfg, "feature_group_mode", getattr(feature_subset, "feature_preset", "all")))),
        feature_subset=tuple(feature_subset.feature_subset) if feature_subset.feature_subset is not None else None,
    )


def _resolve_feature_selection_for_training(
    cfg: TrainConfig,
    feature_names: list[str],
):
    """Resolve feature selection for training without hardcoding indices."""

    feature_group_mode = str(getattr(cfg, "feature_preset", getattr(cfg, "feature_group_mode", "all"))).strip().lower()

    if feature_group_mode != "all":
        return resolve_feature_selection(
            feature_names,
            feature_preset=feature_group_mode,
            explicit_features=None,
        )

    if cfg.use_feature_subset:
        subset_cfg = _normalize_feature_subset_config(cfg)
        return resolve_feature_selection(
            feature_names,
            feature_preset=subset_cfg.feature_preset,
            explicit_features=tuple(subset_cfg.feature_subset) if subset_cfg.feature_subset is not None else None,
        )

    return resolve_feature_selection(
        feature_names,
        feature_preset="all",
        explicit_features=None,
    )


def _print_feature_selection(selection) -> None:
    print("=" * 36)
    print("Feature Selection")
    print("=" * 36)
    print(f"Mode:\n{selection.mode}")
    print("\nSelected Features:")
    for name in selection.selected_features:
        print(f"- {name}")
    print("\nRemoved Features:")
    for name in selection.removed_features:
        print(f"- {name}")
    print("\nNumber of Features:")
    print(f"{len(selection.selected_features)}")


def _resolve_subject_splits(metadata_seq, cfg: TrainConfig):
    # metadata_seq has Subject column.
    subjects = sorted(metadata_seq["Subject"].astype(str).unique().tolist())

    if cfg.train_subjects and cfg.val_subjects and cfg.test_subjects:
        return cfg.train_subjects, cfg.val_subjects, cfg.test_subjects

    # Use (almost) all subjects while keeping a dedicated val and test.
    # Deterministic: take first val_size_subjects for val, next test_size_subjects for test.
    # Train gets the remaining subjects (no additional truncation at sequence level).
    val_subjects = subjects[: cfg.val_size_subjects]
    test_subjects = subjects[
        cfg.val_size_subjects : cfg.val_size_subjects + cfg.test_size_subjects
    ]
    train_subjects = [
        s
        for s in subjects
        if s not in set(val_subjects).union(set(test_subjects))
    ]

    return train_subjects, val_subjects, test_subjects


def _undersample_balanced_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata,
    seed: int,
    split_name: str = "train",
):
    """Undersample a split so each class has the same number of samples.

    This keeps validation/test untouched and only reduces the training split.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    if len(classes) <= 1:
        print(f"[Undersampling] {split_name}: only one class present; skipping.")
        return X, y, metadata

    min_count = int(counts.min())
    keep_indices = []
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        if idx_c.shape[0] <= min_count:
            keep_idx_c = idx_c
        else:
            keep_idx_c = rng.choice(idx_c, size=min_count, replace=False)
        keep_indices.append(keep_idx_c)

    keep_indices = np.concatenate(keep_indices, axis=0)
    rng.shuffle(keep_indices)

    X_new = X[keep_indices]
    y_new = y[keep_indices]

    try:
        metadata_new = metadata.iloc[keep_indices]
    except AttributeError:
        metadata_new = metadata[keep_indices]

    before = {int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())}
    after_classes, after_counts = np.unique(y_new, return_counts=True)
    after = {int(c): int(n) for c, n in zip(after_classes.tolist(), after_counts.tolist())}
    print(
        f"[Undersampling] {split_name}: min_count={min_count} | "
        f"before={before} | after={after} | total={X.shape[0]} -> {X_new.shape[0]}"
    )

    return X_new, y_new, metadata_new



def create_sequences_and_split(cfg: TrainConfig):
    ds = load_processed_dataset(cfg.processed_dir)

    feature_names = load_feature_names(cfg.processed_dir)
    feature_selection = _resolve_feature_selection_for_training(cfg, feature_names)
    _print_feature_selection(feature_selection)
    setattr(cfg, "_feature_selection", feature_selection)
    X_selected = select_features_from_X(
        ds.X,
        feature_names=feature_names,
        selected_features=feature_selection.selected_features,
    )

    # Option: use prebuilt samples directly from processed/X.npy and processed/y.npy
    # (skips SequenceCreator to avoid re-running expensive preprocessing/sequence building).
    if getattr(cfg, "use_prebuilt_sequences", False):
        X_seq = np.asarray(X_selected)
        y_seq = np.asarray(ds.y)
        metadata_seq = ds.metadata

        # Ensure shapes expected by the training code: X is (N,T,F).
        # If X is already 2D, it means it represents a single timestep/window (T=1).
        # In that case, training will not be temporal anymore; fail fast when
        # the config expects temporal context.
        if X_seq.ndim == 2:
            if int(getattr(cfg, "sequence_length", 1)) != 1:
                raise ValueError(
                    "use_prebuilt_sequences=true but processed/X.npy is 2D (T=1). "
                    "TCN/GRU/LSTM require time dimension > 1. "
                    "Fix: set use_prebuilt_sequences=false (so SequenceCreator builds sequences) "
                    "or set sequence_length=1 to intentionally train non-temporal."
                )
            X_seq = X_seq[:, None, :]
        elif X_seq.ndim != 3:
            raise ValueError(
                f"Expected X to have ndim 2 or 3 in prebuilt mode, got {X_seq.ndim}"
            )


        y_seq = y_seq.astype(np.int64, copy=False)

    else:
        seq_creator = SequenceCreator(
            sequence_length=cfg.sequence_length,
            step=cfg.step,
            require_contiguous=cfg.require_contiguous,
        )

        X_seq, y_seq, metadata_seq = seq_creator.build_sequences(
            X_selected, ds.y, ds.metadata
        )


    if len(X_seq) == 0:
        raise RuntimeError(
            "No sequences were created. Check sequence_length/contiguity assumptions."
        )
    # Normalize features using train split only.
    train_subjects, val_subjects, test_subjects = _resolve_subject_splits(metadata_seq, cfg)

    splitter = SubjectWiseSplitter(
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    
    (X_train, y_train, train_metadata), (X_val, y_val, val_metadata), (X_test, y_test, test_metadata) = splitter.split(
        X_seq, y_seq, metadata_seq, return_metadata=True
    )

    # Save the original training distribution for class_weight computation.
    # This lets us combine undersampling with weighted loss: the training set is reduced,
    # but the loss still reflects the pre-undersample imbalance.
    setattr(cfg, "_class_weight_y_train_source", np.asarray(y_train).copy())

    if cfg.use_undersampling:
        X_train, y_train, train_metadata = _undersample_balanced_split(
            X_train,
            y_train,
            train_metadata,
            seed=cfg.undersample_seed if cfg.undersample_seed is not None else cfg.seed,
            split_name="train",
        )

    print("Sequence length:", X_train.shape[1])
    print("Unique y_train:", np.unique(y_train))
    print("Train counts :", np.bincount(y_train))
    print("Val counts   :", np.bincount(y_val))
    print("Test counts  :", np.bincount(y_test))
    print("Counts:", np.bincount(y_train))

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_metadata,
        (train_subjects, val_subjects, test_subjects),
    )


def _noise_oversample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_std: float,
    seed: int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augment X_train/y_train with noise-based oversampling to balance classes.

    Untuk setiap kelas minoritas, sample di-duplicate (dengan replacement)
    dan ditambahkan Gaussian noise per-feature:
        noise ~ N(0, noise_std * feature_std)
    dimana feature_std adalah std dari masing-masing fitur di X_train.

    Total sample per kelas akan sama dengan kelas mayoritas.
    Hanya X_train/y_train yang di-augment (bukan val/test).
    Dilakukan SEBELUM scaling agar scaler mempelajari distribusi seimbang.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) <= 1:
        print("[NoiseOversample] Only one class present; skipping.")
        return X_train, y_train

    if rng is None:
        rng = np.random.default_rng(seed)

    majority_count = int(counts.max())
    # Cek apakah semua kelas sudah seimbang
    if int(counts.min()) == majority_count:
        print("[NoiseOversample] Classes already balanced; skipping.")
        return X_train, y_train

    # Compute per-feature std across entire training set (N, T, F) -> (1, 1, F)
    feature_std = X_train.std(axis=(0, 1), keepdims=True)  # shape (1, 1, F)
    # Avoid division by zero for constant features
    feature_std = np.where(feature_std < 1e-12, 1e-12, feature_std)

    augmented_X_list = [X_train]
    augmented_y_list = [y_train]

    for c in classes:
        idx_c = np.flatnonzero(y_train == c)
        n_c = int(idx_c.shape[0])
        n_needed = majority_count - n_c
        if n_needed <= 0:
            continue

        # Duplicate existing samples with replacement
        chosen_idx = rng.choice(idx_c, size=n_needed, replace=True)
        dup_X = X_train[chosen_idx].copy()

        # Add Gaussian noise: scale per-feature
        noise = rng.normal(
            loc=0.0,
            scale=noise_std * feature_std,  # broadcasts over (n_needed, T, F)
            size=dup_X.shape,
        ).astype(dup_X.dtype)
        dup_X += noise

        dup_y = y_train[chosen_idx].copy()

        augmented_X_list.append(dup_X)
        augmented_y_list.append(dup_y)

    X_new = np.concatenate(augmented_X_list, axis=0)
    y_new = np.concatenate(augmented_y_list, axis=0)

    # Shuffle
    shuffle_idx = rng.permutation(X_new.shape[0])
    X_new = X_new[shuffle_idx]
    y_new = y_new[shuffle_idx]

    print(
        f"[NoiseOversample] noise_std={noise_std:.3f} | "
        f"Before: {dict(zip(classes, counts.tolist()))} | "
        f"After: {dict(zip(*np.unique(y_new, return_counts=True)))} | "
        f"Total: {X_train.shape[0]} -> {X_new.shape[0]}"
    )

    return X_new, y_new


def _scale_sequence_data(X_train, X_val, X_test):
    # X_* shape: (N, T, F)
    n_train, t, f = X_train.shape
    scaler = Scaler()

    X_train_2d = X_train.reshape(-1, f)
    scaler.fit(X_train_2d)

    X_val_2d = X_val.reshape(-1, f)
    X_test_2d = X_test.reshape(-1, f)

    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, t, f)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape[0], t, f)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape[0], t, f)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_model(
    cfg: TrainConfig,
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    feature_names: list[str] | None = None,
):
    if cfg.model_type.lower() == "gru":
        model = build_gru_model(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            l2_weight_decay=cfg.l2_weight_decay,
        )
    elif cfg.model_type.lower() == "tcn":
        model = build_tcn_model(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            l2_weight_decay=cfg.l2_weight_decay,
        )

    elif cfg.model_type.lower() == "gru_real":
        model = build_gru_real_model(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            l2_weight_decay=cfg.l2_weight_decay,
        )

    elif cfg.model_type.lower() == "tcn_lstm":
        model = build_tcn_lstm_model(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_units=cfg.hidden_units,
            filters=cfg.filters,
            kernel_size=cfg.kernel_size,
            dilations=cfg.dilations,
            dropout=cfg.dropout,
            use_separable_conv=cfg.use_separable_conv,
        )

    elif cfg.model_type.lower() == "cnn":
        model = build_cnn_model(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes,
            filters=cfg.filters,
            kernel_size=cfg.kernel_size,
            dilations=cfg.dilations,
            dropout=cfg.dropout,
            l2_weight_decay=cfg.l2_weight_decay,
        )

    elif cfg.model_type.lower() == "simple_rnn":
        model = build_simple_rnn_model(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            l2_weight_decay=cfg.l2_weight_decay,
        )

    elif cfg.model_type.lower() == "multi_tcn":
        if feature_names is None:
            raise ValueError("feature_names must be provided for model_type='multi_tcn'")
        model = build_multi_tcn_model(
            sequence_length=sequence_length,
            feature_dim=feature_dim,
            num_classes=num_classes,
            feature_names=feature_names,
            dropout=cfg.dropout,
            kernel_size=cfg.kernel_size,
            branch_filters={
                "movement": 48,
                "hrv": 48,
                "hr": 32,
                "temporal": 16,
            },
        )

    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    # Override LR if configured.
    # Jika PSO mengaktifkan pencarian class_weight, learning rate diturunkan agar pembelajaran
    # tidak terlalu agresif di epoch awal (mitigasi overfit/early Scollapse).
    effective_lr = cfg.learning_rate
    if getattr(cfg, "use_pso_weight", False):
        # konservatif: 1e-3 -> 1e-4
        effective_lr = min(effective_lr, 1e-4)

    model.optimizer.learning_rate.assign(effective_lr)

    return model


def _make_class_weight_balanced(
    y_train: np.ndarray,
    multipliers_by_class: np.ndarray | None = None,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> Dict[int, float]:
    """Compute class_weight='balanced' and optionally scale it by PSO-searchable multipliers.

    This function intentionally does NOT apply any exponent/power transform.
    The PSO should optimize multipliers only.

    multipliers_by_class is applied in the order of np.unique(y_train) class ids.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    if multipliers_by_class is None:
        scaled = weights
    else:
        multipliers_by_class = np.asarray(multipliers_by_class, dtype=np.float64)
        if multipliers_by_class.shape[0] != weights.shape[0]:
            raise ValueError(
                f"multipliers_by_class length mismatch: got {multipliers_by_class.shape[0]} but need {weights.shape[0]}"
            )
        scaled = weights * multipliers_by_class

    scaled = np.clip(scaled, clip_min, clip_max)
    return {int(c): float(w) for c, w in zip(classes, scaled)}



def _objective_score_from_weights(
    cfg: TrainConfig,
    multipliers_by_class: np.ndarray,
) -> float:

    """Objective for PSO.

    Use per-class weighted recall (to reflect class distribution / test size)
    + Cohen's kappa.
    """

    search_cfg = cfg
    setattr(search_cfg, "epochs", cfg.pso_epochs_search)
    # PSO should optimize multipliers only.
    setattr(
        search_cfg,
        "_pso_multipliers_by_class",
        np.asarray(multipliers_by_class, dtype=np.float64),
    )


    payload = train_one(search_cfg)

    # evaluate_model exports:
    # - recall_per_class: list[float]
    # - support_per_class: list[int]
    # - cohen_kappa: float
    recalls = np.asarray(payload.get("recall_per_class", []), dtype=np.float64)
    supports = np.asarray(payload.get("support_per_class", []), dtype=np.float64)
    kappa = float(payload.get("cohen_kappa", 0.0))

    # train_one attaches history and includes val losses.
    history = payload.get("history", {}) or {}
    best_val_loss = float(history.get("best_val_loss", history.get("final_val_loss", np.inf)))

    # If evaluation dict missing expected keys, fall back to accuracy.
    if recalls.size == 0 or supports.size != recalls.size:

        val_acc_proxy = float(payload.get("accuracy", 0.0))
        acc_weight = float(getattr(cfg, "pso_accuracy_weight", 0.5))
        kappa_weight = float(getattr(cfg, "pso_kappa_weight", 0.5))
        # val_loss term (lower is better). Use reciprocal to keep scale similar.
        val_loss_weight = float(getattr(cfg, "pso_val_loss_weight", 0.1))
        val_loss_term = 1.0 / (best_val_loss + 1e-12)
        return acc_weight * val_acc_proxy + kappa_weight * kappa + val_loss_weight * val_loss_term

    # Weighted per-class recall; weights depend on testcase amount per class.
    w = supports / (supports.sum() + 1e-12)
    weighted_recall = float(np.sum(w * recalls))

    # val_accuracy: tidak ada dalam payload test evaluation; gunakan accuracy sebagai proksi.
    val_accuracy_proxy = float(payload.get("accuracy", 0.0))

    # val_loss: dari history yang dikembalikan train_one
    # Gunakan bobot langsung (higher val_loss hurts) dengan normalisasi yang stabil.
    # score = ... - w * (normalized_val_loss)
    val_loss_weight = float(getattr(cfg, "pso_val_loss_weight", 0.1))
    # normalisasi sederhana: bagi dengan (1+best_val_loss) agar tetap bounded
    val_loss_penalty = best_val_loss / (1.0 + best_val_loss + 1e-12)


    # Combine terms with configurable weights.
    kappa_weight = float(getattr(cfg, "pso_kappa_weight", 0.5))
    acc_weight = float(getattr(cfg, "pso_accuracy_weight", 0.5))
    recall_weight = float(getattr(cfg, "pso_recall_weight", 1.0))

    return (
        recall_weight * weighted_recall
        + acc_weight * val_accuracy_proxy
        + kappa_weight * kappa
        # Kurangi pengaruh val_loss agar tidak selalu memilih epoch 1.
        # (val_loss term sudah kita bentuk sebagai penalty.)
        - (val_loss_weight * 0.25) * val_loss_penalty



    )





def train_one(cfg: TrainConfig) -> Dict[str, Any]:
    _set_seed(cfg.seed)




    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_metadata,
        subjects,
    ) = create_sequences_and_split(cfg)

    if y_train.size == 0:
        raise RuntimeError(
            "y_train kosong (train set tidak punya data). "
            "Cek debug log dari create_sequences_and_split (undersampling & subject-wise split) "
            "serta pastikan jumlah subject/sequence_length cukup."
        )

    num_classes = int(np.unique(y_train).max() + 1)
    sequence_length = X_train.shape[1]
    feature_dim = X_train.shape[2]

    # Noise-based oversampling (only train, before scaling).
    if (not cfg.use_undersampling) and cfg.use_noise_oversampling:
        noise_rng = np.random.default_rng(cfg.seed)
        X_train, y_train = _noise_oversample_train(
            X_train,
            y_train,
            noise_std=cfg.noise_oversample_std,
            seed=cfg.seed,
            rng=noise_rng,
        )
        print("Train counts after noise oversample:", np.bincount(y_train))

    X_train_s, X_val_s, X_test_s, scaler = _scale_sequence_data(
        X_train, X_val, X_test
    )

    # Sanity checks for NaN/Inf to locate training collapse.
    def _check(name: str, arr: np.ndarray) -> None:
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        if has_nan or has_inf:
            print(f"[SANITY] {name} contains nan={has_nan} inf={has_inf}. shape={arr.shape}")
            # Count per-feature NaNs to help localize.
            nan_counts = np.isnan(arr).sum(axis=(0, 1)) if arr.ndim >= 3 else np.isnan(arr).sum(axis=0)
            inf_counts = np.isinf(arr).sum(axis=(0, 1)) if arr.ndim >= 3 else np.isinf(arr).sum(axis=0)
            print(f"[SANITY] {name} nan_counts_per_feature={nan_counts}")
            print(f"[SANITY] {name} inf_counts_per_feature={inf_counts}")
            raise RuntimeError(f"NaN/Inf detected in {name}")

    _check("X_train_s", X_train_s)
    _check("X_val_s", X_val_s)
    _check("X_test_s", X_test_s)


    selected_feature_names = None
    if getattr(cfg, "_feature_selection", None) is not None:
        selected_feature_names = list(cfg._feature_selection.selected_features)

    model = build_model(
        cfg,
        sequence_length,
        feature_dim,
        num_classes,
        feature_names=selected_feature_names,
    )

    class_weight = None
    if cfg.use_class_weight:
        class_weight_source = np.asarray(
            getattr(cfg, "_class_weight_y_train_source", y_train)
        )
        classes = np.unique(class_weight_source)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=class_weight_source,
        )


        print("\n=== Balanced class weights (sklearn) ===")
        for c, w in zip(classes, weights):
            print(f"Class {c}: {w:.4f}")

        # Multiplier scaling is controlled by PSO (if enabled).
        # Keep class_weight 'balanced' as the baseline and only apply multipliers.


        # Default multipliers per-class in class-id sorted order.
        default_multipliers = np.array([1, 1, 1, 1], dtype=np.float64)

        # Multipliers are per-class in the order of np.unique(y_train).
        # Ensure length matches number of present classes.
        multipliers = np.asarray(
            getattr(cfg, "_pso_multipliers_by_class", default_multipliers), dtype=np.float64
        )
        classes_present = np.unique(y_train)
        n_present = int(len(classes_present))
        if multipliers.shape[0] != n_present:
            # If defaults/P0 were built for 5 classes but we have 4, slice.
            # If we have more than 5, fall back by repeating then trimming.
            if multipliers.shape[0] > n_present:
                multipliers = multipliers[:n_present]
            else:
                reps = int(np.ceil(n_present / multipliers.shape[0]))
                multipliers = np.tile(multipliers, reps)[:n_present]

        # Compute class weights and clip for stability.
        class_weight = _make_class_weight_balanced(
            y_train=class_weight_source,
            multipliers_by_class=multipliers,
            clip_min=0.70,
            clip_max=1.1
        )

        print("\n=== Final class_weight used for training ===")
        for c in sorted(class_weight.keys()):
            print(f"Class {c}: {class_weight[c]:.4f}")






    outputs_dir = Path(cfg.outputs_dir)
    ckpt_dir = outputs_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        # Stop when val_loss stops improving so we keep the best generalizing
        # epoch instead of the last overfit one.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # ReduceLROnPlateau is most useful when the base LR is still high enough
    # that smaller steps can unlock a better minimum. If the effective LR is
    # already extremely small, the scheduler rarely helps and can be skipped.
    # Can be disabled via config: use_reduce_lr_on_plateau: false
    if cfg.use_reduce_lr_on_plateau and model.optimizer.learning_rate.numpy() > 3e-5:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=max(1, cfg.patience // 2),
                min_lr=1e-6,
                verbose=1,
            )
        )



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


    # Inference/eval
    # Evaluate with the best validation checkpoint to keep notebook metrics consistent
    # with the epoch metrics observed during training.
    best_ckpt_path = ckpt_dir / "best.keras"
    if best_ckpt_path.exists():
        model_for_eval = tf.keras.models.load_model(best_ckpt_path)
    else:
        model_for_eval = model

    eval_result = evaluate_model(model_for_eval, X_test_s, y_test)
    # ============================================================
    # Save prediction arrays for visualization
    # ============================================================

    logits = model_for_eval.predict(X_test_s, verbose=0)

    y_pred = np.argmax(logits, axis=1)

    if logits.shape[1] > 1:
        y_prob = tf.nn.softmax(logits, axis=1).numpy()
    else:
        y_prob = logits

    viz_dir = outputs_dir / "visualization"
    viz_dir.mkdir(exist_ok=True)

    np.save(viz_dir / "y_true.npy", y_test)
    np.save(viz_dir / "y_pred.npy", y_pred)
    np.save(viz_dir / "y_prob.npy", y_prob)

    # sequence index
    np.save(
        viz_dir / "sequence_index.npy",
        np.arange(len(y_test))
    )

    test_metadata.to_csv(
        viz_dir / "test_metadata.csv",
        index=False
    )
    payload = save_eval_outputs(
        eval_result,
        outputs_dir=outputs_dir,
        tag=f"{cfg.model_type}_seq{cfg.sequence_length}",
    )


    scaler_path = outputs_dir / "scaler.json"
    scaler.save(scaler_path)

    # Save model
    keras_path = outputs_dir / "saved_model.keras"
    saved_model_dir = outputs_dir / "saved_model"

    # Save native Keras model
    model.save(keras_path)

    # Export TensorFlow SavedModel
    model.export(saved_model_dir)
    print(f"Keras model  : {keras_path}")
    print(f"SavedModel   : {saved_model_dir}")
    
    # Summary (keep for backward compatibility)
    payload["history"] = {
        "best_val_loss": float(min(history.history.get("val_loss", [np.nan]))),
        "final_train_loss": float(history.history.get("loss", [np.nan])[-1]),
        "final_val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
    }

    # Full per-epoch history for visualization
    # Example keys: loss, val_loss, accuracy, val_accuracy
    payload["history_per_epoch"] = {
        k: [float(x) for x in v]
        for k, v in (history.history or {}).items()
        if isinstance(v, (list, tuple))
    }

    payload["class_weight"] = class_weight
    feature_selection = getattr(cfg, "_feature_selection", None)
    if feature_selection is not None:
        payload["feature_selection"] = {
            "mode": feature_selection.mode,
            "selected_features": list(feature_selection.selected_features),
            "removed_features": list(feature_selection.removed_features),
            "n_selected_features": len(feature_selection.selected_features),
        }


    payload["subjects"] = {
        "train": subjects[0],
        "val": subjects[1],
        "test": subjects[2],
    }

    # Quantize (optional)
    if cfg.quantize_int8:

        from src.training.quantize_tflite import (
            quantize_savedmodel_to_int8,
        )

        tflite_path = (
            outputs_dir
            / "checkpoints"
            / f"{cfg.model_type}_int8.tflite"
        )

        quantize_savedmodel_to_int8(
            saved_model_dir=saved_model_dir,
            tflite_path=tflite_path,
            representative_data=X_train_s,
            representative_steps=cfg.representative_steps,
        )

        payload["tflite_int8_path"] = str(tflite_path)

    return payload


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    cfg = TrainConfig(**data)
    if isinstance(cfg.feature_subset, dict):
        cfg.feature_subset = FeatureSubsetConfig(**cfg.feature_subset)
    return cfg


def _pso_optimize_class_weight(cfg: TrainConfig) -> Tuple[np.ndarray, float]:
    """Return (best_multipliers, best_score)."""

    rng = np.random.default_rng(cfg.seed)

    # Determine number of classes from y_train by doing one quick split.
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
    print("Sequence length:", X_train.shape[1])
    print("Unique y_train:", np.unique(y_train))
    print("Train counts :", np.bincount(y_train))
    y_train_classes = np.unique(y_train)
    n_classes = int(len(y_train_classes))

    # Search space: per-class multipliers.
    # Base weights are already 'balanced', so keep multipliers tight for stability.
    m_min, m_max = 0.8, 1.2


    # PSO hyperparameters (small for stability)
    # PSO params (lebih konservatif supaya tidak langsung overfit)
    w_inertia = 0.35
    c1 = 1.0
    c2 = 1.0


    dim = n_classes  # multipliers only


    pop = int(cfg.pso_particles)
    max_iters = int(cfg.pso_max_iters)

    # Initialize swarm
    positions = np.zeros((pop, dim), dtype=np.float64)
    velocities = np.zeros((pop, dim), dtype=np.float64)

    # Initialize multipliers only.
    positions[:, :] = rng.uniform(m_min, m_max, size=(pop, n_classes))


    # Personal best
    pbest_pos = positions.copy()
    pbest_score = np.full((pop,), -np.inf, dtype=np.float64)

    # Global best
    gbest_pos = positions[0].copy()
    gbest_score = -np.inf

    # Evaluate function needs access to cfg; we'll call train_one via cfg overrides.
    for it in range(max_iters):
        for i in range(pop):
            multipliers = positions[i, :].copy()

            score = _objective_score_from_weights(
                cfg=cfg,
                multipliers_by_class=multipliers,
            )



            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest_pos[i] = positions[i].copy()

            if score > gbest_score:
                gbest_score = score
                gbest_pos = positions[i].copy()

        best_multipliers = gbest_pos.copy()
        print(f"[PSO] iter={it+1}/{max_iters} best_accuracy={gbest_score:.4f}")


        if gbest_score >= cfg.pso_target_accuracy:
            break

        # Update velocities/positions
        r1 = rng.random((pop, dim))
        r2 = rng.random((pop, dim))

        velocities = (
            w_inertia * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )

        positions = positions + velocities

        # Clamp to bounds
        positions[:, 0] = np.clip(positions[:, 0], alpha_min, alpha_max)
        positions[:, 1:] = np.clip(positions[:, 1:], m_min, m_max)

    return gbest_pos.copy(), float(gbest_score)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.use_pso_weight:
        best_multipliers, best_acc = _pso_optimize_class_weight(cfg)
        # keep exponent disabled/fixed; PSO now optimizes multipliers only.
        setattr(cfg, "_pso_multipliers_by_class", best_multipliers)
        # keep original cfg.epochs

    else:
        best_alpha, best_multipliers, best_acc = None, None, None

    results = train_one(cfg)
    results["pso_best"] = {
        "enabled": bool(cfg.use_pso_weight),
        "best_accuracy": best_acc,
        "best_exponent_alpha": None,

        "best_multipliers_by_class": best_multipliers.tolist() if best_multipliers is not None else None,
        "pso_particles": cfg.pso_particles,
        "pso_max_iters": cfg.pso_max_iters,
        "pso_epochs_search": cfg.pso_epochs_search,
        "pso_target_accuracy": cfg.pso_target_accuracy,
    }

    out_path = Path(cfg.outputs_dir) / "run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure valid JSON (history_per_epoch may contain lists)
    import json
    out_path.write_text(json.dumps(results, indent=2))

    feature_selection = results.get("feature_selection")
    if feature_selection is not None:
        feature_selection_path = Path(cfg.outputs_dir) / "feature_selection.json"
        feature_selection_path.write_text(json.dumps(feature_selection, indent=2))

    print("Run finished. Outputs in:", cfg.outputs_dir)




if __name__ == "__main__":
    main()
