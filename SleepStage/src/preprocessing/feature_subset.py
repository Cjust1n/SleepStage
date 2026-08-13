"""feature_subset.py

Runtime feature selection utility with YAML-driven presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import json
import numpy as np


LEGACY_DEFAULT_FEATURES: tuple[str, ...] = (
    "relative_position",
    "mean_hr",
    "rmssd",
    "sd2",
    "lf",
    "hf",
    "lf_hf",
    "nn50",
    "mean_acc",
    "std_acc",
    "energy",
    "variance",
    "movement_count",
    "zero_crossing",
    "time_of_night",
)


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "temporal": ("relative_position", "time_of_night", "sin_time_of_night", "cos_time_of_night"),
    "hrv": ("mean_hr", "rmssd", "sd2", "lf", "hf", "lf_hf", "nn50", "hr_delta", "rolling_mean_hr", "rolling_std_hr", "hr_slope", "rolling_hr_range"),
    "movement": ("mean_acc", "std_acc", "energy", "movement_count", "movement_ratio", "acceleration_jerk", "zero_crossing", "variance", "rolling_mean_acc", "rolling_std_acc"),
}


VALID_FEATURE_PRESETS = {
    "all",
    "minimal",
    "hrv_only",
    "movement_only",
    "temporal_only",
    "no_hrv",
    "no_movement",
    "no_temporal",
}


@dataclass(frozen=True)
class FeatureSubsetConfig:
    processed_dir: str = "processed"
    feature_preset: str = "all"
    feature_subset: Sequence[str] | None = None


@dataclass(frozen=True)
class FeatureSelectionResult:
    mode: str
    selected_features: tuple[str, ...]
    removed_features: tuple[str, ...]


def load_feature_names(processed_dir: str | Path) -> List[str]:
    p = Path(processed_dir) / "feature_names.json"
    if not p.exists():
        raise FileNotFoundError(f"feature_names.json not found at: {p}")
    return json.loads(p.read_text())


def _validate_unique(feature_names: Sequence[str], *, context: str) -> None:
    dupes = sorted({f for f in feature_names if feature_names.count(f) > 1})
    if dupes:
        raise ValueError(f"Duplicate feature(s) in {context}: {', '.join(dupes)}")


def _normalize_preset(preset: str | None) -> str:
    preset_norm = (preset or "all").strip().lower()
    if preset_norm not in VALID_FEATURE_PRESETS:
        raise ValueError("feature_preset must be one of: " + ", ".join(sorted(VALID_FEATURE_PRESETS)))
    return preset_norm


def _validate_feature_names(feature_names: Sequence[str], expected_features: Sequence[str], *, context: str) -> None:
    feature_name_set = set(feature_names)
    missing = [f for f in expected_features if f not in feature_name_set]
    if missing:
        raise KeyError(f"Missing feature(s) for {context}: " + ", ".join(missing))


def _resolve_preset(feature_names: Sequence[str], preset: str) -> tuple[str, ...]:
    if preset == "all":
        return tuple(feature_names)
    if preset == "minimal":
        return tuple(LEGACY_DEFAULT_FEATURES)
    if preset == "hrv_only":
        return FEATURE_GROUPS["hrv"]
    if preset == "movement_only":
        return FEATURE_GROUPS["movement"]
    if preset == "temporal_only":
        return FEATURE_GROUPS["temporal"]
    if preset == "no_hrv":
        excluded = set(FEATURE_GROUPS["hrv"])
        return tuple(f for f in feature_names if f not in excluded)
    if preset == "no_movement":
        excluded = set(FEATURE_GROUPS["movement"])
        return tuple(f for f in feature_names if f not in excluded)
    if preset == "no_temporal":
        excluded = set(FEATURE_GROUPS["temporal"])
        return tuple(f for f in feature_names if f not in excluded)
    raise ValueError(f"Unhandled feature_preset: {preset}")


def resolve_feature_selection(
    feature_names: Sequence[str],
    *,
    feature_preset: str = "all",
    explicit_features: Sequence[str] | None = None,
) -> FeatureSelectionResult:
    feature_names = list(feature_names)
    _validate_unique(feature_names, context="feature_names")
    preset = _normalize_preset(feature_preset)

    if explicit_features is not None and len(explicit_features) > 0:
        selected = list(explicit_features)
        _validate_unique(selected, context="explicit_features")
        _validate_feature_names(feature_names, selected, context="explicit feature subset")
        mode = "feature_subset"
    else:
        selected = list(_resolve_preset(feature_names, preset))
        mode = preset

    _validate_feature_names(feature_names, selected, context=f"feature_preset={mode}")

    removed = [f for f in feature_names if f not in set(selected)]
    return FeatureSelectionResult(
        mode=mode,
        selected_features=tuple(selected),
        removed_features=tuple(removed),
    )


def infer_feature_selection_from_dim(feature_names: Sequence[str], target_dim: int) -> FeatureSelectionResult | None:
    feature_names = list(feature_names)
    candidates: list[FeatureSelectionResult] = []
    for preset in sorted(VALID_FEATURE_PRESETS):
        try:
            selection = resolve_feature_selection(feature_names, feature_preset=preset, explicit_features=None)
        except KeyError:
            continue
        if len(selection.selected_features) == int(target_dim):
            candidates.append(selection)

    legacy_selection = resolve_feature_selection(
        feature_names,
        feature_preset="all",
        explicit_features=LEGACY_DEFAULT_FEATURES,
    )
    if len(legacy_selection.selected_features) == int(target_dim):
        candidates.append(
            FeatureSelectionResult(
                mode="legacy_default",
                selected_features=legacy_selection.selected_features,
                removed_features=legacy_selection.removed_features,
            )
        )

    return candidates[0] if len(candidates) == 1 else None


def select_features_from_X(
    X: np.ndarray,
    *,
    feature_names: Sequence[str],
    selected_features: Sequence[str],
) -> np.ndarray:
    feature_names = list(feature_names)
    selected_features = list(selected_features)
    _validate_unique(feature_names, context="feature_names")
    _validate_unique(selected_features, context="selected_features")

    if X.ndim not in (2, 3):
        raise ValueError(f"Expected X to be 2D or 3D, got shape={X.shape}")

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [f for f in selected_features if f not in name_to_idx]
    if missing:
        raise KeyError("Missing selected feature(s) in processed/feature_names.json: " + ", ".join(missing))

    idxs = [name_to_idx[f] for f in selected_features]
    return X[:, idxs] if X.ndim == 2 else X[:, :, idxs]


def apply_runtime_feature_subset(X_seq: np.ndarray, *, cfg: FeatureSubsetConfig) -> np.ndarray:
    feature_names = load_feature_names(cfg.processed_dir)
    selection = resolve_feature_selection(
        feature_names,
        feature_preset=cfg.feature_preset,
        explicit_features=cfg.feature_subset,
    )
    return select_features_from_X(X_seq, feature_names=feature_names, selected_features=selection.selected_features)


def apply_runtime_feature_group_selection(
    X_seq: np.ndarray,
    *,
    processed_dir: str | Path,
    feature_preset: str,
) -> tuple[np.ndarray, FeatureSelectionResult]:
    feature_names = load_feature_names(processed_dir)
    selection = resolve_feature_selection(feature_names, feature_preset=feature_preset, explicit_features=None)
    return select_features_from_X(X_seq, feature_names=feature_names, selected_features=selection.selected_features), selection
