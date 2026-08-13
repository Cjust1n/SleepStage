"""Simple feature selection for SleepStage.

Runs two filters:
1) Random Forest feature importance (MDI via .feature_importances_):
   - train a short RandomForestClassifier
   - drop features whose importance is near zero

2) Correlation redundancy filter (Pearson):
   - if two features are highly correlated (default >= 0.99),
     drop one (the one with smaller RF importance)

Input assumptions:
- Uses processed arrays from `processed/` created by build_all_processed_dataset.py
  containing:
    - X.npy or X_normalized.npy : shape (n_samples, n_features)
    - y.npy : shape (n_samples,)
    - metadata.csv (optional but expected for debugging)
    - processed/feature_names.json : list[str] length n_features

Note:
- This script operates on the *epoch-level* feature matrix X (not on sequences).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


@dataclass
class SelectionConfig:
    processed_dir: str
    feature_names_path: str | None = None

    # Subject split for leakage-safe feature selection.
    # If explicit subject lists are not provided, the script falls back to a
    # deterministic subject-wise split using val/test subject counts.
    train_subjects: list[str] | None = None
    val_subjects: list[str] | None = None
    test_subjects: list[str] | None = None
    val_size_subjects: int = 1
    test_size_subjects: int = 1

    # RF
    rf_n_estimators: int = 250
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 2
    rf_random_state: int = 42
    rf_importance_floor: float = 1e-6

    # Correlation
    corr_method: str = "spearman"  # pearson|spearman
    corr_threshold: float = 0.90

    # Feature exclude (drop before RF/correlation)
    exclude_features: list[str] | None = None

    # Output
    output_dir: str = "feature_selection_outputs"


def _normalize_feature_name(s: str) -> str:
    # Make matching robust to accidental whitespace/case changes.
    return "_".join(s.strip().lower().split())



def _load_feature_names(processed_dir: Path, explicit: str | None) -> List[str]:
    if explicit is not None:
        p = Path(explicit)
    else:
        p = processed_dir / "feature_names.json"
    if not p.exists():
        raise FileNotFoundError(f"feature_names.json not found at: {p}")
    return json.loads(p.read_text())


def _load_xy(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    x_candidates = [processed_dir / "X_normalized.npy", processed_dir / "X.npy"]
    x = None
    for cand in x_candidates:
        if cand.exists():
            x = np.load(cand, allow_pickle=True)
            break
    if x is None:
        raise FileNotFoundError(f"Could not find X.npy in {processed_dir}")

    y_path = processed_dir / "y.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing y.npy at {y_path}")
    y = np.load(y_path, allow_pickle=True)
    y = np.asarray(y)
    if y.dtype == object:
        y = y.astype(np.int64)
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D, got shape={y.shape}")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: X={x.shape[0]}, y={y.shape[0]}")

    # If X is accidentally stored as sequences (N,T,F), flatten to epochs.
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
        # If y is epoch-level, lengths may not match; assume user uses correct processed dir.
    elif x.ndim == 1:
        x = x.reshape(-1, 1)

    return x.astype(np.float32), y.astype(int)


def _load_metadata(processed_dir: Path) -> pd.DataFrame:
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv at {meta_path}")
    metadata = pd.read_csv(meta_path)
    if "Subject" not in metadata.columns:
        raise ValueError("metadata.csv must contain a Subject column")
    return metadata


def _resolve_subject_splits(
    metadata: pd.DataFrame,
    cfg: SelectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    subjects = sorted(metadata["Subject"].astype(str).unique().tolist())

    if cfg.train_subjects and cfg.val_subjects and cfg.test_subjects:
        train_subjects = list(cfg.train_subjects)
        val_subjects = list(cfg.val_subjects)
        test_subjects = list(cfg.test_subjects)

        unknown = sorted(
            set(train_subjects + val_subjects + test_subjects) - set(subjects)
        )
        if unknown:
            raise ValueError(
                "Unknown subject(s) in explicit split lists: " + ", ".join(unknown)
            )

        overlap = (
            set(train_subjects).intersection(val_subjects)
            | set(train_subjects).intersection(test_subjects)
            | set(val_subjects).intersection(test_subjects)
        )
        if overlap:
            raise ValueError(
                "Subject split lists must be disjoint; overlap found: "
                + ", ".join(sorted(overlap))
            )
    else:
        val_subjects = subjects[: cfg.val_size_subjects]
        test_subjects = subjects[
            cfg.val_size_subjects : cfg.val_size_subjects + cfg.test_size_subjects
        ]
        train_subjects = [
            s for s in subjects if s not in set(val_subjects).union(set(test_subjects))
        ]

    def select(subject_list: list[str]) -> np.ndarray:
        return metadata[metadata["Subject"].astype(str).isin(subject_list)].index.to_numpy()

    train_idx = select(train_subjects)
    val_idx = select(val_subjects)
    test_idx = select(test_subjects)

    split_info = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }
    return train_idx, val_idx, test_idx, split_info


def _safe_to_dataframe(X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape={X.shape}")
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Feature names length mismatch: len(names)={len(feature_names)} vs X.shape[1]={X.shape[1]}"
        )
    return pd.DataFrame(X, columns=feature_names)


def run_feature_selection(cfg: SelectionConfig) -> Dict[str, Any]:
    processed_dir = Path(cfg.processed_dir)
    feature_names = _load_feature_names(processed_dir, cfg.feature_names_path)
    X, y = _load_xy(processed_dir)
    metadata = _load_metadata(processed_dir)

    feature_df = _safe_to_dataframe(X, feature_names)

    if len(metadata) != len(feature_df):
        raise ValueError(
            f"Length mismatch between metadata and X: metadata={len(metadata)} X={len(feature_df)}"
        )

    train_idx, val_idx, test_idx, split_info = _resolve_subject_splits(metadata, cfg)

    if train_idx.size == 0:
        raise RuntimeError("No training samples selected for feature selection.")

    # Exclude requested features (drop columns) before any filtering.
    exclude_list = cfg.exclude_features or ["sdnn", "sdsd", "sd1", "mean_ibi"]
    exclude_norm = {_normalize_feature_name(x) for x in exclude_list}

    keep_cols: list[str] = []
    excluded_actual: list[str] = []
    for f in feature_names:
        if _normalize_feature_name(f) in exclude_norm:
            excluded_actual.append(f)
        else:
            keep_cols.append(f)

    if len(keep_cols) == 0:
        raise RuntimeError(
            "All features were excluded; check cfg.exclude_features / feature_names.json"
        )

    feature_df = feature_df[keep_cols]

    feature_df_train = feature_df.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]

    # 1) Random Forest feature importances
    rf = RandomForestClassifier(

        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        n_jobs=-1,
        random_state=cfg.rf_random_state,
        class_weight="balanced",
    )

    rf.fit(feature_df_train.values, y_train)
    importances = rf.feature_importances_

    # Map importance->feature (use the kept feature names, not original ones)
    rf_rank = (
        pd.DataFrame(
            {"feature": list(feature_df.columns), "rf_importance": importances}
        )
        .sort_values("rf_importance", ascending=False)
        .reset_index(drop=True)
    )


    keep_by_importance = rf_rank[rf_rank["rf_importance"] > cfg.rf_importance_floor][
        "feature"
    ].tolist()

    # 2) Correlation redundancy filtering on kept set
    corr_method = cfg.corr_method.lower()
    if corr_method not in {"pearson", "spearman"}:
        raise ValueError("corr_method must be pearson or spearman")

    kept_importance_map = dict(zip(rf_rank["feature"], rf_rank["rf_importance"]))

    X_kept = feature_df_train[keep_by_importance]

    # Correlation matrix
    corr = X_kept.corr(method=corr_method).abs()

    # Greedy keep: always keep higher-importance when removing correlated pairs.
    selected: List[str] = []
    # Sort kept features by importance descending
    kept_sorted = sorted(keep_by_importance, key=lambda f: kept_importance_map[f], reverse=True)

    # Track dropped
    dropped: List[str] = []

    # For speed, work with numpy and indices.
    feat_to_idx = {f: i for i, f in enumerate(keep_by_importance)}
    # But selected will be subset; easiest: compute on-the-fly using corr.loc.

    for f in kept_sorted:
        should_drop = False
        for s in selected:
            if corr.loc[f, s] >= cfg.corr_threshold:
                should_drop = True
                break
        if should_drop:
            dropped.append(f)
        else:
            selected.append(f)

    # Package outputs
    out: Dict[str, Any] = {
        "config": cfg.__dict__,
        "n_samples": int(X.shape[0]),
        "n_train_samples": int(len(train_idx)),
        "n_val_samples": int(len(val_idx)),
        "n_test_samples": int(len(test_idx)),
        "n_features_total": int(X.shape[1]),
        "n_features_after_importance": int(len(keep_by_importance)),
        "n_features_final": int(len(selected)),
        "split_subjects": split_info,
        "rf_importances_sorted": rf_rank.to_dict(orient="records"),
        "excluded_features": excluded_actual,
        "dropped_by_importance": [
            f for f in feature_names if f not in keep_by_importance
        ],
        "final_selected_features": selected,
        "dropped_by_correlation": dropped,
    }


    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--feature_names_path", type=str, default=None)

    parser.add_argument("--train_subjects", type=str, default=None, help="comma-separated")
    parser.add_argument("--val_subjects", type=str, default=None, help="comma-separated")
    parser.add_argument("--test_subjects", type=str, default=None, help="comma-separated")
    parser.add_argument("--val_size_subjects", type=int, default=1)
    parser.add_argument("--test_size_subjects", type=int, default=1)

    parser.add_argument("--rf_n_estimators", type=int, default=250)
    parser.add_argument("--rf_max_depth", type=int, default=None)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=2)
    parser.add_argument("--rf_random_state", type=int, default=42)
    parser.add_argument("--rf_importance_floor", type=float, default=1e-6)

    parser.add_argument("--corr_method", type=str, default="pearson")
    parser.add_argument("--corr_threshold", type=float, default=0.99)

    parser.add_argument(
        "--exclude_features",
        type=str,
        default="sdnn,sdsd,sd1,mean_ibi",
        help="comma-separated feature names to drop before RF/correlation filters",
    )

    parser.add_argument("--output_dir", type=str, default="feature_selection_outputs")


    args = parser.parse_args()

    def parse_subject_list(value: str | None) -> list[str] | None:
        if value is None:
            return None
        items = [x.strip() for x in value.split(",") if x.strip()]
        return items or None

    exclude_list = (
        [x.strip() for x in args.exclude_features.split(",") if x.strip()]
        if args.exclude_features
        else None
    )

    cfg = SelectionConfig(
        processed_dir=args.processed_dir,
        feature_names_path=args.feature_names_path,
        train_subjects=parse_subject_list(args.train_subjects),
        val_subjects=parse_subject_list(args.val_subjects),
        test_subjects=parse_subject_list(args.test_subjects),
        val_size_subjects=args.val_size_subjects,
        test_size_subjects=args.test_size_subjects,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_random_state=args.rf_random_state,
        rf_importance_floor=args.rf_importance_floor,
        corr_method=args.corr_method,
        corr_threshold=args.corr_threshold,
        exclude_features=exclude_list,
        output_dir=args.output_dir,
    )


    result = run_feature_selection(cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "feature_selection_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    (out_dir / "selected_features.txt").write_text("\n".join(result["final_selected_features"]))
    (out_dir / "dropped_by_importance.txt").write_text(
        "\n".join(result["dropped_by_importance"])
    )
    (out_dir / "dropped_by_correlation.txt").write_text(
        "\n".join(result["dropped_by_correlation"])
    )

    print("Feature selection finished.")
    print("Total features:", result["n_features_total"])
    print("After RF importance filter:", result["n_features_after_importance"])
    print("Final features:", result["n_features_final"])
    print("Outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
