from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

import numpy as np

from Preprocessing.normalizer import FeatureNormalizer


# ==========================
# Load dataset
# ==========================

X = np.load(
    PROJECT_ROOT / "processed" / "X.npy"
)

print("Original Shape :", X.shape)

# ==========================
# Normalize
# ==========================

normalizer = FeatureNormalizer()

X_norm = normalizer.fit_transform(X)

normalizer.summary()

# ==========================
# Save
# ==========================

normalizer.save(
    PROJECT_ROOT / "processed" / "normalization.json"
)

np.save(
    PROJECT_ROOT / "processed" / "X_normalized.npy",
    X_norm,
)

print()

print("=" * 60)

print("Normalized Dataset")

print("=" * 60)

print("Shape :", X_norm.shape)

print()

print("Mean")

print(np.mean(X_norm, axis=0))

print()

print("Std")

print(np.std(X_norm, axis=0))