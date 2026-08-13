"""evaluate.py

Compute metrics and confusion matrix.

Outputs:
- outputs/confusion_matrix/confusion_matrix.csv
- outputs/confusion_matrix/confusion_matrix.png
- outputs/metrics.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


@dataclass
@dataclass
class EvalResult:
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    cohen_kappa: float
    mcc: float
    precision_per_class: list[float]
    recall_per_class: list[float]
    f1_per_class: list[float]
    support_per_class: list[int]
    confusion_matrix: np.ndarray
    classification_report: dict


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> EvalResult:

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    kappa = cohen_kappa_score(y_test, y_pred)

    mcc = matthews_corrcoef(y_test, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    print("Normalized confusion matrix")
    print(np.round(cm_norm,3))
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return EvalResult(
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        cohen_kappa=float(kappa),
        mcc=float(mcc),
        precision_per_class=precision.tolist(),
        recall_per_class=recall.tolist(),
        f1_per_class=f1.tolist(),
        support_per_class=support.tolist(),
        confusion_matrix=cm,
        classification_report=report,
    )


def save_eval_outputs(
    result: EvalResult,
    outputs_dir: str | Path,
    tag: str = "baseline",
    class_names: list[str] | None = None,
) -> Dict[str, Any]:
    outputs_dir = Path(outputs_dir)
    cm_dir = outputs_dir / "confusion_matrix"
    cm_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = outputs_dir / "metrics.json"

    payload = {
        "tag": tag,

        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,

        "f1_macro": result.f1_macro,
        "f1_weighted": result.f1_weighted,

        "cohen_kappa": result.cohen_kappa,
        "mcc": result.mcc,

        "precision_per_class": result.precision_per_class,
        "recall_per_class": result.recall_per_class,
        "f1_per_class": result.f1_per_class,
        "support_per_class": result.support_per_class,

        "classification_report": result.classification_report,

        "confusion_matrix": result.confusion_matrix.tolist(),
    }

    metrics_path.write_text(json.dumps(payload, indent=2))

    # CSV
    cm_csv = cm_dir / "confusion_matrix.csv"
    np.savetxt(
        cm_csv,
        result.confusion_matrix,
        delimiter=",",
        fmt="%d",
    )

    # PNG
    cm_png = cm_dir / "confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    plt.imshow(result.confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({tag})")
    plt.colorbar()

    n_classes = result.confusion_matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = result.confusion_matrix.max() / 2.0 if result.confusion_matrix.size else 0
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(
                j,
                i,
                format(result.confusion_matrix[i, j], "d"),
                horizontalalignment="center",
                color="white" if result.confusion_matrix[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()

    return payload

