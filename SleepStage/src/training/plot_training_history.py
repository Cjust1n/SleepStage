from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _get_series(history_dict: dict[str, Any] | None, key: str) -> list[float] | None:
    if not history_dict:
        return None
    v = history_dict.get(key)
    if v is None:
        return None
    if not isinstance(v, list):
        return None
    try:
        return [float(x) for x in v]
    except Exception:
        return None


def plot_from_run_summary(run_summary_path: Path, out_dir: Path) -> Path:
    run_summary_path = run_summary_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = run_summary_path.read_text().strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Backward compatibility: older run_summary.json was written as a python dict string
        import ast

        data = ast.literal_eval(raw)

    history_per_epoch = data.get("history_per_epoch")


    # Determine epochs length from any available series
    loss = _get_series(history_per_epoch, "loss")
    val_loss = _get_series(history_per_epoch, "val_loss")
    acc = _get_series(history_per_epoch, "accuracy")
    val_acc = _get_series(history_per_epoch, "val_accuracy")

    if loss is None and val_loss is None and acc is None and val_acc is None:
        raise RuntimeError(
            f"No per-epoch history found in {run_summary_path}. "
            "Expected keys like loss/val_loss/accuracy/val_accuracy."
        )

    n_epochs = None
    for s in [loss, val_loss, acc, val_acc]:
        if s is not None:
            n_epochs = len(s)
            break

    if n_epochs is None:
        raise RuntimeError("Could not infer number of epochs from history data")

    epochs = list(range(1, n_epochs + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax0 = axes[0]
    plotted_any_loss = False
    if loss is not None:
        ax0.plot(epochs, loss, label="Train loss")
        plotted_any_loss = True
    if val_loss is not None:
        ax0.plot(epochs, val_loss, label="Val loss")
        plotted_any_loss = True

    if plotted_any_loss:
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.set_title("Training vs Validation Loss")
        ax0.grid(True, alpha=0.3)
        ax0.legend()
    else:
        ax0.axis("off")

    # --- Accuracy ---
    ax1 = axes[1]
    plotted_any_acc = False
    if acc is not None:
        ax1.plot(epochs, acc, label="Train accuracy")
        plotted_any_acc = True
    if val_acc is not None:
        ax1.plot(epochs, val_acc, label="Val accuracy")
        plotted_any_acc = True

    if plotted_any_acc:
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Training vs Validation Accuracy")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.axis("off")

    fig.suptitle(run_summary_path.parent.name + ": per-epoch training history", y=1.02)
    fig.tight_layout()

    # Save outputs
    out_path = out_dir / "training_history.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_summary",
        type=str,
        default="outputs/run_summary.json",
        help="Path to outputs/run_summary.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Directory to store training_history.png",
    )
    args = parser.parse_args()

    run_summary_path = Path(args.run_summary)
    out_dir = Path(args.out_dir)

    out_path = plot_from_run_summary(run_summary_path=run_summary_path, out_dir=out_dir)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

