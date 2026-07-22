from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def regression_metrics(targets, predictions) -> dict[str, float]:
    actual = np.asarray(targets, dtype=np.float64)
    estimated = np.asarray(predictions, dtype=np.float64)
    residuals = estimated - actual
    mse = float(np.mean(residuals**2))
    total_variation = np.sum((actual - actual.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residuals**2) / total_variation)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": r2,
    }


def save_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_training_plot(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, train_loss, label="train")
    axis.plot(epochs, validation_loss, label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE on scaled target")
    axis.set_title("JAX MLP training history")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
