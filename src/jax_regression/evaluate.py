from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _matching_arrays(targets, predictions) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(targets, dtype=np.float64)
    estimated = np.asarray(predictions, dtype=np.float64)
    if actual.shape != estimated.shape:
        raise ValueError("targets and predictions must have the same shape")
    if actual.size == 0:
        raise ValueError("targets and predictions must not be empty")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(estimated)):
        raise ValueError("targets and predictions must contain only finite values")
    return actual, estimated


def regression_metrics(targets, predictions) -> dict[str, float]:
    actual, estimated = _matching_arrays(targets, predictions)
    residuals = estimated - actual
    mse = float(np.mean(residuals**2))
    total_variation = np.sum((actual - actual.mean()) ** 2)
    if total_variation == 0:
        r2 = 1.0 if np.allclose(residuals, 0.0) else 0.0
    else:
        r2 = 1.0 - float(np.sum(residuals**2) / total_variation)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": r2,
    }


def residual_summary(targets, predictions) -> dict[str, float]:
    actual, estimated = _matching_arrays(targets, predictions)
    residuals = estimated - actual
    return {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "residual_std": float(np.std(residuals)),
        "p10_residual": float(np.percentile(residuals, 10)),
        "p90_residual": float(np.percentile(residuals, 90)),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def empirical_interval_summary(
    targets,
    predictions,
    coverage: float = 0.90,
) -> dict[str, float]:
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    actual, estimated = _matching_arrays(targets, predictions)
    absolute_errors = np.abs(estimated - actual)
    radius = float(np.quantile(absolute_errors, coverage))
    covered = np.abs(estimated - actual) <= radius
    return {
        "target_coverage": float(coverage),
        "observed_coverage": float(np.mean(covered)),
        "interval_radius": radius,
        "mean_interval_width": float(2.0 * radius),
    }


def binned_residual_summary(
    targets,
    predictions,
    bins: int = 4,
) -> list[dict[str, float | int]]:
    if bins < 1:
        raise ValueError("bins must be at least 1")
    actual, estimated = _matching_arrays(targets, predictions)
    residuals = estimated - actual

    if float(actual.min()) == float(actual.max()):
        edges = np.array([actual.min(), actual.max()], dtype=np.float64)
    else:
        edges = np.linspace(actual.min(), actual.max(), bins + 1)

    rows = []
    for index in range(len(edges) - 1):
        left = edges[index]
        right = edges[index + 1]
        if index == len(edges) - 2:
            mask = (actual >= left) & (actual <= right)
        else:
            mask = (actual >= left) & (actual < right)
        if not np.any(mask):
            continue

        bin_residuals = residuals[mask]
        rows.append(
            {
                "bin": index + 1,
                "target_min": float(left),
                "target_max": float(right),
                "examples": int(np.sum(mask)),
                "mean_actual": float(np.mean(actual[mask])),
                "mean_prediction": float(np.mean(estimated[mask])),
                "mean_residual": float(np.mean(bin_residuals)),
                "mae": float(np.mean(np.abs(bin_residuals))),
            }
        )
    return rows


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


def save_residual_plot(targets, predictions, path: Path) -> None:
    actual = np.asarray(targets, dtype=np.float64)
    residuals = np.asarray(predictions, dtype=np.float64) - actual
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(actual, residuals, alpha=0.75, edgecolor="none")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Actual target")
    axis.set_ylabel("Prediction residual")
    axis.set_title("MLP residuals on test set")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
