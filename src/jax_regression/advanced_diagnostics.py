from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .train import mse_loss, regression_loss, tree_l2_norm


def residual_distribution_report(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    """Summarize asymmetry and tail shape that RMSE alone can hide."""

    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.ndim != 1 or targets.shape != predictions.shape:
        raise ValueError("targets and predictions must be aligned one-dimensional arrays")
    if len(targets) < 3:
        raise ValueError("at least three observations are required")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise ValueError("targets and predictions must contain only finite values")

    residuals = predictions - targets
    centered = residuals - residuals.mean()
    variance = float(np.mean(centered**2))
    if variance <= np.finfo(np.float64).eps:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        scale = np.sqrt(variance)
        skewness = float(np.mean((centered / scale) ** 3))
        excess_kurtosis = float(np.mean((centered / scale) ** 4) - 3.0)

    return {
        "n_samples": int(len(residuals)),
        "mean_residual": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "underprediction_rate": float(np.mean(residuals < 0.0)),
        "overprediction_rate": float(np.mean(residuals > 0.0)),
    }


def target_quantile_error_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    bins: int = 4,
) -> dict[str, int | list[dict[str, float | int]]]:
    """Measure whether errors concentrate in low or high target ranges."""

    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.ndim != 1 or predictions.ndim != 1 or targets.shape != predictions.shape:
        raise ValueError("targets and predictions must be aligned one-dimensional arrays")
    if bins < 2 or bins > len(targets):
        raise ValueError("bins must be between 2 and the number of observations")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise ValueError("targets and predictions must contain only finite values")

    edges = np.quantile(targets, np.linspace(0.0, 1.0, bins + 1))
    assignments = np.searchsorted(edges[1:-1], targets, side="right")
    rows = []
    for index in range(bins):
        mask = assignments == index
        if not np.any(mask):
            continue
        residuals = predictions[mask] - targets[mask]
        rows.append(
            {
                "bin": index,
                "count": int(mask.sum()),
                "target_min": float(targets[mask].min()),
                "target_max": float(targets[mask].max()),
                "mae": float(np.mean(np.abs(residuals))),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mean_residual": float(residuals.mean()),
            }
        )
    return {"requested_bins": bins, "populated_bins": len(rows), "bins": rows}
