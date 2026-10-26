from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .train import mse_loss, regression_loss, tree_l2_norm
from .model import mlp_apply


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


def ridge_leverage_report(
    features: np.ndarray,
    alpha: float = 0.0,
    top_k: int = 5,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    """Report diagonal hat-matrix leverage for a ridge design matrix."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")
    if alpha < 0.0:
        raise ValueError("alpha must not be negative")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    design = np.column_stack([features, np.ones(len(features))])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    inverse = np.linalg.pinv(design.T @ design + penalty)
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    order = np.argsort(-leverage, kind="stable")[: min(top_k, len(features))]
    threshold = 2.0 * design.shape[1] / len(features)
    return {
        "n_samples": int(len(features)),
        "effective_parameters": float(leverage.sum()),
        "mean_leverage": float(leverage.mean()),
        "max_leverage": float(leverage.max()),
        "high_leverage_threshold": float(threshold),
        "high_leverage_count": int(np.sum(leverage > threshold)),
        "top_observations": [
            {"index": int(index), "leverage": float(leverage[index])}
            for index in order
        ],
    }


def ridge_influence_report(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.0,
    top_k: int = 5,
) -> dict[str, float | int | list[dict[str, float | int]]]:
    """Rank training rows by an approximate Cook's distance for ridge regression."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned matrix and vector inputs")
    if len(features) <= features.shape[1] + 1:
        raise ValueError("more rows than fitted coefficients are required")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("features and targets must contain only finite values")
    if alpha < 0.0 or top_k < 1:
        raise ValueError("alpha must not be negative and top_k must be positive")

    design = np.column_stack([features, np.ones(len(features))])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    inverse = np.linalg.pinv(design.T @ design + penalty)
    coefficients = inverse @ design.T @ targets
    residuals = targets - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    degrees_of_freedom = max(1, len(features) - design.shape[1])
    residual_variance = float(np.sum(residuals**2) / degrees_of_freedom)
    denominator = design.shape[1] * max(residual_variance, np.finfo(float).eps)
    cooks_distance = (residuals**2 / denominator) * (
        leverage / np.maximum((1.0 - leverage) ** 2, np.finfo(float).eps)
    )
    order = np.argsort(-cooks_distance, kind="stable")[: min(top_k, len(features))]
    threshold = 4.0 / len(features)
    return {
        "n_samples": int(len(features)),
        "residual_variance": residual_variance,
        "influence_threshold": float(threshold),
        "influential_count": int(np.sum(cooks_distance > threshold)),
        "max_cooks_distance": float(cooks_distance.max()),
        "top_observations": [
            {
                "index": int(index),
                "cooks_distance": float(cooks_distance[index]),
                "leverage": float(leverage[index]),
                "residual": float(residuals[index]),
            }
            for index in order
        ],
    }


def hessian_trace_estimate(
    parameters,
    features: np.ndarray,
    targets: np.ndarray,
    key: jax.Array,
    probes: int = 16,
    l2_penalty: float = 0.0,
) -> dict[str, float | int]:
    """Estimate aggregate loss curvature with Hutchinson probes."""

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned matrix and vector inputs")
    if probes < 1:
        raise ValueError("probes must be positive")
    if l2_penalty < 0.0:
        raise ValueError("l2_penalty must not be negative")

    x = jnp.asarray(features)
    y = jnp.asarray(targets)
    gradient = jax.grad(lambda params: mse_loss(params, x, y, l2_penalty))
    leaves, treedef = jax.tree_util.tree_flatten(parameters)
    estimates = []
    for probe_key in jax.random.split(key, probes):
        leaf_keys = jax.random.split(probe_key, len(leaves))
        vectors = [
            jax.random.rademacher(leaf_key, leaf.shape, dtype=leaf.dtype)
            for leaf_key, leaf in zip(leaf_keys, leaves)
        ]
        direction = jax.tree_util.tree_unflatten(treedef, vectors)
        _, hessian_vector = jax.jvp(gradient, (parameters,), (direction,))
        estimates.append(
            sum(
                float(jnp.vdot(vector, hv))
                for vector, hv in zip(vectors, jax.tree_util.tree_leaves(hessian_vector))
            )
        )

    values = np.asarray(estimates, dtype=np.float64)
    return {
        "probes": int(probes),
        "trace_estimate": float(values.mean()),
        "probe_std": float(values.std()),
        "standard_error": float(values.std() / np.sqrt(probes)),
    }


def empirical_ntk_spectrum(
    parameters,
    features: np.ndarray,
    max_samples: int = 64,
) -> dict[str, float | int | list[float]]:
    """Summarize the spectrum of the empirical neural tangent kernel."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    selected = jnp.asarray(features[:max_samples])
    gradient_fn = jax.grad(lambda params, row: mlp_apply(params, row))
    rows = []
    for row in selected:
        gradient = gradient_fn(parameters, row)
        rows.append(
            np.concatenate(
                [np.asarray(leaf).reshape(-1) for leaf in jax.tree_util.tree_leaves(gradient)]
            )
        )
    jacobian = np.stack(rows)
    kernel = jacobian @ jacobian.T / jacobian.shape[1]
    eigenvalues = np.linalg.eigvalsh(kernel)
    eigenvalues = np.clip(eigenvalues, 0.0, None)[::-1]
    positive = eigenvalues[eigenvalues > max(1e-12, eigenvalues[0] * 1e-8)]
    probabilities = positive / positive.sum() if len(positive) else np.array([1.0])
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    condition_number = float(positive[0] / positive[-1]) if len(positive) else 0.0
    return {
        "samples": int(len(selected)),
        "parameters": int(jacobian.shape[1]),
        "numerical_rank": int(len(positive)),
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "trace": float(eigenvalues.sum()),
        "top_eigenvalues": [float(value) for value in eigenvalues[:5]],
    }


def gradient_noise_scale(
    parameters,
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 32,
    loss: str = "mse",
    huber_delta: float = 1.0,
) -> dict[str, float | int]:
    """Compare minibatch gradient variance with the full-data gradient signal."""

    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned matrix and vector inputs")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    x = jnp.asarray(features)
    y = jnp.asarray(targets)
    gradient_fn = jax.grad(
        lambda params, batch_x, batch_y: regression_loss(
            params,
            batch_x,
            batch_y,
            loss=loss,
            huber_delta=huber_delta,
        )
    )
    full_gradient = gradient_fn(parameters, x, y)
    full_leaves = [np.asarray(leaf).reshape(-1) for leaf in jax.tree_util.tree_leaves(full_gradient)]
    full_vector = np.concatenate(full_leaves)

    batch_vectors = []
    for start in range(0, len(features), batch_size):
        batch_gradient = gradient_fn(
            parameters,
            x[start : start + batch_size],
            y[start : start + batch_size],
        )
        batch_vectors.append(
            np.concatenate(
                [np.asarray(leaf).reshape(-1) for leaf in jax.tree_util.tree_leaves(batch_gradient)]
            )
        )
    stacked = np.stack(batch_vectors)
    variance = float(np.mean(np.sum((stacked - full_vector) ** 2, axis=1)))
    signal = float(np.sum(full_vector**2))
    return {
        "batches": int(len(batch_vectors)),
        "batch_size": int(batch_size),
        "gradient_signal_squared": signal,
        "gradient_variance": variance,
        "noise_scale": float(variance / max(signal, np.finfo(float).eps)),
    }
