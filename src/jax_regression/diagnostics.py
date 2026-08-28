from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .model import mlp_apply


def _validated_feature_matrix(features: np.ndarray) -> jax.Array:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if values.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must contain only finite values")
    return jnp.asarray(values)


def input_gradients(
    parameters: tuple[dict[str, jax.Array], ...],
    features: np.ndarray,
) -> np.ndarray:
    feature_array = _validated_feature_matrix(features)
    gradient_fn = jax.grad(lambda row: mlp_apply(parameters, row))
    return np.asarray(jax.vmap(gradient_fn)(feature_array))


def feature_sensitivity(
    parameters: tuple[dict[str, jax.Array], ...],
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    gradients = input_gradients(parameters, features)
    if gradients.ndim != 2:
        raise ValueError("expected a two-dimensional gradient matrix")
    if gradients.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match feature columns")

    rows = []
    for index, name in enumerate(feature_names):
        column = gradients[:, index]
        rows.append(
            {
                "feature": name,
                "mean_gradient": float(np.mean(column)),
                "mean_abs_gradient": float(np.mean(np.abs(column))),
                "std_gradient": float(np.std(column)),
            }
        )
    rows.sort(key=lambda row: row["mean_abs_gradient"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def permutation_importance(
    features: np.ndarray,
    targets: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: tuple[str, ...],
    repeats: int = 5,
    seed: int = 42,
) -> list[dict[str, float | int | str]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("features must be 2-D and targets must be 1-D")
    if len(x) != len(y):
        raise ValueError("features and targets must contain matching rows")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match feature columns")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("features and targets must contain only finite values")

    baseline_predictions = _checked_predictions(predict_fn(x), y.shape)
    baseline_mse = float(np.mean((baseline_predictions - y) ** 2))
    rng = np.random.default_rng(seed)
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        deltas = []
        for _ in range(repeats):
            permuted = x.copy()
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            predictions = _checked_predictions(predict_fn(permuted), y.shape)
            mse = float(np.mean((predictions - y) ** 2))
            deltas.append(mse - baseline_mse)
        rows.append(
            {
                "feature": feature_name,
                "baseline_mse": baseline_mse,
                "mean_mse_increase": float(np.mean(deltas)),
                "std_mse_increase": float(np.std(deltas)),
                "repeats": repeats,
            }
        )

    positive_total = sum(max(0.0, float(row["mean_mse_increase"])) for row in rows)
    for row in rows:
        row["normalized_importance"] = (
            max(0.0, float(row["mean_mse_increase"])) / positive_total
            if positive_total > 0.0
            else 0.0
        )
    rows.sort(key=lambda row: row["mean_mse_increase"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _checked_predictions(predictions: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError("predict_fn must return one prediction per row")
    if not np.all(np.isfinite(values)):
        raise ValueError("predict_fn returned non-finite predictions")
    return values
