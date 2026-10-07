from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .model import mlp_apply
from .train import mse_loss, tree_l2_norm


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


def input_gradient_check(
    parameters: tuple[dict[str, jax.Array], ...],
    features: np.ndarray,
    epsilon: float = 1e-3,
    max_rows: int = 8,
) -> dict[str, float | int]:
    """Compare autodiff input gradients with central finite differences."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    feature_array = np.asarray(_validated_feature_matrix(features)[:max_rows])
    autodiff = input_gradients(parameters, feature_array)
    finite_difference = np.empty_like(autodiff)
    for column in range(feature_array.shape[1]):
        positive = feature_array.copy()
        negative = feature_array.copy()
        positive[:, column] += epsilon
        negative[:, column] -= epsilon
        positive_prediction = np.asarray(jax.vmap(lambda row: mlp_apply(parameters, row))(positive))
        negative_prediction = np.asarray(jax.vmap(lambda row: mlp_apply(parameters, row))(negative))
        finite_difference[:, column] = (
            positive_prediction - negative_prediction
        ) / (2.0 * epsilon)

    absolute_error = np.abs(autodiff - finite_difference)
    scale = np.maximum(np.abs(autodiff), np.abs(finite_difference))
    relative_error = absolute_error / np.maximum(scale, 1e-6)
    return {
        "rows_checked": int(len(feature_array)),
        "features_checked": int(feature_array.shape[1]),
        "epsilon": float(epsilon),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_relative_error": float(np.mean(relative_error)),
        "max_relative_error": float(np.max(relative_error)),
    }


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


def integrated_gradient_importance(
    parameters: tuple[dict[str, jax.Array], ...],
    features: np.ndarray,
    feature_names: tuple[str, ...],
    steps: int = 32,
) -> list[dict[str, float | int | str]]:
    """Aggregate path attributions from the standardized feature mean."""

    if steps < 1:
        raise ValueError("steps must be positive")
    feature_array = _validated_feature_matrix(features)
    if feature_array.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match feature columns")

    baseline = jnp.zeros(feature_array.shape[1], dtype=feature_array.dtype)
    alphas = jnp.linspace(0.0, 1.0, steps + 1)
    gradient_fn = jax.grad(lambda row: mlp_apply(parameters, row))

    def attribute(row):
        path = baseline + alphas[:, None] * (row - baseline)
        path_gradients = jax.vmap(gradient_fn)(path)
        average_gradient = jnp.mean(
            (path_gradients[:-1] + path_gradients[1:]) * 0.5,
            axis=0,
        )
        return (row - baseline) * average_gradient

    attributions = np.asarray(jax.vmap(attribute)(feature_array))
    rows = []
    for index, name in enumerate(feature_names):
        values = attributions[:, index]
        rows.append(
            {
                "feature": name,
                "mean_attribution": float(np.mean(values)),
                "mean_abs_attribution": float(np.mean(np.abs(values))),
            }
        )
    total = sum(float(row["mean_abs_attribution"]) for row in rows)
    rows.sort(key=lambda row: float(row["mean_abs_attribution"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["normalized_importance"] = (
            float(row["mean_abs_attribution"]) / total if total > 0 else 0.0
        )
    return rows


def feature_correlation_pairs(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    top_k: int = 8,
) -> list[dict[str, float | str]]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match feature columns")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(x) < 2:
        raise ValueError("at least two rows are needed for correlation")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must contain only finite values")

    correlations = np.corrcoef(x, rowvar=False)
    rows = []
    for left in range(len(feature_names)):
        for right in range(left + 1, len(feature_names)):
            value = float(correlations[left, right])
            if not np.isfinite(value):
                value = 0.0
            rows.append(
                {
                    "feature_a": feature_names[left],
                    "feature_b": feature_names[right],
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )
    rows.sort(key=lambda row: float(row["abs_correlation"]), reverse=True)
    return rows[:top_k]


def design_matrix_report(features: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must contain at least one row and one column")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must contain only finite values")

    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    if singular_values.size == 0:
        effective_rank = 0.0
        condition_number = 0.0
        smallest = 0.0
        largest = 0.0
    else:
        largest = float(singular_values[0])
        smallest = float(singular_values[-1])
        safe_values = singular_values[singular_values > 1e-12]
        if safe_values.size == 0:
            effective_rank = 0.0
            condition_number = 0.0
        else:
            weights = safe_values / safe_values.sum()
            entropy = -float(np.sum(weights * np.log(weights)))
            effective_rank = float(np.exp(entropy))
            condition_number = largest / max(float(safe_values[-1]), 1e-12)

    return {
        "rows": int(x.shape[0]),
        "columns": int(x.shape[1]),
        "rank": int(np.linalg.matrix_rank(centered)),
        "effective_rank": effective_rank,
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_number": float(condition_number),
    }


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


def random_parameter_direction(parameters, key: jax.Array):
    leaves, treedef = jax.tree_util.tree_flatten(parameters)
    keys = jax.random.split(key, len(leaves))
    direction_leaves = [
        jax.random.normal(leaf_key, shape=leaf.shape, dtype=leaf.dtype)
        for leaf_key, leaf in zip(keys, leaves)
    ]
    direction = jax.tree_util.tree_unflatten(treedef, direction_leaves)
    norm = tree_l2_norm(direction)
    return jax.tree_util.tree_map(lambda value: value / (norm + 1e-8), direction)


def tree_dot(left, right) -> float:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        raise ValueError("trees must have the same structure")
    total = sum(
        (jnp.vdot(left_leaf, right_leaf) for left_leaf, right_leaf in zip(left_leaves, right_leaves)),
        jnp.array(0.0),
    )
    return float(total)


def directional_curvature(
    parameters: tuple[dict[str, jax.Array], ...],
    features: np.ndarray,
    targets: np.ndarray,
    key: jax.Array,
    probes: int = 4,
    l2_penalty: float = 0.0,
) -> dict[str, float | int]:
    if probes < 1:
        raise ValueError("probes must be at least 1")
    feature_array = _validated_feature_matrix(features)
    target_array = np.asarray(targets, dtype=np.float32)
    if target_array.ndim != 1 or len(target_array) != feature_array.shape[0]:
        raise ValueError("targets must be a one-dimensional array with one value per row")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("targets must contain only finite values")

    target_array = jnp.asarray(target_array)
    grad_fn = jax.grad(lambda params: mse_loss(params, feature_array, target_array, l2_penalty))
    curvatures = []
    for probe_key in jax.random.split(key, probes):
        direction = random_parameter_direction(parameters, probe_key)
        _, hessian_vector = jax.jvp(grad_fn, (parameters,), (direction,))
        curvatures.append(tree_dot(direction, hessian_vector))

    values = np.asarray(curvatures, dtype=np.float64)
    return {
        "probes": int(probes),
        "mean_directional_curvature": float(values.mean()),
        "std_directional_curvature": float(values.std()),
        "min_directional_curvature": float(values.min()),
        "max_directional_curvature": float(values.max()),
    }


def _checked_predictions(predictions: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != expected_shape:
        raise ValueError("predict_fn must return one prediction per row")
    if not np.all(np.isfinite(values)):
        raise ValueError("predict_fn returned non-finite predictions")
    return values
