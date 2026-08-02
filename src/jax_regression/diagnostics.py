from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .model import mlp_apply


def input_gradients(
    parameters: tuple[dict[str, jax.Array], ...],
    features: np.ndarray,
) -> np.ndarray:
    feature_array = jnp.asarray(features)
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
