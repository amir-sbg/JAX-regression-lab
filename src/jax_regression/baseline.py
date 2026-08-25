from __future__ import annotations

import jax.numpy as jnp


def _validated_regression_arrays(features, targets=None):
    x = jnp.asarray(features)
    if x.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if x.shape[0] == 0:
        raise ValueError("features must contain at least one row")

    if targets is None:
        return x

    y = jnp.asarray(targets)
    if y.ndim != 1:
        raise ValueError("targets must be a one-dimensional array")
    if y.shape[0] != x.shape[0]:
        raise ValueError("features and targets must contain the same number of rows")
    return x, y


def fit_ridge(features, targets, alpha: float = 1.0):
    """Solve the closed-form ridge objective with an unregularized bias."""
    if alpha < 0:
        raise ValueError("alpha must not be negative")
    x, y = _validated_regression_arrays(features, targets)
    augmented = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    penalty = jnp.diag(jnp.concatenate([jnp.ones(x.shape[1]), jnp.zeros(1)]))
    system = augmented.T @ augmented + alpha * penalty
    return jnp.linalg.solve(system, augmented.T @ y)


def predict_ridge(parameters, features):
    x = _validated_regression_arrays(features)
    params = jnp.asarray(parameters)
    if params.ndim != 1 or params.shape[0] != x.shape[1] + 1:
        raise ValueError("parameters must contain one coefficient per feature plus a bias")
    augmented = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    return augmented @ params
