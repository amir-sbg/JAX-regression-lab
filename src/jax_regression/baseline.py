from __future__ import annotations

import jax.numpy as jnp


def fit_ridge(features, targets, alpha: float = 1.0):
    """Solve the closed-form ridge objective with an unregularized bias."""
    x = jnp.asarray(features)
    y = jnp.asarray(targets)
    augmented = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    penalty = jnp.diag(jnp.concatenate([jnp.ones(x.shape[1]), jnp.zeros(1)]))
    system = augmented.T @ augmented + alpha * penalty
    return jnp.linalg.solve(system, augmented.T @ y)


def predict_ridge(parameters, features):
    x = jnp.asarray(features)
    augmented = jnp.concatenate([x, jnp.ones((x.shape[0], 1))], axis=1)
    return augmented @ parameters
