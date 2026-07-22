from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .model import mlp_apply


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.90
    l2_penalty: float = 1e-4
    patience: int = 30


@dataclass(frozen=True)
class TrainingResult:
    parameters: tuple[dict[str, jax.Array], ...]
    history: list[dict[str, float]]
    best_epoch: int


def mse_loss(parameters, features, targets, l2_penalty: float = 0.0):
    predictions = mlp_apply(parameters, features)
    residuals = predictions - targets
    weight_penalty = sum(jnp.sum(layer["weights"] ** 2) for layer in parameters)
    return jnp.mean(residuals**2) + l2_penalty * weight_penalty


@jax.jit
def gradient_step(
    parameters,
    velocity,
    features,
    targets,
    learning_rate: float,
    momentum: float,
    l2_penalty: float,
):
    loss, gradients = jax.value_and_grad(mse_loss)(
        parameters,
        features,
        targets,
        l2_penalty,
    )
    velocity = jax.tree_util.tree_map(
        lambda previous, gradient: momentum * previous + gradient,
        velocity,
        gradients,
    )
    parameters = jax.tree_util.tree_map(
        lambda parameter, update: parameter - learning_rate * update,
        parameters,
        velocity,
    )
    return parameters, velocity, loss


def train_model(
    parameters,
    features: np.ndarray,
    targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    config: TrainingConfig,
    key: jax.Array,
) -> TrainingResult:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if not 0 <= config.momentum < 1:
        raise ValueError("momentum must be between 0 and 1")
    if config.l2_penalty < 0 or config.patience < 1:
        raise ValueError("l2_penalty must not be negative and patience must be positive")

    velocity = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    best_parameters = parameters
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        key, permutation_key = jax.random.split(key)
        permutation = np.asarray(
            jax.random.permutation(permutation_key, len(features))
        )
        for start in range(0, len(features), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            parameters, velocity, _ = gradient_step(
                parameters,
                velocity,
                features[indices],
                targets[indices],
                config.learning_rate,
                config.momentum,
                config.l2_penalty,
            )

        train_loss = float(mse_loss(parameters, features, targets))
        validation_loss = float(
            mse_loss(parameters, validation_features, validation_targets)
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_validation_loss - 1e-6:
            best_parameters = parameters
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    return TrainingResult(
        parameters=best_parameters,
        history=history,
        best_epoch=best_epoch,
    )
