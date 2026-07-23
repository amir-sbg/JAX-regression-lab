from __future__ import annotations

import numpy as np
import pytest
import jax

from jax_regression.baseline import fit_ridge, predict_ridge
from jax_regression.config import ExperimentConfig
from jax_regression.data import load_regression_data
from jax_regression.evaluate import regression_metrics
from jax_regression.model import (
    init_mlp,
    load_parameters,
    parameter_count,
    predict_batch,
    save_parameters,
)
from jax_regression.train import TrainingConfig, train_model


def test_data_split_and_scaling_are_deterministic() -> None:
    first = load_regression_data(seed=7)
    second = load_regression_data(seed=7)

    assert first.x_train.shape == (265, 10)
    assert first.x_validation.shape == (88, 10)
    assert first.x_test.shape == (89, 10)
    np.testing.assert_allclose(first.x_train, second.x_train)
    np.testing.assert_allclose(first.y_test, second.y_test)
    np.testing.assert_allclose(first.x_train.mean(axis=0), 0, atol=1e-6)


def test_ridge_baseline_matches_linear_relationship() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    targets = 2 * features[:, 0] + 1
    parameters = fit_ridge(features, targets, alpha=0.0)
    predictions = predict_ridge(parameters, features)
    np.testing.assert_allclose(predictions, targets, atol=1e-5)


def test_mlp_shapes_and_parameter_count() -> None:
    parameters = init_mlp(3, (8, 4), jax.random.PRNGKey(0))
    predictions = predict_batch(parameters, np.zeros((5, 3), dtype=np.float32))
    assert predictions.shape == (5,)
    assert parameter_count(parameters) == 73


def test_saved_parameters_round_trip(tmp_path) -> None:
    parameters = init_mlp(3, (5,), jax.random.PRNGKey(4))
    checkpoint = tmp_path / "parameters.npz"
    save_parameters(parameters, checkpoint)

    restored = load_parameters(checkpoint)
    features = np.ones((4, 3), dtype=np.float32)
    np.testing.assert_allclose(
        predict_batch(parameters, features),
        predict_batch(restored, features),
    )


def test_training_improves_a_small_regression_problem() -> None:
    features = np.linspace(-1, 1, 24, dtype=np.float32).reshape(-1, 1)
    targets = (0.75 * features[:, 0] - 0.25).astype(np.float32)
    parameters = init_mlp(1, (8,), jax.random.PRNGKey(1))
    result = train_model(
        parameters,
        features,
        targets,
        features,
        targets,
        TrainingConfig(epochs=20, batch_size=8, learning_rate=0.05, patience=20),
        jax.random.PRNGKey(2),
    )
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]


def test_metrics_include_standard_regression_values() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert metrics["mse"] == 0.5
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mae"] == 0.5


def test_experiment_config_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig(learning_rate=0)
