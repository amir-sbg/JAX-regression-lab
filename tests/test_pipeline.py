from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_regression.baseline import fit_ridge, predict_ridge
from jax_regression.config import ExperimentConfig
from jax_regression.data import load_regression_data
from jax_regression.diagnostics import feature_sensitivity
from jax_regression.evaluate import (
    binned_residual_summary,
    empirical_interval_summary,
    interval_calibration_curve,
    regression_metrics,
    residual_summary,
    split_conformal_interval_summary,
)
from jax_regression.model import (
    init_mlp,
    load_parameters,
    parameter_count,
    predict_batch,
    save_parameters,
)
from jax_regression.train import TrainingConfig, clip_gradients, train_model, tree_l2_norm


def test_data_split_and_scaling_are_deterministic() -> None:
    first = load_regression_data(seed=7)
    second = load_regression_data(seed=7)

    assert first.x_train.shape == (265, 10)
    assert first.x_validation.shape == (88, 10)
    assert first.x_test.shape == (89, 10)
    np.testing.assert_allclose(first.x_train, second.x_train)
    np.testing.assert_allclose(first.y_test, second.y_test)
    np.testing.assert_allclose(first.x_train.mean(axis=0), 0, atol=1e-6)


def test_data_loader_accepts_custom_split_sizes() -> None:
    data = load_regression_data(seed=7, validation_size=0.10, test_size=0.30)

    assert len(data.x_train) + len(data.x_validation) + len(data.x_test) == 442
    assert len(data.x_validation) < len(data.x_test)


def test_ridge_baseline_matches_linear_relationship() -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    targets = 2 * features[:, 0] + 1
    parameters = fit_ridge(features, targets, alpha=0.0)
    predictions = predict_ridge(parameters, features)
    np.testing.assert_allclose(predictions, targets, atol=1e-5)


def test_ridge_baseline_validates_input_shapes() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        fit_ridge(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="same number"):
        fit_ridge(np.ones((3, 2)), np.ones(2))
    with pytest.raises(ValueError, match="alpha"):
        fit_ridge(np.ones((3, 2)), np.ones(3), alpha=-0.1)
    with pytest.raises(ValueError, match="one coefficient"):
        predict_ridge(np.ones(2), np.ones((3, 2)))


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
    assert result.best_epoch >= 1
    assert result.best_validation_loss == pytest.approx(
        result.history[result.best_epoch - 1]["validation_loss"]
    )
    assert "gradient_norm" in result.history[-1]


def test_gradient_clipping_rescales_large_updates() -> None:
    gradients = ({"weights": jnp.array([3.0, 4.0], dtype=jnp.float32)},)
    clipped, norm = clip_gradients(gradients, max_norm=1.0)

    assert float(norm) == pytest.approx(5.0)
    assert float(tree_l2_norm(clipped)) == pytest.approx(1.0)


def test_metrics_include_standard_regression_values() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert metrics["mse"] == 0.5
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["mae"] == 0.5


def test_metrics_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="must not be empty"):
        regression_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        regression_metrics(np.array([1.0]), np.array([np.nan]))


def test_metrics_handle_constant_targets() -> None:
    perfect = regression_metrics(np.array([2.0, 2.0]), np.array([2.0, 2.0]))
    imperfect = regression_metrics(np.array([2.0, 2.0]), np.array([2.0, 3.0]))

    assert perfect["r2"] == 1.0
    assert imperfect["r2"] == 0.0


def test_residual_summary_reports_error_shape() -> None:
    summary = residual_summary(np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 4.0]))

    assert summary["mean_residual"] == pytest.approx(1 / 3)
    assert summary["max_abs_residual"] == 1.0


def test_residual_summary_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        residual_summary(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="must not be empty"):
        residual_summary(np.array([]), np.array([]))


def test_binned_residual_summary_groups_target_ranges() -> None:
    rows = binned_residual_summary(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.5, 1.5, 1.5, 2.5]),
        bins=2,
    )

    assert len(rows) == 2
    assert rows[0]["examples"] == 2
    assert rows[0]["mae"] == 0.5


def test_binned_residual_summary_rejects_bad_bin_count() -> None:
    with pytest.raises(ValueError, match="bins"):
        binned_residual_summary(np.array([1.0]), np.array([1.0]), bins=0)


def test_empirical_interval_summary_uses_residual_quantile() -> None:
    summary = empirical_interval_summary(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 3.0, 1.0]),
        coverage=0.50,
    )

    assert summary["target_coverage"] == 0.50
    assert summary["interval_radius"] == pytest.approx(0.5)
    assert summary["observed_coverage"] == pytest.approx(0.5)


def test_empirical_interval_summary_rejects_bad_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        empirical_interval_summary(np.array([1.0]), np.array([1.0]), coverage=1.0)


def test_split_conformal_interval_uses_validation_residuals() -> None:
    summary = split_conformal_interval_summary(
        calibration_targets=np.array([0.0, 1.0, 2.0, 3.0]),
        calibration_predictions=np.array([0.0, 1.25, 2.5, 4.0]),
        test_targets=np.array([10.0, 20.0, 30.0]),
        test_predictions=np.array([10.2, 21.2, 30.7]),
        coverage=0.75,
    )

    assert summary["interval_radius"] == pytest.approx(1.0)
    assert summary["observed_coverage"] == pytest.approx(2 / 3)
    assert summary["calibration_examples"] == 4
    assert summary["test_examples"] == 3


def test_interval_calibration_curve_reports_each_requested_level() -> None:
    rows = interval_calibration_curve(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.2, 3.0]),
        np.array([0.0, 1.0]),
        np.array([0.1, 1.6]),
        coverages=(0.50, 0.90),
    )

    assert [row["target_coverage"] for row in rows] == [0.50, 0.90]


def test_split_conformal_interval_rejects_bad_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        split_conformal_interval_summary(
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            coverage=0.0,
        )


def test_feature_sensitivity_ranks_input_gradients() -> None:
    parameters = (
        {
            "weights": jnp.array([[2.0], [0.25]], dtype=jnp.float32),
            "bias": jnp.zeros((1,), dtype=jnp.float32),
        },
    )
    features = np.ones((3, 2), dtype=np.float32)
    rows = feature_sensitivity(parameters, features, ("strong", "weak"))

    assert rows[0]["feature"] == "strong"
    assert rows[0]["rank"] == 1
    assert rows[0]["mean_abs_gradient"] == pytest.approx(2.0)


def test_experiment_config_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig(learning_rate=0)


def test_experiment_config_rejects_invalid_gradient_clip() -> None:
    with pytest.raises(ValueError, match="gradient_clip"):
        ExperimentConfig(gradient_clip=0)


def test_experiment_config_rejects_invalid_split_sizes() -> None:
    with pytest.raises(ValueError, match="sum to less than 1"):
        ExperimentConfig(validation_size=0.5, test_size=0.5)
