from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_regression.baseline import fit_ridge, predict_ridge
from jax_regression.config import ExperimentConfig
from jax_regression.data import load_regression_data
from jax_regression.diagnostics import (
    directional_curvature,
    feature_correlation_pairs,
    feature_sensitivity,
    permutation_importance,
    random_parameter_direction,
    tree_dot,
)
from jax_regression.evaluate import (
    binned_residual_summary,
    empirical_interval_summary,
    interval_calibration_curve,
    model_comparison_summary,
    regression_metrics,
    residual_summary,
    split_conformal_interval_summary,
    training_convergence_summary,
)
from jax_regression.model import (
    init_mlp,
    load_parameters,
    parameter_count,
    predict_batch,
    save_parameters,
)
from jax_regression.train import (
    TrainingConfig,
    clip_gradients,
    learning_rate_for_epoch,
    regression_loss,
    train_model,
    tree_l2_norm,
)


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


def test_mlp_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        init_mlp(0, (4,), jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="hidden_dims"):
        init_mlp(2, (), jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="hidden_dims"):
        init_mlp(2, (0,), jax.random.PRNGKey(0))


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


def test_load_parameters_rejects_incompatible_layers(tmp_path) -> None:
    checkpoint = tmp_path / "bad.npz"
    np.savez(
        checkpoint,
        layer_0_weights=np.ones((3, 4), dtype=np.float32),
        layer_0_bias=np.ones(4, dtype=np.float32),
        layer_1_weights=np.ones((3, 1), dtype=np.float32),
        layer_1_bias=np.ones(1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="dimensions do not line up"):
        load_parameters(checkpoint)


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
    assert "learning_rate" in result.history[-1]


def test_training_rejects_bad_arrays_before_jit() -> None:
    parameters = init_mlp(1, (4,), jax.random.PRNGKey(3))
    with pytest.raises(ValueError, match="same rows"):
        train_model(
            parameters,
            np.ones((3, 1), dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
            np.ones(2, dtype=np.float32),
            TrainingConfig(epochs=2),
            jax.random.PRNGKey(4),
        )
    invalid = np.ones((2, 1), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        train_model(
            parameters,
            invalid,
            np.ones(2, dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
            np.ones(2, dtype=np.float32),
            TrainingConfig(epochs=2),
            jax.random.PRNGKey(4),
        )


def test_huber_loss_downweights_large_residuals() -> None:
    parameters = (
        {
            "weights": jnp.array([[0.0]], dtype=jnp.float32),
            "bias": jnp.zeros((1,), dtype=jnp.float32),
        },
    )
    features = np.ones((2, 1), dtype=np.float32)
    targets = np.array([0.5, 5.0], dtype=np.float32)

    mse = regression_loss(parameters, features, targets, loss="mse")
    huber = regression_loss(parameters, features, targets, loss="huber", huber_delta=1.0)

    assert float(huber) < float(mse)
    assert float(huber) == pytest.approx(2.3125)


def test_learning_rate_schedule_warms_up_and_decays() -> None:
    config = TrainingConfig(
        epochs=6,
        learning_rate=0.1,
        warmup_epochs=2,
        final_learning_rate_ratio=0.1,
    )

    values = [learning_rate_for_epoch(epoch, config) for epoch in range(1, 7)]

    assert values[0] == pytest.approx(0.05)
    assert values[1] == pytest.approx(0.10)
    assert values[-1] == pytest.approx(0.01)
    assert values[2] > values[3] > values[4] > values[5]


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


def test_model_comparison_summary_ranks_by_primary_metric() -> None:
    summary = model_comparison_summary(
        {
            "ridge": {"rmse": 4.0, "mae": 3.0},
            "mlp": {"rmse": 3.0, "mae": 2.5},
        }
    )

    assert summary["winner"] == "mlp"
    assert summary["mlp_delta_vs_ridge"] == pytest.approx(-1.0)
    assert summary["mlp_relative_change_vs_ridge"] == pytest.approx(-0.25)


def test_model_comparison_summary_requires_primary_metric() -> None:
    with pytest.raises(ValueError, match="missing rmse"):
        model_comparison_summary({"ridge": {"mae": 1.0}})


def test_residual_summary_reports_error_shape() -> None:
    summary = residual_summary(np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 4.0]))

    assert summary["mean_residual"] == pytest.approx(1 / 3)
    assert summary["max_abs_residual"] == 1.0


def test_residual_summary_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        residual_summary(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="must not be empty"):
        residual_summary(np.array([]), np.array([]))


def test_training_convergence_summary_tracks_best_epoch() -> None:
    summary = training_convergence_summary(
        [
            {
                "epoch": 1.0,
                "train_loss": 4.0,
                "validation_loss": 5.0,
                "gradient_norm": 3.0,
                "learning_rate": 0.1,
            },
            {
                "epoch": 2.0,
                "train_loss": 2.0,
                "validation_loss": 2.5,
                "gradient_norm": 1.0,
                "learning_rate": 0.05,
            },
            {
                "epoch": 3.0,
                "train_loss": 1.5,
                "validation_loss": 2.8,
                "gradient_norm": 0.5,
                "learning_rate": 0.01,
            },
        ]
    )

    assert summary["epochs_ran"] == 3
    assert summary["best_epoch"] == 2
    assert summary["train_loss_reduction"] == pytest.approx(0.625)
    assert summary["mean_gradient_norm"] == pytest.approx(1.5)
    assert summary["max_gradient_norm"] == pytest.approx(3.0)
    assert summary["final_gradient_norm"] == pytest.approx(0.5)


def test_training_convergence_summary_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="at least one epoch"):
        training_convergence_summary([])


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


def test_feature_sensitivity_rejects_bad_feature_matrix() -> None:
    parameters = (
        {
            "weights": jnp.array([[1.0]], dtype=jnp.float32),
            "bias": jnp.zeros((1,), dtype=jnp.float32),
        },
    )

    with pytest.raises(ValueError, match="two-dimensional"):
        feature_sensitivity(parameters, np.array([1.0, 2.0]), ("x",))
    with pytest.raises(ValueError, match="at least one row"):
        feature_sensitivity(parameters, np.empty((0, 1), dtype=np.float32), ("x",))
    with pytest.raises(ValueError, match="finite"):
        feature_sensitivity(parameters, np.array([[np.nan]], dtype=np.float32), ("x",))


def test_feature_correlation_pairs_rank_strong_relationships() -> None:
    features = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 0.0],
            [3.0, 6.0, -1.0],
        ],
        dtype=np.float32,
    )

    pairs = feature_correlation_pairs(features, ("a", "b", "c"), top_k=2)

    assert pairs[0]["feature_a"] == "a"
    assert pairs[0]["feature_b"] == "b"
    assert pairs[0]["correlation"] == pytest.approx(1.0)
    assert len(pairs) == 2


def test_permutation_importance_ranks_predictive_feature() -> None:
    features = np.array(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 2.0],
        ],
        dtype=np.float32,
    )
    targets = 2.0 * features[:, 0] + 0.1 * features[:, 1]

    rows = permutation_importance(
        features,
        targets,
        lambda batch: 2.0 * batch[:, 0] + 0.1 * batch[:, 1],
        ("strong", "weak"),
        repeats=4,
        seed=3,
    )

    assert rows[0]["feature"] == "strong"
    assert rows[0]["rank"] == 1
    assert rows[0]["mean_mse_increase"] > rows[1]["mean_mse_increase"]
    assert sum(row["normalized_importance"] for row in rows) == pytest.approx(1.0)


def test_permutation_importance_rejects_bad_predictor_output() -> None:
    features = np.ones((3, 2), dtype=np.float32)
    targets = np.ones(3, dtype=np.float32)

    with pytest.raises(ValueError, match="one prediction"):
        permutation_importance(
            features,
            targets,
            lambda batch: np.ones((len(batch), 1), dtype=np.float32),
            ("x", "y"),
        )


def test_random_parameter_direction_is_unit_norm() -> None:
    parameters = init_mlp(2, (4,), jax.random.PRNGKey(10))
    direction = random_parameter_direction(parameters, jax.random.PRNGKey(11))

    assert float(tree_l2_norm(direction)) == pytest.approx(1.0)
    assert tree_dot(direction, direction) == pytest.approx(1.0)


def test_directional_curvature_reports_finite_values() -> None:
    parameters = init_mlp(2, (4,), jax.random.PRNGKey(12))
    features = np.ones((5, 2), dtype=np.float32)
    targets = np.linspace(-1, 1, 5, dtype=np.float32)

    report = directional_curvature(
        parameters,
        features,
        targets,
        jax.random.PRNGKey(13),
        probes=3,
    )

    assert report["probes"] == 3
    assert np.isfinite(report["mean_directional_curvature"])


def test_experiment_config_rejects_invalid_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig(learning_rate=0)


def test_experiment_config_rejects_invalid_gradient_clip() -> None:
    with pytest.raises(ValueError, match="gradient_clip"):
        ExperimentConfig(gradient_clip=0)


def test_experiment_config_rejects_bad_schedule() -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        ExperimentConfig(epochs=4, warmup_epochs=4)
    with pytest.raises(ValueError, match="final_learning_rate_ratio"):
        ExperimentConfig(final_learning_rate_ratio=1.5)


def test_experiment_config_rejects_bad_loss_settings() -> None:
    with pytest.raises(ValueError, match="loss"):
        ExperimentConfig(loss="mae")
    with pytest.raises(ValueError, match="huber_delta"):
        ExperimentConfig(loss="huber", huber_delta=0)


def test_experiment_config_rejects_invalid_split_sizes() -> None:
    with pytest.raises(ValueError, match="sum to less than 1"):
        ExperimentConfig(validation_size=0.5, test_size=0.5)


def test_experiment_config_rejects_bad_permutation_repeats() -> None:
    with pytest.raises(ValueError, match="permutation_repeats"):
        ExperimentConfig(permutation_repeats=0)


def test_experiment_config_rejects_bad_curvature_probe_count() -> None:
    with pytest.raises(ValueError, match="curvature_probes"):
        ExperimentConfig(curvature_probes=0)
