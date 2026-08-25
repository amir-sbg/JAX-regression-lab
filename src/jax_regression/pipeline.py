from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from .baseline import fit_ridge, predict_ridge
from .config import ExperimentConfig, prepare_output_directories
from .data import load_regression_data
from .diagnostics import feature_sensitivity
from .evaluate import (
    binned_residual_summary,
    empirical_interval_summary,
    interval_calibration_curve,
    regression_metrics,
    residual_summary,
    save_interval_calibration_plot,
    save_json,
    save_residual_plot,
    save_training_plot,
    split_conformal_interval_summary,
)
from .model import init_mlp, parameter_count, predict_batch, save_parameters
from .train import TrainingConfig, train_model


def _config_payload(config: ExperimentConfig) -> dict:
    payload = asdict(config)
    payload["hidden_dims"] = list(config.hidden_dims)
    payload["output_dir"] = str(config.output_dir)
    payload["report_dir"] = str(config.report_dir)
    return payload


def run(config: ExperimentConfig) -> dict:
    prepare_output_directories(config)
    data = load_regression_data(
        seed=config.seed,
        validation_size=config.validation_size,
        test_size=config.test_size,
    )
    key = jax.random.PRNGKey(config.seed)
    model_key, training_key = jax.random.split(key)

    ridge_parameters = fit_ridge(
        data.x_train,
        data.y_train,
        alpha=config.ridge_alpha,
    )
    ridge_validation_predictions = np.asarray(
        predict_ridge(ridge_parameters, data.x_validation)
    )
    ridge_predictions = np.asarray(predict_ridge(ridge_parameters, data.x_test))

    parameters = init_mlp(
        input_dim=data.x_train.shape[1],
        hidden_dims=config.hidden_dims,
        key=model_key,
    )
    training = train_model(
        parameters=parameters,
        features=data.x_train,
        targets=data.y_train,
        validation_features=data.x_validation,
        validation_targets=data.y_validation,
        config=TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            l2_penalty=config.l2_penalty,
            patience=config.patience,
            gradient_clip=config.gradient_clip,
        ),
        key=training_key,
    )
    mlp_validation_predictions = np.asarray(
        predict_batch(training.parameters, data.x_validation)
    )
    mlp_predictions = np.asarray(predict_batch(training.parameters, data.x_test))

    validation_targets = data.inverse_target(data.y_validation)
    ridge_validation_predictions = data.inverse_target(ridge_validation_predictions)
    mlp_validation_predictions = data.inverse_target(mlp_validation_predictions)
    actual_targets = data.inverse_target(data.y_test)
    ridge_predictions = data.inverse_target(ridge_predictions)
    mlp_predictions = data.inverse_target(mlp_predictions)
    metrics = {
        "ridge": regression_metrics(actual_targets, ridge_predictions),
        "mlp": regression_metrics(actual_targets, mlp_predictions),
    }
    residuals = pd.DataFrame(
        {
            "actual": actual_targets,
            "ridge_prediction": ridge_predictions,
            "ridge_residual": ridge_predictions - actual_targets,
            "mlp_prediction": mlp_predictions,
            "mlp_residual": mlp_predictions - actual_targets,
        }
    )
    residual_report = {
        "ridge": residual_summary(actual_targets, ridge_predictions),
        "mlp": residual_summary(actual_targets, mlp_predictions),
    }
    residual_bins = {
        "ridge": binned_residual_summary(actual_targets, ridge_predictions),
        "mlp": binned_residual_summary(actual_targets, mlp_predictions),
    }
    interval_report = {
        "ridge": empirical_interval_summary(actual_targets, ridge_predictions),
        "mlp": empirical_interval_summary(actual_targets, mlp_predictions),
    }
    conformal_interval_report = {
        "ridge": split_conformal_interval_summary(
            validation_targets,
            ridge_validation_predictions,
            actual_targets,
            ridge_predictions,
        ),
        "mlp": split_conformal_interval_summary(
            validation_targets,
            mlp_validation_predictions,
            actual_targets,
            mlp_predictions,
        ),
    }
    interval_calibration = {
        "ridge": interval_calibration_curve(
            validation_targets,
            ridge_validation_predictions,
            actual_targets,
            ridge_predictions,
        ),
        "mlp": interval_calibration_curve(
            validation_targets,
            mlp_validation_predictions,
            actual_targets,
            mlp_predictions,
        ),
    }
    sensitivity_report = {
        "description": (
            "Mean input gradients for the trained MLP on standardized test features. "
            "Values are in the scaled target space used during optimization."
        ),
        "features": feature_sensitivity(
            training.parameters,
            data.x_test,
            data.feature_names,
        ),
    }

    save_parameters(training.parameters, config.output_dir / "mlp_parameters.npz")
    np.save(config.output_dir / "ridge_parameters.npy", np.asarray(ridge_parameters))
    pd.DataFrame(training.history).to_csv(
        config.output_dir / "training_history.csv",
        index=False,
    )
    residuals.to_csv(config.report_dir / "residuals.csv", index=False)
    save_training_plot(training.history, config.report_dir / "training_history.png")
    save_residual_plot(actual_targets, mlp_predictions, config.report_dir / "residuals.png")
    save_interval_calibration_plot(
        interval_calibration,
        config.report_dir / "interval_calibration.png",
    )
    save_json(metrics, config.report_dir / "metrics.json")
    save_json(residual_report, config.report_dir / "residual_summary.json")
    save_json(residual_bins, config.report_dir / "residual_bins.json")
    save_json(interval_report, config.report_dir / "interval_summary.json")
    save_json(conformal_interval_report, config.report_dir / "conformal_intervals.json")
    save_json(interval_calibration, config.report_dir / "interval_calibration.json")
    save_json(sensitivity_report, config.report_dir / "feature_sensitivity.json")
    save_json(
        {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "samples": {
                "train": len(data.x_train),
                "validation": len(data.x_validation),
                "test": len(data.x_test),
            },
            "feature_names": list(data.feature_names),
            "feature_mean": list(data.feature_mean),
            "feature_scale": list(data.feature_scale),
            "target_mean": data.target_mean,
            "target_scale": data.target_scale,
            "model": {
                "hidden_dims": list(config.hidden_dims),
                "parameter_count": parameter_count(training.parameters),
                "best_epoch": training.best_epoch,
                "best_validation_loss": training.best_validation_loss,
            },
            "metrics": metrics,
            "residuals": residual_report,
            "residual_bins": residual_bins,
            "intervals": interval_report,
            "conformal_intervals": conformal_interval_report,
            "top_feature_sensitivity": sensitivity_report["features"][:5],
        },
        config.report_dir / "run_summary.json",
    )
    save_json(_config_payload(config), config.report_dir / "run_config.json")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the JAX regression pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.90)
    parser.add_argument("--l2-penalty", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    values = vars(args).copy()
    values["hidden_dims"] = tuple(values["hidden_dims"])
    return ExperimentConfig(**values)


if __name__ == "__main__":
    run(config_from_args(build_parser().parse_args()))
