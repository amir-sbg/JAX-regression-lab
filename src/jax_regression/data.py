from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegressionData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    target_mean: float
    target_scale: float

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_scale + self.target_mean


def load_regression_data(
    seed: int = 42,
    validation_size: float = 0.20,
    test_size: float = 0.20,
) -> RegressionData:
    if not 0 < validation_size < 1 or not 0 < test_size < 1:
        raise ValueError("validation_size and test_size must be between 0 and 1")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must sum to less than 1")

    dataset = load_diabetes(as_frame=False)
    features = dataset.data.astype(np.float32)
    targets = dataset.target.astype(np.float32)
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        targets,
        test_size=validation_size + test_size,
        random_state=seed,
    )
    test_fraction = test_size / (validation_size + test_size)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=test_fraction,
        random_state=seed,
    )

    feature_scaler = StandardScaler().fit(x_train)
    target_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    return RegressionData(
        x_train=feature_scaler.transform(x_train).astype(np.float32),
        y_train=target_scaler.transform(y_train.reshape(-1, 1)).ravel().astype(np.float32),
        x_validation=feature_scaler.transform(x_validation).astype(np.float32),
        y_validation=target_scaler.transform(y_validation.reshape(-1, 1)).ravel().astype(np.float32),
        x_test=feature_scaler.transform(x_test).astype(np.float32),
        y_test=target_scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32),
        feature_names=tuple(dataset.feature_names),
        feature_mean=tuple(float(value) for value in feature_scaler.mean_),
        feature_scale=tuple(float(value) for value in feature_scaler.scale_),
        target_mean=float(target_scaler.mean_[0]),
        target_scale=float(target_scaler.scale_[0]),
    )
