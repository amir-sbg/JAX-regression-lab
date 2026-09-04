from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    validation_size: float = 0.20
    test_size: float = 0.20
    hidden_dims: tuple[int, ...] = (64, 32)
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.90
    l2_penalty: float = 1e-4
    patience: int = 30
    gradient_clip: float | None = None
    warmup_epochs: int = 0
    final_learning_rate_ratio: float = 1.0
    ridge_alpha: float = 1.0
    permutation_repeats: int = 5
    output_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0 < self.validation_size < 1 or not 0 < self.test_size < 1:
            raise ValueError("validation_size and test_size must be between 0 and 1")
        if self.validation_size + self.test_size >= 1:
            raise ValueError("validation_size and test_size must sum to less than 1")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be between 0 and 1")
        if self.l2_penalty < 0 or self.ridge_alpha < 0:
            raise ValueError("regularization values must not be negative")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive when provided")
        if self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be non-negative and smaller than epochs")
        if not 0.0 < self.final_learning_rate_ratio <= 1.0:
            raise ValueError("final_learning_rate_ratio must be in (0, 1]")
        if self.permutation_repeats < 1:
            raise ValueError("permutation_repeats must be at least 1")


def prepare_output_directories(config: ExperimentConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
