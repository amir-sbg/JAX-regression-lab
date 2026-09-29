# JAX Regression Lab

A small end-to-end regression experiment built with JAX. It compares a closed-form ridge baseline with an MLP trained from first principles on the scikit-learn diabetes dataset.

The project keeps the model intentionally small and makes the learning mechanics visible: JAX PyTrees hold parameters and optimizer state, `jax.value_and_grad` computes updates, `jax.jit` compiles the step function, and `jax.vmap` handles batched prediction. The pipeline also checks preprocessing, validation performance, residual behavior, feature sensitivity, and local curvature.

## Included

- deterministic train/validation/test splits with train-only standardization
- ridge regression solved with `jax.numpy.linalg.solve`
- MLP training with momentum, L2 regularization, early stopping, optional warmup, cosine decay, gradient clipping, MSE, or Huber loss
- JIT-compiled updates and vectorized prediction
- RMSE, MAE, R², residual, convergence, and model-comparison reports
- conformal interval checks, permutation importance, feature collinearity, input sensitivity, and Hessian-vector curvature probes
- `.npz` parameter checkpoints with shape and finite-value validation

The dataset has 442 samples, 10 numeric features, and a continuous target. Metrics and plots are reported on the original target scale.

## Setup

```bash
git clone https://github.com/amir-sbg/JAX-regression-lab.git
cd JAX-regression-lab
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

JAX selects the available backend at runtime. CPU is sufficient for the default experiment; an accelerator can be used when available.

## Run

```bash
python -m jax_regression.pipeline
```

The main settings are configurable from the command line:

```bash
python -m jax_regression.pipeline \
  --validation-size 0.20 \
  --test-size 0.20 \
  --epochs 300 \
  --batch-size 32 \
  --hidden-dims 64 32 \
  --learning-rate 0.01 \
  --gradient-clip 5.0 \
  --loss huber \
  --permutation-repeats 5 \
  --curvature-probes 4
```

## Outputs

The run writes model artifacts to `artifacts/` and reports to `reports/`, including:

- `metrics.json` and `model_comparison.json` for ridge/MLP test performance
- `training_convergence.json` and `training_history.png` for loss, learning-rate, and gradient behavior
- residual summaries and target-range bins
- conformal interval and calibration reports
- feature sensitivity, feature-matrix conditioning, permutation importance, and curvature diagnostics
- `mlp_parameters.npz` and `ridge_parameters.npy` for saved model parameters

## Project layout

```text
src/jax_regression/
├── config.py       experiment settings and validation
├── data.py         dataset loading and scaling
├── baseline.py     closed-form ridge regression
├── model.py        MLP, prediction, and checkpointing
├── train.py        losses, gradients, JIT update, and early stopping
├── evaluate.py     metrics, diagnostics, and plots
└── pipeline.py     command-line experiment
tests/test_pipeline.py
```
