# JAX Regression Lab

A small regression project built around JAX. It compares a closed-form ridge-regression baseline with a multilayer perceptron trained from first principles on the scikit-learn diabetes dataset.

## What is included

The pipeline covers:

- deterministic train, validation, and test splits
- feature and target standardization fitted on training data only
- a ridge solution implemented with `jax.numpy.linalg.solve`
- a fully connected MLP represented as a JAX parameter PyTree
- automatic differentiation with `jax.value_and_grad`
- JIT-compiled momentum updates with `jax.jit`
- batched prediction with `jax.vmap`
- validation-based early stopping and a held-out test report
- residual diagnostics for checking bias and error spread

The data contains 442 samples, 10 numeric features, and a continuous disease-progression target. Predictions and error metrics are reported in the original target scale.

## Mathematical setup

The baseline solves the ridge objective:

```text
min_w  ||Xw - y||² + α||w||²
```

using the normal-equation system with an unregularized bias term. The neural model uses two `tanh` hidden layers and minimizes mean squared error with an L2 penalty on the weight matrices.

The training update is momentum gradient descent:

```text
vₜ = μvₜ₋₁ + ∇L(θₜ₋₁)
θₜ = θₜ₋₁ - ηvₜ
```

The update is expressed as a pure function over nested parameter and optimizer-state trees. JAX handles differentiation, compilation, and device placement without an external deep-learning training framework.

## Run the project

```bash
git clone https://github.com/amir-sbg/JAX.git
cd JAX

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Run the default experiment:

```bash
python -m jax_regression.pipeline
```

Training options can be changed from the command line:

```bash
python -m jax_regression.pipeline \
  --epochs 300 \
  --batch-size 32 \
  --hidden-dims 64 32 \
  --learning-rate 0.01 \
  --momentum 0.90 \
  --patience 30
```

JAX uses the available backend at runtime. The selected backend, device list, split sizes, preprocessing parameters, model size, and metrics are saved with the run.

## Outputs

```text
artifacts/
├── mlp_parameters.npz
├── ridge_parameters.npy
└── training_history.csv

reports/
├── metrics.json
├── run_config.json
├── run_summary.json
├── residual_summary.json
├── residuals.csv
├── residuals.png
└── training_history.png
```

`metrics.json` reports MSE, RMSE, MAE, and R² for both the ridge baseline and the JAX MLP. The residual report keeps per-sample errors and summary statistics in the original target scale, which makes it easier to see whether the neural model is biased high or low on the held-out set.

## Project structure

```text
.
├── src/jax_regression/
│   ├── config.py       # experiment settings and validation
│   ├── data.py         # dataset loading and scaling
│   ├── baseline.py     # closed-form ridge regression
│   ├── model.py        # MLP parameters, vmap prediction, serialization
│   ├── train.py        # loss, gradients, JIT update, early stopping
│   ├── evaluate.py     # regression metrics and plots
│   └── pipeline.py     # command-line experiment
├── tests/test_pipeline.py
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```
