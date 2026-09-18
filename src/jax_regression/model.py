from __future__ import annotations

from numbers import Integral
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def init_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    key: jax.Array,
) -> tuple[dict[str, jax.Array], ...]:
    if not isinstance(input_dim, Integral) or input_dim < 1:
        raise ValueError("input_dim must be a positive integer")
    if not hidden_dims or any(
        not isinstance(width, Integral) or width < 1 for width in hidden_dims
    ):
        raise ValueError("hidden_dims must contain positive integer widths")
    layer_sizes = (input_dim, *hidden_dims, 1)
    keys = jax.random.split(key, len(layer_sizes) - 1)
    parameters = []
    for input_size, output_size, layer_key in zip(
        layer_sizes[:-1], layer_sizes[1:], keys
    ):
        scale = jnp.sqrt(2.0 / input_size)
        parameters.append(
            {
                "weights": jax.random.normal(
                    layer_key, (input_size, output_size)
                )
                * scale,
                "bias": jnp.zeros(output_size),
            }
        )
    return tuple(parameters)


def mlp_apply(
    parameters: tuple[dict[str, jax.Array], ...],
    features: jax.Array,
) -> jax.Array:
    activations = features
    for layer in parameters[:-1]:
        activations = jnp.tanh(activations @ layer["weights"] + layer["bias"])
    output_layer = parameters[-1]
    output = activations @ output_layer["weights"] + output_layer["bias"]
    return output[..., 0]


@jax.jit
def predict_batch(parameters, features):
    return jax.vmap(mlp_apply, in_axes=(None, 0))(parameters, features)


def parameter_count(parameters) -> int:
    return sum(int(np.prod(layer["weights"].shape) + np.prod(layer["bias"].shape)) for layer in parameters)


def save_parameters(
    parameters: tuple[dict[str, jax.Array], ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, layer in enumerate(parameters):
        arrays[f"layer_{index}_weights"] = np.asarray(layer["weights"])
        arrays[f"layer_{index}_bias"] = np.asarray(layer["bias"])
    np.savez(path, **arrays)


def load_parameters(path: Path) -> tuple[dict[str, jax.Array], ...]:
    with np.load(path, allow_pickle=False) as arrays:
        indices = sorted(
            int(key.removeprefix("layer_").removesuffix("_weights"))
            for key in arrays.files
            if key.startswith("layer_") and key.endswith("_weights")
        )
        if not indices or indices != list(range(len(indices))):
            raise ValueError("checkpoint must contain consecutive MLP layers")

        parameters = []
        previous_output = None
        for index in indices:
            weights_key = f"layer_{index}_weights"
            bias_key = f"layer_{index}_bias"
            if bias_key not in arrays.files:
                raise ValueError(f"checkpoint is missing {bias_key}")
            weights = np.asarray(arrays[weights_key])
            bias = np.asarray(arrays[bias_key])
            if weights.ndim != 2 or bias.ndim != 1:
                raise ValueError("checkpoint layers must contain matrix weights and vector bias")
            if weights.shape[1] != bias.shape[0]:
                raise ValueError(f"checkpoint layer {index} has incompatible weights and bias")
            if previous_output is not None and weights.shape[0] != previous_output:
                raise ValueError("checkpoint layer dimensions do not line up")
            if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
                raise ValueError("checkpoint parameters must contain only finite values")
            previous_output = weights.shape[1]
            parameters.append(
                {
                    "weights": jnp.asarray(weights),
                    "bias": jnp.asarray(bias),
                }
            )
        if parameters[-1]["weights"].shape[1] != 1:
            raise ValueError("checkpoint output layer must have one unit")
    return tuple(parameters)
