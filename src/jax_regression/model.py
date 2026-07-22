from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def init_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    key: jax.Array,
) -> tuple[dict[str, jax.Array], ...]:
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
