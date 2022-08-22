from functools import partial
from typing import Tuple
import jax
from jax import numpy as jnp


def fit_background_single(spot_color: jnp.ndarray, weight_shift: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This determines the coefficient of the background vectors.
    Coefficients determined using a weighted dot product as to avoid over-fitting
    and accounting for the fact that background coefficients are not updated after this.

    !!! note
        `background_vectors[i]` is 1 in channel `i` for all rounds and 0 otherwise.
        It is then normalised to have L2 norm of 1 when summed over all rounds and channels.

    Args:
        spot_color: `float [n_rounds x n_channels]`.
            Spot color normalised to equalise intensities between channels (and rounds).
        weight_shift: shift to apply to weighting of each background vector to limit boost of weak spots.

    Returns:
        - residual - `float [n_rounds x n_channels]`.
            `spot_color` after background removed.
        - coefs - `float [n_channels]`.
            coefficient value for each background vector.
        - background_vectors `float [n_channels x n_rounds x n_channels]`.
            background_vectors[c] is the background vector for channel c.
    """
    n_rounds, n_channels = spot_color.shape
    background_vectors = jnp.repeat(jnp.expand_dims(jnp.eye(n_channels), axis=1), n_rounds, axis=1)
    # give background_vectors an L2 norm of 1 so can compare coefficients with other genes.
    background_vectors = background_vectors / jnp.linalg.norm(background_vectors, axis=(1, 2), keepdims=True)
    # array of correct shape containing the non-zero value of background_vectors everywhere.
    background_nz_value = jnp.full((n_rounds, n_channels), background_vectors[0, 0, 0])

    weight_factor = 1 / (jnp.abs(spot_color) + weight_shift)
    spot_weight = spot_color * weight_factor
    background_weight = background_nz_value * weight_factor
    coefs = jnp.sum(spot_weight * background_weight, axis=0) / jnp.sum(background_weight ** 2, axis=0)
    residual = spot_color - coefs * background_nz_value
    return residual, coefs, background_vectors


@partial(jax.jit, static_argnums=1)
def fit_background(spot_colors: jnp.ndarray, weight_shift: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This determines the coefficient of the background vectors for each spot.
    Coefficients determined using a weighted dot product as to avoid overfitting
    and accounting for the fact that background coefficients are not updated after this.

    !!! note
        `background_vectors[i]` is 1 in channel `i` for all rounds and 0 otherwise.
        It is then normalised to have L2 norm of 1 when summed over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        weight_shift: shift to apply to weighting of each background vector to limit boost of weak spots.

    Returns:
        - residual - `float [n_spots x n_rounds x n_channels]`.
            `spot_colors` after background removed.
        - coef - `float [n_spots, n_channels]`.
            coefficient value for each background vector found for each spot.
        - background_vectors `float [n_channels x n_rounds x n_channels]`.
            background_vectors[c] is the background vector for channel c.
    """
    return jax.vmap(fit_background_single, in_axes=(0, None), out_axes=(0, 0, None))(spot_colors, weight_shift)
