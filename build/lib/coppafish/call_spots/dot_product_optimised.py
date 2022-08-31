from functools import partial
import jax
from jax import numpy as jnp


def dot_product_score_single(spot_colors: jnp.ndarray, bled_codes: jnp.ndarray, norm_shift: float,
                             weight_squared: jnp.ndarray) -> jnp.ndarray:
    n_genes, n_round_channels = bled_codes.shape
    spot_colors = spot_colors / (jnp.linalg.norm(spot_colors) + norm_shift)
    bled_codes = bled_codes / jnp.linalg.norm(bled_codes, axis=1, keepdims=True)
    spot_colors = spot_colors * weight_squared
    score = spot_colors @ bled_codes.transpose()
    score = score / jnp.sum(weight_squared) * n_round_channels
    return score


@partial(jax.jit, static_argnums=2)
def dot_product_score(spot_colors: jnp.ndarray, bled_codes: jnp.ndarray, norm_shift: float,
                      weight_squared: jnp.ndarray) -> jnp.ndarray:
    """
    Computes `sum(W**2(s * b) / W**2)` where `s` is a `spot_color`, `b` is a `bled_code` and `W**2` is weight_squared
    for a particular `spot_color`. Sum is over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x (n_rounds x n_channels)]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        weight_squared: `float [n_spots x (n_rounds x n_channels)]`.
            squared weight to apply to each round/channel for each spot when computing dot product.

    Returns:
        `float [n_spots x n_genes]`.
            `score` such that `score[d, c]` gives dot product between `spot_colors` vector `d`
            with `bled_codes` vector `c`.
    """
    score = jax.vmap(dot_product_score_single, in_axes=(0, None, None, 0), out_axes=0)(spot_colors, bled_codes,
                                                                                       norm_shift, weight_squared)
    return score


def dot_product_score_no_weight_single(spot_colors: jnp.ndarray, bled_codes: jnp.ndarray,
                                       norm_shift: float) -> jnp.ndarray:
    spot_colors = spot_colors / (jnp.linalg.norm(spot_colors) + norm_shift)
    bled_codes = bled_codes / jnp.linalg.norm(bled_codes, axis=1, keepdims=True)
    return spot_colors @ bled_codes.transpose()


@partial(jax.jit, static_argnums=2)
def dot_product_score_no_weight(spot_colors: jnp.ndarray, bled_codes: jnp.ndarray, norm_shift: float) -> jnp.ndarray:
    """
    Computes `sum((s * b))` where `s` is a `spot_color`, `b` is a `bled_code`.
    Sum is over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x (n_rounds x n_channels)]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.

    Returns:
        `float [n_spots x n_genes]`.
            `score` such that `score[d, c]` gives dot product between `spot_colors` vector `d`
            with `bled_codes` vector `c`.
    """
    score = jax.vmap(dot_product_score_no_weight_single, in_axes=(0, None, None), out_axes=0)(spot_colors, bled_codes,
                                                                                              norm_shift)
    return score