from functools import partial
import jax
import jax.numpy as jnp


def dot_product_score_single(spot_colours: jnp.ndarray, bled_codes: jnp.ndarray, norm_shift: float,
                             weight_squared: jnp.ndarray) -> jnp.ndarray:
    n_genes, n_round_channels = bled_codes.shape
    spot_colours = spot_colours / (jnp.linalg.norm(spot_colours) + norm_shift)
    spot_colours = spot_colours * weight_squared
    score = spot_colours @ bled_codes.transpose()
    score = n_round_channels * score / jnp.sum(weight_squared)
    return score


@partial(jax.jit, static_argnums=2)
def dot_product_score(spot_colours: jnp.ndarray, bled_codes: jnp.ndarray, norm_shift: float = 0,
                      weight_squared: jnp.ndarray = None) -> jnp.ndarray:
    """
    Computes `sum(W**2(s * b) / W**2)` where `s` is a `spot_color`, `b` is a `bled_code` and `W**2` is weight_squared
    for a particular `spot_color`. Sum is over all rounds and channels.

    Args:
        spot_colours (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`): spot colours.
        bled_codes (`[n_genes x (n_rounds * n_channels_use)] ndarray[float]`): normalised bled codes.
        norm_shift (float, optional): added to the norm of each spot colour to avoid boosting weak spots too much. 
            Default: 0.
        weight_squared (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`, optional): array of weights. Default: 
            all ones, i.e. no effect.

    Returns:
        `[n_spots x n_genes] ndarray[float]`: `score` such that `score[d, c]` gives dot product between `spot_colours` 
            vector `d` with `bled_codes` vector `c`.
    """
    n_spots, n_genes = spot_colours.shape[0], bled_codes.shape[0]
    n_rounds_channels_use = spot_colours.shape[1]
    # If no weighting is given, use equal weighting
    if weight_squared is None:
        weight_squared = jnp.ones((n_spots, n_rounds_channels_use))
    # Normalise `bled_codes` outside of the for loop, since it does not loop over n_spots
    bled_codes = bled_codes / jnp.linalg.norm(bled_codes, axis=1, keepdims=True)
    score = jax.vmap(dot_product_score_single, in_axes=(0, None, None, 0), out_axes=0)(spot_colours,
                                                                                       bled_codes, 
                                                                                       norm_shift, 
                                                                                       weight_squared)
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