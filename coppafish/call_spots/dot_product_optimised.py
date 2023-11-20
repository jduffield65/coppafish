from functools import partial
import jax
import jax.numpy as jnp


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
    
    # Ensure bled_codes is normalised for each gene
    bled_codes = bled_codes / jnp.linalg.norm(bled_codes, axis=1, keepdims=True)
    weight_squared = weight_squared / jnp.sum(weight_squared, axis=1)[:, None]
    spot_colours = spot_colours / (jnp.linalg.norm(spot_colours, axis=1)[:, None] + norm_shift)
    spot_colours = n_rounds_channels_use * spot_colours * weight_squared

    # Now we can obtain the dot product score for each spot and each gene
    all_scores = spot_colours @ bled_codes.T
    return all_scores
