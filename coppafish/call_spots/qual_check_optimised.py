import jax
from jax import numpy as jnp


@jax.jit
def get_spot_intensity(spot_colors: jnp.ndarray) -> jnp.ndarray:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.

    Args:
        spot_colors (`[n_spots x n_rounds x n_channels] ndarray[float]`: spot colors normalised to equalise intensities 
            between channels (and rounds).

    Returns:
        `[n_spots] ndarray[float]`: index `s` is the intensity of spot `s`.

    Notes:
        Logic is that we expect spots that are genes to have at least one large intensity value in each round
        so high spot intensity is more indicative of a gene.
    """
    # Max over all channels, then median over all rounds
    return jnp.median(jnp.max(spot_colors, axis=2), axis=1)
