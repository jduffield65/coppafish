import jax
from jax import numpy as jnp


@jax.jit
def get_spot_intensity(spot_colors: jnp.ndarray) -> jnp.ndarray:
    """
    Finds the max intensity for each imaging round across all imaging channels for each spot.
    Then median of these max round intensities is returned.
    Logic is that we expect spots that are genes to have at least one large intensity value in each round
    so high spot intensity is more indicative of a gene.

    Args:
        spot_colors: ```float [n_spots x n_rounds x n_channels]```.
            Spot colors normalised to equalise intensities between channels (and rounds).

    Returns:
        ```float [n_spots]```.
            ```[s]``` is the intensity of spot ```s```.
    """
    return jax.vmap(lambda x: jnp.median(jnp.max(x, axis=1)), in_axes=0, out_axes=0)(spot_colors)
