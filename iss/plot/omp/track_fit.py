from typing import Optional, Tuple
import numpy as np
from ...setup import Notebook
from ...omp.coefs import get_all_coefs


def get_track_info(nb: Notebook, spot_no: int, method: str, dp_thresh: Optional[float] = None,
                   max_genes: Optional[int] = None) -> Tuple[dict, np.ndarray, float]:
    """
    This runs omp while tracking the residual at each stage.

    Args:
        nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        spot_no: Spot of interest to get track_info for.
        method: `'anchor'` or `'omp'`.
            Which method of gene assignment used i.e. `spot_no` belongs to `ref_spots` or `omp` page of Notebook.
        dp_thresh: If None, will use value in omp section of config file.
        max_genes: If None, will use value in omp section of config file.

    Returns:
        `track_info` - dictionary containing info about genes added at each step returned:

            - `background_codes` - `float [n_channels x n_rounds x n_channels]`.
                `background_codes[c]` is the background vector for channel `c` with L2 norm of 1.
            - `background_coefs` - `float [n_channels]`.
                `background_coefs[c]` is the coefficient value for `background_codes[c]`.
            - `gene_added` - `int [n_genes_added + 2]`.
                `gene_added[0]` and `gene_added[1]` are -1.
                `gene_added[2+i]` is the `ith` gene that was added.
            - `residual` - `float [(n_genes_added + 2) x n_rounds x n_channels]`.
                `residual[0]` is the initial `pixel_color`.
                `residual[1]` is the post background `pixel_color`.
                `residual[2+i]` is the `pixel_color` after removing gene `gene_added[2+i]`.
            - `coef` - `float [(n_genes_added + 2) x n_genes]`.
                `coef[0]` and `coef[1]` are all 0.
                `coef[2+i]` are the coefficients for all genes after the ith gene has been added.
            - `dot_product` - `float [n_genes_added + 2]`.
                `dot_product[0]` and `dot_product[1]` are 0.
                `dot_product[2+i]` is the dot product for the gene `gene_added[2+i]`.
            - `inverse_var` - `float [(n_genes_added + 2) x n_rounds x n_channels]`.
                `inverse_var[0]` and `inverse_var[1]` are all 0.
                `inverse_var[2+i]` is the weighting used to compute `dot_product[2+i]`,
                 which down-weights rounds/channels for which a gene has already been fitted.
        `bled_codes` - `float [n_genes x n_use_rounds x n_use_channels]`.
            gene `bled_codes` used in omp with L2 norm = 1.
        `dp_thresh` - threshold dot product score, above which gene is fitted.
    """
    color_norm = nb.call_spots.color_norm_factor[np.ix_(nb.basic_info.use_rounds,
                                                        nb.basic_info.use_channels)]
    n_use_rounds, n_use_channels = color_norm.shape
    if method.lower() == 'omp':
        page_name = 'omp'
        config_name = 'omp'
    else:
        page_name = 'ref_spots'
        config_name = 'call_spots'
    spot_color = nb.__getattribute__(page_name).colors[spot_no][
                     np.ix_(nb.basic_info.use_rounds, nb.basic_info.use_channels)] / color_norm
    n_genes = nb.call_spots.bled_codes_ge.shape[0]
    bled_codes = np.asarray(
        nb.call_spots.bled_codes_ge[np.ix_(np.arange(n_genes),
                                           nb.basic_info.use_rounds, nb.basic_info.use_channels)])
    # ensure L2 norm is 1 for bled codes
    norm_factor = np.expand_dims(np.linalg.norm(bled_codes, axis=(1, 2)), (1, 2))
    norm_factor[norm_factor == 0] = 1  # For genes with no dye in use_dye, this avoids blow up on next line
    bled_codes = bled_codes / norm_factor

    # Get info to run omp
    dp_norm_shift = nb.call_spots.dp_norm_shift * np.sqrt(n_use_rounds)
    config = nb.get_config()
    if dp_thresh is None:
        dp_thresh = config['omp']['dp_thresh']
    alpha = config[config_name]['alpha']
    beta = config[config_name]['beta']
    if max_genes is None:
        max_genes = config['omp']['max_genes']
    weight_coef_fit = config['omp']['weight_coef_fit']

    # Run omp with track to get residual at each stage
    track_info = get_all_coefs(spot_color[np.newaxis], bled_codes, nb.call_spots.background_weight_shift,
                               dp_norm_shift, dp_thresh, alpha, beta, max_genes, weight_coef_fit, True)[2]
    return track_info, bled_codes, dp_thresh
