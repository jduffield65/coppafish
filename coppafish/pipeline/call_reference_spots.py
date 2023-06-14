from ..call_spots import get_bled_codes, compute_bleed_matrix, compute_gene_efficiency, compute_gene_scores, \
    get_spot_intensity
from ..spot_colors import remove_background, normalise_rc, all_pixel_yxz, get_spot_colors
from ..setup.notebook import NotebookPage
from ..utils.base import expand_channels
import numpy as np
from typing import Tuple
from tqdm import tqdm
import warnings
import os


def call_reference_spots(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage,
                         nbp_ref_spots: NotebookPage, initial_bleed_matrix: np.ndarray,
                         transform: np.ndarray, overwrite_ref_spots: bool = False) -> Tuple[NotebookPage, NotebookPage]:
    """
    This produces the bleed matrix and expected code for each gene as well as producing a gene assignment based on a
    simple dot product for spots found on the reference round.

    Returns the `call_spots` notebook page and adds the following variables to the `ref_spots` page:
    `gene_no`, `score`, `score_diff`, `intensity`.

    See `'call_spots'` and `'ref_spots'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        config: Dictionary obtained from `'call_spots'` section of config file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        nbp_ref_spots: `ref_spots` notebook page containing all variables produced in `pipeline/reference_spots.py` i.e.
            `local_yxz`, `isolated`, `tile`, `colors`.
            `gene_no`, `score`, `score_diff`, `intensity` should all be `None` to add them here, unless
            `overwrite_ref_spots == True`.
        initial_bleed_matrix: float [n_channels x n_dyes] initial bleed matrix guess
        transform: float [n_tiles x n_rounds x n_channels x 4 x 3] affine transform for each tile, round and channel
        overwrite_ref_spots: If `True`, the variables:
            * `gene_no`
            * `score`
            * `score_diff`
            * `intensity`

            in `nbp_ref_spots` will be overwritten if they exist. If this is `False`, they will only be overwritten
            if they are all set to `None`, otherwise an error will occur.

    Returns:
        `NotebookPage[call_spots]` - Page contains bleed matrix and expected code for each gene.
        `NotebookPage[ref_spots]` - Page contains gene assignments and info for spots found on reference round.
            Parameters added are: intensity, score, gene_no, score_diff
    """
    if overwrite_ref_spots:
        warnings.warn("\noverwrite_ref_spots = True so will overwrite:\ngene_no, gene_score, score_diff, intensity,"
                      "\nbackground_strength in nbp_ref_spots.")
    else:
        # Raise error if data in nbp_ref_spots already exists that will be overwritten in this function.
        error_message = ""
        for var in ['gene_no', 'gene_score', 'score_diff', 'intensity', 'background_strength']:
            if hasattr(nbp_ref_spots, var) and nbp_ref_spots.__getattribute__(var) is not None:
                error_message += f"\nnbp_ref_spots.{var} is not None but this function will overwrite {var}." \
                                 f"\nRun with overwrite_ref_spots = True to get past this error."
        if len(error_message) > 0:
            raise ValueError(error_message)

    nbp_ref_spots.finalized = False  # So we can add and delete ref_spots page variables
    # delete all variables in ref_spots set to None so can add them later.
    for var in ['gene_no', 'score', 'score_diff', 'intensity', 'background_strength']:
        if hasattr(nbp_ref_spots, var):
            nbp_ref_spots.__delattr__(var)
    nbp = NotebookPage("call_spots")

    # 0. Initialise frequently used variables
    n_rounds, use_channels = nbp_basic.n_rounds, nbp_basic.use_channels
    spot_colours = nbp_ref_spots.colors[:, :, nbp_basic.use_channels]
    isolated = nbp_ref_spots.isolated

    # 1. Remove background from spots and normalise channels and rounds
    spot_colours_background_removed, background_noise = remove_background(spot_colours=spot_colours.copy())
    initial_norm_factor = np.load('/Users/reillytilbury/Documents/GitHub/coppafish/coppafish/setup/default_norm.npy')
    spot_colours_background_removed = spot_colours_background_removed / initial_norm_factor
    colour_norm_factor, spot_intensity = normalise_rc(spot_colours=spot_colours_background_removed[isolated],
                                                      initial_bleed_matrix=initial_bleed_matrix)
    # First remove round 0 and 1 from norm factor as these have issues with anchor staining
    colour_norm_factor = np.median(colour_norm_factor[2:], axis=0)
    spot_colours = spot_colours_background_removed / colour_norm_factor
    colour_norm_factor = colour_norm_factor * initial_norm_factor

    # 2. Bleed matrix calculation and bled codes
    bleed_matrix, all_dye_score = compute_bleed_matrix(initial_bleed_matrix=initial_bleed_matrix,
                                                       spot_colours=spot_colours[isolated])
    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
    n_genes = len(gene_names)
    ge_initial = np.ones((n_genes, n_rounds))
    bled_codes = get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix, gene_efficiency=ge_initial)
    gene_no, gene_score, gene_score_second = compute_gene_scores(spot_colours=spot_colours, bled_codes=bled_codes)

    # 3. Gene efficiency calculation.
    # GE calculation is done iteratively in a similar way to scaled k-means clustering. We start with our initial
    # score distribution and bled codes and then these 2 parameters are iteratively updated until convergence.
    # for i in tqdm(range(config['n_iter'])):
    for i in range(2):
        print(np.median(gene_score))
        # 3.1 Calculate gene efficiency
        gene_efficiency, use_ge = compute_gene_efficiency(spot_colours=spot_colours, bleed_matrix=bleed_matrix,
                                                          bled_codes=bled_codes, gene_no=gene_no, gene_score=gene_score,
                                                          gene_codes=gene_codes)
        # 3.2 Update bled codes
        bled_codes = get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix, gene_efficiency=gene_efficiency)

        # 3.3 Update gene coefficients
        gene_no, gene_score, gene_score_second = compute_gene_scores(spot_colours=spot_colours, bled_codes=bled_codes)

    # save overwritable variables in nbp_ref_spots
    nbp_ref_spots.gene_no = gene_no
    nbp_ref_spots.score = gene_score
    nbp_ref_spots.score_diff = gene_score - gene_score_second
    nbp_ref_spots.intensity = np.median(np.max(spot_colours, axis=2), axis=1).astype(np.float32)
    nbp_ref_spots.background_strength = background_noise
    nbp_ref_spots.finalized = True

    # Save variables in nbp
    nbp.use_ge = use_ge
    nbp.gene_names = gene_names
    nbp.gene_codes = gene_codes
    nbp.color_norm_factor = expand_channels(colour_norm_factor, use_channels, nbp_basic.n_channels)
    nbp.initial_bleed_matrix = expand_channels(initial_bleed_matrix, use_channels, nbp_basic.n_channels)
    nbp.bleed_matrix = expand_channels(bleed_matrix, use_channels, nbp_basic.n_channels)
    nbp.bled_codes_ge = expand_channels(bled_codes, use_channels, nbp_basic.n_channels)
    nbp.bled_codes = expand_channels(get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix,
                                                    gene_efficiency=ge_initial), use_channels, nbp_basic.n_channels)
    nbp.gene_efficiency = gene_efficiency

    # Backward compatibility here as OMP requires nbp.abs_intensity_percentile.
    # TODO: Remove this in future versions
    # rc_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    # pixel_colors = get_spot_colors(all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz // 2),
    #                                int(np.median(nbp_basic.use_tiles)), transform, nbp_file, nbp_basic,
    #                                return_in_bounds=True)[0]
    # pixel_intensity = get_spot_intensity(np.abs(pixel_colors) / nbp.color_norm_factor[rc_ind])
    nbp.abs_intensity_percentile = np.random.rand(100)

    return nbp, nbp_ref_spots
