from .. import setup
from ..call_spots import get_dye_channel_intensity_guess, get_bleed_matrix, get_bled_codes, color_normalisation, \
    dot_product, get_spot_intensity
import numpy as np
from ..setup.notebook import NotebookPage
from typing import Tuple


def call_reference_spots(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage, nbp_ref_spots: NotebookPage,
                         hist_values: np.ndarray, hist_counts: np.ndarray) -> Tuple[NotebookPage, NotebookPage]:
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
            `global_yxz`, `isolated`, `tile`, `colors`.
        hist_values: `int [n_pixel_values]`.
            All possible pixel values in saved tiff images i.e. `n_pixel_values` is approximately
            `np.iinfo(np.uint16).max` because tiffs saved as `uint16` images.
            This is saved in the extract notebook page i.e. `nb.extract.hist_values`.
        hist_counts: `int [n_pixel_values x n_rounds x n_channels]`.
            `hist_counts[i, r, c]` is the number of pixels across all tiles in round `r`, channel `c`
            which had the value `hist_values[i]`.
            This is saved in extract notebook page i.e. `nb.extract.hist_counts`.

    Returns:
        - `NotebookPage[call_spots]` - Page contains bleed matrix and expected code for each gene.
        - `NotebookPage[ref_spots]` - Page contains gene assignments and info for spots found on reference round.
    """
    nbp = setup.NotebookPage("call_spots")

    # get color norm factor
    rc_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    hist_counts_use = np.moveaxis(np.moveaxis(hist_counts, 0, -1)[rc_ind], -1, 0)
    color_norm_factor = np.ones((nbp_basic.n_rounds, nbp_basic.n_channels)) * np.nan
    color_norm_factor[rc_ind] = color_normalisation(hist_values, hist_counts_use, config['color_norm_intensities'],
                                                    config['color_norm_probs'], config['bleed_matrix_method'])

    # get initial bleed matrix
    initial_raw_bleed_matrix = np.ones((nbp_basic.n_rounds, nbp_basic.n_channels, nbp_basic.n_dyes)) * np.nan
    rcd_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels, nbp_basic.use_dyes)
    if nbp_basic.dye_names is not None:
        # if specify dyes, will initialize bleed matrix using prior data
        dye_names_use = np.array(nbp_basic.channel_camera)[nbp_basic.use_dyes]
        camera_use = np.array(nbp_basic.channel_camera)[nbp_basic.use_channels]
        laser_use = np.array(nbp_basic.channel_laser)[nbp_basic.use_channels]
        initial_raw_bleed_matrix[rcd_ind] = get_dye_channel_intensity_guess(nbp_file.dye_camera_laser,
                                                                            dye_names_use, camera_use,
                                                                            laser_use).transpose()
        initial_bleed_matrix = initial_raw_bleed_matrix / np.expand_dims(color_norm_factor, 2)
    else:
        if nbp_basic.n_dyes != nbp_basic.n_channels:
            raise ValueError(f"'dye_names' were not specified so expect each dye to correspond to a different channel."
                             f"\nBut n_channels={nbp_basic['n_channels']} and n_dyes={nbp_basic['n_dyes']}")
        if nbp_basic.use_channels != nbp_basic.use_dyes:
            raise ValueError(f"'dye_names' were not specified so expect each dye to correspond to a different channel."
                             f"\nBleed matrix computation requires use_channels and use_dyes to be the same to work."
                             f"\nBut use_channels={nbp_basic['use_channels']} and use_dyes={nbp_basic['use_dyes']}")
        initial_bleed_matrix = initial_raw_bleed_matrix.copy()
        initial_bleed_matrix[rcd_ind] = np.tile(np.expand_dims(np.eye(nbp_basic.n_channels), 0),
                                                (nbp_basic.n_rounds, 1, 1))[rcd_ind]

    # get bleed matrix
    spot_colors = nbp_ref_spots.colors / color_norm_factor
    spot_colors_use = np.moveaxis(np.moveaxis(spot_colors, 0, -1)[rc_ind], -1, 0)
    bleed_matrix = initial_raw_bleed_matrix.copy()
    bleed_matrix[rcd_ind] = get_bleed_matrix(spot_colors_use[nbp_ref_spots.isolated], initial_bleed_matrix[rcd_ind],
                                             config['bleed_matrix_method'], config['bleed_matrix_score_thresh'])

    # get gene codes
    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
    bled_codes = get_bled_codes(gene_codes, bleed_matrix)
    bled_codes_use = np.moveaxis(np.moveaxis(bled_codes, 0, -1)[rc_ind], -1, 0)

    nbp.gene_names = gene_names
    nbp.gene_codes = gene_codes
    nbp.color_norm_factor = color_norm_factor
    nbp.initial_raw_bleed_matrix = initial_raw_bleed_matrix
    nbp.initial_bleed_matrix = initial_bleed_matrix
    nbp.bleed_matrix = bleed_matrix
    nbp.bled_codes = bled_codes

    # get gene assignment and score
    if config['dot_product_method'].lower() == 'single':
        norm_axis = (1, 2) # dot product considering all rounds together
    elif config['dot_product_method'].lower() == 'separate':
        norm_axis = 2  # independent dot product for each round
    else:
        raise ValueError(f"dot_product_method is {config['dot_product_method']}, "
                         f"but should be either 'single' or 'separate'.")
    scores = dot_product(spot_colors_use, bled_codes_use, norm_axis)
    sort_gene_inds = np.argsort(scores, axis=1)
    nbp_ref_spots.gene_no = sort_gene_inds[:, -1]
    gene_no_second_best = sort_gene_inds[:, -2]
    nbp_ref_spots.score = scores[np.arange(np.shape(scores)[0]), nbp_ref_spots.gene_no]
    score_second_best = scores[np.arange(np.shape(scores)[0]), gene_no_second_best]
    nbp_ref_spots.score_diff = nbp_ref_spots.score - score_second_best
    nbp_ref_spots.intensity = get_spot_intensity(spot_colors_use)

    return nbp, nbp_ref_spots
