import numpy as np
from .. import utils
from ..call_spots.base import fit_background, dot_product_score
from ..extract.deconvolution import get_isolated_points, get_spot_images, get_average_spot_image
from ..find_spots import detect_spots
from typing import Tuple, Optional, Union, List
from tqdm import tqdm
import warnings
from scipy.sparse import csr_matrix


def fitting_standard_deviation(bled_codes: np.ndarray, coef: np.ndarray, alpha: float, beta: float = 1) -> np.ndarray:
    """
    Based on maximum likelihood estimation, this finds the standard deviation accounting for all genes fit in
    each round/channel. The more genes added, the greater the standard deviation so if the inverse is used as a
    weighting for omp fitting, the rounds/channels which already have genes in will contribute less.

    Args:
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        coef: `float [n_pixels x n_genes]`.
            Coefficient of each `bled_code` for each pixel found on the previous OMP iteration.
        alpha: By how much to increase variance as genes added.
        beta: The variance with no genes added (`coef=0`) is `beta**2`.

    Returns:
        `float [n_pixels x n_rounds x n_channels]`
            Standard deviation of each pixel in each round/channel based on genes fit.
    """
    n_genes, n_rounds, n_channels = bled_codes.shape
    n_pixels = coef.shape[0]

    if not utils.errors.check_shape(coef, [n_pixels, n_genes]):
        raise utils.errors.ShapeError('coef', coef.shape, (n_pixels, n_genes))

    var = np.ones((n_pixels, n_rounds, n_channels)) * beta ** 2
    for g in range(n_genes):
        var = var + alpha * np.expand_dims(coef[:, g] ** 2, (1, 2)) * np.expand_dims(bled_codes[g] ** 2, 0)

    sigma = np.sqrt(var)
    return sigma


def fit_coefs(bled_codes: np.ndarray, pixel_colors: np.ndarray, weight: Optional[np.ndarray] = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    This finds the least squared solution for how the `n_genes` `bled_codes` can best explain each `pixel_color`.
    Can also find weighted least squared solution if `weight` provided.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]`.
            Flattened then transposed pixel colors which usually has the shape `[n_genes x n_rounds x n_channels]`.
        weight: `float [(n_rounds x n_channels) x 1]`.
            Weight to be applied to each data value when computing coefficient of each `bled_code` for each pixel.

    Returns:
        - residual - `float [(n_rounds x n_channels) x n_pixels]`.
            Residual pixel_colors are removing bled_codes with coefficients specified by coef.
        - coef - `float [n_pixels x n_genes]`.
            coefficient found through least squares fitting for each gene.

    """
    if weight is not None:
        pixel_colors = pixel_colors * weight
        bled_codes = bled_codes * weight
    if bled_codes.shape[1] == 1:
        # can do many pixels at once if just one gene and is quicker this way.
        coefs = np.sum(bled_codes * pixel_colors, axis=0) / np.sum(bled_codes ** 2)
        residual = pixel_colors - coefs * bled_codes
        coefs = coefs.reshape(1, -1)
    else:
        # TODO: maybe iterate over all unique combinations of added genes instead of over all spots.
        #  Would not work if do weighted coef fitting though.
        coefs = np.linalg.lstsq(bled_codes, pixel_colors, rcond=None)[0]
        residual = pixel_colors - bled_codes @ coefs
    if weight is not None:
        residual = residual / weight
    return residual, coefs.transpose()


def get_best_gene(residual_pixel_colors: np.ndarray, bled_codes: np.ndarray, coefs: np.ndarray, norm_shift: float,
                  score_thresh: float, alpha: float, beta: float,
                  ignore_genes: Optional[Union[np.ndarray, List]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.


    !!! note
        `best_gene` will be set to -1 if dot product is less than `score_thresh` or if the `best_gene` has already
        been added to the pixel or if best_gene is in ignore_genes.

    Args:
        residual_pixel_colors: `float [n_pixels x n_rounds x n_channels]`.
            Residual pixel colors from previous iteration of omp.
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        coefs: `float [n_pixels x n_genes]`.
            `coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm on previous iteration.
             Most are zero.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        alpha: Used for `fitting_standard_deviation`, by how much to increase variance as genes added.
        beta: Used for `fitting_standard_deviation`, the variance with no genes added (`coef=0`) is `beta**2`.
        ignore_genes: `int [n_genes_ignore]`.
            Genes indices which if they are the best gene, best_gene will be set to -1.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next. It is -1 if no more genes should be added.
        - sigma - `float [n_pixels x n_rounds x n_channels]`.
            Standard deviation of each pixel in each round/channel based on genes fit on previous iteration.

    """
    if ignore_genes is None:
        ignore_genes = []
    sigma = fitting_standard_deviation(bled_codes, coefs, alpha, beta)
    all_scores = dot_product_score(residual_pixel_colors, bled_codes, norm_shift, 1 / sigma)
    best_gene = np.argmax(np.abs(all_scores), 1)
    all_scores[coefs != 0] = 0  # best gene cannot be a gene which has already been added.
    for g in ignore_genes:
        all_scores[:, g] = 0  # best gene cannot be a gene in ignore_genes.
    best_score = all_scores[np.arange(all_scores.shape[0]), best_gene]
    best_gene[np.abs(best_score) <= score_thresh] = -1
    return best_gene, sigma


def get_all_coefs(pixel_colors: np.ndarray, bled_codes: np.ndarray, background_shift: float,
                  dp_shift: float, dp_thresh: float, alpha: float, beta: float, max_genes: int,
                  weight_coef_fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    This performs omp on every pixel, the stopping criterion is that the dot_product_score
    when selecting the next gene to add exceeds dp_thresh or the number of genes added to the pixel exceeds max_genes.

    !!! note
        Background vectors are fitted first and then not updated again.

    Args:
        pixel_colors: `float [n_pixels x n_rounds x n_channels]`.
            Pixel colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        background_shift: When fitting background,
            this is applied to weighting of each background vector to limit boost of weak pixels.
        dp_shift: When finding `dot_product_score` between residual `pixel_colors` and `bled_codes`,
            this is applied to normalisation of `pixel_colors` to limit boost of weak pixels.
        dp_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added at each iteration.
        alpha: Used for `fitting_standard_deviation`, by how much to increase variance as genes added.
        beta: Used for `fitting_standard_deviation`, the variance with no genes added (`coef=0`) is `beta**2`.
        max_genes: Maximum number of genes that can be added to a pixel i.e. number of iterations of OMP.
        weight_coef_fit: If False, coefs are found through normal least squares fitting.
            If True, coefs are found through weighted least squares fitting using 1/sigma as the weight factor.

    Returns:
        - gene_coefs - `float [n_pixels x n_genes]`.
            `gene_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm. Most are zero.
        - background_coefs - `float [n_pixels x n_channels]`.
            coefficient value for each background vector found for each pixel.
    """
    diff_to_int = np.round(pixel_colors).astype(int) - pixel_colors
    if np.abs(diff_to_int).max() == 0:
        raise ValueError("spot_intensities should be found using normalised spot_colors. "
                         "\nBut all values in spot_colors given are integers indicating they are the raw intensities.")
    del diff_to_int
    n_genes, n_rounds, n_channels = bled_codes.shape
    n_pixels = pixel_colors.shape[0]
    if not utils.errors.check_shape(pixel_colors, [n_pixels, n_rounds, n_channels]):
        raise utils.errors.ShapeError('pixel_colors', pixel_colors.shape, (n_pixels, n_rounds, n_channels))
    no_verbose = n_pixels < 1000  # show progress bar with more than 1000 pixels.

    # Fit background and override initial pixel_colors
    all_coefs = np.zeros((n_pixels, n_genes + n_channels))  # coefs of all genes and background
    pixel_colors, all_coefs[:, -n_channels:], background_codes = fit_background(pixel_colors, background_shift)
    background_genes = np.arange(n_genes, n_genes + n_channels)

    # colors and codes for get_best_gene function
    # Includes background as if background is the best gene, iteration ends.
    # uses residual color as used to find next gene to add.
    all_codes = np.concatenate((bled_codes, background_codes))
    residual_pixel_colors = pixel_colors.copy()

    # colors and codes for fit_coefs function (No background as this is not updated again).
    # always uses post background color as coefficients for all genes re-estimated at each iteration.
    pixel_colors = pixel_colors.reshape((n_pixels, -1)).transpose()
    bled_codes = bled_codes.reshape((n_genes, -1)).transpose()

    added_genes = np.ones((n_pixels, max_genes), dtype=int) * -1
    sigma = np.zeros((n_pixels, n_rounds, n_channels))
    continue_pixels = np.arange(n_pixels)
    for i in range(max_genes):
        # only continue with pixels for which dot product score exceeds threshold
        added_genes[continue_pixels, i], sigma[continue_pixels] = get_best_gene(residual_pixel_colors[continue_pixels],
                                                                                all_codes, all_coefs[continue_pixels],
                                                                                dp_shift, dp_thresh, alpha, beta,
                                                                                background_genes)
        residual_pixel_colors = residual_pixel_colors.reshape((n_pixels, -1)).transpose()
        continue_pixels = added_genes[:, i] >= 0
        n_continue = sum(continue_pixels)
        if n_continue == 0:
            break
        with tqdm(total=n_continue, disable=no_verbose) as pbar:
            pbar.set_postfix({'iter': i, 'n_pixels': n_continue})
            if i == 0:
                # When adding only 1 gene, can do many pixels at once if neglect weighting.
                # Neglecting weighting seems reasonable as small effect with just background.
                for g in range(n_genes):
                    use = added_genes[:, i] == g
                    residual_pixel_colors[:, use], all_coefs[use, g:g + 1] = fit_coefs(bled_codes[:, g:g + 1],
                                                                                       pixel_colors[:, use])
                    pbar.update(np.sum(use))
            else:
                if weight_coef_fit:
                    weight = 1 / sigma.reshape((n_pixels, -1)).transpose()
                for s in np.where(continue_pixels)[0]:
                    # s:s+1 is so shape is correct for fit_coefs function.
                    if weight_coef_fit:
                        residual_pixel_colors[:, s:s + 1], all_coefs[s, added_genes[s, :i + 1]] = \
                            fit_coefs(bled_codes[:, added_genes[s, :i + 1]], pixel_colors[:, s:s + 1],
                                      weight[:, s:s + 1])
                    else:
                        # TODO: maybe do this fitting with all unique combinations of genes so can do
                        #  multiple spots at once.
                        residual_pixel_colors[:, s:s + 1], all_coefs[s, added_genes[s, :i + 1]] = \
                            fit_coefs(bled_codes[:, added_genes[s, :i + 1]], pixel_colors[:, s:s + 1])
                    pbar.update(1)
        pbar.close()
        residual_pixel_colors = residual_pixel_colors.transpose().reshape((n_pixels, n_rounds, n_channels))

    gene_coefs = all_coefs[:, :n_genes]
    background_coefs = all_coefs[:, -n_channels:]
    return gene_coefs, background_coefs


def detect_spots_all_genes(pixel_coefs: np.array, pixel_yxz: np.ndarray, radius_xy: int,
                           radius_z: Optional[int] = None) -> np.ndarray:
    """
    Finds all local maxima in coef image of each gene with threshold of 0.

    Args:
        pixel_coefs: `float [n_pixels x n_genes]`.
            `pixel_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm.
             Most are zero hence sparse form used.
        pixel_yxz: ```int [n_pixels x 3]```.
            ```pixel_yxz[s, :2]``` are the local yx coordinates in ```yx_pixels``` for pixel ```s```.
            ```pixel_yxz[s, 2]``` is the local z coordinate in ```z_pixels``` for pixel ```s```.
        radius_xy: Radius of dilation structuring element in xy plane (approximately spot radius).
        radius_z: Radius of dilation structuring element in z direction (approximately spot radius).
            If ```None```, 2D filter is used.

    Returns:
        `int [n_spots x 4]`
            ```spot_yxzg[s, :2]``` are the local yx coordinates in ```yx_pixels``` for spot ```s```.
            ```spot_yxzg[s, 2]``` is the local z coordinate in ```z_pixels``` for spot ```s```.
            ```spot_yxzg[s, 3]``` is the gene that this spot is assigned to.
    """
    # shift pixel_yxz so min is 0 in each axis so smaller image can be formed.
    coord_shift = pixel_yxz.min(axis=0)
    pixel_yxz = pixel_yxz - coord_shift
    n_genes = pixel_coefs.shape[1]
    n_y, n_x, n_z = pixel_yxz.max(axis=0) + 1
    # initialise with big array assuming each pixel is a spot.
    spot_yxzg = np.zeros((pixel_coefs.shape[0], 4), dtype=int)
    n_spots = 0
    for g in range(n_genes):
        # coef_image at pixels not indicated by pixel_yxz is set to 0.
        if n_z == 1:
            coef_image = np.zeros((n_y, n_x))
        else:
            coef_image = np.zeros((n_y, n_x, n_z))
        coef_image[tuple([pixel_yxz[:, j] for j in range(coef_image.ndim)])] = pixel_coefs[:, g]
        spot_yxz, _ = detect_spots(coef_image, 0, radius_xy, radius_z, False)
        spot_yxzg[n_spots: n_spots + spot_yxz.shape[0], :coef_image.ndim] = spot_yxz
        spot_yxzg[n_spots: n_spots + spot_yxz.shape[0], 3] = g
        n_spots = n_spots + spot_yxz.shape[0]
    spot_yxzg = spot_yxzg[:n_spots]  # Only keep spots found.
    spot_yxzg[:, :3] = spot_yxzg[:, :3] + coord_shift
    return spot_yxzg


def count_spot_neighbours(image: np.ndarray, spot_yxz: np.ndarray, pos_filter: np.ndarray,
                          neg_filter: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Counts the number of positive (and negative) pixels in a neighbourhood about each spot.

    Args:
        image: `float [n_y x n_x (x n_z)]`.
            image spots were found on.
        spot_yxz: `int [n_peaks x image.ndim]`.
            yx or yxz location of spots found.
        pos_filter: `int [filter_sz_y x filter_sz_x (x filter_sz_z)]`.
            Number of positive pixels counted in this neighbourhood about each spot in image.
            Only contains values 0 and 1.
        neg_filter: `int [filter_sz_y x filter_sz_x (x filter_sz_z)]`.
            Number of negative pixels counted in this neighbourhood about each spot in image.
            Only contains values 0 and 1.
            `None` means don't find `n_neg_neighbours`.

    Returns:
        - n_pos_neighbours - `float [n_pixels x n_genes]`.
            `gene_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm. Most are zero.
        - n_neg_neighbours - `float [n_pixels x n_channels]`.
            coefficient value for each background vector found for each pixel.
            Only returned if 'neg_filter is not None'
    """
    # Correct for 2d cases where an empty dimension has been used for some variables.
    if all([image.ndim == spot_yxz.shape[1] - 1, np.max(np.abs(spot_yxz[:, -1])) == 0]):
        # Image 2D but spots 3D
        spot_yxz = spot_yxz[:, :image.ndim]
    if all([image.ndim == spot_yxz.shape[1] + 1, image.shape[-1] == 1]):
        # Image 3D but spots 2D
        image = np.mean(image, axis=image.ndim - 1)  # average over last dimension just means removing it.
    if all([image.ndim == pos_filter.ndim - 1, pos_filter.shape[-1] == 1]):
        # Image 2D but pos_filter 3D
        pos_filter = np.mean(pos_filter, axis=pos_filter.ndim - 1)
    if neg_filter is not None:
        if all([image.ndim == neg_filter.ndim - 1, neg_filter.shape[-1] == 1]):
            # Image 2D but neg_filter 3D
            neg_filter = np.mean(neg_filter, axis=neg_filter.ndim - 1)
        if not np.isin(neg_filter, [0, 1]).all():
            raise ValueError('neg_filter contains values other than 0 or 1.')

    if not np.isin(pos_filter, [0, 1]).all():
        raise ValueError('pos_filter contains values other than 0 or 1.')

    # Check all spots in image
    max_yxz = np.array(image.shape) - 1
    spot_oob = [val for val in spot_yxz if val.min() < 0 or any(val > max_yxz)]
    if len(spot_oob) > 0:
        raise utils.errors.OutOfBoundsError("spot_yxz", spot_oob[0], [0] * image.ndim, max_yxz)

    # make binary images indicating sign of image.
    # TODO: give option of providing pos_image and neg_image instead of image as less memory.
    pos_image = (image > 0).astype(int)
    # filter these to count neighbours at each pixel.
    pos_neighbour_image = utils.morphology.imfilter(pos_image, pos_filter.astype(int), 'reflect')
    # find number of neighbours at each spot.
    n_pos_neighbours = pos_neighbour_image[tuple([spot_yxz[:, j] for j in range(image.ndim)])]
    if neg_filter is None:
        return n_pos_neighbours
    else:
        neg_image = (image < 0).astype(int)
        neg_neighbour_image = utils.morphology.imfilter(neg_image, neg_filter.astype(int), 'reflect')
        n_neg_neighbours = neg_neighbour_image[tuple([spot_yxz[:, j] for j in range(image.ndim)])]
        return n_pos_neighbours, n_neg_neighbours


def spot_neighbourhood(pixel_coef: np.array, pixel_yxz: np.ndarray, spot_yxzg: np.ndarray,
                       max_size: Union[np.ndarray, List], pos_neighbour_thresh: int, isolation_dist: float,
                       z_scale: float, mean_sign_thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the expected sign the coefficient should have in the neighbourhood about a spot.

    Args:
        pixel_coef: `float [n_pixels x n_genes]`.
            `pixel_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm.
             Most are zero hence sparse form used.
        pixel_yxz: ```int [n_pixels x 3]```.
            ```pixel_yxz[s, :2]``` are the local yx coordinates in ```yx_pixels``` for pixel ```s```.
            ```pixel_yxz[s, 2]``` is the local z coordinate in ```z_pixels``` for pixel ```s```.
        spot_yxzg: ```int [n_spots x 4]```.
            ```spot_yxzg[s, :2]``` are the local yx coordinates in ```yx_pixels``` for spot ```s```.
            ```spot_yxzg[s, 2]``` is the local z coordinate in ```z_pixels``` for spot ```s```.
            ```spot_yxzg[s, 3]``` is the gene that this spot is assigned to.
        max_size: `int [n_spots x 3]`.
            max YXZ size of spot shape returned. Zeros at extremities will be cropped.
        pos_neighbour_thresh: For spot to be used to find av_spot_image, it must have this many pixels
            around it on the same z-plane that have a positive coefficient.
            If 3D, also, require 1 positive pixel on each neighbouring plane (i.e. 2 is added to this value).
            Typical = 9.
        isolation_dist: Spots are isolated if nearest neighbour (across all genes) is further away than this.
            Only isolated spots are used to find av_spot_image.
        z_scale: Scale factor to multiply z coordinates to put them in units of yx pixels.
            I.e. ```z_scale = pixel_size_z / pixel_size_yx``` where both are measured in microns.
            typically, ```z_scale > 1``` because ```z_pixels``` are larger than the ```yx_pixels```.
        mean_sign_thresh: If the mean absolute coefficient sign is less than this in a region near a spot,
            we set the expected coefficient in av_spot_image to be 0.

    Returns:
        - av_spot_image - `int [av_shape_y x av_shape_x x av_shape_z]`
            Expected sign of omp coefficient in neighbourhood centered on spot.
        - spot_indices_used - `int [n_spots_used]`.
            indices of spots in `spot_yxzg` used to make av_spot_image.
        - av_spot_image_float - `float [max_size[0] x max_size[1] (x max_size[2])]`
            Mean sign of omp coefficient in neighbourhood centered on spot.
            This is before cropping and thresholding.
    """
    # TODO: Maybe provide pixel_coef_sign instead of pixel_coef as less memory or use csr_matrix.
    # shift coordinates so min is 0 in each axis so smaller image can be formed.
    coord_shift = np.min([pixel_yxz.min(axis=0), spot_yxzg[:, :3].min(axis=0)], axis=0)
    spot_yxzg[:, :3] = spot_yxzg[:, :3] - coord_shift
    pixel_yxz = pixel_yxz - coord_shift
    n_y, n_x, n_z = pixel_yxz.max(axis=0) + 1
    n_genes = pixel_coef.shape[1]
    pos_filter_shape_yx = np.ceil(np.sqrt(pos_neighbour_thresh)).astype(int)
    if pos_filter_shape_yx % 2 == 0:
        # Shape must be odd
        pos_filter_shape_yx = pos_filter_shape_yx + 1
    if n_z == 1:
        pos_filter_shape_z = 1
    else:
        pos_filter_shape_z = 3
    pos_filter = np.zeros((pos_filter_shape_yx, pos_filter_shape_yx, pos_filter_shape_z), dtype=int)
    pos_filter[:, :, np.floor(pos_filter_shape_z/2).astype(int)] = 1
    if n_z > 1:
        mid_yx = np.floor(pos_filter_shape_yx/2).astype(int)
        pos_filter[mid_yx, mid_yx, 0] = 1
        pos_filter[mid_yx, mid_yx, 2] = 1

    max_size = np.array(max_size)
    if n_z == 1:
        max_size[2] = 1
    max_size_odd_loc = np.where(np.array(max_size) % 2 == 0)[0]
    if max_size_odd_loc.size > 0:
        max_size[max_size_odd_loc] += 1  # ensure shape is odd

    # get image centred on each spot.
    n_spots = spot_yxzg.shape[0]
    spot_images = np.zeros((n_spots, *max_size), dtype=int)  # Big image shape which will be cropped later
    spots_used = np.zeros(n_spots, dtype=bool)
    for g in range(n_genes):
        coef_sign_image = np.zeros((n_y, n_x, n_z), dtype=int)
        coef_sign_image[tuple([pixel_yxz[:, j] for j in range(coef_sign_image.ndim)])] = \
            np.sign(pixel_coef[:, g]).astype(int)
        use = spot_yxzg[:, -1] == g
        if use.any():
            # Only keep spots with all neighbourhood having positive coefficient.
            n_pos_neighb = count_spot_neighbours(coef_sign_image, spot_yxzg[use, :-1], pos_filter)
            use[np.where(use)[0][n_pos_neighb != pos_filter.sum()]] = False
            if use.any():
                # Maybe need float coef_sign_image here
                spot_images[use] = get_spot_images(coef_sign_image, spot_yxzg[use, :-1], max_size)
                spots_used[use] = True

    # Compute average spot image from all isolated spots
    spot_images = spot_images[spots_used]
    isolated = get_isolated_points(spot_yxzg[spots_used, :-1] * [1, 1, z_scale], isolation_dist)
    av_spot_image = get_average_spot_image(spot_images[isolated].astype(float), 'mean', 'annulus_3d')
    av_spot_image_float = av_spot_image.copy()
    spot_indices_used = np.where(spots_used)[0][isolated]

    # Where mean sign is low, set to 0.
    av_spot_image[np.abs(av_spot_image) < mean_sign_thresh] = 0
    av_spot_image = np.sign(av_spot_image).astype(int)

    # Crop image to remove zeros at extremities
    # may get issue here if there is a positive sign pixel further away than negative but think unlikely.
    av_spot_image = av_spot_image[:, :, ~np.all(av_spot_image == 0, axis=(0, 1))]
    av_spot_image = av_spot_image[:, ~np.all(av_spot_image == 0, axis=(0, 2)), :]
    av_spot_image = av_spot_image[~np.all(av_spot_image == 0, axis=(1, 2)), :, :]

    if np.sum(av_spot_image == 1) == 0:
        warnings.warn(f"In av_spot_image, no pixels have a value of 1.\n"
                      f"Maybe mean_sign_thresh = {mean_sign_thresh} is too high.")
    if np.sum(av_spot_image == -1) == 0:
        warnings.warn(f"In av_spot_image, no pixels have a value of -1.\n"
                      f"Maybe mean_sign_thresh = {mean_sign_thresh} is too high.")
    if np.sum(av_spot_image == 0) == 0:
        warnings.warn(f"In av_spot_image, no pixels have a value of 0.\n"
                      f"Maybe mean_sign_thresh = {mean_sign_thresh} is too low.")

    return av_spot_image, spot_indices_used, av_spot_image_float
