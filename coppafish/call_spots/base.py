from scipy.spatial import KDTree
import numpy as np
from .. import utils
from typing import Union, List


def get_non_duplicate(tile_origin: np.ndarray, use_tiles: List, tile_centre: np.ndarray,
                      spot_local_yxz: np.ndarray, spot_tile: np.ndarray) -> np.ndarray:
    """
    Find duplicate spots as those detected on a tile which is not tile centre they are closest to.

    Args:
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch` notebook page i.e. `nb.stitch.tile_origin`.
        use_tiles: ```int [n_use_tiles]```.
            Tiles used in the experiment.
        tile_centre: ```float [3]```
            ```tile_centre[:2]``` are yx coordinates in ```yx_pixels``` of the centre of the tile that spots in
            ```yxz``` were found on.
            ```tile_centre[2]``` is the z coordinate in ```z_pixels``` of the centre of the tile.
            E.g. for tile of ```yxz``` dimensions ```[2048, 2048, 51]```, ```tile_centre = [1023.5, 1023.5, 25]```
            Each entry in ```tile_centre``` must be an integer multiple of ```0.5```.
        spot_local_yxz: ```int [n_spots x 3]```.
            Coordinates of a spot s on tile spot_tile[s].
            ```yxz[s, :2]``` are the yx coordinates in ```yx_pixels``` for spot ```s```.
            ```yxz[s, 2]``` is the z coordinate in ```z_pixels``` for spot ```s```.
        spot_tile: ```int [n_spots]```.
            Tile each spot was found on.

    Returns:
        ```bool [n_spots]```.
            Whether spot_tile[s] is the tile that spot_global_yxz[s] is closest to.
    """
    tile_centres = tile_origin[use_tiles] + tile_centre
    # Do not_duplicate search in 2D as overlap is only 2D
    tree_tiles = KDTree(tile_centres[:, :2])
    if np.isnan(tile_origin[np.unique(spot_tile)]).any():
        nan_tiles = np.unique(spot_tile)[np.unique(np.where(np.isnan(tile_origin[np.unique(spot_tile)]))[0])]
        raise ValueError(f"tile_origin for tiles\n{nan_tiles}\ncontains nan values but some spot_tile "
                         f"also contains these tiles. Maybe remove these from use_tiles to continue.\n"
                         f"Also, consider coppafish.plot.n_spots_grid to check if these tiles have few spots.")
    spot_global_yxz = spot_local_yxz + tile_origin[spot_tile]
    all_nearest_tile_ind = tree_tiles.query(spot_global_yxz[:, :2])[1]
    not_duplicate = np.asarray(use_tiles)[all_nearest_tile_ind.flatten()] == spot_tile
    return not_duplicate


def color_normalisation(hist_values: np.ndarray, hist_counts: np.ndarray,
                        thresh_intensities: Union[float, List[float], np.ndarray],
                        thresh_probs: Union[float, List[float], np.ndarray], method: str) -> np.ndarray:
    """
    This finds the normalisations for each round, ```r```, and channel, ```c```, such that if ```norm_spot_color[r,c] =
    spot_color[r,c] / norm_factor[r,c]```, the probability of ```norm_spot_color``` being larger than ```
    thresh_intensities[i]``` is less than ```thresh_probs[i]``` for every ```i```.
    Where the probability is based on all pixels from all tiles in that round and channel.

    Args:
        hist_values: ```int [n_pixel_values]```.
            All possible pixel values in saved tiff images i.e. n_pixel_values is approximately
            ```np.iinfo(np.uint16).max``` because tiffs saved as ```uint16``` images.
        hist_counts: ```int [n_pixel_values x n_rounds x n_channels]```.
            ```hist_counts[i, r, c]``` is the number of pixels across all tiles in round ```r```, channel ```c```
            which had the value ```hist_values[i]```.
        thresh_intensities: ```float [n_thresholds]```.
            Thresholds such that the probability of having a normalised spot_color greater than this are quite low.
            Need to be ascending.
            Typical: ```[0.5, 1, 5]``` i.e. we want most of ```normalised spot_colors``` to be less than ```0.5``` so
            high normalised spot color is on the order of ```1```.
        thresh_probs: ```float [n_thresholds]```.
            Probability of normalised spot color being greater than ```thresh_intensities[i]``` must be less than
            ```thresh_probs[i]```. Needs to be same shape as thresh_intensities and descending.
            Typical: ```[0.01, 5e-4, 1e-5]``` i.e. want almost all non spot pixels to have
            normalised intensity less than ```0.5```.
        method: Must be one of the following:

            - ```'single'``` - A single normalisation factor is produced for all rounds of each channel
                i.e. ```norm_factor[r, b]``` for a given ```b``` value, will be the same for all ```r``` values.
            - ```'separate'``` - A different normalisation factor is made for each round and channel.

    Returns:
        ```float [n_rounds x n_channels]```.
            ```norm_factor``` such that ```norm_spot_color[s,r,c] = spot_color[s,r,c] / norm_factor[r,c]```.
    """
    if not utils.errors.check_shape(hist_values, hist_counts.shape[:1]):
        raise utils.errors.ShapeError('hist_values', hist_values.shape, hist_counts.shape[:1])
    # if only one value provided, turn to a list
    if isinstance(thresh_intensities, (float, int)):
        thresh_intensities = [thresh_intensities]
    if isinstance(thresh_probs, (float, int)):
        thresh_probs = [thresh_probs]
    if not utils.errors.check_shape(np.array(thresh_intensities), np.array(thresh_probs).shape):
        raise utils.errors.ShapeError('thresh_intensities', np.array(thresh_intensities).shape,
                                      np.array(thresh_probs).shape)

    # sort thresholds and check that thresh_probs descend as thresh_intensities increase
    ind = np.argsort(thresh_intensities)
    thresh_intensities = np.array(thresh_intensities)[ind]
    thresh_probs = np.array(thresh_probs)[ind]
    if not np.all(np.diff(thresh_probs) <= 0):
        raise ValueError(f"thresh_probs given, {thresh_probs}, do not all descend as thresh_intensities,"
                         f" {thresh_intensities}, increase.")

    n_rounds, n_channels = hist_counts.shape[1:]
    norm_factor = np.zeros((n_rounds, n_channels))
    for r_ind in range(n_rounds):
        if method.lower() == 'single':
            r = np.arange(n_rounds)
        elif method.lower() == 'separate':
            r = r_ind
        else:
            raise ValueError(f"method given was {method} but should be either 'single' or 'separate'")
        for b in range(n_channels):
            hist_counts_rb = np.sum(hist_counts[:, r, b].reshape(hist_values.shape[0], -1), axis=1)
            # if not np.int32, get error in windows when cumsum goes negative.
            cum_sum_rb = np.cumsum(hist_counts_rb.astype(np.int64))
            n_pixels = cum_sum_rb[-1]
            norm_factor_rb = -np.inf
            for thresh_intensity, thresh_prob in zip(thresh_intensities, thresh_probs):
                prob = np.sum(hist_counts_rb[hist_values >= thresh_intensity * norm_factor_rb]) / n_pixels
                if prob > thresh_prob:
                    norm_factor_rb = hist_values[np.where(cum_sum_rb > (1 - thresh_prob) * n_pixels)[0][1]
                                     ] / thresh_intensity
            norm_factor[r, b] = norm_factor_rb
        if r_ind == 0 and method.lower() == 'single':
            break

    return norm_factor


def get_bled_codes(gene_codes: np.ndarray, bleed_matrix: np.ndarray) -> np.ndarray:
    """
    This gets ```bled_codes``` such that the spot_color of a gene ```g``` in round ```r``` is expected to be a constant
    multiple of ```bled_codes[g, r]```.
    This function should be run with full bleed_matrix with any rounds/channels/dyes outside those using set to nan.
    Otherwise, will get confusion with dye indices in `gene_codes` being outside size of `bleed_matrix`.

    !!! note
        All bled_codes returned with an L2 norm of 1 when summed over all rounds and channels
        with any nan values assumed to be 0.

    Args:
        gene_codes: ```int [n_genes x n_rounds]```.
            ```gene_codes[g, r]``` indicates the dye that should be present for gene ```g``` in round ```r```.
        bleed_matrix: ```float [n_rounds x n_channels x n_dyes]```.
            Expected intensity of dye ```d``` in round ```r``` is a constant multiple of ```bleed_matrix[r, :, d]```.

    Returns:
        ```float [n_genes x n_rounds x n_channels]```.
            ```bled_codes``` such that ```spot_color``` of a gene ```g```
            in round ```r``` is expected to be a constant multiple of ```bled_codes[g, r]```.
    """
    n_genes = gene_codes.shape[0]
    n_rounds, n_channels, n_dyes = bleed_matrix.shape
    if not utils.errors.check_shape(gene_codes, [n_genes, n_rounds]):
        raise utils.errors.ShapeError('gene_codes', gene_codes.shape, (n_genes, n_rounds))
    if gene_codes.max() >= n_dyes:
        ind_1, ind_2 = np.where(gene_codes == gene_codes.max())
        raise ValueError(f"gene_code for gene {ind_1[0]}, round {ind_2[0]} has a dye with index {gene_codes.max()}"
                         f" but there are only {n_dyes} dyes.")
    if gene_codes.min() < 0:
        ind_1, ind_2 = np.where(gene_codes == gene_codes.min())
        raise ValueError(f"gene_code for gene {ind_1[0]}, round {ind_2[0]} has a dye with a negative index:"
                         f" {gene_codes.min()}")

    bled_codes = np.zeros((n_genes, n_rounds, n_channels))
    for g in range(n_genes):
        for r in range(n_rounds):
            for c in range(n_channels):
                bled_codes[g, r, c] = bleed_matrix[r, c, gene_codes[g, r]]

    # Give all bled codes an L2 norm of 1 assuming any nan values are 0
    norm_factor = np.expand_dims(np.linalg.norm(np.nan_to_num(bled_codes), axis=(1, 2)), (1, 2))
    norm_factor[norm_factor == 0] = 1   # For genes with no dye in any rounds, this avoids blow up on next line
    bled_codes = bled_codes / norm_factor
    return bled_codes


def get_gene_efficiency(spot_colors: np.ndarray, spot_gene_no: np.ndarray, gene_codes: np.ndarray,
                        bleed_matrix: np.ndarray, min_spots,
                        max_gene_efficiency: float = np.inf,
                        min_gene_efficiency: float = 0,
                        min_gene_efficiency_factor: float = 1) -> np.ndarray:
    """
    `gene_efficiency[g,r]` gives the expected intensity of gene `g` in round `r` compared to that expected
    by the `bleed_matrix`. It is computed based on the average of all `spot_colors` assigned to that gene.

    Args:
        spot_colors: `float [n_spots x n_rounds x n_channels]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        spot_gene_no: `int [n_spots]`.
            Gene each spot was assigned to.
        gene_codes: `int [n_genes, n_rounds]`.
            `gene_codes[g, r]` indicates the dye that should be present for gene `g` in round `r`.
        bleed_matrix: `float [n_rounds x n_channels x n_dyes]`.
            For a spot, `s` matched to gene with dye `d` in round `r`, we expect `spot_colors[s, r]`",
            to be a constant multiple of `bleed_matrix[r, :, d]`"
        min_spots: If number of spots assigned to a gene less than or equal to this, `gene_efficiency[g]=1`
            for all rounds. Typical = 30.
        max_gene_efficiency: Maximum allowed gene efficiency, i.e. any one round can be at most this times more
            important than the median round for every gene. Typical = 6.
        min_gene_efficiency: At most `ceil(min_gene_efficiency_factor * n_rounds)` rounds can have
            `gene_efficiency` below `min_gene_efficiency` for any given gene. Typical = 0.05
        min_gene_efficiency_factor: At most `ceil(min_gene_efficiency_factor * n_rounds)` rounds can have
            `gene_efficiency` below `min_gene_efficiency` for any given gene. Typical = 0.2

    Returns:
        `float [n_genes x n_rounds]`.
            `gene_efficiency[g,r]` gives the expected intensity of gene `g` in round `r` compared to that expected
            by the `bleed_matrix`.
    """
    # Check n_spots, n_rounds, n_channels, n_genes consistent across all variables.
    if not utils.errors.check_shape(spot_colors[0], bleed_matrix[:, :, 0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      (spot_colors.shape[0],) + bleed_matrix[:, :, 0].shape)
    if not utils.errors.check_shape(spot_colors[:, 0, 0], spot_gene_no.shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      spot_gene_no.shape + bleed_matrix[:, :, 0].shape)
    n_genes, n_rounds = gene_codes.shape
    if not utils.errors.check_shape(spot_colors[0, :, 0].squeeze(), gene_codes[0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      spot_gene_no.shape + (n_rounds,) + (bleed_matrix.shape[1],))

    gene_no_oob = [val for val in spot_gene_no if val < 0 or val >= n_genes]
    if len(gene_no_oob) > 0:
        raise utils.errors.OutOfBoundsError("spot_gene_no", gene_no_oob[0], 0, n_genes - 1)

    gene_efficiency = np.ones([n_genes, n_rounds])
    n_min_thresh = int(np.ceil(min_gene_efficiency_factor * n_rounds))
    for g in range(n_genes):
        use = spot_gene_no == g
        if np.sum(use) > min_spots:
            round_strength = np.zeros([np.sum(use), n_rounds])
            for r in range(n_rounds):
                dye_ind = gene_codes[g, r]
                # below is equivalent to MATLAB spot_colors / bleed_matrix.
                round_strength[:, r] = np.linalg.lstsq(bleed_matrix[r, :, dye_ind:dye_ind + 1],
                                                       spot_colors[use, r].transpose(), rcond=None)[0]

            # find a reference round for each gene as that with median strength.
            av_round_strength = np.median(round_strength, 0)
            av_round = np.abs(av_round_strength - np.median(av_round_strength)).argmin()

            # for each spot, find strength of each round relative to strength in
            # av_round. Need relative strength not absolute strength
            # because expect spot color to be constant multiple of bled code.
            # So for all genes, gene_efficiency[g, av_round] = 1 but av_round is different between genes.

            # Only use spots whose strength in RefRound is positive.
            use = round_strength[:, av_round] > 0
            if np.sum(use) > min_spots:
                relative_round_strength = round_strength[use] / np.expand_dims(round_strength[use, av_round], 1)
                # Only use spots with gene efficiency below the maximum allowed.
                below_max = np.max(relative_round_strength, axis=1) < max_gene_efficiency
                # Only use spots with at most n_min_thresh rounds below the minimum.
                above_min = np.sum(relative_round_strength < min_gene_efficiency, axis=1) <= n_min_thresh
                use = np.array([below_max, above_min]).all(axis=0)
                if np.sum(use) > min_spots:
                    gene_efficiency[g] = np.median(relative_round_strength[use], 0)

    # set negative values to 0
    gene_efficiency = np.clip(gene_efficiency, 0, np.inf)
    return gene_efficiency
