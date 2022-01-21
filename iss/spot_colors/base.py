import numpy as np
from .. import pcr, utils


def get_spot_colors(yxz_base, t, transforms, nbp_file, nbp_basic, use_rounds=None, use_channels=None):
    """
    Takes some spots found on the reference round, and computes the corresponding spot intensity
    in specified imaging rounds/channels.

    :param yxz_base: numpy integer array [n_spots x 3]
        local yxz coordinates of spots found in the reference round/reference channel of tile t
        yx coordinates are in units of yx pixels. z coordinates are in units of z pixels
    :param t: integer
        tile that spots were found on
    :param transforms: numpy float array [n_tiles x n_rounds x n_channels x 4 x 3]
        transforms[t, r, c] is the affine transform to get from tile t, ref_round, ref_channel to
        tile t, round r, channel c
    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param use_rounds: list, optional
        rounds you would like to find the spot color for
        (error will raise if transform is zero for particular round)
        default: None, meaning use all rounds in nbp_basic['use_rounds']
    :param use_channels:
        channels you would like to find the spot color for
        (error will raise if transform is zero for particular channel)
        default: None, meaning use all channels in nbp_basic['use_channels']
    :return:
        spot_colors: numpy float array [n_spots x n_rounds x n_channels]
            spot_colors[s, r, c] is the spot color for spot s in round r, channel c.
            It will be nan if the registered coordinate of spot s is outside the tile in round r, channel c or
            if  r/c is not in use_rounds/use_channels.
            Note n_rounds/n_channels are total number of rounds/channels in raw nd2 file as saved in nbp_basic.
    """
    if use_rounds is None:
        use_rounds = nbp_basic['use_rounds']
    if use_channels is None:
        use_channels = nbp_basic['use_channels']
    z_scale = nbp_basic['pixel_size_z']/nbp_basic['pixel_size_xy']
    tile_sz = [nbp_basic['tile_sz'], nbp_basic['tile_sz'], nbp_basic['nz']]
    if not nbp_basic['3d']:
        tile_sz[2] = 1

    # note using nan means can't use integer even though data is integer
    spot_colors = np.ones((yxz_base.shape[0], nbp_basic['n_rounds'], nbp_basic['n_channels'])) * np.nan
    for r in use_rounds:
        for c in use_channels:
            if transforms[t, r, c, 0, 0] == 0:
                raise ValueError(f"Transform for tile {t}, round {r}, channel {c} is zero:"
                                 f"\n{transforms[t, r, c]}")
            yxz_transform = pcr.apply_transform(yxz_base, transforms[t, r, c], nbp_basic['tile_centre'], z_scale)
            in_range = np.logical_and(np.min(yxz_transform >= [0, 0, 0], axis=1),
                                      np.min(yxz_transform < tile_sz, axis=1))  # set color to nan if out range
            image = utils.tiff.load_tile(nbp_file, nbp_basic, t, r, c)
            if nbp_basic['3d']:
                spot_colors[in_range, r, c] = image[yxz_transform[in_range, 0], yxz_transform[in_range, 1],
                                                    yxz_transform[in_range, 2]]
            else:
                spot_colors[in_range, r, c] = image[yxz_transform[in_range, 0], yxz_transform[in_range, 1]]
    return spot_colors
