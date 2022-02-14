import numpy as np
from .. import pcr, utils
from ..setup.notebook import NotebookPage
from typing import List, Optional


def get_spot_colors(yxz_base: np.ndarray, t: int, transforms: np.ndarray, nbp_file: NotebookPage,
                    nbp_basic: NotebookPage, use_rounds: Optional[List[int]] = None,
                    use_channels: Optional[List[int]] = None) -> np.ndarray:
    """
    Takes some spots found on the reference round, and computes the corresponding spot intensity
    in specified imaging rounds/channels.

    Args:
        yxz_base: `int [n_spots x 3]`.
            Local yxz coordinates of spots found in the reference round/reference channel of tile `t`
            yx coordinates are in units of `yx_pixels`. z coordinates are in units of `z_pixels`.
        t: Tile that spots were found on.
        transforms: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.
            `transforms[t, r, c]` is the affine transform to get from tile `t`, `ref_round`, `ref_channel` to
            tile `t`, round `r`, channel `c`.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        use_rounds: `int [n_use_rounds]`.
            Rounds you would like to find the `spot_color` for.
            Error will raise if transform is zero for particular round.
            If `None`, all rounds in `nbp_basic.use_rounds` used.
        use_channels: `int [n_use_channels]`.
            Channels you would like to find the `spot_color` for.
            Error will raise if transform is zero for particular channel.
            If `None`, all channels in `nbp_basic.use_channels` used.

    Returns:
        `float [n_spots x n_rounds x n_channels]`.

        `spot_colors[s, r, c]` is the spot color for spot `s` in round `r`, channel `c`.

        It will be nan if the registered coordinate of spot `s` is outside the tile in round `r`, channel `c` or
        if  `r`/`c` is not in `use_rounds`/`use_channels`.

        Note `n_rounds`/`n_channels` are total number of rounds/channels in raw nd2 file as saved in `nbp_basic`.
    """
    if use_rounds is None:
        use_rounds = nbp_basic.use_rounds
    if use_channels is None:
        use_channels = nbp_basic.use_channels
    z_scale = nbp_basic.pixel_size_z/nbp_basic.pixel_size_xy
    tile_sz = [nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz]
    if not nbp_basic.is_3d:
        tile_sz[2] = 1

    # note using nan means can't use integer even though data is integer
    spot_colors = np.ones((yxz_base.shape[0], nbp_basic.n_rounds, nbp_basic.n_channels)) * np.nan
    for r in use_rounds:
        for c in use_channels:
            if transforms[t, r, c, 0, 0] == 0:
                raise ValueError(f"Transform for tile {t}, round {r}, channel {c} is zero:"
                                 f"\n{transforms[t, r, c]}")
            yxz_transform = pcr.apply_transform(yxz_base, transforms[t, r, c], nbp_basic.tile_centre, z_scale)
            in_range = np.logical_and(np.min(yxz_transform >= [0, 0, 0], axis=1),
                                      np.min(yxz_transform < tile_sz, axis=1))  # set color to nan if out range
            image = utils.tiff.load_tile(nbp_file, nbp_basic, t, r, c)
            if nbp_basic.is_3d:
                spot_colors[in_range, r, c] = image[yxz_transform[in_range, 0], yxz_transform[in_range, 1],
                                                    yxz_transform[in_range, 2]]
            else:
                spot_colors[in_range, r, c] = image[yxz_transform[in_range, 0], yxz_transform[in_range, 1]]
    return spot_colors
