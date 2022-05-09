import numpy as np
from .. import pcr, utils
from ..setup.notebook import NotebookPage
from typing import List, Optional, Tuple, Union


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
        `int [n_spots x n_rounds x n_channels]`.

        `spot_colors[s, r, c]` is the spot color for spot `s` in round `r`, channel `c`.

        nan_value = -nbp_basic.tile_pixel_value_shift - 1 is the lowest possible value saved in the tiff file minus 1,
        so it is impossible for spot_color to be this. Hence I use this as integer nan.
        It will be `nan_value` if the registered coordinate of spot `s` is outside the tile in round `r`, channel `c` or
        if  `r`/`c` is not in `use_rounds`/`use_channels`.

        Note `n_rounds`/`n_channels` are total number of rounds/channels in raw nd2 file as saved in `nbp_basic`.
    """
    nan_value = -nbp_basic.tile_pixel_value_shift - 1  # impossible for spot_color to be this.
    if use_rounds is None:
        use_rounds = nbp_basic.use_rounds
    if use_channels is None:
        use_channels = nbp_basic.use_channels
    z_scale = nbp_basic.pixel_size_z/nbp_basic.pixel_size_xy
    tile_sz = [nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz]
    if not nbp_basic.is_3d:
        tile_sz[2] = 1

    # note using nan means can't use integer even though data is integer
    spot_colors = np.ones((yxz_base.shape[0], nbp_basic.n_rounds, nbp_basic.n_channels), dtype=int) * nan_value
    for r in use_rounds:
        for c in use_channels:
            if transforms[t, r, c, 0, 0] == 0:
                raise ValueError(f"Transform for tile {t}, round {r}, channel {c} is zero:"
                                 f"\n{transforms[t, r, c]}")
            yxz_transform = pcr.apply_transform(yxz_base, transforms[t, r, c], nbp_basic.tile_centre, z_scale)
            in_range = np.logical_and(np.min(yxz_transform >= [0, 0, 0], axis=1),
                                      np.min(yxz_transform < tile_sz, axis=1))  # set color to nan if out range
            yxz_transform = yxz_transform[in_range]

            # only load in section of image required for speed.
            yxz_min = np.min(yxz_transform, axis=0)
            yxz_max = np.max(yxz_transform, axis=0)
            load_y = np.arange(yxz_min[0], yxz_max[0] + 1)
            load_x = np.arange(yxz_min[1], yxz_max[1] + 1)
            load_z = np.arange(yxz_min[2], yxz_max[2] + 1)
            image = utils.tiff.load_tile(nbp_file, nbp_basic, t, r, c, load_y, load_x, load_z)

            yxz_transform = yxz_transform - yxz_min  # shift yxz so load in correct colors from cropped image.
            if image.ndim == 3:
                spot_colors[in_range, r, c] = image[yxz_transform[:, 0], yxz_transform[:, 1],
                                                    yxz_transform[:, 2]]
            else:
                spot_colors[in_range, r, c] = image[yxz_transform[:, 0], yxz_transform[:, 1]]
    return spot_colors


def get_all_pixel_colors(t: int, transforms: np.ndarray, nbp_file: NotebookPage,
                         nbp_basic: NotebookPage,
                         z_planes: Union[int, np.ndarray, List] = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds colors for every pixel in a tile.
    Keeping only pixels within tile bounds on each round and channel in nbp_basic.use_rounds/channels.

    !!! note
        Returned pixel colors have dimension `n_pixels x len(nbp_basic.use_rounds) x len(nbp_basic.use_channels)` not
        `n_pixels x nbp_basic.n_rounds x nbp_basic.n_channels`.

    Args:
        t: Tile that spots were found on.
        transforms: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.
            `transforms[t, r, c]` is the affine transform to get from tile `t`, `ref_round`, `ref_channel` to
            tile `t`, round `r`, channel `c`.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        z_planes: z_planes to load all pixels for.

    Returns:
        - ```pixel_colors``` - `int [n_pixels x n_rounds_use x n_channels_use]`.
            `pixel_colors[s, r, c]` is the color at `pixel_yxz[s]` in round `use_rounds[r]`, channel `use_channels[c]`.
        - ```pixel_yxz``` - `float [n_pixels x 3]`.
            Local yxz coordinates of pixels in the reference round/reference channel of tile `t`
            yx coordinates are in units of `yx_pixels`. z coordinates are in units of `z_pixels`.
    """
    nan_value = -nbp_basic.tile_pixel_value_shift - 1
    if isinstance(z_planes, int):
        z_planes = np.array([z_planes])
    pixel_yxz_all = np.array(np.meshgrid(np.arange(nbp_basic.tile_sz),
                                         np.arange(nbp_basic.tile_sz), np.arange(1))).T.reshape(-1, 3)
    pixel_colors = np.zeros((0, len(nbp_basic.use_rounds), len(nbp_basic.use_channels)), dtype=int)
    pixel_yxz = np.zeros((0, 3), dtype=int)
    for z in z_planes:
        pixel_yxz_all[:, 2] = z
        pixel_colors_all = get_spot_colors(pixel_yxz_all, t, transforms, nbp_file, nbp_basic)
        # only keep used rounds/channels to save memory.
        pixel_colors_all = pixel_colors_all[np.ix_(np.arange(pixel_colors_all.shape[0]), nbp_basic.use_rounds,
                                                   nbp_basic.use_channels)]
        # only keep spots in all rounds/channels meaning no nan values
        keep = ~np.any(pixel_colors_all == nan_value, axis=(1, 2))
        pixel_colors = np.append(pixel_colors, pixel_colors_all[keep].astype(int), axis=0)
        pixel_yxz = np.append(pixel_yxz, pixel_yxz_all[keep], axis=0)
    utils.errors.check_spot_color_nan(pixel_colors, nbp_basic)
    return pixel_colors, pixel_yxz
