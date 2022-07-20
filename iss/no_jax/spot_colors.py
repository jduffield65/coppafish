from typing import Optional, List
import numpy as np
from tqdm import tqdm
from .. import utils
from ..setup import NotebookPage


def apply_transform(yxz: np.ndarray, transform: np.ndarray, tile_centre: np.ndarray, z_scale: float) -> np.ndarray:
    """
    This transforms the coordinates yxz based on an affine transform.
    E.g. to find coordinates of spots on the same tile but on a different round and channel.

    Args:
        yxz: ```int [n_spots x 3]```.
            ```yxz[i, :2]``` are the non-centered yx coordinates in ```yx_pixels``` for spot ```i```.
            ```yxz[i, 2]``` is the non-centered z coordinate in ```z_pixels``` for spot ```i```.
            E.g. these are the coordinates stored in ```nb['find_spots']['spot_details']```.
        transform: ```float [4 x 3]```.
            Affine transform to apply to ```yxz```, once centered and z units changed to ```yx_pixels```.
            ```transform[3, 2]``` is approximately the z shift in units of ```yx_pixels```.
            E.g. this is one of the transforms stored in ```nb['register']['transform']```.
        tile_centre: ```float [3]```.
            ```tile_centre[:2]``` are yx coordinates in ```yx_pixels``` of the centre of the tile that spots in
            ```yxz``` were found on.
            ```tile_centre[2]``` is the z coordinate in ```z_pixels``` of the centre of the tile.
            E.g. for tile of ```yxz``` dimensions ```[2048, 2048, 51]```, ```tile_centre = [1023.5, 1023.5, 25]```
            Each entry in ```tile_centre``` must be an integer multiple of ```0.5```.
        z_scale: Scale factor to multiply z coordinates to put them in units of yx pixels.
            I.e. ```z_scale = pixel_size_z / pixel_size_yx``` where both are measured in microns.
            typically, ```z_scale > 1``` because ```z_pixels``` are larger than the ```yx_pixels```.

    Returns:
        ```int [n_spots x 3]```.
            ```yxz_transform``` such that
            ```yxz_transform[i, [1,2]]``` are the transformed non-centered yx coordinates in ```yx_pixels```
            for spot ```i```.
            ```yxz_transform[i, 2]``` is the transformed non-centered z coordinate in ```z_pixels``` for spot ```i```.
    """
    if (utils.round_any(tile_centre, 0.5) == tile_centre).min() == False:
        raise ValueError(f"tile_centre given, {tile_centre}, is not a multiple of 0.5 in each dimension.")
    yxz_pad = np.pad((yxz - tile_centre) * [1, 1, z_scale], [(0, 0), (0, 1)], constant_values=1)
    yxz_transform = np.matmul(yxz_pad, transform)
    yxz_transform = np.round((yxz_transform / [1, 1, z_scale]) + tile_centre).astype(np.int16)
    return yxz_transform


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

        `invalid_value = -nbp_basic.tile_pixel_value_shift` is the lowest possible value saved in the npy file minus 1
        (due to clipping in extract step), so it is impossible for spot_color to be this.
        Hence I use this as integer nan.
        It will be `invalid_value` if the registered coordinate of spot `s` is outside the tile in round `r`, channel
        `c`.

        Note `n_rounds`/`n_channels` are total number of rounds/channels in raw nd2 file as saved in `nbp_basic`.
    """
    invalid_value = -nbp_basic.tile_pixel_value_shift  # impossible for spot_color to be this.
    if use_rounds is None:
        use_rounds = nbp_basic.use_rounds
    if use_channels is None:
        use_channels = nbp_basic.use_channels
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
    tile_sz = [nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz]
    if not nbp_basic.is_3d:
        tile_sz[2] = 1

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10000
    # note using nan means can't use integer even though data is integer
    spot_colors = np.ones((n_spots, nbp_basic.n_rounds, nbp_basic.n_channels), dtype=int) * invalid_value
    n_images = len(use_rounds) * len(nbp_basic.use_channels)
    with tqdm(total=n_images, disable=no_verbose) as pbar:
        pbar.set_description(f"Reading {n_spots} spot_colors found on tile {t} from npy files.")
        for r in use_rounds:
            for c in use_channels:
                pbar.set_postfix({'round': r, 'channel': c})
                if transforms[t, r, c, 0, 0] == 0:
                    raise ValueError(f"Transform for tile {t}, round {r}, channel {c} is zero:"
                                     f"\n{transforms[t, r, c]}")
                yxz_transform = apply_transform(yxz_base, transforms[t, r, c], nbp_basic.tile_centre, z_scale)
                in_range = np.logical_and(np.min(yxz_transform >= [0, 0, 0], axis=1),
                                          np.min(yxz_transform < tile_sz, axis=1))  # set color to nan if out range
                if in_range.any():
                    yxz_transform = yxz_transform[in_range]

                    # only load in section of image required for speed.
                    yxz_min = np.min(yxz_transform, axis=0)
                    yxz_max = np.max(yxz_transform, axis=0)
                    load_y = np.arange(yxz_min[0], yxz_max[0] + 1)
                    load_x = np.arange(yxz_min[1], yxz_max[1] + 1)
                    load_z = np.arange(yxz_min[2], yxz_max[2] + 1)
                    image = utils.npy.load_tile(nbp_file, nbp_basic, t, r, c,
                                                [load_y, load_x, load_z][:2+nbp_basic.is_3d])

                    yxz_transform = yxz_transform - yxz_min  # shift yxz so load in correct colors from cropped image.
                    if image.ndim == 3:
                        spot_colors[in_range, r, c] = image[yxz_transform[:, 0], yxz_transform[:, 1],
                                                            yxz_transform[:, 2]]
                    else:
                        spot_colors[in_range, r, c] = image[yxz_transform[:, 0], yxz_transform[:, 1]]
                pbar.update(1)
    return spot_colors
