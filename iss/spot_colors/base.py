import numpy as np
from .. import pcr, utils
from ..setup.notebook import NotebookPage
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import jax.numpy as jnp


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
        for r in use_rounds:
            for c in use_channels:
                pbar.set_description(f"Reading {n_spots} spot_colors found on tile {t} from tiff files.")
                pbar.set_postfix({'round': r, 'channel': c})
                if transforms[t, r, c, 0, 0] == 0:
                    raise ValueError(f"Transform for tile {t}, round {r}, channel {c} is zero:"
                                     f"\n{transforms[t, r, c]}")
                yxz_transform = pcr.apply_transform(yxz_base, transforms[t, r, c], nbp_basic.tile_centre, z_scale)
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


def get_spot_colors_jax(yxz_base: jnp.ndarray, t: int, transforms: jnp.ndarray, nbp_file: NotebookPage,
                        nbp_basic: NotebookPage, use_rounds: Optional[List[int]] = None,
                        use_channels: Optional[List[int]] = None,
                        return_in_bounds: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, jnp.ndarray]]:
    """
    Takes some spots found on the reference round, and computes the corresponding spot intensity
    in specified imaging rounds/channels.
    By default, will run on `nbp_basic.use_rounds` and `nbp_basic.use_channels`.

    !!! note
        Returned spot colors have dimension `n_spots x len(nbp_basic.use_rounds) x len(nbp_basic.use_channels)` not
        `n_pixels x nbp_basic.n_rounds x nbp_basic.n_channels`.

    !!! note
        `invalid_value = -nbp_basic.tile_pixel_value_shift` is the lowest possible value saved in the npy file
        minus 1 (due to clipping in extract step), so it is impossible for spot_color to be this.
        Hence I use this as integer nan. It will be `invalid_value` if the registered coordinate of
        spot `s` is outside the tile in round `r`, channel `c`.

    Args:
        yxz_base: `int16 [n_spots x 3]`.
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
        return_in_bounds: if `True`, then only `spot_colors` which are within the tile bounds in all
            `use_rounds` / `use_channels` will be returned.
            The corresponding `yxz_base` coordinates will also be returned in this case.
            Otherwise, `spot_colors` will be returned for all the given `yxz_base` but if spot `s` is out of bounds on
            round `r`, channel `c`, then `spot_colors[s, r, c] = invalid_value = -nbp_basic.tile_pixel_value_shift`.
            This is the only scenario for which `spot_colors = invalid_value` due to clipping in the extract step.


    Returns:
        - `spot_colors` - `int32 [n_spots x n_rounds_use x n_channels_use]` or
            `int32 [n_spots_in_bounds x n_rounds_use x n_channels_use]`.
            `spot_colors[s, r, c]` is the spot color for spot `s` in round `use_rounds[r]`, channel `use_channels[c]`.
        - `yxz_base` - `int16 [n_spots_in_bounds x 3]`.
            If `return_in_bounds`, the `yxz_base` corresponding to spots in bounds for all `use_rounds` / `use_channels`
            will be returned. It is likely that `n_spots_in_bounds` won't be the same as `n_spots`.
    """
    if use_rounds is None:
        use_rounds = nbp_basic.use_rounds
    if use_channels is None:
        use_channels = nbp_basic.use_channels
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10000
    # note using nan means can't use integer even though data is integer
    n_use_rounds = len(use_rounds)
    n_use_channels = len(use_channels)
    # spots outside tile bounds on particular r/c will initially be set to 0.
    spot_colors = np.zeros((n_spots, n_use_rounds, n_use_channels), dtype=np.int32)
    tile_centre = jnp.array(nbp_basic.tile_centre)
    if not nbp_basic.is_3d:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        tile_sz = jnp.array([nbp_basic.tile_sz, nbp_basic.tile_sz, 1], dtype=jnp.int16)
    else:
        tile_sz = jnp.array([nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz], dtype=jnp.int16)

    with tqdm(total=n_use_rounds * n_use_channels, disable=no_verbose) as pbar:
        for r in range(n_use_rounds):
            if not nbp_basic.is_3d:
                # If 2D, load in all channels first
                image_all_channels = np.load(nbp_file.tile[t][use_rounds[r]], mmap_mode='r')
            for c in range(n_use_channels):
                transform_rc = transforms[t, use_rounds[r], use_channels[c]]
                pbar.set_description(f"Reading {n_spots} spot_colors found on tile {t} from tiff files.")
                pbar.set_postfix({'round': use_rounds[r], 'channel': use_channels[c]})
                if transform_rc[0, 0] == 0:
                    raise ValueError(
                        f"Transform for tile {t}, round {use_rounds[r]}, channel {use_channels[c]} is zero:"
                        f"\n{transform_rc}")
                yxz_transform, in_range = pcr.apply_transform_jax(yxz_base, transform_rc, tile_centre, z_scale, tile_sz)
                yxz_transform = np.asarray(yxz_transform)
                in_range = np.asarray(in_range)
                yxz_transform = yxz_transform[in_range]
                if yxz_transform.shape[0] > 0:
                    # Read in the shifted uint16 colors here, and remove shift later.
                    if nbp_basic.is_3d:
                        spot_colors[in_range, r, c] = utils.npy.load_tile(nbp_file, nbp_basic, t, use_rounds[r],
                                                                          use_channels[c], yxz_transform,
                                                                          apply_shift=False)
                    else:
                        spot_colors[in_range, r, c] = image_all_channels[use_channels[c]][
                            tuple(np.asarray(yxz_transform[:, i]) for i in range(2))]
                pbar.update(1)
    # Remove shift so now spots outside bounds have color equal to - nbp_basic.tile_pixel_shift_value.
    # It is impossible for any actual spot color to be this due to clipping at the extract stage.
    spot_colors = spot_colors - nbp_basic.tile_pixel_value_shift
    invalid_value = - nbp_basic.tile_pixel_value_shift
    if return_in_bounds:
        good = ~np.any(spot_colors == invalid_value, axis=(1, 2))
        return spot_colors[good], yxz_base[good]
    else:
        return spot_colors
