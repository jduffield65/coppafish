from functools import partial
import numpy as np
from .. import utils
from ..setup.notebook import NotebookPage
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import jax.numpy as jnp
import jax


def apply_transform_single(yxz: jnp.ndarray, transform: jnp.ndarray, tile_sz: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
    yxz_pad = jnp.pad(yxz, [(0, 1)], constant_values=1)
    yxz_transform = jnp.matmul(yxz_pad, transform)
    yxz_transform = jnp.round(yxz_transform).astype(jnp.int16)
    in_range = jnp.logical_and((yxz_transform >= jnp.array([0, 0, 0])).all(),
                               (yxz_transform < tile_sz).all())  # set color to nan if out range
    return yxz_transform, in_range


@partial(jax.jit, static_argnums=3)
def apply_transform(yxz: jnp.ndarray, transform: jnp.ndarray, tile_sz: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This transforms the coordinates yxz based on an affine transform.
    E.g. to find coordinates of spots on the same tile but on a different round and channel.
    Args:
        yxz: ```int16 [n_spots x 3]```.
            ```yxz[i, :2]``` are the non-centered yx coordinates in ```yx_pixels``` for spot ```i```.
            ```yxz[i, 2]``` is the non-centered z coordinate in ```z_pixels``` for spot ```i```.
            E.g. these are the coordinates stored in ```nb['find_spots']['spot_details']```.
        transform: ```float [4 x 3]```.
            Affine transform to apply to ```yxz```, once centered and z units changed to ```yx_pixels```.
            ```transform[3, 2]``` is approximately the z shift in units of ```yx_pixels```.
            E.g. this is one of the transforms stored in ```nb['register']['transform']```.
        tile_sz: ```int16 [3]```.
            YXZ dimensions of tile

    Returns:
        - `yxz_transform` - ```int [n_spots x 3]```.
            ```yxz_transform``` such that
            ```yxz_transform[i, [1,2]]``` are the transformed non-centered yx coordinates in ```yx_pixels```
            for spot ```i```.
            ```yxz_transform[i, 2]``` is the transformed non-centered z coordinate in ```z_pixels``` for spot ```i```.
        - ```in_range``` - ```bool [n_spots]```.
            Whether spot s was in the bounds of the tile when transformed to round `r`, channel `c`.
    """
    return jax.vmap(apply_transform_single, in_axes=(0, None, None), out_axes=(0, 0))(yxz, transform, tile_sz)


def get_spot_colors(yxz_base: jnp.ndarray, t: int, transforms: jnp.ndarray, nbp_file: NotebookPage,
                    nbp_basic: NotebookPage, nbp_extract: NotebookPage, use_rounds: Optional[List[int]] = None,
                    use_channels: Optional[List[int]] = None, return_in_bounds: bool = False,
                    bg_scale: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, jnp.ndarray]]:
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
        nbp_file: `file_names` notebook page.
        nbp_basic: `basic_info` notebook page.
        nbp_extract: `extract` notebook page.
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
        bg_scale: 'float [n_tiles x n_rounds x n_channels_use x 2]' normalisation factor for each
            of the tiles/rounds and channels. bg_round[t, c] * bg_scale_offset[t, r, c]
            will equalise the background brightness profile to the same as that of tile t, round r, channel c. If None,
            no normalisation will be performed.

    Returns:
        - `spot_colors` - `int32 [n_spots x n_rounds_use x n_channels_use]` or
            `int32 [n_spots_in_bounds x n_rounds_use x n_channels_use]`.
            `spot_colors[s, r, c]` is the spot color for spot `s` in round `use_rounds[r]`, channel `use_channels[c]`.
        - `yxz_base` - `int16 [n_spots_in_bounds x 3]`.
            If `return_in_bounds`, the `yxz_base` corresponding to spots in bounds for all `use_rounds` / `use_channels`
            will be returned. It is likely that `n_spots_in_bounds` won't be the same as `n_spots`.
        - `bg_colours` - `int32 [n_spots x n_channels_use]`.
    """
    if bg_scale is not None:
        assert nbp_basic.use_preseq, "Can't subtract background if preseq round doesn't exist!"
        use_bg = True
    else:
        use_bg = False
    if use_rounds is None:
        use_rounds = nbp_basic.use_rounds
    if use_channels is None:
        use_channels = nbp_basic.use_channels

    if nbp_extract.file_type == '.npy':
        from ..utils.npy import load_tile
    elif nbp_extract.file_type == '.zarr':
        from ..utils.zarray import load_tile

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10000
    # note using nan means can't use integer even though data is integer
    n_use_rounds = len(use_rounds)
    n_use_channels = len(use_channels)
    # spots outside tile bounds on particular r/c will initially be set to 0.
    # TODO: Should this be jax.numpy?
    spot_colors = np.zeros((n_spots, n_use_rounds, n_use_channels), dtype=np.int32)
    if use_bg:
        bg_colours = np.zeros((n_spots, n_use_channels), dtype=np.int32)
    if not nbp_basic.is_3d:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        tile_sz = jnp.array([nbp_basic.tile_sz, nbp_basic.tile_sz, 1], dtype=jnp.int16)
    else:
        tile_sz = jnp.array([nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz], dtype=jnp.int16)

    with tqdm(total=n_use_rounds * n_use_channels, disable=no_verbose) as pbar:
        pbar.set_description(f"Reading {n_spots} spot_colors found on tile {t} from npy files")
        for r in range(n_use_rounds):
            if not nbp_basic.is_3d:
                # If 2D, load in all channels first
                image_all_channels = np.load(nbp_file.tile[t][use_rounds[r]], mmap_mode='r')
            for c in range(n_use_channels):
                transform_rc = transforms[t, use_rounds[r], use_channels[c]]
                pbar.set_postfix({'round': use_rounds[r], 'channel': use_channels[c]})
                if transform_rc[0, 0] == 0:
                    raise ValueError(
                        f"Transform for tile {t}, round {use_rounds[r]}, channel {use_channels[c]} is zero:"
                        f"\n{transform_rc}")
                yxz_transform, in_range = apply_transform(yxz_base, transform_rc, tile_sz)
                yxz_transform = np.asarray(yxz_transform)
                in_range = np.asarray(in_range)
                yxz_transform = yxz_transform[in_range]
                if yxz_transform.shape[0] > 0:
                    # Read in the shifted uint16 colors here, and remove shift later.
                    if nbp_basic.is_3d:
                        spot_colors[in_range, r, c] = load_tile(nbp_file, nbp_basic, t, use_rounds[r],
                                                                          use_channels[c], yxz_transform,
                                                                          apply_shift=False)
                    else:
                        spot_colors[in_range, r, c] = image_all_channels[use_channels[c]][
                            tuple(np.asarray(yxz_transform[:, i]) for i in range(2))]
                pbar.update(1)
    if use_bg:
        with tqdm(total=n_use_channels, disable=no_verbose) as pbar:
            pbar.set_description(f"Reading {n_spots} background spot_colors found on tile {t} from npy files")
            for c in range(n_use_channels):
                transform_rc = transforms[t, nbp_basic.pre_seq_round, use_channels[c]]
                pbar.set_postfix({'round': use_rounds[r], 'channel': use_channels[c]})
                if transform_rc[0, 0] == 0:
                    raise ValueError(
                        f"Transform for tile {t}, round {nbp_basic.pre_seq_round}, channel {use_channels[c]} is zero:"
                        f"\n{transform_rc}")
                yxz_transform, in_range = apply_transform(yxz_base, transform_rc, tile_sz)
                yxz_transform = yxz_transform[in_range]
                if yxz_transform.shape[0] > 0:
                    # Read in the shifted uint16 colors here, and remove shift later.
                    if nbp_basic.is_3d:
                        bg_colours[in_range, c] = \
                            load_tile(nbp_file, nbp_basic, t, nbp_basic.pre_seq_round, use_channels[c],
                                                yxz_transform, apply_shift=False)
                pbar.update(1)
        # subtract tile pixel shift value so that bg_colours are in range -15_000 to 50_000 (approx)
        valid = bg_colours > 0
        bg_colours[valid] = bg_colours[valid] - nbp_basic.tile_pixel_value_shift
        # repeat bg_colours so it is the same shape as spot_colors
        bg_colours = np.repeat(bg_colours[:, None, :], n_use_rounds, axis=1)
        bg_colours[valid] = bg_colours[valid] * bg_scale[t, use_rounds, use_channels]
    # Remove shift so now spots outside bounds have color equal to - nbp_basic.tile_pixel_shift_value.
    # It is impossible for any actual spot color to be this due to clipping at the extract stage.
    spot_colors = spot_colors - nbp_basic.tile_pixel_value_shift
    if use_bg and not return_in_bounds:
        spot_colors = spot_colors - bg_colours
        return spot_colors, bg_colours
    invalid_value = -nbp_basic.tile_pixel_value_shift
    if use_bg and return_in_bounds:
        good = ~np.any(spot_colors == invalid_value, axis=(1, 2))
        return spot_colors[good], yxz_base[good], bg_colours[good]
    elif not use_bg and return_in_bounds:
        good = ~np.any(spot_colors == invalid_value, axis=(1, 2))
        return spot_colors[good], yxz_base[good]
    else:
        return spot_colors


def all_pixel_yxz(y_size: int, x_size: int, z_planes: Union[List, int, np.ndarray]) -> jnp.ndarray:
    """
    Returns the yxz coordinates of all pixels on the indicated z-planes of an image.

    Args:
        y_size: number of pixels in y direction of image.
        x_size: number of pixels in x direction of image.
        z_planes: `int [n_z_planes]` z_planes, coordinates are desired for.

    Returns:
        `int16 [y_size * x_size * n_z_planes, 3]`
            yxz coordinates of all pixels on `z_planes`.
    """
    if isinstance(z_planes, int):
        z_planes = jnp.array([z_planes])
    elif isinstance(z_planes, list):
        z_planes = jnp.array(z_planes)
    return jnp.array(jnp.meshgrid(jnp.arange(y_size), jnp.arange(x_size), z_planes), dtype=jnp.int16).T.reshape(-1, 3)
