import numpy as np
from .. import pcr, utils
from ..setup.notebook import NotebookPage
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import tifffile
import jax.numpy as jnp
import jax
from functools import partial


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
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
    tile_sz = [nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz]
    if not nbp_basic.is_3d:
        tile_sz[2] = 1

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10000
    # note using nan means can't use integer even though data is integer
    spot_colors = np.ones((n_spots, nbp_basic.n_rounds, nbp_basic.n_channels), dtype=int) * nan_value
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
                    try:
                        image = utils.tiff.load_tile(nbp_file, nbp_basic, t, r, c, load_y, load_x, load_z)
                    except IndexError:
                        # Sometimes get index error when it will only load in all z-planes.
                        image = utils.tiff.load_tile(nbp_file, nbp_basic, t, r, c, load_y, load_x)
                        image = image[:, :, load_z]

                    yxz_transform = yxz_transform - yxz_min  # shift yxz so load in correct colors from cropped image.
                    if image.ndim == 3:
                        spot_colors[in_range, r, c] = image[yxz_transform[:, 0], yxz_transform[:, 1],
                                                            yxz_transform[:, 2]]
                    else:
                        spot_colors[in_range, r, c] = image[yxz_transform[:, 0], yxz_transform[:, 1]]
                pbar.update(1)
    return spot_colors


def get_all_pixel_colors(t: int, transforms: jnp.ndarray, nbp_file: NotebookPage,
                         nbp_basic: NotebookPage,
                         z_planes: Union[int, np.ndarray, List] = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        z_planes = jnp.array([z_planes])
    pixel_yxz_all = jnp.array(jnp.meshgrid(np.arange(nbp_basic.tile_sz),
                                           jnp.arange(nbp_basic.tile_sz), jnp.arange(1))).T.reshape(-1, 3)
    pixel_colors = jnp.zeros((0, len(nbp_basic.use_rounds), len(nbp_basic.use_channels)), dtype=int)
    pixel_yxz = jnp.zeros((0, 3), dtype=int)
    for z in z_planes:
        pixel_yxz_all = pixel_yxz_all * jnp.array([1, 1, 0]) + jnp.array([0, 0, z])
        # IMPORTANT!! Rounding error as jnp is float32 not float64 causes some differences between get_spot_colors_jax
        # and get_spot_colors. I think all because transforms are 1 pixel apart.
        # TODO: unit test between get_spot_colors and get_spot_colors_jax.
        pixel_colors_all = get_spot_colors_jax(jnp.array(pixel_yxz_all), t, transforms, nbp_file, nbp_basic)
        # pixel_colors_all = get_spot_colors(pixel_yxz_all, t, transforms, nbp_file, nbp_basic)
        # only keep used rounds/channels to save memory.
        # pixel_colors_all = pixel_colors_all[jnp.ix_(jnp.arange(pixel_colors_all.shape[0]), nbp_basic.use_rounds,
        #                                            nbp_basic.use_channels)]
        # only keep spots in all rounds/channels meaning no nan values
        keep = ~jnp.any(pixel_colors_all == nan_value, axis=(1, 2))
        if keep.any():
            pixel_colors = jnp.append(pixel_colors, pixel_colors_all[keep].astype(int), axis=0)
            pixel_yxz = jnp.append(pixel_yxz, pixel_yxz_all[keep], axis=0)
    # # Don't include nan check as jnp to np conversion takes time.
    # if pixel_colors.shape[0] > 0:
    #     utils.errors.check_color_nan(pixel_colors, nbp_basic)
    return pixel_colors, pixel_yxz


def read_spot_color_single(image: jnp.ndarray, z0_plane: int, yxz_base: jnp.ndarray, transform: jnp.ndarray,
                           tile_centre: jnp.ndarray, z_scale: float, nan_value: float, tile_sz: jnp.ndarray) -> int:
    """
    Reads in color of a single round/channel at desired coordinate.

    Args:
        image: `int [image_szY x image_szX x image_szZ]`.
            image_szZ should be 1 if 2D.
            Image to find colors on which is the round/channel which matches the `transform` given.
        z0_plane: The first z-plane of image, `image[:, :, 0]`, is plane `z0_plane` in the tiff_file and in `yxz_base`.
            I.e. spots with `yxz_base[:, 2] == z0_plane` will be looked for in `image[:, :, 0]`.
        yxz_base: `int [3]`.
            Local yxz coordinates of spot found in the reference round/reference channel of tile `t`
            yx coordinates are in units of `yx_pixels`. z coordinates are in units of `z_pixels`.
        transform: ```float [4 x 3]```.
            Affine transform to apply to ```yxz```, once centered and z units changed to ```yx_pixels```.
            ```transform[3, 2]``` is approximately the z shift in units of ```yx_pixels```.
            E.g. this is one of the transforms stored in ```nb.register.transform```.
        tile_centre: ```float [3]```.
            ```tile_centre[:2]``` are yx coordinates in ```yx_pixels``` of the centre of the tile that spots in
            ```yxz``` were found on.
            ```tile_centre[2]``` is the z coordinate in ```z_pixels``` of the centre of the tile.
            E.g. for tile of ```yxz``` dimensions ```[2048, 2048, 51]```, ```tile_centre = [1023.5, 1023.5, 25]```
            Each entry in ```tile_centre``` must be an integer multiple of ```0.5```.
        z_scale: Scale factor to multiply z coordinates to put them in units of yx pixels.
            I.e. ```z_scale = pixel_size_z / pixel_size_yx``` where both are measured in microns.
            typically, ```z_scale > 1``` because ```z_pixels``` are larger than the ```yx_pixels```.
        nan_value: Value to set color if transformed yxz coordinate is out of range.
            Typically set to `-nbp.basic_info.tile_pixel_value_shift - 1`.
        tile_sz: `int [3]`
            YXZ shape of tile_sz to determine whether transformed yxz coordinates are out of bounds.

    Returns:
        Pixel value in `image` at `yxz_transform` which is `yxz_base` transformed according to `transform`.
    """
    # subtract z0_plane so correct coordinates for image which is cropped n z.
    yxz_transform = pcr.apply_transform_jax_single(yxz_base, transform, tile_centre, z_scale
                                                   ) - jnp.array([0, 0, z0_plane])
    in_range = jnp.logical_and(jnp.min(yxz_transform >= jnp.array([0, 0, 0])),
                               jnp.min(yxz_transform < tile_sz))  # set color to nan if out range
    # Below is one line way to give nan if out of range else value read from image.
    # Out of range is not flagged as an error in jax so need to be careful.
    return image[yxz_transform[0], yxz_transform[1], yxz_transform[2]] * in_range + jnp.invert(in_range) * nan_value


@partial(jax.jit, static_argnums=(5, 6))
def read_spot_color(image: jnp.ndarray, z0_plane: int, yxz_base: jnp.ndarray, transform: jnp.ndarray,
                    tile_centre: jnp.ndarray, z_scale: float, nan_value: float, tile_sz: jnp.ndarray) -> jnp.ndarray:
    """
    Reads in colors of a single round/channel at desired coordinates.

    Args:
        image: `int [image_szY x image_szX x image_szZ]`.
            image_szZ should be 1 if 2D.
            Image to find colors on which is the round/channel which matches the `transform` given.
        z0_plane: The first z-plane of image, `image[:, :, 0]`, is plane `z0_plane` in the tiff_file and in `yxz_base`.
            I.e. spots with `yxz_base[:, 2] == z0_plane` will be looked for in `image[:, :, 0]`.
        yxz_base: `int [n_spots x 3]`.
            Local yxz coordinates of spot found in the reference round/reference channel of tile `t`
            yx coordinates are in units of `yx_pixels`. z coordinates are in units of `z_pixels`.
        transform: ```float [4 x 3]```.
            Affine transform to apply to ```yxz```, once centered and z units changed to ```yx_pixels```.
            ```transform[3, 2]``` is approximately the z shift in units of ```yx_pixels```.
            E.g. this is one of the transforms stored in ```nb.register.transform```.
        tile_centre: ```float [3]```.
            ```tile_centre[:2]``` are yx coordinates in ```yx_pixels``` of the centre of the tile that spots in
            ```yxz``` were found on.
            ```tile_centre[2]``` is the z coordinate in ```z_pixels``` of the centre of the tile.
            E.g. for tile of ```yxz``` dimensions ```[2048, 2048, 51]```, ```tile_centre = [1023.5, 1023.5, 25]```
            Each entry in ```tile_centre``` must be an integer multiple of ```0.5```.
        z_scale: Scale factor to multiply z coordinates to put them in units of yx pixels.
            I.e. ```z_scale = pixel_size_z / pixel_size_yx``` where both are measured in microns.
            typically, ```z_scale > 1``` because ```z_pixels``` are larger than the ```yx_pixels```.
        nan_value: Value to set color if transformed yxz coordinate is out of range.
            Typically set to `-nbp.basic_info.tile_pixel_value_shift - 1`.
        tile_sz: `int [3]`
            YXZ shape of tile_sz to determine whether transformed yxz coordinates are out of bounds.

    Returns:
        `int [n_spots]`
        Pixel value in `image` at `yxz_transform` which is `yxz_base` transformed according to `transform`.
    """
    return jax.vmap(read_spot_color_single, in_axes=(None, None, 0, None, None, None, None, None),
                    out_axes=0)(image, z0_plane, yxz_base, transform, tile_centre, z_scale, nan_value, tile_sz)


def get_spot_colors_jax(yxz_base: jnp.ndarray, t: int, transforms: jnp.ndarray, nbp_file: NotebookPage,
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
        `int [n_spots x n_rounds_use x n_channels_use]`.

        `spot_colors[s, r, c]` is the spot color for spot `s` in round `use_rounds[r]`, channel `use_channels[c]`.

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
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10000
    # note using nan means can't use integer even though data is integer
    n_use_rounds = len(use_rounds)
    n_use_channels = len(use_channels)
    spot_colors = np.ones((n_spots, n_use_rounds, n_use_channels), dtype=int) * nan_value
    if nbp_basic.is_3d:
        z_base_min_max = jnp.unique(yxz_base[:, 2])  # all z-planes to consider.
        # add 0.25 as a security incase rotation or scaling pushes spot to another z-plane.
        z_base_min_max = np.array([z_base_min_max[0]-0.25, z_base_min_max[-1]+0.25])
    else:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        z_transform = np.array([0])
    # Consider consecutive planes so can just subtract min_plane.
    tile_centre = jnp.array(nbp_basic.tile_centre)

    with tqdm(total=n_use_rounds * n_use_channels, disable=no_verbose) as pbar:
        for r in range(n_use_rounds):
            if not nbp_basic.is_3d:
                image_all_channels = jnp.moveaxis(
                    jnp.array(tifffile.imread(nbp_file.tile[t][use_rounds[r]], key=use_channels).astype(int)), 0, -1) \
                                     - nbp_basic.tile_pixel_value_shift
            for c in range(n_use_channels):
                transform_rc = transforms[t, use_rounds[r], use_channels[c]]
                pbar.set_description(f"Reading {n_spots} spot_colors found on tile {t} from tiff files.")
                pbar.set_postfix({'round': use_rounds[r], 'channel': use_channels[c]})
                if transform_rc[0, 0] == 0:
                    raise ValueError(
                        f"Transform for tile {t}, round {use_rounds[r]}, channel {use_channels[c]} is zero:"
                        f"\n{transform_rc}")
                # z_transform will always be [0] in 2D.
                if nbp_basic.is_3d:
                    z_transform_min_max = np.round(z_base_min_max + transform_rc[3, 2] / z_scale).astype(int)
                    z_transform = np.arange(z_transform_min_max[0], z_transform_min_max[1]+1)
                    # Only include possible z-planes in tiff file.
                    z_transform = z_transform[np.logical_and(z_transform >= 0, z_transform < nbp_basic.nz)]
                if len(z_transform) > 0:
                    tile_sz = jnp.array([nbp_basic.tile_sz, nbp_basic.tile_sz,
                                         len(z_transform)])  # size of image passed to read_spot_color
                    if nbp_basic.is_3d:
                        try:
                            image = jnp.array(tifffile.imread(
                                nbp_file.tile[t][use_rounds[r]][use_channels[c]], key=z_transform).astype(int)
                                              ) - nbp_basic.tile_pixel_value_shift

                        except IndexError:
                            # Sometimes get index error when it will only load in all z-planes.
                            image = tifffile.imread(nbp_file.tile[t][use_rounds[r]][use_channels[c]])
                            image = jnp.array(image[z_transform].astype(int)) - nbp_basic.tile_pixel_value_shift
                        if len(z_transform) > 1:
                            image = jnp.moveaxis(image, 0, 2)
                    else:
                        image = image_all_channels[:, :, c]
                    if image.ndim == 2:
                        image = image[:, :, jnp.newaxis]
                    # having spot_colors as np.array makes assignment quicker than at/set notation in jax.
                    spot_colors[:, r, c] = np.asarray(
                        read_spot_color(image, z_transform.min(), yxz_base, transform_rc, tile_centre,
                                        z_scale, nan_value, tile_sz))
                pbar.update(1)
    return jnp.array(spot_colors)
