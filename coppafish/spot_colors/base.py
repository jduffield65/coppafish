from typing import Optional, List, Union, Tuple
import numpy as np
from tqdm import tqdm
from .. import utils
from ..setup import NotebookPage


def apply_transform(yxz: np.ndarray, transform: np.ndarray,
                    tile_sz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        tile_sz: ```int16 [3]```.
            YXZ dimensions of tile

    Returns:
        ```int [n_spots x 3]```.
            ```yxz_transform``` such that
            ```yxz_transform[i, [1,2]]``` are the transformed non-centered yx coordinates in ```yx_pixels```
            for spot ```i```.
            ```yxz_transform[i, 2]``` is the transformed non-centered z coordinate in ```z_pixels``` for spot ```i```.
        - ```in_range``` - ```bool [n_spots]```.
            Whether spot s was in the bounds of the tile when transformed to round `r`, channel `c`.
    """
    yxz_pad = np.pad(yxz, [(0, 0), (0, 1)], constant_values=1)
    yxz_transform = yxz_pad @ transform
    yxz_transform = np.round(yxz_transform).astype(np.int16)
    in_range = np.logical_and((yxz_transform >= np.array([0, 0, 0])).all(axis=1),
                              (yxz_transform < tile_sz).all(axis=1))  # set color to nan if out range
    return yxz_transform, in_range


def get_spot_colors(yxz_base: np.ndarray, t: int, transforms: np.ndarray, nbp_file: NotebookPage,
                    nbp_basic: NotebookPage, use_rounds: Optional[List[int]] = None,
                    use_channels: Optional[List[int]] = None, return_in_bounds: bool = False,
                    bg_scale_offset: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
        Hence, I use this as integer nan. It will be `invalid_value` if the registered coordinate of
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
        bg_scale_offset: 'float [n_tiles x n_rounds x n_channels_use x 2]' normalisation factor for each
            of the tiles/rounds and channels. bg_round[t, c] * bg_scale_offset[t, r, c, 0] + bg_scale_offset[t, r, c, 1]
            will equalise the background brightness profile to the same as that of tile t, round r, channel c. If None,
            no normalisation will be performed.


    Returns:
        - `spot_colors` - `int32 [n_spots x n_rounds_use x n_channels_use]` or
            `int32 [n_spots_in_bounds x n_rounds_use x n_channels_use]`.
            `spot_colors[s, r, c]` is the spot color for spot `s` in round `use_rounds[r]`, channel `use_channels[c]`.
        - `yxz_base` - `int16 [n_spots_in_bounds x 3]`.
            If `return_in_bounds`, the `yxz_base` corresponding to spots in bounds for all `use_rounds` / `use_channels`
            will be returned. It is likely that `n_spots_in_bounds` won't be the same as `n_spots`.
        - `bg_colours` - `int32 [n_spots_in_bounds x n_rounds x n_channels_use]`. Background colour for each spot
            in each round and channel. Only returned if `bg_scale_offset` is not None.
    """
    if bg_scale_offset is not None:
        assert nbp_basic.use_preseq, "Can't subtract background if preseq round doesn't exist!"
        use_bg = True
    else:
        use_bg = False
    if use_rounds is None:
        use_rounds = nbp_basic.use_rounds
    if use_channels is None:
        use_channels = nbp_basic.use_channels

    n_spots = yxz_base.shape[0]
    no_verbose = n_spots < 10000
    # note using nan means can't use integer even though data is integer
    n_use_rounds = len(use_rounds)
    n_use_channels = len(use_channels)
    # spots outside tile bounds on particular r/c will initially be set to 0.
    spot_colors = np.zeros((n_spots, n_use_rounds, n_use_channels), dtype=np.int32)
    if use_bg:
        bg_colours = np.zeros((n_spots, n_use_channels), dtype=np.int32)
    if not nbp_basic.is_3d:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        tile_sz = np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, 1], dtype=np.int16)
    else:
        tile_sz = np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz], dtype=np.int16)

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
                            utils.npy.load_tile(nbp_file, nbp_basic, t, nbp_basic.pre_seq_round, use_channels[c],
                                                yxz_transform, apply_shift=False)
                pbar.update(1)
        # subtract tile pixel shift value so that bg_colours are in range -15_000 to 50_000 (approx)
        valid = bg_colours > 0
        bg_colours[valid] = bg_colours[valid] - nbp_basic.tile_pixel_value_shift
        # repeat bg_colours so it is the same shape as spot_colors
        bg_colours = np.repeat(bg_colours[:, None, :], n_use_rounds, axis=1)
        bg_colours[valid] = bg_colours[valid] * bg_scale_offset[t, use_rounds, use_channels, 0][None] + \
                     bg_scale_offset[t, use_rounds, use_channels, 1][None]
    # Remove shift so now spots outside bounds have color equal to - nbp_basic.tile_pixel_shift_value.
    # It is impossible for any actual spot color to be this due to clipping at the extract stage.
    spot_colors = spot_colors - nbp_basic.tile_pixel_value_shift
    if use_bg:
        spot_colors = spot_colors - bg_colours
        return spot_colors, bg_colours
    invalid_value = -nbp_basic.tile_pixel_value_shift
    if return_in_bounds:
        good = ~np.any(spot_colors == invalid_value, axis=(1, 2))
        return spot_colors[good], yxz_base[good]
    else:
        return spot_colors


def all_pixel_yxz(y_size: int, x_size: int, z_planes: Union[List, int, np.ndarray]) -> np.ndarray:
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
        z_planes = np.array([z_planes])
    elif isinstance(z_planes, list):
        z_planes = np.array(z_planes)
    return np.array(np.meshgrid(np.arange(y_size), np.arange(x_size), z_planes), dtype=np.int16).T.reshape(-1, 3)


def normalise_rc(pixel_colours: np.ndarray, spot_colours: np.ndarray, cutoff_intensity_percentile: float = 75,
                 num_spots: int = 100) -> np.ndarray:
    """
    Takes in the pixel colours for a single z-plane of a tile, for all rounds and channels. Then performs 2
    normalisations. The first of these is normalising by round and the second is normalising by channel.
    Args:
        pixel_colours: 'int [n_pixels x n_rounds x n_channels_use]' pixel colours for a single z-plane of a tile.
        # NOTE: It is assumed these images are all aligned and have the same dimensions.
        spot_colours: 'int [n_spots x n_rounds x n_channels_use]' spot colours for whole dataset.
        cutoff_intensity_percentile: 'float' upper percentile of pixel intensities to use for regression in
        round normalisation.
        num_spots: 'int' number of spots to use for each round/channel in channel normalisation.
    Returns:
        norm_factor: [n_rounds x n_channels_use]` normalisation factor for each of the rounds/channels.
    """
    # 1. Normalise by round. Do this by performing a linear regression on low brightness pixels that will not be spots.
    # First, for each channel, find a good round to use for normalisation. We will take this round to be the one with
    # the median of the means of all rounds.
    n_spots, n_rounds, n_channels = pixel_colours.shape
    round_slopes = np.zeros((n_rounds, n_channels))
    for c in range(n_channels):
        brightness = np.mean(np.abs(pixel_colours)[:, :, c], axis=0)
        median_brightness = np.median(brightness)
        # Find the round with the median brightness
        median_round = np.where(brightness == median_brightness)[0][0]
        # Now perform the regression of each round against the median round
        cutoff_intensity = np.percentile(pixel_colours[:, median_round, c], cutoff_intensity_percentile)
        image_mask = pixel_colours[:, median_round, c] < cutoff_intensity
        base_image = pixel_colours[:, median_round, c][image_mask]
        for r in range(n_rounds):
            target_image = pixel_colours[:, r, c][image_mask]
            round_slopes[r, c] = np.linalg.lstsq(base_image[:, None], target_image, rcond=None)[0]
            pixel_colours[:, r, c] = pixel_colours[:, r, c] / round_slopes[r, c]
    # 2. Normalise by channel. For this we want to use spots. As spots are not aligned between rounds, we will
    # concatenate all rounds of a given channel and match the intensities across channels.
    bright_spots = np.zeros((n_rounds, n_channels, num_spots))
    max_channel = np.argmax(spot_colours, axis=2)
    # channel_strength is the median of the mean spot intensities for each channel
    rc_spot_strength = np.zeros((n_rounds, n_channels))
    for r in range(n_rounds):
        for c in range(n_channels):
            possible_spots = np.where(max_channel[:, r] == c)[0]
            possible_colours = spot_colours[possible_spots, r, c]
            # take the brightest spots
            bright_spots[r][c] = possible_colours[np.argsort(possible_colours)[-num_spots:]]
            rc_spot_strength[r, c] = np.median(bright_spots[r][c])

    norm_factor = rc_spot_strength * round_slopes
    return norm_factor


def remove_background(spot_colours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes background from spot colours
    Args:
        spot_colours: 'float [n_spots x n_rounds x n_channels_use]' spot colours to remove background from.
    Returns:
        'spot_colours: [n_spots x n_rounds x n_channels_use]' spot colours with background removed.
        background_noise: [n_spots x n_channels_use]' background noise for each spot and channel.
    """
    background_noise = np.zeros((spot_colours.shape[0], spot_colours.shape[2]))
    # Loop through all channels and remove the background from each channel.
    for c in tqdm(range(spot_colours.shape[2])):
        background_code = np.zeros(spot_colours[0].shape)
        background_code[:, c] = 1
        # now loop through all spots and remove the component of the background from the spot colour
        for s in range(spot_colours.shape[0]):
            background_noise[s, c] = np.percentile(spot_colours[s, :, c], 25)
            spot_colours[s] = spot_colours[s] - background_noise[s, c] * background_code

    return spot_colours, background_noise


def neighbour_normalisation(im1: np.ndarray, im2: np.ndarray):
    """
    im1 and im2 will come from adjacent tiles and will be their registered overlap.
    We want to normalise the intersection so that their intensities match.
    Args:
        im1: 'uint16 n_pixels' image from tile1 overlap
        im2: 'uint16 n_pixels' image from tile2 overlap

    Returns:
        alpha: 'float32' normalisation factor for im1
    """
