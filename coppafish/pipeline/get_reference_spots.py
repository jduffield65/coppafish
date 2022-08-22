from ..spot_colors import get_spot_colors
from ..call_spots import get_non_duplicate
from ..find_spots import spot_yxz
import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp
from ..setup.notebook import NotebookPage


def get_reference_spots(nbp_file: NotebookPage, nbp_basic: NotebookPage, spot_details: np.ndarray,
                        tile_origin: np.ndarray, transform: np.ndarray) -> NotebookPage:
    """
    This takes each spot found on the reference round/channel and computes the corresponding intensity
    in each of the imaging rounds/channels.

    See `'ref_spots'` section of `notebook_comments.json` file
    for description of the variables in the page.
    The following variables:

    * `gene_no`
    * `score`
    * `score_diff`
    * `intensity`

    will be set to `None` so the page can be added to a *Notebook*. `call_reference_spots` should then be run
    to give their actual values. This is so if there is an error in `call_reference_spots`,
    `get_reference_spots` won't have to be re-run.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        spot_details: `int [n_spots x 7]`.
            `spot_details[s]` is `[tile, round, channel, isolated, y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.spot_details`.
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch` notebook page i.e. `nb.stitch.tile_origin`.
        transform: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.
            `transform[t, r, c]` is the affine transform to get from tile `t`, `ref_round`, `ref_channel` to
            tile `t`, round `r`, channel `c`.
            This is saved in the register notebook page i.e. `nb.register.transform`.

    Returns:
        `NotebookPage[ref_spots]` - Page containing intensity of each reference spot on each imaging round/channel.
    """
    nbp = NotebookPage("ref_spots")
    r = nbp_basic.ref_round
    c = nbp_basic.ref_channel

    # all means all spots found on the reference round / channel
    all_local_yxz = np.zeros((0, 3), dtype=np.int16)
    all_isolated = np.zeros(0, dtype=bool)
    all_local_tile = np.zeros(0, dtype=np.int16)
    for t in nbp_basic.use_tiles:
        t_local_yxz, t_isolated = spot_yxz(spot_details, t, r, c, return_isolated=True)
        if np.shape(t_local_yxz)[0] > 0:
            all_local_yxz = np.append(all_local_yxz, t_local_yxz, axis=0)
            all_isolated = np.append(all_isolated, t_isolated.astype(bool), axis=0)
            all_local_tile = np.append(all_local_tile, np.ones_like(t_isolated, dtype=np.int16) * t)

    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    not_duplicate = get_non_duplicate(tile_origin, nbp_basic.use_tiles, nbp_basic.tile_centre, all_local_yxz,
                                      all_local_tile)

    # nd means all spots that are not duplicate
    nd_local_yxz = all_local_yxz[not_duplicate]
    nd_isolated = all_isolated[not_duplicate]
    nd_local_tile = all_local_tile[not_duplicate]
    invalid_value = -nbp_basic.tile_pixel_value_shift
    # Only save used rounds/channels initially
    n_use_rounds = len(nbp_basic.use_rounds)
    n_use_channels = len(nbp_basic.use_channels)
    use_tiles = np.array(nbp_basic.use_tiles.copy())
    n_use_tiles = len(use_tiles)
    nd_spot_colors_use = np.zeros((nd_local_tile.shape[0], n_use_rounds, n_use_channels), dtype=np.int32)
    transform = jnp.asarray(transform)
    print('Reading in spot_colors for ref_round spots')
    for t in nbp_basic.use_tiles:
        in_tile = nd_local_tile == t
        if np.sum(in_tile) > 0:
            print(f"Tile {np.where(use_tiles==t)[0][0]+1}/{n_use_tiles}")
            # this line will return invalid_value for spots outside tile bounds on particular r/c.
            nd_spot_colors_use[in_tile] = get_spot_colors(jnp.asarray(nd_local_yxz[in_tile]), t,
                                                          transform, nbp_file, nbp_basic)
    # good means all spots that were in bounds of tile on every imaging round and channel that was used.
    # nd_spot_colors_use = np.moveaxis(nd_spot_colors_use, 0, -1)
    # use_rc_index = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    # nd_spot_colors_use = np.moveaxis(nd_spot_colors_use[use_rc_index], -1, 0)
    good = ~np.any(nd_spot_colors_use == invalid_value, axis=(1, 2))

    good_local_yxz = nd_local_yxz[good]
    good_isolated = nd_isolated[good]
    good_local_tile = nd_local_tile[good]
    # add in un-used rounds with invalid_value
    n_good = np.sum(good)
    good_spot_colors = np.full((n_good, nbp_basic.n_rounds,
                                nbp_basic.n_channels), invalid_value, dtype=np.int32)
    good_spot_colors[np.ix_(np.arange(n_good), nbp_basic.use_rounds, nbp_basic.use_channels)] = nd_spot_colors_use[good]

    # save spot info to notebook
    nbp.local_yxz = good_local_yxz
    nbp.isolated = good_isolated
    nbp.tile = good_local_tile
    nbp.colors = good_spot_colors

    # Set variables added in call_reference_spots to None so can save to Notebook.
    # I.e. if call_reference_spots hit error, but we did not do this,
    # we would have to run get_reference_spots again.
    nbp.gene_no = None
    nbp.score = None
    nbp.score_diff = None
    nbp.intensity = None
    return nbp
