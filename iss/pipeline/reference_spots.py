from .. import setup
from ..spot_colors import get_spot_colors
from ..find_spots import spot_yxz
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
from ..setup.notebook import NotebookPage


def reference_spots(nbp_file: NotebookPage, nbp_basic: NotebookPage, spot_details: np.ndarray,
                    tile_origin: np.ndarray, transform: np.ndarray) -> NotebookPage:
    """
    This takes each spot found on the reference round/channel and computes the corresponding intensity
    in each of the imaging rounds/channels.

    See `'ref_spots'` section of `notebook_comments.json` file
    for description of the variables in the page.

    Args:
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        spot_details: `int [n_spots x 7]`.
            `spot_details[s]` is `[tile, round, channel, isolated, y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.spot_details`.
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch_debug` notebook page i.e. `nb.stitch_debug.tile_origin`.
        transform: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.
            `transform[t, r, c]` is the affine transform to get from tile `t`, `ref_round`, `ref_channel` to
            tile `t`, round `r`, channel `c`.
            This is saved in the register notebook page i.e. `nb.register.transform`.

    Returns:
        `NotebookPage[ref_spots]` - Page containing intensity of each reference spot on each imaging round/channel.
    """
    nbp = setup.NotebookPage("ref_spots")
    r = nbp_basic.ref_round
    c = nbp_basic.ref_channel

    # all means all spots found on the reference round / channel
    all_local_yxz = np.zeros((0, 3), dtype=int)
    all_global_yxz = np.zeros((0, 3), dtype=float)
    all_isolated = np.zeros(0, dtype=bool)
    all_local_tile = np.zeros(0, dtype=int)
    for t in range(nbp_basic.n_tiles):
        t_local_yxz, t_isolated = spot_yxz(spot_details, t, r, c, return_isolated=True)
        if np.shape(t_local_yxz)[0] > 0:
            all_local_yxz = np.append(all_local_yxz, t_local_yxz, axis=0)
            all_global_yxz = np.append(all_global_yxz, t_local_yxz + tile_origin[t], axis=0)
            all_isolated = np.append(all_isolated, t_isolated.astype(bool), axis=0)
            all_local_tile = np.append(all_local_tile, np.ones_like(t_isolated, dtype=int) * t)

    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    # Do this in 2d as overlap is only 2d
    tile_centres = tile_origin + nbp_basic.tile_centre
    tree_tiles = NearestNeighbors(n_neighbors=1).fit(tile_centres[:, :2])
    _, all_nearest_tile = tree_tiles.kneighbors(all_global_yxz[:, :2])
    not_duplicate = all_nearest_tile.flatten() == all_local_tile

    # nd means all spots that are not duplicate
    nd_local_yxz = all_local_yxz[not_duplicate]
    nd_global_yxz = all_global_yxz[not_duplicate]
    nd_isolated = all_isolated[not_duplicate]
    nd_local_tile = all_local_tile[not_duplicate]
    nd_spot_colors = np.zeros((nd_local_tile.shape[0], nbp_basic.n_rounds, nbp_basic.n_channels))
    for t in tqdm(range(nbp_basic.n_tiles), desc='Current Tile'):
        in_tile = nd_local_tile == t
        if sum(in_tile) > 0:
            nd_spot_colors[in_tile] = get_spot_colors(nd_local_yxz[in_tile], t, transform, nbp_file, nbp_basic)

    # good means all spots that were in bounds of tile on every imaging round and channel that was used.
    nd_spot_colors_use = np.moveaxis(nd_spot_colors, 0, -1)
    use_rc_index = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    nd_spot_colors_use = np.moveaxis(nd_spot_colors_use[use_rc_index], -1, 0)
    good = ~np.any(np.isnan(nd_spot_colors_use), axis=(1, 2))

    good_global_yxz = nd_global_yxz[good]
    good_isolated = nd_isolated[good]
    good_local_tile = nd_local_tile[good]
    good_spot_colors = nd_spot_colors[good]

    # save spot info to notebook
    nbp.global_yxz = good_global_yxz
    nbp.isolated = good_isolated
    nbp.tile = good_local_tile
    nbp.colors = good_spot_colors
    return nbp
