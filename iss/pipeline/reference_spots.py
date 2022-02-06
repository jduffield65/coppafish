from .. import setup
from ..spot_colors import get_spot_colors
from ..find_spots import spot_yxz
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm


def reference_spots(nbp_file, nbp_basic, spot_details, tile_origin, transform):
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

    # good means all spots that were in bounds of tile on every imaging round and channel
    good = ~np.any(np.isnan(nd_spot_colors), axis=(1, 2))
    good_global_yxz = nd_global_yxz[good]
    good_isolated = nd_isolated[good]
    good_local_tile = nd_local_tile[good]
    good_spot_colors = nd_spot_colors[good].astype(int)  # nans where only non integer values before

    # save spot info to notebook
    nbp.global_yxz = good_global_yxz
    nbp.isolated = good_isolated
    nbp.tile = good_local_tile
    nbp.colors = good_spot_colors
    return nbp
