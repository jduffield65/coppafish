import numpy as np
import os


def get_tilepos(xy_pos, tile_sz):
    """
    tilepos_yx_nd2[i, 0] is y index of tile with fov index i in nd2 file.
    tilepos_yx_nd2[i, 1] is x index of tile with fov index i in nd2 file.
    tilepos_yx_tiff[i, 0] is y index of tile with tile directory (tiff files) index i.
    tilepos_yx_tiff[i, 1] is x index of tile with tile directory (tiff files) index i.

    :param xy_pos: numpy array [nTiles x 2]
        xy position of tiles in pixels.
    :param tile_sz: integer
        xy dimension of tile in pixels.
    :return:
        tilepos_yx_nd2: integer numpy array [nTiles x 2]
        tilepos_yx_tiff: integer numpy array [nTiles x 2]
    """
    tilepos_yx_nd2 = np.zeros_like(xy_pos, dtype=int)
    if np.shape(xy_pos)[0] != 1:
        # say y coordinate changes when successive tiles have pixel separation of more than tile_sz/2
        change_y_coord = np.abs(np.ediff1d(xy_pos[:, 1])) > tile_sz / 2
        if False in change_y_coord:
            # sometimes get faulty first xy_pos
            # know that if there are more than one y coordinates, then the first
            # and second tile must have the same y coordinate.
            change_y_coord[0] = False
        ny = sum(change_y_coord) + 1
        nx = np.shape(xy_pos)[0] / ny
        if round(nx) != nx:
            raise ValueError('nx is not an integer')
        tilepos_yx_nd2[:, 0] = np.flip(np.arange(ny).repeat(nx))
        tilepos_yx_nd2[:, 1] = np.tile(np.concatenate((np.flip(np.arange(nx)), np.arange(nx))),
                                          np.ceil(ny / 2).astype(int))[:np.shape(xy_pos)[0]]
    t_tiff_index = get_tile_file_indices(tilepos_yx_nd2)
    tilepos_yx_tiff = tilepos_yx_nd2[np.argsort(t_tiff_index)]
    return tilepos_yx_nd2, tilepos_yx_tiff


def get_tile_file_indices(tilepos_yx_nd2):
    """
    Tile with nd2 fov index i is at yx position tilepos_yx_nd2[i, :]
    and tiff file has tile index given by tile_file_index[i].
    Tile file indices are different because want tile_file_index[0]
    to refer to tile at yx position [0, 0].

    :param tilepos_yx_nd2: integer numpy array [nTiles x 2]
        tilepos_yx[i, 0] is y index of tile with fov index i in nd2 file.
        tilepos_yx[i, 1] is x index of tile with fov index i in nd2 file.
    :return:
        tile_file_index: integer numpy array [nTiles,]
    """
    ny, nx = tuple(np.max(tilepos_yx_nd2, 0) + 1)
    tile_file_index = np.ravel_multi_index(np.array([tilepos_yx_nd2[:, 0], tilepos_yx_nd2[:, 1]]), (ny, nx))
    return tile_file_index


def get_tile_name(tile_directory, file_base, r, t, c=None):
    """

    :param tile_directory: path to folder where tiles tiff files saved.
    :param file_base: object numpy array or list [n_rounds,].
        file_base[r] is identifier for round r.
    :param r: round
    :param t: tiff tile index
    :param c: channel
    :return: string giving full path of tile.
    """
    if c is None:
        tile_name = os.path.join(tile_directory, '{}_t{}.tif'.format(file_base[r], t))
    else:
        tile_name = os.path.join(tile_directory, '{}_t{}c{}.tif'.format(file_base[r], t, c))
    return tile_name


def get_tile_file_names(tile_directory, file_base, tilepos_yx_nd2, matlab_tile_names, n_channels=0):
    """

    :param tile_directory: path to folder where tiles tiff files saved.
    :param file_base: object numpy array or list [n_rounds,].
        file_base[r] is identifier for round r.
    :param tilepos_yx_nd2: integer numpy array [nTiles x 2]
        tilepos_yx_nd2[i, 0] is y index of tile with fov index i in nd2 file.
        tilepos_yx_nd2[i, 1] is x index of tile with fov index i in nd2 file.
    :param matlab_tile_names: boolean
        if true tile files will have t and c index starting at 1 else will start at 0
    :param n_channels: total number of imaging channels if using 3D, optional.
        0 if using 2D pipeline as all channels saved in same file.
        default: 0.
    :return: tiles_files. object numpy array
        [n_tiles x n_rounds] if 2D
        [n_tiles x n_rounds x n_channels] if 3D
    """
    t_tiff = get_tile_file_indices(tilepos_yx_nd2)
    n_tiles = np.shape(tilepos_yx_nd2)[0]
    n_rounds = len(file_base)
    index_shift = int(matlab_tile_names)
    if n_channels == 0:
        # 2D
        tile_files = np.zeros((n_tiles, n_rounds), dtype=object)
        for r in range(n_rounds):
            for t_nd2 in range(n_tiles):
                tile_files[t_tiff[t_nd2], r] = \
                    get_tile_name(tile_directory, file_base, r, t_tiff[t_nd2] + index_shift)
    else:
        # 3D
        tile_files = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)
        for r in range(n_rounds):
            for t_nd2 in range(n_tiles):
                for c in range(n_channels):
                    tile_files[t_tiff[t_nd2], r, c] = \
                        get_tile_name(tile_directory, file_base, r, t_tiff[t_nd2] + index_shift,
                                      c + index_shift)
    return tile_files
