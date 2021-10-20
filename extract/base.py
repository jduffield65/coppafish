import numpy as np


def get_nd2_tile_ind(tile_ind_tiff, tile_pos_yx):
    """
    :param tile_ind_tiff: integer
        index of tiff file
    :param tile_pos_yx: dictionary
        ['nd2']: numpy array[nTiles x 2] [i,:] contains YX position of tile with nd2 index i.
        ['tiff']: numpy array[nTiles x 2] [i,:] contains YX position of tile with tiff index i.
    :return: integer, corresponding index in nd2 file.
    """
    return np.where(np.sum(tile_pos_yx['nd2'] == tile_pos_yx['tiff'][tile_ind_tiff], 1) == 2)[0][0]



# def get_filter(r1, r2):
#
# def fstack():
#
# def get_extract_scale():
#
# def filter():
#
# def filter_dapi():
#
# def get_hist_counts():
#
# def get_auto_thresh():

