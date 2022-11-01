import numpy as np
from skimage.exposure import equalize_hist
import tifffile as tf
import yaml
from coppafish import Notebook, NotebookPage


def fuse_image(nbp_basic: NotebookPage, tile_dir: str, name: str, channel: int):

    tile_pos_original = nbp_basic.tilepos_yx
    tile_permutation = np.zeros(tile_pos_original.shape[0], dtype=int)

    with open(tile_dir + '/stitch.yml', 'r') as file:
        file_matrix = yaml.safe_load(file)['filematrix']

    tile_origin = np.zeros((len(file_matrix), 3), dtype=int)
    tile_pos = np.zeros((len(file_matrix), 2), dtype=int)
    size = [file_matrix[0]['nfrms'], file_matrix[0]['ysize'], file_matrix[0]['xsize']]

    for i in range(len(file_matrix)):
        tile_origin[i] = [file_matrix[i]['Zs'], file_matrix[i]['Ys'], file_matrix[i]['Xs']]
        tile_pos[i] = [int(file_matrix[i]['Y']), int(file_matrix[i]['X'])]
        y_tile = [j for j in range(len(file_matrix)) if tile_pos_original[j, 0] == tile_pos[i, 0]]
        x_tile = [j for j in range(len(file_matrix)) if tile_pos_original[j, 1] == tile_pos[i, 1]]
        index_set = list(set(y_tile).intersection(set(x_tile)))
        index = index_set[0]
        tile_permutation[i] = index

    canvas_shape = np.max(tile_origin, axis=0) + size
    large_image = np.zeros(canvas_shape)

    for i in range(len(file_matrix)):
        small_image = np.load(tile_dir + name + str(tile_permutation[i]) + 'c' + str(channel) + '.npy')
        large_image[tile_origin[i, 0]:tile_origin[i, 0] + size[0], tile_origin[i, 1]:tile_origin[i, 1] + size[1],
                    tile_origin[i, 2]:tile_origin[i, 2] + size[2]] = small_image

    np.save(tile_dir + name + '.npy', large_image)

nb = Notebook('//ZARU\Subjects\ISS\Izzie/221020_IF_primary_then_seq\output/IF_npy_tiles/6E10/notebook.npz')
fuse_image(nb.basic_info,'//ZARU\Subjects\ISS\Izzie/221020_IF_primary_then_seq\output/IF_npy_tiles/6E10/', 'IF_round_t',
           19)


