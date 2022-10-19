from coppafish import Notebook
from coppafish.find_spots import spot_yxz
from coppafish.register.base import get_single_affine_transform
import numpy as np

# Load in notebooks from 1 tile and 5 tile dataset
nb1 = Notebook('C:/Users/Reilly/Desktop/Sample Notebooks/Izzie/notebook_single_tile.npz')
nb5 = Notebook('C:/Users/Reilly/Desktop/Sample Notebooks/Izzie/notebook_5_tile.npz')

delta_shift = np.zeros((nb1.basic_info.n_rounds, nb1.basic_info.n_channels))

# Load in point clouds from each round channel combo for tile 55 and compute their shifts
for r in nb1.basic_info.use_rounds:
    for c in nb1.basic_info.use_channels:
        pc1 = spot_yxz(nb1.find_spots.spot_details, 55, r, c, nb1.find_spots.spot_no)
        pc5 = spot_yxz(nb5.find_spots.spot_details, 55, r, c, nb5.find_spots.spot_no)
        pca1 = spot_yxz(nb1.find_spots.spot_details, 55, nb1.basic_info.ref_round, nb1.basic_info.ref_round,
                        nb1.find_spots.spot_no)
        pca5 = spot_yxz(nb5.find_spots.spot_details, 55, nb5.basic_info.ref_round, nb5.basic_info.ref_round,
                        nb5.find_spots.spot_no)
        transform1, n_matches1, error1, is_converged1 = get_single_affine_transform(pca1, pc1, 0.8 / 0.26,
                                                                                    0.8 / 0.26, np.vstack(
                (np.eye(3), nb1.register_initial.shift[55, 5])),
                                                                                    5, [2304 // 2, 2304 // 2, 53 // 2],
                                                                                    100)
        transform5, n_matches5, error5, is_converged5 = get_single_affine_transform(pca5, pc5, 0.8 / 0.26,
                                                                                    0.8 / 0.26, np.vstack(
                (np.eye(3), nb5.register_initial.shift[55, 5])),
                                                                                    5, [2304 // 2, 2304 // 2, 53 // 2],
                                                                                    100)
        delta_shift[r, c] = np.sum(np.abs(transform1[3, :]-transform5[3, :]))

print('Hello World')