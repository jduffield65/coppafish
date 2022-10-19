from coppafish import Notebook
from coppafish.plot.raw import get_raw_images
from skimage.registration import phase_cross_correlation
import numpy as np

nb = Notebook('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Sep Round/Full/notebook_NSC_3_sc18.npz')
nb_og = Notebook('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Sep Round/Full/Original/notebook_NSC_3.npz')
ref_round = nb.basic_info.ref_round
ref_channel = nb.basic_info.ref_channel

transform = nb.register.transform
shift = transform[:, :, nb_og.basic_info.use_channels, 3, :]

transform_og = nb_og.register.transform
shift_og = transform_og[:, :, nb_og.basic_info.use_channels, 3, :]

del transform, transform_og

diffs = np.sum(np.abs(shift_og-shift), axis=3)

large_diff_indices = np.argwhere(diffs > 0.9*np.max(diffs))
pcc_shifts = np.zeros((large_diff_indices.shape[0], 3))

for i in range(large_diff_indices.shape[0]):
    anchor_image = get_raw_images(nb, [large_diff_indices[i, 0]], [ref_round], [ref_channel], nb.basic_info.use_z)
    offset_image = get_raw_images(nb, [large_diff_indices[i, 0]], [large_diff_indices[i, 1]],
                                  [large_diff_indices[i, 2]], nb.basic_info.use_z)

    anchor_image = anchor_image[anchor_image.shape[0]//2 - 10: anchor_image.shape[0]//2 + 10,
                    anchor_image.shape[1]//2 - 500: anchor_image.shape[1]//2 + 500,
                    anchor_image.shape[2]//2 - 500: anchor_image.shape[2]//2 + 500]

    offset_image = offset_image[offset_image.shape[0] // 2 - 10: offset_image.shape[0] // 2 + 10,
                   offset_image.shape[1] // 2 - 500: offset_image.shape[1] // 2 + 500,
                   offset_image.shape[2] // 2 - 500: offset_image.shape[2] // 2 + 500]

    phase_cross_correlation(offset_image, anchor_image)[0]

np.savez('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Sep Round/Full/shift_info.npz', large_diff_indices,
         pcc_shifts, shift, shift_og)
