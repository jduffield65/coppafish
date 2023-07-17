import numpy as np
from .. import utils
from typing import Tuple, List, Union
import warnings

def get_dye_channel_intensity_guess(csv_file_name: str, dyes: Union[List[str], np.ndarray],
                                    cameras: Union[List[int], np.ndarray],
                                    lasers: Union[List[int], np.ndarray]) -> np.ndarray:
    """
    This gets an estimate for the intensity of each dye in each channel (before any channel normalisation)
    which is then used as the starting point for the bleed matrix computation.

    Args:
        csv_file_name: Path to csv file which has 4 columns with headers Dye, Camera, Laser, Intensity:

            - Dye is a column of names of different dyes
            - Camera is a column of integers indicating the wavelength in nm of the camera.
            - Laser is a column of integers indicating the wavelength in nm of the laser.
            - Intensity```[i]``` is the approximate intensity of Dye```[i]``` in a channel with Camera```[i]``` and
                Laser```[i]```.
        dyes: ```str [n_dyes]```.
            Names of dyes used in particular experiment.
        cameras: ```int [n_channels]```.
            Wavelength of camera in nm used in each channel.
        lasers: ```int [n_channels]```.
            Wavelength of laser in nm used in each channel.

    Returns:
        ```float [n_dyes x n_channels]```.
            ```[d, c]``` is estimate of intensity of dye ```d``` in channel ```c```.
    """
    n_dyes = len(dyes)
    cameras = np.array(cameras)
    lasers = np.array(lasers)
    n_channels = cameras.shape[0]
    if not utils.errors.check_shape(cameras, lasers.shape):
        raise utils.errors.ShapeError('cameras', cameras.shape, lasers.shape)

    # load in csv info
    csv_dyes = np.genfromtxt(csv_file_name, delimiter=',', usecols=0, dtype=str, skip_header=1)
    csv_cameras = np.genfromtxt(csv_file_name, delimiter=',', usecols=1, dtype=int, skip_header=1)
    csv_lasers = np.genfromtxt(csv_file_name, delimiter=',', usecols=2, dtype=int, skip_header=1)
    csv_intensities = np.genfromtxt(csv_file_name, delimiter=',', usecols=3, dtype=float, skip_header=1)

    # read in intensity from csv info for desired dyes in each channel
    dye_channel_intensity = np.zeros((n_dyes, n_channels))
    for d in range(n_dyes):
        correct_dye = csv_dyes == dyes[d].upper()
        for c in range(n_channels):
            correct_camera = csv_cameras == cameras[c]
            correct_laser = csv_lasers == lasers[c]
            correct_all = np.all((correct_dye, correct_camera, correct_laser), axis=0)
            if sum(correct_all) != 1:
                raise ValueError(f"Expected intensity for dye {dyes[d]}, camera {cameras[c]} and laser {lasers[c]} "
                                 f"to be found once in csv_file. Instead, it was found {sum(correct_all)} times.")
            dye_channel_intensity[d, c] = csv_intensities[np.where(correct_all)[0][0]]

    return dye_channel_intensity


def compute_bleed_matrix(initial_bleed_matrix: np.ndarray, spot_colours: np.ndarray,
                         round_split: bool = False, n_dyes: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes in a default bleed matrix (channels x dyes) and a collection of spots' colour vectors. Then computes the
    dot product of each colour vector with each of the columns (normalised of the bleed matrix). Then learns the dye
    association. Returns the same bleed matrix with scaled columns where the scale is the median of the scores for all
    spots of this dye.
    Args:
        initial_bleed_matrix: n_channels x n_dyes bleed matrix
        spot_colours: n_spots x n_rounds x n_channels colour matrix for each isolated spot
        round_split: If True, then the bleed matrix is computed separately for each round. If False, then the bleed
            matrix is computed for all rounds together.
        n_dyes: int. Number of dyes used in experiment. Default = 7
    Returns:
        bleed_matrix: n_channels x n_dyes updated bleed matrix or n_rounds x n_channels x n_dyes updated bleed matrix
        all_dye_score: n_spots x n_dyes array of scores for each spot and dye
    """
    # Dye score will be a list of n_dyes arrays. dye_score[i] gives an array of all dye scores for all spots allocated
    # to dye i
    n_spots, n_rounds, n_channels_use = spot_colours.shape[0], spot_colours.shape[1], spot_colours.shape[2]
    bleed_matrix_norm = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    # Convert colour_matrices to colour_vectors
    if round_split:
        colour_vector = np.swapaxes(spot_colours, 0, 1)
        bleed_matrix = np.zeros((n_rounds, initial_bleed_matrix.shape[0], initial_bleed_matrix.shape[1]))
    else:
        # If not round split then we just take all rounds together. Let's throw away the first 2 rounds as we have
        # a lot of anchor bleed in these rounds
        spot_colours = spot_colours[:, 2:, :]
        colour_vector = np.reshape(spot_colours, (n_spots * (n_rounds - 2), n_channels_use))[np.newaxis, :, :]
        bleed_matrix = np.zeros((1, initial_bleed_matrix.shape[0], initial_bleed_matrix.shape[1]))

    # Now we loop over the first dimension of the colour vector and compute the dye score for each spot. This allows us
    # to treat the 2 cases of round split and not round split in the same way
    for i in range(colour_vector.shape[0]):
        dye_score = []
        scale = []
        # Now compute the dot product of each spot with each dye. This gives an n_spots x n_dyes matrix
        all_dye_score = colour_vector[i] @ bleed_matrix_norm
        # Now assign each spot a dye which is its highest score
        spot_dye = np.argmax(all_dye_score, axis=1)
        spot_dye_score = np.max(all_dye_score, axis=1)
        spot_dye_score_second = np.sort(all_dye_score, axis=1)[:, -2]
        # Now we want to remove all spots which have a score of 0 or less than twice the second highest score
        dye_score_valid = (spot_dye_score > 0) * (spot_dye_score > 2 * spot_dye_score_second)
        spot_dye_score = spot_dye_score[dye_score_valid]
        spot_dye = spot_dye[dye_score_valid]

        # Loop through all dyes and partition our data nicely.
        # Also compute the median of each of these, this will be our scale factor for each column
        for d in range(n_dyes):
            score_d = spot_dye_score[spot_dye == d]
            scale_d = np.median(score_d)
            # Append these
            dye_score.append(score_d)
            scale.append(scale_d)
            # Add scaled copy of each of the cols to our new bleed matrix
            bleed_matrix[i, :, d] = scale_d * bleed_matrix_norm[:, d]

    # Repeat the first dimension of bleed_matrix if round_split is False
    if round_split is False:
        bleed_matrix = np.repeat(bleed_matrix, n_rounds, axis=0)

    return bleed_matrix, all_dye_score
