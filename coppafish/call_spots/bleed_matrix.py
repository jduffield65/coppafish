import numpy as np
from .. import utils
from typing import Tuple, List, Union
import itertools
import matplotlib.pyplot as plt


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


def compute_bleed_matrix(initial_bleed_matrix: np.ndarray, spot_colours: np.ndarray, spot_tile: np.ndarray,
                         n_tiles: int, tile_split: bool = False, round_split: bool = False,
                         n_dyes: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes in a bleed matrix template and a bunch of isolated spots.
    Then assigns spots in each round to dyes by maximising cosine similarity between the spot colour and the dye colour.
    Then updates the bleed matrix by averaging the colour of each dye across all spots allocated to that dye.
    Averaging is not quite right, we take the first singular vector of the spot colours assigned to each dye.

    Args:
        initial_bleed_matrix: n_channels x n_dyes bleed matrix
        spot_colours: n_spots x n_rounds x n_channels colour matrix for each isolated spot
        spot_tile: n_spots array of tile numbers for each spot
        n_tiles: int. Number of tiles in experiment
        tile_split: bool. If True, then compute bleed matrix for each tile separately. Default = False
        round_split: bool. If True, then compute bleed matrix for each round separately. Default = False
        n_dyes: int. Number of dyes used in experiment. Default = 7
    Returns:
        bleed_matrix: n_channels x n_dyes updated bleed matrix or n_rounds x n_channels x n_dyes updated bleed matrix
        all_dye_score: n_spots x n_dyes array of scores for each spot and dye
    """
    # Check inputs
    if round_split:
        bleed_rounds = spot_colours.shape[1]
    else:
        bleed_rounds = 1

    if tile_split:
        bleed_tiles = n_tiles
    else:
        bleed_tiles = 1

    n_tiles, n_rounds, n_channels = len(set(spot_tile)), spot_colours.shape[1], spot_colours.shape[2]
    # First define the bleed matrix (normalised so that each dye has unit norm)
    bleed_matrix_norm = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    # Now define bleed matrix
    bleed_matrix = np.zeros((bleed_tiles, bleed_rounds, bleed_matrix_norm.shape[0], bleed_matrix_norm.shape[1]))
    colour_vector = np.zeros((bleed_tiles, bleed_rounds, 0)).tolist()

    # Populate colour vector with spot colours
    for t, r in itertools.product(range(bleed_tiles), range(bleed_rounds)):
        # Cover 4 cases: tile_split, round_split, both, neither
        if tile_split and round_split:
            colour_vector[t][r] = spot_colours[spot_tile == t, r, :]
        elif tile_split and not round_split:
            colour_vector[t][r] = spot_colours[spot_tile == t, :, :].reshape(-1, n_channels)
        elif not tile_split and round_split:
            colour_vector[t][r] = spot_colours[:, r, :]
        else:
            colour_vector[t][r] = spot_colours.reshape(-1, n_channels)

    # Now compute the bleed matrices
    for t, r in itertools.product(range(bleed_tiles), range(bleed_rounds)):
        # Now compute the dot product of each spot with each dye. This gives an n_spots_use x n_dyes matrix.
        all_dye_score = colour_vector[t][r] @ bleed_matrix_norm
        # Now assign each spot a dye which is its highest score
        spot_dye, spot_dye_score = np.argmax(all_dye_score, axis=1), np.max(all_dye_score, axis=1)
        spot_dye_score_second = np.sort(all_dye_score, axis=1)[:, -2]
        # Now we want to remove all spots which have a score of 0 or less than twice the second-highest score
        keep = (spot_dye_score > 0) * (spot_dye_score > 1.5 * spot_dye_score_second)

        # Now we loop over the dyes and fine tune the spectrum for each dye
        for d in range(n_dyes):
            # Get all spots which have been confidently assigned to dye d
            d_spots = colour_vector[t][r][(spot_dye == d) * keep]
            # Compute the first singular vector of these spots
            d_spectrum, d_score = fine_tune_dye_spectrum(spot_colours=d_spots)
            bleed_matrix[t, r, :, d] = d_spectrum

        # Now normalise the bleed matrix
        bleed_matrix[t, r] = bleed_matrix[t, r] / np.linalg.norm(bleed_matrix[t, r])

    return bleed_matrix, all_dye_score


def fine_tune_dye_spectrum(spot_colours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes in a collection of spots' colour vectors assigned to a single dye. Then computes the first singular vector.
    Args:
        spot_colours: n_spots x n_channels colour matrix for each isolated spot assigned to a single dye

    Returns:
        dye_spectrum: n_channels x 1 array of the first singular vector
        dye_score: n_spots x 1 array of the score of each spot for this dye

    """
    # Compute the outer product of the colour vectors and then compute the first singular vector as the eigenvector
    # corresponding to the largest eigenvalue. This is the best vector that each spot is roughly a multiple of.
    outer_prod = spot_colours.T @ spot_colours
    eig_val, eig_vec = np.linalg.eig(outer_prod)
    dye_spectrum = np.real(eig_vec[:, np.argmax(eig_val)])

    # If the largest absolute value of the spectrum is negative, then we flip the spectrum
    max_index = np.argmax(np.abs(dye_spectrum))
    if dye_spectrum[max_index] < 0:
        dye_spectrum = -dye_spectrum

    # Now compute the score of each spot for this dye. This is just the dot product of the spot colour with the
    # dye spectrum (which is normalised to have unit norm)
    dye_score = spot_colours @ dye_spectrum
    dye_spectrum = dye_spectrum * np.max(eig_val)
    return dye_spectrum, dye_score
