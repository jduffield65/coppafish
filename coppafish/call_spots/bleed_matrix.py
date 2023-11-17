import numpy as np
from typing import Tuple, List, Union
import itertools

from .. import utils


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
                         dye_score_thresh: float = 0) -> np.ndarray:
    """
    Takes in spots from a single tile and round and then computes the bleed matrix for that tile and round.
    Args:
        initial_bleed_matrix: n_channels x n_dyes bleed matrix (each dye normalised across channels)
        spot_colours: n_spots x n_channels colour spectrum for each dye
        dye_score_thresh: float, threshold for cosine similarity score for each dye. If the score is below this,
        then the dye won't be used for the calculation of the bleed matrix.
    Returns:
        bleed_matrix: n_channels x n_dyes bleed matrix (each dye normalised across channels)
    """
    # Check inputs
    n_channels, n_dyes = initial_bleed_matrix.shape
    # Now define bleed matrix
    initial_bleed_matrix = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    bleed_matrix = np.zeros((n_channels, n_dyes))

    # Now compute the dot product of each spot with each dye. This gives an n_spots_use x n_dyes matrix.
    all_dye_score = spot_colours @ initial_bleed_matrix
    # Now assign each spot a dye which is its highest score
    spot_dye, spot_dye_score = np.argmax(all_dye_score, axis=1), np.max(all_dye_score, axis=1)
    keep = spot_dye_score > dye_score_thresh

    # Now we loop over the dyes and fine tune the spectrum for each dye
    for d in range(n_dyes):
        # Get all spots which have been confidently assigned to dye d
        d_spots = spot_colours[(spot_dye == d) * keep]
        # Compute the first singular vector of these spots
        d_spectrum = fine_tune_dye_spectrum(d_spots)
        bleed_matrix[:, d] = d_spectrum

    return bleed_matrix


def fine_tune_dye_spectrum(spot_colours: np.ndarray) -> np.ndarray:
    """
    Takes in a collection of spots' colour vectors assigned to a single dye. Then computes the first singular vector.
    Args:
        spot_colours: n_spots x n_channels colour matrix for each isolated spot assigned to a single dye

    Returns:
        dye_spectrum: n_channels x 1 array of the first singular vector

    """
    # Compute the outer product of the colour vectors and then compute the first singular vector as the eigenvector
    # corresponding to the largest eigenvalue. This is the best vector that each spot is roughly a multiple of.
    outer_prod = spot_colours.T @ spot_colours
    eig_val, eig_vec = np.linalg.eig(outer_prod)
    dye_spectrum = np.real(eig_vec[:, np.argmax(eig_val)])
    dye_score = spot_colours @ dye_spectrum

    # We expect the dye_spectrum to be positive multiples of each row. If median is negative, then flip the spectrum
    if np.median(dye_score) < 0:
        dye_spectrum = -dye_spectrum
        dye_score = -dye_score

    scale = np.median(dye_score)
    dye_spectrum = dye_spectrum * scale

    return dye_spectrum
