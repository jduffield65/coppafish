import numpy as np
from .. import utils
from typing import Tuple, List, Union
import warnings


def scaled_k_means(x: np.ndarray, initial_cluster_mean: np.ndarray,
                   score_thresh: Union[float, np.ndarray] = 0, min_cluster_size: int = 10,
                   n_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Does a clustering that minimizes the norm of ```x[i] - g[i] * cluster_mean[cluster_ind[i]]```
    for each data point ```i``` in ```x```, where ```g``` is the gain which is not explicitly computed.

    Args:
        x: ```float [n_points x n_dims]```.
            Data set of vectors to build cluster means from.
        initial_cluster_mean: ```float [n_clusters x n_dims]```.
            Starting point of mean cluster vectors.
        score_thresh: `float` or give different score for each cluster as `float [n_clusters]`
            Scalar between ```0``` and ```1```.
            Points in ```x``` with dot product to a cluster mean vector greater than this
            contribute to new estimate of mean vector.
        min_cluster_size: If less than this many points assigned to a cluster,
            that cluster mean vector will be set to ```0```.
        n_iter: Maximum number of iterations performed.

    Returns:
        - norm_cluster_mean - ```float [n_clusters x n_dims]```.
            Final normalised mean cluster vectors.
        - cluster_eig_value - ```float [n_clusters]```.
            First eigenvalue of outer product matrix for each cluster.
        - cluster_ind - ```int [n_points]```.
            Index of cluster each point was assigned to. ```-1``` means fell below score_thresh and not assigned.
        - top_score - ```float [n_points]```.
            `top_score[i]` is the dot product score between `x[i]` and `norm_cluster_mean[cluster_ind[i]]`.
        - cluster_ind0 - ```int [n_points]```.
            Index of cluster each point was assigned to on first iteration.
            ```-1``` means fell below score_thresh and not assigned.
        - top_score0 - ```float [n_points]```.
            `top_score0[i]` is the dot product score between `x[i]` and `initial_cluster_mean[cluster_ind0[i]]`.
    """
    # normalise starting points and original data
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(initial_cluster_mean, axis=1).reshape(-1, 1)
    x_norm = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    n_clusters = initial_cluster_mean.shape[0]
    n_points, n_dims = x.shape
    cluster_ind = np.ones(x.shape[0], dtype=int) * -2  # set all to -2 so won't end on first iteration
    cluster_eig_val = np.zeros(n_clusters)

    if not utils.errors.check_shape(initial_cluster_mean, [n_clusters, n_dims]):
        raise utils.errors.ShapeError('initial_cluster_mean', initial_cluster_mean.shape, (n_clusters, n_dims))

    if len(np.array([score_thresh]).flatten()) == 1:
        # if single threshold, set the same for each cluster
        score_thresh = np.ones(n_clusters) * score_thresh

    if not utils.errors.check_shape(score_thresh, [n_clusters]):
        raise utils.errors.ShapeError('score_thresh', score_thresh.shape, (n_clusters,))

    for i in range(n_iter):
        cluster_ind_old = cluster_ind.copy()

        # project each point onto each cluster. Use normalized so we can interpret score
        score = x_norm @ norm_cluster_mean.transpose()
        cluster_ind = np.argmax(score, axis=1)  # find best cluster for each point
        top_score = score[np.arange(n_points), cluster_ind]
        top_score[np.where(np.isnan(top_score))[0]] = score_thresh.min()-1  # don't include nan values
        cluster_ind[top_score < score_thresh[cluster_ind]] = -1  # unclusterable points
        if i == 0:
            top_score0 = top_score.copy()
            cluster_ind0 = cluster_ind.copy()

        if (cluster_ind == cluster_ind_old).all():
            break

        for c in range(n_clusters):
            my_points = x[cluster_ind == c]  # don't use normalized, to avoid overweighting weak points
            n_my_points = my_points.shape[0]
            if n_my_points < min_cluster_size:
                norm_cluster_mean[c] = 0
                warnings.warn(f"Cluster c only had {n_my_points} vectors assigned to it.\n "
                              f"This is less than min_cluster_size = {min_cluster_size} so setting this cluster to 0.")
                continue
            eig_vals, eigs = np.linalg.eig(my_points.transpose() @ my_points / n_my_points)
            best_eig_ind = np.argmax(eig_vals)
            norm_cluster_mean[c] = eigs[:, best_eig_ind] * np.sign(eigs[:, best_eig_ind].mean())  # make them positive
            cluster_eig_val[c] = eig_vals[best_eig_ind]

    return norm_cluster_mean, cluster_eig_val, cluster_ind, top_score, cluster_ind0, top_score0


def get_bleed_matrix(spot_colors: np.ndarray, initial_bleed_matrix: np.ndarray, method: str, score_thresh: float = 0,
                     min_cluster_size: int = 10, n_iter: int = 100, score_thresh_anneal: bool = True,
                     debug: int = -1) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    This returns a bleed matrix such that the expected intensity of dye ```d``` in round ```r```
    is a constant multiple of ```bleed_matrix[r, :, d]```.

    Args:
        spot_colors: ```float [n_spots x n_rounds x n_channels]```.
            Intensity found for each spot in each round and channel, normalized in some way to equalize channel
            intensities typically, the normalisation will be such that spot_colors vary between around
            ```-5``` to ```10``` with most near ```0```.
        initial_bleed_matrix: ```float [n_rounds x n_channels x n_dyes]```.
            Initial guess for intensity we expect each dye to produce in each channel and round.
            Should be normalized in same way as spot_colors.
        method: Must be one of the following:

            - ```'single'``` - A single bleed matrix is produced for all rounds.
            - ```'separate'``` - A different bleed matrix is made for each round.
        score_thresh: Scalar between ```0``` and ```1```.
            Threshold used for ```scaled_k_means``` affecting which spots contribute to bleed matrix estimate.
        min_cluster_size: If less than this many points assigned to a dye, that dye mean vector will be set to ```0```.
        n_iter: Maximum number of iterations performed in ```scaled_k_means```.
        score_thresh_anneal: If `True`, `scaled_k_means` will be performed twice.
            The second time starting with the output of the first and with `score_thresh` for cluster `i`
            set to the median of the scores assigned to cluster `i` in the first run.
            This limits the influence of bad spots to the bleed matrix.
        debug: If this is >=0, then the `debug_info` dictionary will also be returned.
            If `method == 'separate'`, this specifies the round of the `bleed_matrix` calculation to return
            debugging info for.

    Returns:
        `bleed_matrix` - ```float [n_rounds x n_channels x n_dyes]```.
            ```bleed_matrix``` such that the expected intensity of dye ```d``` in round ```r```
            is a constant multiple of ```bleed_matrix[r, _, d]```.
        `debug_info` - dictionary containing useful information for debugging bleed matrix calculation.
            Each variable has size=3 in first dimension. `var[0]` refers to value with `initial_bleed_matrix`.
            `var[1]` refers to value after first `scaled_k_means`.
            `var[2]` refers to value after second k means (only if `score_thresh_anneal == True`).

            - `cluster_ind`: `int8 [3 x n_vectors]`. Index of dye each vector was assigned to in `scaled_k_means`.
                ```-1``` means fell below score_thresh and not assigned.
            - `cluster_score`: `float16 [3 x n_vectors]`. Value of dot product between each vector and dye assigned to.
            - `bleed_matrix`: `float16 [3 x n_channels x n_dyes]`.
                `bleed_matrix` computed at each stage of calculation.
    """
    n_rounds, n_channels = spot_colors.shape[1:]
    n_dyes = initial_bleed_matrix.shape[2]
    if not utils.errors.check_shape(initial_bleed_matrix, [n_rounds, n_channels, n_dyes]):
        raise utils.errors.ShapeError('initial_bleed_matrix', initial_bleed_matrix.shape, (n_rounds, n_channels,
                                                                                           n_dyes))

    bleed_matrix = np.zeros((n_rounds, n_channels, n_dyes))  # Round, Measured, Real
    debug_info = None
    if method.lower() == 'separate':
        for r in range(n_rounds):
            spot_channel_intensity = spot_colors[:, r, :]
            # get rid of any nan codes
            spot_channel_intensity = spot_channel_intensity[~np.isnan(spot_channel_intensity).any(axis=1)]
            if r == debug:
                debug_info = {'cluster_ind': np.zeros((2+score_thresh_anneal, spot_channel_intensity.shape[0]),
                                                      dtype=np.int8),
                              'cluster_score': np.zeros((2+score_thresh_anneal, spot_channel_intensity.shape[0]),
                                                        dtype=np.float16),
                              'bleed_matrix': np.zeros((2+score_thresh_anneal, n_channels, n_dyes)),
                              'round': r}
            dye_codes, dye_eig_vals, cluster_ind, cluster_score, cluster_ind0, cluster_score0 = \
                scaled_k_means(spot_channel_intensity, initial_bleed_matrix[r].transpose(),
                               score_thresh, min_cluster_size, n_iter)
            if r == debug:
                debug_info['bleed_matrix'][0] = initial_bleed_matrix[r]
                debug_info['cluster_ind'][0] = cluster_ind0
                debug_info['cluster_ind'][1] = cluster_ind
                debug_info['cluster_score'][0] = cluster_score0
                debug_info['cluster_score'][1] = cluster_score
                for d in range(n_dyes):
                    debug_info['bleed_matrix'][1, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
            if score_thresh_anneal:
                # repeat with higher score_thresh so bad spots contribute less.
                score_thresh2 = np.zeros(n_dyes)
                for d in range(n_dyes):
                    score_thresh2[d] = np.median(cluster_score[cluster_ind == d])
                dye_codes, dye_eig_vals, cluster_ind, cluster_score = \
                    scaled_k_means(spot_channel_intensity, dye_codes, score_thresh2, min_cluster_size, n_iter)[:4]
                if r == debug:
                    debug_info['cluster_ind'][2] = cluster_ind
                    debug_info['cluster_score'][2] = cluster_score
                    for d in range(n_dyes):
                        debug_info['bleed_matrix'][2, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
            for d in range(n_dyes):
                bleed_matrix[r, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
    elif method.lower() == 'single':
        initial_bleed_matrix_round_diff = initial_bleed_matrix.max(axis=0) - initial_bleed_matrix.min(axis=0)
        if np.max(np.abs(initial_bleed_matrix_round_diff)) > 1e-10:
            raise ValueError(f"method is {method}, but initial_bleed_matrix is different for different rounds.")

        spot_channel_intensity = spot_colors.reshape(-1, n_channels)
        # get rid of any nan codes
        spot_channel_intensity = spot_channel_intensity[~np.isnan(spot_channel_intensity).any(axis=1)]
        if debug >= 0:
            debug_info = {'cluster_ind': np.zeros((2 + score_thresh_anneal, spot_channel_intensity.shape[0]),
                                                  dtype=np.int8),
                          'cluster_score': np.zeros((2 + score_thresh_anneal, spot_channel_intensity.shape[0]),
                                                    dtype=np.float16),
                          'bleed_matrix': np.zeros((2 + score_thresh_anneal, n_channels, n_dyes))}
        dye_codes, dye_eig_vals, cluster_ind, cluster_score, cluster_ind0, cluster_score0 = \
            scaled_k_means(spot_channel_intensity, initial_bleed_matrix[0].transpose(),
                           score_thresh, min_cluster_size, n_iter)
        if debug >= 0:
            debug_info['bleed_matrix'][0] = initial_bleed_matrix[0]
            debug_info['cluster_ind'][0] = cluster_ind0
            debug_info['cluster_ind'][1] = cluster_ind
            debug_info['cluster_score'][0] = cluster_score0
            debug_info['cluster_score'][1] = cluster_score
            for d in range(n_dyes):
                debug_info['bleed_matrix'][1, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
        if score_thresh_anneal:
            # repeat with higher score_thresh so bad spots contribute less.
            score_thresh2 = np.zeros(n_dyes)
            for d in range(n_dyes):
                score_thresh2[d] = np.median(cluster_score[cluster_ind == d])
            dye_codes, dye_eig_vals, cluster_ind, cluster_score = \
                scaled_k_means(spot_channel_intensity, dye_codes, score_thresh2, min_cluster_size, n_iter)[:4]
            if debug >= 0:
                debug_info['cluster_ind'][2] = cluster_ind
                debug_info['cluster_score'][2] = cluster_score
                for d in range(n_dyes):
                    debug_info['bleed_matrix'][2, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
        for r in range(n_rounds):
            for d in range(n_dyes):
                bleed_matrix[r, :, d] = dye_codes[d] * np.sqrt(dye_eig_vals[d])
    else:
        raise ValueError(f"method given was {method} but should be either 'single' or 'separate'")
    if debug_info is not None:
        return bleed_matrix, debug_info
    else:
        return bleed_matrix


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
