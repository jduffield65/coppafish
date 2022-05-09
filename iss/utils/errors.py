import numpy as np
from typing import Tuple, Union, Optional
from ..setup.notebook import NotebookPage


class OutOfBoundsError(Exception):
    def __init__(self, var_name: str, oob_val: float, min_allowed: float, max_allowed: float):
        """
        Error raised because `oob_val` is outside expected range between
        `min_allowed` and `max_allowed` inclusive.

        Args:
            var_name: Name of variable testing.
            oob_val: Value in array that is not in expected range.
            min_allowed: Smallest allowed value i.e. `>= min_allowed`.
            max_allowed: Largest allowed value i.e. `<= max_allowed`.
        """
        self.message = f"\n{var_name} contains the value {oob_val}." \
                       f"\nThis is outside the expected inclusive range between {min_allowed} and {max_allowed}"
        super().__init__(self.message)


class NoFileError(Exception):
    def __init__(self, file_path: str):
        """
        Error raised because `file_path` does not exist.

        Args:
            file_path: Path to file of interest.
        """
        self.message = f"\nNo file with the following path:\n{file_path}\nexists"
        super().__init__(self.message)


class EmptyListError(Exception):
    def __init__(self, var_name: str):
        """
        Error raised because the variable indicated by `var_name` contains no data.

        Args:
            var_name: Name of list or numpy array
        """
        self.message = f"\n{var_name} contains no data"
        super().__init__(self.message)


def check_shape(array: np.ndarray, expected_shape: Union[list, tuple, np.ndarray]) -> bool:
    """
    Checks to see if `array` has the shape indicated by `expected_shape`.

    Args:
        array: Array to check the shape of.
        expected_shape: `int [n_array_dims]`.
            Expected shape of array.

    Returns:
        `True` if shape of array is correct.
    """
    correct_shape = array.ndim == len(expected_shape)  # first check if number of dimensions are correct
    if correct_shape:
        correct_shape = np.abs(np.array(array.shape) - np.array(expected_shape)).max() == 0
    return correct_shape


class ShapeError(Exception):
    def __init__(self, var_name: str, var_shape: tuple, expected_shape: tuple):
        """
        Error raised because variable indicated by `var_name` has wrong shape.

        Args:
            var_name: Name of numpy array.
            var_shape: Shape of numpy array.
            expected_shape: Expected shape of numpy array.
        """
        self.message = f"\nShape of {var_name} is {var_shape} but should be {expected_shape}"
        super().__init__(self.message)


class TiffError(Exception):
    def __init__(self, scale_tiff: float, scale_nbp: float, shift_tiff: int, shift_nbp: int):
        """
        Error raised because parameters used to produce tiff files are different to those in the current notebook.

        Args:
            scale_tiff: Scale factor applied to tiff. Found from tiff description.
            scale_nbp: Scale factor applied to tiff. Found from `nb.extract_debug.scale`.
            shift_tiff: Shift applied to tiff to ensure pixel values positive. Found from tiff description.
            shift_nbp: Shift applied to tiff to ensure pixel values positive.
                Found from `nb.basic_info.tile_pixel_value_shift`.
        """
        self.message = f"\nThere are differences between the parameters used to make the tiffs and the parameters " \
                       f"in the Notebook:"
        if scale_tiff != scale_nbp:
            self.message = self.message + f"\nScale used to make tiff was {scale_tiff}." \
                                          f"\nCurrent scale in extract_params notebook page is {scale_nbp}."
        if shift_tiff != shift_nbp:
            self.message = self.message + f"\nShift used to make tiff was {shift_tiff}." \
                                          f"\nCurrent tile_pixel_value_shift in basic_info notebook page is " \
                                          f"{shift_nbp}."
        super().__init__(self.message)


def check_spot_color_nan(spot_colors: np.ndarray, nbp_basic: NotebookPage):
    """
    `spot_colors` should only contain the `nan_value = -nbp_basic.tile_pixel_value_shift - 1` in rounds/channels not in
    use_rounds/channels. This raises an error if this is not the case or
    if a round/channel not in use_rounds/channels contains a value other than `nan_value`.

    Args:
        spot_colors: `int [n_spots x n_rounds x n_channels]`
            `spot_colors[s, r, c]` is the spot color for spot `s` in round `r`, channel `c`.
        nbp_basic: basic_info NotebookPage
    """
    diff_to_int = np.round(spot_colors).astype(int) - spot_colors
    if np.abs(diff_to_int).max() != 0:
        raise ValueError("check_nan should be found using non-normalised spot_colors. "
                         "\nBut all values in spot_colors given are floats indicating they are"
                         " the normalised intensities.")

    # decide which rounds/channels should be ignored i.e. only contain nan_value.
    n_spots, n_rounds, n_channels = spot_colors.shape
    nan_value = -nbp_basic.tile_pixel_value_shift - 1
    if n_rounds == nbp_basic.n_rounds and n_channels == nbp_basic.n_channels:
        use_rounds = nbp_basic.use_rounds
        use_channels = nbp_basic.use_channels
    elif n_rounds == len(nbp_basic.use_rounds) and n_channels == len(nbp_basic.use_channels):
        use_rounds = np.arange(n_rounds)
        use_channels = np.arange(n_channels)
    else:
        raise SpotColorNanError(spot_colors, nbp_basic)

    ignore_rounds = np.setdiff1d(np.arange(n_rounds), use_rounds)
    for r in ignore_rounds:
        unique_vals = np.unique(spot_colors[:, r, :])
        for val in unique_vals:
            if not nan_value in unique_vals:
                raise SpotColorNanError(spot_colors, nbp_basic, round_no=r)
            if val != nan_value:
                raise SpotColorNanError(spot_colors, nbp_basic, round_no=r)

    ignore_channels = np.setdiff1d(np.arange(n_channels), use_channels)
    for c in ignore_channels:
        unique_vals = np.unique(spot_colors[:, :, c])
        for val in unique_vals:
            if not nan_value in unique_vals:
                raise SpotColorNanError(spot_colors, nbp_basic, channel_no=c)
            if val != nan_value:
                raise SpotColorNanError(spot_colors, nbp_basic, channel_no=c)

    # see if any spots contain nan_values.
    use_spot_colors = spot_colors[np.ix_(np.arange(n_spots), use_rounds, use_channels)]
    nan_spots = np.where(use_spot_colors == nan_value)
    n_nan_spots = nan_spots[0].size
    if n_nan_spots > 0:
        s = nan_spots[0][0]
        # round, channel number in spot_colors different from in use_spot_colors.
        r = np.arange(n_rounds)[nan_spots[1][0]]
        c = np.arange(n_channels)[nan_spots[2][0]]
        raise SpotColorNanError(spot_colors, nbp_basic, round_no=r, channel_no=c, spot_no=s)


class SpotColorNanError(Exception):
    def __init__(self, spot_colors: np.ndarray, nbp_basic: NotebookPage, round_no: Optional[int] = None,
                 channel_no: Optional[int] = None, spot_no: Optional[int] = None):
        """
        Error raised because `spot_colors` contains a `nan_value` where it should not.

        Args:
            spot_colors: `int [n_spots x n_rounds x n_channels]`
                `spot_colors[s, r, c]` is the spot color for spot `s` in round `r`, channel `c`.
            nbp_basic: basic_info NotebookPage
            round_no: round to flag error for.
            channel_no: channel to flag error for.
            spot_no: Spot index to flag error for.
        """
        n_spots, n_rounds, n_channels = spot_colors.shape
        nan_value = -nbp_basic.tile_pixel_value_shift - 1
        if round_no is not None and spot_no is None:
            self.message = f"spot_colors contains a value other than nan_value={nan_value} in round {round_no}\n" \
                           f"which is not in use_rounds = {nbp_basic.use_rounds}."
        elif channel_no is not None and spot_no is None:
            self.message = f"spot_colors contains a value other than nan_value={nan_value} in channel {channel_no}\n" \
                           f"which is not in use_channels = {nbp_basic.use_channels}."
        elif round_no is not None and channel_no is not None and spot_no is not None:
            self.message = f"spot_colors contains a nan_value={nan_value} for spot {spot_no}, round {round_no}, " \
                           f"channel {channel_no}.\n" \
                           f"There should be no nan_values in this round and channel."
        else:
            self.message = f"spot_colors has n_rounds = {n_rounds} and n_channels = {n_channels}.\n" \
                           f"This is neither matches the total_rounds = {nbp_basic.n_rounds} and " \
                           f"total_channels = {nbp_basic.n_channels}\n" \
                           f"nor the number of use_rounds = {len(nbp_basic.use_rounds)} and use_channels = " \
                           f"{len(nbp_basic.use_channels)}"
        super().__init__(self.message)
