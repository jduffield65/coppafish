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


def check_color_nan(colors: np.ndarray, nbp_basic: NotebookPage):
    """
    `colors` should only contain the `invalid_value` in rounds/channels not in use_rounds/channels.
    This raises an error if this is not the case or if a round/channel not in use_rounds/channels
    contains a value other than `invalid_value`.
    `invalid_value = -nbp_basic.tile_pixel_value_shift - 1` if colors is integer i.e. the non-normalised colors,
    usually spot_colors.
    `invalid_value = -np.nan` if colors is float i.e. the normalised colors or most likely the bled_codes.


    Args:
        colors: `int or float [n_codes x n_rounds x n_channels]`
            `colors[s, r, c]` is the color for code `s` in round `r`, channel `c`.
            This is likely to be `spot_colors` if `int` or `bled_codes` if `float`.
        nbp_basic: basic_info NotebookPage
    """
    diff_to_int = np.round(colors).astype(int) - colors
    if np.abs(diff_to_int).max() == 0:
        # if not normalised, then invalid_value is an integer value that is impossible for a spot_color to be
        invalid_value = -nbp_basic.tile_pixel_value_shift
    else:
        # if is normalised then expect nan value to be normal np.nan.
        invalid_value = np.nan

    # decide which rounds/channels should be ignored i.e. only contain invalid_value.
    n_spots, n_rounds, n_channels = colors.shape
    if n_rounds == nbp_basic.n_rounds and n_channels == nbp_basic.n_channels:
        use_rounds = nbp_basic.use_rounds
        use_channels = nbp_basic.use_channels
    elif n_rounds == len(nbp_basic.use_rounds) and n_channels == len(nbp_basic.use_channels):
        use_rounds = np.arange(n_rounds)
        use_channels = np.arange(n_channels)
    else:
        raise ColorInvalidError(colors, nbp_basic, invalid_value)

    ignore_rounds = np.setdiff1d(np.arange(n_rounds), use_rounds)
    for r in ignore_rounds:
        unique_vals = np.unique(colors[:, r, :])
        for val in unique_vals:
            if not invalid_value in unique_vals:
                raise ColorInvalidError(colors, nbp_basic, invalid_value, round_no=r)
            if not np.array_equal(val, invalid_value, equal_nan=True):
                raise ColorInvalidError(colors, nbp_basic, invalid_value, round_no=r)

    ignore_channels = np.setdiff1d(np.arange(n_channels), use_channels)
    for c in ignore_channels:
        unique_vals = np.unique(colors[:, :, c])
        for val in unique_vals:
            if not invalid_value in unique_vals:
                raise ColorInvalidError(colors, nbp_basic, invalid_value, channel_no=c)
            if not np.array_equal(val, invalid_value, equal_nan=True):
                raise ColorInvalidError(colors, nbp_basic, invalid_value, channel_no=c)

    # see if any spots contain invalid_values.
    use_colors = colors[np.ix_(np.arange(n_spots), use_rounds, use_channels)]
    if np.array_equal(invalid_value, np.nan, equal_nan=True):
        nan_codes = np.where(np.isnan(use_colors))
    else:
        nan_codes = np.where(use_colors == invalid_value)
    n_nan_spots = nan_codes[0].size
    if n_nan_spots > 0:
        s = nan_codes[0][0]
        # round, channel number in spot_colors different from in use_spot_colors.
        r = np.arange(n_rounds)[nan_codes[1][0]]
        c = np.arange(n_channels)[nan_codes[2][0]]
        raise ColorInvalidError(colors, nbp_basic, invalid_value, round_no=r, channel_no=c, code_no=s)


class ColorInvalidError(Exception):
    def __init__(self, colors: np.ndarray, nbp_basic: NotebookPage, invalid_value: float, round_no: Optional[int] = None,
                 channel_no: Optional[int] = None, code_no: Optional[int] = None):
        """
        Error raised because `spot_colors` contains a `invalid_value` where it should not.

        Args:
            colors: `int or float [n_codes x n_rounds x n_channels]`
                `colors[s, r, c]` is the color for code `s` in round `r`, channel `c`.
                This is likely to be `spot_colors` if `int` or `bled_codes` if `float`.
            nbp_basic: basic_info NotebookPage
            invalid_value: This is the value that colors should only be in rounds/channels not used.
                Likely to be np.nan if colors is float or -nbp_basic.tile_pixel_value_shift if integer.
            round_no: round to flag error for.
            channel_no: channel to flag error for.
            code_no: Spot or gene index to flag error for.
        """
        n_spots, n_rounds, n_channels = colors.shape
        if round_no is not None and code_no is None:
            self.message = f"colors contains a value other than invalid_value={invalid_value} in round {round_no}\n" \
                           f"which is not in use_rounds = {nbp_basic.use_rounds}."
        elif channel_no is not None and code_no is None:
            self.message = f"colors contains a value other than invalid_value={invalid_value} in channel {channel_no}\n" \
                           f"which is not in use_channels = {nbp_basic.use_channels}."
        elif round_no is not None and channel_no is not None and code_no is not None:
            self.message = f"colors contains a invalid_value={invalid_value} for code {code_no}, round {round_no}, " \
                           f"channel {channel_no}.\n" \
                           f"There should be no invalid_values in this round and channel."
        else:
            self.message = f"colors has n_rounds = {n_rounds} and n_channels = {n_channels}.\n" \
                           f"This is neither matches the total_rounds = {nbp_basic.n_rounds} and " \
                           f"total_channels = {nbp_basic.n_channels}\n" \
                           f"nor the number of use_rounds = {len(nbp_basic.use_rounds)} and use_channels = " \
                           f"{len(nbp_basic.use_channels)}"
        super().__init__(self.message)
