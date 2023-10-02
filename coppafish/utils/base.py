import numpy as np
from typing import Union


def round_any(x: Union[float, np.ndarray], base: float, round_type: str = 'round') -> Union[float, np.ndarray]:
    """
    Rounds `x` to the nearest multiple of `base` with the rounding done according to `round_type`.

    Args:
        x: Number or array to round.
        base: Rounds `x` to nearest integer multiple of value of `base`.
        round_type: One of the following, indicating how to round `x` -

            - `'round'`
            - `'ceil'`
            - `'floor'`

    Returns:
        Rounded version of `x`.

    Example:
        ```
        round_any(3, 5) = 5
        round_any(3, 5, 'floor') = 0
        ```
    """
    if round_type == 'round':
        return base * np.round(x / base)
    elif round_type == 'ceil':
        return base * np.ceil(x / base)
    elif round_type == 'floor':
        return base * np.floor(x / base)
    else:
        raise ValueError(f"round_type specified was {round_type} but it should be one of the following:\n"
                         f"round, ceil, floor")


def setdiff2d(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Finds all unique elements in `array1` that are also not in `array2`. Each element is appended along the first \
    axis. E.g.

    If `array1` has `[4,0]` twice, `array2` does not have `[4,0]`, returned array will have `[4,0]` once.

    If `array1` has `[4,0]` twice, `array2` has `[4,0]` once, returned array will not have `[4,0]`.

    Args:
        array1: `float [n_elements1 x element_dim]`.
        array2: `float [n_elements2 x element_dim]`.

    Returns:
        `float [n_elements_diff x element_dim]`.
    """
    set1 = set([tuple(x) for x in array1])
    set2 = set([tuple(x) for x in array2])
    return np.array(list(set1-set2))


def expand_channels(array: np.ndarray, use_channels: list, n_channels: int) -> np.ndarray:
    """
    Expands `array` to have `n_channels` channels, with the values in `array` being in the channels specified by
    `use_channels`.

    Args:
        array: `float [n1 x n2 x ... x n_k x n_channels_use]`.
        use_channels: list of channels to use from `array`.
        n_channels: Number of channels to expand `array` to.

    Returns:
        expanded_array: `float [n1 x n2 x ... x n_k x n_channels_use].
    """
    old_array_shape = np.array(array.shape)
    new_array_shape = old_array_shape.copy()
    new_array_shape[-1] = n_channels
    expanded_array = np.zeros(new_array_shape)

    for i, channel in enumerate(use_channels):
        expanded_array[..., channel] = array[..., i]

    return expanded_array
