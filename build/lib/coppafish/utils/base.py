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
            - `'float'`

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
    Finds all elements in `array1` that are not in `array2`.
    Returned array will only contain unique elements E.g.

    If `array1` has `[4,0]` twice, `array2` has `[4,0]` once, returned array will not have `[4,0]`.

    If `array1` has `[4,0]` twice, `array2` does not have `[4,0]`, returned array will have `[4,0]` once.

    Args:
        array1: `float [n_elements1 x element_dim]`.
        array2: `float [n_elements2 x element_dim]`.

    Returns:
        `float [n_elements_diff x element_dim]`.
    """
    set1 = set([tuple(x) for x in array1])
    set2 = set([tuple(x) for x in array2])
    return np.array(list(set1-set2))
