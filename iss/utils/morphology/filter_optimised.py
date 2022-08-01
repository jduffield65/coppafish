import numbers
from typing import Union, Tuple
import jax
import numpy as np
from jax import numpy as jnp
from .base import ensure_odd_kernel


def get_shifts_from_kernel(kernel: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns where kernel is positive as shifts in y, x and z.
    I.e. `kernel=jnp.ones((3,3,3))` would return `y_shifts = x_shifts = z_shifts = -1, 0, 1`.
    Args:
        kernel: int [kernel_szY x kernel_szX x kernel_szY]

    Returns:
        - `int [n_shifts]`.
            y_shifts.
        - `int [n_shifts]`.
            x_shifts.
        - `int [n_shifts]`.
            z_shifts.
    """
    shifts = list(jnp.where(kernel > 0))
    for i in range(kernel.ndim):
        shifts[i] = (shifts[i] - (kernel.shape[i] - 1) / 2).astype(int)
    return tuple(shifts)


def manual_convolve_single(image: jnp.ndarray, y_kernel_shifts: jnp.ndarray, x_kernel_shifts: jnp.asarray,
                           z_kernel_shifts: jnp.ndarray, coord: jnp.ndarray) -> float:
    return jnp.sum(image[coord[0] + y_kernel_shifts, coord[1] + x_kernel_shifts, coord[2] + z_kernel_shifts])


@jax.jit
def manual_convolve(image: jnp.ndarray, y_kernel_shifts: jnp.ndarray, x_kernel_shifts: jnp.asarray,
                    z_kernel_shifts: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Finds result of convolution at specific locations indicated by `coords` with binary kernel.
    I.e. instead of convolving whole `image`, just find result at these `points`.

    !!! note
        image needs to be padded before this function is called otherwise get an error when go out of bounds.

    Args:
        image: `int [image_szY x image_szX x image_szZ]`.
            Image to be filtered. Must be 3D.
        y_kernel_shifts: `int [n_nonzero_kernel]`
            Shifts indicating where kernel equals 1.
            I.e. if `kernel = np.ones((3,3))` then `y_shift = x_shift = z_shift = [-1, 0, 1]`.
        x_kernel_shifts: `int [n_nonzero_kernel]`
            Shifts indicating where kernel equals 1.
            I.e. if `kernel = np.ones((3,3))` then `y_shift = x_shift = z_shift = [-1, 0, 1]`.
        z_kernel_shifts: `int [n_nonzero_kernel]`
            Shifts indicating where kernel equals 1.
            I.e. if `kernel = np.ones((3,3))` then `y_shift = x_shift = z_shift = [-1, 0, 1]`.
        coords: `int [n_points x 3]`.
            yxz coordinates where result of filtering is desired.

    Returns:
        `int [n_points]`.
            Result of filtering of `image` at each point in `coords`.
    """
    return jax.vmap(manual_convolve_single, in_axes=(None, None, None, None, 0),
                    out_axes=0)(image, y_kernel_shifts, x_kernel_shifts,z_kernel_shifts, coords)


def imfilter_coords(image: np.ndarray, kernel: np.ndarray, coords: np.ndarray, padding: Union[float, str] = 0,
                    corr_or_conv: str = 'corr') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Copy of MATLAB `imfilter` function with `'output_size'` equal to `'same'`.
    Only finds result of filtering at specific locations.

    !!! note
        image and image2 need to be np.int8 and kernel needs to be int otherwise will get cython error.

    Args:
        image: `int [image_szY x image_szX (x image_szZ)]`.
            Image to be filtered. Must be 2D or 3D.
        kernel: `int [kernel_szY x kernel_szX (x kernel_szZ)]`.
            Multidimensional filter, expected to be binary i.e. only contains 0 and/or 1.
        coords: `int [n_points x image.ndims]`.
            Coordinates where result of filtering is desired.
        padding: One of the following, indicated which padding to be used.

            - numeric scalar - Input array values outside the bounds of the array are assigned the value `X`.
                When no padding option is specified, the default is `0`.
            - `‘symmetric’` - Input array values outside the bounds of the array are computed by
                mirror-reflecting the array across the array border.
            - `‘edge’`- Input array values outside the bounds of the array are assumed to equal
                the nearest array border value.
            - `'wrap'` - Input array values outside the bounds of the array are computed by implicitly
                assuming the input array is periodic.
        corr_or_conv:
            - `'corr'` - Performs multidimensional filtering using correlation.
                This is the default when no option specified.
            - `'conv'` - Performs multidimensional filtering using convolution.

    Returns:
        `int [n_points]`.
            Result of filtering of `image` at each point in `coords`.
    """
    if corr_or_conv == 'corr':
        kernel = np.flip(kernel)
    elif corr_or_conv != 'conv':
        raise ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")
    kernel = ensure_odd_kernel(kernel, 'end')

    # Ensure shape of image and kernel correct
    if image.ndim != coords.shape[1]:
        raise ValueError(f"Image has {image.ndim} dimensions but coords only have {coords.shape[1]} dimensions.")
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    elif image.ndim != 3:
        raise ValueError(f"image must have 2 or 3 dimensions but given image has {image.ndim}.")
    if kernel.ndim == 2:
        kernel = np.expand_dims(kernel, 2)
    elif kernel.ndim != 3:
        raise ValueError(f"kernel must have 2 or 3 dimensions but given image has {image.ndim}.")
    if kernel.max() > 1:
        raise ValueError(f'kernel is expected to be binary, only containing 0 or 1 but kernel.max = {kernel.max()}')

    if coords.shape[1] == 2:
        # set all z coordinates to 0 if 2D.
        coords = np.append(coords, np.zeros((coords.shape[0], 1), dtype=int), axis=1)
    if (coords.max(axis=0) >= np.array(image.shape)).any():
        raise ValueError(f"Max yxz coordinates provided are {coords.max(axis=0)} but image has shape {image.shape}.")

    pad_size = [(int((ax_size-1)/2),)*2 for ax_size in kernel.shape]
    pad_coords = jnp.asarray(coords) + jnp.array([val[0] for val in pad_size])
    if isinstance(padding, numbers.Number):
        image_pad = jnp.pad(jnp.asarray(image), pad_size, 'constant', constant_values=padding).astype(int)
    else:
        image_pad = jnp.pad(jnp.asarray(image), pad_size, padding).astype(int)
    y_shifts, x_shifts, z_shifts = get_shifts_from_kernel(jnp.asarray(np.flip(kernel)))
    return np.asarray(manual_convolve(image_pad, y_shifts, x_shifts, z_shifts, pad_coords))
