import numbers
from typing import Union, Tuple
import numpy as np
from scipy.ndimage import correlate, convolve
from scipy.signal import oaconvolve
from .base import ensure_odd_kernel


def imfilter(image: np.ndarray, kernel: np.ndarray, padding: Union[float, str] = 0,
             corr_or_conv: str = 'corr', oa: bool = True) -> np.ndarray:
    """
    Copy of MATLAB `imfilter` function with `'output_size'` equal to `'same'`.

    Args:
        image: `float [image_sz1 x image_sz2 x ... x image_szN]`.
            Image to be filtered.
        kernel: `float [kernel_sz1 x kernel_sz2 x ... x kernel_szN]`.
            Multidimensional filter.
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
            - `'conv'` - Performs multidimensional filtering using convolution.
        oa: Whether to use oaconvolve or scipy.ndimage.convolve.
            scipy.ndimage.convolve seems to be quicker for smoothing in extract step (3s vs 20s for 50 z-planes).

    Returns:
        `float [image_sz1 x image_sz2 x ... x image_szN]`.
            `image` after being filtered.
    """
    if oa:
        if corr_or_conv == 'corr':
            kernel = np.flip(kernel)
        elif corr_or_conv != 'conv':
            raise ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")
        kernel = ensure_odd_kernel(kernel, 'end')
        if kernel.ndim < image.ndim:
            kernel = np.expand_dims(kernel, axis=tuple(np.arange(kernel.ndim, image.ndim)))
        pad_size = [(int((ax_size-1)/2),)*2 for ax_size in kernel.shape]
        if isinstance(padding, numbers.Number):
            return oaconvolve(np.pad(image, pad_size, 'constant', constant_values=padding), kernel, 'valid')
        else:
            return oaconvolve(np.pad(image, pad_size, padding), kernel, 'valid')
    else:
        if padding == 'symmetric':
            padding = 'reflect'
        elif padding == 'edge':
            padding = 'nearest'
        # Old method, about 3x slower for filtering large 3d image with small 3d kernel
        if isinstance(padding, numbers.Number):
            pad_value = padding
            padding = 'constant'
        else:
            pad_value = 0.0  # doesn't do anything for non-constant padding
        if corr_or_conv == 'corr':
            kernel = ensure_odd_kernel(kernel, 'start')
            return correlate(image, kernel, mode=padding, cval=pad_value)
        elif corr_or_conv == 'conv':
            kernel = ensure_odd_kernel(kernel, 'end')
            return convolve(image, kernel, mode=padding, cval=pad_value)
        else:
            raise ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")


def imfilter_coords(image: np.ndarray, kernel: np.ndarray, coords: np.ndarray, padding: Union[float, str] = 0,
                    corr_or_conv: str = 'corr') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Copy of MATLAB `imfilter` function with `'output_size'` equal to `'same'`.
    Only finds result of filtering at specific locations but still filters entire image.

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
            - `'conv'` - Performs multidimensional filtering using convolution.

    Returns:
        `int [n_points]`.
            Result of filtering of `image` at each point in `coords`.
    """
    im_filt = imfilter(image.astype(int), kernel, padding, corr_or_conv, oa=False)
    return im_filt[tuple([coords[:, j] for j in range(im_filt.ndim)])]
