import numpy as np
import scipy.signal
from scipy.ndimage.morphology import grey_dilation
from scipy.ndimage import convolve, correlate
import numbers
from . import errors
import cv2
from typing import Optional, Union
from scipy.signal import oaconvolve


def ftrans2(b: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Produces a 2D convolve kernel that corresponds to the 1D convolve kernel, `b`, using the transform, `t`.
    Copied from [MATLAB `ftrans2`](https://www.mathworks.com/help/images/ref/ftrans2.html).

    Args:
        b: `float [Q]`.
            1D convolve kernel.
        t: `float [M x N]`.
            Transform to make `b` a 2D convolve kernel.
            If `None`, McClellan transform used.

    Returns:
        `float [(M-1)*(Q-1)/2+1 x (N-1)*(Q-1)/2+1]`.
            2D convolve kernel.
    """
    if t is None:
        # McClellan transformation
        t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]]) / 8

    # Convert the 1-D convolve_2d b to SUM_n a(n) cos(wn) form
    n = int(round((len(b) - 1) / 2))
    b = b.reshape(-1, 1)
    b = np.rot90(np.fft.fftshift(np.rot90(b)))
    a = np.concatenate((b[:1], 2 * b[1:n + 1]))

    inset = np.floor((np.array(t.shape) - 1) / 2).astype(int)

    # Use Chebyshev polynomials to compute h
    p0 = 1
    p1 = t
    h = a[1] * p1
    rows = inset[0]
    cols = inset[1]
    h[rows, cols] += a[0] * p0
    for i in range(2, n + 1):
        p2 = 2 * scipy.signal.convolve2d(t, p1)
        rows = rows + inset[0]
        cols = cols + inset[1]
        p2[rows, cols] -= p0
        rows = inset[0] + np.arange(p1.shape[0])
        cols = (inset[1] + np.arange(p1.shape[1])).reshape(-1, 1)
        hh = h.copy()
        h = a[i] * p2
        h[rows, cols] += hh
        p0 = p1.copy()
        p1 = p2.copy()
    h = np.rot90(h)
    return h


def hanning_diff(r1: int, r2: int) -> np.ndarray:
    """
    Gets difference of two hanning window 2D convolve kernel.
    Central positive, outer negative with sum of `0`.

    Args:
        r1: radius in pixels of central positive hanning convolve kernel.
        r2: radius in pixels of outer negative hanning convolve kernel.

    Returns:
        `float [2*r2+1 x 2*r2+1]`.
            Difference of two hanning window 2D convolve kernel.
    """
    if not 0 <= r1 <= r2-1:
        raise errors.OutOfBoundsError("r1", r1, 0, r2-1)
    if not r1+1 <= r2 <= np.inf:
        raise errors.OutOfBoundsError("r2", r1+1, np.inf)
    h_outer = np.hanning(2 * r2 + 3)[1:-1]  # ignore zero values at first and last index
    h_outer = -h_outer / h_outer.sum()
    h_inner = np.hanning(2 * r1 + 3)[1:-1]
    h_inner = h_inner / h_inner.sum()
    h = h_outer.copy()
    h[r2 - r1:r2 + r1 + 1] += h_inner
    h = ftrans2(h)
    return h


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolves `image` with `kernel`, padding by replicating border pixels.

    Args:
        image: `float [image_sz1 x image_sz2]`.
            Image to convolve.
        kernel: `float [kernel_sz1 x kernel_sz2]`.
            2D kernel

    Returns:
        `float [image_sz1 x image_sz2]`.
            `image` after being convolved with `kernel`.

    !!! note
        `np.flip` is used to give same result as `convn` with replicate padding in MATLAB.
    """
    return cv2.filter2D(image.astype(float), -1, np.flip(kernel), borderType=cv2.BORDER_REPLICATE)


def ensure_odd_kernel(kernel: np.ndarray, pad_location: str = 'start') -> np.ndarray:
    """
    This ensures all dimensions of `kernel` are odd by padding even dimensions with zeros.
    Replicates MATLAB way of dealing with even kernels.

    Args:
        kernel: `float [kernel_sz1 x kernel_sz2 x ... x kernel_szN]`.
        pad_location: One of the following, indicating where to pad with zeros -

            - `'start'` - Zeros at start of kernel.
            - `'end'` - Zeros at end of kernel.

    Returns:
        `float [odd_kernel_sz1 x odd_kernel_sz2 x ... x odd_kernel_szN]`.
            `kernel` padded with zeros so each dimension is odd.

    Example:
        If `pad_location` is `'start'` then `[[5,4];[3,1]]` becomes `[[0,0,0],[0,5,4],[0,3,1]]`.
    """
    even_dims = (np.mod(kernel.shape, 2) == 0).astype(int)
    if max(even_dims) == 1:
        if pad_location == 'start':
            pad_dims = [tuple(np.array([1, 0]) * val) for val in even_dims]
        elif pad_location == 'end':
            pad_dims = [tuple(np.array([0, 1]) * val) for val in even_dims]
        else:
            raise ValueError(f"pad_location has to be either 'start' or 'end' but value given was {pad_location}.")
        return np.pad(kernel, pad_dims, mode='constant')
    else:
        return kernel


def top_hat(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Does tophat filtering of `image` with `kernel`.

    Args:
        image: `float [image_sz1 x image_sz2]`.
            Image to filter.
        kernel: `np.uint8 [kernel_sz1 x kernel_sz2]`.
            Top hat `kernel` containing only zeros or ones.

    Returns:
        `float [image_sz1 x image_sz2]`.
            `image` after being top hat filtered with `kernel`.
    """
    if kernel.dtype != np.uint8:
        if sum(np.unique(kernel) == [0, 1]) == len(np.unique(kernel)):
            kernel = kernel.astype(np.uint8)  # kernel must be uint8
        else:
            raise ValueError(f'kernel is of type {kernel.dtype} but must be of data type np.uint8.')
    image_dtype = image.dtype   # so returned image is of same dtype as input
    if image.dtype == int:
        if image.min() >= 0 and image.max() <= np.iinfo(np.uint16).max:
            image = image.astype(np.uint16)
    if not (image.dtype == float or image.dtype == np.uint16):
        raise ValueError(f'image is of type {image.dtype} but must be of data type np.uint16 or float.')
    if np.max(np.mod(kernel.shape, 2) == 0):
        # With even kernel, gives different results to MATLAB
        raise ValueError(f'kernel dimensions are {kernel.shape}. Require all dimensions to be odd.')
    # kernel = ensure_odd_kernel(kernel)  # doesn't work for tophat at start or end.
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel).astype(image_dtype)


def dilate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Dilates `image` with `kernel`, using zero padding.

    Args:
        image: `float [image_sz1 x image_sz2]`.
            Image to be dilated.
        kernel: `int [kernel_sz1 x kernel_sz2]`.
            Dilation kernel containing only zeros or ones.

    Returns:
        `float [image_sz1 x image_sz2]`.
            `image` after being dilated with `kernel`.
    """
    kernel = ensure_odd_kernel(kernel)
    # mode refers to the padding. We pad with zeros to keep results the same as MATLAB
    return grey_dilation(image, footprint=kernel, mode='constant')
    # return morphology.dilation(image, kernel)


def imfilter(image: np.ndarray, kernel: np.ndarray, padding: Union[float, str] = 0,
             corr_or_conv: str = 'corr') -> np.ndarray:
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
                This is the default when no option specified.
            - `'conv'` - Performs multidimensional filtering using convolution.

    Returns:
        `float [image_sz1 x image_sz2 x ... x image_szN]`.
            `image` after being filtered.
    """
    if corr_or_conv == 'corr':
        kernel = np.flip(kernel)
    elif corr_or_conv != 'conv':
        raise ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")
    kernel = ensure_odd_kernel(kernel, 'end')
    pad_size = [(int((ax_size-1)/2),)*2 for ax_size in kernel.shape]
    if isinstance(padding, numbers.Number):
        return oaconvolve(np.pad(image, pad_size, 'constant', constant_values=padding), kernel, 'valid')
    else:
        return oaconvolve(np.pad(image, pad_size, padding), kernel, 'valid')

    # Old method, about 3x slower for filtering large 3d image with small 3d kernel
    # if isinstance(padding, numbers.Number):
    #     pad_value = padding
    #     padding = 'constant'
    # else:
    #     pad_value = 0.0  # doesn't do anything for non-constant padding
    # if corr_or_conv == 'corr':
    #     kernel = ensure_odd_kernel(kernel, 'start')
    #     return correlate(image, kernel, mode=padding, cval=pad_value)
    # elif corr_or_conv == 'conv':
    #     kernel = ensure_odd_kernel(kernel, 'end')
    #     return convolve(image, kernel, mode=padding, cval=pad_value)
    # else:
    #     raise ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")
