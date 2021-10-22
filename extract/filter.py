import numpy as np
import scipy.signal
import utils.errors


def ftrans2(b, t=None):
    """
    Produces a 2D filter that corresponds to the 1D filter b, using the transform t
    Copied from MATLAB ftrans2: https://www.mathworks.com/help/images/ref/ftrans2.html

    :param b: float numpy array [Q,]
    :param t: float numpy array [M, N], optional.
        default: McClellan transform
    :return: float numpy array [(M-1)*(Q-1)/2+1, (N-1)*(Q-1)/2+1]
    """
    if t is None:
        # McClellan transformation
        t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]]) / 8

    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
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
    for i in range(2, n+1):
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


def get_filter(r1, r2):
    """
    gets difference of two hanning window filter
    (central positive, outer negative) with sum of 0.

    :param r1: integer
        radius in pixels of central positive hanning filter
    :param r2: integer, must be greater than r1
        radius in pixels of outer negative hanning filter
    :return: float numpy array [2*r2 + 1, 2*r2 + 1]
    """
    utils.errors.out_of_bounds('r1', r1, 0, r2)
    utils.errors.out_of_bounds('r2', r2, r1, np.inf)
    h_outer = np.hanning(2 * r2 + 3)[1:-1]  # ignore zero values at first and last index
    h_outer = -h_outer / h_outer.sum()
    h_inner = np.hanning(2 * r1 + 3)[1:-1]
    h_inner = h_inner / h_inner.sum()
    h = h_outer.copy()
    h[r2 - r1:r2 + r1 + 1] += h_inner
    h = ftrans2(h)
    return h
