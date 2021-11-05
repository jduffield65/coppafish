import numpy as np
import scipy.signal
import iss.utils.errors
import cv2


def get_pixel_length(length_microns, pixel_size):
    """
    Converts a length in units of microns into a length in units of pixels

    :param length_microns: float
        length in units of microns (microns)
    :param pixel_size: float
        size of a pixel in microns (microns/pixels)
    :return: integer, desired length in units of pixels (pixels)
    """
    return int(round(length_microns / pixel_size))


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


def hanning_diff(r1, r2):
    """
    gets difference of two hanning window filter
    (central positive, outer negative) with sum of 0.

    :param r1: integer
        radius in pixels of central positive hanning filter
    :param r2: integer, must be greater than r1
        radius in pixels of outer negative hanning filter
    :return: float numpy array [2*r2 + 1, 2*r2 + 1]
    """
    iss.utils.errors.out_of_bounds('r1', r1, 0, r2)
    iss.utils.errors.out_of_bounds('r2', r2, r1, np.inf)
    h_outer = np.hanning(2 * r2 + 3)[1:-1]  # ignore zero values at first and last index
    h_outer = -h_outer / h_outer.sum()
    h_inner = np.hanning(2 * r1 + 3)[1:-1]
    h_inner = h_inner / h_inner.sum()
    h = h_outer.copy()
    h[r2 - r1:r2 + r1 + 1] += h_inner
    h = ftrans2(h)
    return h


def periodic_line_strel(p, v):
    """
    creates a flat structuring element
    containing 2*p+1 members.  v is a two-element vector containing
    integer-valued row and column offsets.  One structuring element member
    is located at the origin.  The other members are located at 1*v, -1*v,
    2*v, -2*v, ..., p*v, -p*v.
    copy of MATLAB strel('periodicline')
    """
    pp = np.repeat(np.arange(-p, p + 1).reshape(-1, 1), 2, axis=1)
    rc = pp * v
    r = rc[:, 0]
    c = rc[:, 1]
    M = 2 * np.abs(r).max() + 1
    N = 2 * np.abs(c).max() + 1
    nhood = np.zeros((M, N), dtype=bool)
    # idx = np.ravel_multi_index([r + np.abs(r).max(), c + np.abs(c).max()], (M, N))
    nhood[r + np.abs(r).max(), c + np.abs(c).max()] = True
    return nhood.astype(np.uint8)


def disk_strel(r, n=4):
    """
    creates a flat disk-shaped structuring element
    with the specified radius, r.  r must be a nonnegative integer.  n must
    be 0, 4, 6, or 8.  When n is greater than 0, the disk-shaped structuring
    element is approximated by a sequence of n (or sometimes n+2)
    periodic-line structuring elements.  When n is 0, no approximation is
    used, and the structuring element members comprise all pixels whose
    centers are no greater than r away from the origin.  n can be omitted,
    in which case its default value is 4.  Note: Morphological operations
    using disk approximations (n>0) run much faster than when n=0.  Also,
    the structuring elements resulting from choosing n>0 are suitable for
    computing granulometries, which is not the case for n=0.  Sometimes it
    is necessary for STREL to use two extra line structuring elements in the
    approximation, in which case the number of decomposed structuring
    elements used is n+2.
    copy of MATLAB strel('disk')
    """
    if r < 3:
        # Radius is too small to use decomposition, so force n=0.
        n = 0
    if n == 0:
        xx, yy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        nhood = xx ** 2 + yy ** 2 <= r ** 2
    else:
        """
        Reference for radial decomposition of disks:  Rolf Adams, "Radial
        Decomposition of Discs and Spheres," CVGIP:  Graphical Models and
        Image Processing, vol. 55, no. 5, September 1993, pp. 325-332.
        
        The specific decomposition technique used here is radial
        decomposition using periodic lines.  The reference is:  Ronald
        Jones and Pierre Soille, "Periodic lines: Definition, cascades, and
        application to granulometries," Pattern Recognition Letters,
        vol. 17, 1996, pp. 1057-1063.
        
        Determine the set of "basis" vectors to be used for the
        decomposition.  The rows of v will be used as offset vectors for
        periodic line strels.
        """
        if n == 4:
            v = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]])
        elif n == 6:
            v = np.array([[1, 0], [1, 2], [2, 1], [0, 1], [-1, 2], [-2, 1]])
        elif n == 8:
            v = np.array([[1, 0], [2, 1], [1, 1], [1, 2], [0, 1], [-1, 2], [-1, 1], [-2, 1]])
        else:
            raise ValueError(f'Value of n provided ({n}) is not 0, 4, 6 or 8.')
        # Determine k, which is the desired radial extent of the periodic
        # line strels.  For the origin of this formula, see the second
        # paragraph on page 328 of the Rolf Adams paper.
        theta = np.pi / (2 * n)
        k = 2 * r / (1 / np.tan(theta) + 1 / np.sin(theta))

        # For each periodic line strel, determine the repetition parameter,
        # rp.  The use of floor() in the computation means that the resulting
        # strel will be a little small, but we will compensate for this
        # below.
        nhood = np.ones((2*r-1, 2*r-1), np.uint8) * -np.inf
        nhood[int((nhood.shape[0]-1)/2), int((nhood.shape[0]-1)/2)] = 1
        for q in range(n):
            rp = int(np.floor(k / np.linalg.norm(v[q, :])))
            decomposition = periodic_line_strel(rp, v[q, :])
            nhood = cv2.dilate(nhood, decomposition)
        nhood = nhood > 0

        # Now we are going to add additional vertical and horizontal line
        # strels to compensate for the fact that the strel resulting from the
        # above decomposition tends to be smaller than the desired size.
        extra_strel_size = int(sum(np.sum(nhood, axis=1) == 0) + 1)
        if extra_strel_size > 0:
            # Update the computed neighborhood to reflect the additional strels in
            # the decomposition.
            nhood = cv2.dilate(nhood.astype(np.uint8), np.ones((1, extra_strel_size), dtype=np.uint8))
            nhood = cv2.dilate(nhood, np.ones((extra_strel_size,  1), dtype=np.uint8))
            nhood = nhood > 0
    return nhood.astype(np.uint8)


def filter_imaging(image, kernel):
    """
    convolves image with kernel, padding by replicating border pixels
    np.flip is to give same as convn with replicate padding in MATLAB

    :param image: numpy array [image_sz1 x image_sz2]
    :param kernel: numpy float array
    :return: numpy float array [image_sz1 x image_sz2]
    """
    return cv2.filter2D(image.astype(float), -1, np.flip(kernel), borderType=cv2.BORDER_REPLICATE)


def filter_dapi(image, kernel):
    """
    does tophat filtering of image with kernel

    :param image: numpy float array [image_sz1 x image_sz2]
    :param kernel: numpy uint8 array
    :return: numpy float array [image_sz1 x image_sz2]
    """
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel.astype(np.uint8))
