import cython
import numpy as np
cimport numpy as cnp
cimport scipy.linalg.cython_blas
cimport scipy.linalg.cython_lapack
from scipy.linalg.lapack import dgels
from scipy.linalg.blas import dgemv

ctypedef cnp.float64_t REAL_t
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

# Useful example function in Cython similar to mat_mul
# https://stackoverflow.com/questions/44710838/calling-blas-lapack-directly-using-the-scipy-interface-and-cython
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef mat_mul(_A, _x, _output, REAL_t beta = 0, char *transpose = 'n'):
    """
    Updates `_output` to the new value `_output = beta * _output + _A @ _x`.
    If `transpose == 't'`, then `_output = beta * _output + _A.transpose() @ _x`.
    Not much quicker than scipy-dgemv.
    
    Args:
        _A: `float [M x N]`.
        _x: `float [N]` if transpose == 'n'.
            `float [M]` if transpose == 't'.
        _output: `float [M]` if transpose == 'n'.
                 `float [N]` if transpose == 't'.
        beta: float
        transpose: 't' to transpose _A first.
    """
    # LAPACK-BLAS function and variables explained here:
    # http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html

    cdef int M = _A.shape[0]
    cdef int N = _A.shape[1]
    cdef REAL_t *A = <REAL_t *> (cnp.PyArray_DATA(_A))
    cdef REAL_t *x = <REAL_t *> (cnp.PyArray_DATA(_x))
    cdef REAL_t *output = <REAL_t *> (cnp.PyArray_DATA(_output))
    with nogil:
        scipy.linalg.cython_blas.dgemv(transpose, &M, &N, &ONEF, A, &M, x, &ONE, &beta, output, &ONE)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef least_squares(_A, _B):
    """
    Finds x such that B - A*x is minimised.
    This is slower than non-cython scipy-dgels.
    
    Args:
        _A: `float [M x N]`.
        _B: `float [M x 1]` if transpose == 'n'.
                 `float [N]` if transpose == 't'.

    Returns:
        x: `float [N]`.
    """
    # LAPACK-least squares function:
    # https://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_ga94bd4a63a6dacf523e25ff617719f752.html
    cdef char *transpose = 't'
    # Have to copy so don't overwrite data in A and B.
    _A = _A.copy().transpose()
    _B = _B.copy()
    cdef int M = _A.shape[0]
    cdef int N = _A.shape[1]
    cdef REAL_t *A = <REAL_t *> (cnp.PyArray_DATA(_A))
    cdef REAL_t *B = <REAL_t *> (cnp.PyArray_DATA(_B))
    cdef int RCOND = -1
    cdef int LDB = max(M, N)
    cdef int LWORK = min(M, N) * 2
    cdef int INFO = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _WORK = np.zeros(LWORK)
    cdef REAL_t *WORK = <REAL_t *> (cnp.PyArray_DATA(_WORK))
    with nogil:
        scipy.linalg.cython_lapack.dgels(transpose, &M, &N, &ONE, A, &M, B, &LDB, WORK, &LWORK, &INFO)
    return _B[:M, 0]


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_fit_coefs(cnp.ndarray[double, ndim=2] bled_codes, cnp.ndarray[double, ndim=2] pixel_colors,
                 cnp.ndarray[Py_ssize_t, ndim=2] genes):
    """
    This finds the least squared solution for how `n_fit_genes` `bled_codes` can best explain each `pixel_color`.

    Args:
        bled_codes: `float [(n_rounds x n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds x n_channels) x n_pixels]`.
            Flattened then transposed pixel colors which usually has the shape `[n_genes x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_fit_genes]`.
            `genes[s]` indicates which `n_fit_genes` `bled_codes` to fit to `pixel_colors[:, s]`.

    Returns:
        - residual - `float [(n_rounds x n_channels) x n_pixels]`.
            Residual pixel_colors are removing bled_codes with coefficients specified by coef.
        - coef - `float [n_pixels x n_fit_genes]`.
            coefficient found through least squares fitting for each gene.
    """
    cdef Py_ssize_t s, g
    cdef Py_ssize_t n_pixels = pixel_colors.shape[1]
    cdef int n_fit_genes = genes.shape[1]
    cdef int n_round_channels = pixel_colors.shape[0]
    cdef cnp.ndarray[double, ndim=2] coefs = np.zeros((n_pixels, n_fit_genes))
    cdef cnp.ndarray[double, ndim=2] residual = pixel_colors.copy()
    cdef cnp.ndarray[double, ndim=1] mat_mul_res = np.zeros(pixel_colors.shape[0])

    for s in range(n_pixels):
        # TODO: make cython version of dgels.
        s_bled_codes = bled_codes[:, genes[s]]
        coefs[s] = dgels(s_bled_codes, pixel_colors[:, s])[1][:n_fit_genes]
        #coefs[s] = least_squares(bled_codes[:, genes[s]].transpose(), pixel_colors[:, s:s+1])
        #residual[:, s] = pixel_colors[:, s] - dgemv(1, bled_codes[:, genes[s]], coefs[s])
        mat_mul(s_bled_codes, coefs[s], mat_mul_res)
        residual[:, s] -= mat_mul_res
    return residual, coefs
