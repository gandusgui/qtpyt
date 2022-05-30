import numpy as np
from numba import njit, prange, vectorize


@njit("(c16,c16[:,:],c16[:,:,:])", cache=True, parallel=True, fastmath=True)
def build_diag(z, mat_ii, hs_ii):
    """Implemets

        mat_ii = z*S_ii - H_ii
    """
    for i in prange(hs_ii.shape[0]):
        for j in range(hs_ii.shape[0]):
            mat_ii[i, j] = z * hs_ii[1, i, j] - hs_ii[0, i, j]


@njit("(c16,c16[:,:],c16[:,:,:],c16[:,:])", cache=True, parallel=True, fastmath=True)
def build_diag_sigma(z, mat_ii, hs_ii, sigma):
    """Implemets

        mat_ii = z*S_ii - H_ii - Sigma_ii
    """
    for i in prange(hs_ii.shape[0]):
        for j in range(hs_ii.shape[0]):
            mat_ii[i, j] = z * hs_ii[1, i, j] - hs_ii[0, i, j] - sigma[i, j]


@njit("(c16,c16[:,:],c16[:,:],c16[:,:,:],i8)", cache=True, parallel=True, fastmath=True)
def build_offdiags(z, mat_ij, mat_ji, hs_ij, block_size):
    """Implemets

        mat_ij = z*S_ij - H_ij
        mat_ji = z*S_ij.T.conj()  H_ij.T.conj(),

    in parallel, for sub-blocks. This helps reuse of data.
    """
    m = mat_ij.shape[0] // block_size
    n = mat_ij.shape[1] // block_size
    rest_m = mat_ij.shape[0] % block_size
    rest_n = mat_ij.shape[1] % block_size

    for I in prange(m):
        C = I * block_size
        for J in range(n):
            D = J * block_size
            for i in range(block_size):
                A = C + i
                for j in range(block_size):
                    B = D + j
                    mat_ij[A, B] = z * hs_ij[1, A, B] - hs_ij[0, A, B]
                    mat_ji[B, A] = (
                        z * hs_ij[1, A, B].conjugate() - hs_ij[0, A, B].conjugate()
                    )
        D = n * block_size
        for i in range(block_size):
            A = C + i
            for j in range(rest_n):
                B = D + j
                mat_ij[A, B] = z * hs_ij[1, A, B] - hs_ij[0, A, B]
                mat_ji[B, A] = (
                    z * hs_ij[1, A, B].conjugate() - hs_ij[0, A, B].conjugate()
                )

    C = m * block_size
    for J in prange(n):
        D = J * block_size
        for i in range(rest_m):
            A = C + i
            for j in range(block_size):
                B = D + j
                mat_ij[A, B] = z * hs_ij[1, A, B] - hs_ij[0, A, B]
                mat_ji[B, A] = (
                    z * hs_ij[1, A, B].conjugate() - hs_ij[0, A, B].conjugate()
                )
    D = n * block_size
    for i in range(rest_m):
        A = C + i
        for j in range(rest_n):
            B = D + j
            mat_ij[A, B] = z * hs_ij[1, A, B] - hs_ij[0, A, B]
            mat_ji[B, A] = z * hs_ij[1, A, B].conjugate() - hs_ij[0, A, B].conjugate()
