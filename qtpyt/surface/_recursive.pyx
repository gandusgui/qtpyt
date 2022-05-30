# cython: infer_types=True
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from scipy import linalg as la
from cython.parallel import prange, parallel
from scipy.linalg.cython_lapack cimport zgetrf, zgetrs, zgetri, zlange, zlacpy, zgesv
from scipy.linalg.cython_blas cimport zgemm, zaxpy, zcopy


cdef double conv = 1e-8


ctypedef fused double_or_complex:
    complex
    double


cdef inline complex conj(complex a) nogil:
    return a.real - a.imag*1j


cdef inline void init_vecs(complex* h_ii, complex* s_ii,
                           complex[:,:] h_ij, complex[:,:] s_ij,
                           complex* v_00, complex* v_01,
                           complex* v_10, complex* v_11,
                           complex z, int m) nogil:

    cdef int i, j

    for i in prange(m, schedule='static'):
        for j in range(m):
            v_00[i*m+j] = z * conj(s_ii[i*m+j]) - conj(h_ii[i*m+j])
            v_11[i*m+j] = v_00[i*m+j]
            v_01[i*m+j] = z * conj(s_ij[i,j]) - conj(h_ij[i,j])

    for i in prange(m, schedule='static'):
        for j in range(m):
            v_10[i*m+j] = z * s_ij[j,i] - h_ij[j,i]


cdef void get_Ginv(complex* v_00, complex* v_01,
                   complex* v_10, complex* v_11,
                   complex[::1,:] a, complex[::1,:] b,
                   complex[::1,:] lu, complex[::1,:] v_01_dot_b,
                   int[:] ipiv, int m) nogil:
    # """The inverse of the retarded surface Green function"""

    cdef int i, j, info, inc = 1
    cdef int m2 = m*m
    cdef char trans = b'N', norm = b'M', UPLO = b'A'
    cdef complex alpha = 1.+0.j, beta = -1.+0.j, gamma = 0.+0.j

    cdef double delta = conv + 1
    cdef double work[0]

    while delta > conv:

        zlacpy(&UPLO,&m,&m,&v_11[0],&m,&lu[0,0],&m)
        zgetrf(&m,&m,&lu[0,0],&m,&ipiv[0],&info)
        zlacpy(&UPLO,&m,&m,&v_01[0],&m,&a[0,0],&m)
        zgetrs(&trans,&m,&m,&lu[0,0],&m,&ipiv[0],&a[0,0],&m,&info)
        zlacpy(&UPLO,&m,&m,&v_10[0],&m,&b[0,0],&m)
        zgetrs(&trans,&m,&m,&lu[0,0],&m,&ipiv[0],&b[0,0],&m,&info)
        zgemm(&trans,&trans,&m,&m,&m,&alpha,&v_01[0],&m,&b[0,0],
              &m,&gamma,&v_01_dot_b[0,0],&m)
        zaxpy(&m2,&beta,&v_01_dot_b[0,0],&inc,&v_00[0],&inc)
        zgemm(&trans,&trans,&m,&m,&m,&beta,&v_10[0],&m,&a[0,0],
              &m,&alpha,&v_11[0],&m)
        zaxpy(&m2,&beta,&v_01_dot_b[0,0],&inc,&v_11[0],&inc)
        zlacpy(&UPLO,&m,&m,&v_01[0],&m,&lu[0,0],&m)
        zgemm(&trans,&trans,&m,&m,&m,&beta,&lu[0,0],&m,&a[0,0],
              &m,&gamma,&v_01[0],&m)
        zlacpy(&UPLO,&m,&m,&v_10[0],&m,&lu[0,0],&m)
        zgemm(&trans,&trans,&m,&m,&m,&beta,&lu[0,0],&m,&b[0,0],
              &m,&gamma,&v_10[0],&m)
        delta = zlange(&norm,&m,&m,&v_01[0],&m,&work[0])

cdef inline void inv(complex* a, complex[::1] work, int[:] ipiv, int m) nogil:

    cdef int info, lwork = work.shape[0]

    zgetrf(&m,&m,&a[0],&m,&ipiv[0],&info)
    zgetri(&m,&a[0],&m,&ipiv[0],&work[0],&lwork,&info)

# cdef void get_Sigma(complex* Ginv, complex* v_01, complex* v_10, 
#                     int[:] ipiv, int m) nogil:

#     cdef int info
#     cdef complex alpha = 1.+0.j, beta = 0.+0.j
#     cdef char trans = b'N'
    
#     zgesv(&m,&m,&Ginv[0],&m,&ipiv[0],&v_10[0],&m,&info)
#     zgemm(&trans,&trans,&m,&m,&m,&alpha,&v_01[0],&m,&v_10[0],
#           &m,&beta,&Ginv[0],&m)

def get_G(G_kMM, H_kii, S_kii, H_kij, S_kij, double_or_complex energy, 
          double eta=1e-5, double bias=0., solve_sigma=False):

    nkpts = H_kii.shape[0]
    m = G_kMM.shape[1]
    n = G_kMM.shape[2]
    lwork = 32000

    # v_xx = z * OP(h_xx) - OP(s_xx)
    # OP :: .T or .H
    v_00 = np.zeros((m,n), complex, order='F')
    cdef complex[:,:] _v_00 = v_00
    v_01 = np.zeros((m,n), complex, order='F')
    cdef complex[:,:] _v_01 = v_01
    v_10 = np.zeros((m,n), complex, order='F')
    cdef complex[:,:] _v_10 = v_10
    v_11 = np.zeros((m,n), complex, order='F')
    cdef complex[:,:] _v_11 = v_11
    # if solve_sigma:
    # m_01 = np.zeros((m,n), complex, order='F')
    # cdef complex[:,:] _m_01 = m_01
    # m_10 = np.zeros((m,n), complex, order='F')
    # cdef complex[:,:] _m_10 = m_10
    # cdef int incx = 1, incy = 1, _m = m
        
    # Loop internal vars.
    a = np.zeros((m,n), complex, order='F')
    b = np.zeros((m,n), complex, order='F')
    lu = np.zeros((m,n), complex, order='F')
    v_01_dot_b = np.zeros((m,n), complex, order='F')

    # Lapack internal vars.
    ipiv = np.zeros(m, np.int32)
    work = np.zeros(lwork, complex)

    # Green's function at transverse k-points
    cdef complex[:,:] h_ii
    cdef complex[:,:] s_ii
    cdef complex[:,:] h_ij
    cdef complex[:,:] s_ij
    cdef complex z
    z = energy - bias + eta * 1.j
    for i in range(nkpts):
        h_ii = H_kii[i]
        s_ii = S_kii[i]
        h_ij = H_kij[i]
        s_ij = S_kij[i]
        init_vecs(&h_ii[0,0],&s_ii[0,0],H_kij[i],S_kij[i],
                  &_v_00[0,0],&_v_01[0,0],&_v_10[0,0],&_v_11[0,0],
                  z,m)
        # if solve_sigma:
        #     zcopy(&_m,&_v_10[0,0],&incx,&_m_10[0,0],&incy)
        #     zcopy(&_m,&_v_01[0,0],&incx,&_m_01[0,0],&incy)
        #     get_Ginv(&_v_00[0,0],&_v_01[0,0],&_v_10[0,0],&_v_11[0,0],
        #             a,b,lu,v_01_dot_b,ipiv,m)
        #     get_Sigma(&_v_00[0,0],&_m_01[0,0],&_m_10[0,0],ipiv,m)
        # else:
        get_Ginv(&_v_00[0,0],&_v_01[0,0],&_v_10[0,0],&_v_11[0,0],
                a,b,lu,v_01_dot_b,ipiv,m)
        inv(&_v_00[0,0],work,ipiv,m)
        G_kMM[i] = v_00
