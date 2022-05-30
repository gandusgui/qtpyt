import numpy as np
import numba
from numba import prange


@numba.njit(parallel=True, fastmath=True)
def _geamT(z, h, s, m):
    for i in prange(h.shape[0]):
        for j in range(h.shape[1]):
            m[j, i] = z * s[i, j] - h[i, j]


@numba.njit(parallel=True, fastmath=True)
def _geamC(z, h, s, m):
    for i in prange(h.shape[0]):
        for j in range(h.shape[1]):
            m[i, j] = z * s[i, j].conjugate() - h[i, j].conjugate()


@numba.njit(parallel=False, fastmath=True)
def _get_lambda(sigma, lambd):
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            lambd[j, i] = 1.0j * (sigma[i, j] - sigma[j, i].conjugate())


def geamT(z, h, s, m=None):
    assert h.flags.c_contiguous & s.flags.c_contiguous
    if m is not None:
        assert m.flags.f_contiguous
    else:
        m = np.empty(h.shape, complex, order="F")
    _geamT(z, h, s, m)
    return m


def geamC(z, h, s, m=None):
    assert h.flags.c_contiguous & s.flags.c_contiguous
    if m is not None:
        assert m.flags.f_contiguous
    else:
        m = np.empty(h.shape, complex, order="F")
    _geamC(z, h, s, m)
    return m


def get_lambda(sigma, lambd=None):
    assert sigma.flags.c_contiguous
    if lambd is not None:
        assert lambd.flags.f_contiguous
    else:
        lambd = np.empty(sigma.shape, complex, order="F")
    _get_lambda(sigma, lambd)
    return lambd


if __name__ == "__main__":
    z = complex(2.0)

    h = np.random.random((900, 900)).astype(complex)
    s = np.random.random((900, 900)).astype(complex)
    m = np.empty((900, 900), complex, order="F")

    geamT(z, h, s, m)
    expected = z * s.T - h.T
    assert np.allclose(expected, m)

    geamC(z, h, s, m)
    expected = z * s.conj() - h.conj()
    assert np.allclose(expected, m)

    sigma = np.random.random((900, 900)).astype(complex)
    lambd = np.empty((900, 900), complex, order="F")

    get_lambda(sigma, lambd)
    expected = 1.0j * (sigma - sigma.T.conj()).T
    assert np.allclose(expected, lambd)
