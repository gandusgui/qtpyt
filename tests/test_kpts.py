import numpy as np
from qtpyt.surface.kpts import fourier_sum


def test_fourier_sum():
    nk = 10
    n = 4

    A = np.random.random(nk * n * n).reshape(nk, n, n).astype(complex)
    A += np.transpose(A, (0, 2, 1))
    A /= 2.0

    kpts = np.random.random((nk, 3))
    R = np.random.randint(0, 10, nk * 3).reshape(nk, 3)
    Ak = fourier_sum(A, kpts, R, inverse=True)
    Ar = fourier_sum(Ak, kpts, R)

    np.testing.assert_allclose(A, Ar, atol=1e-13)
