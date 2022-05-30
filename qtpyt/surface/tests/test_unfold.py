import numpy as np

from qtpyt.surface.unfold import bloch_unfold


def test_1D():
    nr = nc = 2
    N = (1, 3, 1)
    nk = np.prod(N)
    A = np.broadcast_to(range(nk), (nr, nc, nk)).T.astype(complex)
    """
    array([[[0, 0],
            [0, 0]],

           [[1, 1],
            [1, 1]],

           [[2, 2],
            [2, 2]]])"""
    kpts = np.zeros((nk, 3))
    out = bloch_unfold(A * nk, kpts, N)
    np.testing.assert_allclose(out, 3 * np.ones((np.prod(N) * nr, np.prod(N) * nc)))


def test_2D():
    nr = nc = 2
    N = (1, 3, 4)
    nk = np.prod(N)
    A = np.broadcast_to(range(nk), (nr, nc, nk)).T.astype(complex)
    """
    array([[[0, 0],
            [0, 0]],

           [[2, 2],
            [2, 2]],

               :
               :
               :
            
           [[11, 11],
            [11, 11]])"""
    kpts = np.zeros((nk, 3))
    out = bloch_unfold(A * nk, kpts, N)
    np.testing.assert_allclose(out, 66 * np.ones((np.prod(N) * nr, np.prod(N) * nc)))


if __name__ == "__main__":
    test_1D()
    test_2D()
