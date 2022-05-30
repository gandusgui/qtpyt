import numpy as np
from qtpyt.block_tridiag._linalg_numba_cpu import build_offdiags


def test_build_offdiags():
    z = 1.0 + 0.025j
    m = 200
    n = 300
    mat_ij = np.zeros((m, n), complex)
    mat_ji = mat_ij.T.copy()
    hs_ij = np.zeros((2, m, n), complex)
    hs_ij[0] = np.ones(
        (m, n), complex
    )  # np.random.random((m,n)) + 1.j*np.random.random((m,n))
    hs_ij[1] = np.ones(
        (m, n), complex
    )  # np.random.random((m,n)) + 1.j*np.random.random((m,n))
    block_size = 32

    build_offdiags(z, mat_ij, mat_ji, hs_ij, block_size)

    expected_ij = z * hs_ij[1] - hs_ij[0]
    expected_ji = z * hs_ij[1].T.conj() - hs_ij[0].T.conj()

    np.testing.assert_allclose(expected_ij, mat_ij)
    np.testing.assert_allclose(expected_ji, mat_ji)


if __name__ == "__main__":
    test_build_offdiags()
