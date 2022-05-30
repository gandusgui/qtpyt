import numpy as np
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.surface.expand import expand_couplings, seminf_expand, tritoeplz_expand


def test_expand_periodic():

    A = np.ones((2, 2)).astype(int)
    B = np.arange(4).reshape((2, 2)).astype(int)

    out = tritoeplz_expand(3, A, B)

    desired = [
        [1, 1, 0, 1, 0, 0],
        [1, 1, 2, 3, 0, 0],
        [0, 2, 1, 1, 0, 1],
        [1, 3, 1, 1, 2, 3],
        [0, 0, 0, 2, 1, 1],
        [0, 0, 1, 3, 1, 1],
    ]

    np.testing.assert_allclose(out, desired)


def test_expand_seminfinite():

    h_ii = np.random.random((2, 2))
    h_ii += h_ii.T
    s_ii = np.eye(2)

    h_ij = np.random.random((2, 2))
    s_ij = np.zeros((2, 2))

    L = LeadSelfEnergy((h_ii, s_ii), (h_ij, s_ij))
    R = LeadSelfEnergy((h_ii, s_ii), (h_ij, s_ij), id="right")

    sigL = L.retarded(0.0)
    sigR = R.retarded(0.0)
    z = L.eta * 1.0j + 0.0

    out = seminf_expand(
        3, (L.h_ii, L.s_ii), (L.h_ij, L.s_ij), sigL, sigR, z, (R.h_ij, R.s_ij)
    )

    H = tritoeplz_expand(3, h_ii, h_ij)
    S = tritoeplz_expand(3, s_ii, s_ij)

    Ginv = z * S - H
    Ginv[:2, :2] -= sigL
    Ginv[-2:, -2:] -= sigR

    desired = np.linalg.inv(Ginv)

    np.testing.assert_allclose(out, desired)


if __name__ == "__main__":
    test_expand_periodic()
    test_expand_seminfinite()
