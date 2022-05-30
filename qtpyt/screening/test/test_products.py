import numpy as np
from qtpyt.screening.products import (
    exchange_product,
    hartree_product,
    polarization_product,
)


def _get_random(n, m, dtype=float):
    x = np.random.random((n, m))
    if dtype is complex:
        x = x + 1.0j * np.random.random((n, m))
    return x


n = 7
p, q = n ** 2, 13
U_pq = _get_random(p, q).astype(complex)
U_qp = U_pq.T.conj()
V_qq = _get_random(q, q).astype(complex)
V_pp = U_pq.dot(V_qq).dot(U_qp)
g = _get_random(n, n, complex)
gd = _get_random(n, n, complex)


def test_kron_pol():
    desired = np.kron(g, gd)
    actual = polarization_product(g, gd)
    np.testing.assert_allclose(desired, actual)


def test_pol_prod():
    desired = U_qp.dot(polarization_product(g, gd)).dot(U_pq)
    actual = polarization_product(g, gd, rot=U_pq)
    np.testing.assert_allclose(desired, actual)


def test_tensordot_exch():
    desired = np.tensordot(g, V_pp.reshape(n, n, n, n), axes=([0, 1], [1, 3]))
    actual = exchange_product(g, V_pp)
    np.testing.assert_allclose(desired, actual)


def test_exch_prod():
    desired = exchange_product(g, V_pp)
    actual = exchange_product(g, V_qq, rot=U_pq)
    np.testing.assert_allclose(desired, actual)


def test_hartree_prod():
    desired = hartree_product(g, V_pp)
    actual = hartree_product(g, V_qq, rot=U_pq)
    np.testing.assert_allclose(desired, actual)


if __name__ == "__main__":
    test_kron_pol()
    test_pol_prod()
    test_tensordot_exch()
    test_exch_prod()
    test_hartree_prod()
