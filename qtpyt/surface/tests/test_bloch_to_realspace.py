import numpy as np

from qtpyt.tightbinding import bloch_to_realspace
from qtpyt.surface.kpts import (
    build_lattice_vectors,
    expand_inversion_symm,
    fourier_sum,
    monkhorst_pack,
)


def test_2D_odd():
    h_full_k, s_full_k = np.load("2D/odd/no-sym/hs_pl_k.npy")
    h_k, s_k = np.load("2D/odd/hs_pl_k.npy")
    kpts = monkhorst_pack((3, 3, 1))
    R = build_lattice_vectors((3, 3, 1))
    # from IPython import embed; embed()
    expected = fourier_sum(h_full_k, kpts, R)
    _, h_expand_k = expand_inversion_symm(kpts, h_k)
    expected = fourier_sum(h_full_k, kpts, R)
    computed_expand = fourier_sum(h_expand_k, kpts, R)
    computed = bloch_to_realspace(h_k, kpts, R)
    np.testing.assert_allclose(expected, computed_expand, atol=1e-8)
    np.testing.assert_allclose(expected, computed, atol=1e-8)


def test_2D_even():
    h_full_k, s_full_k = np.load("2D/even/no-sym/hs_pl_k.npy")
    h_k, s_k = np.load("2D/even/hs_pl_k.npy")
    kpts = monkhorst_pack((3, 2, 1), offset=(0, 1 / 4, 0))
    R = build_lattice_vectors((3, 2, 1))
    # from IPython import embed; embed()
    expected = fourier_sum(h_full_k, kpts, R)
    _, h_expand_k = expand_inversion_symm(kpts, h_k)
    expected = fourier_sum(h_full_k, kpts, R)
    computed_expand = fourier_sum(h_expand_k, kpts, R)
    # computed = bloch_to_realspace(h_k, kpts, R)
    np.testing.assert_allclose(expected, computed_expand, atol=1e-8)
    # np.testing.assert_allclose(expected, computed, atol=1e-8)


if __name__ == "__main__":
    test_2D_odd()
    test_2D_even()
