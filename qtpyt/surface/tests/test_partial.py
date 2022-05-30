import numpy as np
from qtpyt.surface.kpts import (
    monkhorst_pack,
    apply_inversion_symm,
    build_partial_kpts,
    fourier_sum,
    partial_bloch_sum,
)


def common_setup(size, offset):
    kpts = monkhorst_pack(size, offset=offset)
    kpts_p, kpts_t = build_partial_kpts(size, offset=offset)
    ikpts = apply_inversion_symm(kpts)[0]
    return kpts, ikpts, kpts_p, kpts_t


def run_test_partial(h_k, h_full_k, kpts, ikpts, kpts_p, kpts_t):
    computed = partial_bloch_sum(h_k, ikpts, [0, 0, 0], kpts_p, kpts_t)
    expected = partial_bloch_sum(h_full_k, kpts, [0, 0, 0], kpts_p, kpts_t)
    np.testing.assert_allclose(computed, expected, atol=1e-9)


def test_2D_even_symm():
    kpts, ikpts, kpts_p, kpts_t = common_setup((3, 2, 1), (0.0, 0.25, 0.0))
    h_k = np.load("2D/even/hs_pl_k.npy")[0]
    h_full_k = np.load("2D/even/no-sym/hs_pl_k.npy")[0]
    run_test_partial(h_k, h_full_k, kpts, ikpts, kpts_p, kpts_t)


def test_2D_odd_symm():
    kpts, ikpts, kpts_p, kpts_t = common_setup((3, 3, 1), (0.0, 0.0, 0.0))
    h_k = np.load("2D/odd/hs_pl_k.npy")[0]
    h_full_k = np.load("2D/odd/no-sym/hs_pl_k.npy")[0]
    run_test_partial(h_k, h_full_k, kpts, ikpts, kpts_p, kpts_t)


def test_3D_even_symm():
    kpts, ikpts, kpts_p, kpts_t = common_setup((5, 4, 2), (0.0, 0.125, 0.25))
    h_k = np.load("3D/even/hs_pl_k.npy")[0]
    h_full_k = np.load("3D/even/no-sym/hs_pl_k.npy")[0]
    run_test_partial(h_k, h_full_k, kpts, ikpts, kpts_p, kpts_t)


def test_3D_even_nogamma_symm():
    kpts, ikpts, kpts_p, kpts_t = common_setup((5, 4, 2), (0.0, 0.0, 0.0))
    h_k = np.load("3D/even/no-gamma/hs_pl_k.npy")[0]
    h_full_k = np.load("3D/even/no-gamma/no-sym/hs_pl_k.npy")[0]
    run_test_partial(h_k, h_full_k, kpts, ikpts, kpts_p, kpts_t)


if __name__ == "__main__":
    test_2D_even_symm()
    test_2D_odd_symm()
    test_3D_even_symm()
    test_3D_even_nogamma_symm()
