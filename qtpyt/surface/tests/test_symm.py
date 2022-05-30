import numpy as np
from qtpyt.surface.kpts import monkhorst_pack, apply_inversion_symm


def run_test_symm(h_k, ik2k, time_reverse, h_full_k):
    h_expand_k = h_k[ik2k]
    h_expand_k[time_reverse] = np.transpose(h_expand_k[time_reverse], axes=(0, 2, 1))
    np.testing.assert_allclose(h_full_k, h_expand_k, atol=1e-8)


def test_2D_even_symm():
    kpts = monkhorst_pack((3, 2, 1), offset=(0.0, 0.25, 0.0))
    _, _, ik2k, time_reverse = apply_inversion_symm(kpts)
    h_k = np.load("2D/even/hs_pl_k.npy")[0]
    h_full_k = np.load("2D/even/no-sym/hs_pl_k.npy")[0]
    run_test_symm(h_k, ik2k, time_reverse, h_full_k)


def test_2D_odd_symm():
    kpts = monkhorst_pack((3, 3, 1), offset=(0.0, 0.0, 0.0))
    _, _, ik2k, time_reverse = apply_inversion_symm(kpts)
    h_k = np.load("2D/odd/hs_pl_k.npy")[0]
    h_full_k = np.load("2D/odd/no-sym/hs_pl_k.npy")[0]
    run_test_symm(h_k, ik2k, time_reverse, h_full_k)


def test_3D_even_symm():
    kpts = monkhorst_pack((5, 4, 2), offset=(0.0, 0.125, 0.25))
    _, _, ik2k, time_reverse = apply_inversion_symm(kpts)
    h_k = np.load("3D/even/hs_pl_k.npy")[0]
    h_full_k = np.load("3D/even/no-sym/hs_pl_k.npy")[0]
    run_test_symm(h_k, ik2k, time_reverse, h_full_k)


def test_3D_even_nogamma_symm():
    kpts = monkhorst_pack((5, 4, 2), offset=(0.0, 0.0, 0.0))
    _, _, ik2k, time_reverse = apply_inversion_symm(kpts)
    h_k = np.load("3D/even/no-gamma/hs_pl_k.npy")[0]
    h_full_k = np.load("3D/even/no-gamma/no-sym/hs_pl_k.npy")[0]
    run_test_symm(h_k, ik2k, time_reverse, h_full_k)


if __name__ == "__main__":
    test_2D_even_symm()
    test_2D_odd_symm()
    test_3D_even_symm()
    test_3D_even_nogamma_symm()
