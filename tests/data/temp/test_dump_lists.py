import numpy as np
import os
import pickle

os.system("dump_hs_lists.py ../poly2_scatt.gpw 36")

hs_ii, hs_ij = pickle.load(open("hs_poly2_scatt_lists.pckl", "rb"))
HS_ii, HS_ij = pickle.load(open("../poly2_hs_lists.pckl", "rb"))


def test_hs(hs, HS):
    H, S = HS
    h, s = hs
    for s0, S0 in zip(s, S):
        np.testing.assert_allclose(s0, S0)
    for h0, H0 in zip(h, H):
        np.testing.assert_allclose(h0, H0)


if __name__ == "__main__":
    test_hs(hs_ii, HS_ii)
    test_hs(hs_ij, HS_ij)
