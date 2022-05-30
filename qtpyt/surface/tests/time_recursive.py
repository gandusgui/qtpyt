import numpy as np
from time import perf_counter

from ase.io import read

from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.basis import Basis
from qtpyt.surface._recursive import get_G

hs_pl_k = np.load("3D/even/hs_pl_k.npy")
pl_basis = Basis.from_dictionary(read("3D/even/pl.xyz"), {"Au": 9, "C": 13})
kpts_t, h_kii, s_kii, h_kij, s_kij = prepare_leads_matrices(
    pl_basis, hs_pl_k[0], hs_pl_k[1], (5, 4, 2), (0.0, 0.125, 0.25)
)
g_kii = np.empty_like(h_kii)


def run(times):
    elapsed = 0.0
    for _ in range(times):
        s = perf_counter()
        get_G(g_kii, h_kii, s_kii, h_kij, s_kij, energy=0.0, eta=1e-5)
        elapsed += perf_counter() - s
    # np.testing.assert_allclose(out, sum(range(11*9))*np.ones((np.prod(N)*nr,np.prod(N)*nc)))
    return elapsed / times


if __name__ == "__main__":
    print("InitTime: ", run(1))
    print("RunTime", run(5))
