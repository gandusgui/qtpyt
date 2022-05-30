import numpy as np
from numba import vectorize
from time import perf_counter

from _linalg_numba_cpu import build_offdiags

z = 1.0 + 0.025j
m = 500
n = 600
mat_ij = np.zeros((m, n), complex)
mat_ji = mat_ij.T.copy()
hs_ij = np.zeros((2, m, n), complex)
hs_ij[0] = np.ones((m, n), complex)
hs_ij[1] = np.ones((m, n), complex)
block_size = 15


def time_build_offdiags():

    elapsed = 0.0
    for _ in range(4):
        s = perf_counter()
        _ = build_offdiags(z, mat_ij, mat_ji, hs_ij, block_size)
        elapsed += perf_counter() - s
    elapsed = elapsed / 4

    print("Parallel", elapsed)


if __name__ == "__main__":
    time_build_offdiags()
