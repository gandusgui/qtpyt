import numpy as np
from qtpyt.surface.unfold import bloch_unfold
from time import perf_counter

nr = nc = 100
N = (1, 11, 9)

nk = np.prod(N)
A = nk * np.broadcast_to(range(nk), (nr, nc, nk)).T.astype(complex)
kpts = np.zeros((nk, 3))
out = np.empty((np.prod(N) * nr, np.prod(N) * nc), complex)


def run(times):
    elapsed = 0.0
    for _ in range(times):
        s = perf_counter()
        _ = bloch_unfold(A, kpts, N, out)
        elapsed += perf_counter() - s
    # np.testing.assert_allclose(out, sum(range(11*9))*np.ones((np.prod(N)*nr,np.prod(N)*nc)))
    return elapsed / times


if __name__ == "__main__":
    print("InitTime: ", run(1))
    print("RunTime", run(3))
