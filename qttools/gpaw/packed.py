import numpy as np

from gpaw.utilities.blas import rk
from gpaw.utilities.tools import tri2full

"""
This module unpacks a compressed matrix of pair orbitals.

Example:

Orbitals:
    ['a','b']

Pair Orbitals compressed (upper triangle):
    ['aa':0, 'ab':1, 'bb':2]

The following packed pairorbital array

    array([[(0, 0), (0, 1), (0, 2)],  *
           [(1, 0), (1, 1), (1, 2)],  **
           [(2, 0), (2, 1), (2, 2)]]) ***

is unpacked to

    array([[
        [(0, 0), (0, 1)],      [(1, 0), (1, 1)],   
    *   [(0, 1), (0, 2)],      [(1, 1), (1, 2)],  **

        [(1, 0), (1, 1)],      [(2, 0), (2, 1)],
   **   [(1, 1), (1, 2)],      [(2, 1), (2, 2)]   ***
    ]])
    
Each row is packed both in elements and in position.

"""


def unpack(D_cc, Nw, D_pp=None, UL="L"):
    """Unpack compressed matrix."""
    Nc = Nw * (Nw + 1) // 2
    Np = Nw ** 2
    # check inputs
    assert Nc == D_cc.shape[0], "Invalid combinations with replacement."
    if D_pp is not None:
        assert Np == D_pp.shape[0], "Invalid product."
    else:
        D_pp = np.empty((Np, Np), D_cc.dtype)
    # prepare inputs
    tri2full(D_cc, UL)
    D_pp = D_pp.reshape(Nw, Nw, Nw, Nw)
    # temporary work array
    i1, i2 = np.triu_indices(Nw)
    work = np.empty((Nw, Nw), D_cc.dtype)
    for r in np.ndindex(Nc):
        work[i1, i2] = D_cc[r]
        tri2full(work, UL="U")
        D_pp[i1[r], i2[r]] = D_pp[i2[r], i1[r]] = work
    return D_pp.reshape(Np, Np)


if __name__ == "__main__":
    Nw = 8
    Nc = Nw * (Nw + 1) // 2
    Np = Nw ** 2
    w = np.arange(Nw) + 1  # np.random.random(Nw)
    D_cc = np.zeros((Nc, Nc))
    actual_D_pp = np.zeros((Np, Np))
    desired_D_pp = np.zeros((Np, Np))
    # combo with repetitions
    f = np.zeros(Nc)
    p = 0
    for r in range(Nw):
        for c in range(r, Nw):
            f[p] = w[r] * w[c]
            p += 1
    rk(1.0, f[..., None], 0.0, D_cc)
    unpack(D_cc, Nw, actual_D_pp)
    # full product
    f = np.zeros(Np)
    p = 0
    for r in range(Nw):
        for c in range(Nw):
            f[p] = w[r] * w[c]
            p += 1
    rk(1.0, f[..., None], 0.0, desired_D_pp)
    tri2full(desired_D_pp, UL="L")
    # test
    np.testing.assert_allclose(actual_D_pp, desired_D_pp)
