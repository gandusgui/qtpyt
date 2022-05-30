import numpy as np

from ._unfold_numba import _bloch_unfold


def bloch_unfold(A, kpts, N, out=None):
    """Unfold matrices evaluated at k-points onto a supercell.

    The supercell (SC) consists of N repetitions of a smaller 
    building block described by the k-matrices. At the moment,
    unfolding is implemented only along two directions. This means
    that even though N is compsed of (3,) integers, only 2 of them
    can be greater than 1. Finally, note that the last dimension 
    != 1 is used in the innermost unfolding.
    
    """

    assert A.dtype == complex, "Input arrays must be complex128!"

    i, j, k = N
    nr, nc = A.shape[1:]

    # NOTE that for 1D expansion only 2nd column of kpts are used
    if i == 1:
        if k == 1:  # (1D) y-expand
            m, n = 1, j
            kpts = kpts[:, [0, 1]]
        else:  # (1D) z- or (2D) yz-expand
            m, n = j, k
            kpts = kpts[:, [1, 2]]
    else:  # i>1
        if j == 1:
            if k == 1:  # (1D) x-expand
                m, n = 1, i
                kpts = kpts[:, [0, 0]]  # Hack to swap x-coord to last position.
            else:  # (2D) xz-expand
                m, n = i, k
                kpts = kpts[:, [0, 2]]
        else:
            if k == 1:  # xy-expand
                m, n = i, j
                kpts = kpts[:, [0, 1]]
            else:
                raise NotImplementedError("Cannot expand 3D supercells.")

    mxn = m * n
    if out is None:
        out = np.zeros((mxn, nr, mxn, nc), complex)
    else:
        assert out.dtype == complex, "Output matrix must be complex128!"
        if out.ndim != 4:
            out.shape = (mxn, nr, mxn, nc)

    _bloch_unfold(A, kpts, m, n, out)

    out.shape = (mxn * nr, mxn * nc)
    return out
