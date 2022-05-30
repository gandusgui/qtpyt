from functools import singledispatch
from typing import Any

import numpy as np

from qtpyt import xp
from qtpyt.base.greenfunction import BaseGreenFunction, GreenFunction
from qtpyt.block_tridiag.btmatrix import BTMatrix, _BTBuffer


class ProjectedGreenFunction(BaseGreenFunction):
    """Projected green's function onto an orthogonal space."""

    def __init__(self, gf: GreenFunction, indices) -> None:
        self.parent = gf
        self.projector = Projector(gf.S, indices)
        H = self.projector.cut(gf.H)
        super().__init__(H, self.projector.S, gf.eta, gf.mu, gf.kt)
        self.energy = None

    def __setattr__(self, name: str, value: Any) -> Any:
        if name in ["eta", "kt", "mu"]:
            setattr(self.parent, name, value)
        super().__setattr__(name, value)

    def retarded(self, energy):
        if self.energy != energy:
            self.energy = energy
            Gr = self.parent.retarded(energy)
            self.Gr = self.projector.project(Gr)
        return self.Gr


class CutGreenFunction(ProjectedGreenFunction):
    def __init__(self, gf, indices) -> None:
        super().__init__(gf, indices)
        self.projector.project = self.projector.cut


class Projector:
    """Projector to an orthogonal subspace.
    
    Can handle both block tridiagonal and dense parent spaces.
    
    NOTE : For block tridiagonal. `indices` can be any of the following:
            1. q : block index.
            2. (q,indices) : block index and relative relative from block.
            3. indices : gloabal indices from parent.
    """

    def __new__(self, S, indices):
        if isinstance(S, xp.ndarray):
            return BaseProjector(S, indices)

        else:
            # Preprocess input
            if isinstance(indices, int):
                q = indices
                indx = None
            elif len(indices) == 2 and hasattr(indices[1], "__len__"):
                q = indices[0]
                indx = indices[1]
            else:
                nodes = S._nodes
                # basis function at node belongs to right block
                q = np.searchsorted(nodes[1:], indices[0], side="right")
                indx = indices - nodes[q]
            return BTProjector(S, q, indx)


class BaseProjector:
    """Base projector. 
    
    Matrices of parent space are full arrays.
    
    Args:
        greenfunction : parent greenfunction (or Projector!!)
        indices : indices of projection
            
    """

    def __init__(self, S, indices):
        self._S_pp = S
        self.indices = indices
        self.S = self.cut(self._S_pp)

    def project(self, X):
        return project(X, self._S_pp, self.indices)

    def expand(self, X):
        return expand(X, self._S_pp, self.indices)

    def cut(self, X):
        return X[np.ix_(self.indices, self.indices)]


class BTProjector:
    """Block tridiagonal project. 
    
    Matrices of parent space are block tridiagonal.

    Args:
        q : block index.
        indices : subindices relative to block `q`.
        
    """

    def __init__(self, S, q, indices=None):
        self._S_pp = S
        self.q = q
        self.indices = indices
        self.S = self.cut(self._S_pp)

    def project(self, X):
        return project(X, self._S_pp, self.q, self.indices)

    def expand(self, X):
        pass
        # return BTMatrix(...)

    def cut(self, X):
        X = X[self.q, self.q]
        if self.indices is not None:
            X = X[np.ix_(self.indices, self.indices)]
        return X


def rotate_matrix(M, U):
    return U.T.conj().dot(M.dot(U))


@singledispatch
def project(A, S, q):
    """Project A onto subspace q orthogonal to the rest.
    Args : 
        A : (xp.ndarray)
        S : (xp.ndarray) overlap matrix
        q : indices of subspace
    """
    return rotate_matrix(rotate_matrix(A, S[:, q]), xp.linalg.inv(S[xp.ix_(q, q)]))


@project.register(BTMatrix)
def _(A, S, q, indices=None):
    """Project A onto subspace q orthogonal to the rest.
    Args : 
        A : (BTMatrix)
        S : (BTMatrix) overlap matrix
        q : index of subspace's block
        indices : (optional) subspace indices within block q.
    """
    # [(D_21*H_11 + D_22*H_21)*S_12
    # + (D_22*H_23 + D_23*H_33)*S_32
    # + (D_21*H_12 + D_22*H_22 + D_23*H_32)*S_22]
    D = S[q, q].T.dot(A[q, q])
    B = xp.zeros_like(A[q, q])
    if (q - 1) >= 0:
        D += S[q, q - 1].dot(A[q - 1, q])
        T = S[q, q].dot(A[q, q - 1])
        T += S[q, q - 1].dot(A[q - 1, q - 1])
        B += T.dot(S[q - 1, q])
    if (q + 1) < A.N:
        D += S[q, q + 1].dot(A[q + 1, q])
        T = S[q, q].dot(A[q, q + 1])
        T += S[q, q + 1].dot(A[q + 1, q + 1])
        B += T.dot(S[q + 1, q])
    D.dot(S[q, q], out=D)
    B += D
    B = rotate_matrix(B, xp.linalg.inv(S[q, q]))
    if indices is not None:
        return project(B, S[q, q], indices)
    return B


class ExpandSelfEnergy:
    """Expand selfenergy defined in a subspace."""

    def __init__(self, S, selfenergy, indices) -> None:
        self.S = S
        self.selfenergy = selfenergy
        self.indices = indices

    def expand(self, X):
        return expand(self.S, X, self.indices)

    def retarded(self, energy):
        return self.expand(self.selfenergy.retarded(energy))


@singledispatch
def expand(S, A, q):
    """Unfold projected matrix onto original space."""
    return rotate_matrix(rotate_matrix(A, S[xp.ix_(q, q)]), S[q, :])


@expand.register(BTMatrix)
def _(S, A, q, indices=None):
    # [0,              0,              0,              0]
    # [0, G_22*S_12*S_21, G_22*S_12*S_22, G_22*S_12*S_23]
    # [0, G_22*S_21*S_22,   G_22*S_22**2, G_22*S_22*S_23]
    # [0, G_22*S_21*S_32, G_22*S_22*S_32, G_22*S_23*S_32]
    N = S.N
    B = _BTBuffer(N)
    if indices is not None:
        A = expand(A, S[q, q], indices)
    B[q, q] = rotate_matrix(A, S[q, q])
    if q > 0:
        B[q - 1, q] = S[q - 1, q].dot(A).dot(S[q, q])
        B[q, q - 1] = S[q, q].dot(A).dot(S[q, q - 1])
        B[q - 1, q - 1] = S[q - 1, q].dot(A).dot(S[q, q - 1])
    if (q + 1) < N:
        B[q + 1, q] = S[q + 1, q].dot(A).dot(S[q, q])
        B[q, q + 1] = S[q, q].dot(A).dot(S[q, q + 1])
        B[q + 1, q + 1] = S[q + 1, q].dot(A).dot(S[q, q + 1])
    return B  # BTMatrix(B.m_qii, B.m_qij, B.m_qji)
