from itertools import accumulate
import numpy as np
from numba import njit
from functools import singledispatch

#
from qtpyt.block_tridiag.btmatrix import BTMatrix, _BTBuffer

# NeighborList
from qtpyt.basis import get_neighbors

"""
Module's reference:

    Crossed graphene nanoribbons as beam splitters and mirrors
    for electron quantum optics. Sofia Sanz, et al.

"""


@njit
def _orbital_current(A, H, index_i, index_j, A_dag, H_dag):
    """Compute local current matrix between subset of indices.
    
    J   = H   A    -  H    A
     ij     ji  ij      ij   ji
    """
    ni = len(index_i)
    nj = len(index_j)
    J_ij = np.zeros((ni, nj))
    for i, j in np.ndindex(ni, nj):
        ii = index_i[i]
        jj = index_j[j]
        J_ij[i, j] = np.imag(H_dag[jj, ii] * A[ii, jj] - H[ii, jj] * A_dag[jj, ii])
    return J_ij


def _orbital_current_btd(A, H, index_i, index_j):
    """Block-tridiagonal version of _orbitals_current"""
    nodes = np.cumsum([m.shape[0] for m in H.m_qii])
    # Indices of atoms in recursive block
    # np.searchsorted(..,side='right') because
    # if indices_i[0] == nodes[q=0], then atom i
    # goes to next block q=1
    #
    qi = np.searchsorted(nodes, index_i, side="right")
    qj = np.searchsorted(nodes, index_j, side="right")

    # Check
    assert len(set(qi)) == 1
    assert len(set(qj)) == 1
    # If OK, all orbitals belong to same block
    qi = qi[0]
    qj = qj[0]

    M_q = np.insert(nodes, 0, 0)[:-1]
    index_i = np.asarray(index_i) - M_q[qi]
    index_j = np.asarray(index_j) - M_q[qj]

    if qi == qj:
        A_ij = A.m_qii[qi]
        H_ij = H.m_qii[qi]
        A_ji = A_ij
        H_ji = H_ij
    elif qj == qi + 1:
        A_ij = A.m_qij[qi]
        H_ij = H.m_qij[qi]
        A_ji = A.m_qji[qi]
        H_ji = H.m_qji[qi]
    elif qi == qj + 1:
        A_ij = A.m_qji[qj]
        H_ij = H.m_qji[qj]
        A_ji = A.m_qij[qj]
        H_ji = H.m_qij[qj]

    return _orbital_current(A_ij, H_ij, index_i, index_j, A_ji, H_ji)


def orbital_current(A, H, index_i, index_j):
    """Dispatcher."""
    if isinstance(A, (BTMatrix, _BTBuffer)):
        return _orbital_current_btd(A, H, index_i, index_j)
    else:
        return _orbital_current(A, H, index_i, index_j, A, H)


def bond_current(A, H, bfs_ai, nlists):
    J_aa = [[None for _ in nlist] for nlist in nlists]
    for a0, nlist in enumerate(nlists):
        index_i = bfs_ai[a0]
        for j, a1 in enumerate(nlist):
            index_j = bfs_ai[a1]
            J_aa[a0][j] = orbital_current(A, H, index_i, index_j).sum()
    return J_aa


def quiver_plot(atoms, J_aa, nlists, **kwargs):
    """ Get arrows info
    X - origins
    U - vectors with legth of corresponding bond
    W - weigths
    """
    X = []
    U = []
    W = []
    for a in atoms:
        nlist = nlists[a.index]
        pos = np.tile(a.position, (len(nlist), 1))
        dist = atoms[nlist].positions - a.position  # \
        # / atoms.get_distances(a.index, nlist)[:,None]
        X.extend(pos.tolist())
        U.extend(dist.tolist())
        W.extend(J_aa[a.index])

    X = np.asarray(X)
    U = np.asarray(U)
    W = np.asarray(W)
    Wneg = W < 0
    # Shift arrow to moddle of plot
    X[Wneg] += U[Wneg] / 2
    # Reverse arrow
    U[Wneg] *= -1
    W = np.abs(W)
    return X, U, W


def normalize_weigths(W, normalize=False, vmin=None, vmax=None, clip=False):
    if normalize:
        W = (W - W.min()) / (W.max() - W.min())
    if clip:
        W = np.clip(W, vmin, vmax)
    return W


class DensityCurrent(object):
    def __init__(self, gf, basis, indices=None):
        self.gf = gf
        self.indices = indices or np.arange(len(basis.atoms))
        self.atoms = basis.atoms[self.indices]
        self.bfs_ai = [basis.get_indices(a) for a in self.indices]
        self.energy = None
        self._set_nlists()

    def __call__(self, energy):
        # Spectral functions
        if energy != self.energy:
            gf = self.gf
            self.sepectrals = gf.get_spectrals(energy)
        J = []
        for A in self.sepectrals:
            J.append(bond_current(A, gf.H, self.bfs_ai, self.nlists))

        return J

    def _set_nlists(self):
        atoms = self.atoms
        pbc = atoms.pbc.copy()
        # Hack to avoid arrow between boundaries
        atoms.set_pbc(False)
        self.nlists = get_neighbors(atoms)
        atoms.set_pbc(pbc)

    def add_neighbors(self, index, neighbors):
        self.nlists[index] = np.insert(self.nlists[index], 0, neighbors)

    def pop_neighbors(self, index, neighbors):
        self.nlists[index] = np.delete(self.nlists[index], neighbors)

    def display(
        self,
        J_aa,
        nlists,
        projection="2d",
        plane="xy",
        scale=2,
        normalize=True,
        vmin=None,
        vmax=None,
        clip=False,
    ):

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt
        import matplotlib.cm as cm

        # from matplotlib.colors import Normalize
        atoms = self.atoms

        X, U, W = quiver_plot(atoms, J_aa, nlists)
        W = normalize_weigths(W, normalize, vmin, vmax, clip)
        colormap = cm.YlOrBr

        fig = plt.figure(figsize=(atoms.cell[0, 0] / scale, atoms.cell[1, 1] / scale))
        if projection is "3d":
            ax = fig.gca(projection=projection)
            ax.quiver(
                X[:, 0], X[:, 1], X[:, 2], U[:, 0], U[:, 1], U[:, 2], color=colormap(W)
            )
        else:
            ax = fig.gca()
            a = "xyz".index(plane[0])
            b = "xyz".index(plane[1])
            ax.quiver(X[:, a], X[:, b], U[:, a], U[:, b], color=colormap(W))
            ax.scatter(atoms.positions[:, a], atoms.positions[:, b])
        return ax
