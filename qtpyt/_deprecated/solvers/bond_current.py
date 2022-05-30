import numpy as np
from functools import singledispatch

#
from .recursive import multiply, get_diagonal
from qtpyt.greenfunction import RecursiveGF, GreenFunction
from qtpyt.tools import dagger

# NeighborList
from qtpyt.tk_atoms import get_neighbors


def _orbital_current_gf(A_mm, H_mm, index_i, index_j, A_dag_mm, H_dag_mm):
    """ Reference bond current formula:
    Crossed graphene nanoribbons as beam splitters and mirrors
    for electron quantum optics.
    Sofia Sanz, et al.
    """
    ni = len(index_i)
    nj = len(index_j)
    J_ij = np.zeros((ni, nj))
    for i, j in np.ndindex(ni, nj):
        ii = index_i[i]
        jj = index_j[j]
        J_ij[i, j] = np.imag(
            H_dag_mm[jj, ii] * A_mm[ii, jj] - H_mm[ii, jj] * A_dag_mm[jj, ii]
        )
    return J_ij


def _orbital_current_rgf(A_xqmm, H_xqmm, index_i, index_j):
    A_qii, A_qij, A_qji = A_xqmm
    H_qii, H_qij, H_qji = H_xqmm
    sizes = np.cumsum([H_mm.shape[0] for H_mm in H_qii])
    # Indices of atoms in recursive partition
    # np.searchsorted(..,side='right') because
    # if indices_i[0] == sizes[q=0], then atom i
    # goes to next partition q=1
    #
    qi = np.searchsorted(sizes, index_i, side="right")
    qj = np.searchsorted(sizes, index_j, side="right")

    # Check
    assert len(set(qi)) == 1
    assert len(set(qj)) == 1
    # If OK, all orbitals belong to same partition
    qi = qi[0]
    qj = qj[0]

    M_q = np.insert(sizes, 0, 0)[:-1]
    index_i = np.asarray(index_i) - M_q[qi]
    index_j = np.asarray(index_j) - M_q[qj]

    if qi == qj:
        A_mm = A_qii[qi]
        H_mm = H_qii[qi]
        A_dag_mm = A_mm
        H_dag_mm = H_mm
    elif qj == qi + 1:
        A_mm = A_qij[qi]
        H_mm = H_qij[qi]
        A_dag_mm = A_qji[qi]
        H_dag_mm = H_qji[qi]
    elif qi == qj + 1:
        A_mm = A_qji[qj]
        H_mm = H_qji[qj]
        A_dag_mm = A_qij[qj]
        H_dag_mm = H_qij[qj]

    return _orbital_current_gf(A_mm, H_mm, index_i, index_j, A_dag_mm, H_dag_mm)


def orbital_current(A, H, index_i, index_j):
    if isinstance(A, (list, tuple)):
        return _orbital_current_rgf(A, H, index_i, index_j)
    else:
        return _orbital_current_gf(A, H, index_i, index_j, A, H)


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
    def __init__(self, GF, atoms, bfs_ai):
        self.GF = GF
        self.atoms = atoms
        self.bfs_ai = bfs_ai
        self.energy = None
        self._set_nlists()
        self.indices = None

    def __call__(self, energy):
        # Spectral functions
        if energy != self.energy:
            GF = self.GF
            self.A1, self.A2 = GF.get_spectrals(energy)
        bfs_ai = self._get_lists("bfs_ai")
        nlists = self._get_lists("nlists")
        J1_aa = bond_current(self.A1, GF.H, bfs_ai, nlists)
        J2_aa = bond_current(self.A2, GF.H, bfs_ai, nlists)

        return J1_aa, J2_aa, nlists

    def _get_lists(self, name):
        attr = getattr(self, name)
        if self.indices is None:
            return attr
        else:
            return [l for a, l in enumerate(attr) if a in self.indices]

    def _set_nlists(self):
        atoms = self.atoms
        pbc = atoms.pbc.copy()
        # Hack to avoid arrow between boundaries
        atoms.set_pbc(False)
        self.nlists = get_neighbors(atoms)
        atoms.set_pbc(pbc)

    def add_neighbors(self, index, neighbors):
        if isinstance(neighbors, int):
            neighbors = [neighbors]
        elif isinstance(neighbors, np.ndarry):
            neighbors = neighbors.tolist()
        self.nlists[index].extend(neighbors)
        for neighbor in neighbors:
            self.nlists[neighbor].append(index)

    def pop_neighbors(self, index, neighbors):
        if isinstance(neighbors, int):
            neighbors = [neighbors]
        elif isinstance(neighbors, np.ndarry):
            neighbors = neighbors.tolist()
        for neighbor in neighbors:
            ii = self.nlists[index].index(neighbor)
            jj = self.nlists[neighbor].index(index)
            self.nlists[index].pop(ii)
            self.nlists[neighbor].pop(jj)

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
