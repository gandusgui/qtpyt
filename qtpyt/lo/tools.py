import numpy as np
from math import sqrt
from scipy import linalg as la
from scipy.linalg import block_diag


dagger = lambda mat: np.conj(mat.T)


def rotate_matrix(h, u):
    return np.dot(u.T.conj(), np.dot(h, u))


def get_subspace(matrix, index):
    """Get the subspace spanned by the basis function listed in index"""
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    return matrix.take(index, 0).take(index, 1)


def normalize(matrix, s=None):
    """Normalize column vectors.

    ::

      <matrix[:,i]| s |matrix[:,i]> = 1

    """
    for col in matrix.T:
        if s is None:
            col /= np.linalg.norm(col)
        else:
            col /= np.sqrt(np.dot(col.conj(), np.dot(s, col)))


from scipy.linalg import eigh


def get_orthonormal_subspace(h, s, index_j=None):
    """Get orthonormal eigen-values and -vectors of subspace j."""
    if index_j is not None:
        h_jj = get_subspace(h, index_j)
        s_jj = get_subspace(s, index_j)
    else:
        h_jj = h
        s_jj = s
    # from IPython import embed; embed()
    w, v = eigh(h_jj, s_jj)
    return v, w
    # e_j, v_jj = np.linalg.eig(np.linalg.solve(s_jj, h_jj))
    # normalize(v_jj, s_jj)  # normalize: <v_j|s|v_j> = 1
    # permute_list = np.argsort(e_j.real)
    # e_j = np.take(e_j, permute_list)
    # v_jj = np.take(v_jj, permute_list, axis=1)
    # return v_jj, e_j


def subdiagonalize(h, s, index_j):
    """Subdiagonalize a block in the Hamiltonian."""

    v_jj, e_j = get_orthonormal_subspace(h, s, index_j)
    U = np.identity(h.shape[0], h.dtype)
    U[np.ix_(index_j, index_j)] = v_jj[:]
    return U, e_j


def subdiagonalize_atoms(basis, h, s, a=None):
    """Get rotation matrix for subdiagonalization of given(all) atoms."""

    if a is None:
        a = range(len(basis.atoms))
    if isinstance(a, int):
        a = [a]
    U_mm = []
    e_aj = []
    na_tot = len(basis.atoms)
    for a0 in range(na_tot):
        bfs = basis.get_indices(a0)
        if a0 in a:
            v_jj, e_j = get_orthonormal_subspace(h, s, bfs)
            U_mm.append(v_jj)
            e_aj.append(e_j)
        else:
            U_mm.append(np.eye(len(bfs)))
    U_mm = block_diag(*U_mm)
    return U_mm, e_aj


def cutcoupling(h, s, index_n):
    for i in index_n:
        s[:, i] = 0.0
        s[i, :] = 0.0
        s[i, i] = 1.0
        Ei = h[i, i]
        h[:, i] = 0.0
        h[i, :] = 0.0
        h[i, i] = Ei


def lowdin_rotation(h, s, index_j=None):
    s_jj = get_subspace(s, index_j) if index_j is not None else s
    eig, U_jj = np.linalg.eigh(s_jj)
    eig = np.abs(eig)
    U_jj = np.dot(U_jj / np.sqrt(eig), dagger(U_jj))
    if index_j is not None:
        U = np.identity(s.shape[0], h.dtype)
        U[np.ix_(index_j, index_j)] = U_jj[:]
        U_jj = U
    return U_jj


def get_U1(bfs_m, bfs_i, c_MM=None):
    """Get rotation matrix to extract sub-block defined 
    by matrix indices `bfs_m` to the bottom rigth of the 
    matrix. If a rotation c_MM is given apply permutation to c_MM.

    """
    nbf = len(bfs_m) + len(bfs_i)
    if c_MM is None:
        c_MM = np.eye(nbf)
    U1 = np.take(c_MM, np.concatenate([bfs_i, bfs_m]), axis=1)
    """If apply is True permute inplace bfs_m bfs_i"""
    bfs_m[:] = range(nbf - len(bfs_m), nbf)
    bfs_i[:] = range(nbf - len(bfs_m))
    return U1, bfs_m, bfs_i


def get_U2(s1, bfs_m, bfs_i):
    """Get rotation matrix that orthogonalizes sub-blocks spanned
    by indices `bfs_m` and `bfs_i` leaving bfs_m unchanged.
    
    """
    nbf = len(bfs_m) + len(bfs_i)
    s_mm = get_subspace(s1, bfs_m)
    s_mi = s1.take(bfs_m, axis=0).take(bfs_i, axis=1)
    U_mi = -np.linalg.inv(s_mm).dot(s_mi)
    U2 = np.eye(s1.shape[0])
    U2[np.ix_(bfs_m, bfs_i)] = U_mi
    return U2


def get_U2T(s1_MM, bfs_m, bfs_i):
    """Get rotation matrix that orthogonalizes sub-blocks spanned
    by indices `bfs_m` and `bfs_i` leaving bfs_i unchanged.
    
    """
    nbf = len(bfs_m) + len(bfs_i)
    s_ii = get_subspace(s1_MM, bfs_i)
    s_im = s1_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    U_im = -np.linalg.inv(s_ii).dot(s_im)
    U2T = np.eye(s1_MM.shape[0])
    U2T[np.ix_(bfs_i, bfs_m)] = U_im
    return U2T


def extract_orthogonal_subspaces(h, s, bfs_m, U=None, T=False):
    nbf = h.shape[0]
    bfs_i = np.setdiff1d(range(nbf), bfs_m)
    U1, bfs_m, bfs_i = get_U1(bfs_m, bfs_i, U)
    h1 = rotate_matrix(h, U1)
    s1 = rotate_matrix(s, U1)
    U2 = get_U2(s1, bfs_m, bfs_i) if not T else get_U2T(s1, bfs_m, bfs_i)
    h2 = rotate_matrix(h1, U2)
    s2 = rotate_matrix(s1, U2)
    hs_mm, hs_ii, hs_im = split_subspaces(h2, s2, bfs_m, bfs_i)
    U = U1.dot(U2)
    return hs_mm, hs_ii, hs_im, U


def split_subspaces(h_MM, s_MM, bfs_m, bfs_i):
    nbf = h_MM.shape[0]
    h_ii = get_subspace(h_MM, bfs_i)
    s_ii = get_subspace(s_MM, bfs_i)
    h_mm = get_subspace(h_MM, bfs_m)
    s_mm = get_subspace(s_MM, bfs_m)
    h_im = h_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    s_im = s_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    return (h_mm, s_mm), (h_ii, s_ii), (h_im, s_im)


flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]


def get_orthogonal_subspaces(basis, h_MM, s_MM, a=None, cutoff=np.inf, condition=None):
    """Split matrix in hybridization and active spaces.
       Active space contains orbital(s) of atom(s) a with
       abs(energy) < cutoff"""
    c_MM, e_aj = subdiagonalize_atoms(basis, h_MM, s_MM, a)
    bfs_imp = basis.get_indices(a)
    e_j = (e for e_j in e_aj for e in e_j)
    if condition is None:
        condition = lambda e: abs(e) < cutoff
    # bfs_m = list(bfs_imp[j] for j, e in enumerate(e_j) if abs(e)<cutoff)
    bfs_m = list(bfs_imp[j] for j, e in enumerate(e_j) if condition(e))
    return extract_orthogonal_subspaces(h_MM, s_MM, bfs_m, c_MM)


def order_transform(basis, groups=None, nuc=None):
    """Order orbital indices using group of atoms and atomic elements.
    
    For each subset of atoms, group the atoms of the same kind and order
    the orbital indices in scending order. When both `groups` and `nuc` 
    are None, treat the whole set of atoms as a single group.

    Args:
        basis : (Basis object)
        groups : (list of lists) 
            List of the atoms in each group.
        nuc : (int)
            Number of sequential unit cells in atoms.
            Each unit cell is considered a group.
    """
    atoms = basis.atoms
    symbols = atoms.symbols
    natoms = len(atoms)

    if groups is not None:
        try:
            groups[0][0]
        except IndexError:
            raise ValueError("`groups` must be a list of lists of atoms in each group.")
        nuc = len(groups)
        range_uc = lambda uc: groups[uc]
    else:
        if nuc is not None:
            if not natoms % nuc == 0.0:
                raise ValueError("Number of unit cells are not a dividend of atoms.")
            nuc_atoms = natoms // nuc
        else:
            nuc = 1
            nuc_atoms = natoms
        range_uc = lambda uc: range(uc * nuc_atoms, (uc + 1) * nuc_atoms)

    indices = []
    for uc in range(nuc):
        elements = np.unique(atoms[range_uc(uc)].symbols)
        indices.append(
            list(
                list(filter(lambda a: symbols[a] == elem, range_uc(uc)))
                for elem in elements
            )
        )

    perm_indices = []
    for uc in indices:
        for elem in uc:
            perm_indices.extend(
                basis.get_indices(elem).reshape(len(elem), -1).T.flatten()
            )

    U = np.eye(basis.nao).take(perm_indices, axis=1)
    return U, indices


def get_ll_mm_rr(H, S, nbf_l, nbf_m, nbf_r):
    rng_l = list(range(nbf_l))
    rng_m = list(range(nbf_l, nbf_l + nbf_m))
    rng_r = list(range(nbf_l + nbf_m, nbf_l + nbf_m + nbf_r))
    s_lm = S.take(rng_l, 0).take(rng_m, 1)
    h_lm = H.take(rng_l, 0).take(rng_m, 1)
    s_rm = S.take(rng_r, 0).take(rng_m, 1)
    h_rm = H.take(rng_r, 0).take(rng_m, 1)
    h_mm = get_subspace(H, rng_m)
    s_mm = get_subspace(S, rng_m)
    return (h_lm, s_lm), (h_mm, s_mm), (h_rm, s_rm)


def get_U3(S, bfs_l, bfs_m, bfs_r):
    """Get rotation matrix that orthogonalizes sub-blocks spanned
    by indices `m` and `l` and `m` and `r` leaving `m` unchanged 
    
    NOTE: requires zero overlap between `l` and `r`.
    
    """
    mm = np.ix_(bfs_m, bfs_m)
    ml = np.ix_(bfs_m, bfs_l)
    mr = np.ix_(bfs_m, bfs_r)
    U = np.eye(len(bfs_l) + len(bfs_m) + len(bfs_r))
    U[ml] = -np.linalg.inv(S[mm]).dot(S[ml])
    U[mr] = -np.linalg.inv(S[mm]).dot(S[mr])
    return U
