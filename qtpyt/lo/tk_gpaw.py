import numpy as np
from scipy import linalg as la
from ase.calculators.singlepoint import SinglePointCalculator
from gpaw.setup import types2atomtypes
from gpaw.basis_data import Basis
from .tools import (
    subdiagonalize,
    get_orthonormal_subspace,
    get_subspace,
    rotate_matrix,
    dagger,
)
from .block import block


def get_atom_indices(calc, a):
    from .tk_calc import get_info

    M_a, nao_a, nao = get_info(calc, ["M_a", "nao_a", "nao"])
    if isinstance(a, (int, str)):
        a = [a]
    ai = []
    for elem in a:
        if isinstance(elem, int):
            ai.append(elem)
        elif isinstance(elem, str):
            symbol = elem
            for a0 in range(len(calc.atoms)):
                offset = 0
                if a0 >= len(M_a):  # original number of atoms,
                    # e.g. allows calc.atoms = atoms.repeat(..)
                    offset += (a0 // len(M_a)) * len(M_a)
                    a0 %= len(M_a)
                if calc.atoms[a0].symbol == symbol:
                    ai.append(a0 + offset)
        else:
            raise Warning("Non valid input {}".format(elem))
    ai = sorted(ai)
    return ai


def get_bfs_indices(calc, a, method="extend"):
    """Get list of basis function indices of atom(s) a.
    Available methods:
        1. 'extend' - return flatten ao indices
        2. 'append' - return ao.shape(natoms,nao)
    """
    from .tk_calc import get_info

    M_a, nao_a, nao = get_info(calc, ["M_a", "nao_a", "nao"])
    if isinstance(a, (int, np.int32, np.int64)):
        a = [a]
    Mvalues = []
    Mattr = getattr(Mvalues, method)
    for a0 in a:
        M = 0
        M += M_a[a0]
        Mattr(list(range(M, M + nao_a[a0])))
    return Mvalues


def get_bfs_atoms(atoms, basis=None, default="dzp", method="extend"):
    """Get ba
    atoms - ase::atoms
    basis - dict of elements with correspondig basis set type
    default - str
    """
    if basis is None:
        basis = {symbol: default for symbol in set(atoms.symbols)}
    symbols = atoms.get_chemical_symbols()
    basis_a = types2atomtypes(symbols, basis, default)

    nao_a = [Basis(symbol, type).nao for symbol, type in zip(symbols, basis_a)]

    Mvalues = []
    Mattr = getattr(Mvalues, method)
    i0 = 0
    for nao in nao_a:
        i1 = i0 + nao
        Mattr(list(range(i0, i1)))
        i0 = i1
    return Mvalues


def subdiagonalize_atoms(calc, h_ii, s_ii, a=None):
    """Get rotation matrix for subdiagonalization of given(all) atoms."""
    from scipy.linalg import block_diag

    if a is None:
        a = range(len(calc.atoms))
    if isinstance(a, int):
        a = [a]
    U_mm = []
    e_aj = []
    na_tot = len(calc.atoms)
    for a0 in range(na_tot):
        bfs = get_bfs_indices(calc, a0)
        if a0 in a:
            v_jj, e_j = get_orthonormal_subspace(h_ii, s_ii, bfs)
            U_mm.append(v_jj)
            e_aj.append(e_j)
        else:
            U_mm.append(np.eye(len(bfs)))
    U_mm = block_diag(*U_mm)
    return U_mm, e_aj


def subdiagonalize_blocks(calc, h_ii, s_ii, a=None):
    """Get rotation matrix for subdiagonalization of given(all) atoms."""
    from scipy.linalg import block_diag

    U_mm = []
    e_aj = []
    for a0 in a:
        bfs = get_bfs_indices(calc, a0)
        v_jj, e_j = get_orthonormal_subspace(h_ii, s_ii, bfs)
        U_mm.append(v_jj)
        e_aj.append(e_j)
    U_mm = block_diag(*U_mm)
    return U_mm, e_aj


def get_bf_centers(atoms, basis=None):

    from ase.calculators.singlepoint import SinglePointCalculator
    from gpaw.setup import types2atomtypes

    calc = atoms.get_calculator()
    if calc is None or isinstance(calc, SinglePointCalculator):
        symbols = atoms.get_chemical_symbols()
        basis_a = types2atomtypes(symbols, basis, "dzp")
        nao_a = [Basis(symbol, type).nao for symbol, type in zip(symbols, basis_a)]
    else:
        if not calc.initialized:
            calc.initialize(atoms)
        nao_a = [calc.wfs.setups[a].nao for a in range(len(atoms))]
    pos_ic = []
    for pos, nao in zip(atoms.get_positions(), nao_a):
        pos_ic.extend(pos[None].repeat(nao, 0))
    return np.array(pos_ic)


def sum_bf_atom(calc, X_i, n_a=0):
    """Sum basis functions on the same atom."""
    if n_a == 0:
        n_a = len(calc.atoms)
    X_a = np.zeros(n_a, dtype=X_i.dtype)
    i0 = 0
    for a0 in range(n_a):
        i1 = i0 + calc.wfs.setups[a0].nao
        X_a[a0] = sum(X_i[i0:i1])
        i0 = i1
    return X_a


def flatten(iterables):
    return (elem for iterable in iterables for elem in iterable)


def get_U1(bfs_m, bfs_i, c_MM=None, apply=False):
    nbf = len(bfs_m) + len(bfs_i)
    if c_MM is None:
        c_MM = np.eye(nbf)
    U_mm = c_MM.take(bfs_m, axis=1)
    U_ii = c_MM.take(bfs_i, axis=1)
    U1 = np.array(block([[1, 0], [U_ii, U_mm]]))
    """If apply is True permute inplace bfs_m bfs_i"""
    if apply:
        bfs_m[:] = range(nbf - len(bfs_m), nbf)
        bfs_i[:] = range(nbf - len(bfs_m))
    return U1


def get_U2(s1_MM, bfs_m, bfs_i):
    s_mm = get_subspace(s1_MM, bfs_m)
    s_mi = s1_MM.take(bfs_m, axis=0).take(bfs_i, axis=1)
    U_mi = -la.inv(s_mm).dot(s_mi)
    U_ii = np.eye(len(bfs_i))
    U2 = np.array(block([[U_ii, 0], [U_mi, np.eye(len(bfs_m))]]))
    return U2


def get_mm_ii_im(h_MM, s_MM, bfs_m, bfs_i):
    h_ii = get_subspace(h_MM, bfs_i)
    s_ii = get_subspace(s_MM, bfs_i)
    h_mm = get_subspace(h_MM, bfs_m)
    s_mm = get_subspace(s_MM, bfs_m)
    h_im = h_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    s_im = s_MM.take(bfs_i, axis=0).take(bfs_m, axis=1)
    return (h_mm, s_mm), (h_ii, s_ii), (h_im, s_im)


def extract_orthogonal_subspaces(
    h_MM, s_MM, bfs_m, bfs_i, c_MM=None, orthogonal=True, apply=True
):
    U1 = get_U1(bfs_m, bfs_i, c_MM, apply=True)  # Modifies bfs_m, bfs_i
    h1_MM = rotate_matrix(h_MM, U1)
    s1_MM = rotate_matrix(s_MM, U1)
    """If orthogonal is True set s_im to zeros."""
    if orthogonal:
        U2 = get_U2(s1_MM, bfs_m, bfs_i)
        h2_MM = rotate_matrix(h_MM, U1.dot(U2))
        s2_MM = rotate_matrix(s_MM, U1.dot(U2))
    else:
        h2_MM = h1_MM
        s2_MM = s1_MM
    if apply is False:
        return U1.dot(U2)
    return get_mm_ii_im(h2_MM, s2_MM, bfs_m, bfs_i)


def get_orthogonal_subspaces(calc, h_MM, s_MM, a=None, cutoff=np.inf, orthogonal=True):
    """Split matrix in hybridization and active spaces.
       Active space contains orbital(s) of atom(s) a with
       abs(energy) < cutoff"""
    nbf = len(h_MM)
    c_MM, e_aj = subdiagonalize_atoms(calc, h_MM, s_MM, a)
    bfs_imp = get_bfs_indices(calc, a)
    bfs_m = [
        bfs_imp[i] for i, eigval in enumerate(flatten(e_aj)) if abs(eigval) < cutoff
    ]
    bfs_i = list(np.setdiff1d(range(nbf), bfs_m))
    return extract_orthogonal_subspaces(h_MM, s_MM, bfs_m, bfs_i, c_MM, orthogonal)


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


def get_lead_orthogonal(greenfunction):
    s_ll = greenfunction.selfenergies[0].s_ii
    s_rr = greenfunction.selfenergies[1].s_ii
    h_ll = greenfunction.selfenergies[0].h_ii
    h_rr = greenfunction.selfenergies[1].h_ii
    s_lm = greenfunction.selfenergies[0].s_im
    s_rm = greenfunction.selfenergies[1].s_im
    h_lm = greenfunction.selfenergies[0].h_im
    h_rm = greenfunction.selfenergies[1].h_im
    h_mm = greenfunction.H
    s_mm = greenfunction.S
    nbf_l = len(h_ll)
    nbf_r = len(h_rr)
    nbf_m = len(h_mm)
    s_ml = dagger(s_lm)
    s_mr = dagger(s_rm)
    h_ml = dagger(h_lm)
    h_mr = dagger(h_rm)
    U_lm = -la.inv(s_ll).dot(s_lm)
    U_rm = -la.inv(s_rr).dot(s_rm)
    U = np.array(
        block(
            [[np.eye(nbf_l), U_lm, 0], [0, np.eye(nbf_m), 0], [0, U_rm, np.eye(nbf_r)]]
        )
    )
    S = np.array(block([[s_ll, s_lm, 0], [s_ml, s_mm, s_mr], [0, s_rm, s_rr]]))
    H = np.array(block([[h_ll, h_lm, 0], [h_ml, h_mm, h_mr], [0, h_rm, h_rr]]))
    S = rotate_matrix(S, U)
    H = rotate_matrix(H, U)
    return get_ll_mm_rr(H, S, nbf_l, nbf_m, nbf_r)
