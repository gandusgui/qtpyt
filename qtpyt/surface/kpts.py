from functools import partial

import numpy as np
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from numba import njit
from scipy.spatial.distance import cdist


def monkhorst_pack(size, offset=(0.0, 0.0, 0.0)):
    """K-points from Monkhorst-Pack."""
    kpts = np.indices(size).reshape(len(size), -1).T
    kpts = (kpts + 0.5) / size - 0.5 + offset
    return kpts


def _get_dirs(direction="x"):
    """Get indices parallel and transverse directions.
    
    Args:
        direction : (str) parallel direction. 
            One of 'x', 'y' or 'z'
    """
    dirs_t = [0, 1, 2]
    dir_p = dirs_t.pop("xyz".index(direction))
    return dir_p, dirs_t


@njit
def _index(array, item):
    """Find item in array."""
    for idx, val in enumerate(array):
        if val == item:
            return idx


def map_ik2k(kpts, k2ik):
    """Get mapping from irreducible (ikpts) to reducible k-points (kpts).
    
    Args:
        kpts : (np.ndarray) k-points
        k2ik : map from kpts to ikpts such that kpts[k2ik] = ikpts

    Returns:
        ik2k : map from ikpts to kpts.
        time_reversal : whether item of map must be time-reversed.    
    """
    # Find if there's a minimum distance with time-reversed k-point.
    # distances = cdist(kpts, np.dot(kpts, -np.eye(3).T)) % 1
    distances = cdist(kpts % 1, -kpts % 1)  #% 1
    equivalent = np.where(distances.min(1) < 1e-7, distances.argmin(1), -1)
    ik2k = np.empty(kpts.shape[0], int)
    ik2k[k2ik] = np.arange(len(k2ik))
    diff = np.setdiff1d(range(kpts.shape[0]), k2ik)
    for idx, val in enumerate(diff):
        try:
            ik2k[val] = _index(k2ik, equivalent[val])
        except:
            raise IndexError
    # ik2k[diff] = [_index(k2ik, item) for item in equivalent[diff]]
    time_reverse = np.zeros(kpts.shape[0], bool)
    time_reverse[diff] = True
    return ik2k, time_reverse


def apply_inversion_symm(kpts, direction="x"):
    """Reduce inversion (time reversal) symmetry along direction.
    
    Args:
        kpts : (np.ndarray) k-points
        direction : time-reverse direction

    Returns:
        ikpts : (np.ndarray) irreducible k-points
        k2ik : mapping from kpts to ikpts, i.e. kpts[k2ik] = ikpts
        ik2k : map from ikpts to kpts.
        time_reversal : whether item of map must be time-reversed.
    
    Example:
        kpts = monkhorst-pack(size)
        ikpts, k2ik, ik2k, time_reverse = apply_inversion_symm(kpts)
        A :: k-matrix at ikpts
        B :: k-matrix at kpts
        B = A[ik2k]
        B[time_reverse] = np.transpose(B[time_reverse],axes=(0,2,1))
    """
    p_dir = "xyz".index(direction)
    ikpts = []
    k2ik = []
    for bzk_index, bzk_k in enumerate(kpts):
        try:
            if bzk_k[np.nonzero(bzk_k)[0][p_dir]] > 0:
                ikpts.append(bzk_k)
                k2ik.append(bzk_index)
            else:
                continue
        # zero case
        except IndexError:
            ikpts.append(bzk_k)
            k2ik.append(bzk_index)
    ikpts = np.array(ikpts)
    k2ik = np.array(k2ik)
    ik2k, time_reverse = map_ik2k(kpts, k2ik)
    return ikpts, k2ik, ik2k, time_reverse


def expand_inversion_symm(kpts, A, direction="x"):
    """Expand the k-matrices A to the full-BZ.

    Given the (ir)redicible k-points `kpts` and the matrices
    A evaluated at the (ir)redicible k-points, expand A
    to the full-BZ.

    Args:
        kpts : (np.ndarray) k-points
        direction : time-reverse direction
        A :  matrices in k-space.
    """
    assert A.shape[0] < kpts.shape[0], "A already in reducible BZ"
    ikpts, k2ik, ik2k, time_reverse = apply_inversion_symm(kpts, direction)
    B = A[ik2k]
    B[time_reverse] = np.transpose(B[time_reverse], axes=(0, 2, 1))
    kpts = ikpts[ik2k]
    kpts[time_reverse] = -kpts[time_reverse]
    return kpts, B


def build_partial_kpts(size, offset=(0.0, 0.0, 0.0), direction="x"):
    """Build parallel and transverse k-points. 
    
    Args:
        size, offset : Monkhorst-Pack.
        direction : parallel direction.

    Returns:
        kpts_p : parallel k-points
        kpts_t : transverse k-points 
    """
    size = np.asarray(size)
    offset = np.asarray(offset)

    dir_p, dirs_t = _get_dirs(direction)

    size_p = np.ones(3, int)
    size_p[dir_p] = size[dir_p]
    offset_p = np.zeros(3)
    offset_p[dir_p] = offset[dir_p]

    kpts_p = monkhorst_pack(size_p, offset_p)

    size_t = np.ones(3, int)
    size_t[dirs_t] = size[dirs_t]
    offset_t = np.zeros(3)
    offset_t[dirs_t] = offset[dirs_t]

    kpts_t = monkhorst_pack(size_t, offset_t)

    return kpts_p, kpts_t


def fourier_sum(A, kpts, R, out=None, inverse=False):
    """K-space Fourier sum.
    
    Args:
        A : (np.ndarray) matrices in k-space.
        kpts : (1D or 2D) k-points at which A's have been (will be) evaluated.
        R : (1D or 2D) lattice vectors at which A's will be (have been) evaluated.
        inverse : compute inverse fourier transform
    """
    # np.asarray avoids copy
    kpts = np.asarray(kpts)
    R = np.asarray(R)
    if kpts.ndim < 2:
        kpts = kpts[None, :]
    squeeze = False
    if R.ndim < 2:
        squeeze = True
        R = R[None, :]
    shape = (R.shape[0],) + A.shape[1:]
    if out is None:
        out = np.empty(shape, dtype=A.dtype)
    if out.ndim < 3:
        out = out[None, :]

    out.shape = (out.shape[0], -1)
    A.shape = (A.shape[0], -1)
    if inverse:
        phase = np.exp(-2.0j * np.pi * np.dot(kpts, R.T))
    else:
        phase = np.exp(2.0j * np.pi * np.dot(R, kpts.T)) / kpts.shape[0]
    np.sum(phase[..., None] * A[None, ...], axis=1, out=out)
    A.shape = (kpts.shape[0],) + shape[1:]
    out.shape = shape

    if squeeze:
        return out[0]
    return out


def partial_bloch_sum(A, kpts, R, kpts_p, kpts_t, out=None):
    """Partial Bloch sum along parallel direction.

    Args:
        A : (np.ndarray) matrices in k-space.
        kpts : (2D) k-points at which A's have been evaluated.
        R : (1D or 2D) lattice vectors at which A's will be transformed.
        kpts_p : parallel k-points
        kpts_t : transverse k-points
    """
    distance = partial(np.linalg.norm, axis=1)

    R = np.asarray(R)
    squeeze = False
    if R.ndim < 2:
        squeeze = True
    R = np.array(R, ndmin=2)
    if out is None:
        out = np.empty((kpts_t.shape[0], R.shape[0]) + A.shape[1:], A.dtype)

    for j, kpt_t in enumerate(kpts_t):
        _kpts = kpts_p + kpt_t
        A_t = []
        for k in _kpts:
            try:
                i = np.where(distance(k - kpts) < 1e-7)[0][0]
                A_t.append(A[i])
            except IndexError:
                # kpts are irreducible representation (ikpts).
                # look for time-reversed k-point k = -k + N.
                i = np.where(distance(((-k) % 1) - kpts % 1) < 1e-7)[0][0]
                A_t.append(A[i].T)
        fourier_sum(np.array(A_t), _kpts, R, out[j])

    if squeeze:
        return out[:, 0, :, :]
    return out


def build_lattice_vectors(size):
    """Lattice vectors in scaled coordinates."""
    return np.indices(size).reshape(len(size), -1).T


def unfold_kpts(Kpts, kpts):
    """Unfold coarse mesh Kpts into fine mesh kpoints 
    sampled with Monkhorst-Pack mesh kpts_size.

    Args:
        Kpts : (np.ndarray, ndimn=2)
            The coarse sampled k-points.
        kpts : (np.ndarray, ndimn=2)
            The fine sampled k-points
        kpts_size : the size of the fine kpts mesh.

    Returns:
        k_groups : The groups of kpoints indices that must be
            unfolded to fold into Kpts.
    """
    distance = partial(np.linalg.norm, axis=1)
    Ksize, Koffset = get_monkhorst_pack_size_and_offset(Kpts)
    ksize, koffset = get_monkhorst_pack_size_and_offset(kpts)

    # assert np.allclose(Koffset, koffset), f'Incompatible mesh grids.\
    # They have different offsets {Koffset} vs. {koffset}. \
    #    Force including the gamma point to match the offsets.'

    Nr = ksize // Ksize
    assert np.allclose(
        Nr, Nr.astype(int)
    ), f"\
Incompatible mesh grids. Fine mesh with size {ksize} is \
not a multiple of coarse mesh with size {Ksize}."

    k_groups = []
    for K in Kpts:
        group_indices = []
        for i, j, k in np.ndindex(tuple(Nr)):
            kpt = (K % 1 + (i, j, k)) / Nr
            idx = np.where(distance(kpt - kpts % 1) < 1e-7)[0][0]
            group_indices.append(idx)
        k_groups.append(group_indices)

    return k_groups
