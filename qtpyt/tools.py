import numpy as np

from qtpyt import xp
from qtpyt.base.selfenergy import BaseSelfEnergy


def remove_pbc(basis, A_kMM, direction="x", eps=-1e-3):
    """Mask matrix elements above diagonal along `direction`.

    """

    # Transport direction
    p_dir = "xyz".index(direction)

    L = basis.atoms.cell[p_dir, p_dir]

    cutoff = L / 2 - eps
    # Coordinates of central unit cell i (along transport)
    centers_p_i = basis.centers[:, p_dir]
    # Coordinates of neighbooring unit cell j
    # centers_p_j = centers_p_i + L
    # Distance between j atoms and i atoms
    dist_p_ji = np.abs(centers_p_i[:, None] - centers_p_i[None, :])
    # Mask j atoms farther than L
    mask_ji = (dist_p_ji < cutoff).astype(A_kMM.dtype)

    A_kMM *= mask_ji[None, :]


def expand_coupling(selfenergy: BaseSelfEnergy, nbf_m, id="left"):
    """Expand selfenergy's coupling matrices to 
    match dimensions of green's function region.

    Args:
        selfenergy : any selfenergy on CPU or GPU.
        nbf_m : number of basis of green's function's region.
        id : 'left' for downfold or 'right' for upfold.
    """
    if type(selfenergy.H).__module__ == "numpy":
        # Force numpy array for principallayer
        # which is only implemented for CPU
        _xp = np
    elif type(selfenergy.H).__module__ == "cupy":
        _xp = xp
    else:
        raise RuntimeError("Array module not recognised.")
    nbf_i = selfenergy.h_ij.shape[0]
    h_ij = _xp.zeros((nbf_i, nbf_m), complex)
    s_ij = _xp.zeros_like(h_ij)
    if id == "left":
        h_ij[:, :nbf_i] = selfenergy.h_ij
        s_ij[:, :nbf_i] = selfenergy.s_ij
    else:
        h_ij[:, -nbf_i:] = selfenergy.h_ij
        s_ij[:, -nbf_i:] = selfenergy.s_ij
    selfenergy.h_ij = h_ij
    selfenergy.s_ij = s_ij
    selfenergy.Sigma = _xp.zeros((nbf_m, nbf_m), complex)


def rotate_couplings(basis, selfenergy, N=(1, 1, 1), order="xyz", positions=None):
    """Rotate couplings inplace to match order of the adjoining atoms
    in a neighbor region.
    
    Args:
        basis : (Basis object)
            basis describing the selfenergy's atoms. Can be a subset
            of total atoms, e.g. a PL.
        N : (tuple,list)
            number of repetitions if `basis` describes a PL.
        order : (str)
            any permutation of 'x','y','z'. Order of adjoining atoms.
        positions : (np.ndarray)
            positions of adjoining atoms.
    """
    from qtpyt.surface.tools import order_indices

    permutation = order_indices(basis, N, order, positions)

    h_ij = selfenergy.h_ij
    s_ij = selfenergy.s_ij
    h_ij.take(permutation, axis=1, out=h_ij)
    s_ij.take(permutation, axis=1, out=s_ij)


def tri2full(H_nn, UL="L", map=np.conj):
    """Fill in values of hermitian or symmetric matrix.

    Fill values in lower or upper triangle of H_nn based on the opposite
    triangle, such that the resulting matrix is symmetric/hermitian.

    UL='U' will copy (conjugated) values from upper triangle into the
    lower triangle.

    UL='L' will copy (conjugated) values from lower triangle into the
    upper triangle.

    The map parameter can be used to specify a different operation than
    conjugation, which should work on 1D arrays.  Example::

      def antihermitian(src, dst):
            np.conj(-src, dst)

      tri2full(H_nn, map=antihermitian)

    """
    N, tmp = H_nn.shape
    assert N == tmp, "Matrix must be square"
    if UL != "L":
        H_nn = H_nn.T

    for n in range(N - 1):
        map(H_nn[n + 1 :, n], H_nn[n, n + 1 :])
