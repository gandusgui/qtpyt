"""Simple tight-binding add-on for GPAWs LCAO module."""

from math import pi

import numpy as np
import scipy.linalg as sla

from qtpyt.surface.kpts import (
    apply_inversion_symm,
    expand_inversion_symm,
    fourier_sum,
    get_monkhorst_pack_size_and_offset,
)


def bloch_to_realspace(A, kpts, R, out=None):
    """K-space Fourier sum.
    
    The main difference w.r.t. fourier_sum from qtpyt.surface.kpts
    is that it accetps A's k-matrices which are evaluated in the irreducible
    BZ. In this case, kpts MUST be a reducible set of kpts.

    Args:
        A : (np.ndarray) matrices in k-space.
        kpts : (1D or 2D) k-points at which A's have been evaluated.
        R : (1D or 2D) lattice vectors at which A's will be transformed.
    """
    size, offset = get_monkhorst_pack_size_and_offset(kpts)
    if not np.allclose(offset, (0, 0, 0)):
        # TODO :: In this case one cannot simply out.conj()
        raise NotImplementedError

    kpts = np.atleast_2d(kpts)
    R = np.atleast_2d(R)
    # Irreducible brillouin zone
    ibz = False
    if kpts.shape[0] > A.shape[0]:
        # A's are at IBZ
        Nk = [len(set(kpts[:, i])) for i in range(3)]
        kpts = apply_inversion_symm(kpts)[0]
        ibz = True
    out = fourier_sum(A, kpts, R, out)
    if ibz:
        out *= kpts.shape[0]
        out += out.conj()
        try:
            gamma = np.where(np.sum(kpts, axis=1) < 1e-9)[0][0]
        except IndexError:
            pass
        else:
            #       ik          -ik
            # A(k) e    + A(-k) e
            #                                 T    *
            # NOTE: A(k) = A(-k)  & A(k) = ( A(k) )
            #
            #         ik   *     *   -ik       T    * -ik          -ik
            # ( A(k) e    )  =  A(k) e   = ( A(-k) )  e    = A(-k) e
            out -= A[gamma]
        out /= np.product(Nk)
    return out


class TightBinding:
    """Simple class for tight-binding calculations.
    
    Example:
        kpts = monkhorst-pack(size)
        h_k :: Hamilton evaulated at IBZ
        s_k :: Overlap evaulated at IBZ
        tb = TightBinding(kpts, h_k, s_k)
    """

    def __init__(self, kpts, H_kMM, S_kMM):
        """
        Args:
            kpts : (1D or 2D) 
                k-points at which A's have been evaluated.
            H_kMM, S_kMM : (np.ndarray) 
                Hamiltonian and overlap matrices in k-space.
        """
        if H_kMM.shape[0] < kpts.shape[0]:
            ks, H_kMM = expand_inversion_symm(kpts, H_kMM)
            assert np.allclose(ks, kpts), "Invalid k-points"
            S_kMM = expand_inversion_symm(kpts, S_kMM)[1]

        self.kpts = kpts
        # here we have set kpts to the reducible representation,
        # so Nk is correctly computed when called from set_num_cells.
        self.set_num_cells()
        self.H_NMM = fourier_sum(H_kMM, self.kpts, self.R_cN.T)
        self.S_NMM = fourier_sum(S_kMM, self.kpts, self.R_cN.T)

    @property
    def Nk(self):
        """# of transverse k-points"""
        return np.array([len(set(self.kpts[:, i])) for i in range(3)])

    def set_num_cells(self, N_c=None):
        """Set number of real-space cells to use.

        Parameters
        ----------
        N_c: tuple or ndarray
            Number of unit cells in each direction of the basis vectors.

        """

        if N_c is None:
            self.N_c = tuple(self.Nk)
        else:
            self.N_c = tuple(N_c)
        try:
            if np.any(np.asarray(self.Nk) < np.asarray(self.N_c)):
                print("WARNING: insufficient k-point sampling.")
        except AttributeError:
            # Nk is not defined
            pass
        # Lattice vectors
        R_cN = np.indices(self.N_c).reshape(3, -1)
        N_c = np.array(self.N_c)[:, np.newaxis]
        R_cN += N_c // 2
        R_cN %= N_c
        R_cN -= N_c // 2
        self.R_cN = R_cN

    def band_structure(self, path_kc, blochstates=False):
        """Calculate dispersion along a path in the Brillouin zone.

        Parameters
        ----------
        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        blochstates: bool
            Return LCAO expansion coefficients when True (default: False).

        """
        R_cN = self.R_cN
        # Lists for eigenvalues and eigenvectors along path
        eps_kn = []
        psi_kn = []

        for k_c in path_kc:
            # Evaluate fourier sum
            phase_N = np.exp(-2.0j * pi * np.dot(k_c, R_cN))
            H_MM = np.sum(phase_N[:, np.newaxis, np.newaxis] * self.H_NMM, axis=0)
            S_MM = np.sum(phase_N[:, np.newaxis, np.newaxis] * self.S_NMM, axis=0)

            if blochstates:
                eps_n, c_Mn = sla.eigh(H_MM, S_MM)
                # Sort eigenmodes according to increasing eigenvalues
                c_nM = c_Mn[:, eps_n.argsort()].transpose()
                psi_kn.append(c_nM)
            else:
                eps_n = sla.eigvalsh(H_MM, S_MM)

            # Sort eigenvalues in increasing order
            eps_n.sort()
            eps_kn.append(eps_n)

        # Convert to eV
        eps_kn = np.array(eps_kn)  # * units.Hartree

        if blochstates:
            return eps_kn, np.array(psi_kn)

        return eps_kn
