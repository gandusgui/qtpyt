from itertools import chain
from typing import Any

import numpy as np
from numba import njit, prange
from qtpyt import parallel
from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.base.leads import LeadSelfEnergy, SanchoRubio
from qtpyt.base.selfenergy import BaseSelfEnergy
from qtpyt.basis import Basis
from qtpyt.surface.expand import seminf_expand, tritoeplz_expand

# from ._recursive import get_G
from .unfold import bloch_unfold


class PrincipalLayer(BaseGreenFunction):
    """Principal Layer (PL) Green's function.
    
    Allows to compute a surface Green's Function for a supercell (SC)
    composed of a repeating principal layer (PL). The inputs are
    converged Hamiltonians and Overlaps for the PL evaluated at 
    tranverse k-points.
    
    """

    def __init__(self, kpts, hs_kii, hs_kij, Nr=None, eta=1e-5):
        """
        Args:
            kpts : np.ndarray (shape = (# of k-points, 3))
                tranverse k-points.
            hs_kii : (tuple, list)
                pointers to onsite Hamiltonians and Overlaps (at kpts).
            hs_kij : (tuple, list)
                pointers to coupling Hamiltonians and Overlaps (at kpts).
            Nr : tuple (size = (3,))
                # of transverse PLs. Can be smaller than the actual
                # of transverse k-points!
        
        NOTE: If Nr is not provided, the number of real space repetitions
        is deduced by the unique k-point values in each dimension. It can
        be useful to sample the BZ with a k-mesh bigger than the realspace
        supercell size.
        """

        self.kpts = kpts
        self.Nr = Nr or self.N_k

        assert (
            self.kpts.shape[1] == 3
        ), "Invalid k-points. Must have (x,y,z) components."
        assert len(self.Nr) == 3, "Invalid # of transverse PLs. Must be (Nx, Ny, Nz)."

        H, S = (self.bloch_unfold(m) for m in hs_kii)

        self.PL = [
            SanchoRubio((hs_kii[0][k], hs_kii[1][k]), (hs_kij[0][k], hs_kij[1][k]), eta)
            for k in range(self.Nk)
        ]

        super().__init__(H, S, eta)

        self.G_kii = np.empty((kpts.shape[0],) + self.PL[0].shape, complex)
        self.G = np.empty(self.shape, complex)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for PL in self.PL:
                setattr(PL, name, value)
        super().__setattr__(name, value)

    @property
    def N_k(self):
        """# of transverse k-points"""
        return np.array([len(set(self.kpts[:, i])) for i in range(3)])

    @property
    def Nk(self):
        """# of parallel repetitions."""
        return self.kpts.shape[0]

    def retarded(self, energy):
        """Get retarded Green function at specified energy."""
        if energy != self.energy:
            self.energy = energy
            for k in range(self.Nk):
                self.G_kii[k] = self.PL[k].retarded(energy)
            self.bloch_unfold(self.G_kii, self.G)
        return self.G

    def bloch_unfold(self, A, out=None):
        """Unfold to supercell matrix.
        
        Args:
            A : Matrices at transverse k-points.
            out : (optional) supercell matrix 
        """
        return bloch_unfold(A, self.kpts, self.Nr, out)


class PrincipalSelfEnergy(BaseSelfEnergy):
    """Principal Layer (PL) Self-energy.
    
    Allows to compute a surface self-energy for a supercell (SC)
    composed of a repeating principal layer (PL). The inputs are
    converged Hamiltonians and Overlaps for the PL evaluated at 
    tranverse k-points.
    

    TODO :: order return!
    """

    ids = ["left", "right"]

    def __init__(self, kpts, hs_kii, hs_kij, Nr=None, id="left", eta=1e-5):

        assert id in self.ids, f"Invalid id. Choose between {self.ids}"

        if id == "right":
            h_kij = hs_kij[0].swapaxes(1, 2).conj()
            s_kij = hs_kij[1].swapaxes(1, 2).conj()
            hs_kij = (h_kij, s_kij)

        self.gf = PrincipalLayer(kpts, hs_kii, hs_kij, Nr, eta)

        hs_ij = tuple(bloch_unfold(m, self.gf.kpts, self.gf.Nr) for m in hs_kij)

        super().__init__(self.gf, hs_ij)


class SurfaceGreenFunction(BaseGreenFunction):
    """The surface Green's function."""

    def __init__(self, kpts, hs_kii, hs_kij, Nr, direction="x", eta=1e-5):

        self.Nr = Nr
        self.kpts = kpts

        self.d = "xyz".index(direction)
        self.Nd = Nr[self.d]  # num. PLs along transport
        self.Nr_t = tuple(
            n if i != self.d else 1 for i, n in enumerate(Nr)
        )  # num. PLs along transverse

        hs_kji = tuple(m.swapaxes(1, 2).conj() for m in hs_kij)

        sel = [
            LeadSelfEnergy((hs_kii[0][k], hs_kii[1][k]), (hs_kij[0][k], hs_kij[1][k]), eta=eta)
            for k in range(self.Nk)
        ]
        
        ser = [
            LeadSelfEnergy((hs_kii[0][k], hs_kii[1][k]), (hs_kji[0][k], hs_kji[1][k]), eta=eta)
            for k in range(self.Nk)
        ]

        self.selfenergies = [sel, ser]
        
        self.work_kMM = np.zeros(
            (self.Nk,) + 2 * (hs_kii[0].shape[1] * self.Nd,), complex
        )  # Internal work array

        H, S = (
            self.unfold_periodic(A, B, C) for A, B, C in zip(hs_kii, hs_kij, hs_kji)
        )

        super().__init__(H, S, eta)

        self.G = np.zeros(self.shape, complex)
        # Store reference for quick leads extraction
        self.hs_kii = hs_kii
        self.hs_kij = hs_kij
        self.hs_kji = hs_kji

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for PL in chain(*self.selfenergies):
                setattr(PL, name, value)
        super().__setattr__(name, value)

    @property
    def Nk(self):
        """# of parallel repetitions."""
        return self.kpts.shape[0]

    def bloch_unfold(self, A, out=None):
        """Unfold k-points to supercell matrix.
        
        Args:
            A : Matrices at transverse k-points.
            out : (optional) supercell matrix 
        """
        return bloch_unfold(A, self.kpts, self.Nr_t, out)

    def unfold_periodic(self, A, B, C, out=None):
        """Expand tridiagonal toeplitz and bloch unfold.
        
        """
        for k in range(self.Nk):
            tritoeplz_expand(self.Nd, A[k], B[k], C[k], self.work_kMM[k])
        return self.bloch_unfold(self.work_kMM, out)

    def retarded(self, energy):
        """The retarded green's function.
        
        """
        if energy != self.energy:
            self.energy = energy
            z = self.z(energy)
            for k in range(self.Nk):
                seminf_expand(
                    self.Nd,
                    (self.hs_kii[0][k], self.hs_kii[1][k]),
                    (self.hs_kij[0][k], self.hs_kij[1][k]),
                    self.selfenergies[0][k].retarded(energy),
                    self.selfenergies[1][k].retarded(energy),
                    z,
                    (self.hs_kji[0][k], self.hs_kji[1][k]),
                    self.work_kMM[k],
                )
            self.bloch_unfold(self.work_kMM, self.G)
        return self.G


@njit(
    ["(c16,c16[:,::1],c16[:,::1],c16[:,::1],c16[:,::1],i8[::1])"], parallel=True,
)
def _fill_sigma_subset(z, H, S, Ginv, Sigma, indices):
    n = len(indices)
    for i in prange(n):
        for j in range(n):
            Sigma[i, j] = (
                z * S[indices[i], indices[j]]
                - H[indices[i], indices[j]]
                - Ginv[indices[i], indices[j]]
            )


@njit(["(c16,c16[:,::1],c16[:,::1],c16[:,::1],c16[:,::1])"], parallel=True)
def _fill_sigma_full(z, H, S, Ginv, Sigma):
    n, m = H.shape
    for i in prange(n):
        for j in prange(m):
            Sigma[i, j] = z * S[i, j] - H[i, j] - Ginv[i, j]


def fill_sigma(z, H, S, Ginv, Sigma=None, indices=None):
    """Solve Dyson equation for boundary self-energy."""
    if Sigma is None:
        Sigma = np.empty_like(Ginv)
    if indices is None:
        _fill_sigma_full(z, H, S, Ginv, Sigma)
    else:
        _fill_sigma_subset(z, H, S, Ginv, Sigma, indices)
    return Sigma


class SurfaceSelfEnergy(BaseSelfEnergy):
    """The surface selfenergy."""

    def __init__(
        self, kpts, hs_kii, hs_kij, Nr, indices=None, direction="x", eta=1e-5
    ) -> None:

        self.gf = SurfaceGreenFunction(kpts, hs_kii, hs_kij, Nr, direction, eta)
        super().__init__(self.gf, (np.empty((0, 0), complex), None))

        self.indices = indices

    @property
    def indices(self):
        """Set subset of orbitals for the self-energy.
        
        Useful when coupling to a structure with missing
        surface atoms, e.g. defects.
        """
        return self._indices

    @indices.setter
    def indices(self, indices):
        if indices is None:
            shape = self.gf.shape
        else:
            shape = 2 * (len(indices),)
        self.Sigma = np.empty(shape, complex)
        self._indices = indices

    def take(self, indices):
        """Take subset of basis function for self-energy."""
        self.indices = indices

    def retarded(self, energy):
        """The retarded selfenergy."""
        if energy != self.energy:
            self.energy = energy
            Ginv = np.linalg.inv(self.gf.retarded(energy))
            fill_sigma(self.z(energy), self.H, self.S, Ginv, self.Sigma, self.indices)
        return self.Sigma


def extract_leads_from_surface(sf: SurfaceSelfEnergy, bpl: Basis, bcc: Basis):
    """Extract left and right leads of SurfaceSelfEnergy.
    
    """
    from scipy.spatial.distance import cdist

    gf = sf.gf
    SL = PrincipalSelfEnergy(gf.kpts, gf.hs_kii, gf.hs_kij, gf.Nr_t, eta=gf.eta)
    SR = PrincipalSelfEnergy(gf.kpts, gf.hs_kii, gf.hs_kji, gf.Nr_t, eta=gf.eta)

    # Override principal layers.
    SL.gf.PL = [se.gf for se in gf.selfenergies[0]]
    SR.gf.PL = [se.gf for se in gf.selfenergies[1]]

    # Find position of left/right atoms in surface.
    bsc = bpl.repeat(gf.Nr_t)
    sc_pos = bsc.atoms.positions
    l_in_c = cdist(sc_pos, bcc.atoms.positions).argmin(1)
    # Shift to the right.
    sc_pos[:, gf.d] += (gf.Nd - 1) * bpl.atoms.cell.diagonal()[gf.d]
    r_in_c = cdist(sc_pos, bcc.atoms.positions).argmin(1)
    # Rearrange couplings.
    for in_c, selfenergy in zip([l_in_c, r_in_c], [SL, SR]):
        perm = bcc[in_c].get_indices()
        h_ij = np.zeros((bsc.nao, bcc.nao), complex)
        s_ij = np.zeros((bsc.nao, bcc.nao), complex)
        h_ij[:, perm] = selfenergy.h_ij
        s_ij[:, perm] = selfenergy.s_ij
        selfenergy.h_ij = h_ij
        selfenergy.s_ij = s_ij
        selfenergy.Sigma = np.zeros(2 * (bcc.nao,), complex)
    return SL, SR

