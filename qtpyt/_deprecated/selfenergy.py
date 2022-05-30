from functools import lru_cache

import numpy as np
from scipy import linalg as la

from .internalselfenergy import InternalSelfEnergy
from .tools import get_subspace, rotate_matrix

# from gpaw.utilities.blas import gemm
# import _cppmodule as _cpp

# class LeadSelfEnergy(_cpp.LeadSelfEnergy, InternalSelfEnergy):
class LeadSelfEnergy(InternalSelfEnergy):
    conv = 1e-8  # Convergence criteria for surface Green function

    def __init__(self, hs_dii, hs_dij, hs_dim, eta=1e-4):
        self.h_ij, self.s_ij = hs_dij  # coupling between principal layers
        # self.bias = 0
        InternalSelfEnergy.__init__(self, hs_dii, hs_dim, eta=eta)

    def set_bias(self, bias):
        self.bias = bias

    def get_lambda(self, energy):
        """Return the lambda (aka Gamma) defined by i(S-S^d).

        Here S is the retarded selfenergy, and d denotes the hermitian
        conjugate.
        """
        Sigma = self.retarded(energy)
        return 1.0j * (Sigma - Sigma.T.conj())

    # # @lru_cache(maxsize=None)
    def get_Ginv(self, energy):
        # """The inverse of the retarded surface Green function"""
        z = energy - self.bias + self.eta * 1.0j

        v_00 = z * self.s_ii.T.conj() - self.h_ii.T.conj()
        v_11 = v_00.copy()
        v_10 = z * self.s_ij - self.h_ij
        v_01 = z * self.s_ij.T.conj() - self.h_ij.T.conj()

        delta = self.conv + 1
        while delta > self.conv:
            lu, piv = la.lu_factor(v_11)
            a = la.lu_solve((lu, piv), v_01)
            b = la.lu_solve((lu, piv), v_10)
            v_01_dot_b = np.dot(v_01, b)
            v_00 -= v_01_dot_b
            v_11 -= np.dot(v_10, a)
            v_11 -= v_01_dot_b
            v_01 = -np.dot(v_01, a)
            v_10 = -np.dot(v_10, b)
            delta = abs(v_01).max()

        return v_00

    def apply_rotation(self, c_mm):
        # self.get_sgfinv.cache_clear()
        self.h_ii = self.h_ii.copy()  # needed because self[0].h_ii is self[1].h_ii
        self.s_ii = self.s_ii.copy()  # as self[0].h_ii -> H1[:nprinc,:nprinc]
        InternalSelfEnergy.apply_rotation(self, c_mm)
        self.h_ij = rotate_matrix(self.h_ij, c_mm)
        self.s_ij = rotate_matrix(self.s_ij, c_mm)

    def cutcoupling_bfs(self, bfs, apply=False):
        # self.get_sgfinv.cache_clear()
        h_pp, s_pp = InternalSelfEnergy.cutcoupling_bfs(self, bfs, apply)
        if apply:
            for m in bfs:
                self.h_ij[m, :] = 0.0
                self.s_ij[m, :] = 0.0
        return h_pp, s_pp

    def take_bfs(self, bfs, apply=False):
        # self.get_sgfinv.cache_clear()
        h_pp, s_pp, c_mm = InternalSelfEnergy.take_bfs(self, bfs, apply)
        if apply:
            self.h_ij = rotate_matrix(
                self.h_ij, c_mm
            )  # np.dot(c_mm.T.conj(), self.h_ij)
            self.s_ij = rotate_matrix(
                self.s_ij, c_mm
            )  # np.dot(c_mm.T.conj(), self.s_ij)
        return h_pp, s_pp


class BoxProbe:
    """Box shaped Buttinger probe.

    Kramers-kroning: real = H(imag); imag = -H(real)
    """

    def __init__(self, eta, a, b, energies, S, T=0.3):
        from qtpyt.Hilbert import hilbert

        se = np.empty(len(energies), complex)
        se.imag = 0.5 * (
            np.tanh(0.5 * (energies - a) / T) - np.tanh(0.5 * (energies - b) / T)
        )
        se.real = hilbert(se.imag)
        se.imag -= 1
        self.selfenergy_e = eta * se
        self.energies = energies
        self.S = S

    def retarded(self, energy):
        return self.selfenergy_e[self.energies.searchsorted(energy)] * self.S
