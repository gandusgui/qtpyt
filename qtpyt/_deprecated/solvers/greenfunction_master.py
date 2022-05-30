import numpy as np
from functools import lru_cache
from .coupledhamiltonian import CoupledHamiltonian
from .block_tridiag import tridiagonalize, cutoff
from .recursive_gf import recursive_gf


class GreenFunction(CoupledHamiltonian):
    """Equilibrium retarded Green function."""

    def __init__(self, H, S=None, selfenergies=[], eta=1e-4):
        self.H = H
        self.S = S
        self.selfenergies = selfenergies
        self.eta = eta
        self.energy = None
        self.Ginv = np.empty(H.shape, complex)
        self.is_recursive = False

    def set_tridiag(self, calc, pl1, pl2=None, cutoff=cutoff):

        self.calc = calc
        self.hs_list_ii, self.hs_list_ij = tridiagonalize(
            calc, self.H, self.S, pl1, pl2, cutoff
        )

        for selfenergy in self.selfenergies:
            selfenergy.set_hs_im(selfenergy.nbf_i, selfenergy.nbf_i)

        self.g_1N = None
        self.is_recursive = True

    # @lru_cache(maxsize=None)
    def retarded(self, energy, inverse=False):
        """Get retarded Green function at specified energy.

        If 'inverse' is True, the inverse Green function is returned (faster).
        """
        if energy != self.energy:
            self.energy = energy
            z = energy + self.eta * 1.0j

            if self.S is None:
                self.Ginv[:] = 0.0
                self.Ginv.flat[:: len(self.S) + 1] = z
            else:
                self.Ginv[:] = z
                self.Ginv *= self.S
            self.Ginv -= self.H

            for selfenergy in self.selfenergies:
                self.Ginv -= selfenergy.retarded(energy)

        if inverse:
            return self.Ginv
        else:
            return np.linalg.inv(self.Ginv)

    def recursive(self, energy):

        if energy != self.energy:
            self.energy = energy
            z = energy + self.eta * 1.0j

            mat_list_ii = []
            mat_list_ij = []
            mat_list_ji = []

            h_list_ij, s_list_ij = self.hs_list_ij
            for h_ij, s_ij in zip(h_list_ij, s_list_ij):

                mat_list_ij.append(z * s_ij - h_ij)
                mat_list_ji.append(z * s_ij.T.conj() - h_ij.T.conj())

            h_list_ii, s_list_ii = self.hs_list_ii
            for h_ii, s_ii in zip(h_list_ii, s_list_ii):
                mat_list_ii.append(z * s_ii - h_ii)

            # from IPython import embed
            # embed()
            mat_list_ii[0] -= self.selfenergies[0].retarded(energy)
            mat_list_ii[-1] -= self.selfenergies[1].retarded(energy)

            self.g_1N = recursive_gf(mat_list_ii, mat_list_ij, mat_list_ji)

        return self.g_1N

    def calculate(self, energy, sigma):
        """XXX is this really needed"""
        ginv = energy * self.S - self.H - sigma
        return np.linalg.inv(ginv)

    def apply_retarded(self, energy, X):
        """Apply retarded Green function to X.

        Returns the matrix product G^r(e) . X
        """
        return np.linalg.solve(self.retarded(energy, inverse=True), X)

    def dos(self, energy):
        """Total density of states -1/pi Im(Tr(GS))"""
        if self.S is None:
            return -self.retarded(energy).imag.trace() / np.pi
        else:
            GS = self.apply_retarded(energy, self.S)
            return -GS.imag.trace() / np.pi

    def pdos(self, energy):
        """Projected density of states -1/pi Im(SGS/S)"""
        if self.S is None:
            return -self.retarded(energy).imag.diagonal() / np.pi
        else:
            S = self.S
            SGS = np.dot(S, self.apply_retarded(energy, S))
            return -(SGS.diagonal() / S.diagonal()).imag / np.pi

    def take_bfs(self, bfs, apply):
        h_pp, s_pp, c_mm = CoupledHamiltonian.take_bfs(self, bfs, apply)
        if apply:
            self.Ginv = np.empty(self.H.shape, complex)
        return h_pp, s_pp, c_mm
