import numpy as np
from numpy import linalg
from scipy.sparse import csr_matrix

# from functools import lru_cache
from .coupledhamiltonian import CoupledHamiltonian
from .tools import dagger

# Recursive GreenFunction helpers
from .solvers.tridiagonal import tridiagonalize, cutoff
from .solvers.recursive import *  # get_mat_lists, recursive_gf, multiply

# Density integral
from .continued_fraction import integrate_pdos
from .tk_gpaw import sum_bf_atom

# GPU
# import cupy as xp
from transport import xp


class GreenFunction(CoupledHamiltonian):
    """Equilibrium retarded Green function."""

    def __init__(self, H, S=None, selfenergies=[], **kwargs):
        self.H = H
        self.S = S
        self.selfenergies = selfenergies
        # self.eta = eta
        self.energy = None
        self.Ginv = np.empty(H.shape, complex)

        self.parameters = {"eta": 1e-5, "align_bf": None}
        self.initialized = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key in ["eta", "align_bf"]:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError("%r not a vaild keyword" % key)

        self.parameters.update(kwargs)

    def initialize(self):  # , calc, pl1, pl2=None, cutoff=cutoff, align_bf=None):
        """
            calc: scattering calculator
            pl1: # of bfs left principal layer
            pl2: # of bfs right principal layer (default to pl1)
            cutoff: cutoff for block neighbor interaction
        """

        if self.initialized:
            return

        p = self.parameters
        align_bf = p["align_bf"]

        # Set eta
        self.eta = p["eta"]

        # Align first basis funciton for Left lead
        if align_bf is not None:
            self.align_bf(align_bf)

        self.initialized = True

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

    def calculate(self, energy, sigma):
        """XXX is this really needed"""
        ginv = energy * self.S - self.H - sigma
        return np.linalg.inv(ginv)

    def apply_retarded(self, energy, X):
        """Apply retarded Green function to X.

        Returns the matrix product G^r(e) . X
        """
        return np.linalg.solve(self.retarded(energy, inverse=True), X)

    def apply_overlap(self, energy, trace=False, diag=False):
        """Apply retarded Green function to S."""
        GS = self.apply_retarded(energy, self.S)
        if trace:
            return np.trace(GS)
        if diag:
            return GS.diagonal()
        return GS

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

    def get_spectrals(self, energy):
        """Specral density of states incoming from leads."""
        G_mm = self.retarded(energy)
        lambda1_mm = self.selfenergies[0].get_lambda(energy)
        lambda2_mm = self.selfenergies[1].get_lambda(energy)
        A1_mm = G_mm.dot(lambda1_mm).dot(dagger(G_mm))
        A2_mm = G_mm.dot(lambda2_mm).dot(dagger(G_mm))

        return A1_mm, A2_mm

    def get_transmission(self, energies, T_e=None):

        if T_e is None:
            T_e = np.zeros(len(energies))

        for e, energy in enumerate(energies):
            Ginv_mm = self.retarded(energy, inverse=True)
            lambda1_mm = self.selfenergies[0].get_lambda(energy)
            lambda2_mm = self.selfenergies[1].get_lambda(energy)
            a_mm = linalg.solve(Ginv_mm, lambda1_mm)
            b_mm = linalg.solve(dagger(Ginv_mm), lambda2_mm)
            T_e[e] = np.trace(np.dot(a_mm, b_mm)).real

        return T_e


class RecursiveGF(CoupledHamiltonian):
    def __init__(self, H=None, S=None, selfenergies=[], **kwargs):

        super().__init__(H, S, selfenergies)

        self.energy = None
        self.g1N = None

        self.parameters = {
            "eta": 1e-5,
            "calc": None,
            "pl1": None,
            "pl2": None,
            "cutoff": cutoff,
            "align_bf": None,
            "H": None,
            "S": None,
            "gpu": False,
        }

        self.initialized = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key in [
                "eta",
                "calc",
                "pl1",
                "pl2",
                "cutoff",
                "align_bf",
                "H",
                "S",
                "gpu",
            ]:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError("%r not a vaild keyword" % key)

        self.parameters.update(kwargs)

    def initialize(
        self, hs_list_ii=None, hs_list_ij=None
    ):  # , calc, pl1, pl2=None, cutoff=cutoff, align_bf=None):
        """
            calc: scattering calculator
            pl1: # of bfs left principal layer
            pl2: # of bfs right principal layer (default to pl1)
            cutoff: cutoff for block neighbor interaction
        """
        if self.initialized:
            return

        p = self.parameters
        align_bf = p["align_bf"]
        calc = p["calc"]
        pl1 = p["pl1"]
        pl2 = p["pl2"]
        cutoff = p["cutoff"]
        H = p["H"]
        S = p["S"]

        if pl1 is None:
            pl1 = self.selfenergies[0].natoms
        if pl2 is None:
            pl2 = pl1

        # Set eta
        self.eta = p["eta"]

        if hs_list_ii is None:
            # Construct block tridiagonal lists
            hs_list_ii, hs_list_ij = tridiagonalize(calc, H, S, pl1, pl2, cutoff)
        # Offload to GPU
        if p["gpu"] is True:
            hs_list_ii = [
                [xp.asarray(h) for h in hs_list_ii[0]],
                [xp.asarray(s) for s in hs_list_ii[1]],
            ]
            hs_list_ij = [
                [xp.asarray(h) for h in hs_list_ij[0]],
                [xp.asarray(s) for s in hs_list_ij[1]],
            ]
        else:
            from .solvers import recursive

            recursive.xp = np
        # else:
        self.hs_list_ii, self.hs_list_ij = list(hs_list_ii), list(hs_list_ij)

        # self.truncate(hs_list_ii)
        # self.truncate(hs_list_ij)

        # Align first basis funciton for Left lead
        if align_bf is not None:
            self.align_bf(align_bf)

        self.hs_list_ji = [
            [h.T for h in self.hs_list_ij[0]],
            [s.T for s in self.hs_list_ij[1]],
        ]

        # self.dense2csr(self.hs_list_ii)
        # self.dense2csr(self.hs_list_ij)
        # self.dense2csr(self.hs_list_ji)

        # Note h_im[:,:pl1]=h_ij for Left and h_im[:,-pl2:]=h_ij for Right
        # for selfenergy in self.selfenergies:
        #    selfenergy.h_im = selfenergy.h_ij
        #    selfenergy.s_im = selfenergy.s_ij
        #    selfenergy.sigma_mm = np.empty((selfenergy.nbf_i, selfenergy.nbf_i),
        #                                    dtype=selfenergy.sigma_mm.dtype)

        self.nbf = sum(h.shape[0] for h in self.hs_list_ii[0])
        self.N = len(self.hs_list_ii[0])
        self.initialized = True

    @staticmethod
    def truncate(hs_list, cutoff=cutoff):
        conds = [abs(h) < cutoff for h in hs_list[0]]
        for h, cond in zip(hs_list[0], conds):
            h[cond] = 0
        for s, cond in zip(hs_list[1], conds):
            s[cond] = 0

    @staticmethod
    def dense2csr(hs_list):
        hs_list[0] = [csr_matrix(mat) for mat in hs_list[0]]
        hs_list[1] = [csr_matrix(mat) for mat in hs_list[1]]

    def align_bf(self, bf):
        h1 = self.selfenergies[0].h_ii
        h_qii, s_qii = self.hs_list_ii
        h_qij, s_qij = self.hs_list_ij
        n = len(h_qii)
        diff = (h_qii[0][bf, bf] - h1[bf, bf].real) / s_qii[0][bf, bf]
        for q in range(n):
            h_qii[q] -= diff * s_qii[q]
        for q in range(n - 1):
            h_qij[q] -= diff * s_qij[q]

    def get_g1N(self, energy):
        #
        if energy != self.energy:
            self.energy = energy

            mat_lists = self._get_mat_lists(energy)

            self.g1N = recursive_gf(*mat_lists)  # _ii, mat_list_ij, mat_list_ji)

        return self.g1N

    def get_transmission(self, energies, T_e=None):

        # xp = xp.get_array_module(self.hs_list_ii[0][0])

        if T_e is None:
            T_e = np.zeros(len(energies))

        if self.parameters["gpu"] is False:
            # It is not convenient to spilt calculation of selfenergies
            # and recursive greenfuncition on same CPU.
            for e, energy in enumerate(energies):

                gr_1N = self.get_g1N(energy)
                ga_1N = xp.conj(gr_1N.T)
                lambda1_11 = self.selfenergies[0].get_lambda(energy)
                lambda2_NN = self.selfenergies[1].get_lambda(energy)
                T_e[e] = xp.trace(lambda1_11.dot(gr_1N).dot(lambda2_NN).dot(ga_1N)).real

        else:

            import threading

            # Threading calculator
            from .tcalc import Tcalc

            calc = Tcalc(self, energies, T_e)
            # Thread on CPU for computation of selfenergies
            bcs = threading.Thread(target=calc.bcs)
            # Thread on CPU for offloading recursive greenfuncition on GPU
            rec = threading.Thread(target=calc.rec)
            # bcs MUST start before rec (see locks)
            bcs.start()
            rec.start()
            # Wait for GPU.
            rec.join()

        return T_e

    def retarded(self, energy, trace=False, diag=False):

        gr_1N, G_qii, G_qij, G_qji = recursive_gf(
            *self._get_mat_lists(energy), dos=True
        )

        return self._return(G_qii, G_qij, G_qji, trace=trace, diag=diag)

    def lesser(self, energy, fL=None, fR=None, trace=False, diag=False):

        s_in = [None for _ in range(self.N)]

        # Allow to consider separate channels
        if fL is not None:
            s_in[0] = self.selfenergies[0].get_lambda(energy) * fL
        if fR is not None:
            s_in[-1] = self.selfenergies[1].get_lambda(energy) * fR

        Gn_qii, Gn_qij, Gn_qji = recursive_gf(
            *self._get_mat_lists(energy), s_in=s_in, dos=True
        )

        return self._return(Gn_qii, Gn_qij, Gn_qji, trace=trace, diag=diag)

    def apply_retarded(self, energy, X_qii, X_qij, X_qji):  # , X_1N=None):
        """Apply retarded Green function to X in tridiagonal form."""

        G_qii, G_qij, G_qji = self.retarded(energy)

        GX_qii = multiply(G_qii, G_qij, G_qji, X_qii, X_qij, X_qji)

        return GX_qii

    def apply_overlap(self, energy, trace=False, diag=False):
        S_qii = self.hs_list_ii[1]
        S_qij = self.hs_list_ij[1]
        S_qji = self.hs_list_ji[1]

        GS_qii = self.apply_retarded(energy, S_qii, S_qij, S_qji)  # , S_1N)

        return self._return(GS_qii, trace=trace, diag=diag)

    def dos(self, energies, dos_e=None):
        if dos_e is None:
            dos_e = np.zeros_like(energies)
        for e, energy in enumerate(energies):
            GS = self.apply_overlap(energy, trace=True)
            dos_e[e] = -GS.imag / np.pi
        return dos_e

    def pdos(self, energies, bfs, sum=True):
        # Sum over basis functions bfs
        if sum:
            get = lambda a: -a.imag.take(bfs).sum() / np.pi
            pdos_e = np.zeros_like(energies)
        # Basis funtion's pdos
        else:
            get = lambda a: -a.imag.take(bfs) / np.pi
            pdos_e = np.zeros((len(energies), len(bfs)))
        for e, energy in enumerate(energies):
            GS_i = self.apply_overlap(energy, diag=True)
            pdos_e[e] = get(GS_i)
        return pdos_e

    def get_spectrals(self, energy):
        """Specral density of states incoming from leads."""
        A1_qii, A1_qij, A1_qji = self.lesser(energy, fL=1, fR=0)
        A2_qii, A2_qij, A2_qji = self.lesser(energy, fL=0, fR=1)
        A1_qmm = A1_qii, A1_qij, A1_qji
        A2_qmm = A2_qii, A2_qij, A2_qji

        return A1_qmm, A2_qmm

    def add_screening(self, V):
        if not hasattr(self, "V"):
            self.V = np.zeros(self.nbf)
        assert V.size == self.nbf
        # Add screening and remove (if exists) current.
        h_qii = self.hs_list_ii[0]
        if sum(self.V) != 0:
            add_diagonal(h_qii, -self.V)
        self.V[:] = V
        add_diagonal(h_qii, self.V)

    def remove_screening(self):
        h_qii = self.hs_list_ii[0]
        add_diagonal(h_qii, -self.V)
        self.V[:] = 0.0

    ## Helper functions

    @property
    def H(self):
        H_qii = self.hs_list_ii[0]
        H_qij = self.hs_list_ij[0]
        H_qji = self.hs_list_ji[0]
        return H_qii, H_qij, H_qji

    @H.setter
    def H(self, H):
        pass

    @property
    def S(self):
        S_qii = self.hs_list_ii[1]
        S_qij = self.hs_list_ij[1]
        S_qji = self.hs_list_ji[1]
        return S_qii, S_qij, S_qji

    @S.setter
    def S(self, S):
        pass

    def _get_mat_lists(self, energy):
        z = energy + self.eta * 1.0j
        if self.selfenergies:
            sigma_L = self.selfenergies[0].retarded(energy)
            sigma_R = self.selfenergies[1].retarded(energy)
        else:
            sigma_L = None
            sigma_R = None
        # mat_list_ii, mat_list_ij, mat_list_ji

        if self.parameters["gpu"] is True:
            sigma_L = xp.asarray(sigma_L)
            sigma_R = xp.asarray(sigma_R)

        return get_mat_lists(
            z, self.hs_list_ii, self.hs_list_ij, self.hs_list_ji, sigma_L, sigma_R
        )

    def _return(self, *args, trace, diag):

        if trace:
            return sum(A.trace() for A in args[0])
        elif diag:
            return get_diagonal(args[0])
        else:
            return args
