import _thread

from transport import xp
from .solvers.recursive import get_mat_lists, recursive_gf


class Tcalc:
    """
    Class to compute T_e concurrently between two threads
    """

    def __init__(self, greenfuncition, energies, T_e):
        self.greenfuncition = greenfuncition
        self.selfenergies = self.greenfuncition.selfenergies
        self.energies = energies
        self.T_e = T_e
        self.sigma_L = None
        self.sigma_R = None
        self.bcs_lock = _thread.allocate_lock()
        self.rec_lock = _thread.allocate_lock()
        #
        self.rec_lock.acquire()

    def bcs(self):

        for energy in self.energies:

            self.bcs_lock.acquire()
            self.sigma_L = self.selfenergies[0].retarded(energy)
            self.sigma_R = self.selfenergies[1].retarded(energy)
            self.rec_lock.release()

    def rec(self):

        for e, energy in enumerate(self.energies):

            self.rec_lock.acquire()
            sigma_L = xp.asarray(self.sigma_L)  # local
            sigma_R = xp.asarray(self.sigma_R)  # local
            self.bcs_lock.release()

            gf = self.greenfuncition
            z = energy + gf.eta * 1.0j
            mat_lists = get_mat_lists(
                z, gf.hs_list_ii, gf.hs_list_ij, gf.hs_list_ji, sigma_L, sigma_R
            )
            lambda1_11 = 1.0j * (sigma_L - xp.conj(sigma_L.T))
            lambda2_NN = 1.0j * (sigma_R - xp.conj(sigma_R.T))
            gr_1N = recursive_gf(*mat_lists)
            ga_1N = xp.conj(gr_1N.T)
            self.T_e[e] = xp.trace(
                lambda1_11.dot(gr_1N).dot(lambda2_NN).dot(ga_1N)
            ).real
