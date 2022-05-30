import functools

import numpy as np
from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.base.selfenergy import BaseSelfEnergy
from scipy.linalg import lu_factor, lu_solve


class SanchoRubio(BaseGreenFunction):
    """Recursive Sancho-Rubio algorithm for surface 
    Green's function.
    
    """

    conv = 1e-8  # Convergence criteria for surface Green function

    def __init__(self, hs_ii, hs_ij, eta=1e-5) -> None:
        self.h_ij, self.s_ij = hs_ij  # Coupling to next cell.
        super().__init__(hs_ii[0], hs_ii[1], eta, 0.0, 0.0)

    def get_Ginv(self, energy):
        """The inverse of the retarded surface Green function"""
        z = self.z(energy)

        v_00 = z * self.S.T.conj() - self.H.T.conj()
        v_11 = v_00.copy()
        v_10 = z * self.s_ij - self.h_ij
        v_01 = z * self.s_ij.T.conj() - self.h_ij.T.conj()

        delta = self.conv + 1
        while delta > self.conv:
            lu, piv = lu_factor(v_11, check_finite=False)
            a = lu_solve((lu, piv), v_01, check_finite=False)
            b = lu_solve((lu, piv), v_10, check_finite=False)
            # a = np.linalg.solve(v_11, v_01)
            # b = np.linalg.solve(v_11, v_10)
            v_01_dot_b = np.dot(v_01, b)
            v_00 -= v_01_dot_b
            v_11 -= np.dot(v_10, a)
            v_11 -= v_01_dot_b
            v_01 = -np.dot(v_01, a)
            v_10 = -np.dot(v_10, b)
            delta = abs(v_01).max()
        return v_00

    def retarded(self, energy):
        if energy != self.energy:
            self.energy = energy
            self.Gr = np.linalg.inv(self.get_Ginv(energy))
        return self.Gr


class LeadSelfEnergy(BaseSelfEnergy):

    ids = ["left", "right"]

    def __init__(self, hs_ii, hs_ij, hs_im=None, nbf_m=None, id="left", eta=1e-5):

        assert id in self.ids, f"Invalid id. Choose between {self.ids}"

        if id == "right":
            h_ij = hs_ij[0].swapaxes(0, 1).conj()
            s_ij = hs_ij[1].swapaxes(0, 1).conj()
            hs_ij = (h_ij, s_ij)
            if hs_im is not None:
                h_im = hs_im[0].swapaxes(0, 1).conj()
                s_im = hs_im[1].swapaxes(0, 1).conj()
                hs_im = (h_im, s_im)
        if hs_im is None:
            hs_im = hs_ij

        gf = SanchoRubio(hs_ii, hs_ij, eta)

        if nbf_m is not None:
            nbf_i = gf.shape[0]
            h_im = np.zeros((nbf_i, nbf_m), complex)
            s_im = np.zeros((nbf_i, nbf_m), complex)
            if id == "left":
                h_im[:, :nbf_i] = hs_im[0]
                s_im[:, :nbf_i] = hs_im[1]
            else:
                h_im[:, -nbf_i:] = hs_im[0]
                s_im[:, -nbf_i:] = hs_im[1]
            hs_im = (h_im, s_im)

        super().__init__(gf, hs_im)
