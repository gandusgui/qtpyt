from math import pi, sqrt

# Also import numpy for array matrices
import numpy as np

from qtpyt import xp
from qtpyt.base.selfenergy import DataSelfEnergy
from qtpyt.parallel.egrid import GridDesc
from qtpyt.parallel.tools import comm_sum
from qtpyt.projector import BaseProjector, ProjectedGreenFunction
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.screening.iterate_product import iterate_product
from qtpyt.screening.langreth import LangrethPair, assert_domain, change_domain
from qtpyt.screening.products import exchange_product, polarization_product
from qtpyt.screening.tools import lesser_from_retarded, roll, rotate


class Chi(LangrethPair):
    """Polarization.
    
    This class computes the polarization function. It is the base class
    for any caluclation that requires the convolution of the the outer
    product of a green's function. 
    
    At the end of an update, the langreth arrays will contain the retarded
    and greater components in energy domain.
    
    Args:
        gf : (DistGreenFunction) object
            the green's function for which to compute the 
            polarization.
            
        U_qp : (optional)
            rotation to go from full pair orbitals to opti-
            mized pair orbitlas. (#q < #p).
            
    Attributes:
        This class and those inheriting follow the convention
        to define a solve and an update method. The first will only
        solve the task of the current class whereas the second
        will also take care of updating the parent classes.
    """

    def __init__(self, gf: DistGreenFunction, U_pq=None, oversample=10) -> None:
        self.gf = gf
        # Decide if polarization is for full or optimized pair orbitals.
        nq = gf.no ** 2 if U_pq is None else U_pq.shape[1]
        super().__init__(gf.global_energies, nq, dtype=complex, oversample=oversample)

        self.U_pq = U_pq if U_pq is None else xp.asarray(U_pq)
        # self.U_qp = None if U_pq is None else self.U_pq.T.copy()

    def get_work_array(self):
        """Get temp work arrays used in product."""
        if self.U_pq is None:
            return None
        return xp.empty_like(self.U_pq)

    def solve_polarization(self):
        """Compute polarization product."""
        #
        #         </>        +     </>      >/<
        #       P   =  2 i U   ( G    x   G    ) U

        assert self.gf.domain == "t"
        Gl = self.gf.arrays["l"]
        Gg = self.gf.arrays["g"]
        Pl = self.arrays["l"]
        Pg = self.arrays["g"]

        # Prefactor: 2 for spin, j. from definition, and de / 2 pi for fft
        pre = 2.0j * self.de / (2.0 * pi)

        # Temporary work array
        work1 = self.get_work_array()
        work2 = self.get_work_array()
        # P^</>_ij,kl(t)      = -i G^</>_ik(t) G^>/<_lj(-t)
        #
        # G_lj(-t) = -G_jl(t)*
        #
        for t in range(Gl.shape[0]):
            for G, Gd, P in ([Gl[t], Gg[t], Pl[t]], [Gg[t], Gl[t], Pg[t]]):
                polarization_product(G, Gd.conj(), pre, P, self.U_pq, work1, work2)

        # # chi< = pre * Ud (G< otimes G>.conj ) U
        # iterate_product(
        #     polarization_product,
        #     Gl,
        #     Gg,
        #     self.arrays["l"],  # use retarded array
        #     pre=pre,
        #     rot=self.U_pq,
        #     # rotd=self.U_qp,
        #     work1=work1,
        #     work2=work2,
        # )

        # # chi> = pre * Ud (G> otimes G<.conj ) U
        # iterate_product(
        #     polarization_product,
        #     Gg,
        #     Gl,
        #     self.arrays["g"],
        #     pre=pre,
        #     rot=self.U_pq,
        #     # rotd=self.U_qp,
        #     work1=work1,
        #     work2=work2,
        # )  # P<(t), P>(t)

        self.domain = "t"
        self.convert_less_and_great_to_ret(-self.zero_index)  #  Pr(e), P>(e)

    def update_polarization(self):
        """Update retarded and greater components."""
        #
        #         r               <    >
        #       P   =  theta(t) (P  - P )
        #
        if self.gf.domain == None:
            self.gf.update()  # Gr(e), G<(e)
        if not "g" in self.gf.arrays.keys():
            self.gf.convert_retarded()  # G<(e), G>(e)
        if self.gf.domain == "e":
            self.gf.convert_domain()  # G<(t), G>(t)
        self.solve_polarization()  # Pr(e), P>(e)


class WRPA(Chi):
    """Screened interaction within RPA.
    
    This class computes the screened interaction whithin the
    RPA approximation.
    
    Inherits from polarization baseclass. An update of this class
    will start from the polarizations computeted by an update of
    the parent and will overwrite the langreth arrays with the 
    retarded and greater components in energy domain.
    
    Args:
        gf : (DistGreenFunction) object
            the green's function for which to compute the 
            polarization.
        
        V_qq : (ndarray)
            coulomb matrix in either full or optimized
            pair orbital basis.
        
        U_qp : (optional)
            rotation to go from full pair orbitals to opti-
            mized pair orbitlas. (#q < #p).
    """

    def __init__(self, gf, V_qq, U_pq=None, oversample=10) -> None:
        super().__init__(gf, U_pq, oversample=oversample)
        self.V_qq = V_qq

    @assert_domain("e")
    def solve_screening(self):
        I = np.eye(self.no)
        Wr = Pr = self.arrays["r"]
        Wg = Pg = self.arrays["g"]

        work = np.empty_like(Wr[0])
        for e in range(self.energies.size):
            # NOTE that Wr is Pr but it's okey since we read-write same energy.
            Wr[e] = np.linalg.solve(I - self.V_qq.dot(Pr[e]), self.V_qq)
            rotate(Pg[e], Wr[e], out=Wg[e], work=work)
            # Wg[e] = Wr[e].dot(Pg[e]).dot(Wr[e].T.conj())

    def update_screening(self):
        """Update retared and greater components.
        
        NOTE : This method will overwrite the retarded and greater polarizations
        with the corresponding screened interaction.
        """
        #       r             r
        #      W    = [I - V P ] V
        #
        #       >     r   <   a
        #      W   = W  P   W
        #
        self.update_polarization()  # Pr(e), P>(e)
        self.solve_screening()  # Wr(e), W>(e)


class GW(LangrethPair):
    """GW selfenergy.
    
    This class computes the GW selfenergy within the RPA
    approximation. 
    
    Inherits from screened interaction class. An update of
    this class starts from the screening computed by the parent,
    converts the langreth pair to the their lesser and grea-
    ter components in energy domain, and overwrites the lesser
    and greater components of the green's function. 
    
    NOTE :This means that at the end of a call to this class the 
    lesser and greater green's function will contain the corresp-
    onding GW selfenergies.
    
    Args:
        gf : (DistGreenFunction) object
            the green's function for which to compute the 
            polarization.
        
        V_qq : (ndarray)
            coulomb matrix in either full or optimized
            pair orbital basis.
        
        U_qp : (optional)
            rotation to go from full pair orbitals to opti-
            mized pair orbitlas. (#q < #p).
    """

    def __init__(
        self, wrpa: WRPA, gf: DistGreenFunction = None, U_pq=None, oversample=10
    ) -> None:
        self.wrpa = wrpa
        if gf is None:
            gf = self.wrpa.gf
        if U_pq is None and hasattr(self.wrpa, "U_pq"):  # W is in pair-orbital basis.
            U_pq = self.wrpa.U_pq
        self.gf = gf
        self.U_pq = U_pq
        super().__init__(
            gf.global_energies, gf.no, dtype=complex, oversample=oversample
        )

    def get_work_array(self):
        """Get temp work arrays used in product."""
        if self.U_pq is None:
            return None
        return xp.empty_like(self.U_pq)

    def solve_correlation(self):

        if self.gf.domain == "e":
            # At this point gf has been updated.
            # If gf is the same as wrpa.gf, then domain is already t
            # Else, it needs to be converted.
            self.gf.convert_domain()
        Gl = self.gf.arrays["l"]
        Gg = self.gf.arrays["g"]
        Wl = self.wrpa.arrays["l"]
        Wg = self.wrpa.arrays["g"]

        self.arrays["g"] = (
            self.arrays["g"] if "g" in self.arrays.keys() else self.arrays.pop("r")
        )
        GWl = self.arrays["l"]
        GWg = self.arrays["g"]

        # Prefactor: j. from definition, and de / 2 pi for fft
        pre = 1.0j * self.de / (2 * np.pi)

        work1 = self.get_work_array()
        work2 = self.get_work_array()

        for t in range(Gl.shape[0]):
            for G, W, GW in [[Gl[t], Wl[t], GWl[t]], [Gg[t], Wg[t], GWg[t]]]:
                exchange_product(
                    G, W, pre=pre, out=GW, rot=self.U_pq, work1=work1, work2=work2,
                )

        self.domain = "t"
        self.convert_less_and_great_to_ret(
            self.zero_index, override="g"
        )  # domain << energy

    def update_correlation(self):
        # update lesser greater green's

        self.wrpa.update_screening()  # Wr(e), W>(e)
        self.wrpa.convert_retarded()  # W<(e), W>(e)
        self.wrpa.convert_domain()  # W<(t), W>(t)
        self.solve_correlation()  # GWr(e), GW<(e)

    def retarded(self, energy):
        """The retarded GW self-energy."""
        return self.arrays["r"][np.searchsorted(self.energies, energy)]

    def lesser(self, energy):
        """The retarded GW self-energy."""
        return self.arrays["l"][np.searchsorted(self.energies, energy)]


# Constrained methods


def cut_rotation(U_pq, indices):
    """Cut pairorbital rotation matrix."""
    no2, nq = U_pq.shape
    no = int(sqrt(no2))
    assert no ** 2 == no2, "# p is not square of orbitals."
    U_cq = np.ascontiguousarray(
        U_pq.reshape(no, no, nq)[np.ix_(indices, indices)].reshape(
            len(indices) ** 2, nq
        )
    )
    return U_cq


class cRPA(LangrethPair):
    """Constrained RPA."""

    def __init__(self, wrpa: WRPA, ic: np.ndarray) -> None:
        self.U_qq = None  # sets to 0. polarization of `c` subspace
        self.U_cq = None
        if wrpa.U_pq is not None:
            U_pq = wrpa.U_pq
            U_qp = U_pq.T
            U_pr = np.eye(U_pq.shape[0], dtype=U_pq.dtype)
            #
            self.U_qq = U_qp.dot(U_pr).dot(U_pq)
            self.U_cq = cut_rotation(U_pq, ic)
        super().__init__(wrpa.global_energies, len(ic)**2, dtype=complex)
        self.wrpa = wrpa
        self.ic = ic
        self.ic4 = np.ix_(self.ic, self.ic, self.ic, self.ic)
        self.arrays["r"] = self.arrays.pop("l")
        self.arrays.pop("g")  # not required
        # self.domain = "e"

    def _screen_polarization(self, P, work=None):
        if self.U_qq is None:
            no = int(sqrt(self.no))
            P = P.reshape(4 * (no,))
            P[self.ic4] = 0.0
        else:
            work = P.dot(self.U_qq, out=work)
            return self.U_qq.T.dot(work, out=P)
            # return rotate(P, self.U_qq, work=work, overwrite_a=True)

    # @assert_domain("e")
    def screen_polarization(self):
        """Set to zero the polarization of the constrained region."""
        Pr = self.wrpa.arrays["r"]
        work = (
            None
            if self.U_qq is None
            else np.empty(self.U_qq.shape, dtype=self.U_qq.dtype)
        )
        for e in range(Pr.shape[0]):
            self._screen_polarization(Pr[e], work=work)

    def _cut_screening(self, W, work=None, out=None):
        if self.U_cq is None:
            no = int(sqrt(self.no))
            W = W.reshape(4 * (no,))
            return W[self.ic4]
        else:
            work = W.dot(self.U_cq.T, out=work)
            return self.U_cq.dot(work, out=out)
            # return rotate(W, self.U_cq, work=work, out=out)

    # @assert_domain("e")
    def solve_unscreening(self):
        Wr = self.wrpa.arrays["r"]
        Ur = self.arrays["r"]
        work = (
            None
            if self.U_cq is None
            else np.empty(self.U_cq.T.shape, dtype=self.U_cq.dtype)
        )
        for e in range(Wr.shape[0]):
            self._cut_screening(Wr[e], work=work, out=Ur[e])

    def update_unscreening(self):
        # update lesser greater green's
        self.wrpa.update_polarization()  # Pr(e), P>(e)
        self.screen_polarization()  # Pr'(e) :: ' means `c` subspace is 0.
        self.wrpa.solve_screening()  # Wr(e)
        self.solve_unscreening()

    # def __init__(self, gf: DistGreenFunction, V_qq, U_pq=None, oversample=10) -> None:
    #     # super().__init__(DistGreenFunction(gf, energies), V_qq, U_pq)
    #     Constrained.__init__(self, gf, U_pq)
    #     WRPA.__init__(self, gf.parent, V_qq, U_pq, oversample=oversample)
    #     self.chi = Chi(self.gfc, oversample=oversample)
    #     super().__init__(energies, no, dtype, oversample)

    #     self.rotate = self._cut if U_pq is None else self._rot

    # def _cut(self, W, out):
    #     out[:] = W.reshape(4 * (self.gf.no,))[self.ix4].reshape(2 * (len(self.chi.no),))
    #     return out

    # def _rot(self, W, out):
    #     return self.U_cq.dot(W, out=self.work).dot(self.U_cq.T, out=out)

    # @property
    # def W(self):
    #     raise NotImplementedError("Not available with this method.")

    # @property
    # def U(self):
    #     return self.chi.arrays["r"]

    # @assert_domain("e")
    # def screen_polarization(self):
    #     """Set to zero the polarization of the constrained region."""
    #     Pr = self.arrays["r"]
    #     no = self.gf.no
    #     rotate = lambda X, U: X
    #     if self.U_pq is not None:
    #         rotate = lambda X, U: U.dot(X).dot(U.T)
    #     for e in range(self.energies.size):
    #         P_pp = rotate(Pr[e], self.U_pq)
    #         P_pp = P_pp.reshape((no,) * 4)
    #         P_pp[self.ix4] = 0.0
    #         P_pp = P_pp.reshape((no ** 2,) * 2)
    #         Pr[e, ...] = rotate(P_pp, self.U_pq.T)

    # @assert_domain("e")
    # def solve_unscreening(self):
    #     W = self.arrays["r"]
    #     self.chi.arrays["r"] = self.chi.arrays.pop("l")
    #     U = self.chi.arrays["r"]
    #     for e in range(self.energies.size):
    #         self.rotate(W[e], out=U[e])

    # def update_unscreening2(self):
    #     self.update_polarization()  # Pr(e), P>(e)
    #     self.screen_polarization()
    #     self.solve_screening()  # Wr(e), W>(e)
    #     self.solve_unscreening()  # Ur(e)


# class Constrained:
#     """Base class for constrained calculations.

#     Args:
#         gf : qtpyt:DistGreenFunction
#             distributed green's function object.
#         indices : 1D array
#             indices of constrained subspace.
#         U_pq : 2D array (optional)
#             see Chi
#     """

#     def __init__(self, gf: DistGreenFunction, U_pq=None) -> None:
#         assert isinstance(
#             gf.gf0.projector, BaseProjector
#         ), "Invalid parent green's function for constrained region."
#         self.gfc = gf
#         idx = self.indices
#         self.ix = np.ix_(idx, idx)
#         self.ix4 = np.ix_(idx, idx, idx, idx)

#         if U_pq is None:
#             U_cq = None

#         else:
#             U_cq = self.cut_rotation(U_pq, idx)

#         self.U_cq = U_cq

#     @property
#     def indices(self):
#         return self.gfc.gf0.projector.indices

#     @staticmethod
#     def cut_rotation(U_pq, indices):
#         """Cut pairorbital rotation matrix."""
#         no2, nq = U_pq.shape
#         no = int(sqrt(no2))
#         assert no ** 2 == no2, "# p is not square of orbitals."
#         U_cq = np.ascontiguousarray(
#             U_pq.reshape(no, no, nq)[np.ix_(indices, indices)].reshape(
#                 len(indices) ** 2, nq
#             )
#         )
#         return U_cq

#     def get_cwork_array(self):
#         if self.U_cq is None:
#             return None
#         else:
#             return xp.empty_like(self.U_cq)


# class cGW(Constrained, GW):
#     """Constrained GW calculation.

#     The GW selfenergy is calculated only for a subspace of
#     the WRPA region where the screening is computed.
#     """

#     def __init__(self, gf: DistGreenFunction, V_qq, U_pq, oversample=10) -> None:
#         # super().__init__(DistGreenFunction(gf, energies), V_qq, U_pq)
#         Constrained.__init__(self, gf, U_pq)
#         GW.__init__(self, gf.parent, V_qq, U_pq, oversample=oversample)

#     def __getattr__(self, attr):
#         """When """
#         if attr in ['gf','U_pq','get_work_array']:
#             return getattr(self.wrpa, attr)
#         raise AttributeError

#     @assert_domain("t")
#     def solve_correlation(self):

#         if self.gf.domain == "e":
#             self.gf.convert_domain()
#         self.gfc.update_from_parent(self.gf)
#         Gl = self.gfc.arrays["l"]
#         Gg = self.gfc.arrays["g"]
#         Wl = self.arrays["l"]
#         Wg = self.arrays["g"]

#         # Prefactor: j. from definition, and de / 2 pi for fft
#         pre = 1.0j * self.de / (2 * np.pi)

#         work1 = self.get_cwork_array()
#         work2 = self.get_cwork_array()

#         for t in range(Gl.shape[0]):
#             for G, W in [[Gl[t], Wl[t]], [Gg[t], Wg[t]]]:
#                 exchange_product(
#                     G,
#                     W,
#                     pre=pre,
#                     out=G,  # OK to overwrite
#                     rot=self.U_cq,
#                     # rotd=self.U_qp,
#                     work1=work1,
#                     work2=work2,
#                 )

#         self.gfc.convert_less_and_great_to_ret(self.zero_index)  # domain << energy
#         self.gfc.domain = None

#     def retarded(self, energy):
#         """The retarded GW self-energy."""
#         return self.gfc.arrays["r"][np.searchsorted(self.energies, energy)]

#     def greater(self, energy):
#         return self.gfc.arrays["g"][np.searchsorted(self.energies, energy)]

#     def lesser(self, energy):
#         Gr = self.retarded(energy)
#         Gg = self.retarded(energy)
#         return lesser_from_retarded(self.retarded(energy), self.gfc.arr

# class cRPA(Constrained, WRPA):
#     """Constrained RPA."""

#     def __init__(self, gf: DistGreenFunction, V_qq, U_pq=None, oversample=10) -> None:
#         # super().__init__(DistGreenFunction(gf, energies), V_qq, U_pq)
#         Constrained.__init__(self, gf, U_pq)
#         WRPA.__init__(self, gf.parent, V_qq, U_pq, oversample=oversample)
#         self.chi = Chi(self.gfc, oversample=oversample)

#         self.work = self.get_cwork_array()
#         self.rotate = self._cut if self.work is None else self._rot

#     def _cut(self, W, out):
#         out[:] = W.reshape(4 * (self.gf.no,))[self.ix4].reshape(2 * (len(self.chi.no),))
#         return out

#     def _rot(self, W, out):
#         return self.U_cq.dot(W, out=self.work).dot(self.U_cq.T, out=out)

#     @property
#     def W(self):
#         return self.chi.arrays["g"]

#     @property
#     def U(self):
#         return self.chi.arrays["r"]

#     @assert_domain("e")
#     def solve_unscreening(self):
#         #       r      r       r  r  ^-1
#         #      U    = W [ I + P  W ]
#         Pr = self.chi.arrays["r"]
#         Pg = self.chi.arrays["g"]
#         Wr = self.arrays["r"]

#         I = np.eye(self.chi.no)

#         for e in range(self.energies.size):
#             cWr = self.rotate(Wr[e], out=Pg[e])
#             cWr.dot(np.linalg.inv(I + Pr[e].dot(cWr)), out=Pr[e])

#     def update_unscreening(self):
#         self.update_screening()  # Wr(e), W>(e)
#         self.gfc.update_from_parent(self.gf)  # g<(t), g>(t)
#         self.chi.update_polarization()  # Pr(e), P>(e)
#         self.solve_unscreening()

