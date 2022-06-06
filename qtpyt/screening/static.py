from qtpyt import xp
from qtpyt.base.selfenergy import ConstSelfEnergy
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.screening.products import exchange_product, hartree_product


class Fock(ConstSelfEnergy):
    """Fock selfenergy.
    
    This class computes the Fock diagram for a given green's
    function. 

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

    def __init__(self, gf: DistGreenFunction, V_qq, U_pq=None) -> None:
        self.gf = gf
        self.V_qq = V_qq
        self.U_pq = U_pq if U_pq is None else xp.asarray(U_pq)
        super().__init__(shape=2 * (gf.no,))

    def get_work_array(self):
        """Get temp work arrays used in product."""
        if self.U_pq is None:
            return None
        return xp.empty_like(self.U_pq)

    def update(self, D=None):
        """Update Fock self-energy
        
        Args:
            D : (optional)
                The density matrix.
        """
        if D is None:
            assert self.gf.domain == "e"
            D = self.gf.get_density_matrix()

        pre = -1.0  # -1. from definition

        work1 = self.get_work_array()
        work2 = self.get_work_array()

        # for g, s in zip(Gl, self.Sigma):
        exchange_product(
            D,
            self.V_qq,
            pre=pre,
            out=self.Sigma,
            rot=self.U_pq,
            # rotd=self.U_qp,
            work1=work1,
            work2=work2,
        )


class Hartree(ConstSelfEnergy):
    """Hartree selfenergy.
    
    
    This class computes the Hartree correction after a change in
    the underlying green's function.

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

    def __init__(self, gf: DistGreenFunction, V_qq, U_pq=None) -> None:
        self.gf = gf
        self.V_qq = V_qq
        self.U_pq = U_pq if U_pq is None else xp.asarray(U_pq)
        self.Sigma0 = xp.zeros(2 * (gf.no,), complex)
        super().__init__(shape=2 * (gf.no,))

    def get_work_array(self):
        """Get temp work arrays used in product."""
        if self.U_pq is None:
            return None
        return xp.empty(self.U_pq.shape[1], complex)

    def initialize(self, D=None):
        self.hartree_product(D, self.Sigma0)

    def hartree_product(self, D=None, Sigma=None):
        """Update Hartree self-energy
        
        Args:
            D : (optional)
                The density matrix.
            Sigma: (optional)
                The self-energy array to update.
        """
        if D is None:
            assert self.gf.domain == "e"
            D = self.gf.get_density_matrix()

        if Sigma is None:
            Sigma = self.Sigma

        pre = 2.0  # 2. from spin

        work1 = self.get_work_array()
        work2 = self.get_work_array()

        # for g, s in zip(Gl, self.Sigma if Sigma is None else Sigma):
        hartree_product(
            D,
            self.V_qq,
            pre=pre,
            out=Sigma,
            rot=self.U_pq,
            # rotd=self.U_qp,
            work1=work1,
            work2=work2,
        )

    def update(self, D):
        self.hartree_product(D, self.Sigma)
        self.Sigma -= self.Sigma0
