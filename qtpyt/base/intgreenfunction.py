from qtpyt import xp
from qtpyt.base._kernels import get_lambda, dottrace, dotdiag, dagger


class GreenFunction:
    """Interacting Green's function.
        
        |A B|
        |C D|

        S = D - (C A^-1 B)
                    ^     
                selfenergy(1)

                       selfenergy(2)
                            v
        |A^-1  +  A^-1 (B S^-1 C) A^-1   - A^-1 B S^-1|
        |    - S^-1 C A^-1                    S^-1    |

    selfenergy(1) = Selfenergy(A, B)
    selfenergy(2) = Selfenergy(D, C, selfenergy(1))
    greenfuntion = Greenfunction(A)
    intgreenfunction = Intgreenfunction(greenfunction, selfenergy(2))
    """

    def __init__(self, greenfunction, selfenergy):
        self.greenfunction = greenfunction
        self.selfenergy = selfenergy

        self.g = None
        self.Sigma = xp.zeros_like(self.greenfunction.Ginv)
        self.energy = None

    def get_partials(self, energy):
        """Get partial contributions to the Green's function.
        This is equivalent to computing the (0,0) element of a
        matrix inverse.
        G(0,0) = g(0,0) + g(0,0) (tau_mi g(1,1) tau_im) g(0,0)
        """
        if self.energy != energy:
            self.g = self.greenfunction.retarded(energy)
            indices, selfenergy = self.selfenergy
            self.Sigma[indices] = selfenergy.retarded(energy)
        return self.g, self.g.dot(self.Sigma).dot(self.g)

    def retarded(self, energy):
        g, dg = self.get_partials(energy)
        return g + dg

    def get_dos(self, energy):
        """Total density of states -1/pi Im(Tr(GS))"""
        S = self.greenfunction.S
        GS = dottrace(self.retarded(energy), S)
        return -GS.imag * 1 / xp.pi

    def get_pdos(self, energy):

        """Total density of states -1/pi Im(Tr(GS))"""
        S = self.greenfunction.S
        GS = dotdiag(self.retarded(energy), S)
        return -GS.imag * 1 / xp.pi

    def get_transmission(self, energy):
        # also update reference green's function selfenergies
        G = self.retarded(energy)
        gamma_L = self.greenfunction.gammas[0]
        gamma_R = self.greenfunction.gammas[1]
        T_e = dottrace(G.dot(gamma_L), dagger(G).dot(gamma_R)).real
        return T_e

    def get_partial_transmissions(self, energy):
        # also update reference green's function selfenergies
        g, dg = self.get_partials(energy)
        gamma_L = self.greenfunction.gammas[0]
        gamma_R = self.greenfunction.gammas[1]
        T_b = dottrace(g.dot(gamma_L), dagger(g).dot(gamma_R)).real
        T_a = dottrace(dg.dot(gamma_L), dagger(dg).dot(gamma_R)).real
        T_i = (
            dottrace(g.dot(gamma_L), dagger(dg).dot(gamma_R)).real
            + dottrace(dg.dot(gamma_L), dagger(g).dot(gamma_R)).real
        )
        return T_b, T_a, T_i
