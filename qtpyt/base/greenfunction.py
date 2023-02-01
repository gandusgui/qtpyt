from math import pi
from typing import Any, List

from qtpyt import xp
from qtpyt.base._kernels import dagger, dotdiag, dottrace, get_lambda
from qtpyt.screening.tools import fermidistribution, rotate


class BaseGreenFunction:
    """Base class for the Green's function."""

    def __init__(self, H, S, eta=1e-5, mu=0.0, kt=0.0) -> None:
        self.H = H  # Hamiltonian
        self.S = S  # Overlap
        self.eta = eta  # immaginary shift
        self.mu = mu  # chemical potential
        self.kt = kt  # boltzmann temperature
        self.energy = None

    @property
    def shape(self):
        """Size of matrices."""
        return self.H.shape

    def z(self, energy):
        """The complex energy above the real axis."""
        return energy + 1.0j * self.eta

    def fermi(self, energy):
        """The fermi distribution"""
        return fermidistribution(energy - self.mu, self.kt)

    def get_Ginv(self, energy):
        """The inverse of the retarded green's function."""
        return None

    def retarded(self, energy):
        """The retarded green's function."""
        return None

    def advanced(self, energy):
        """The advanced green's function."""
        return dagger(self.retarded(energy))

    def lesser(self, energy):
        """The lesser green's function."""
        return -self.fermi(energy) * (self.retarded(energy) - self.advanced(energy))

    def greater(self, energy):
        """The greater green's function."""
        return (1.0 - self.fermi(energy)) * (
            self.retarded(energy) - self.advanced(energy)
        )

    def get_dos(self, energy):
        """Total density of states."""
        return -dottrace(self.retarded(energy), self.S).imag / pi

    def get_pdos(self, energy):
        """Projected density of states."""
        return -dotdiag(self.S, self.retarded(energy).dot(self.S)).imag / pi


class GreenFunction(BaseGreenFunction):
    """Green's function class."""

    def __init__(
        self, H, S, selfenergies=[], idxleads=[0, 1], eta=1e-5, mu=0.0, kt=0.0
    ):
        # Offload to GPU
        H = xp.asarray(H)
        S = xp.asarray(S)

        self.selfenergies = selfenergies
        self.idxleads = sorted(idxleads)
        self.gammas = None

        super().__init__(H, S, eta, mu, kt)

    @property
    def equilibrium(self):
        """Equilibrium Green's function."""
        biases = []
        for i in self.idxleads:
            selfenergy = self.selfenergies[i][1]
            biases.append(selfenergy.bias)
        return xp.all([abs(bias - biases[0]) < 1e-7 for bias in biases])

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for _, selfenergy in self.selfenergies:
                selfenergy.eta = value
        super().__setattr__(name, value)

    def get_Ginv(self, energy):
        Ginv = self.z(energy) * self.S - self.H
        self.gammas = []
        # Add selfenergies
        for i, (indices, selfenergy) in enumerate(self.selfenergies):
            sigma = xp.asarray(selfenergy.retarded(energy))
            Ginv[indices] -= sigma
            if i in self.idxleads:
                self.gammas.append(get_lambda(sigma))

        return Ginv

    def retarded(self, energy):
        """The retarded green's function."""
        if self.energy != energy:
            self.energy = energy
            Ginv = self.get_Ginv(energy)
            self.Gr = xp.linalg.inv(Ginv)
        return self.Gr

    def lesser(self, energy):
        """The lesser green's function."""

        # -f(e-mu) (Gr - Ga)
        if self.equilibrium:
            return super().lesser(energy)

        # Gr S< Ga
        self.Sl = xp.zeros(self.shape, complex)
        for i, (indices, selfenergy) in enumerate(self.selfenergies):
            if hasattr(selfenergy, "lesser"):
                self.Sl[indices] += selfenergy.lesser(energy)
        return self.retarded(energy).dot(self.Sl).dot(self.advanced(energy))

    def greater(self, energy):

        # (1-f(e-mu)) (Gr - Ga)
        if self.equilibrium:
            return super().greater(energy)

        # < = > - r + a
        # > = < + r - a
        return self.lesser(energy) + self.retarded(energy) - self.advanced(energy)

    def get_transmission(self, energy):
        """Get the transmission coeffiecient."""
        # if self.equilibrium:
        # Gr = self.retarded(energy)  # updates gammas
        a_mm = self.retarded(energy).dot(self.gammas[0])
        b_mm = self.advanced(energy).dot(self.gammas[1])
        return dottrace(a_mm, b_mm).real

        # delta = 0.0
        # for se in selist:
        #     if se.Correlated():
        #         ret = se.GetRetarded(e)
        #         delta += 1.j * (ret - ret.T.conj())
        # delta[:] = np.linalg.solve(lambdal + lambdar +
        #                             2 * self.GetInfinitesimal() *
        #                             self.GetIdentityRepresentation(),
        #                             delta)
        # delta.flat[::len(delta) + 1] += 1.0
        # T_e[e] = dots(
        #     lambdal, Gr, lambdar, delta, Gr.T.conj()).trace().real

    def get_current_density(self, energy):
        """The current density."""
        selfenergy = self.selfenergies[self.idxleads[0]][1]
        fermi = selfenergy.gf.fermi(energy)
        Gl = self.lesser(energy)  # updates gammas
        Gg = self.greater(energy)
        Sl = 1.0j * self.gammas[0] * fermi
        Sg = 1.0j * self.gammas[0] * (fermi - 1)
        # return sum(dotdiag(Sl, Gg) - dotdiag(Sg, Gl)).real
        return (dottrace(Sl, Gg) - dottrace(Sg, Gl)).real

    def get_spectrals(self, energy):
        """Get spectral functions."""
        spectrals = []
        for gamma in self.gammas:
            spectrals.append(
                self.retarded(energy).dot(gamma).dot(self.advanced(energy))
            )

        return spectrals


class TransmissionCalculator:
    """Transmission calculator."""

    def __init__(self, gf: GreenFunction, leads: List["LeadSelfEnergy"]) -> None:
        if len(leads) != 2:
            raise NotImplementedError("Only 2 leads implemented.")
        self.gf = gf
        self.leads = leads

    def get_transmission(self, energy):
        # TODO :: improve with single factorization compatible with GPU
        gamma_L = self.leads[0].get_lambda(energy)
        gamma_R = self.leads[1].get_lambda(energy)
        a_mm = self.gf.retarded(energy).dot(gamma_L)
        b_mm = self.gf.advanced(energy).dot(gamma_R)
        return dottrace(a_mm, b_mm).real
