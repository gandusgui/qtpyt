from math import pi
from typing import Any, List

from qtpyt import xp
from qtpyt.base._kernels import dagger, dotdiag, dottrace, get_lambda
from qtpyt.screening.tools import fermidistribution


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
        self.idxleads = idxleads
        self.gammas = [None] * len(idxleads)

        super().__init__(H, S, eta, mu, kt)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "eta":
            for _, selfenergy in self.selfenergies:
                selfenergy.eta = value
        super().__setattr__(name, value)

    def get_Ginv(self, energy):
        Ginv = self.z(energy) * self.S - self.H
        # Add selfenergies
        for i, (indices, selfenergy) in enumerate(self.selfenergies):
            sigma = xp.asarray(selfenergy.retarded(energy))
            Ginv[indices] -= sigma
            if i in self.idxleads:
                self.gammas[i] = get_lambda(sigma)

        return Ginv

    def retarded(self, energy):
        """The retarded green's function."""
        if self.energy != energy:
            self.energy = energy
            Ginv = self.get_Ginv(energy)
            self.Gr = xp.linalg.inv(Ginv)
        return self.Gr

    def get_transmission(self, energy):
        """Get the transmission coeffiecient."""
        a_mm = self.retarded(energy).dot(self.gammas[0])
        b_mm = self.advanced(energy).dot(self.gammas[1])
        return dottrace(a_mm, b_mm).real

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
