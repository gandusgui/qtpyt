from types import MethodType
from typing import Any, Callable, Union

import numpy as np

from qtpyt import xp
from qtpyt.base._kernels import dagger
from qtpyt.base.greenfunction import BaseGreenFunction, GreenFunction


class BaseSelfEnergy:
    """Base class for a self-energy."""

    def __init__(self, gf: BaseGreenFunction, hs_ij) -> None:
        self.gf = gf
        self.h_ij, self.s_ij = hs_ij
        self.Sigma = np.empty(self.shape, complex)
        self.energy = None

    @property
    def shape(self):
        return 2 * (self.h_ij.shape[1],)

    @property
    def H(self):
        return self.gf.H

    @property
    def S(self):
        return self.gf.S

    @property
    def eta(self):
        return self.gf.eta

    @eta.setter
    def eta(self, eta):
        self.gf.eta = eta

    def z(self, energy):
        return self.gf.z(energy)

    def retarded(self, energy):
        """The retarded self-energy."""
        if energy != self.energy:
            z = self.z(energy)
            tau_ij = z * self.s_ij - self.h_ij
            G = self.gf.retarded(energy)
            tau_ji = z * self.s_ij.T.conj() - self.h_ij.T.conj()
            tau_ji.dot(G).dot(tau_ij, out=self.Sigma)

        return self.Sigma

    def advanced(self, energy):
        return dagger(self.retarded(energy))

    def get_lambda(self, energy):
        """Return the lambda (aka Gamma) defined by i(S-S^d)."""
        Sigma = self.retarded(energy)
        return 1.0j * (Sigma - Sigma.T.conj())


class SelfEnergy(BaseSelfEnergy):
    """Self-energy."""

    ids = ["left", "right"]

    def __init__(
        self,
        hs_ii,
        hs_ij,
        selfenergies=[],
        idxleads=[0, 1],
        id="left",
        eta=1e-5,
        mu=0.0,
        kt=0.0,
    ):

        assert id in self.ids, f"Invalid id. Choose between {self.ids}"

        if id == "right":
            h_ij = hs_ij[0].swapaxes(0, 1).conj()
            s_ij = hs_ij[1].swapaxes(0, 1).conj()
            hs_ij = (h_ij, s_ij)

        gf = GreenFunction(hs_ii[0], hs_ii[1], selfenergies, idxleads, eta, mu, kt)

        super().__init__(gf, (xp.asarray(m) for m in hs_ij))


class ConstSelfEnergy:
    """Constant selfenergy."""

    def __init__(self, Sigma=None, shape=None) -> None:
        if Sigma is None:
            Sigma = xp.zeros(shape, complex)
        self.Sigma = Sigma

    def retarded(self, energy):
        return self.Sigma


class DataSelfEnergy(ConstSelfEnergy):
    """Constant selfenergy."""

    def __init__(self, energies, Sigma) -> None:
        self.energies = energies
        self.energy = None
        super().__init__(Sigma=Sigma)

    def retarded(self, energy):
        return self.Sigma[np.searchsorted(self.energies, energy)]


class StackSelfEnergy(ConstSelfEnergy):
    """Stack of selfenergies."""

    def __init__(self, selfenergies: list):
        for selfenergy in selfenergies:
            if hasattr(selfenergy, "Sigma"):
                shape = selfenergy.Sigma.shape
            if shape is None:
                raise RuntimeError("Cannot retreive shape of selfenergy matrix.")
        if len(shape) > 2:  # DataSelfEnergy
            shape = shape[1:]
        super().__init__(shape=shape)
        for e, selfenergy in enumerate(list(selfenergies)):
            if isinstance(selfenergy, np.ndarray):
                # substitute with ConstSelfenergy
                selfenergies.insert(e, ConstSelfEnergy(Sigma=selfenergies.pop(e)))
        self.selfenergies = selfenergies

    def retarded(self, energy):
        self.Sigma[:] = 0.0
        for selfenergy in self.selfenergies:
            self.Sigma += selfenergy.retarded(energy)
        return self.Sigma

    def lesser(self, energy):
        self.Sigma[:] = 0.0
        for selfenergy in self.selfenergies:
            if hasattr(selfenergy, "lesser"):
                self.Sigma += selfenergy.lesser(energy)
        return self.Sigma


def ZeroFilter(selfenergy: Union[BaseSelfEnergy, ConstSelfEnergy], func: Callable):

    Sigma = np.zeros_like(selfenergy.Sigma)

    def filter(self, energy):
        if func(energy):
            out = self._retarded(energy)
            return out
        else:
            return Sigma

    selfenergy._retarded = getattr(selfenergy, "retarded")
    selfenergy.retarded = MethodType(filter, selfenergy)
    return selfenergy
