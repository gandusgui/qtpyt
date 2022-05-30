#!/usr/bin/env python

import numpy as np
from qtpyt.surface.principallayer import PrincipalSelfEnergy

# Test surface self-energy 
# https://www.cambridge.org/core/services/aop-cambridge-core/content/view/BAFFA7FC03D7313FCD829EC0D9291767/9781139164313c8_p183-216_CBO.pdf/level_broadening.pdf
#

t0 = 1.0 # hopping
Ec = -2.0 * t0 # onsite


class SigmaAnalytic:
    """Analytic solution of 1D chain with hopping t0 and onsites Ec.
    
    """
    def __init__(self, t0, Ec):
        self.Ec = Ec
        self.t0 = t0

    def x(self, energy):
        return (energy - self.Ec) / (2 * self.t0)

    def y1(self, x):
        return (x - 1) + np.sqrt(x ** 2 - 2 * x)

    def y2(self, x):
        return (x - 1) - 1.0j * np.sqrt(2 * x - x ** 2)

    def y3(self, x):
        return (x - 1) - np.sqrt(x ** 2 - 2 * x)

    def retarded(self, energy):
        x = self.x(energy)
        if x <= 0:
            return self.y1(x)
        if x >= 2:
            return self.y3(x)
        return self.y2(x)


h0 = np.array([[Ec + 2 * t0]])
h1 = np.array([[-t0]])

s0 = np.eye(1)
s1 = np.zeros((1, 1))

hs_kii = (m[None, :].astype(complex) for m in (h0, s0))
hs_kij = (m[None, :].astype(complex) for m in (h1, s1))
kpts = np.zeros((1, 3))


def test_recursive():
    sep = PrincipalSelfEnergy(kpts, hs_kii, hs_kij)
    sea = SigmaAnalytic(t0, Ec)

    energies = np.linspace(-3 * t0, 3 * t0, 101, endpoint=True)
    sigma_analytic = np.empty(energies.size, complex)
    sigma_pl = np.empty(energies.size, complex)

    for e, energy in enumerate(energies):
        sigma_analytic[e] = sea.retarded(energy)
        sigma_pl[e] = sep.retarded(energy)

    np.testing.assert_allclose(sigma_analytic, sigma_pl, atol=1e-4)


if __name__ == "__main__":
    test_recursive()
