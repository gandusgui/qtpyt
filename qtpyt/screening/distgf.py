from functools import lru_cache
from math import pi
from typing import TypeVar, Union

import numpy as np

from qtpyt import xp
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.projector import ProjectedGreenFunction
from qtpyt.screening.langreth import LangrethPair, assert_domain, change_domain
from qtpyt.screening.tools import greater_from_retarded, smooth

T = TypeVar("T", bound="DistGreenFunction")


class DistGreenFunction(LangrethPair):
    """Distributed green's function"""

    def __init__(
        self,
        gf: Union[ProjectedGreenFunction, GreenFunction],
        energies,
        parent: T = None,
    ) -> None:

        self.gf0 = gf
        self.parent = parent
        super().__init__(energies, gf.H.shape[0], complex)

    def update_from_parent(self, other):
        """Project from parent.
        
        Also update domain and array keys.
        
        Args:
            other : DistGreenFunction
                green's function defined on a larger
                domain that contains this.
        """
        project = self.gf0.projector.project
        old_k = tuple(self.arrays.keys())
        new_k = tuple(other.arrays.keys())
        for ok, nk in zip(old_k, new_k):
            for e in range(self.energies.size):
                self.arrays[ok][e] = project(other.arrays[nk][e])
        self.arrays = {new_k[0]: self.arrays[old_k[0]], new_k[1]: self.arrays[old_k[1]]}
        self.domain = other.domain

    @change_domain("e")
    def update(self, *, callback=None, lg=True):
        """Update r and <.
        
        Args:
            callback : callable, optional
                Called after each iteration, as callback(e, energy)
                
            lg : bool, optional
                Compute the < and > components directly.
                
        """
        # Simply project from parent
        if self.parent is not None:
            self.parent.update(callback=callback, lg=lg)
            self.update_from_parent(self.parent)
        # Actually compute
        else:
            asnumpy = lambda a: a  # Identity
            if xp.__name__ == "cupy":
                asnumpy = xp.asnumpy
            if callback is None:
                callback = lambda e, energy: None  # Identity

            def get(key):
                try:
                    self.arrays[key]
                except KeyError:
                    self.arrays[key] = self.arrays.pop("g")
                return self.arrays[key]

            Gr = get("r")
            Gl = get("l")
            for e, energy in enumerate(self.energies):
                callback(e, energy)
                Gr[e] = asnumpy(self.gf0.retarded(energy))
                Gl[e] = asnumpy(self.gf0.lesser(energy))
            if lg:
                self.domain = "e"
                self.convert_retarded()

    @assert_domain("e")
    def convert_retarded(self):
        """Override the retarded with the greater component."""
        greater_from_retarded(self.arrays["r"], self.arrays["l"])
        self.arrays["g"] = self.arrays.pop("r")

    @assert_domain("e")
    def get_density_matrix(self):
        """Density matrix.
        
        Integral of the lesser green's function.
        This is equivalent to -1/pi Im{Gr}.
        """
        gl = self.arrays["l"]
        # Gl = self.collect_energies(gl)
        # Gl = smooth(self.global_energies, Gl, 1.0, 6.0)
        # self.collect_orbitals(Gl, gl)
        return self.sum_energies(gl, pre=-1.0j / (2 * np.pi))

    @assert_domain("e")
    def get_dos(self, out=None, root=0):
        """Density of states.
        
        Energy resolved DOS computed from
        the spectral matrix multiplied by the overlap.
        """
        # g> - g< = gr - ga
        # A = Re[1.j * (gr - ga)]
        # DOS = Tr[A S] / (2 pi)
        if out is None:
            out = np.empty(self.energies.size)
        for e in range(self.energies.size):
            out[e] = (
                1.0j
                * (self.arrays["g"][e] - self.arrays["l"][e]).dot(self.gf0.S).trace()
            ).real / (2 * np.pi)
        return self.gather_energies(out, root)

    def get_num_electrons(self):
        """Get total number of electrons."""
        D = self.get_density_matrix()
        S = self.gf0.S
        # x2. degenerate spin
        return 2.0 * D.dot(S).real.trace()

    def take_subspace(self, indices):
        """Build a distributed green's function for a subspace.
        
        This instance will be set as the parent so that each call 
        to the subspace will automatically update this instance's arrays."""
        return DistGreenFunction(
            ProjectedGreenFunction(self.gf0, indices), self.global_energies, parent=self
        )
