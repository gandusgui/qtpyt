import numpy as np

from qtpyt import xp
from qtpyt.base.greenfunction import BaseGreenFunction
from qtpyt.base.selfenergy import BaseSelfEnergy
from qtpyt.surface.principallayer import fill_sigma


class Hybridization(BaseSelfEnergy):
    """Compute hybridization of an orthogonal subspace.
    
    Example:
    
    hybrid = Hybridization(gf, indices)
    
    delta = hybrid.retarded(0.)
    h_eff = (hybrid.cut(gf.H) + delta).real
    
    """

    def __init__(self, gf : BaseGreenFunction) -> None:
        super().__init__(gf, (gf.H, None))

    def retarded(self, energy):
        """Get retarded hybridization."""
        #          -1        -1
        # /\(z) = g (z)  -  G (z)
        #          0
        if energy != self.energy:
            self.energy = energy
            Ginv = np.linalg.inv(self.gf.retarded(energy))
            fill_sigma(self.z(energy), self.H, self.S, Ginv, self.Sigma)
            
        return self.Sigma
