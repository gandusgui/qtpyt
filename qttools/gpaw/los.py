import pickle

import numpy as np
from ase.io import write
from ase.units import Hartree
from qtpyt.basis import Basis
from qtpyt.lo.tools import lowdin_rotation, rotate_matrix, subdiagonalize_atoms

from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.utilities.blas import r2k
from gpaw.utilities.tools import tri2full


def makeLOs(
    scatt,
    active,
    gpwfile="scatt.gpw",
    orbitalfile="w_wG.npy",
    xcfile="xc.npy",
    indxscattfile="index_p.npy",
    indxactivefile="index_c.npy",
    hsfile="hs_lolw_k.npy",
    lowdin=True,
):

    atoms, calc = restart(gpwfile, txt=None)
    lcao = LCAOwrap(calc)

    fermi = calc.get_fermi_level()
    H = lcao.get_hamiltonian()
    S = lcao.get_overlap()
    H -= fermi * S

    basis = Basis(atoms, [setup.nao for setup in calc.wfs.setups])

    basis_p = basis[scatt]
    index_p = basis_p.get_indices()
    index_c = basis_p.extract().take(active)

    Usub, eig = subdiagonalize_atoms(basis, H, S, a=scatt)

    H = rotate_matrix(H, Usub)
    S = rotate_matrix(S, Usub)

    if lowdin:
        Ulow = lowdin_rotation(H, S, index_p[index_c])

        H = rotate_matrix(H, Ulow)
        S = rotate_matrix(S, Ulow)

        U = Usub.dot(Ulow)
    else:
        U = Usub

    los = LOs(U[:, index_p].T, lcao)

    for w, w_G in enumerate(los.get_orbitals(index_c)):
        write(f"lo{w}.cube", atoms, data=w_G)

    w_wG = los.get_orbitals()
    np.save(orbitalfile, w_wG)
    np.save(indxscattfile, index_p)
    np.save(indxactivefile, index_c)
    np.save(xcfile, los.get_xc(w_wG))
    np.save(hsfile, (H[None, :], S[None, :]))


class LOs:
    """Local Orbital descriptor.
    
    Args:
        c_wM : (np.ndarray)
            expansion coefficients in LCAO basis.
        lcao : (gpaw::lcao::pwf2::LCAOwrap)
            lcao wrapper of GPAW calculation in LCAO mode.
    
    NOTE: c_wM can also have 1.'s at the positions of a specific AO basis
        in which case the orbital is simply the underlying AO, e.g.
        c_wM[i] = [0.,0.,...,1.,0.....]
    """

    def __init__(self, c_wM, lcao):
        self.lcao = lcao
        self.c_wM = np.ascontiguousarray(c_wM)
        self.Nw = c_wM.shape[0]

    def get_projections(self, indices=None, P_aMi=None):
        if P_aMi is None:
            P_aMi = self.lcao.get_projections()
        if indices is None:
            indices = range(self.Nw)
        c_iM = self.c_wM[indices]
        return {a: c_iM.dot(P_Mi) for a, P_Mi in P_aMi.items()}

    def get_orbitals(self, indices=None):
        if indices is None:
            indices = range(self.Nw)
        Ni = len(indices)
        w_iG = self.lcao.calc.wfs.gd.zeros(Ni, dtype=self.lcao.dtype)
        c_iM = self.c_wM[indices]
        self.lcao.calc.wfs.basis_functions.lcao_to_grid(c_iM, w_iG, q=-1)
        return w_iG

    def get_xc(self, w_wG=None):
        calc = self.lcao.calc
        if calc.density.nt_sg is None:
            calc.density.interpolate_pseudo_density()
        nt_sg = calc.density.nt_sg
        vxct_sg = calc.density.finegd.zeros(calc.wfs.nspins)
        calc.hamiltonian.xc.calculate(calc.density.finegd, nt_sg, vxct_sg)
        vxct_G = calc.wfs.gd.empty()
        calc.hamiltonian.restrict_and_collect(vxct_sg[0], vxct_G)

        # Integrate pseudo part
        if w_wG is None:
            w_wG = self.get_orbitals()
        Nw = len(w_wG)
        xc_ww = np.empty((Nw, Nw))
        r2k(0.5 * calc.wfs.gd.dv, w_wG, vxct_G * w_wG, 0.0, xc_ww)
        tri2full(xc_ww, "L")
        return xc_ww * Hartree
