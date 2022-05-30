import pickle

import numpy as np
from qtpyt.basis import Basis

from gpaw import restart
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.lcao.tools import basis_subset2
from gpaw.utilities.blas import r2k
from gpaw.utilities.tools import tri2full


def makeAOs(
    scatt,
    active=None,
    gpwfile="scatt.gpw",
    orbitalfile="w_wG.npy",
    projectionfile="P_awi.pckl",
    hsfile="hs_lcao_k.npy",
    xcfile="xc.npy",
    Fcorefile="Fcore.npy",
    indxscattfile="index_p.npy",
    indxactivefile="index_c.npy",
    largebasis=None,
    smallbasis=None
):

    atoms, calc = restart(gpwfile, txt=None)
    lcao = LCAOwrap(calc)

    # fermi = calc.get_fermi_level()
    H = lcao.get_hamiltonian()
    S = lcao.get_overlap()
    # H -= fermi * S

    basis = Basis(atoms, [setup.nao for setup in calc.wfs.setups])

    basis_p = basis[scatt]
    index_p = basis_p.get_indices()
    if active is not None:
        index_c = np.searchsorted(index_p, basis[active].get_indices())
    if smallbasis is not None:
        insmall = basis_subset2(atoms[scatt].symbols,largebasis,smallbasis)
        index_p = index_p[insmall]
        if active is not None:
            insmall = basis_subset2(atoms[active].symbols,largebasis,smallbasis)
            index_c = index_c[insmall]

    P_awi = lcao.get_projections(indices=index_p)
    w_wG = lcao.get_orbitals(indices=index_p)
    xc = lcao.get_xc(indices=index_p)
    Fcore = lcao.get_Fcore(indices=index_p)

    pickle.dump(P_awi, open(projectionfile, "wb"))
    np.save(indxscattfile, index_p)
    np.save(hsfile, (H[None, :], S[None, :]))
    np.save(xcfile, xc)
    np.save(orbitalfile, w_wG)
    np.save(Fcorefile, Fcore)
    if active is not None:
        np.save(indxactivefile, index_c)
