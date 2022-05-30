#!/usr/bin/env python

from gpaw import restart
from gpaw.lcao.tools import get_lead_lcao_hamiltonian, remove_pbc
import numpy as np


def get_h_and_s(calc, direction="x"):
    """
        The function below performs
            1. tri2full
            2. *= Hartree
            3. stack [0,0,0],[1,0,0]
    """
    ibz_t_kc, weight_t_k, h_skmm, s_kmm = get_lead_lcao_hamiltonian(calc)
    """
        The code below performs
            1. remove_pbc
            2. align_fermi
    """
    d = "xyz".index(direction)
    nspins, nkpts = h_skmm.shape[:2]
    for s in range(nspins):
        for k in range(nkpts):
            if s == 0:
                remove_pbc(atoms, h_skmm[s, k], s_kmm[k], d)
            else:
                remove_pbc(atoms, h_skmm[s, k], None, d)
            h_skmm[s, k] -= s_kmm[k] * calc.occupations.get_fermi_level()
    return h_skmm, s_kmm


if __name__ == "__main__":
    import sys
    from os.path import basename

    filename = sys.argv[1]
    atoms, calc = restart(filename, txt=None)
    atoms.set_calculator(calc)
    h_skmm, s_kmm = get_h_and_s(calc)
    np.save("hs_{}_sk".format(basename(filename).split(".")[0]), (h_skmm, s_kmm))
