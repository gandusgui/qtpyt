#!/usr/bin/env python3

from gpaw import GPAW
from gpaw.lcao.tools import remove_pbc, get_lead_lcao_hamiltonian
from ase.units import Hartree


def get_h_and_s(calc, spin=0, kpt=0):
    calc.atoms.set_calculator(calc)
    if calc.wfs.S_qMM is None:
        calc.wfs.set_positions(calc.spos_ac)
    tkpts_kc, tweights_k, H_skMM, S_kMM = get_lead_lcao_hamiltonian(calc, direction="x")
    fermi = calc.get_fermi_level()
    s = S_kMM[kpt]
    h = H_skMM[spin, kpt] - fermi * s
    remove_pbc(calc.atoms, h, s, d=0)
    return h, s


if __name__ == "__main__":
    import pickle
    import sys
    from os.path import basename

    filename = sys.argv[1]
    calc = GPAW(filename, txt=None)
    h, s = get_h_and_s(calc)
    pickle.dump(
        (h, s), open("hs_{}.pckl".format(basename(filename).split(".")[0]), "wb"), 2
    )
