#!/usr/bin/env python3

from gpaw import GPAW
from gpaw.lcao.tools import tri2full
from gpaw.lcao.tools import remove_pbc
from ase.units import Hartree


def get_h_and_s(calc, kpt=0):
    calc.atoms.set_calculator(calc)
    if calc.wfs.S_qMM is None:
        calc.wfs.set_positions(calc.spos_ac)
    h = calc.wfs.eigensolver.calculate_hamiltonian_matrix(
        hamiltonian=calc.hamiltonian, wfs=calc.wfs, kpt=calc.wfs.kpt_u[kpt]
    )
    s = calc.wfs.S_qMM[kpt]
    tri2full(h)
    tri2full(s)
    h *= Hartree
    h -= calc.get_fermi_level() * s
    remove_pbc(atoms=calc.atoms, h=h, s=s)
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
