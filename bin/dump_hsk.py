#!/usr/bin/env python3

from gpaw import GPAW
from ase.units import Hartree
from ase import units
from qtpyt.lcao.tk_lcao import h_and_s
from qtpyt.tk_gpaw import initialize_calculator


def get_h_and_s(calc, spin=0, kpt=0):
    initialize_calculator(calc)
    H_kMM, S_kMM = h_and_s(calc)
    # Convert in Hartree units
    H_kMM *= units.Hartree
    # Align Fermi Level
    H_kMM -= calc.fermi * S_kMM
    return H_kMM, S_kMM


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
