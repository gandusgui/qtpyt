from __future__ import print_function

import numpy as np
from ase.io import read
from ase.units import Hartree
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER

atoms = read("scatt.xyz")

calc = GPAW(
    h=0.2,
    xc="PBE",
    basis="szp(dzp)",
    occupations=FermiDirac(width=0.0),
    # kpts={'size': (1,1,1)},
    mode="lcao",
    txt="scatt.txt",
    mixer=Mixer(0.1, 5, weight=100.0),
    parallel=dict(
        band=2, augment_grids=True, sl_auto=True  # use all cores for XC/Poisson
    ),  # enable ScaLAPACK parallelization
    symmetry={"point_group": False, "time_reversal": False},
)

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write("scatt.gpw")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open("fermi_scatt.txt", "w"))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= calc.get_fermi_level() * S_kMM
    np.save("hs_scatt_k.npy", (H_kMM, S_kMM))
