from __future__ import print_function

import numpy as np
from ase.io import read
from ase.units import Hartree
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER

atoms = read("cc.xyz")

calc = GPAW(
    h=0.2,
    xc="PBE",
    basis="szp(dzp)",
    occupations=FermiDirac(width=0.0),
    kpts={"size": (1, 6, 1)},
    mode="lcao",
    txt="cc.txt",
    mixer=Mixer(0.1, 5, weight=100.0),
    # parallel=dict(band=2,
    #              domain=2,
    #              augment_grids=True,  # use all cores for XC/Poisson
    #              sl_auto=True),  	   # enable ScaLAPACK parallelization
    symmetry={"point_group": False, "time_reversal": False},
)

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write("cc.gpw")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open("fermi_cc.txt", "w"))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= calc.get_fermi_level() * S_kMM
    np.save("hs_cc_k.npy", (H_kMM, S_kMM))
