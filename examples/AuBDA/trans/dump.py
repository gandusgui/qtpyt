import numpy as np
from ase.io import read
from ase.units import Hartree
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER

atoms = read('scatt.xyz')

calc = GPAW(restart='scatt.gpw',txt=None)
atoms.set_calculator(calc)
calc.wfs.set_positions(calc.spos_ac)

fermi = calc.get_fermi_level()
print(fermi, file=open('fermi_scatt.txt','w'))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    H_kMM -= fermi * S_kMM
    np.save('hs_cc_k.npy', (H_kMM, S_kMM))
