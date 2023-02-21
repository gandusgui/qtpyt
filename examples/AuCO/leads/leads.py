import numpy as np
from ase.io import read
from ase.units import Hartree
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER

atoms = read('leads.xyz')
basis = {'default':'sz(dzp)'}

calc = GPAW(h=0.2,
            xc='PBE',
            nbands='nao',
            convergence={'bands':'all'},
            basis=basis,
            occupations=FermiDirac(width=0.01),
            kpts=(5, 1, 1),
            mode='lcao',
            txt='leads.txt',
            mixer=Mixer(0.02, 5, weight=100.0),
            symmetry={'point_group': False, 'time_reversal': True})

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('leads.gpw')

fermi = calc.get_fermi_level()
print(repr(fermi), file=open('fermi_leads.txt', 'w'))

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
if rank == 0:
    H_kMM = H_skMM[0]
    np.save('hs_pl_k.npy', (H_kMM, S_kMM))
