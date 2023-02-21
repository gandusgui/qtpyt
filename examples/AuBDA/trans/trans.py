from pathlib import Path
import numpy as np

from ase.io import read

from qtpyt.tools import remove_pbc, rotate_couplings
from qtpyt.block_tridiag import graph_partition, greenfunction
from qtpyt.surface.principallayer import PrincipalSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.basis import Basis
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.screening.distgf import DistGreenFunction

pl_path = Path('../leads/')
cc_path = Path('./')

h_pl_k, s_pl_k = np.load(pl_path/'hs_pl_k.npy')
h_cc_k, s_cc_k = map(lambda m: m.astype(complex), np.load(cc_path/f'hs_cc_k.npy'))

basis = {'Au':6,'H':1,'C':4,'N':4}

atoms_pl = read(pl_path/'leads.xyz')
basis_pl = Basis.from_dictionary(atoms_pl, basis)

atoms_cc = read(cc_path/'scatt.xyz')
basis_cc = Basis.from_dictionary(atoms_cc, basis)

kpts_t, h_pl_kii, s_pl_kii, h_pl_kij, s_pl_kij = prepare_leads_matrices(h_pl_k, s_pl_k, (5,5,3), align=(0,h_cc_k[0,0,0]))
remove_pbc(basis_cc, h_cc_k)
remove_pbc(basis_cc, s_cc_k)

Nr = (1,5,3)

se = [None, None]
se[0] = PrincipalSelfEnergy(kpts_t, (h_pl_kii, s_pl_kii), (h_pl_kij, s_pl_kij), Nr=Nr)
se[1] = PrincipalSelfEnergy(kpts_t, (h_pl_kii, s_pl_kii), (h_pl_kij, s_pl_kij), Nr=Nr, id='right')

rotate_couplings(basis_pl, se[0], Nr)
rotate_couplings(basis_pl, se[1], Nr)

# find block tridiagonal indices
nodes = graph_partition.get_tridiagonal_nodes(basis_cc, h_cc_k[0], len(atoms_pl.repeat(Nr)))

hs_list_ii, hs_list_ij = graph_partition.tridiagonalize(nodes, h_cc_k[0], s_cc_k[0])

de = 0.01
energies = np.arange(-4.,4.+de/2.,de).round(7)
eta = 1e-5

gf = greenfunction.GreenFunction(hs_list_ii, hs_list_ij, [(0, se[0]),
                                                          (len(hs_list_ii)-1, se[1])],
                                                          solver='coupling',
                                                          eta=eta)

gd = GridDesc(energies, 1)
T = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T[e] = gf.get_transmission(energy)

T = gd.gather_energies(T)
if comm.rank == 0:
    np.save(f'ET.npy', (energies, T))
