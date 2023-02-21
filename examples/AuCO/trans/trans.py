from pathlib import Path
import numpy as np

from ase.io import read

from qtpyt.tools import remove_pbc, expand_coupling
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.surface.tools import prepare_leads_matrices
from qtpyt.basis import Basis
from qtpyt.parallel import comm
from qtpyt.parallel.egrid import GridDesc
from qtpyt.screening.distgf import DistGreenFunction

pl_path = Path('../leads/')
cc_path = Path('./')

h_pl_k, s_pl_k = np.load(pl_path/'hs_pl_k.npy')
h_cc_k, s_cc_k = map(lambda m: m.astype(complex), np.load(cc_path/f'hs_cc_k.npy'))

# No. basis functions per atom kind
basis = {'Au':6,'C':4,'O':4}

atoms_pl = read(pl_path/'leads.xyz')
basis_pl = Basis.from_dictionary(atoms_pl, basis)

atoms_cc = read(cc_path/'scatt.xyz')
basis_cc = Basis.from_dictionary(atoms_cc, basis)

# (5,1,1) k-pts as in DFT leads
# (0,h_cc_k[0,0,0]) align 1st basis in leads and scattering region
h_pl_ii, s_pl_ii, h_pl_ij, s_pl_ij = map(lambda m: m[0], prepare_leads_matrices(h_pl_k, s_pl_k, (5,1,1), align=(0,h_cc_k[0,0,0]))[1:])
remove_pbc(basis_cc, h_cc_k)
remove_pbc(basis_cc, s_cc_k)

se = [None, None]
se[0] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij)) # left lead
se[1] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), id='right') # right lead

# expand to dimension of scattering
expand_coupling(se[0], len(h_cc_k[0]))
expand_coupling(se[1], len(h_cc_k[0]), id='right')

de = 0.01
energies = np.arange(-4.,4.+de/2.,de).round(7)
eta = 1e-5

# slice(None) means that we've already expanded the leads to the scattering region
gf = GreenFunction(h_cc_k[0], s_cc_k[0], selfenergies=[(slice(None),se[0]),(slice(None),se[1])], eta=eta)

gd = GridDesc(energies, 1)
T = np.empty(gd.energies.size)

for e, energy in enumerate(gd.energies):
    T[e] = gf.get_transmission(energy)

T = gd.gather_energies(T)
if comm.rank == 0:
    np.save(f'ET', (energies, T))
