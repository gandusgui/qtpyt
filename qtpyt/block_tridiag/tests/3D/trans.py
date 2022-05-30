from __future__ import print_function
import numpy as np
import pickle
from ase.io import read
import time
from gpaw import restart
from qtpyt.lcao.principallayer import PrincipalSelfEnergy
from qtpyt.greenfunction import RecursiveGF
from matplotlib import pyplot as plt

energies = np.arange(-4, 0, 0.1)


atoms = read("scatt.xyz")
patoms, pcalc = restart("leads.gpw", txt=None)
H_kMM, S_kMM = pickle.load(open("hs_leads_k.pckl", "rb"))
hs_list_ii, hs_list_ij = pickle.load(open("hs_scatt_lists.pckl", "rb"))

eta = 1e-5

RSE = [
    PrincipalSelfEnergy(pcalc, scatt=atoms, id=0),
    PrincipalSelfEnergy(pcalc, scatt=atoms, id=1),
]

for selfenergy in RSE:
    selfenergy.set(eta=eta)
    selfenergy.initialize(H_kMM, S_kMM)

RGF = RecursiveGF(selfenergies=RSE)

RGF.set(eta=eta, align_bf=0, gpu=False)
RGF.initialize(hs_list_ii, hs_list_ij)

# dV = np.load('dV.npy')
# RGF.add_screening(dV)

start = time.time()
T_e = RGF.get_transmission(energies)
end = time.time()

print("Elapsed time:", end - start)

# np.testing.assert_allclose(T_e, expected)

np.save("ET_long", (energies, T_e))
# pickle.dump((energies, T_e), open('ET.pckl', 'wb'), 2)
plt.plot(energies, T_e)
plt.savefig("ET.png")
plt.close()
