from __future__ import print_function
import numpy as np
import pickle as pickle
from qtpyt.calculators import TransportCalculator
from matplotlib import pyplot as plt
from qtpyt.mpi import rank

energies = np.arange(-2.0, 2.0, 1.0)
H, S = pickle.load(open("hs_scat.pckl", "rb"))
H1, S1 = pickle.load(open("hs_princ.pckl", "rb"))

tcalc = TransportCalculator(
    h=H.astype(np.complex64),
    s=S.astype(np.complex64),
    h1=H1.astype(np.complex64),
    s1=S1.astype(np.complex64),
    align_bf=0,
)
tcalc.initialize()
tcalc.set(energies=energies)
T = tcalc.get_transmission()

if rank == 0:
    pickle.dump((energies, T), open("ET.pckl", "wb"), 2)
    plt.plot(energies, T)
    plt.savefig("ET.png")
    plt.close()
