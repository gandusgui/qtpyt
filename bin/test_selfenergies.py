#!/usr/bin/env python
# coding: utf-8

# In[11]:
from qtpyt.selfenergy import LeadSelfEnergy as LSE
from qtpyt.lcao.principallayer import PrincipalSelfEnergy as PS
from qtpyt.tools import dagger
from gpaw import GPAW
import pickle
from matplotlib import pyplot as plt
import numpy as np

eta = 1e-5
energies = np.arange(-2, 0, 0.5)

# In[2]:
def get_ase_selfenergies(h, s):
    pl = len(h) // 2
    h_ii = h[:pl, :pl]
    s_ii = s[:pl, :pl]
    h_ij = h[:pl, pl : 2 * pl]
    s_ij = s[:pl, pl : 2 * pl]
    return (h_ii, s_ii), (h_ij, s_ij)


def init_ase_selfenergies(h, s):
    hs_ii, hs_ij = get_ase_selfenergies(h, s)
    hs_ji = [dagger(mat) for mat in hs_ij]
    SE = [LSE(hs_ii, hs_ij, hs_ij, eta=eta), LSE(hs_ii, hs_ji, hs_ji, eta=eta)]
    return SE


def init_principal_selfenergies(pcalc, scalc=None):
    SE = [PS(pcalc, scatt=scalc, id=0, eta=eta), PS(pcalc, scatt=scalc, id=1, eta=eta)]
    for selfenergy in SE:
        selfenergy.initialize()
    return SE


def plot_x(x, color, marker, label):

    if isinstance(x.dtype, (np.complex128, np.complex64)):
        plt.plot(x.real, color + marker, label=label + "_real")
        plt.plot(x.imag, color + marker, label=label + "_imag")
    else:
        plt.plot(x, color + marker, label=label)


def plot_compare(x_ase, x_my, label):

    plot_x(x_ase, "C1", "--", "ase")
    plot_x(x_my, "C2", "x", "my")
    plt.savefig("{}.png".format(label))
    plt.legend()
    plt.close()


def test_selfenergies(lse, pse, energies):

    dos_aseL = np.zeros_like(energies)
    dosL = np.zeros_like(energies)
    dos_aseR = np.zeros_like(energies)
    dosR = np.zeros_like(energies)
    sigma_aseL = np.zeros_like(energies, dtype=complex)
    sigmaL = np.zeros_like(energies, dtype=complex)
    sigma_aseR = np.zeros_like(energies, dtype=complex)
    sigmaR = np.zeros_like(energies, dtype=complex)

    for e, energy in enumerate(energies):
        dos_aseL[e] = lse[0].dos(energy)
        dosL[e] = pse[0].dos(energy)
        sigma_aseL[e] = lse[0].retarded(energy).trace()
        sigmaL[e] = pse[0].retarded(energy).trace()

    for e, energy in enumerate(energies):
        dos_aseR[e] = lse[1].dos(energy)
        dosR[e] = pse[1].dos(energy)
        sigma_aseR[e] = lse[1].retarded(energy).trace()
        sigmaR[e] = pse[1].retarded(energy).trace()

    plot_compare(dos_aseL, dosL, "dos_left")
    plot_compare(dos_aseR, dosR, "dos_right")
    plot_compare(sigma_aseL, sigmaL, "sigma_left")
    plot_compare(sigma_aseR, sigmaR, "sigma_right")


def get_filename(pattern):
    from glob import glob

    filename = glob(pattern)
    if len(filename) > 1:
        filenames = [repr(tu) for tu in enumerate(filename)]
        stdout = "Which file?\n {} \n:".format("\n".join(filenames))
        filename = filename[int(input(stdout))]
    else:
        filename = filename[0]
    return filename


if __name__ == "__main__":
    hs_filename = get_filename("*.pckl")
    pcalc = GPAW(get_filename("*.gpw"), txt=None)
    h, s = pickle.load(open(hs_filename, "rb"))
    lse = init_ase_selfenergies(h, s)
    pse = init_principal_selfenergies(pcalc)
    test_selfenergies(lse, pse, energies)
