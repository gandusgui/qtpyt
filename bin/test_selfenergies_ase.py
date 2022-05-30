#!/usr/bin/env python
# coding: utf-8

# In[11]:
from ase.transport.selfenergy import LeadSelfEnergy as ASE
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
    SE = [ASE(hs_ii, hs_ij, hs_ij, eta=eta), ASE(hs_ii, hs_ji, hs_ji, eta=eta)]
    return SE


def init_principal_selfenergies(pcalc, scalc=None):
    SE = [PS(pcalc, scatt=scalc, id=0, eta=eta), PS(pcalc, scatt=scalc, id=1, eta=eta)]
    for selfenergy in SE:
        selfenergy.initialize()
    return SE


def plot_sigma(sigma, color, marker, label):

    plt.plot(sigma.real, color + marker, label=label + "_real")
    plt.plot(sigma.imag, color + marker, label=label + "_imag")


def plot_compare(sigma_ase, sigma, label):

    plot_sigma(sigma_ase, "C1", "--", "ase")
    plot_sigma(sigma, "C2", "x", "my")
    plt.savefig("sigma_{}.png".format(label))
    plt.legend()
    plt.close()


def test_selfenergies(ASE, PSE, energies):

    sigma_aseL = np.zeros_like(energies, dtype=complex)
    sigmaL = np.zeros_like(energies, dtype=complex)
    sigma_aseR = np.zeros_like(energies, dtype=complex)
    sigmaR = np.zeros_like(energies, dtype=complex)

    for e, energy in enumerate(energies):
        sigma_aseL[e] = ASE[0].retarded(energy).trace()
        sigmaL[e] = PSE[0].retarded(energy).trace()

    for e, energy in enumerate(energies):
        sigma_aseR[e] = ASE[1].retarded(energy).trace()
        sigmaR[e] = PSE[1].retarded(energy).trace()

    plot_compare(sigma_aseL, sigmaL, "left")
    plot_compare(sigma_aseR, sigmaR, "right")


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
    ASE = init_ase_selfenergies(h, s)
    PSE = init_principal_selfenergies(pcalc)
    test_selfenergies(ASE, PSE, energies)
