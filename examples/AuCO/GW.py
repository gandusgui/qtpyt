#!/usr/bin/env python

# build in
# from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np
from ase.io import read
# green's function
from qtpyt.base.greenfunction import GreenFunction
from qtpyt.base.leads import LeadSelfEnergy
from qtpyt.base.selfenergy import ConstSelfEnergy, StackSelfEnergy
from qtpyt.basis import Basis
# comm
from qtpyt.parallel import comm
from qtpyt.parallel.log import get_logger
from qtpyt.projector import ExpandSelfEnergy, ProjectedGreenFunction
# screening
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.screening.polarization import cGW
from qtpyt.screening.static import Fock, Hartree
# Surface
from qtpyt.surface.tools import prepare_leads_matrices
# remove PBC
from qtpyt.tools import remove_pbc
# self-consistent
from scipy.optimize import broyden1

pl_path = Path("../leads/")
cc_path = Path("./")

U_pq = np.load(cc_path / "U_pq.npy")
eps_q = np.load(cc_path / "eps_q.npy")
V_qq = np.load(cc_path / "V_qq.npy")
index_p = np.load(cc_path / "index_p.npy")
xc = np.load(cc_path / "xc.npy")
Fcore = np.load(cc_path / "Fcore.npy")

U_pq = (U_pq * np.sqrt(eps_q)).astype(complex)
V_qq = V_qq.astype(complex)

de = 0.025
energies = np.arange(-150.0, 150.0 + de / 2.0, de).round(7)
eta = 2 * de

h_pl_k, s_pl_k = np.load(pl_path / "hs_leads_k.npy")
h_cc_k, s_cc_k = map(lambda m: m.astype(complex), np.load(cc_path / f"hs_lcao_k.npy"))

# Fermi level.
mu = float(np.loadtxt(cc_path / "fermi_scatt.txt"))

basis = {"Au": 6, "O": 4, "C": 4}

bpl = Basis.from_dictionary(read(pl_path / "leads.txt"), basis)
bsc = Basis.from_dictionary(read(cc_path / "scatt.xyz"), basis)

h_pl_ii, s_pl_ii, h_pl_ij, s_pl_ij = map(
    lambda m: m[0],
    prepare_leads_matrices(h_pl_k, s_pl_k, (5, 1, 1), align=(0, h_cc_k[0, 0, 0]))[1:],
)

remove_pbc(bsc, h_cc_k, direction="x")
remove_pbc(bsc, s_cc_k, direction="x")

se = [None] * 2
se[0] = LeadSelfEnergy((h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), nbf_m=bsc.nao, eta=eta)
se[1] = LeadSelfEnergy(
    (h_pl_ii, s_pl_ii), (h_pl_ij, s_pl_ij), id="right", nbf_m=bsc.nao, eta=eta
)

gf = GreenFunction(
    h_cc_k[0],
    s_cc_k[0],
    selfenergies=[(slice(None), se[0]), (slice(None), se[1])],
    mu=mu,
    kt=8.6e-5 * 200.0,
    eta=eta,
)

index_c = np.searchsorted(index_p, bsc.get_indices([5, 6]))
gf0 = DistGreenFunction(ProjectedGreenFunction(gf, index_p), energies)
gf1 = gf0.take_subspace(index_c)

gw = cGW(gf1, V_qq, U_pq, oversample=10)
fock = Fock(gf1, V_qq, gw.U_cq)
hartree = Hartree(gf1, V_qq, gw.U_cq)
se_xc = ConstSelfEnergy(Sigma=-xc[np.ix_(index_c, index_c)])
se_Fcore = ConstSelfEnergy(Sigma=Fcore[np.ix_(index_c, index_c)])


# Global variable (local in mpi) array used in `store_dos`
DOS = np.empty_like(gf0.energies)
T = np.empty_like(gf0.energies)


def store_1d(e, energy):
    """Callback to store the DOS for the entire region.
    
    NOTE : we don't call the `get_dos` method directly
    cause we want to compute (and meoize) the retareded GF 
    anyway.
    """
    DOS[e] = gf.get_dos(energy)
    T[e] = gf.get_transmission(energy)


def save_1d(a, filename):
    """Save DOS for the entire region to `.npy` file."""
    # non-interacting DOS
    A = gf0.gather_energies(a)

    if comm.rank == 0:
        np.save(filename, (energies, A))


def save_pdos(filename):
    """Save projected density of states."""
    dos = gf1.get_dos()

    if comm.rank == 0:
        np.save(filename, (energies, dos))


# Logger for root process
log = get_logger(__name__, open_log=comm.rank == 0)

project = gf1.gf0.projector.project
S1 = gf1.gf0.S

gf1.update(callback=store_1d)
D0 = gf0.get_density_matrix()
D1 = project(D0)
N1 = 2.0 * D1.dot(S1).real.trace()
log.info(f"# of electrons : {N1}")
save_1d(DOS, "DOS_DFT.npy")
save_1d(T, "T_DFT.npy")
save_pdos("PDOS_DFT.npy")


se_corr = StackSelfEnergy([fock, hartree, se_xc, se_Fcore])
gf.selfenergies.append((slice(None), ExpandSelfEnergy(gf.S, se_corr, index_p[index_c])))

# HF
hartree.initialize(D0)

i = 0
def step(D0_inp):
    global i

    D1_inp = project(D0_inp)
    hartree.update(D1_inp)
    fock.update(D1_inp)

    gf1.update(callback=store_1d)
    D0_out = gf0.get_density_matrix()
    D1_out = project(D0_out)
    N1 = 2.0 * D1_out.dot(S1).real.trace()
    log.info(f"# of electrons : {N1}")
    save_1d(DOS, f"DOS_G{i}W{i}.npy")
    save_1d(T, f"DOS_G{i}W{i}.npy")
    save_pdos(f"PDOS_G{i}W{i}.npy")
    if comm.rank == 0: np.save('D0',D0_out)

    i += 1
    eps = D0_out - D0_inp

    return eps

D_inp = D0
D_out = broyden1(
    step, xin=D0, reduction_method="svd", max_rank=10, f_tol=0.1
)

# GW@HF
gw.update_correlation()
se_corr.selfenergies.append(gw)
gf1.update(callback=store_1d)
D0 = gf0.get_density_matrix()
D1 = project(D0)
N1 = 2.0 * D1.dot(S1).real.trace()
log.info(f"# of electrons : {N1}")
save_1d(DOS, f"DOS_G{i}W{i}.npy")
save_1d(T, f"DOS_G{i}W{i}.npy")
save_pdos(f"PDOS_G{i}W{i}.npy")
