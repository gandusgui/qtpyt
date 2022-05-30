import pickle
from math import pi, sqrt

import numpy as np
from ase.units import Bohr, Hartree
from numpy.fft import fftn
from qttools.gpaw.packed import unpack
from scipy.sparse.linalg import eigsh

from gpaw import GPAW
from gpaw.coulomb import Coulomb
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.mpi import MASTER, serial_comm, world
from gpaw.poisson import FFTPoissonSolver, PoissonSolver
from gpaw.utilities import pack
from gpaw.utilities.blas import gemm, rk
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.tools import construct_reciprocal, tri2full

# http://www-personal.umich.edu/~lorenzon/classes/2004/Handouts/formula-final390-04.pdf
#
#     e^2
#   -------   = 1.44 eV nm
#  4 pi eps0

#
#      1         1
#  --------- = -----
#   |r -r'|    (Bohr)
unit_eV = 1.44 * 10 * 1 / Bohr  # Hartree

# References
#
# [1] https://arxiv.org/pdf/0910.1921.pdf

class Coulomb:
    """Class used to evaluate two index coulomb integrals."""
    def __init__(self, gd, poisson=None):
        """Class should be initialized with a grid_descriptor 'gd' from
           the gpaw module.
        """
        self.gd = gd
        self.poisson = poisson
        k2, N3 = construct_reciprocal(self.gd)
        self.scale = 4 * pi / (k2 * N3)
        gauss = Gaussian(self.gd)
        self.ng = gauss.get_gauss(0) / sqrt(4 * pi)
        self.vg = gauss.get_gauss_pot(0) / sqrt(4 * pi)

    def fftn(self, n_qG):
        Z_q = np.empty(n_qG.shape[0])
        n_kG = np.empty_like(n_qG)
        for q in range(n_qG.shape[0]):
            Z_q[q] = self.gd.integrate(n_qG[q])
            n_kG[q] = fftn(n_qG[q] - Z_q[q] * self.ng)
        return Z_q, n_kG

    def calculate(self, n1, n2, nk1, nk2, Z1, Z2):
        """Evaluates the coulomb integral of n1 and n2

        The coulomb integral is defined by::

                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------,
                      /    /         |r - r'|


                                                   *          *    *
            (n1|n2) = (n1 - Z1 ng|n2 - Z2 ng) + (Z2 n1 + Z1 n2 - Z1 Z2 ng | ng)

        """
        # Determine total charges

        # Determine the integrand of the neutral system
        # (n1 - Z1 ng)* int dr'  (n2 - Z2 ng) / |r - r'|
        I = nk1.conj() * nk2 * self.scale

        # add the corrections to the integrand due to neutralization
        I += (np.conj(Z1) * n2 + Z2 * n1.conj() -
              np.conj(Z1) * Z2 * self.ng) * self.vg
            
        return np.real(self.gd.integrate(I)) * Hartree


def makeU(
    gpwfile="scatt.gpw",
    orbitalfile="w_wG.npy",
    rotationfile="U_pq.npy",
    eigvalsfile="eps_q.npy",
    tolerance=1e-5,
    low_rank_approx=False,
):
    # Tolerance is used for truncation of optimized pairorbitals
    calc = GPAW(gpwfile, txt=None)
    gd = calc.wfs.gd
    # setups = calc.wfs.setups
    del calc

    if world.rank == MASTER:
        w_wG = np.load(orbitalfile)
        # P_awi = pickle.load(open(projectionfile, "rb"))
        Nw = w_wG.shape[0]
    else:
        w_wG = None
        Nw = 0
    if world.size > 1:
        Nw = gd.comm.sum(Nw)
        w_local_wG = gd.empty(n=Nw)
        gd.distribute(w_wG, w_local_wG)
    else:
        w_local_wG = w_wG
    del w_wG

    # unpack from compressed to wannier-wannier indices
    ww_c = [(w1, w2) for w1 in range(Nw) for w2 in range(w1, Nw)]
    Nc = Nw * (Nw + 1) // 2  # Number of compressed pair orbital indices

    # Make pairorbitals
    f_cG = gd.zeros(n=Nc)
    for f_G, (w1, w2) in zip(f_cG, ww_c):
        np.multiply(w_local_wG[w1], w_local_wG[w2], f_G)
    del w_local_wG

    # Make pairorbital overlap (lower triangle only)
    D_cc = np.zeros((Nc, Nc))
    rk(gd.dv, f_cG, 0.0, D_cc)
    if world.size > 1:
        gd.comm.sum(D_cc, MASTER)
    del f_cG
    if world.rank != MASTER:
        del D_cc

    if world.rank == MASTER:

        if low_rank_approx:
            # Determine eigenvalues and vectors
            eps_q, U_cq = np.linalg.eigh(D_cc, UPLO="L")
            del D_cc

            # Truncate
            indices = np.where(eps_q > tolerance)[0][::-1]
            U_cq = np.ascontiguousarray(U_cq[:, indices])
            eps_q = np.ascontiguousarray(eps_q[indices])
            Nq = len(eps_q)

            # Make rotation matrices and unpack the reverse rotation from
            # compressed format
            U_pq = np.empty((Nw ** 2, Nq))
            for c, U_q in enumerate(U_cq):
                i1 = int(Nw + 0.5 - np.sqrt((Nw - 0.5) ** 2 - 2 * (c - Nw)))
                i2 = c - i1 * (2 * Nw - 1 - i1) // 2
                U_pq[i1 + i2 * Nw] = U_pq[i2 + i1 * Nw] = U_q

        else:
            # Unpack from compressed format
            D_pp = unpack(D_cc, Nw, UL="L")
            del D_cc

            # Determine eigenvalues and vectors on master only
            eps_q, U_pq = eigsh(D_pp, len(D_pp)*20//100, which='LA')

            # Truncate
            indices = np.where(eps_q > tolerance)[0][::-1]
            U_pq = np.ascontiguousarray(U_pq[:, indices])
            eps_q = np.ascontiguousarray(eps_q[indices])

        np.save(rotationfile, U_pq)
        np.save(eigvalsfile, eps_q)


def makeV(
    gpwfile="scatt.gpw",
    orbitalfile="w_wG.npy",
    rotationfile="U_pq.npy",
    eigvalsfile="eps_q.npy",
    coulombfile="V_qq.npy",
):

    # Extract data from files
    calc = GPAW(gpwfile, txt=None, communicator=serial_comm)
    coulomb = Coulomb(calc.wfs.gd)
    w_wG = np.load(orbitalfile)
    eps_q = np.load(eigvalsfile)
    U_pq = np.load(rotationfile)
    del calc

    # Make rotation matrix divided by sqrt of norm
    Nq = len(eps_q)
    # Np = len(U_pq)
    Ni = len(w_wG)
    Uisq_qp = (U_pq / np.sqrt(eps_q)).T.copy()
    Uisq_qij = Uisq_qp.reshape(Nq, Ni, Ni)
    del eps_q, U_pq

    # Determine number of opt. pairorb on each cpu
    Ncpu = world.size
    # Ntriu_r = np.fromiter(
    #     ((Nq - i) for i in range(Nq)), int
    # )  # number of triu elems per row
    # Ntriu = sum(Ntriu_r) // Ncpu  # tot number of triu elems div by number of cpu.
    # nodes = np.arange(1, Ncpu + 1) * Ntriu  # round sequential triu elems cut per cpu
    # nodes_r = abs(nodes[:, None] - np.cumsum(Ntriu_r)[None, :]).argmin(1) + 1
    # nodes_r[-1] = Nq
    # nq_r = np.empty(Ncpu, int)
    # nq_r[0] = nodes_r[0]
    # nq_r[1:] = np.diff(nodes_r)
    # assert sum(nq_r) == Nq
    nq, R = divmod(Nq, Ncpu)
    nq_r = nq * np.ones(Ncpu, int)
    if R > 0:
        nq_r[-R:] += 1

    # Determine number of opt. pairorb on this cpu
    nq1 = nq_r[world.rank]
    q1end = nq_r[: world.rank + 1].sum()
    q1start = q1end - nq1
    V_qq = np.zeros((Nq, Nq), float)

    def make_optimized(qstart, qend):
        g_qG = np.zeros((qend - qstart,) + w_wG.shape[1:], float)
        for w1, w1_G in enumerate(w_wG):
            gemm(1.0, w_wG * w1_G, Uisq_qij[qstart:qend, w1, :].copy(), 1.0, g_qG)
        return g_qG

    g1_qG = make_optimized(q1start, q1end)
    Z1_q, g1_kG = coulomb.fftn(g1_qG)

    for block, nq2 in enumerate(nq_r):
        if block < world.rank:
            continue
        if block == world.rank:
            g2_qG = g1_qG
            q2start, q2end = q1start, q1end
            Z2_q, g2_kG = Z1_q, g1_kG
        else:
            q2end = nq_r[: block + 1].sum()
            q2start = q2end - nq2
            g2_qG = make_optimized(q2start, q2end)
            Z2_q, g2_kG = coulomb.fftn(g2_qG)

        for q1, q2 in np.ndindex(nq1, nq2):
            V_qq[q1 + q1start, q2 + q2start] = coulomb.calculate(
                g1_qG[q1], g2_qG[q2], g1_kG[q1], g2_kG[q2], Z1_q[q1], Z2_q[q2])

    world.sum(V_qq, MASTER)
    if world.rank == MASTER:
        tri2full(V_qq, UL="U")
        np.save(coulombfile, V_qq)
