import pickle
from math import pi, sqrt

import numpy as np
from ase.units import Bohr, Hartree
from qttools.gpaw.packed import unpack
from scipy.sparse.linalg import eigsh

from gpaw import GPAW
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.mpi import MASTER, serial_comm, world
from gpaw.poisson import FFTPoissonSolver, PoissonSolver
from gpaw.utilities import pack
from gpaw.utilities.blas import gemm, rk
from gpaw.utilities.tools import tri2full

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
    def __init__(self, gd, setups, spos_ac, fft=False):
        assert gd.comm.size == 1
        self.rhot1_G = gd.empty()
        self.rhot2_G = gd.empty()
        self.pot_G = gd.empty()
        self.dv = gd.dv
        if fft:
            self.poisson = FFTPoissonSolver()
        else:
            self.poisson = PoissonSolver(name="fd", nn=3, eps=1e-12)
        self.poisson.set_grid_descriptor(gd)
        self.setups = setups

        # Set coarse ghat
        self.Ghat = LFC(gd, [setup.ghat_l for setup in setups], integral=sqrt(4 * pi))
        self.Ghat.set_positions(spos_ac)

    def calculate(self, rhot_G, pot_G, P1_ap=None, P2_ap=None):
        I = 0.0
        if (P1_ap is not None) and (P2_ap is not None):
            I = self.atomic_corrections(P1_ap, P2_ap)
        I += np.vdot(rhot_G, pot_G) * self.dv
        return I * Hartree

    def atomic_corrections(self, P1_ap, P2_ap):
        I = 0.0
        for a, P1_p in P1_ap.items():
            P2_p = P2_ap[a]
            setup = self.setups[a]
            I += 2 * np.dot(P1_p, np.dot(setup.M_pp, P2_p))
        return I

    def poisson_solve(self, pot_G, rhot_G):
        """V(r) = \int dr \rho(r')/|r-r'|"""
        self.poisson.solve(pot_G, rhot_G, charge=None, zero_initial_phi=True)

    def add_compensation_charge(self, nt_G, P_ap):
        Q_aL = {}
        for a, P_p in P_ap.items():
            setup = self.setups[a]
            Q_aL[a] = np.dot(P_p, setup.Delta_pL)
        self.Ghat.add(nt_G, Q_aL)


def makeU_simple(calc, w_wG, P_awi, S_w=None, tolerance=1e-5):
    # TODO :
    # 1. parallelize.
    # 2. handle case S_w non identity.

    atoms = calc.atoms
    gd = calc.wfs.gd
    setups = calc.wfs.setups

    Nw = len(w_wG)

    # Make pairorbitals
    f_pG = gd.zeros(n=Nw ** 2)
    for p, (w1, w2) in enumerate(np.ndindex(Nw, Nw)):
        np.multiply(w_wG[w1], w_wG[w2], f_pG[p])
    assert f_pG.flags.contiguous

    # Make pairorbital overlap (lower triangle only)
    D_pp = np.zeros((Nw ** 2, Nw ** 2))
    rk(gd.dv, f_pG, 0.0, D_pp)

    # Add atomic corrections to pairorbital overlap
    # for atom in atoms:
    # a = atom.index
    # if setups[a].type != "ghost":
    #     P_pp = np.array(
    #         [
    #             pack(np.outer(P_awi[a][w1], P_awi[a][w2]))
    #             for w1, w2 in np.ndindex(Nw, Nw)
    #         ]
    #     )
    #     I4_pp = setups[a].four_phi_integrals()
    #     A = np.zeros((len(I4_pp), len(P_pp)))
    #     gemm(1.0, P_pp, I4_pp, 0.0, A, "t")
    #     gemm(1.0, A, P_pp, 1.0, D_pp)

    # Renormalize pair-orbital overlap matrix.
    if S_w is not None:
        S2 = np.sqrt(S_w)
        for pa, (wa1, wa2) in enumerate(np.ndindex(Nw, Nw)):
            for pb, (wb1, wb2) in enumerate(np.ndindex(Nw, Nw)):
                D_pp[pa, pb] /= S2[wa1] * S2[wa2] * S2[wb1] * S2[wb2]

    # Determine eigenvalues and vectors on master only
    eps_q, U_pq = np.linalg.eigh(D_pp, UPLO="L")
    indices = np.argsort(-eps_q.real)
    eps_q = np.ascontiguousarray(eps_q.real[indices])
    U_pq = np.ascontiguousarray(U_pq[:, indices])

    # Truncate
    indices = eps_q > tolerance
    U_pq = np.ascontiguousarray(U_pq[:, indices])
    eps_q = np.ascontiguousarray(eps_q[indices])

    return U_pq, eps_q


def makeV_simple(calc, w_wG, P_awi, eps_q, U_pq):
    # TODO : parallelize

    coulomb = Coulomb(calc.wfs.gd, calc.wfs.setups, calc.spos_ac, fft=True)

    # Make rotation matrix divided by sqrt of norm
    Nq = len(eps_q)
    Ni = len(w_wG)
    Uisq_iqj = (U_pq / np.sqrt(eps_q)).reshape(Ni, Ni, Nq).swapaxes(1, 2).copy()

    V_qq = np.zeros((Nq, Nq), float)

    def make_optimized(qstart, qend):
        # Make optimized pair orbitals
        g_qG = np.zeros((qend - qstart,) + w_wG.shape[1:], float)
        P_aqp = {}
        for a, P_wi in P_awi.items():
            ni = P_wi.shape[1]
            nii = ni * (ni + 1) // 2
            P_aqp[a] = np.zeros((qend - qstart, nii), float)
        for w1, w1_G in enumerate(w_wG):
            U = Uisq_iqj[w1, qstart:qend].copy()
            gemm(1.0, w1_G * w_wG, U, 1.0, g_qG)
            for a, P_wi in P_awi.items():
                P_wp = np.array(
                    [pack(np.outer(P_wi[w1], P_wi[w2])) for w2 in range(Ni)]
                )
                gemm(1.0, P_wp, U, 1.0, P_aqp[a])
        return g_qG, P_aqp

    g1_qG, P1_aqp = make_optimized(0, Nq)
    for q1 in range(Nq):
        P1_ap = dict([(a, P_qp[q1]) for a, P_qp in P1_aqp.items()])
        coulomb.add_compensation_charge(g1_qG[q1], P1_ap)

    V1_gG = np.empty_like(g1_qG)
    for q1 in range(Nq):
        coulomb.poisson_solve(V1_gG[q1], g1_qG[q1])

    for q1 in range(Nq):
        for q2 in range(q1, Nq):
            P1_ap = dict([(a, P_qp[q1]) for a, P_qp in P1_aqp.items()])
            P2_ap = dict([(a, P_qp[q2]) for a, P_qp in P1_aqp.items()])
            V_qq[q1, q2] = coulomb.calculate(g1_qG[q1], V1_gG[q2], P1_ap, P2_ap)
    tri2full(V_qq, UL="U")
    return V_qq


def makeU(
    gpwfile="scatt.gpw",
    orbitalfile="w_wG.npy",
    projectionfile="P_awi.pckl",
    rotationfile="U_pq.npy",
    eigvalsfile="eps_q.npy",
    tolerance=1e-5,
    low_rank_approx=False,
):
    # Tolerance is used for truncation of optimized pairorbitals
    calc = GPAW(gpwfile, txt=None)
    gd = calc.wfs.gd
    setups = calc.wfs.setups
    del calc

    if world.rank == MASTER:
        w_wG = np.load(orbitalfile)
        P_awi = pickle.load(open(projectionfile, "rb"))
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
        # Add atomic corrections to pairorbital overlap
        for a, P_wi in P_awi.items():
            P_cp = np.array([pack(np.outer(P_wi[w1], P_wi[w2])) for w1, w2 in ww_c])
            I4_pp = setups[a].four_phi_integrals()
            D_cc += np.dot(P_cp, np.dot(I4_pp, P_cp.T))

        if low_rank_approx:
            # Determine eigenvalues and vectors
            eps_q, U_cq = np.linalg.eigh(D_cc, UPLO="L")
            del D_cc

            # indices = np.argsort(-eps_q.real)
            # eps_q = eps_q.real[indices]
            # U_cq = U_cq[:, indices]

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
            eps_q, U_pq = eigsh(D_pp, len(D_pp) * 20 // 100, which="LA")
            # eps_q, U_pq = np.linalg.eigh(D_pp, UPLO="L")

            # indices = np.argsort(-eps_q.real)
            # eps_q = np.ascontiguousarray(eps_q.real[indices])
            # U_pq = np.ascontiguousarray(U_pq[:, indices])

            # Truncate
            indices = np.where(eps_q > tolerance)[0][::-1]
            U_pq = np.ascontiguousarray(U_pq[:, indices])
            eps_q = np.ascontiguousarray(eps_q[indices])

        np.save(rotationfile, U_pq)
        np.save(eigvalsfile, eps_q)


def makeV(
    gpwfile="scatt.gpw",
    orbitalfile="w_wG.npy",
    projectionfile="P_awi.pckl",
    rotationfile="U_pq.npy",
    eigvalsfile="eps_q.npy",
    coulombfile="V_qq.npy",
):

    # Extract data from files
    calc = GPAW(gpwfile, txt=None, communicator=serial_comm)
    coulomb = Coulomb(
        calc.wfs.gd,
        calc.wfs.setups,
        calc.spos_ac,
        fft=True if all(calc.atoms.pbc) else False,
    )
    w_wG = np.load(orbitalfile)
    P_awi = pickle.load(open(projectionfile, "rb"))
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
    Ntriu_r = np.fromiter(
        ((Nq - i) for i in range(Nq)), int
    )  # number of triu elems per row
    Ntriu = sum(Ntriu_r) // Ncpu  # tot number of triu elems div by number of cpu.
    nodes = np.arange(1, Ncpu + 1) * Ntriu  # round sequential triu elems cut per cpu
    nodes_r = abs(nodes[:, None] - np.cumsum(Ntriu_r)[None, :]).argmin(1) + 1
    nodes_r[-1] = Nq
    nq_r = np.empty(Ncpu, int)
    nq_r[0] = nodes_r[0]
    nq_r[1:] = np.diff(nodes_r)
    assert sum(nq_r) == Nq
    # nq, R = divmod(Nq, Ncpu)
    # nq_r = nq * np.ones(Ncpu, int)
    # if R > 0:
    #     nq_r[-R:] += 1

    # Determine number of opt. pairorb on this cpu
    nq1 = nq_r[world.rank]
    q1end = nq_r[: world.rank + 1].sum()
    q1start = q1end - nq1
    V_qq = np.zeros((Nq, Nq), float)

    def make_optimized(qstart, qend):
        g_qG = np.zeros((qend - qstart,) + w_wG.shape[1:], float)
        for w1, w1_G in enumerate(w_wG):
            gemm(1.0, w_wG * w1_G, Uisq_qij[qstart:qend, w1, :].copy(), 1.0, g_qG)
        P_aqp = {}
        for a, P_wi in P_awi.items():
            P_aqp[a] = np.dot(
                Uisq_qp[qstart:qend],
                np.array(
                    [
                        pack(np.outer(P_wi[w1], P_wi[w2]))
                        for w1, w2 in np.ndindex(Ni, Ni)
                    ]
                ),
            )
        return g_qG, P_aqp

    g1_qG, P1_aqp = make_optimized(q1start, q1end)
    for q1 in range(nq1):
        P1_ap = dict([(a, P_qp[q1]) for a, P_qp in P1_aqp.items()])
        coulomb.add_compensation_charge(g1_qG[q1], P1_ap)

    V1_gG = np.empty_like(g1_qG)
    for q1 in range(nq1):
        coulomb.poisson_solve(V1_gG[q1], g1_qG[q1])

    for block, nq2 in enumerate(nq_r):
        if block < world.rank:
            continue
        if block == world.rank:
            g2_qG, P2_aqp = g1_qG, P1_aqp
            q2start, q2end = q1start, q1end
        else:
            q2end = nq_r[: block + 1].sum()
            q2start = q2end - nq2
            g2_qG, P2_aqp = make_optimized(q2start, q2end)
            for q2 in range(nq2):
                P2_ap = dict([(a, P_qp[q2]) for a, P_qp in P2_aqp.items()])
                coulomb.add_compensation_charge(g2_qG[q2], P2_ap)

        for q1, q2 in np.ndindex(nq1, nq2):
            P1_ap = dict([(a, P_qp[q1]) for a, P_qp in P1_aqp.items()])
            P2_ap = dict([(a, P_qp[q2]) for a, P_qp in P2_aqp.items()])
            V_qq[q2 + q2start, q1 + q1start] = coulomb.calculate(
                g2_qG[q2], V1_gG[q1], P2_ap, P1_ap
            )

    world.sum(V_qq, MASTER)
    if world.rank == MASTER:
        # V can be slightly asymmetric due to numerics
        # V_qq = 0.5 * (V_qq + V_qq.T)
        tri2full(V_qq, UL="L")
        np.save(coulombfile, V_qq)
