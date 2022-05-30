from functools import singledispatch

import numpy as np
import scipy.linalg as la

# Physical quantities
from scipy.constants import e, k

from qtpyt.base._kernels import dagger, dottrace
from qtpyt.base.greenfunction import GreenFunction as GF
from qtpyt.block_tridiag.greenfunction import GreenFunction as BTGF

"""References:

http://www1.spms.ntu.edu.sg/~ydchong/teaching/08_contour_integration.pdf
"""


def zero_fermi(nzp):
    """Compute poles (zp) and residues (Rp) of fermi function."""
    N = nzp
    M = 2 * N

    A = np.zeros((2 * M, 2 * M))
    B = np.zeros((2 * M, 2 * M))

    zp = np.zeros(2 + M)
    Rp = np.zeros(2 + M)

    for i in range(1, M + 1):
        B[i, i] = 2 * i - 1

    for i in range(1, M):
        A[i, i + 1] = -0.5
        A[i + 1, i] = -0.5

    a = np.zeros(M * M)
    b = np.zeros(M * M)

    for i in range(M):
        for j in range(M):
            a[j * M + i] = A[i + 1, j + 1]
            b[j * M + i] = B[i + 1, j + 1]

    a.shape = (M, M)
    b.shape = (M, M)

    eigvals, eigvecs = la.eigh(a, b)

    zp[:M] = eigvals

    for i in range(M, 0, -1):
        zp[i] = zp[i - 1]

    for i in range(1, M + 1):
        zp[i] = 1.0 / zp[i]

    a = eigvecs.T.flatten()

    for i in range(0, M):
        Rp[i + 1] = -a[i * M] * a[i * M] * zp[i + 1] * zp[i + 1] * 0.250

    zp = -zp[1 : N + 1]
    Rp = Rp[1 : N + 1]

    return zp, Rp


@singledispatch
def integrate_dos(G, mu=0, T=300, nzp=100, R=1e10):

    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = 0.0

    R = 1e10
    mu_0 = 1j * R * dottrace(G.retarded(1j * R), G.S)

    mu_1 = complex(0)
    for i in range(N):
        mu_1 += dottrace(G.retarded(a_p[i]), G.S) * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


@integrate_dos.register(BTGF)
def _(G, mu=0, T=300, nzp=100, R=1e10):

    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = 0.0

    R = 1e10
    mu_0 = 1j * R * G.retarded(1j * R).dottrace(G.S)

    mu_1 = complex(0)  # np.zeros(len(mu_0),complex)
    for i in range(N):
        mu_1 += G.retarded(a_p[i]).dottrace(G.S) * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


@singledispatch
def integrate_pdos(G, mu=0, T=300, nzp=100, R=1e10):

    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = 0.0

    R = 1e10
    mu_0 = 1j * R * G.apply_retarded(1j * R, G.S).diagonal()

    mu_1 = complex(0)
    for i in range(N):
        mu_1 += G.apply_retarded(a_p[i], G.S).diagonal() * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


@integrate_pdos.register(BTGF)
def _(G, mu=0, T=300, nzp=100, R=1e10):

    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = 0.0

    R = 1e10
    mu_0 = 1j * R * G.retarded(1j * R).dotdiag(G.S)

    mu_1 = np.zeros(len(mu_0), complex)
    for i in range(N):
        mu_1 += G.retarded(a_p[i]).dotdiag(G.S) * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


def density_matrix(G, mu=0, T=300, nzp=100, R=1e10):
    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = 0.0

    R = 1e10
    mu_0 = 1j * R * G.apply_retarded(1j * R, G.S)

    mu_1 = complex(0)
    for i in range(N):
        mu_1 += G.apply_retarded(a_p[i], G.S) * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho
