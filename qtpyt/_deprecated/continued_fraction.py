import numpy as np
import scipy.linalg as la

# Physical quantities
from scipy.constants import e, k


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

    eigvas, eigvecs = la.eigh(a, b)

    zp[:M] = eigvas

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


def integrate_dos(G, mu=0, T=300, nzp=100, R=1e10):

    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = complex(0.0)

    R = 1e10
    mu_0 = 1j * R * G.apply_overlap(1j * R, trace=True)

    mu_1 = complex(0)
    for i in range(N):
        mu_1 += G.apply_overlap(a_p[i], trace=True) * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


def integrate_pdos(G, mu=0, T=300, nzp=100, R=1e10):

    zp, Rp = zero_fermi(nzp)
    N = nzp

    k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05
    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = complex(0.0)

    R = 1e10
    # SGS = G.S @ G.apply_overlap(1j*R)
    mu_0 = (
        1j * R * G.apply_overlap(1j * R, diag=True)
    )  # SGS.diagonal() / G.S.diagonal()

    mu_1 = np.zeros(len(mu_0), complex)
    for i in range(N):
        # SGS = G.S @ G.apply_overlap(a_p[i])
        mu_1 += (
            G.apply_overlap(a_p[i], diag=True) * Rp[i]
        )  # SGS.diagonal() / G.S.diagonal() * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho
