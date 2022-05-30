import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, eigh
from scipy.constants import e, k

k_B = k / e  # Boltzmann constant [eV/K] 8.6173303e-05


def fd(energy, ef, temp):
    return 1.0 / (1.0 + np.exp((energy - ef) / (k_B * temp)))


def approximant(energy, poles, residues):

    arg = np.array(energy)
    ans = np.zeros(arg.shape) + 0.5

    for j in range(len(poles)):
        if poles[j] > 0:
            ans = (
                ans
                - residues[j] / (arg - 1j / poles[j])
                - residues[j] / (arg + 1j / poles[j])
            )

    return ans


def approximant_diff(energy, poles, residues):

    arg = np.array(energy)
    ans = np.zeros(arg.shape) + 0.5 * 0

    for j in range(len(poles)):
        if poles[j] > 0:
            ans = (
                ans
                - residues[j] / (arg - 1j / poles[j]) ** 2
                - residues[j] / (arg + 1j / poles[j]) ** 2
            )

    return ans


def poles_and_residues(cutoff=2):

    b_mat = [
        1 / (2.0 * np.sqrt((2 * (j + 1) - 1) * (2 * (j + 1) + 1)))
        for j in range(0, cutoff - 1)
    ]
    b_mat = np.matrix(np.diag(b_mat, -1)) + np.matrix(np.diag(b_mat, 1))

    poles, residues = eig(b_mat)

    residues = np.array(np.matrix(residues))
    # arg = np.argmax(np.abs(residues), axis=0)

    residues = 0.25 * np.array(
        [
            np.abs(residues[0, j]) ** 2 / (poles[j] ** 2)
            for j in range(residues.shape[0])
        ]
    )

    return poles, residues


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

    eigvas, eigvecs = eigh(a, b)

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


def integrate_pdos(G, zp, Rp, mu=0, T=300):  # nzp=100, R=1e10):

    # zp, Rp = zero_fermi(nzp)
    N = len(zp)  # nzp

    beta = 1 / (k_B * T)
    a_p = mu + 1j * zp / beta

    eta = G.eta
    G.eta = complex(0.0)

    R = 1e10
    mu_0 = 1j * R * G.apply_retarded(1j * R, G.S).diagonal()

    mu_1 = np.zeros(len(G.H), complex)
    for i in range(N):
        mu_1 += G.apply_retarded(a_p[i], G.S).diagonal() * Rp[i]
    mu_1 *= -1j * 4 / beta

    rho = np.real(mu_0) + np.imag(mu_1)

    G.eta = eta

    return rho


if __name__ == "__main__":

    a1, b1 = poles_and_residues(cutoff=2)
    a2, b2 = poles_and_residues(cutoff=10)
    a3, b3 = poles_and_residues(cutoff=30)
    a4, b4 = poles_and_residues(cutoff=50)
    a5, b5 = poles_and_residues(cutoff=100)

    energy = np.linspace(-5.7, 5.7, 3000)

    temp = 100
    fd0 = fd(energy, 0, temp)

    k_B = 8.61733e-5  # Boltzmann constant in eV
    energy = energy / (k_B * temp)

    ans1 = approximant(energy, a1, b1)
    ans2 = approximant(energy, a2, b2)
    ans3 = approximant(energy, a3, b3)
    ans4 = approximant(energy, a4, b4)
    ans5 = approximant(energy, a5, b5)

    plt.plot(fd0)
    plt.plot(ans1)
    plt.plot(ans2)
    plt.plot(ans3)
    plt.plot(ans4)
    plt.plot(ans5)
    plt.show()
