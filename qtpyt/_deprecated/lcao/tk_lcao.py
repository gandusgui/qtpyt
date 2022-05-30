import numpy as np

# from qtpyt.tools import tri2full
from gpaw.utilities.tools import tri2full
from ase.units import Hartree


def symm_reduce(bzk_kc):
    """This function reduces inversion symmetry along 1st dimension."""
    ibzk_kc = []
    bzk2ibzk_k = []
    for bzk_index, bzk_k in enumerate(bzk_kc):
        try:
            if bzk_k[np.nonzero(bzk_k)[0][0]] > 0:
                ibzk_kc.append(bzk_k)
                bzk2ibzk_k.append(bzk_index)
            else:
                continue
        # zero case
        except IndexError:
            ibzk_kc.append(bzk_k)
            bzk2ibzk_k.append(bzk_index)
    return np.array(ibzk_kc), np.array(bzk2ibzk_k)


def fourier_sum(A_kMM, k_kc, R_c, A_x):
    """This function evaluates fourier sum"""
    shape = A_kMM.shape
    if A_x is None:
        A_x = np.zeros(shape[1:], dtype=A_kMM.dtype)
    A_x.shape = np.prod(shape[1:])
    A_kx = A_kMM.reshape(shape[0], -1)
    phase_k = np.exp(2.0j * np.pi * np.dot(k_kc, R_c))
    np.sum(phase_k[:, None] * A_kx, axis=0, out=A_x)
    A_x.shape = shape[1:]  # A_MM
    A_MM = A_x
    return A_MM


def h_and_s(calc):
    """Return LCAO Hamiltonian and overlap matrix in fourier-space."""
    # Extract Bloch Hamiltonian and overlap matrix
    H_kMM = []
    S_kMM = []

    h = calc.hamiltonian
    wfs = calc.wfs
    kpt_u = wfs.kpt_u

    for kpt in kpt_u:
        H_MM = wfs.eigensolver.calculate_hamiltonian_matrix(h, wfs, kpt)
        S_MM = wfs.S_qMM[kpt.q]
        # XXX Converting to full matrices here
        tri2full(H_MM)
        tri2full(S_MM)
        H_kMM.append(H_MM)  # * Hartree)
        S_kMM.append(S_MM)

    # Convert to arrays
    H_kMM = np.array(H_kMM)
    S_kMM = np.array(S_kMM)

    return H_kMM, S_kMM


def build_surface(N_c, A_NMM):

    n_r, M, N = A_NMM.shape
    dtype = A_NMM.dtype
    mat = np.zeros((n_r, M, n_r, N), dtype=dtype)

    n, m = N_c
    A_nmMM = A_NMM.reshape(n, m, M, N)
    # Supercell row index
    count_r = 0
    for i, j in np.ndindex(n, m):
        row = A_nmMM[np.ix_(np.roll(range(n), i), np.roll(range(m), j))]
        row.shape = (n * m, M, N)
        # Supercell column index
        count_c = 0
        for elem in row:
            mat[count_r, :, count_c, :] = elem
            # Increment column in supercell
            count_c += 1
        # Increment row in supercell
        count_r += 1

    mat.shape = (n_r * M, n_r * N)
    return mat


def get_partial_tightbindings(principal_layer):
    """Construct tightbindings from partial bloch transformations."""
    from .tightbinding import TightBinding

    tbs = []
    # On-site
    H_kii = principal_layer.H_kii
    S_kii = principal_layer.S_kii
    # Nearest neighbor
    H_kij = principal_layer.H_kij
    S_kij = principal_layer.S_kij
    # Number of k-points remaining from bartial Bloch sum
    Nk = H_kii.shape[0]
    # Iterate
    for k in range(Nk):
        Hk_NMM = np.stack([H_kii[k], H_kij[k], H_kij[k].T.conj()])
        Sk_NMM = np.stack([S_kii[k], S_kij[k], S_kij[k].T.conj()])
        tbs.append(TightBinding.from_real_space(Hk_NMM, Sk_NMM, N_c=(3, 1, 1)))
    return tbs


def encode_colors(psi_knM):
    from matplotlib import pyplot as plt

    n_k, n_n, n_M = psi_knM.shape
    colors_Mc = plt.cm.jet(np.linspace(0, 1, n_M))
    colors_knc = np.zeros((n_k, n_n, 4))
    for k, n in np.ndindex(n_k, n_n):
        # Avg rgb as
        # r = w0*r[0] + .. wN*r[N]
        # g = w0*g[0] + .. wN*g[N]
        # b = w0*b[0] + .. wN*b[N]
        proj = (psi_knM[k, n] * psi_knM[k, n].conj()).real
        proj /= sum(proj)
        colors_knc[k, n] = (proj[:, None] * colors_Mc).sum(0)

    # No opacity
    colors_knc[:, :, -1] = 1.0
    colors_knc = np.clip(colors_knc, 0.0, 1.0)

    return colors_knc, colors_Mc


def plot_colors(colors, ax=None):
    from matplotlib import pyplot as plt

    # Colors
    nc = colors.shape[0]
    lines = np.arange(nc)
    lines = np.tile(lines, (nc, 1)).T
    if ax is None:
        f, ax = plt.subplots()
    for i, line in enumerate(lines):
        ax.plot(line, c=colors[i])
    ax.set_yticks(ticks=range(len(colors)))
    return ax


def plot_proj_bands(kpts, eps_kn, colors_knc, ax=None):
    from matplotlib import pyplot as plt

    if ax is None:
        f, ax = plt.subplots()
    _ = ax.scatter(
        np.repeat(kpts[:, None], eps_kn.shape[1], axis=1).flat,
        eps_kn.flat,
        c=colors_knc.reshape(-1, 4),
    )
    return ax
