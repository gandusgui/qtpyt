import numpy as np
from qtpyt.surface.unfold import bloch_unfold


def tritoeplz_expand(N, A, B, C=None, out=None):
    """Tile a pricipal layer along a semi infinite direction `N` times.
    
    Args:
        N : # of repetitions
        A : Diagonal block
        B : Upper diagonal block
    """
    # C = B.T.conj()
    #
    #   A B
    #   C A B
    #   0  C A B
    #        .  .  .
    n = A.shape[0]

    if C is None:
        C = B.T.conj()

    if out is None:
        out = np.zeros((N * n, N * n), dtype=A.dtype)

    out = out.reshape(N, n, N, n)
    for i in range(N - 1):
        out[i, :, i, :] = A
        out[i, :, i + 1, :] = B
        out[i + 1, :, i, :] = C
    out[N - 1, :, N - 1, :] = A

    return out.reshape(n * N, n * N)


def tritoeplz_expand_bounds(N, B, id="left", out=None):
    """"""
    n = B.shape[0]

    if out is None:
        out = np.zeros((n * N, n * N), dtype=B.dtype)

    out[:] = 0.0
    out = out.reshape(N, n, N, n)
    if id == "left":
        out[-1, :, 0, :] = B
    else:
        out[0, :, -1, :] = B
    return out.reshape(n * N, n * N)


def seminf_expand(N, hs_ii, hs_ij, sigmaL, sigmaR, z, hs_ji=None, out=None):
    """Exapand the green's function along a semi infinite direction."""
    #
    #   G00  G01  G02  G03 .
    #   G10  G00  G01  G02 .
    #   G20  G10  G00  G01 .
    #   G30  G20  G10  G00 .
    #    .    .    .    .  .
    #
    h_ii, s_ii = hs_ii
    h_ij, s_ij = hs_ij
    if hs_ji is None:
        hs_ji = h_ij.T.conj(), s_ij.T.conj()
    h_ji, s_ji = hs_ji

    n = h_ii.shape[0]

    if out is None:
        out = np.zeros((N * n, N * n), dtype=complex)

    out = out.reshape(N, n, N, n)

    m_ii = z * s_ii - h_ii
    m_ij = z * s_ij - h_ij
    m_ji = z * s_ji - h_ji

    G = np.linalg.inv(m_ii - sigmaL - sigmaR)
    Y = np.linalg.solve(m_ii - sigmaL, m_ij)
    X = np.linalg.solve(m_ii - sigmaR, m_ji)

    out[0, :, 0, :] = out[N - 1, :, N - 1, :] = G

    for i in range(1, N):
        out[i, :, 0, :] = -X.dot(out[i - 1, :, 0, :])

    for i in range(N - 2, -1, -1):
        out[i, :, N - 1, :] = -Y.dot(out[i + 1, :, N - 1, :])

    out = out.reshape(n * N, n * N)

    for i in range(1, N):
        out[i * n : (i + 1) * n, n : (i + 1) * n] = out[(i - 1) * n : i * n, : i * n]

    for i in range(N - 3, -1, -1):
        out[i * n : (i + 1) * n, (i + 1) * n : (N - 1) * n] = out[
            (i + 1) * n : (i + 2) * n, (i + 2) * n : N * n
        ]

    return out

