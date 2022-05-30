import numpy as np
from scipy import linalg as la

# Cutoff for cutting block matrices
from .tridiagonal import cutoff
from qtpyt.tools import dagger

# import cupy as cp
from transport import xp


def recursive_gf(mat_list_ii, mat_list_ij, mat_list_ji, s_in=None, dos=False):

    # xp = cp.get_array_module(mat_list_ii[0][0])

    N = len(mat_list_ii)
    #     mat_shapes = [mat.shape[0] for mat in mat_list_ii]

    # np.matrix alias
    m_qii = mat_list_ii
    m_qij = mat_list_ij
    m_qji = mat_list_ji

    # Left connected green's function
    grL_qii = [None for _ in range(N)]
    # Initalize
    grL_qii[0] = xp.linalg.inv(m_qii[0])  # ([eS-H]_11-Sigma_L)^-1
    # First row green's function
    gr_1i = grL_qii[0].copy()

    # assert xp == cp.get_array_module(gr_1i)

    # Left connected recursion
    for q in range(1, N):
        # Left
        grL_qii[q] = xp.linalg.inv(
            m_qii[q] - xp.dot(xp.dot(m_qji[q - 1], grL_qii[q - 1]), m_qij[q - 1])
        )
        # 1st row
        gr_1i = xp.dot(xp.dot(gr_1i, m_qij[q - 1]), grL_qii[q])

    # Only transport
    if not dos:
        # Return g1N
        return gr_1i

    # Full green's function
    Gr_qii = [None for _ in range(N)]
    Gr_qji = [None for _ in range(N - 1)]
    Gr_qij = [None for _ in range(N - 1)]
    # Initialize
    Gr_qii[-1] = grL_qii[
        -1
    ]  # G_NN = gL_NN = ([eS-H]_NN - [eS-H]_NN-1 * grL_N-1N-1 * [eS-H]_N-1N - Sigma_R)^-1

    # Full recursion
    for q in range(N - 2, -1, -1):
        Gr_qji[q] = -xp.dot(Gr_qii[q + 1], xp.dot(m_qji[q], grL_qii[q]))
        Gr_qij[q] = -xp.dot(grL_qii[q], xp.dot(m_qij[q], Gr_qii[q + 1]))
        Gr_qii[q] = grL_qii[q] - xp.dot(grL_qii[q], xp.dot(m_qij[q], Gr_qji[q]))

    # Return retarded
    if s_in is None:
        # DOS
        return gr_1i, Gr_qii, Gr_qij, Gr_qji

    # Electron correlation function
    if isinstance(s_in, list):

        gnL_qii = [None for _ in range(N)]
        # Initalize
        gnL_qii[0] = grL_qii[0] @ s_in[0] @ np.conj(grL_qii[0])

        # Left connected recursion
        for q in range(1, N):
            sl = m_qji[q - 1] @ gnL_qii[q - 1] @ np.conj(m_qij[q - 1])
            if s_in[q] is not None:
                sl += s_in[q]
            gnL_qii[q] = grL_qii[q] @ sl @ np.conj(grL_qii[q])

        Gn_qii = [None for _ in range(N)]
        Gn_qij = [None for _ in range(N - 1)]
        Gn_qji = [None for _ in range(N - 1)]
        # Initialize
        Gn_qii[-1] = gnL_qii[-1]

        # Full recursion
        for q in range(N - 2, -1, -1):
            a = Gn_qii[q + 1] @ np.conj(m_qji[q]) @ np.conj(grL_qii[q])
            Gn_qji[q] = -Gr_qii[q + 1] @ m_qji[q] @ gnL_qii[q] - a
            Gn_qii[q] = (
                gnL_qii[q]
                + grL_qii[q] @ m_qij[q] @ a
                - gnL_qii[q] @ np.conj(m_qij[q]) @ np.conj(Gr_qji[q])
                - Gr_qij[q] @ m_qji[q] @ gnL_qii[q]
            )
            Gn_qij[q] = dagger(Gn_qji[q])

    # Return electron correlation
    return Gn_qii, Gn_qij, Gn_qji

    # # Right connected green's function
    # hgh_qii = [None for _ in range(N-1)]
    # # Initalize
    # grR_inv = m_qii[-1] # ([eS-H]_NN-Sigma_R)^-1
    #
    # # Right connected recursion
    # for q in range(N-2, -1, -1):
    #     # Left
    #     hgh_qii[q] = m_qij[q] @ la.solve(grR_inv, m_qji[q])
    #     grR_inv = m_qii[q] - hgh_qii[q]
    #
    # # Full green's function
    # Gr_qii = [None for _ in range(N)]
    # Gr_qii[-1] = grL_qii[-1]       #Actual gNN
    # Gr_qii[0]  = la.inv(grR_inv)   #Actual g11
    # for q in range(1, N-1):
    #     Gr_qii[q] = grL_qii[q] @ la.inv(
    #     np.eye(mat_shapes[q]) - grL_qii[q] @ hgh_qii[q])
    #
    # return Gr_qii, None, None
    #


def get_mat_lists(z, hs_list_ii, hs_list_ij, hs_list_ji, sigma_L=None, sigma_R=None):

    mat_list_ii = []
    mat_list_ij = []
    mat_list_ji = []
    # Upper
    h_list_ij, s_list_ij = hs_list_ij
    for h_ij, s_ij in zip(h_list_ij, s_list_ij):

        mat_list_ij.append(z * s_ij - h_ij)
    # Lower
    h_list_ji, s_list_ji = hs_list_ji
    for h_ji, s_ji in zip(h_list_ji, s_list_ji):

        mat_list_ji.append(z * s_ji - h_ji)
    # Diagonal
    h_list_ii, s_list_ii = hs_list_ii
    for h_ii, s_ii in zip(h_list_ii, s_list_ii):
        mat_list_ii.append(z * s_ii - h_ii)
    # Add selfenergies
    if sigma_L is not None:
        # Left
        mat_list_ii[0] -= sigma_L
    if sigma_R is not None:
        # Right
        mat_list_ii[-1] -= sigma_R

    return mat_list_ii, mat_list_ij, mat_list_ji


def multiply(A_qii, A_qij, A_qji, B_qii, B_qij, B_qji):
    """Helper function to multiply two block tridiagonal
    matrices."""
    N = len(A_qii)
    # Diagonal sum
    AB_qii = [xp.dot(a, b) for a, b in zip(A_qii, B_qii)]
    # Upper diagonal sum
    for q in range(N - 1):
        AB_qii[q][:] += xp.dot(A_qij[q], B_qji[q])
    # Lower diagonal sum
    for q in range(1, N):
        AB_qii[q][:] += xp.dot(A_qji[q - 1], B_qij[q - 1])

    return AB_qii


def add_diagonal(A_qii, V):
    i0 = 0
    for A_ii in A_qii:
        i1 = i0 + len(A_ii)
        A_ii.flat[:: len(A_ii) + 1] += V[i0:i1]
        i0 = i1


def get_diagonal(A_qii):
    if xp.__name__ == "cupy":
        get = lambda a: xp.diagonal(a).get()
    else:
        get = lambda a: xp.diagonal(a)
    nao = sum(len(A) for A in A_qii)
    A_i = np.zeros(nao, A_qii[0].dtype)
    # Loop over diagonal elements
    i0 = 0
    for A_ii in A_qii:
        i1 = i0 + len(A_ii)
        A_i[i0:i1] = get(A_ii)  # np.diagonal(A_ii) #.diagonal()
        i0 = i1
    return A_i
