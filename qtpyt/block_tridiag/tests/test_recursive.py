import numpy as np
from itertools import accumulate
from copy import deepcopy

from qtpyt.block_tridiag.recursive import *


def build_simple(m, n, real=False):
    if real:
        return np.random.random((m, n))
    return np.random.random((m, n)) + 1.0j * np.random.random((m, n))


def build_hermitian(m, real):
    a = build_simple(m, m, real)
    a += a.T.conj()
    a /= 2
    return a


get_nodes = lambda sizes: list(accumulate([0,] + sizes, lambda x, y: x + y))
get_ii = lambda M, nodes, i: M[nodes[i] : nodes[i + 1], nodes[i] : nodes[i + 1]]
get_ij = lambda M, nodes, i: M[nodes[i - 1] : nodes[i], nodes[i] : nodes[i + 1]]
get_ji = lambda M, nodes, i: M[nodes[i] : nodes[i + 1], nodes[i - 1] : nodes[i]]


def build_matrix(m_qii, m_qij, m_qji):
    sizes = list(mat.shape[0] for mat in m_qii)
    nodes = get_nodes(sizes)
    M = np.zeros(2 * (sum(sizes),), complex)

    for i in range(len(sizes)):
        mat = get_ii(M, nodes, i)
        mat[:] = m_qii[i]

    for i in range(1, len(sizes)):
        mat = get_ij(M, nodes, i)
        mat[:] = m_qij[i - 1]
        mat = get_ji(M, nodes, i)
        mat[:] = m_qji[i - 1]

    return M


def split_matrix(M, sizes):
    nodes = get_nodes(sizes)
    m_qii = []
    m_qij = []
    m_qji = []
    for i in range(len(sizes)):
        m_qii.append(get_ii(M, nodes, i))
    for i in range(1, len(sizes)):
        m_qij.append(get_ij(M, nodes, i))
        m_qji.append(get_ji(M, nodes, i))
    return m_qii, m_qij, m_qji


def build_setup(sizes=[3, 5, 3, 2], hermitian=False, real=False):
    nodes = get_nodes(sizes)

    m_qii = []
    m_qij = []
    m_qji = []

    z = np.random.random()  # + 1.e-5j
    for i in range(len(sizes)):
        if hermitian:
            s = build_hermitian(sizes[i], real)
            h = build_hermitian(sizes[i], real)
            m_qii.append(z * s - h)
            # m_qii.append(build_hermitian(sizes[i]))
        else:
            m_qii.append(build_simple(sizes[i], sizes[i], real))

    for i in range(1, len(sizes)):
        if hermitian:
            s = build_simple(sizes[i - 1], sizes[i], real)
            h = build_simple(sizes[i - 1], sizes[i], real)
            m_qij.append(z * s - h)
            m_qji.append(z * s.T.conj() - h.T.conj())
            # m_qij.append(build_simple(sizes[i-1],sizes[i]))
            # m_qji.append(m_qij[-1].T.conj())
        else:
            m_qij.append(build_simple(sizes[i - 1], sizes[i], real))
            m_qji.append(build_simple(sizes[i], sizes[i - 1], real))

    M = build_matrix(m_qii, m_qij, m_qji)

    return M, m_qii, m_qij, m_qji


def setup_transmission(M, m_qii, m_qij, m_qji, sizes, hermitian=False):
    nodes = get_nodes(sizes)
    if hermitian:
        sigma_L = build_hermitian(m_qii[0].shape[0])
        sigma_R = build_hermitian(m_qii[-1].shape[0])
    else:
        sigma_L = build_simple(*m_qii[0].shape)
        sigma_R = build_simple(*m_qii[-1].shape)
    sL = np.zeros_like(M)
    sR = np.zeros_like(M)
    mat = get_ii(sL, nodes, 0)
    mat[:] = sigma_L
    mat = get_ii(sR, nodes, len(sizes) - 1)
    mat[:] = sigma_R
    M -= sL + sR
    G = np.linalg.inv(M)  # -sL-sR)
    gL = 1.0j * (sL - sL.T.conj())
    gR = 1.0j * (sR - sR.T.conj())
    T_e = np.einsum("ij,jk,kl,lm->im", gL, dagger(G), gR, G, optimize=True).real.trace()
    m_qii[0] -= sigma_L
    m_qii[-1] -= sigma_R
    gamma_L = 1.0j * (sigma_L - sigma_L.T.conj())
    gamma_R = 1.0j * (sigma_R - sigma_R.T.conj())
    return T_e, gamma_L, gamma_R, gL, gR


def test_recursive_coupling_method():
    sizes = [30, 43, 21, 40, 25]
    M, m_qii, m_qij, m_qji = build_setup(sizes)
    g_1N = coupling_method_1N(deepcopy(m_qii), m_qij, m_qji)
    g_N1 = coupling_method_N1(deepcopy(m_qii), m_qij, m_qji)
    G = np.linalg.inv(M)
    expected_1N = G[: m_qii[0].shape[0], -m_qii[-1].shape[-1] :]
    expected_N1 = G[-m_qii[-1].shape[0] :, : m_qii[0].shape[0]]
    np.testing.assert_allclose(g_1N, expected_1N)
    np.testing.assert_allclose(g_N1, expected_N1)


def test_transmission_coupling_method():
    sizes = [30, 43, 21, 40, 25]
    M, m_qii, m_qij, m_qji = build_setup(sizes, hermitian=True)
    T_e, gamma_L, gamma_R = setup_transmission(M, m_qii, m_qij, m_qji, sizes)[:3]
    g_N1 = coupling_method_N1(deepcopy(m_qii), deepcopy(m_qij), deepcopy(m_qji))
    g_1N = coupling_method_1N(m_qii, m_qij, m_qji)
    T_e_N1 = np.einsum(
        "ij,jk,kl,lm->im", gamma_R, g_N1, gamma_L, dagger(g_N1), optimize=True
    ).real.trace()
    T_e_1N = np.einsum(
        "ij,jk,kl,lm->im", gamma_L, g_1N, gamma_R, dagger(g_1N), optimize=True
    ).real.trace()
    np.testing.assert_allclose(T_e_N1, T_e)
    np.testing.assert_allclose(T_e_1N, T_e)


def test_spectral_method():
    sizes = [2, 2, 2, 2]
    nodes = get_nodes(sizes)
    M, m_qii, m_qij, m_qji = build_setup(sizes, hermitian=True)
    _, gamma_L, gamma_R, gL, gR = setup_transmission(M, m_qii, m_qij, m_qji, sizes)
    G = np.linalg.inv(M)
    A1, A2 = spectral_method(m_qii, m_qij, m_qji, gamma_L, gamma_R)
    A = G.dot(gL + gR).dot(G.T.conj())
    np.testing.assert_allclose(A, 1.0j * (G - G.T.conj()))
    for i in range(len(sizes)):
        np.testing.assert_allclose(A1[0][i] + A2[0][i], get_ii(A, nodes, i))
    for i in range(1, len(sizes)):
        np.testing.assert_allclose(A1[1][i - 1] + A2[1][i - 1], get_ij(A, nodes, i))
        np.testing.assert_allclose(A1[2][i - 1] + A2[2][i - 1], get_ji(A, nodes, i))


def test_transport_spectral_method():
    sizes = [2, 2, 2, 2]
    M, m_qii, m_qij, m_qji = build_setup(sizes, hermitian=True)
    T_e, gamma_L, gamma_R, gL, gR = setup_transmission(M, m_qii, m_qij, m_qji, sizes)
    A1, A2 = spectral_method(m_qii, m_qij, m_qji, gamma_L, gamma_R)
    np.testing.assert_allclose(T_e, gamma_L.dot(A2[0][0]).trace())
    np.testing.assert_allclose(T_e, gamma_R.dot(A1[0][-1]).trace())


# def test_transmission_overlap_method():
#     sizes = [3,2,5,4,4]
#     M, m_qii, m_qij, m_qji = build_setup(sizes)
#     T_e, gamma_L, gamma_R = setup_transmission(M,m_qii,m_qij,m_qji,sizes)
#     T_e_overlap = overlap_method(m_qii, m_qij, m_qji)
#     np.testing.assert_allclose(T_e_overlap, T_e)


def test_recursive_dyson_method():
    sizes = [3, 5, 3, 4]
    nodes = get_nodes(sizes)
    M, m_qii, m_qij, m_qji = build_setup(sizes)
    g_1N, (G_qii, G_qij, G_qji) = dyson_method(m_qii, m_qij, m_qji, trans=True)
    G = np.linalg.inv(M)
    np.testing.assert_allclose(g_1N, G[: m_qii[0].shape[0], -m_qii[-1].shape[-1] :])
    for i in range(len(sizes)):
        np.testing.assert_allclose(G_qii[i], get_ii(G, nodes, i))
    for i in range(1, len(sizes)):
        np.testing.assert_allclose(G_qij[i - 1], get_ij(G, nodes, i))
        np.testing.assert_allclose(G_qji[i - 1], get_ji(G, nodes, i))


# def test_partial_dyson_method():
#     sizes = [3,5,3,4]
#     nodes = get_nodes(sizes)
#     M, m_qii, m_qij, m_qji = build_setup(sizes,hermitian=True)
#     gamma_L, gamma_R, gL, gR = setup_transmission(M,m_qii,m_qij,m_qji,sizes)[1:]
#     (G_qii, G_qij, G_qji) = dyson_method(m_qii,m_qij,m_qji,gamma_L,gamma_R)
#     G = np.linalg.inv(M)
#     A = G.dot(gL+gR).dot(G.conj())
#     # A = 1.j*(G-G.conj()) A, 1.j*(G-G.conj()))
#     # np.testing.assert_allclose(g_1N, G[:m_qii[0].shape[0],-m_qii[-1].shape[-1]:])
#     for i in range(len(sizes)):
#         np.testing.assert_allclose(G_qii[i], get_ii(A,nodes,i))
#     for i in range(1,len(sizes)):
#         np.testing.assert_allclose(G_qij[i-1], get_ij(A,nodes,i))
#         np.testing.assert_allclose(G_qji[i-1], get_ji(A,nodes,i))


if __name__ == "__main__":
    test_spectral_method()
    test_recursive_coupling_method()
    test_transmission_coupling_method()
    test_recursive_dyson_method()
