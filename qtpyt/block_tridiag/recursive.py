# Cutoff for cutting block matrices
from qtpyt import xp, xla

solve = xp.linalg.solve  # la.solve
inv = xp.linalg.inv  # la.inv
dot = xp.dot
dagger = lambda mat: xp.conj(mat.T)


def coupling_method_1N(m_qii, m_qij, m_qji):
    """
    TODO :: 
    (1) a_mm = solve(..,m_qij) inplace m_qij
    """
    N = len(m_qii)

    q = 1
    a_mm = solve(m_qii[q - 1], m_qij[q - 1])
    m_qii[q] -= dot(m_qji[q - 1], a_mm)
    c_mm = a_mm
    for q in range(2, N):
        a_mm = solve(m_qii[q - 1], m_qij[q - 1])
        m_qii[q] -= dot(m_qji[q - 1], a_mm)
        c_mm = dot(c_mm, a_mm)

    c_mm *= (-1) ** (N - 1)
    # solve xA=B =: x=BA^-1
    # (xA)^T = B^T
    # A^Tx^T = B^T
    g_1N = solve(m_qii[N - 1].T, c_mm.T).T

    return g_1N


def coupling_method_N1(m_qii, m_qij, m_qji):
    """
    TODO :: 
    (1) a_mm = solve(..,m_qji) inplace m_qji
    """
    N = len(m_qii)

    q = N - 2
    a_mm = solve(m_qii[q + 1], m_qji[q])
    m_qii[q] -= dot(m_qij[q], a_mm)
    c_mm = a_mm
    for q in range(N - 3, -1, -1):
        a_mm = solve(m_qii[q + 1], m_qji[q])
        m_qii[q] -= dot(m_qij[q], a_mm)
        c_mm = dot(c_mm, a_mm)

    c_mm *= (-1) ** (N - 1)
    # solve xA=B =: x=BA^-1
    # (xA)^T = B^T
    # A^Tx^T = B^T
    g_N1 = solve(m_qii[0].T, c_mm.T).T

    return g_N1


def spectral_method(m_qii, m_qij, m_qji, gamma_L, gamma_R):

    N = len(m_qii)

    n_qii = [None for _ in range(N)]
    n_qji = [None for _ in range(N - 1)]
    n_qij = [None for _ in range(N - 1)]

    n_qii[-1] = m_qii[-1]
    for q in range(N - 2, -1, -1):
        n_qji[q] = solve(n_qii[q + 1], m_qji[q])  # can't overwrite
        n_qii[q] = m_qii[q] - dot(m_qij[q], n_qji[q])  # can't overwrite

    for q in range(1, N):
        m_qij[q - 1] = solve(m_qii[q - 1], m_qij[q - 1])  # can overwrite
        m_qii[q] -= dot(m_qji[q - 1], m_qij[q - 1])  # can overwrite

    a_mm = solve(n_qii[0], gamma_L)
    n_qii[0] = solve(n_qii[0].conj(), a_mm.T).T
    for q in range(N - 1):
        a_mm = -n_qji[q]
        n_qji[q] = dot(a_mm, n_qii[q])  # , out=n_qji[q])
        n_qij[q] = dot(n_qii[q], dagger(a_mm))  # , out=n_qij[q])
        n_qii[q + 1] = dot(a_mm, n_qij[q])  # , out=n_qii[q+1])

    a_mm = solve(m_qii[-1], gamma_R)
    m_qii[-1] = solve(m_qii[-1].conj(), a_mm.T).T
    for q in range(N - 2, -1, -1):
        a_mm = -m_qij[q]
        m_qij[q] = dot(a_mm, m_qii[q + 1])  # , out=m_qij[q])
        m_qji[q] = dot(m_qii[q + 1], dagger(a_mm))  # , out=m_qji[q].T)
        m_qii[q] = dot(a_mm, m_qji[q])  # , out=m_qii[q])

    return (n_qii, n_qij, n_qji), (m_qii, m_qij, m_qji)


def dyson_method(m_qii, m_qij, m_qji, trans=False):

    N = len(m_qii)

    # Left connected green's function
    grL_qii = m_qii  # [None for _ in range(N)]
    # Initalize
    grL_qii[0] = inv(m_qii[0])  # ([eS-H]_11-Sigma_L)^-1

    # Left connected recursion
    for q in range(1, N):
        a_ij = dot(grL_qii[q - 1], m_qij[q - 1])
        grL_qii[q] = inv(m_qii[q] - dot(m_qji[q - 1], a_ij))

    if trans:
        # First row green's function
        gr_1N = grL_qii[0]
        for q in range(1, N):
            gr_1N = dot(gr_1N, dot(m_qij[q - 1], grL_qii[q]))
        gr_1N *= (-1) ** (N - 1)

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
        a_ij = dot(-grL_qii[q], m_qij[q])
        a_ji = dot(-m_qji[q], grL_qii[q])
        Gr_qji[q] = dot(Gr_qii[q + 1], a_ji)
        Gr_qij[q] = dot(a_ij, Gr_qii[q + 1])
        Gr_qii[q] = grL_qii[q] + dot(a_ij, Gr_qji[q])

    if trans:
        return gr_1N, (Gr_qii, Gr_qij, Gr_qji)
    return (Gr_qii, Gr_qij, Gr_qji)

    # gnL_qii = [None for _ in range(N)]
    # s_in = [None for _ in range(N)]
    # s_in[0] = gamma_L; s_in[-1] = gamma_R

    # gnL_qii[q] = dot(grL_qii[q], dot(s_in[0], grL_qii[q].conj()))
    # for q in range(1, N):
    #     sl = dot(m_qji[q-1], dot(gnL_qii[q-1], m_qij[q-1].conj()))
    #     if s_in[q] is not None: sl += s_in[q]
    #     gnL_qii[q] = dot(grL_qii[q], dot(sl, grL_qii[q].conj()))

    # Gn_qii = [None for _ in range(N)]
    # Gn_qij = [None for _ in range(N-1)]
    # Gn_qji = [None for _ in range(N-1)]

    # Gn_qii[-1] = gnL_qii[-1]
    # for q in range(N-2,-1,-1):
    #     a = dot(Gn_qii[q+1], dot(m_qji[q].conj(), grL_qii[q].conj()))
    #     Gn_qji[q] = - (dot(Gr_qii[q+1], dot(m_qji[q], gnL_qii[q])) + a)
    #     Gn_qii[q] = gnL_qii[q] + \
    #                 dot(grL_qii[q], dot(m_qij[q],  a)) - \
    #                 dot(gnL_qii[q], dot(m_qij[q].conj(), Gr_qji[q].conj())) - \
    #                 dot(Gr_qij[q], dot(m_qji[q], gnL_qii[q]))
    #     Gn_qij[q] = dagger(Gn_qji[q])

    # return (Gn_qii, Gn_qij, Gn_qji)


# def overlap_method(m_qii, m_qij, m_qji):

#     N = len(m_qii)
#     k = xp.argmin([m_qii[q].shape[0] for q in range(N)])

#     for q in range(1,k):
#         a_mm = solve(m_qii[q-1], m_qij[q-1]) # Here we could overwrite m_qij[q-1]
#         m_qii[q] -= dot(m_qji[q-1], a_mm)

#     q = k
#     sL_kk = dot(m_qji[q-1], solve(m_qii[q-1], m_qij[q-1]))  # Here we could overwrite m_qji[q+1]

#     for q in range(N-2,k,-1):
#         a_mm = solve(m_qii[q+1], m_qji[q])
#         m_qii[q] -= dot(m_qij[q], a_mm)

#     q = k
#     sR_kk = dot(m_qij[q], solve(m_qii[q+1], m_qji[q]))

#     gammaL_kk = 1.j * (sL_kk - sL_kk.T.conj())
#     gammaR_kk = 1.j * (sR_kk - sR_kk.T.conj())
#     m_qii[k] -= sL_kk + sR_kk
#     g_kk = inv(m_qii[k]) # Assert allclose with G[kk].inv() OK

#     a_mm = solve(m_qii[k], gammaL_kk)
#     b_mm = solve(dagger(m_qii[k]), gammaR_kk)

#     return trace(dot(a_mm, b_mm)).real
