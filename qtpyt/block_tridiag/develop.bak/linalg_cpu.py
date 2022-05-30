import numpy as np
from _linalg_numba_cpu import build_diag, build_diag_sigma, build_offdiags


def build_mat_lists(
    z,
    hs_list_ii,
    hs_list_ij,
    mat_list_ii,
    mat_list_ij,
    mat_list_ji,
    sigma_L=None,
    sigma_R=None,
):
    """Construct matrices of the form (zS-H).
    
    If Sigma_L or Sigma_R are given, then construct (zS-H-Sigma).
    """

    N = len(hs_list_ii)

    if sigma_L is None:
        lb = 0
    else:
        lb = 1
        build_diag_sigma(z, mat_list_ii[0], hs_list_ii[0], sigma_L)

    if sigma_R is None:
        ub = N
    else:
        ub = N - 1
        build_diag_sigma(z, mat_list_ii[-1], hs_list_ii[-1], sigma_R)

    for q in range(lb, ub):
        build_diag(z, mat_list_ii[q], hs_list_ii[q])

    for q in range(N - 1):
        build_offdiags(z, mat_list_ij[q], mat_list_ji[q], hs_list_ij[q], 32)


def multiply(A_qii, A_qij, B_qii, B_qij, A_qji=None, B_qji=None):
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
