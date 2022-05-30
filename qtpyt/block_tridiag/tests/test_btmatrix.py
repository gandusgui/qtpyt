from posixpath import split
import numpy as np

from qtpyt.block_tridiag.btmatrix import BTMatrix
from qtpyt.projector import project
from test_recursive import build_matrix, build_setup, split_matrix, get_nodes, get_ii

sizes = [3, 5, 3, 2]


def test_dot():

    A, a_qii, a_qij, a_qji = build_setup(sizes)
    B, b_qii, b_qij, b_qji = build_setup(sizes)
    btA = BTMatrix(a_qii, a_qij, a_qji)
    btB = BTMatrix(b_qii, b_qij, b_qji)
    btC = btA.dot(btB)
    C = A.dot(B)

    btC_expect = BTMatrix(*split_matrix(C, sizes))

    # Test diagonal components are equivalent
    for compute, expect in zip(btC.m_qii, btC_expect.m_qii):
        np.testing.assert_allclose(compute, expect)


def test_rotate():
    A, a_qii, a_qij, a_qji = build_setup(sizes)
    btA = BTMatrix(a_qii, a_qij, a_qji)

    # permutation matrix
    U = np.eye(A.shape[0])
    u_qii, u_qij, u_qji = split_matrix(U, sizes)
    for i in [2, 3]:
        permute = np.arange(sizes[i])
        np.random.shuffle(permute)
        u_qii[i] = u_qii[i].take(permute, axis=0).take(permute, axis=1)
    U = build_matrix(u_qii, u_qij, u_qji)

    At = U.T.dot(A).dot(U)
    btAt = btA.rotate(u_qii)

    btAt_expect = BTMatrix(*split_matrix(At, sizes))

    btAt == btAt_expect


def test_project():
    A, a_qii, a_qij, a_qji = build_setup(sizes)
    B, b_qii, b_qij, b_qji = build_setup(
        sizes, hermitian=True, real=True
    )  # rotation matrix should be real and symmetric
    btA = BTMatrix(a_qii, a_qij, a_qji)
    btB = BTMatrix(b_qii, b_qij, b_qji)

    # test project terminal blocks + central block
    nodes = get_nodes(sizes)

    # 0 block
    c_expect = project(A, B, np.arange(nodes[0], nodes[1]))
    c_compute = project(btA, btB, 0)
    np.testing.assert_allclose(c_compute, c_expect)

    # (N-1) block
    c_expect = project(A, B, np.arange(nodes[-2], nodes[-1]))
    c_compute = project(btA, btB, len(sizes) - 1)
    np.testing.assert_allclose(c_compute, c_expect)

    # i-th block
    c_expect = project(A, B, np.arange(nodes[1], nodes[2]))
    c_compute = project(btA, btB, 1)
    np.testing.assert_allclose(c_compute, c_expect)

    # test project with subindices

    # 0 block
    indices = np.arange(nodes[0], nodes[1])[[0, 1]]
    c_expect = project(A, B, indices)
    c_compute = project(btA, btB, 0, indices - nodes[0])
    np.testing.assert_allclose(c_compute, c_expect)

    # i-th block
    indices = np.arange(nodes[1], nodes[2], 2)
    c_expect = project(A, B, indices)
    c_compute = project(btA, btB, 1, indices - nodes[1])
    np.testing.assert_allclose(c_compute, c_expect)


if __name__ == "__main__":
    test_dot()
    test_rotate()
    test_project()
