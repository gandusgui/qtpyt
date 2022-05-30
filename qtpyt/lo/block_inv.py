import numpy as np
from numpy.core.defchararray import array
from numpy.core.shape_base import block


def block_inv(M):
    """"Block inverision of 2x2 matrix using Shur complement.
    
    Args:
        M : (np.array(object), tuple, list), shape 2x2
            Block matrix to invert.
    """
    Minv = [[None, None], [None, None]]
    A = M[0][0]
    B = M[0][1]
    C = M[1][0]
    D = M[1][1]
    Ainv = np.linalg.inv(A)
    S = D - C.dot(Ainv).dot(B)
    Sinv = np.linalg.inv(S)
    Minv[1][1] = Sinv
    Ainv_dot_B = Ainv.dot(B)
    C_dot_Ainv = C.dot(Ainv)
    Minv[0][1] = -Ainv_dot_B.dot(Sinv)
    Minv[1][0] = -Sinv.dot(C_dot_Ainv)
    Minv[0][0] = Ainv
    Minv[0][0] += Ainv_dot_B.dot(Sinv).dot(C_dot_Ainv)
    return Minv


def test_block_inv():
    mat = np.random.random((4, 4))
    M = [[mat[:2, :2], mat[:2, 2:]], [mat[2:, :2], mat[2:, 2:]]]
    computed = np.block(block_inv(M))
    expected = np.linalg.inv(mat)
    np.testing.assert_allclose(expected, computed)


if __name__ == "__main__":
    test_block_inv()
