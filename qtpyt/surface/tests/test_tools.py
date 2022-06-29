import numpy as np
from ase import Atoms
from qtpyt.basis import Basis, argsort
from qtpyt.surface.tools import map_B2A

unit_cell = Atoms("C2", positions=[[0, 0, 0], [0.5, 0.5, 0]], cell=[1, 1, 0])


def test_map_same_size_diff_ord():
    #
    #   |  5|      |  5|
    #   |4  |      |2  |
    #   |  3|      |  4|
    #   |2  |      |1  |
    #   |  1|      |  3|
    #   |0  |      |0  |
    pA = unit_cell.repeat((1, 3, 1)).positions
    pB = pA[argsort(pA)]
    B2A = map_B2A(pA, pB)
    np.testing.assert_allclose(B2A, [0, 3, 1, 4, 2, 5])
    np.testing.assert_allclose(pA, pB[B2A])


def test_map_diff_size_same_ord():
    #
    #   |  5|      |  9     11|
    #   |4  |      |8    10   |
    #   |  3|      |  5     7 |
    #   |2  |      |4    6    |
    #   |  1|      |  1     3 |
    #   |0  |      |0    2    |
    pA = unit_cell.repeat((1, 3, 1)).positions
    pB = unit_cell.repeat((2, 1, 1)).repeat((1, 3, 1)).positions
    B2A = map_B2A(pA, pB)
    np.testing.assert_allclose(B2A, [0, 1, 4, 5, 8, 9])
    np.testing.assert_allclose(pA, pB[B2A])


def test_map_diff_size_diff_ord():
    #
    #   |  5|      |  5     11|
    #   |4  |      |2    8    |
    #   |  3|      |  4     10|
    #   |2  |      |1    7    |
    #   |  1|      |  3     9 |
    #   |0  |      |0    6    |
    pA = unit_cell.repeat((1, 3, 1)).positions
    pB = unit_cell.repeat((2, 3, 1)).positions
    pB = pB[argsort(pB)]
    B2A = map_B2A(pA, pB)
    np.testing.assert_allclose(B2A, [0, 3, 1, 4, 2, 5])
    np.testing.assert_allclose(pA, pB[B2A])
