import numpy as np
from qtpyt.parallel.transpose import transpose


def test_transpose():

    m, n = np.random.randint(0, 50, 2)
    a = np.arange(m * n) + 1.0j * 1e-5
    expected = a.T.copy()

    a = transpose(a)

    np.testing.assert_allclose(a.T, expected)
