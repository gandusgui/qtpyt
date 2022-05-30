import cupy as cp

from qtpyt.kernels.linalg_gpu import solve, lrsolve, tsolve


def test_solve():
    a = cp.random.random((100, 100)) + 1.0j * cp.random.random((100, 100))
    b = cp.random.random((100, 100)) + 1.0j * cp.random.random((100, 100))

    a_f = cp.asfortranarray(a)
    b_f = cp.asfortranarray(b)

    c = solve(a_f, b_f)

    expected = cp.linalg.solve(a, b)

    assert cp.allclose(c, expected)


def test_lrsolve():
    a = cp.random.random((100, 100)) + 1.0j * cp.random.random((100, 100))
    b = cp.random.random((100, 100)) + 1.0j * cp.random.random((100, 100))

    a_f = cp.asfortranarray(a)
    b_f = cp.asfortranarray(b)

    c = lrsolve(a_f, b_f)

    tmp = cp.linalg.solve(a, b)
    expected = cp.linalg.solve(a.conj(), tmp.T).T

    assert cp.allclose(c, expected)


def test_tsolve():
    a = cp.random.random((100, 100)) + 1.0j * cp.random.random((100, 100))
    b = cp.random.random((100, 100)) + 1.0j * cp.random.random((100, 100))
    c = cp.random.random((100, 100)) + 1.0j * cp.random.random((100, 100))

    a_f = cp.asfortranarray(a)
    b_f = cp.asfortranarray(b)
    c_f = cp.asfortranarray(c)

    d_f1, d_f2 = tsolve(a_f, b_f, c_f)

    expected1 = cp.linalg.solve(a, b)
    expected2 = cp.linalg.solve(a.T.conj(), c)

    assert cp.allclose(d_f1, expected1)
    assert cp.allclose(d_f2, expected2)
