import numpy as np

from qtpyt.kernels.linalg_cpu import solve, lrsolve, tsolve, dot


def test_dot():

    a = np.ones((50, 100))
    b = np.ones((100, 200))
    c = np.empty((50, 200))

    expected = np.dot(a, b)

    c = dot(a, b, c, beta=0.0)

    np.testing.assert_allclose(c, expected)


def test_solve():

    a = np.random.random((100, 100)) + 1.0j * np.random.random((100, 100))
    b = np.random.random((100, 200)) + 1.0j * np.random.random((100, 200))

    expected = np.linalg.solve(a, b)

    a_f = a.copy(order="F")
    b_f = b.copy(order="F")

    c = solve(a, b)

    np.testing.assert_allclose(c, expected)

    c = solve(a_f, b_f)

    np.testing.assert_allclose(c, expected)


def test_lrsolve():

    a = np.random.random((100, 100)) + 1.0j * np.random.random((100, 100))
    b = np.random.random((100, 100)) + 1.0j * np.random.random((100, 100))

    tmp = np.linalg.solve(a, b)
    expected = np.linalg.solve(a.conj(), tmp.T).T

    a_f = a.copy(order="F")
    b_f = b.copy(order="F")

    c = lrsolve(a, b)

    np.testing.assert_allclose(c, expected)

    c = lrsolve(a_f, b_f)

    np.testing.assert_allclose(c, expected)


def test_tsolve():

    a = np.random.random((100, 100)) + 1.0j * np.random.random((100, 100))
    b = np.random.random((100, 100)) + 1.0j * np.random.random((100, 100))
    c = np.random.random((100, 100)) + 1.0j * np.random.random((100, 100))

    expected1 = np.linalg.solve(a, b)
    expected2 = np.linalg.solve(a.T.conj(), c)

    a_f = a.copy(order="F")
    b_f = b.copy(order="F")
    c_f = c.copy(order="F")

    d1, d2 = tsolve(a, b, c)

    np.testing.assert_allclose(d1, expected1)
    np.testing.assert_allclose(d2, expected2)

    d1, d2 = tsolve(a_f, b_f, c_f)

    np.testing.assert_allclose(d1, expected1)
    np.testing.assert_allclose(d2, expected2)


def time_dgemm():
    import timeit

    m = 1000
    k = 2000
    n = 700
    a = np.ones((m, k))
    b = np.ones((k, n))
    c = np.empty((m, n))

    elapsed = timeit.timeit(
        "np.dot(a,b,out=c)", setup="import numpy as np", globals=locals(), number=500
    )

    print("numpy dot :", elapsed)

    elapsed = timeit.timeit(
        "dot(a,b,c,beta=0.)",
        setup="from __main__ import dot",
        globals=locals(),
        number=500,
    )

    print("This :", elapsed)


def time_solve():
    import timeit

    n = 1000
    a = np.random.random((n, n)).astype(complex)
    nrhs = 2000
    b = np.random.random((n, nrhs)).astype(complex)

    # elapsed = timeit.timeit("la.solve(a,b,overwrite_b=True,check_finite=False)",
    #     setup='from scipy import linalg as la', globals=locals(), number=500)

    # print("scipy solve :", elapsed)

    elapsed = timeit.timeit(
        "la.solve(a,b)",
        setup="from numpy import linalg as la",
        globals=locals(),
        number=100,
    )

    print("numpy solve :", elapsed)

    elapsed = timeit.timeit(
        "solve(a,b,overwrite_b=True)",
        setup="from __main__ import solve",
        globals=locals(),
        number=100,
    )

    print("This :", elapsed)


def time_lrsolve():
    import timeit

    n = 1000
    a = np.random.random((n, n)).astype(complex)
    nrhs = 1000
    b = np.random.random((n, nrhs)).astype(complex)

    a_f = a.copy(order="F")
    b_f = b.copy(order="F")

    elapsed = timeit.timeit(
        "tmp=la.solve(a,b);la.solve(a,tmp.T).T",
        setup="from numpy import linalg as la",
        globals=locals(),
        number=100,
    )

    print("numpy solve :", elapsed)

    # elapsed = timeit.timeit("ainv=la.inv(a);ainv.dot(b).dot(ainv.T)",
    #     setup='from numpy import linalg as la', globals=locals(), number=100)

    # print("numpy inv :", elapsed)

    elapsed = timeit.timeit(
        "lrsolve(a_f,b_f)",
        setup="from __main__ import lrsolve",
        globals=locals(),
        number=100,
    )

    print("This solve:", elapsed)


def time_tsolve():
    import timeit

    n = 1000
    a = np.random.random((n, n)).astype(complex)
    nrhs = 1000
    b = np.random.random((n, nrhs)).astype(complex)
    c = np.random.random((n, nrhs)).astype(complex)

    a_f = a.copy(order="F")
    b_f = b.copy(order="F")
    c_f = c.copy(order="F")

    elapsed = timeit.timeit(
        "expected1=la.solve(a,b);expected2=la.solve(a.T.conj(),c)",
        setup="from numpy import linalg as la",
        globals=locals(),
        number=100,
    )

    print("numpy solve :", elapsed)

    elapsed = timeit.timeit(
        "d1,d2=tsolve(a_f,b_f,c_f)",
        setup="from __main__ import tsolve",
        globals=locals(),
        number=100,
    )

    print("This solve:", elapsed)


if __name__ == "__main__":
    # time_dgemm()
    time_solve()
    time_lrsolve()
    time_tsolve()
