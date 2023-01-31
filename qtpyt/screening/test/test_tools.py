import numpy as np

from qtpyt.screening.tools import (greater_from_retarded, lesser_from_retarded,
                                   roll, rotate, smooth, translate_along_axis)


def test_roll():

    for _ in range(5):
        a = np.random.random((np.random.randint(0, 120), 3, 3))
        for _ in range(5):
            shift = np.random.randint(0, a.shape[0])
            b = translate_along_axis(a, shift)
            c = roll(a.copy(), shift)
            np.testing.assert_allclose(b, c)


def test_rotate():
    rnd = lambda m: np.random.random((m, m))

    m = 20
    a = rnd(m) + 1.0j * rnd(m)
    b = rnd(m) + 1.0j * rnd(m)
    c = a.copy()

    out = rotate(a, b, overwrite_a=True)
    np.testing.assert_allclose(a, out)
    np.testing.assert_allclose(out, b.dot(c).dot(b.T.conj()))


def test_lesser_from_retarded():

    ne = 10
    n = 4
    r = np.random.random((ne, n, n)) + 1.0j * np.random.random((ne, n, n))
    g = np.random.random((ne, n, n)) + 1.0j * np.random.random((ne, n, n))
    desired = g - r + r.swapaxes(1, 2).conj()
    lesser_from_retarded(r, g)
    actual = r
    np.testing.assert_allclose(actual, desired)


def test_greater_from_retarded():

    ne = 10
    n = 4
    r = np.random.random((ne, n, n)) + 1.0j * np.random.random((ne, n, n))
    l = np.random.random((ne, n, n)) + 1.0j * np.random.random((ne, n, n))
    desired = l + r - r.swapaxes(1, 2).conj()
    greater_from_retarded(r, l)
    actual = r
    np.testing.assert_allclose(actual, desired)


def test_smooth():
    x = np.arange(4)
    A = np.arange(len(x) * 3 ** 2).reshape(len(x), 3, 3)
    A2 = smooth(x, A, cutfreq=1, oversample=2)
    np.testing.assert_allclose(A, A2)


if __name__ == "__main__":
    test_roll()
    test_rotate()
    test_lesser_from_retarded()
    test_greater_from_retarded()
