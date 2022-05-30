from qtpyt import xp
from qtpyt.base._kernels import _mulc, _mul, dottrace, dotdiag


def test_mul():
    a = xp.random.random((10, 10)).astype(complex)
    b = a.copy()
    c = a.copy()
    z = complex(2.0)

    _mul(z, a, b, out=c)
    assert xp.allclose(z * b - a, c)


def test_mulc():
    a = xp.random.random((10, 10)).astype(complex)
    b = a.copy()
    c = a.copy()
    z = complex(2.0)

    _mulc(z, a.T, b.T, out=c)
    assert xp.allclose(z * b.T.conj() - a.T.conj(), c)


def test_dotx():
    a = xp.random.random((10, 10)).astype(complex)
    b = a.copy()

    res = dottrace(a, b)
    assert xp.allclose(res, a.dot(b).trace())

    res = xp.zeros(a.shape[0], complex)
    dotdiag(a, b, res)
    assert xp.allclose(res, a.dot(b).diagonal())
