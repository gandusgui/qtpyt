from numba import vectorize

from qtpyt import xp

dagger = lambda mat: xp.conj(mat.T)

target = "cpu" if xp.__name__ == "numpy" else "cuda"

kwargs = {"target": target}
if target == "cpu":
    kwargs["fastmath"] = True

# @vectorize(['c16(c16,c16,c16)'],**kwargs)
def _mul(z, h, s):
    return z * s - h


# @vectorize(['c16(c16,c16,c16)'],**kwargs)
def _mulc(z, h, s):
    return z * s.conj() - h.conj()
    # return z*s.conjugate()-h.conjugate()


def get_lambda(sigma):
    return 1.0j * (sigma - sigma.T.conj())


def dotdiag(x, y, res=None):
    if res is None:
        res = xp.zeros(x.shape[0], x.dtype)
    for i in range(x.shape[0]):
        res[i] += xp.sum(x[i, :] * y[:, i])
    return res


def dottrace(x, y, res=0.0):
    for i in range(x.shape[0]):
        res += xp.sum(x[i, :] * y[:, i])
    return res
