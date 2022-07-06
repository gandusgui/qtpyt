import numpy as np
from numba import njit, prange
from scipy.constants import e, k
from scipy.interpolate import interp1d
from scipy.linalg.blas import get_blas_funcs

kB = k / e  # Boltzmann in eV.


def fermidistribution(energy, kt):
    # fermi level is fixed to zero
    # energy can be a single number or a list
    assert kt >= 0.0, "Negative temperature encountered!"

    if kt == 0:
        if isinstance(energy, float):
            return int(energy / 2.0 <= 0)
        else:
            return (energy / 2.0 <= 0).astype(int)
    else:
        return 1.0 / (1.0 + np.exp(energy / kt))


@njit("(c16[:,:,::1],c16[:,:,::1])", parallel=True, fastmath=True)
def lesser_from_retarded(r, g):
    # < = > - r + a
    ne, n = r.shape[:2]
    for e in prange(ne):
        a = np.empty((n, n), r.dtype)
        # diag
        for i in range(n):
            r[e, i, i] = g[e, i, i] - r[e, i, i] + np.conjugate(r[e, i, i])
        # utrid
        for i in range(n - 1):
            for j in range(i + 1, n):
                a[i, j] = np.conjugate(r[e, i, j])
                r[e, i, j] = g[e, i, j] - r[e, i, j] + np.conjugate(r[e, j, i])
        # ltrid
        for i in range(1, n):
            for j in range(i):
                r[e, i, j] = g[e, i, j] - r[e, i, j] + a[j, i]


@njit("(c16[:,:,::1],c16[:,:,::1])", parallel=True, fastmath=True)
def greater_from_retarded(r, l):
    # > = < + r - a
    ne, n = r.shape[:2]
    for e in prange(ne):
        a = np.empty((n, n), r.dtype)
        # diag
        for i in range(n):
            r[e, i, i] = l[e, i, i] + r[e, i, i] - np.conjugate(r[e, i, i])
        # utrid
        for i in range(n - 1):
            for j in range(i + 1, n):
                a[i, j] = np.conjugate(r[e, i, j])
                r[e, i, j] = l[e, i, j] + r[e, i, j] - np.conjugate(r[e, j, i])
        # ltrid
        for i in range(1, n):
            for j in range(i):
                r[e, i, j] = l[e, i, j] + r[e, i, j] - a[j, i]


def translate_along_axis(array, translation):
    """Optimized method for translating along axis 0"""
    newarray = array.copy()
    if translation == 0:
        return newarray
    newarray[:translation] = array[-translation:]
    newarray[translation:] = array[:-translation]
    return newarray


def roll(a, shift):
    """Translate and array in place."""
    n = a.shape[0]
    shift = shift % n
    perm = np.roll(np.arange(n), shift)
    tmpa = np.atleast_1d(np.empty_like(a[0]))

    def swap(a, i, j, tmp):
        tmp[:] = a[i]
        a[i] = a[j]
        a[j] = tmp[:]

    for i in range(n):
        swap(a, i, perm[i], tmpa)
        argmin = np.where((perm - i) == 0)[0][-1]
        perm[argmin] = perm[i]
        perm[i] = 0

    return a


def rotate(a, u, work=None, out=None, overwrite_a=False):

    gemm = get_blas_funcs("gemm", (a, u))
    if work is None:
        work = np.empty_like(a)
    if overwrite_a:
        out = a
    elif out is None:  # overwrite_a == False
        out = np.empty_like(a)
    # U.dot(A).dot(U.T)
    gemm(1.0, a.T, u.T, 0.0, work.T, overwrite_c=True)
    gemm(1.0, u.T, work.T, 0.0, out.T, trans_a=2, overwrite_c=True)
    return out


def increase2pow2(N):
    """Increase number to power of two.s"""
    count = 0
    while N != 1:
        if N % 2 == 1:
            N += 1
        N /= 2
        count += 1
    return 2 ** count


def get_extended_energies(energies, oversample=10):
    ne = len(energies)

    # Make sure that grid lenght is a power of 2
    ne_ext = increase2pow2(oversample * ne)

    begin = energies[0]
    de = energies[1] - begin
    extenergies = np.arange(ne_ext) * de + begin

    return np.around(extenergies, 8)


def interpolate(X, energies, indx2eval):
    """Interpolate array inplace at `indx2eval` energies along the 0-th axis.
    
    X.shape[0] and len(energies) are the same. The rest of the indices are
    assumed to be the energy points at which X has been evaluated.
    
    Example:
        X = func(energies[indx])
        interpolate(X, energies, np.setdiff1d(range(energies.size), indx))
    
    """
    assert X.shape[0] == len(energies), "X.shape[0] differs from len(energies)."
    indx = np.setdiff1d(range(len(energies)), indx2eval)
    X_interp = interp1d(energies[indx], X[indx], kind="slinear", axis=0)
    for e, energy in zip(indx2eval, energies[indx2eval]):
        X[e] = X_interp(energy)


def get_interp_indices(energies):
    a = 2
    b = energies[0] / 10.0
    c = energies[-1] / 10.0
    indices = []
    count = 0
    while count < len(energies):
        indices.append(count)
        x = energies[count]
        if x <= 0:
            y = int(1 + pow(abs(x), a) / pow(abs(b), a))
        else:
            y = int(1 + pow(abs(x), a) / pow(abs(c), a))
        count += y
    return np.array(indices)


def linear_interp(a1, indices, size):
    # Interpolate from array a1 to array a2. a1 is sampled on indices
    # Treat first index as special case

    # Don't do anything if a1 and a2 are identical
    if len(a1) == size:
        return a1
    s = (size,) + a1.shape[1:]
    a2 = np.zeros(s, a1.dtype)
    slope0 = (a1[1] - a1[0]) / (1.0 * indices[1] - indices[0])
    for i in range(indices[0]):
        a2[i] = a1[0] + slope0 * (1.0 * i - indices[0])
    for i in range(1, len(indices)):
        for j in range(indices[i - 1], indices[i]):
            a2[j] = a1[i - 1] + (a1[i] - a1[i - 1]) * (1.0 * j - indices[i - 1]) / (
                indices[i] - indices[i - 1]
            )
    slope1 = (a1[-1] - a1[-2]) / (1.0 * indices[-1] - indices[-2])
    for i in range(indices[-1], a2.shape[0]):
        a2[i] = a1[-1] + slope1 * (1.0 * i - indices[-1])
    return a2
