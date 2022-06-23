from math import cos, sin

import numpy as np
from numba import njit, prange
from qtpyt.screening import fft, ifft
from qtpyt.screening.tools import roll

"""
The fourier transform we define by::

   F(w) = int dt exp(iwt) * f(t)

and the inverse by::

   f(t) = 1 / (2 * pi) int dw exp(-iwt) * F(w)
"""


def fourier_integral(f, delta, N, sign=1, data=None):
    """ The fourier transform of a function represented by the array
    f. len(f) does not have to be an integer power of two!
    NOTE: Assumes that the fouriertransform is along axis 0.
    WARNING: this methods leads to a kink in the transform in the middle of
             the data range! It should only be used if the data is
             oversampled (zero-padded) by at least a factor 2.

    ``delta`` is the increment of the tranform axis.

    sign=1 indicates an inverse fft ( return F(w) from f(t) )
    sign=-1 indicates a forward fft ( return f(t) from F(w) )
    """
    # N = len(f)
    if data is None:
        data = get_fourier_data(N)
    W, cor = data
    endpts = np.concatenate([f[:4], f[-1:-5:-1]])
    # endpts = np.concatenate([f[N // 2 - 5 : N // 2 - 1], f[N // 2 : N // 2 + 4]])

    if sign == 1:
        F = ifft(f, N)
        _dftshift(F, W, cor, endpts, 1)
        F *= delta
    else:
        F = fft(f, N)
        _dftshift(F, W, cor, endpts, -1)
        F *= delta / (2 * np.pi)
    return F


@njit(["(c16[::1],c16[::1],c16[:,::1],c16[::1],i8)"], parallel=True, cache=False)
def _dftshift(f, W, cor, endpts, sign):

    if sign == 1:
        N = f.size
    else:
        N = 1

    for i in prange(len(cor)):
        shift = 0.0
        for j in range(4):
            spt = endpts[j]
            ept = endpts[j + 4]
            c = cor[i, j]
            shift += c * spt + c.conjugate() * ept
        f[i] = W[i] * N * f[i] + shift


def get_fourier_data(N):
    theta = np.arange(-N / 2, N / 2) * 2 * np.pi / N
    theta = np.fft.ifftshift(theta)
    return dftcor(theta)  # end point corrections
    # W, cor = dftcor(theta)  # end point corrections
    # return np.fft.ifftshift(W, axes=[0]), np.fft.ifftshift(cor, axes=[0])


def dftcor(theta):
    """ End point corrections for FFT
    
    Numerical Recipies in C, Chap. 13.9
    Formula 13.9.14
    """
    theta_small = 0.01
    N = len(theta)
    cor = np.zeros((N, 4), complex)
    W = np.zeros(N, complex)

    mask = abs(theta) < theta_small

    _dftcor(mask, W, cor, theta)
    return W, cor


@njit(["(b1[::1],c16[::1],c16[:,::1],f8[::1])"], parallel=True, cache=False)
def _dftcor(mask, W, cor, theta):

    for i in prange(len(mask)):

        t = theta[i]

        if mask[i]:
            ts2 = t * t
            ts4 = ts2 * ts2
            ts6 = ts4 * ts2
            W[i] = 1 - 11.0 / 720 * ts4 + 23.0 / 15120 * ts6
            cor[i, 0] = (
                -2.0 / 3
                + 1.0 / 45 * ts2
                + 103.0 / 15120 * ts4
                - 169.0 / 226800 * ts6
                + 1j
                * t
                * (2.0 / 45 + 2.0 / 105 * ts2 - 8.0 / 2835 * ts4 + 86.0 / 467775 * ts6)
            )
            cor[i, 1] = (
                7.0 / 24
                - 7.0 / 180 * ts2
                + 5.0 / 3456 * ts4
                - 7.0 / 259200 * ts6
                + 1j
                * t
                * (
                    7.0 / 72
                    - 1.0 / 168 * ts2
                    + 11.0 / 72576 * ts4
                    - 13.0 / 5987520 * ts6
                )
            )
            cor[i, 2] = (
                -1.0 / 6
                + 1.0 / 45 * ts2
                - 5.0 / 6048 * ts4
                + 1.0 / 64800 * ts6
                + 1j
                * t
                * (
                    -7.0 / 90
                    + 1.0 / 210 * ts2
                    - 11.0 / 90720 * ts4
                    + 13.0 / 7484400 * ts6
                )
            )
            cor[i, 3] = (
                1.0 / 24
                - 1.0 / 180 * ts2
                + 5.0 / 24192 * ts4
                - 1.0 / 259200 * ts6
                + 1j
                * t
                * (
                    7.0 / 360
                    - 1.0 / 840 * ts2
                    + 11.0 / 362880 * ts4
                    - 13.0 / 29937600 * ts6
                )
            )

        else:
            ts2 = t * t
            ts3 = ts2 * t
            ts4 = ts2 * ts2
            costb = cos(t)
            cos2tb = cos(2 * t)
            sintb = sin(t)
            sin2tb = sin(2 * t)
            W[i] = ((6 + ts2) / (3 * ts4)) * (3 - 4 * costb + cos2tb)
            cor[i, 0] += (-42 + 5 * ts2 + (6 + ts2) * (8 * costb - cos2tb)) / (
                6 * ts4
            ) + 1j * (-12 * t + 6 * ts3 + sin2tb * (6 + ts2)) / (6 * ts4)

            cor[i, 1] += (14 * (3 - ts2) - 7 * (6 + ts2) * costb) / (6 * ts4) + 1j * (
                30 * t - 5 * (6 + ts2) * sintb
            ) / (6 * ts4)
            cor[i, 2] += (-4 * (3 - ts2) + 2 * (6 + ts2) * costb) / (3 * ts4) + 1j * (
                -12 * t + 2 * (6 + ts2) * sintb
            ) / (3 * ts4)
            cor[i, 3] += (2 * (3 - ts2) - (6 + ts2) * costb) / (6 * ts4) + 1j * (
                6 * t - (6 + ts2) * sintb
            ) / (6 * ts4)


if __name__ == "__main__":
    # Run a self-test, also serving as an example.
    import pylab as pl

    def gauss(x, x0=0.0, width=2.5):
        return np.exp(-0.5 * (x - x0) ** 2 / width ** 2) / (width * np.sqrt(2 * np.pi))

    def lorentz(x, x0=0.0, width=2.5):
        return width / ((x - x0) ** 2 + width ** 2) / np.pi

    x = np.linspace(-8, 15, 2 ** 10)
    de, N = x[1] - x[0], len(x)
    Next = N * 2  # oversample

    # Make Lorentzian and FFT to time
    eta = 0.2
    y = lorentz(x, width=eta)
    fy = np.fft.fft(y, n=Next)
    print("Integral of Lorentzian", np.trapz(y, x))

    # Make retarded
    r = fy.copy()
    r[len(r) // 2 :] = 0.0

    # inverse FFT back to energy
    a1 = 2 * np.fft.ifft(r)[:N]
    a2 = 2 * fourier_integral(r, de, sign=1)[:N] / (Next * de)

    fig = pl.figure(1)
    pl.subplot(211)
    pl.plot(x, y.real, label="Lorentz")
    pl.plot(x, a1.real, label="Real of ret np.ifft")
    pl.plot(x, a1.imag, label="Imag of ret np.ifft")
    pl.plot(x, a2.real, label="Real of ret fourierint")
    pl.plot(x, a2.imag, label="Imag of ret fourierint")
    pl.title("Energy space")
    pl.xlabel("Energy")
    pl.axis("tight")
    pl.legend()

    pl.subplot(212)
    pl.plot(fy.real, label="Real of np.fft")
    pl.plot(fy.imag, label="Imag of np.fft")
    pl.title("Time space")
    pl.xlabel("Time")
    pl.legend()
    pl.axis("tight")

    pl.show()
