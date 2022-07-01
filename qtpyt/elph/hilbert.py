import numpy as np
from qtpyt.screening import fft, ifft
from qtpyt.screening.tools import increase2pow2
from scipy import ifft

"""
Methods for perfoming the Hilbert transform of a function::

                     +oo
            1       /     f(x) 
  H[f](y) = -- p.v. | dx -----
            pi      /    x - y  
                     -oo

See en.wikipedia.org/wiki/Hilbert_transform (opposite sign)
"""


def hilbert_kernel_simple(n):
    """Construct Hilbert kernel with n grid points.
    
    This is just the discrete Fourier transform of 1 / x.
    """
    ker = np.zeros(n, dtype=complex)
    ker[1 : n // 2] = 1.0j
    ker[n // 2 + 1 :] = -1.0j
    return ker


def hilbert_kernel_interpolate(n):
    """Construct Hilbert kernel with n grid points.
    
    This is just the discrete Hilbert transform of the linear
    interpolation kernel `L(s) = (1 - |s|) Heaviside(1 - |s|)`.
    """
    # middle grid point
    mid = (n + 1) // 2

    # Make auxiliary array
    aux = np.arange(mid + 1, dtype=float)
    np.multiply(aux[1:], np.log(aux[1:]), aux[1:])

    # Make kernel
    ker = np.zeros(n, float)
    ker[1:mid] = aux[2:] - 2 * aux[1:-1] + aux[:-2]
    ker[-1:-mid:-1] = -ker[1:mid]

    return -fft(ker) / np.pi


def hilbert(f, oversample=10, axis=0, ker=None, kerneltype="interpolate"):
    """Perform Hilbert transform."""
    # Number of transform grid points
    n = f.shape[axis]

    # Generate new kernel if needed
    if ker == None:
        if oversample > 0:
            nfft = increase2pow2(oversample * n)
        else:
            nfft = n    
        ker = eval("hilbert_kernel_" + kerneltype)(nfft)
    else:
        assert ker.size == nfft

    # Reshape kernel
    ker_shape = [1,] * len(f.shape)
    ker_shape[axis] = nfft
    ker.shape = tuple(ker_shape)

    # Make convolution of f and kernel
    hil = ifft(fft(f, n=nfft, axis=axis) * ker, axis=axis)
    return hil[0:n]
