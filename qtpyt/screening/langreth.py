from functools import wraps
from math import pi
from typing import Any

import numpy as np

from qtpyt.elph.hilbert import hilbert
from qtpyt.parallel.egrid import GridDesc
from qtpyt.screening import fft, ifft
from qtpyt.screening.fourier import fourier_integral, get_fourier_data
from qtpyt.screening.tools import (
    finer,
    get_extended_energies,
    get_interp_indices,
    increase2pow2,
    lesser_from_retarded,
    linear_interp,
    roll,
    smooth,
)


def get_retarded_from_lesser_and_greater(
    less, great, energies, ret=None, oversample=10
):
    """
    Calculates the retarded function from

    r(t) = theta(t)[ >(t) - <(t) ]
    """
    extenergies = get_extended_energies(energies, oversample)
    fourierdata = get_fourier_data(len(extenergies))
    de_ext = extenergies[1] - extenergies[0]
    Ne = len(energies)
    # Ne_ext = increase2pow2(oversample * Ne)  # len(extenergies)
    Ne_ext = len(extenergies)
    Nq = less.shape[1]
    assert len(less) == Ne

    if ret is None:
        ret = np.zeros((Ne, Nq), complex)

    for i in range(Nq):
        # Make the difference (> - <)
        # g = fft(great[:, i], n=Ne_ext)
        # g -= fft(less[:, i], n=Ne_ext)
        # ret_i = g
        # ret_i = fft(great[:, i], n=Ne_ext) - fft(less[:, i], n=Ne_ext)
        # ret_i = fft(less[:, i], n=Ne_ext) - fft(great[:, i], n=Ne_ext)
        ret_i = great[:, i] - less[:, i]

        # Pad by zeros to extended grid, and FFT to time
        ret_i = fft(ret_i, n=Ne_ext)

        # multiply by theta(t)
        ret_i[Ne_ext // 2 :] = 0.0  # this will zero all t<0
        # ret_i[: Ne_ext // 2] = 0.0  # this will zero all t<0

        # iFFT back to energy using endpoint corrections
        # ret_i = ifft(ret_i)  # <- the equivalent without end-point cor
        ret_i = fourier_integral(ret_i, de_ext, Ne_ext, sign=1, data=fourierdata) / (
            de_ext * Ne_ext
        )
        # Reduce ext grid to full grid and then take nu grid only to store
        ret[:, i] = ret_i[:Ne]
    return ret


def assert_domain(*domains):
    """Assert transformation is supported for domain(s)"""

    def inner(f):
        @wraps(f)
        def wrapper(self, *a, **k):
            assert (
                self.domain in domains
            ), f"{self.domain} is invalid for transormation {f.__name__}"
            return f(self, *a, **k)

        return wrapper

    return inner


def change_domain(domain):
    """Change domain at output."""

    def inner(f):
        @wraps(f)
        def wrapper(self, *a, **k):
            out = f(self, *a, **k)
            self.domain = domain
            return out

        return wrapper

    return inner


def swap_domain(f):
    """Swap domain from/to energy/time."""

    @wraps(f)
    def wrapper(self, *a, **k):
        assert self.domain is not None, f"{self.__name__} not initialized."
        out = f(self, *a, **k)
        self.domain = "et".replace(self.domain, "")
        return out

    return wrapper


class LangrethPair(GridDesc):
    """Descriptor of a pair of distributed arrays following Langreth 
    transformation rules.

    The langreth pairs are contained in a dictionary specifying
    the type as a keyword. An array can represent one of the
    
        l : lesser
        g : greater
        r : retarded
        a : advanced
    
    components of the inheriting class. This structure is thought
    so that a class can re-use effectively the allocated memory and
    keep track of what the arrays acutally represent. Additionally,
    the pair can leave in either of the three domains
    
        None : uninitialized
        e : energy
        t : time
        
    Note that both arrays will always live in the same domain.
    
    Attributes:
        convert_domain : converts the domain from/to energy
            or time.
        
                x(e), y(e) <</>> x(t), y (t)
            
        convert_retared : converts the retarded component to
            the lesser one. This method requires the current 
            domain to be energy.
            
                l(e), g(e) << r(e), g(e)
            
        convert_lesser : converts the lesser and greater components
            given in time domain to the retarded and greater ones
            in energy domain.

                r(e), g(e) << l(t), g(t)
        
    A call to these method will always trigger a domain consistency
    check. If the domain is invalid for the given transformation
    an assertation error is raised. At the end of the transformation
    the domain is consistently changed to the correct one.

    """

    _valid_keys = set(["l", "g", "r", "a"])
    _valid_domains = set([None, "e", "t"])

    def __init__(self, energies, no, dtype=complex, oversample=10) -> None:
        super().__init__(energies, no, dtype)
        self.arrays = {
            "l": self.empty_aligned_orbs(),
            "g": self.empty_aligned_orbs(),
        }
        self.oversample = oversample
        self.domain = None

    @assert_domain("e", "t")
    @swap_domain
    def convert_domain(self):
        """Convert pair from/to energy/time."""
        # x(t), y(t) <</>> x(e), y(e)
        # _fft = fft if self.domain == "e" else ifft
        # if self.domain == "e":
        # indices = get_interp_indices(self.global_energies)
        # indices = (self.global_energies <= -50.0) | (self.global_energies >= 50.0)
        for a in self.arrays.values():
            A = self.collect_energies(a)
            if self.domain == "e":
                # A[indices] = 0.0
                # A = smooth(self.global_energies, A, 1.0, 6.0)
                A = fft(A, axis=0)
                # A = fft(linear_interp(A[indices], indices, self.ne), axis=0)
            else:
                A = ifft(A, axis=0)
            self.collect_orbitals(A, a)

    @assert_domain("e")
    def convert_retarded(self):
        """Convert retarded to lesser."""
        # l(e) << r(e)
        # l(e) = g(e) - r(e) + a(e)
        lesser_from_retarded(self.arrays["r"], self.arrays["g"])
        self.arrays["l"] = self.arrays.pop("r")

    @assert_domain("t")
    @change_domain("e")
    def convert_less_and_great_to_ret(self, zero_index=None, override="l"):
        """Convert lessert to retarded."""
        # r(e), g(e) << l(t), g(t)
        # r(t) = theta(t) [ <(t) - >(t) ]
        if zero_index is None:
            zero_index = self.zero_index
        l = self.collect_energies(self.arrays["l"])
        g = self.collect_energies(self.arrays["g"])
        ####  -1-  ####
        # l = ifft(l, axis=0)
        # g = ifft(g, axis=0)
        # r = 0.5 * (g - l)
        # r -= 1.0j * hilbert(r, oversample=self.oversample)
        ###############

        ####  -2-  ####
        l = roll(ifft(l, axis=0), zero_index)
        g = roll(ifft(g, axis=0), zero_index)
        if override == "l":
            r = l
            x = g
            keep = "g"
        else:
            r = g
            x = l
            keep = "l"
        get_retarded_from_lesser_and_greater(
            l, g, self.global_energies, r, oversample=self.oversample
        )
        ###############
        self.arrays["r"] = self.arrays.pop(override)
        self.collect_orbitals(r, self.arrays["r"])
        self.collect_orbitals(x, self.arrays[keep])

    def __lshift__(self, other):
        old_k = tuple(self.arrays.keys())
        new_k = tuple(other.arrays.keys())
        for ok, nk in zip(old_k, new_k):
            self.arrays[ok][:] = other.arrays[nk]
        self.arrays = {new_k[0]: self.arrays[old_k[0]], new_k[1]: self.arrays[old_k[1]]}
        self.domain = other.domain
