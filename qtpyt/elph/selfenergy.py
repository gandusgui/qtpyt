import numpy as np
from qtpyt.elph.hilbert import hilbert, hilbert_kernel_interpolate
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.screening.langreth import LangrethPair, get_retarded_from_lesser_and_greater
from qtpyt.screening.tools import increase2pow2, roll

kB = 8.6e-5  # Boltzmann
nB = lambda w, kt: 1.0 / (np.exp(w / kt) - 1.0)


def translate(a, s, out, w=1.0):
    """Translate an array by s."""
    if s > 0:
        # f^-(x) = f(x-y)
        out[s:] += w * a[:-s]
    elif s < 0:
        # f^+(x) = f(x+y)
        s = abs(s)
        out[:-s] += w * a[s:]
    else:
        out += w * s


def setzeros(*arrays):
    """Set values of arrays to zero."""
    for a in arrays:
        a[:] = 0.0


class ElphFockSelfEnergy(LangrethPair):
    """Electron-phonon Fock self-energy.
    
    
    Args:
        gf : DistGreenFunction
            electron Green's function.
        g_lii : np.ndarray, shape = (# modes, # basis, # basis)
            electron-phonon coupling matrix.
        w_l : np.ndarray, shape = (# modes)
            phonon frequencies.
    """

    def __init__(
        self, gf: DistGreenFunction, g_lii: np.ndarray, w_l: np.ndarray, oversample=10
    ) -> None:
        self.gf = gf
        self.g_lii = g_lii
        self.w_l = w_l
        self.nph_l = nB(w_l, self.gf.gf0.kt)
        super().__init__(
            gf.global_energies, gf.no, dtype=complex, oversample=oversample
        )
        # We need retarded and lesser
        self.arrays["r"] = self.arrays.pop("g")
        setzeros(*(a for a in self.arrays.values()))

    def update(self):

        Gl = self.collect_energies(self.gf.arrays["l"])
        Gg = self.collect_energies(self.gf.arrays["g"])

        Gl_minus = np.empty_like(Gl)
        Gl_plus = np.empty_like(Gl)
        Gg_minus = np.empty_like(Gl)
        Gg_plus = np.empty_like(Gl)

        Sr = self.arrays["r"]
        Sl = self.arrays["l"]
        setzeros(Sr, Sl)

        for w, g_ii, nph in zip(self.w_l, self.g_lii, self.nph_l):

            s = w / self.de
            r = s % 1  # number from (0,1)
            s = int(s - r)

            setzeros(Gl_minus, Gl_plus, Gg_minus, Gg_plus)

            # interpolate
            for weight in (1 - r, r):
                if round(weight, 5) > 0.0:
                    translate(Gl, s, Gl_minus, weight)
                    translate(Gg, s, Gg_minus, weight)
                    translate(Gl, -s, Gl_plus, weight)
                    translate(Gg, -s, Gg_plus, weight)
                s += 1

            l = nph * Gl_minus + (1 + nph) * Gl_plus
            g = nph * Gg_minus + (1 + nph) * Gg_plus

            # l = roll(l, -self.zero_index)
            # g = roll(g, -self.zero_index)

            # r = get_retarded_from_lesser_and_greater(
            #     l, g, self.global_energies, g, self.oversample
            # )

            # l = roll(l, self.zero_index)
            # r = roll(r, self.zero_index)
            r = 0.5 * (g - l)
            r -= 1.0j * hilbert(r, oversample=self.oversample)

            l = self.collect_orbitals(l)
            r = self.collect_orbitals(r)

            l = np.einsum("ij,ejk,kl->eil", g_ii, l, g_ii, optimize=True)
            r = np.einsum("ij,ejk,kl->eil", g_ii, r, g_ii, optimize=True)

            Sl += l
            Sr += r

    def retarded(self, energy):
        return self.arrays["r"][np.searchsorted(self.energies, energy)]
