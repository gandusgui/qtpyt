from typing import Any

import numpy as np
from numba import njit, prange
from qtpyt.base._kernels import dagger
from qtpyt.parallel.tools import comm_sum
from qtpyt.screening.distgf import DistGreenFunction
from qtpyt.tools import tri2full

diag = lambda m: m.diagonal(axis1=1, axis2=2)


@njit("(f8,f8[:],c16[:,:,:,::1],c16[:,:,:,::1],c16[:,:,::1])", parallel=True)
def _mix(alpha, coeff, inp, out, new):
    """Mix input-output arrays.
         __
         \
    new =     (1-alpha) coeff[n] inp[n] + alpha coeff[n] out[n]
         /_ n  
    """
    ne, m, n = out.shape[1:]
    nk = len(coeff)
    for e in prange(ne):
        for i in range(m):
            for j in range(n):
                new[e, i, j] = 0.0
                for k in range(nk):
                    new[e, i, j] += (1 - alpha) * coeff[k] * inp[
                        k, e, i, j
                    ] + alpha * coeff[k] * out[k, e, i, j]


class ArrayList:
    """Array list to store mixing iterations.
    
    Args:
        N : # iterations to store
        
    Internal:
        array : array of shape = (N,) + shape
            Arrays in the N-th previous iterations.
        indices : list
            Indices that order the internal array.
            Example
                array[indices] = order array from oldest to newest.
        i = current index

    Example:
        In [2]: a = ArrayList(2)
        In [3]: a.append(np.ones(3))
        In [6]: a.append(np.ones(3)*2)
        In [9]: a.array
        Out[9]: 
        array([[1., 1., 1.],
               [2., 2., 2.]])
        In [10]: a.append(np.ones(3)*3)
        In [13]: a.array
        Out[13]: 
        array([[3., 3., 3.],
               [2., 2., 2.]])
        In [13]: a.array[a.indices]
        Out[13]: 
        array([[2., 2., 2.],
               [3., 3., 3.]])

    """

    def __init__(self, N) -> None:
        self.array = None
        self.N = N
        self.indices = []
        self.i = -1

    def append(self, X):
        if self.array is None:
            self.array = np.empty((self.N,) + X.shape, X.dtype)
        self.i += 1
        self.i %= self.N
        self.indices.append(self.i)
        if len(self) > self.N:
            self.indices.pop(0)
        self.array[self.i] = X

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.array[i]


# class Pulay:
#     """Pulay mixing scheme.

#          __
#          \
#     new =     (1-alpha) coeff[n] inp[n] + alpha coeff[n] out[n]
#          /_ n

#     Args:
#         alpha : float
#             mixing factor
#         N : # of iterations to store
#     """

#     def __init__(self, alpha=0.4, N=3):
#         self.alpha = alpha
#         self.N = N
#         self.Gr_inp = []  # ArrayList(N)
#         self.Gr_out = []  # ArrayList(N)
#         self.eps = []

#     def _append(self, G_list, G):
#         G_list.append(G)
#         if len(G_list) > self.N:
#             G_list.pop(0)

#     def append_input(self, gf: DistGreenFunction):
#         """Append input."""
#         self._append(self.Gr_inp, gf.arrays["r"].copy())

#     def append_output(self, gf: DistGreenFunction):
#         """Append output."""
#         self._append(self.Gr_out, gf.arrays["r"].copy())
#         # Compute inner product with all previous (and current) residuals.
#         N = len(self.Gr_out)
#         i = N - 1
#         eps = [None] * N
#         for j in range(N):
#             eps[j] = np.dot(
#                 (self.Gr_out[i] - self.Gr_inp[i]).flat,
#                 (self.Gr_out[j] - self.Gr_inp[j]).flat,
#             ) + np.dot(
#                 (self.Gr_out[i].real - self.Gr_inp[i].real).flat,
#                 (self.Gr_out[j].real - self.Gr_inp[j].real).flat,
#             )
#         # Append lower(upper) triangular entries of Pulay matrix.
#         self.eps.append(eps)
#         if len(self.eps) > self.N:
#             self.eps.pop(0)  # Discard row
#             for eps in self.eps[:-1]:
#                 eps.pop(0)  # Discard column

#     def compute_new_coeff(self):
#         """Compute Pulay's mixing coefficients."""
#         N = len(self.Gr_out)
#         A = np.zeros((N + 1, N + 1))
#         B = np.zeros(N + 1)
#         B[N] = 1.0
#         # Fill lower triangular.
#         for i, eps in enumerate(self.eps):
#             A[i, : i + 1] = eps
#         # Fill symmetric.
#         tri2full(A)
#         A = comm_sum(A)
#         A[:N, N] = 1.0
#         A[N, :N] = 1.0
#         self.A = A
#         self.coeff = np.linalg.solve(A, B)[:-1]
#         return self.coeff

#     def compute_new_input(self, G_inp: ArrayList, G_out: ArrayList, new=None):
#         """Mix previous inputs and outputs and produce new input."""
#         if new is None:
#             new = np.zeros_like(G_inp[0])
#         else:
#             new[:] = 0.0
#         coeff = self.compute_new_coeff()
#         for i in range(len(coeff)):
#             new += (1 - self.alpha) * coeff[i] * G_inp[i] + self.alpha * coeff[
#                 i
#             ] * G_out[i]
#         # _mix(self.alpha, coeff, G_inp.array, G_out.array, new)
#         return new

#     def update_input(self, gf: DistGreenFunction):
#         """Update Green's function arrays with input computed from Pulay."""
#         Gr = gf.arrays["r"]
#         Gl = gf.arrays["l"]
#         self.compute_new_input(self.Gr_inp, self.Gr_out, Gr)
#         fermi = gf.gf0.fermi
#         for e, energy in enumerate(gf.energies):
#             Gl[e] = -fermi(energy) * (Gr[e] - dagger(Gr[e]))
#         self.append_input(gf)

#     def initialize(self, gf: DistGreenFunction, **kwargs):
#         """Update Green's function and initialize Pulay."""
#         child = None
#         if gf.parent is not None:
#             child = gf
#             gf = gf.parent
#         gf.update(lg=False, **kwargs)
#         self.append_input(gf)
#         gf.convert_retarded()
#         if child is not None:
#             child.update_from_parent(gf)

#     def step(self, gf: DistGreenFunction, **kwargs):
#         """Update Green's function and Pulay's internals."""
#         child = None
#         if gf.parent is not None:
#             child = gf
#             gf = gf.parent
#         gf.update(lg=False, **kwargs)
#         self.append_output(gf)
#         self.update_input(gf)  # << Gr, G<
#         gf.convert_retarded()  # << G<, G>
#         if child is not None:
#             child.update_from_parent(gf)


class PulayMixer:
    """Pulay mixing scheme.

         __
         \
    new =     (1-alpha) coeff[n] inp[n] + alpha coeff[n] out[n]
         /_ n  
         
    Args:
        alpha : float
            mixing factor
        N : # of iterations to store
    """

    def __init__(self, alpha=0.4, N=3):
        self.alpha = alpha
        self.N = N
        self.inp = []  # ArrayList(N)
        self.out = []  # ArrayList(N)
        self.eps = []

    def _append(self, G_list, G):
        G_list.append(G)
        if len(G_list) > self.N:
            G_list.pop(0)

    def append_input(self, D: np.ndarray):
        """Append input."""
        self._append(self.inp, D)

    def append_output(self, D: np.ndarray):
        """Append output."""
        self._append(self.out, D)
        # Compute inner product with all previous (and current) residuals.
        N = len(self.out)
        i = N - 1
        eps = [None] * N
        for j in range(N):
            eps[j] = np.vdot(self.out[i] - self.inp[i], self.out[j] - self.inp[j]).real
        # Append lower(upper) triangular entries of Pulay matrix.
        self.eps.append(eps)
        if len(self.eps) > self.N:
            self.eps.pop(0)  # Discard row
            for eps in self.eps[:-1]:
                eps.pop(0)  # Discard column

    def compute_new_coeff(self):
        """Compute Pulay's mixing coefficients."""
        N = len(self.out)
        A = np.zeros((N + 1, N + 1))
        B = np.zeros(N + 1)
        B[N] = 1.0
        # Fill lower triangular.
        for i, eps in enumerate(self.eps):
            A[i, : i + 1] = eps
        # Fill symmetric.
        tri2full(A)
        # A = comm_sum(A)
        A[:N, N] = 1.0
        A[N, :N] = 1.0
        self.A = A
        self.coeff = np.linalg.solve(A, B)[:-1]
        return self.coeff

    def compute_new_input(self, new=None):
        """Mix previous inputs and outputs and produce new input."""
        if new is None:
            new = np.zeros_like(self.inp[0])
        else:
            new[:] = 0.0
        coeff = self.compute_new_coeff()
        for i in range(len(coeff)):
            new += (1 - self.alpha) * coeff[i] * self.inp[i] + self.alpha * coeff[
                i
            ] * self.out[i]
        return new

    def __call__(self, func, D_inp, tol=1e-5, max_iter=100) -> Any:
        self.append_input(D_inp)
        iter = 0
        eps = np.inf
        while eps > tol and iter < max_iter:
            iter += 1
            D_out = func(D_inp)
            self.append_output(D_out)
            eps = abs((D_out - D_inp).real.sum())
            D_inp = self.compute_new_input()
            self.append_input(D_inp)
