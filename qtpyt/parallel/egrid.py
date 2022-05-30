import itertools
from functools import wraps
from warnings import warn

import numpy as np
from qtpyt.parallel import MPI, comm
from qtpyt.parallel.pencil import Pencil, Subcomm, _blockdist
# from qtpyt.parallel.tools import comm_sum
from qtpyt.parallel.transpose import transpose


def check_contiguous(dist_func):
    """Check input arrays are c_contiguous or transpose inplace."""

    def c_contiguous(a: np.ndarray):
        if a.flags.f_contiguous:
            return transpose(a.T)
        return a

    @wraps(dist_func)
    def wrapper(self, a, b=None):
        """Transpose input and output inplace if necessary."""
        a = c_contiguous(a)
        b = dist_func(self, a, b)
        return c_contiguous(b)

    return wrapper


def check_serial(dist_func):
    if comm.size != 1:
        return dist_func

    @wraps(dist_func)
    def collect_serial(self, a, b=None):
        if b is not None and a is not b:
            b[:] = a
            return b
        return a

    return collect_serial


def reshape(arg):
    def reshape_array(a, N):
        if a is not None:
            a = a.reshape(N)
        return a

    if arg == "input":

        def reshape_input(dist_func):
            """Reshape input array from energy-orbital matrix
            to energy-flatten orbital index."""

            @wraps(dist_func)
            def wrapper(self, a, b=None):
                a = reshape_array(a, self.N2D)
                return dist_func(self, a, b)

            return wrapper

        return reshape_input

    elif arg == "output":

        def reshape_output(dist_func):
            """Same as `reshape_input` for output array but also
            to the inverse reshaping at the end."""

            @wraps(dist_func)
            def wrapper(self, a, b=None):
                b = reshape_array(b, self.N2D)
                b = dist_func(self, a, b)
                return reshape_array(b, self.N3D)

            return wrapper

        return reshape_output


class DistGrid:
    """Distributed grid."""

    def __init__(self, ne, no, dtype=complex) -> None:
        subcomms = Subcomm(comm, [0, 1])
        # Array aligned in orbital (1-axis).
        self._pencil_aligned_orbs = Pencil(subcomms, (ne, no ** 2), 1)
        # Array aligned in energy (0-axis).
        # try:
        self._pencil_aligned_eners = self._pencil_aligned_orbs.pencil(0)
        self.transfer = self._pencil_aligned_orbs.transfer(
            self._pencil_aligned_eners, dtype
        )
        # except:
        #     warn("Tranfer not implemented.")
        self.dtype = dtype
        # Array shapes
        self.N2D = self._pencil_aligned_orbs.subshape
        self.N3D = (self._pencil_aligned_orbs.subshape[0], no, no)
        # Global shapes
        self.ne = ne
        self.no = no

    def shape(self):
        return (self.ne, self.no)

    def _assert_dtypes(self, *arrays: np.ndarray):
        for a in arrays:
            assert a.dtype == self.dtype

    @reshape("input")
    @check_contiguous
    @check_serial
    def collect_energies(self, a: np.ndarray, b: np.ndarray = None):
        """From orbital-contiguous to energy-contiguous."""
        if b is None:
            b = self.empty_aligned_eners()
        self._assert_dtypes(a, b)
        self.transfer.forward(a, b)
        return b

    @reshape("output")
    @check_contiguous
    @check_serial
    def collect_orbitals(self, a: np.ndarray, b: np.ndarray = None):
        """From energy-contiguous to orbital-contiguous."""
        if b is None:
            b = self.empty_aligned_orbs().reshape(self.N2D)
        self._assert_dtypes(a, b)
        self.transfer.backward(a, b)
        return b

    def empty_aligned_eners(self):
        return np.empty(self._pencil_aligned_eners.subshape, self.dtype)

    def empty_aligned_orbs(self):
        return np.empty(self.N3D, self.dtype)


class GridDesc(DistGrid):
    """Grid descriptor."""

    def __init__(self, energies, no, dtype=complex) -> None:
        self.global_energies = energies
        super().__init__(energies.size, no, dtype)
        # Helpers for subclasses
        self.zero_index = abs(self.global_energies).argmin()
        self.de = abs(self.global_energies[1] - self.global_energies[0])

    @property
    def oslice(self):
        """Orbital slice."""
        substart = self._pencil_aligned_eners.substart[1]
        subshape = self._pencil_aligned_eners.subshape[1]
        return slice(substart, substart + subshape)

    @property
    def eslice(self):
        """Energy slice."""
        substart = self._pencil_aligned_orbs.substart[0]
        subshape = self._pencil_aligned_orbs.subshape[0]
        return slice(substart, substart + subshape)

    @property
    def energies(self):
        """Local energies."""
        return self.global_energies[self.eslice]

    def get_eintegral_weights(self):
        """Get energy grid weights for trapeziodal integration."""
        if not hasattr(self, "weights"):
            # Trapezoidal rule
            weights = np.empty(self.ne, self.dtype)
            weights[1:-1] = self.global_energies[2:] - self.global_energies[:-2]
            weights[0] = self.global_energies[1] - self.global_energies[0]
            weights[-1] = self.global_energies[-1] - self.global_energies[-2]
            self.weights = weights[self.eslice]
            self.weights /= 2.0
        return self.weights

    def sum_energies(self, a: np.ndarray, pre=1.0):
        """Sum orbital aligned array over global energies."""
        assert a.shape == self.N3D, "Can only sum orbital aligned arrays."
        self.weights = self.get_eintegral_weights()
        a = np.tensordot(pre * self.weights, a, axes=([0], [0]))
        if comm.size == 1:
            return a
        b = np.empty_like(a)
        mpitype = MPI._typedict[a.dtype.char]
        comm.Allreduce([a, a.size, mpitype], [b, mpitype], MPI.SUM)
        return b

    def gather_energies(self, a: np.ndarray, root: int = 0):
        """Gather 1D array distributed over energies to root."""
        assert a.ndim == 1, "Can only gather 1D arrays."
        if comm.size == 1:
            return a
        if comm.rank == root:
            b = np.empty(self.ne, a.dtype)
        else:
            b = None
        mpitype = MPI._typedict[a.dtype.char]
        counts = np.empty(comm.size, int)
        displs = np.empty(comm.size, int)
        for r in range(comm.size):
            counts[r], displs[r] = _blockdist(self.ne, comm.size, r)
        comm.Gatherv([a, a.size, mpitype], [b, counts, displs, mpitype], root)
        return b

    def write(self, a, filename, condition=None):
        """Write array to file in parallel.
        
        Args :
            a : array aligned along energies.
            condition : global condition on gloabl energies,
                e.g. (energies >= 0.) & (energies <= 6.)
        """
        assert (
            np.prod(a.shape[1:]) == self._pencil_aligned_orbs.subshape[1]
        ), "Must be orbital-aligned array."
        # indx = np.flatnonzero(condition)[self.eslice]
        if condition is None:
            condition = self.global_energies < np.inf
        indx = np.asarray(condition, bool)[self.eslice]
        indx = np.flatnonzero(indx)
        # comm = self.pencil_aligned_orbs.subcomm[0]
        send_buff = np.array([len(indx)], np.int32)
        recv_buff = np.empty(comm.size, np.int32)
        comm.Allgather([send_buff, 1, MPI.INT32_T], [recv_buff, 1, MPI.INT32_T])
        # Write to file
        amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
        fh = MPI.File.Open(comm, filename, amode)
        nbytes = a.itemsize * self._pencil_aligned_orbs.subshape[1]
        offset = recv_buff[: comm.rank].sum() * nbytes
        fh.Write_at_all(offset, a[indx])
        fh.Close()
