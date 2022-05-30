import numpy as np
from qtpyt.parallel import MPI, comm


def comm_sum(a : np.ndarray):
    if a.ndim > 2:
        a = a.sum(0)
    if comm.size == 1:
        return a
    b = np.empty_like(a)
    mpitype = MPI._typedict[a.dtype.char]
    comm.Allreduce([a, a.size, mpitype], [b, mpitype], MPI.SUM)
    return b
