from mpi4py import MPI
import numpy as np
import warnings

world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

MPI_type = {
    "int64": MPI.LONG,
    "float64": MPI.DOUBLE,
    "complex64": MPI.COMPLEX,
    "complex128": MPI.DOUBLE_COMPLEX,
}


def print(obj, root=0, comm=world):
    if comm.Get_rank() == root:
        print(obj)


def get_stride(array, comm=world):
    n = array.size
    stride = n // comm.Get_size()
    if n % comm.Get_size() != 0:
        Warning("Cannot parallelize array.")
    return stride


def get_domain(array, comm=world):
    rank = comm.Get_rank()
    stride = get_stride(array, comm)
    out = np.empty(stride, dtype=array.dtype)
    out[:] = array[rank * stride : (rank + 1) * stride]
    return out


def scatter(array, root=0, comm=world):
    stride = get_stride(array, comm)
    send_data = None
    send_count = 0
    send_type = MPI_type[str(array.dtype)]
    recv_count = stride
    recv_data = np.empty(recv_count, dtype=array.dtype)
    recv_type = send_type
    if comm.Get_rank() == root:
        send_data = array
        send_count = stride
    sendbuf = [send_data, send_count, send_type]
    recvbuf = [recv_data, recv_count, recv_type]
    comm.Scatter(sendbuf, recvbuf, root)
    return recv_data


def gather(array, root=0, comm=world):
    recv_data = None
    recv_count = 0
    recv_type = MPI_type[str(array.dtype)]
    send_data = array
    send_type = recv_type
    send_count = array.size
    if comm.Get_rank() == root:
        recv_count = send_count
        recv_data = np.empty(recv_count * comm.Get_size(), dtype=array.dtype)
    sendbuf = [send_data, send_count, send_type]
    recvbuf = [recv_data, recv_count, recv_type]
    comm.Gather(sendbuf, recvbuf, root)
    return recv_data
