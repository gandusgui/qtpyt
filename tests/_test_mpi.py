from transport import mpi
import numpy as np
import pytest

N = 40

rank = mpi.rank
size = mpi.size
root = 0


@pytest.mark.skip
@pytest.mark.parametrize("dtype", ["int64", "float64", "complex64", "complex128"])
def test_scatter_dtype(dtype):
    # N = 4
    array = np.arange(N, dtype=dtype)
    stride = mpi.get_stride(array)
    out = mpi.scatter(array)
    assert np.allclose(out, array[rank * stride : (rank + 1) * stride])


@pytest.mark.skip
def test_scatter_shape():
    # N = 40
    array = np.arange(N, dtype="float").reshape(4, 10)
    stride = mpi.get_stride(array)
    out = mpi.scatter(array)
    expected = array.flat[rank * stride : (rank + 1) * stride]
    assert np.allclose(out, expected)


@pytest.mark.skip
def test_gather():
    # N = 10
    array = np.arange(N * rank, N * (rank + 1), dtype="float")
    out = mpi.gather(array)
    if rank == root:
        expected = np.arange(N * size, dtype="float")
        assert np.allclose(out, expected)
    else:
        expected = None
        assert out == expected
