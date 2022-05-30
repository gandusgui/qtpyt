import numpy as np
import cupy as cp

device = cp.cuda.device

cublas = device.cublas
cusolver = device.cusolver

cublas_handle = device.get_cublas_handle()
cusolver_handle = device.get_cusolver_handle()


def _assert_farray(*arrays):
    for array in arrays:
        assert array.flags.f_contiguous


def solve(a, b, overwrite_b=False):
    _assert_farray(a, b)
    if overwrite_b:
        b = b.copy("A")

    getrf = cusolver.zgetrf
    getrf_buffer = cusolver.zgetrf_bufferSize
    getrs = cusolver.zgetrs
    trans = cublas.CUBLAS_OP_N

    m, n = a.shape
    nrhs = b.shape[1]
    lda = m
    ldb = m

    dev_info = cp.empty(1, dtype=np.int32)
    ipiv = cp.empty(m, np.int32)

    buffersize = getrf_buffer(cusolver_handle, m, n, a.data.ptr, lda)
    workspace = cp.empty(buffersize, dtype=a.dtype)

    getrf(
        cusolver_handle,
        m,
        n,
        a.data.ptr,
        lda,
        workspace.data.ptr,
        ipiv.data.ptr,
        dev_info.data.ptr,
    )
    del workspace

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        b.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    return b


def lrsolve(a, b):
    _assert_farray(a, b)

    getrf = cusolver.zgetrf
    getrf_buffer = cusolver.zgetrf_bufferSize
    getrs = cusolver.zgetrs
    trans = cublas.CUBLAS_OP_N

    m, n = a.shape
    nrhs = b.shape[1]
    lda = m
    ldb = m

    dev_info = cp.empty(1, dtype=np.int32)
    ipiv = cp.empty(m, np.int32)

    buffersize = getrf_buffer(cusolver_handle, m, n, a.data.ptr, lda)
    workspace = cp.empty(buffersize, dtype=a.dtype)

    getrf(
        cusolver_handle,
        m,
        n,
        a.data.ptr,
        lda,
        workspace.data.ptr,
        ipiv.data.ptr,
        dev_info.data.ptr,
    )
    del workspace

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        b.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    cp.conj(a, out=a)

    c = b.copy(order="C")

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        c.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    return c


def tsolve(a, b, c):
    _assert_farray(a, b, c)

    getrf = cusolver.zgetrf
    getrf_buffer = cusolver.zgetrf_bufferSize
    getrs = cusolver.zgetrs
    trans = cublas.CUBLAS_OP_N

    m, n = a.shape
    nrhs = b.shape[1]
    lda = m
    ldb = m

    dev_info = cp.empty(1, dtype=np.int32)
    ipiv = cp.empty(m, np.int32)

    buffersize = getrf_buffer(cusolver_handle, m, n, a.data.ptr, lda)
    workspace = cp.empty(buffersize, dtype=a.dtype)

    getrf(
        cusolver_handle,
        m,
        n,
        a.data.ptr,
        lda,
        workspace.data.ptr,
        ipiv.data.ptr,
        dev_info.data.ptr,
    )
    del workspace

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        b.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    trans = cublas.CUBLAS_OP_C

    getrs(
        cusolver_handle,
        trans,
        n,
        nrhs,
        a.data.ptr,
        lda,
        ipiv.data.ptr,
        c.data.ptr,
        ldb,
        dev_info.data.ptr,
    )

    return b, c
