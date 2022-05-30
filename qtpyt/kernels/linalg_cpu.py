import numpy as np
import ctypes

mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")

i4 = ctypes.c_int
f8 = ctypes.c_double
c16 = ctypes.c_longdouble


def dot(a, b, c, alpha=1.0, beta=0.0):

    Order = i4(101)
    TransA = i4(111)
    TransB = i4(111)

    a_ptr = np.ctypeslib.as_ctypes(a)
    b_ptr = np.ctypeslib.as_ctypes(b)
    c_ptr = np.ctypeslib.as_ctypes(c)

    m = i4(a.shape[0])
    n = i4(b.shape[1])
    k = i4(a.shape[1])

    lda = k
    ldb = n
    ldc = n

    mkl.cblas_dgemm(
        Order,
        TransA,
        TransB,
        m,
        n,
        k,
        f8(alpha),
        a_ptr,
        lda,
        b_ptr,
        ldb,
        f8(beta),
        c_ptr,
        ldc,
    )

    return c


def solve(a, b, overwrite_b=False):
    """Solve the equation:   
        Ax=B
    
    Returns: A^(-1)B
    """
    Order = i4(102) if a.flags.f_contiguous else i4(101)

    _b = b if overwrite_b else b.copy(order="A")

    a_ptr = a.ctypes.data_as(ctypes.POINTER(c16))
    b_ptr = _b.ctypes.data_as(ctypes.POINTER(c16))

    n = i4(a.shape[0])
    nrhs = i4(b.shape[1])
    ipiv = np.empty(a.shape[0], np.int32)
    ipiv_ptr = ipiv.ctypes.data_as(ctypes.POINTER(i4))

    lda = n
    ldb = n if b.flags.f_contiguous else nrhs

    mkl.lapacke_zgesv(Order, n, nrhs, a_ptr, lda, ipiv_ptr, b_ptr, ldb)

    return _b


def lrsolve(a, b):
    """Solve the equation:   
        Ax=B 
        A^(H)y=x

    Returns: A^(-1)BA^(H)^(-1)
    """
    Order = i4(102) if a.flags.f_contiguous else i4(101)

    a_ptr = a.ctypes.data_as(ctypes.POINTER(c16))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(c16))

    m = i4(a.shape[0])
    n = i4(a.shape[1])
    lda = m if a.flags.f_contiguous else n

    ipiv = np.empty(a.shape[0], np.int32)
    ipiv_ptr = ipiv.ctypes.data_as(ctypes.POINTER(c16))

    mkl.lapacke_zgetrf(Order, m, n, a_ptr, lda, ipiv_ptr)

    trans = ctypes.c_char(b"N")
    nrhs = i4(b.shape[1])
    ldb = m if b.flags.f_contiguous else nrhs

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, b_ptr, ldb)

    np.conj(a, out=a)

    c = b.copy(order="C") if a.flags.f_contiguous else b.copy(order="F")
    c_ptr = c.ctypes.data_as(ctypes.POINTER(c16))

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, c_ptr, ldb)

    return c


def tsolve(a, b, c):
    """Solve the equation:   
        Ax=B 
        A^(H)y=C

    Returns: A^(-1)B, A^(H)^(-1)C
    """
    Order = i4(102) if a.flags.f_contiguous else i4(101)

    a_ptr = a.ctypes.data_as(ctypes.POINTER(c16))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(c16))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(c16))

    m = i4(a.shape[0])
    n = i4(a.shape[1])
    lda = m if a.flags.f_contiguous else n

    ipiv = np.empty(a.shape[0], np.int32)
    ipiv_ptr = ipiv.ctypes.data_as(ctypes.POINTER(c16))

    mkl.lapacke_zgetrf(Order, m, n, a_ptr, lda, ipiv_ptr)

    trans = ctypes.c_char(b"N")
    nrhs = i4(b.shape[1])
    ldb = m if b.flags.f_contiguous else nrhs

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, b_ptr, ldb)

    trans = ctypes.c_char(b"C")
    nrhs = i4(c.shape[1])
    ldc = m if c.flags.f_contiguous else nrhs

    mkl.lapacke_zgetrs(Order, trans, n, nrhs, a_ptr, lda, ipiv_ptr, c_ptr, ldc)

    return b, c
