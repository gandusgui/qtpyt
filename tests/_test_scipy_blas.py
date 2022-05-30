import numpy as np
from scipy import linalg as la
from gpaw.utilities.blas import gemm as gpaw_gemm
from timeit import default_timer as timer
import operator


n = 1000
m = 1000
dtype = np.complex64


def time(f, *args, **kwargs):
    start = timer()
    res = f(*args, **kwargs)
    end = timer()
    print(f, "took", end - start)
    return res


# gemm  =>  c = alpha*a*b + beta*c


########## c = a * b ############

# F
order = "F"

a_c = np.ones((n, m), dtype=dtype, order=order)
b_c = np.ones((m, n), dtype=dtype, order=order)
c_c = np.zeros_like(a_c)  # ((m,n),dtype=dtype,order=order)


res_np_dot = time(np.dot, a_c, b_c)  # np.dot(a_c,b_c)
gemm = la.blas.get_blas_funcs("gemm", (a_c,))
time(gemm, 1, a_c, b_c, 0, c_c, overwrite_c=True)  # np.dot(a_c,b_c)
assert np.allclose(res_np_dot, c_c)

# C
order = "C"

a_c = np.ones((n, m), dtype=dtype, order=order)
b_c = np.ones((m, n), dtype=dtype, order=order)
c_c = np.zeros_like(a_c)  # ((m,n),dtype=dtype,order=order)


res_np_dot = time(np.dot, a_c, b_c)  # np.dot(a_c,b_c)
gemm = la.blas.get_blas_funcs("gemm", (a_c,))
c_c = time(gemm, 1, a_c, b_c, 0, c_c, overwrite_c=True)  # np.dot(a_
assert np.allclose(res_np_dot, c_c)


########## c -= a * b ############

# F
order = "F"

a_c = np.ones((n, m), dtype=dtype, order=order)
b_c = np.ones((m, n), dtype=dtype, order=order)
c_c = np.zeros_like(a_c)  # ((m,n),dtype=dtype,order=order)
res_np_dot = c_c.copy()

start = timer()
res_np_dot -= np.dot(a_c, b_c)
end = timer()
print(np.dot, "took", end - start)
gemm = la.blas.get_blas_funcs("gemm", (a_c,))
time(gemm, -1, a_c, b_c, 1, c_c, overwrite_c=True)  # np.dot(a_c,b_c)
assert np.allclose(res_np_dot, c_c)


# C
order = "C"

a_c = np.ones((n, m), dtype=dtype, order=order)
b_c = np.ones((m, n), dtype=dtype, order=order)
c_c = np.zeros_like(a_c)  # ((m,n),dtype=dtype,order=order)
res_np_dot = c_c.copy()

start = timer()
res_np_dot -= np.dot(a_c, b_c)
end = timer()
print(np.dot, "took", end - start)
gemm = la.blas.get_blas_funcs("gemm", (a_c,))
c_c = time(gemm, -1, a_c, b_c, 1, c_c, overwrite_c=True)  # np.dot(a_c,b_c)
assert np.allclose(res_np_dot, c_c)

########## a -= a * b ############


# F
order = "F"

a_c = np.ones((n, m), dtype=dtype, order=order)
b_c = np.ones((m, n), dtype=dtype, order=order)
res_np_dot = a_c.copy()

start = timer()
res_np_dot -= np.dot(a_c, b_c)
end = timer()
print(np.dot, "took", end - start)
gemm = la.blas.get_blas_funcs("gemm", (a_c,))
time(gemm, -1, a_c.copy(), b_c, 1, a_c, overwrite_c=True)  # np.dot(a_c,b_c)
assert np.allclose(res_np_dot, a_c)

# F
order = "C"

a_c = np.ones((n, m), dtype=dtype, order=order)
b_c = np.ones((m, n), dtype=dtype, order=order)
res_np_dot = a_c.copy()

start = timer()
res_np_dot -= np.dot(a_c, b_c)
end = timer()
print(np.dot, "took", end - start)
gemm = la.blas.get_blas_funcs("gemm", (a_c,))
a_c = time(gemm, -1, a_c.copy(), b_c, 1, a_c, overwrite_c=True)  # np.dot(a_c,b_c)
assert np.allclose(res_np_dot, a_c)
