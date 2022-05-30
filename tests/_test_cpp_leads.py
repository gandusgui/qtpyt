import numpy as np

from transport import _cpp
from qtpyt.selfenergy import LeadSelfEnergy
from ase.transport.selfenergy import LeadSelfEnergy as aseLSE
from timeit import default_timer as timer


def execute_function_n_times_and_return_result(f, num_iterations, tag):
    start = timer()
    first_result = f(1.0)
    end = timer()
    time_first_execution = end - start

    start = timer()
    for _ in range(num_iterations):
        f(1.0)
    end = timer()
    time_n_executions = end - start
    print(
        f"{tag} first: {time_first_execution}, multiple iterations: {time_n_executions}"
    )  # Time in seconds, e.g. 5.38091952400282

    return first_result


val = 3
eta = 1e-5
m = 100
i = 500
dtype = np.complex64
order = "F"

h_ii = np.array(np.diag([val] * i), dtype=dtype, order=order)
s_ii = np.eye(i, dtype=dtype, order=order)
h_im = np.zeros((i, m), dtype=dtype, order=order)
h_im[0, 0] = 1.0
h_im[1, 1] = 1.0
s_im = np.zeros((i, m), dtype=dtype, order=order)
h_ij = np.eye(i, dtype=dtype, order=order)
s_ij = np.zeros((i, i), dtype=dtype, order=order)


# lse = _cpp.LeadSelfEnergy(h_ii,s_ii,h_ij,s_ij,h_im,s_im,eta)
c_but_with_python_wrapper = LeadSelfEnergy(
    (h_ii, s_ii), (h_ij, s_ij), (h_im, s_im), eta=eta
)  # c++ with py-wrapper
pure_python = aseLSE((h_ii, s_ii), (h_ij, s_ij), (h_im, s_im), eta=eta)  # pure python

num_iterations = 1

pure_python_result = execute_function_n_times_and_return_result(
    pure_python.retarded, num_iterations, "python"
)
python_wrapped_c_result = execute_function_n_times_and_return_result(
    c_but_with_python_wrapper.retarded, num_iterations, "c_wrapped_python"
)
assert np.allclose(pure_python_result, python_wrapped_c_result)
