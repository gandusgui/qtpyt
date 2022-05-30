import numpy as np
from numba import boolean, njit

# https://stackoverflow.com/questions/9227747/in-place-transposition-of-a-matrix


@njit(["f8[:,::1](f8[:,::1])", "c16[:,::1](c16[:,::1])"], cache=True)
def transpose(a):
    """Transpose a matrix inplace.
    
    Example:
    
    In [1]: a = np.arange(8).reshape(2,4).astype(float)

    In [2]: a
    Out[2]: 
            array([[0., 1., 2., 3.],
                [4., 5., 6., 7.]])

    In [3]: transpose(a)

    In [4]: a
    Out[4]: 
            array([[0., 4., 1., 5.],
                [2., 6., 3., 7.]])
    """
    n, m = a.shape
    a = a.reshape(-1)
    nm1 = n * m - 1
    visited = np.zeros(a.size, boolean)
    cycle = -1
    while cycle != a.size - 1:
        cycle += 1
        if visited[cycle]:
            continue
        w = cycle
        while True:
            w = nm1 if w == nm1 else (n * w) % nm1
            # swap
            tmp = a[w]
            a[w] = a[cycle]
            a[cycle] = tmp
            visited[w] = True
            if w == cycle:
                break
    return a.reshape(m, n)
