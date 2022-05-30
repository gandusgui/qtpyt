import numpy as np
import multiprocessing as mp


def tonumpyarray(mp_arr, shape, dtype):
    """Convert shared multiprocessing array to numpy array.

    no data copying
    """
    return np.frombuffer(mp_arr, dtype=dtype).reshape(shape)


def slices(nitems, mslices):
    """Split nitems on mslices pieces.

    >>> list(slices(10, 3))
    [slice(0, 4, None), slice(4, 8, None), slice(8, 10, None)]
    >>> list(slices(1, 3))
    [slice(0, 1, None), slice(1, 1, None), slice(2, 1, None)]
    """
    step = nitems // mslices + 1
    for i in range(mslices):
        yield slice(i * step, min(nitems, (i + 1) * step))
