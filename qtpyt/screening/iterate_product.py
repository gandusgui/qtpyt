import numpy as np
from qtpyt import xp

# pinned_memory_pool = xp.cuda.PinnedMemoryPool()
# xp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)


# def _pin_memory(array):
#     mem = xp.cuda.alloc_pinned_memory(array.nbytes)
#     ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
#     ret[...] = array
#     return ret


def iterate_gpu(func, x, y, out, **kwargs):
    # Note that x and y should be pinned memory

    x_gpu = xp.empty(x.shape[1:], x.dtype)
    y_gpu = xp.empty(y.shape[1:], y.dtype)
    out_gpu = xp.empty(out.shape[1:], out.dtype)

    kernel_stream = xp.cuda.stream.Stream(non_blocking=True)
    H2D_stream = xp.cuda.stream.Stream(non_blocking=True)
    D2H_stream = xp.cuda.stream.Stream(non_blocking=True)

    for i in range(x.shape[0]):
        with H2D_stream:
            x_gpu.set(x[i])
            y_gpu.set(y[i].conj())

        H2D_event = H2D_stream.record()
        kernel_stream.wait_event(H2D_event)

        with kernel_stream:
            func(x_gpu, y_gpu, out=out_gpu, **kwargs)

        kernel_event = kernel_stream.record()
        D2H_stream.wait_event(kernel_event)

        with D2H_stream:
            out_gpu.get(out=out[i])

    end_event = D2H_stream.record()
    end_event.synchronize()


def iterate_cpu(func, x, y, out, **kwargs):

    for i in range(x.shape[0]):
        func(x[i], y[i].conj(), out=out[i], **kwargs)


iterate_product = iterate_cpu

if xp.__name__ == "cupy":
    iterate_product = iterate_gpu
