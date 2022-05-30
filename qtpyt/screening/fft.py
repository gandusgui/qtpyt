import numpy as np

from . import use_pyfftw


class pyFFTwrap:
    def __init__(self, n, dtype) -> None:
        self.n = n
        self.a = pyfftw.empty_aligned(n, dtype)
        self.b = pyfftw.empty_aligned(n, dtype)
        self.fft = pyfftw.FFTW(
            self.a, self.b, flags=("FFTW_DESTROY_INPUT", "FFTW_MEASURE")
        )
        self.ifft = pyfftw.FFTW(
            self.b,
            self.a,
            direction="FFTW_BACKWARD",
            flags=("FFTW_DESTROY_INPUT", "FFTW_MEASURE"),
        )

    def __eq__(self, inp):
        self.fft.input_array[: self.n] = inp

    def forward(self, a):
        return self.fft(a)

    def backward(self, b):
        return self.ifft(b)


class npFFTwrap:
    def __init__(self, n, dtype) -> None:
        self.n = n
        self.dtype = dtype

    def __eq__(self, a):
        self.a = a

    def forward(self, a):
        return np.fft.fft(a, n=self.n)

    def backward(self, b):
        return np.fft.ifft(b)
