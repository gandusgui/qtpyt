import numpy as np

try:
    import mkl_fft
except:
    fft = np.fft.fft
    ifft = np.fft.ifft
else:
    fft = mkl_fft.fft
    ifft = mkl_fft.ifft
