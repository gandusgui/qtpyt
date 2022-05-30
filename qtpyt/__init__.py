# from .calculators import TransportCalculator
# __all__ = ['TransportCalculator']
import numpy as np
from scipy import linalg

xp = np
xla = linalg

try:
    import cupy as cp
except:
    pass
else:
    xp = cp
    xla = cp.linalg


# try:
#    import _cppmodule as _cpp
# except ImportError as e:
#    if "GLIBCXX" in str(e):
#        msg = ("Cannot import __cppmodule")
#        raise ImportError(msg).with_traceback(e.__traceback__)
#    else:
#        raise
