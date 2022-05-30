from transport import mpi
from qtpyt.calculators import TransportCalculator
import numpy as np
import pytest

N = 40

rank = mpi.rank
size = mpi.size
root = 0

# @pytest.mark.skip
@pytest.fixture
def setup():
    def inner():
        m = 10
        i2 = 3 * 2
        H = np.random.random((m, m))
        H += H.T.conj()
        S = np.random.random((m, m))
        S += S.T.conj()
        H1 = np.random.random((i2, i2)).astype(complex)
        H1 += H1.T.conj()
        S1 = np.random.random((i2, i2)).astype(complex)
        S1 += S1.T.conj()
        return TransportCalculator(h=H, s=S, h1=H1, s1=S1)

    return inner


@pytest.mark.parametrize("energies", [np.arange(-2, 2, 0.1), np.arange(-2.3, 2, 0.1)])
def test_parallel(energies, setup):
    tcalc = setup()
    tcalc.set(energies=energies)
    tcalc.get_transmission()
    if rank == 0:
        assert np.allclose(tcalc.energies, energies[: (energies.size // size) * size])
        assert len(tcalc.energies) == len(tcalc.T_e)
    else:
        assert (tcalc.energies is None) and (tcalc.T_e is None)
