#!/usr/bin/env python

import numpy as np
import pytest


@pytest.mark.parametrize("prefix", ["poly2"])
@pytest.mark.parametrize(
    "method",
    [  #'from_gf',
        #'from_hs',
        "from_rgf"
    ],
)
# pytest.param('from_rgf',pytest.mark.recursive)])
def test_transmission(prefix, method, get_expected, setup):

    tcalc = setup(method, prefix)
    E, T = get_expected(prefix, "transmission")
    tcalc.initialize()
    tcalc.set(energies=np.arange(-2, -1.8, 0.05))
    tcalc.update()
    # assert tcalc.greenfunction.H[0,0]==tcalc.selfenergies[0].h_ii[0,0]
    assert np.allclose(E, tcalc.energies)
    assert np.allclose(T, tcalc.T_e)


# test_transmission(setup)
