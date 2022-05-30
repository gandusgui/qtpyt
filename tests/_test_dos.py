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
def test_dos(prefix, method, get_expected, setup):

    tcalc = setup(method, prefix)
    energies, D = get_expected(prefix, "dos")
    tcalc.initialize()
    GF = tcalc.greenfunction
    dos_e = np.zeros_like(energies)
    for e, energy in enumerate(energies):
        dos_e[e] = GF.dos(energy)
    assert np.allclose(D, dos_e)


# test_transmission(setup)
