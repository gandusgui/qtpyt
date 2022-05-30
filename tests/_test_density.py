#!/usr/bin/env python

import numpy as np
import pytest

from qtpyt.continued_fraction import integrate_pdos


@pytest.mark.parametrize("prefix", ["poly2"])
@pytest.mark.parametrize(
    "method",
    [  #'from_gf',
        #'from_hs',
        "from_rgf"
    ],
)
# pytest.param('from_rgf',pytest.mark.recursive)])
def test_density(prefix, method, get_expected, setup):

    tcalc = setup(method, prefix)
    tcalc.initialize()
    N = get_expected(prefix, "density")
    #
    GF = tcalc.greenfunction
    density = GF.density()
    assert np.allclose(density, N)


# test_transmission(setup)
