from gpaw import GPAW
from qtpyt.lcao.tightbinding import TightBinding
import numpy as np
import pickle

calc = GPAW("data/graphene.gpw", txt=None)
# tb = TightBinding.from_calculator(calc.atoms, calc)
tb = TightBinding.from_monkhorst_pack(calc.atoms, calc.wfs.kd.N_c)
h_kmm, s_kmm = pickle.load(open("data/graphene_hs1_k.pckl", "rb"))
tb.set_h_and_s(h_kmm, s_kmm)
cell = calc.atoms.cell
cell.array[-1, -1] = 0.0
bp = cell.bandpath(npoints=100)
eps_kn = tb.band_structure(bp.kpts)
np.testing.assert_allclose(eps_kn, np.load("data/graphene_eps_kn.npy"))
