import pickle

from ase import *
from ase.io import read
from gpaw import *

atoms = read("scatt.xyz")
basis = {"default": "szp(dzp)"}

calc = GPAW(
    h=0.2,
    xc="PBE",
    nbands="nao",
    convergence={"bands": "all"},
    basis=basis,
    occupations=FermiDirac(width=0.01),
    kpts=(1, 1, 1),
    mode="lcao",
    txt="scatt.txt",
    mixer=Mixer(0.02, 5, 100),
    parallel=dict(
        band=2,  # band parallelization
        augment_grids=True,  # use all cores for XC/Poisson
        sl_auto=True,  # enable ScaLAPACK parallelization
    ),
)
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write("scatt.gpw")

fermi = calc.get_fermi_level()
print(repr(fermi), file=open("fermi_scatt.txt", "w"))
