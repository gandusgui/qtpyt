import numpy as np

#### NeighborList
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist


def get_neighbors(atoms):

    cov_radii = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(cov_radii, bothways=True, self_interaction=False)
    nl.update(atoms)

    nlists = []
    for a in atoms:
        nlists.append(nl.get_neighbors(a.index)[0])

    return nlists
