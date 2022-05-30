#!/usr/bin/env python3

from utilities import *
from ase.io import read

if __name__ == "__main__":
    filename = get_filename("*.xyz")
    atoms = read(filename)
    order_atoms(atoms)
    filename = get_basename(filename) + "_ord.xyz"
    atoms.write(filename)
