#!/usr/bin/env python3

from utilities import *
from ase.io import read

if __name__ == "__main__":
    filename = get_filename("*.xyz")
    atoms = read(filename)
    dir = input("Direction to repeat unit cells (x, y, z): ")
    str_sides = input("Extend at start (0) at end (1) or start and end (01): ")
    n_uc = int(input("Number of unit cell atoms (in tranverse directions) : "))
    n_rep = int(input("Number of unit cell repetitions (in transport direction) : "))
    sides = []
    for s in str_sides:
        sides.append(int(s))
    extend_atoms(atoms, n_uc, n_rep, direction=dir, sides=sides)
    filename = get_basename(filename) + "_ext.xyz"
    order_atoms(atoms)
    atoms.write(filename)
