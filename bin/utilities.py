#!/usr/bin/env python

import numpy as np
from glob import glob
from os.path import join, basename

#### ADD Hydrogens
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist


def get_filename(pattern, assert_single=False):
    filename = glob(pattern)
    if assert_single:
        assert len(filename) == 1
    if len(filename) > 1:
        filenames = [repr(tu) for tu in enumerate(filename)]
        stdout = "Which file?\n {} \n:".format("\n".join(filenames))
        filename = filename[int(input(stdout))]
    else:
        filename = filename[0]
    return filename


def get_basename(filename):
    return basename(filename).split(".")[0]


def map_atoms(atoms, map):
    for name, a in atoms.arrays.items():
        atoms.arrays[name] = a[map]


def add_hydrogens(atoms):

    cov_radii = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(cov_radii, bothways=True, self_interaction=False)
    nl.update(atoms)

    need_a_H = []
    for a in atoms:
        nlist = nl.get_neighbors(a.index)[0]
        if len(nlist) < 3:
            if a.symbol == "C":
                need_a_H.append(a.index)

    print("Added missing Hydrogen atoms: ", need_a_H)

    dCH = 1.1
    for a in need_a_H:
        vec = np.zeros(3)
        indices, offsets = nl.get_neighbors(atoms[a].index)
        for i, offset in zip(indices, offsets):
            vec += -atoms[a].position + (
                atoms.positions[i] + np.dot(offset, atoms.get_cell())
            )
        vec = -vec / np.linalg.norm(vec) * dCH
        vec += atoms[a].position
        htoadd = ase.Atom("H", vec)
        atoms.append(htoadd)


def extend_atoms(atoms, n_uc, n_rep, direction="x", sides=[0, 1]):
    ## sides indicates whether to extend start (0) and/or end (1) along transport

    if not atoms.cell.orthorhombic:
        raise NotImplementedError("Cell must be orthorhombic")

    if isinstance(sides, int):
        sides = [sides]

    orig = atoms
    order_atoms(atoms, direction)

    d = "xyz".index(direction)

    for n in range(n_rep):

        cell = atoms.cell[d, d]
        uc_ldl = atoms[-1 : -n_uc - 1 : -1]
        uc_ldr = atoms[0:n_uc]

        uc_ldr.positions[:, d] += cell
        uc_ldl.positions[:, d] -= cell

        # Store cell
        uc_cell = atoms[n_uc].position[d] - atoms[0].position[d]
        # Expand Atoms and unit cell
        if 0 in sides:
            atoms = uc_ldl + atoms
            atoms.cell[d, d] += uc_cell
        if 1 in sides:
            atoms = atoms + uc_ldr
            atoms.cell[d, d] += uc_cell

        atoms.center()
        order_atoms(atoms, direction)
    for name, a in orig.arrays.items():
        orig.arrays[name] = atoms.arrays[name]
    orig.cell = atoms.cell


def order_coordinate(arr, axes=None):
    map = np.arange(len(arr))
    if axes is None:
        dims = arr.shape[-1]
        # Reverse so first axis is the most important
        axes = range(dims - 1, -1, -1)
    else:
        # Reverse so first axis is the most important
        axes = axes[::-1]
    for i in axes:  # range(dims-1,-1,-1):
        order = arr[map][:, i].argsort(kind="mergesort")
        map = map[order]
    return map


def order_atoms(atoms, direction="x"):

    if direction is "x":
        axes = [0, 1, 2]
    elif direction is "y":
        axes = [1, 0, 2]
    elif direction is "z":
        axes = [2, 0, 1]
    else:
        raise NotImplementedError("Valid directions are: 'x' and 'y'")

    map = order_coordinate(np.round(atoms.positions, 1), axes)

    for name, a in atoms.arrays.items():
        atoms.arrays[name] = a[map]
