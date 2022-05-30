#!/usr/bin/env python3

from gpaw import GPAW
from gpaw.lcao.tools import tri2full
from gpaw.lcao.tools import remove_pbc
from ase.units import Hartree
from ase.io import read
from qtpyt.solvers.tridiagonal import tridiagonalize, cutoff

from utilities import *


def get_h_and_s(calc, kpt=0):
    calc.atoms.set_calculator(calc)
    if calc.wfs.S_qMM is None:
        calc.wfs.set_positions(calc.spos_ac)
    h = calc.wfs.eigensolver.calculate_hamiltonian_matrix(
        hamiltonian=calc.hamiltonian, wfs=calc.wfs, kpt=calc.wfs.kpt_u[kpt]
    )
    s = calc.wfs.S_qMM[kpt]
    tri2full(h)
    tri2full(s)
    h *= Hartree
    h -= calc.get_fermi_level() * s
    remove_pbc(atoms=calc.atoms, h=h, s=s)
    return h, s


def laod_csr(filename, shape):
    from scipy.sparse import csr_matrix

    data = np.load(filename)
    mat = csr_matrix((data[:, 2], (data[:, 0] - 1, data[:, 1])), shape=shape)
    return mat.todense()


def basis_from_file(filename):
    dt = np.dtype([("elem", str), ("nao", int)])
    arr = np.loadtxt(filename, dtype=dt)
    d = dict(zip(arr["elem"], arr["nao"]))
    return d


if __name__ == "__main__":
    import pickle
    import sys
    from os.path import basename
    from scipy.sparse import csr_matrix

    fn_atoms = sys.argv[1]
    fn_basis = sys.argv[2]
    pl1 = int(sys.argv[3])
    pl2 = pl1 if len(sys.argv) == 4 else int(sys.argv[4])
    file_h = get_filename("*KS-SPIN*", assert_single=True)
    file_s = get_filename("*S-SPIN*", assert_single=True)
    h = load_csr(file_h, shape)
    s = load_csr(file_s, shape)
    atoms = read(fn_atoms)
    basis = basis_from_file(fn_basis)
    calc = make_calc(atoms, basis)
    hs_list_ii, hs_list_ij = tridiagonalize(calc, h, s, pl1, pl2, cutoff)
    pickle.dump(
        (hs_list_ii, hs_list_ij),
        open("hs_{}_lists.pckl".format(basename(fn_atoms).split(".")[0]), "wb"),
        2,
    )
