import numpy as np
from collections import namedtuple

fields = ["M_a", "nao_a", "nao", "atoms"]
Calc = namedtuple("calc", fields)


def initialize_calculator(calc):
    calc.atoms.set_calculator(calc)
    if calc.wfs.S_qMM is None:
        # Initialize calculator
        calc.wfs.set_positions(calc.spos_ac)


def make_calc(atoms, basis):
    M_a = np.zeros(len(atoms), dtype=int)
    nao_a = np.zeros(len(atoms), dtype=int)
    nao = 0
    # Init
    M_a[0] = 0
    nao_a[0] = basis[atoms[0].symbol]
    nao += nao_a[0]
    for a0 in range(1, len(atoms)):
        nao_a[a0] = basis[atoms[a0].symbol]
        M_a[a0] = M_a[a0 - 1] + nao_a[a0 - 1]
        nao += nao_a[a0]
    return Calc(M_a, nao_a, nao, atoms)


def get_info(calc, attributes):
    info = []
    gpaw = hasattr(calc, "wfs")
    # From gpaw calculator
    if gpaw:
        initialize_calculator(calc)
        natoms = len(calc.atoms)
        for attr in attributes:
            if attr in ["M_a", "nao"]:
                info.append(getattr(calc.setups, attr))
            elif attr in ["nao_a"]:
                info.append([calc.setups[a0].nao for a0 in range(natoms)])
            else:
                raise NotImplementedError("{}".format(attr))
    else:
        # From namedtuple
        for attr in attributes:
            info.append(getattr(calc, attr))
    return info


from .tk_gpaw import get_atom_indices, get_bfs_indices, get_bfs_atoms


class calcWRAP:
    def __init__(self, atoms, basis=None, default="dzp"):
        self.bfs_ai = get_bfs_atoms(
            atoms, basis=basis, default=default, method="append"
        )
        M_a, nao_a, nao = self.bfs2info(self.bfs_ai)
        self.nao_a = nao_a
        self.nao = nao
        self.M_a = M_a
        self.atoms = atoms
        # super().__init__(M_a, nao_a, nao, atoms)

    @staticmethod
    def bfs2info(bfs_ai):
        nao_a = np.array([len(bfs) for bfs in bfs_ai])
        M_a = [0] + np.cumsum(nao_a)[:-1].tolist()
        M_a = np.array(M_a)
        nao = sum(nao_a)
        return M_a, nao_a, nao

    def get_bfs_indices(self, a, method="extend"):
        return get_bfs_indices(self, a, method=method)

    def get_atom_indices(self, a):
        return get_atom_indices(self, a)
