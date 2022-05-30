import numpy as np
from functools import reduce

from .tools import (
    lowdin_rotation,
    subdiagonalize,
    rotate_matrix,
    get_subspace,
    subdiagonalize_atoms,
    get_U1,
    get_U2,
)


class CoupledHamiltonian:
    def __init__(self, H, S, selfenergies=[]):
        self.H = H
        self.S = S
        self.selfenergies = selfenergies
        self.U = []

    def align_bf(self, align_bf):
        h_mm = self.H
        s_mm = self.S
        h1_ii = self.selfenergies[0].h_ii
        if align_bf is not None:
            diff = (
                h_mm[align_bf, align_bf] - np.real(h1_ii[align_bf, align_bf])
            ) / s_mm[align_bf, align_bf]
            # print('# Aligning scat. H to left lead H. diff=', diff)
            h_mm -= diff * s_mm
        return diff

    def apply_rotation(self, U):
        h_mm = self.H
        s_mm = self.S
        h_mm[:] = rotate_matrix(h_mm, U)
        s_mm[:] = rotate_matrix(s_mm, U)
        # Rotate coupling between lead and central region
        for indices, selfenergy in self.selfenergies:
            selfenergy.h_im[:] = np.dot(selfenergy.h_im, U[indices])
            selfenergy.s_im[:] = np.dot(selfenergy.s_im, U[indices])
        self.U.append(U)

    def diagonalize(self):
        nbf = len(self.H)
        self.subdiagonalize(range(nbf))

    def subdiagonalize(self, bfs):
        U, eig = subdiagonalize(self.H, self.S, bfs)
        self.apply_rotation(U)

    def take(self, bfs, apply=False):
        nbf = len(self.H)
        U = np.eye(nbf).take(bfs, 1)
        self.H = rotate_matrix(self.H, U)
        self.S = rotate_matrix(self.S, U)
        for indices, selfenergy in self.selfenergies:
            selfenergy.h_im = np.dot(selfenergy.h_im, U[indices])
            selfenergy.s_im = np.dot(selfenergy.s_im, U[indices])
        self.U.append(U)

    def lowdin_rotation(self, bfs=None):
        U = lowdin_rotation(self.H, self.S, bfs)
        self.apply_rotation(U)

    def subdiagonalize_atoms(self, basis, a=None):
        U, eig = subdiagonalize_atoms(basis, self.H, self.S, a)
        self.apply_rotation(U)

    def partition_orthogonal(self, bfs_m, bfs_i):
        U1, bfs_m, bfs_i = get_U1(bfs_m, bfs_i)
        self.apply_rotation(U1)
        U2 = get_U2(self.S, bfs_m, bfs_i)
        self.apply_rotation(U2)

    def get_rotation(self):
        return reduce(lambda a, b: np.dot(a, b), self.U)
