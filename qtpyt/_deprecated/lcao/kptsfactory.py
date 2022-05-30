import numpy as np
from .principallayer import PrincipalLayer
from qtpyt.selfenergy import LeadSelfEnergy
from qtpyt.greenfunction import RecursiveGF

class KSelfenergyFactory(PrincipalLayer):

    def __init__(self, calc, direction='x', id=0, eta=1e-5, nbf_m=None):

        super().__init__(calc, direction)

        self.id = id
        self.eta = eta
        self.nbf_m = nbf_m

    def initialize(self, H_kMM=None, S_kMM=None):

        super().initialize(H_kMM, S_kMM)

        self.remove_pbc(self.H_kij)
        self.remove_pbc(self.S_kij)

        # Right lead
        if self.id == 1:
            self.H_kij = self.H_kji
            self.S_kij = self.S_kji

        nbf_i = self.H_kii.shape[0]
        if self.nbf_m is None:
            nbf_m = bfs_i
        self.nbf_i = nbf_i
        self.nbf_m = nbf_m

        # Selfenergies
        self.selfenergies = []
        for h_ii, s_ii, h_ij, s_ij in zip(self.H_kii,self.S_kii,
                                          self.H_kij,self.S_kij):

            # Coupling to central region
            dtype = self.h_ij.dtype
            self.h_im  = np.zeros((nbf_i,nbf_m),dtype=dtype)
            self.s_im  = np.zeros((nbf_i,nbf_m),dtype=dtype)

            if self.id == 0:
                self.h_im[:nbf_i, :nbf_i] = self.h_ij
                self.s_im[:nbf_i, :nbf_i] = self.s_ij

            elif self.id == 1:
                self.h_im[-nbf_i:, -nbf_i:] = self.h_ij
                self.s_im[-nbf_i:, -nbf_i:] = self.s_ij

            self.selfenergies.append(LeadSelfEnergy((h_ii,s_ii),
                                                    (h_ij,s_ij),
                                                    (h_ij,s_ij),
                                                    eta=self.eta))


class KGreenFunctionFactory(PrincipalLayer):

    def __init__(self, calc, direction='x', selfenergies=[], **kwargs):

        super().__init__(calc, direction)

        self.selfenergies_ki = selfenergies
    ############## Copied from RGF #####################
        self.parameters = {'eta': 1e-5,
                           'calc': None,
                           'pl1': None,
                           'pl2': None,
                           'cutoff': cutoff,
                           'align_bf': None,
                           'H': None,
                           'S': None}

        self.initialized = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key in ['eta','calc', 'pl1', 'pl2',
                       'cutoff', 'align_bf','H','S']:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError('%r not a vaild keyword' % key)

        self.parameters.update(kwargs)
    ####################################################

    def initialize(self, H_kMM=None, S_kMM=None):

        # Warning!! super() also initialize H_ij, S_ij
        super().initialize(H_kMM, S_kMM)

        self.remove_pbc(self.H_kii)
        self.remove_pbc(self.S_kii)

        # Right lead
        self.H_kij = None
        self.S_kij = None

        # Selfenergies
        self.greenfunctions = []
        for h_ii, s_ii, selfenergies_i in zip(self.H_kii,self.S_kii,
                                              self.sefenergies_ki):

            greenfunction = RecursiveGF(h_ii, s_ii,
                                        selfenergies_i,
                                        **self.parameters))

            greenfunction.initialize()
            self.greenfunctions.append(greenfunction)

    def remove_pbc(self, A_kMM, eps=-1e-3):

        # atoms of principal layer
        atoms = self.calc.atoms
        atoms.set_calculator(self.calc)

        # Transport direction
        p_dir = self.get_directions()[0]

        L = atoms.cell[p_dir, p_dir]

        centers_ic = get_bf_centers(atoms)
        cutoff = 0.5 * L - eps
        # Coordinates of central unit cell i (along transport)
        centers_p_i = centers_ic[:, p_dir]
        # Distance between j atoms and i atoms
        dist_p_ji = np.abs(centers_p_i[:, None] - centers_p_i[None, :])
        # Mask j atoms farther than cutoff
        mask_ji = (dist_p_ji > cutoff).astype(A_kMM.dtype)

        A_kMM *= mask_ji[None, :]


def get_transmission(greenfunctions, energies, T_e=None):

    p_dir, t_dirs = self.get_directions()

    if T_e is None:
        T_e = np.zeros(len(energies))

    for greenfunction in self.greenfunctions:
        T_e += greenfunction.get_transmission(energies)

    T_e /= np.prod(self.Nk_c[t_dirs])

    return T_e
