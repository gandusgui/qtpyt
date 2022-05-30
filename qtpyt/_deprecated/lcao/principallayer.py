import numpy as np
from ase import units

# from gpaw.symmetry import Symmetry
from ase.dft.kpoints import monkhorst_pack
from qtpyt.block import get_toeplitz

# Density integral
from qtpyt.continued_fraction import integrate_pdos
from qtpyt.selfenergy import LeadSelfEnergy
from qtpyt.tk_gpaw import *

# from qtpyt.tk_gpaw import *get_bf_centers, get_bfs_indices, \
#                              flatten, initialize_calculator
from qtpyt.tools import dagger, get_subspace, rotate_matrix
from scipy import linalg as la
from scipy.spatial.distance import cdist

from .tk_lcao import *

# from qtpyt.tk_gpaw import sum_bf_atom


class PrincipalLayer:
    def __init__(self, calc, direction="x"):

        # Initialize calculator
        initialize_calculator(calc)

        self.calc = calc
        self.direction = direction

        kd = self.calc.wfs.kd
        self.ibzk_kc = kd.ibzk_kc
        self.Nk_c = kd.N_c
        self.offset_c = kd.offset_c
        self.fermi = self.calc.get_fermi_level()

        self.update()

    def initialize(
        self,
        H_kMM=None,
        S_kMM=None,
        H_kii=None,
        S_kii=None,
        H_kij=None,
        S_kij=None,
        ibzk_t_kc=None,
    ):

        if H_kMM is None:
            H_kMM, S_kMM = h_and_s(self.calc)
            # Convert in Hartree units
            H_kMM *= units.Hartree
            # Align Fermi Level
            self.align_fermi(H_kMM, S_kMM)

        # Transport direction
        # self.update()
        p_dir, t_dirs = self.get_directions()

        R_c = [0, 0, 0]
        self.H_kii = self.bloch_to_real_space_p(H_kMM, R_c, ibzk_t_kc, H_kii)
        self.S_kii = self.bloch_to_real_space_p(S_kMM, R_c, ibzk_t_kc, S_kii)

        R_c[p_dir] = 1
        self.H_kij = self.bloch_to_real_space_p(H_kMM, R_c, ibzk_t_kc, H_kij)
        self.S_kij = self.bloch_to_real_space_p(S_kMM, R_c, ibzk_t_kc, S_kij)

    @property
    def H_kji(self):
        return self.H_kij.swapaxes(1, 2).conj()

    @property
    def S_kji(self):
        return self.S_kij.swapaxes(1, 2).conj()

    def get_directions(self):
        # Define transport and transverse directions
        p_dir = "xyz".index(self.direction)
        t_dirs = np.delete([0, 1, 2], p_dir)
        return p_dir, t_dirs

    def set_num_cells(self, Nr_c=None):
        # Number of realspace cells. Defaults to k-points transverse
        if Nr_c is None:
            t_dirs = self.get_directions()[1]
            Nr_c = self.Nk_c[t_dirs]

        t_dirs = self.get_directions()[1]

        # Lattice vectors
        R_cN = np.indices(Nr_c).reshape(2, -1)
        N_c = np.array(Nr_c)[:, np.newaxis]
        R_cN += N_c // 2
        R_cN %= N_c
        R_cN -= N_c // 2
        self.R_cN = R_cN
        self.Nr_c = Nr_c

    def align_fermi(self, H_kMM, S_kMM):

        H_kMM -= self.fermi * S_kMM
        # self.fermi = 0.

    def update(self):

        # Define transport and transverse directions
        p_dir, t_dirs = self.get_directions()

        # K-points in the transport direction
        offset_p_c = np.zeros((3,))
        offset_p_c[p_dir] = self.offset_c[p_dir]
        bzk_p_kc = monkhorst_pack((self.Nk_c[p_dir], 1, 1)) + offset_p_c

        # K-points in the transverse directions
        offset_t_c = np.zeros((3,))
        offset_t_c[: len(t_dirs)] = self.offset_c[t_dirs]
        bzk_t_kc = monkhorst_pack(tuple(self.Nk_c[t_dirs]) + (1,)) + offset_t_c

        # Time-reversal symmetry
        ibzk_p_kc, bzk2ibzk_p_k = symm_reduce(bzk_p_kc)
        ibzk_t_kc = symm_reduce(bzk_t_kc)[0]

        # Take dimensions
        self.bzk_p_k = bzk_p_kc[:, 0]
        # self.ibzk_t_kc = ibzk_t_kc[:, :2]
        self.ibzk_t_kc = bzk_t_kc[:, :2]
        self.bzk_t_kc = bzk_t_kc[:, :2]

        # Update number of cells
        self.set_num_cells()

    def bloch_to_real_space_p(self, A_kMM, R_c, ibzk_t_kc, A_NMM):

        # A_NMM Output matrix evaluated at (each>?1) transverse k-point.
        if ibzk_t_kc is None:
            ibzk_t_kc = self.ibzk_t_kc
            A_NMM = np.zeros((len(ibzk_t_kc), *A_kMM.shape[1:]), dtype=A_kMM.dtype)

        ibz2bz = cdist(ibzk_t_kc, self.ibzk_t_kc).argmin(1)
        # Transport and transverse k-points
        p_dir, t_dirs = self.get_directions()

        # For each transverse k-point, Fourier transform in trasport direction
        for j, kt_c in zip(ibz2bz, ibzk_t_kc):
            # Transport k-point that is fourier transformed in transport direction.
            k_kc = np.zeros((len(self.bzk_p_k), 3))
            k_kc[:, p_dir] = self.bzk_p_k
            # Fix tranverse directions.
            k_kc[:, t_dirs] = kt_c
            # Matrices at transverse k-point
            A_xMM = []
            for k_c in k_kc:
                try:
                    residue_k = np.linalg.norm(k_c - self.ibzk_kc, axis=1)
                    kc2ibzk = np.where(np.abs(residue_k) < 1e-7)[0][0]
                    A_xMM.append(A_kMM[kc2ibzk])
                except IndexError:
                    # e.g. k_c = (-0.4,0.4) -> -k_c = (0.4,-0.4)
                    # A_kMM[k_c] = A_kMM[-k_c].conj()
                    # Since we require A_kMM[k_c], we can take A_kMM[-k_c].conj(),
                    # which is equivalente and present in lcao calculation.
                    try:
                        residue_k = np.linalg.norm((-k_c) - self.ibzk_kc, axis=1)
                        kc2ibzk = np.where(np.abs(residue_k) < 1e-7)[0][0]
                        A_xMM.append(A_kMM[kc2ibzk].conj())
                    except IndexError:
                        raise IndexError("k-point {} not found".format(k_c))
            fourier_sum(np.array(A_xMM), k_kc, R_c, A_NMM[j])

            A_NMM[j] /= self.Nk_c[p_dir]

        return A_NMM

    def bloch_to_real_space_t(self, A_kMM, R_Nc=None, A_NMM=None):

        # Parameters
        shape = A_kMM.shape
        if R_Nc is None:
            R_Nc = self.R_cN.T
            # Output matrix evaluated at (each>?1) transverse k-point.
            A_NMM = np.zeros((len(R_Nc), *shape[1:]), dtype=A_kMM.dtype)

        # Transport and transverse k-points
        p_dir, t_dirs = self.get_directions()

        # Detect gamma point
        gamma_point = True
        try:
            gamma = np.where(np.linalg.norm(self.ibzk_kc, axis=1) < 1e-7)[0][0]
        except IndexError:
            gamma_point = False

        # For each real space point, Fourier transform in trasverse direction
        for i, R_c in enumerate(R_Nc):
            # Evaualte fourier sum in tranverse direction
            fourier_sum(A_kMM, self.ibzk_t_kc, R_c, A_NMM[i])

            if len(self.ibzk_t_kc) < len(self.bzk_t_kc):
                # Add conjugate and subtract double counted Gamma (in transport component)
                if gamma_point:
                    A0_MM = A_kMM[gamma]
                    A_MM += A_MM.conj() - A0_MM
                else:
                    A_MM += A_MM.conj()

            A_NMM[i] /= np.prod(self.Nk_c[t_dirs])

        return A_NMM

    def remove_pbc(self, A_kMM, eps=-1e-3):

        # atoms of principal layer
        atoms = self.calc.atoms
        atoms.set_calculator(self.calc)

        # Transport direction
        p_dir = self.get_directions()[0]

        L = atoms.cell[p_dir, p_dir]

        centers_ic = get_bf_centers(atoms)
        cutoff = L - eps
        # Coordinates of central unit cell i (along transport)
        centers_p_i = centers_ic[:, p_dir]
        # Coordinates of neighbooring unit cell j
        centers_p_j = centers_p_i + L
        # Distance between j atoms and i atoms
        dist_p_ji = np.abs(centers_p_j[:, None] - centers_p_i[None, :])
        # Mask j atoms farther than L
        mask_ji = (dist_p_ji > cutoff).astype(A_kMM.dtype)

        A_kMM *= mask_ji[None, :]

    def bloch_to_real_space_block(self, A_kMM, A_NMM=None):

        if A_NMM is None:
            A_NMM = self.bloch_to_real_space_t(A_kMM)
        else:
            # Fourier transform in transverse directions
            self.bloch_to_real_space_t(A_kMM, self.R_cN.T, A_NMM)

        # The new dimension (x) equals M \times the number of rows
        # A_xx = get_toeplitz(rows=A_NMM)
        A_xx = build_surface(self.Nr_c, A_NMM)

        return A_xx

    def set_order(self, atoms):

        # Transverse directions and # of k-point
        p_dir, t_dirs = self.get_directions()
        # N_c = self.Nk_c.copy()
        # N_c[p_dir] = 1
        N_c = np.ones(3, dtype=int)
        N_c[t_dirs] = self.Nr_c

        # number of repeted unitcells and atoms
        M = np.prod(N_c)
        n = len(self.calc.atoms)
        na = M * n

        # Reverse for yes-pbc to no-pbc
        R_Nc = np.indices(N_c).reshape(3, -1).T

        # Positions in repeted structure
        pos_ac = np.tile(self.calc.atoms.positions, (M, 1))
        i0 = 0
        for R_c in R_Nc:
            i1 = i0 + n
            # add unit cell
            pos_ac[i0:i1] += np.dot(R_c, self.calc.atoms.cell)
            # next unit cell
            i0 = i1

        # index of repeted structure (a) in input atoms (x)
        a2x_a = np.argmin(
            np.linalg.norm(pos_ac[:, None] - atoms.positions[None, :], axis=2), axis=0
        )

        # Basis function indices in repeted structure
        bfs_a = get_bfs_indices(self.calc, range(n), "append") * M
        nao = self.calc.setups.nao
        for a in range(len(bfs_a)):
            count = (a // n) * nao
            bfs_a[a] = [bf + count for bf in bfs_a[a]]

        # index of bfs (a) in (x)
        bfa2bfx_i = []
        for a in a2x_a:
            bfa2bfx_i.extend(bfs_a[a])

        # Store
        self.bfa2bfx_i = bfa2bfx_i
        self.a2x_a = a2x_a

        # Decorate bloch_to_real_space_block to order return matrix
        self.bloch_to_real_space_block = self._order_return(
            self.bloch_to_real_space_block
        )

    def _order_return(self, func):

        # Hack to order matrix according to (x) bfs.
        def _order(func, *args, **kwargs):
            def _inner(*args, **kwargs):
                # Get matrix
                M = _f(*args, **kwargs)
                # Order
                return get_subspace(M, self.bfa2bfx_i)

            _f = func
            return _inner

        return _order(func)

    def band_structure(self, path_kc, blochstates=False):
        """Calculate dispersion along bandpath for each transverse k-points"""
        from .tk_lcao import get_partial_tightbindings

        if not hasattr(self, "tbs_t_k"):
            self.tbs_t_k = get_partial_tightbindings(self)
        # Number of transverse k-points
        Ntk = len(self.ibzk_t_kc)
        # Number of bandpath k-points
        Nk = len(path_kc)
        # Number of bands
        nao = self.H_kij[0].shape[-1]
        # Initialize outputs
        eps_t_kkn = np.zeros((Ntk, Nk, nao))
        if blochstates:
            psi_t_kknM = np.zeros((Ntk, Nk, nao, nao), dtype=complex)
        # Iterate over transverse tightbindings
        for kt, tb in enumerate(self.tbs_t_k):
            if blochstates:
                eps_t_kkn[kt], psi_t_kknM[kt] = tb.band_structure(path_kc, blochstates)
            else:
                eps_t_kkn[kt] = tb.band_structure(path_kc, blochstates)
        #
        if blochstates:
            return eps_t_kkn, psi_t_kknM
        else:
            return eps_t_kkn


class PrincipalSelfEnergy(PrincipalLayer):
    def __init__(self, calc, direction="x", scatt=None, id=0, **kwargs):

        super().__init__(calc, direction)

        self.scatt = scatt
        # self.eta = 1e-5
        self.energy = None
        self.bias = 0.0
        self.id = id

        self.parameters = {
            "eta": 1e-5,
            "bias": 0,
            # number of realspace cells. None defaults to Nk_c[t_dirs]
            "Nr_c": None,
            "nbf_m": None,
        }

        self.initialized = False
        self.set(**kwargs)

    def set(self, **kwargs):
        for key in kwargs:
            if key in ["scatt", "id"]:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError("%r not a vaild keyword" % key)

        self.parameters.update(kwargs)

    def set_bias(self, bias):
        self.bias = bias
        for selfenergy in self.selfenergies:
            selfenergy.set_bias(bias)

    def initialize(self, H_kMM=None, S_kMM=None):

        ibzk_t_kc = self.ibzk_t_kc
        dtype = H_kMM.dtype

        shape_kMM = (len(ibzk_t_kc),) + H_kMM.shape[1:]
        sz_kMM = int(np.prod(shape_kMM))

        self.H_kii = np.empty(shape_kMM, dtype=dtype)
        self.S_kii = np.empty(shape_kMM, dtype=dtype)
        self.H_kij = np.empty(shape_kMM, dtype=dtype)
        self.S_kij = np.empty(shape_kMM, dtype=dtype)
        self.G_kMM = np.empty(shape_kMM, dtype=dtype)

        super().initialize(
            H_kMM, S_kMM, self.H_kii, self.S_kii, self.H_kij, self.S_kij, ibzk_t_kc
        )

        p = self.parameters

        self.eta = p["eta"]
        Nr_c = p["Nr_c"]
        nbf_m = p["nbf_m"]

        if Nr_c is not None:
            assert len(Nr_c) == 2, "Invalid length of Nr_c. It must be 2."
            self.set_num_cells(Nr_c)

        # Real space Green's function
        shape_rMM = (int(np.prod(self.Nr_c)),) + H_kMM.shape[1:]
        sz_rMM = int(np.prod(shape_rMM))

        self.G_NMM = np.empty(shape_rMM, dtype=dtype)

        self.remove_pbc(self.H_kij)
        self.remove_pbc(self.S_kij)

        # Right lead
        if self.id == 1:
            self.H_kij = self.H_kji
            self.S_kij = self.S_kji

        # Selfenergies
        self.selfenergies = [
            LeadSelfEnergy((h_ii, s_ii), (h_ij, s_ij), (h_ij, s_ij), eta=self.eta)
            for h_ii, s_ii, h_ij, s_ij in zip(
                self.H_kii, self.S_kii, self.H_kij, self.S_kij
            )
        ]

        # Number of basis functions leads (i) and scattering (m) regions
        nbf_i = self.calc.setups.nao * len(self.R_cN.T)
        self.natoms = len(self.calc.atoms) * len(self.R_cN.T)
        if self.scatt:
            self.set_order(self.scatt[: self.natoms])

        if nbf_m is None:
            nbf_m = nbf_i  # * self.scatt.setups.nao

        self.h_ii = self.bloch_to_real_space_block(self.H_kii)
        self.s_ii = self.bloch_to_real_space_block(self.S_kii)
        self.h_ij = self.bloch_to_real_space_block(self.H_kij)
        self.s_ij = self.bloch_to_real_space_block(self.S_kij)

        # Coupling to central region
        self.h_im = np.zeros((nbf_i, nbf_m), dtype=dtype)
        self.s_im = np.zeros((nbf_i, nbf_m), dtype=dtype)

        if self.id == 0:
            self.h_im[:nbf_i, :nbf_i] = self.h_ij
            self.s_im[:nbf_i, :nbf_i] = self.s_ij

        elif self.id == 1:
            self.h_im[-nbf_i:, -nbf_i:] = self.h_ij
            self.s_im[-nbf_i:, -nbf_i:] = self.s_ij

        self.Sigma = np.zeros((nbf_m, nbf_m), dtype=dtype)

        self.nbf_m = nbf_m
        self.nbf_i = nbf_i

        # self.initialized = True

    def retarded(self, energy):
        """Return self-energy (sigma) evaluated at specified energy."""
        if energy != self.energy:
            self.energy = energy
            z = energy - self.bias + self.eta * 1.0j
            tau_im = z * self.s_im - self.h_im
            G = self.get_G(energy)
            tau_mi = z * self.s_im.T.conj() - self.h_im.T.conj()
            self.Sigma[:] = tau_mi.dot(G).dot(tau_im)

        return self.Sigma

    def get_lambda(self, energy):
        """Return the lambda (aka Gamma) defined by i(S-S^d).

        Here S is the retarded selfenergy, and d denotes the hermitian
        conjugate.
        """
        Sigma = self.retarded(energy)
        return 1.0j * (Sigma - Sigma.T.conj())

    def get_G(self, energy):

        from .cython.recursive_sgf import get_G

        # Green's functions at thanverse k-points
        G_kMM = self.G_kMM
        func = LeadSelfEnergy.get_Ginv

        ### Compute self-energies at transverse k-points

        # Dask version
        # from dask import delayed, compute
        # func1 = [delayed(func)(selfenergy, energy)
        #             for selfenergy in self.selfenergies]
        # func2 = [delayed(la.inv)(f1, overwrite_a=True, check_finite=True)
        #             for f1 in func1]
        # func3 = delayed(np.asarray)(func2)
        # G_kMM = compute(func3, schedule='processes')[0]

        # for i, selfenergy in enumerate(self.selfenergies):
        #    G_kMM[i] = la.inv(func(selfenergy, energy),
        #                       overwrite_a=True, check_finite=False)

        # Cython version
        get_G(
            G_kMM, self.H_kii, self.S_kii, self.H_kij, self.S_kij, energy, eta=self.eta
        )

        # embed()

        # Compute quantities in realspace
        G = self.bloch_to_real_space_block(G_kMM, A_NMM=self.G_NMM)
        return G

    def apply_overlap(self, energy, trace=False, diag=False):
        """Apply retarded Green function to S."""
        GS = self.get_G(energy) @ self.S
        if trace:
            return np.trace(GS)
        if diag:
            return GS.diagonal()
        return GS

    def dos(self, energy):
        """Total density of states -1/pi Im(Tr(GS))"""
        if not hasattr(self, "S"):
            return -self.retarded(energy).imag.trace() / np.pi
        else:
            G = self.get_G(energy)
            S = self.S
            return -G.dot(S).imag.trace() / np.pi

    def pdos(self, energy):
        """Projected density of states -1/pi Im(SGS/S)"""
        if not hasattr(self, "S"):
            return -self.get_G(energy).imag.diagonal() / np.pi
        else:
            G = self.get_G(energy)
            S = self.S
            return -G.dot(S).imag.diagonal() / np.pi
            #
            # SGS = np.dot(S, G.dot(S))
            # return -(SGS.diagonal() / S.diagonal()).imag / np.pi

    def density(self, energies=None):
        scatt = self.scatt
        n_a = self.natoms
        if energies is None:
            pdos = integrate_pdos(self)
            rho = sum_bf_atom(scatt, pdos, n_a)
        else:
            energies = np.array(energies, ndmin=1)
            rho = np.zeros((energies, n_a))
            for e, energy in energies:
                rho[e] = sum_bf_atom(scalc, self.pdos(energy))
        return rho

    ####### CONVENIENT ALISES ########

    @property
    def H(self):
        return self.h_ii

    @property
    def S(self):
        return self.s_ii

    ######## MODIFIERS ################

    def lowdin_rotation(self, apply=True):
        # Lowding rotation at each k-point

        # number of k-point(s) and basis functions
        nkt, nbf = self.H_kii.shape[:2]

        # Indices of H_kii and H_kij subblocks in H_kmm
        index_kii = np.ix_(range(nkt), range(nbf), range(nbf))
        index_kij = np.ix_(range(nkt), range(nbf), range(nbf, 2 * nbf))

        # Construct bigger matrices
        H_kmm = np.block([[self.H_kii, self.H_kij], [self.H_kji, self.H_kii]])
        S_kmm = np.block([[self.S_kii, self.S_kij], [self.S_kji, self.S_kii]])

        # Lowdin transform at every k-point
        for h_mm, s_mm in zip(H_kmm, S_kmm):
            eig, rot_mm = np.linalg.eigh(s_mm)
            eig = np.abs(eig)
            rot_mm = np.dot(rot_mm / np.sqrt(eig), dagger(rot_mm))
            if apply:
                self.uptodate = False
                h_mm[:] = rotate_matrix(h_mm, rot_mm)  # rotate C region
                s_mm[:] = rotate_matrix(s_mm, rot_mm)

        # Update
        self.H_kii[:] = H_kmm[index_kii]
        self.S_kii[:] = S_kmm[index_kii]
        self.H_kij[:] = H_kmm[index_kij]
        self.S_kij[:] = S_kmm[index_kij]
