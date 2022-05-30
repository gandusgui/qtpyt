from gpaw import *
from .analysis_tools import *

"""The class deals with all kinds of wavefunction plotting
   like plotting KS orbitals
   subdiagonalizing and plotting fragment orbitals...
"""


class Plotter:
    def __init__(self, calc, k=0, sp=0):
        self.calc = calc
        self.atoms = calc.atoms
        self.atoms.calc = calc
        self.kpt = k
        self.sp = sp

    def plot_molecular_MOs(
        self, lowor_en, upor_en
    ):  # lowor_en and upor_en are the lower and upper energy of orbitals of interest with the fermi energy as reference
        H, S, atom_mol, nhomo = get_molecular_hs(
            self.calc, self.atoms, self.kpt, self.sp
        )
        ev, V = diagonalize_h(H, S)
        write_eigenvalues_into_file(ev, self.kpt, self.sp)

        needed_evlist = []
        for i in range(len(ev)):
            if float(ev[i].real) > lowor_en and float(ev[i].real) < upor_en:
                needed_evlist.append(ev[i].real)

        counterlow, counterhigh = -1, 1
        for i in range(len(needed_evlist)):
            if needed_evlist[i] < 0:
                counterlow += 1
            if needed_evlist[i] > 0:
                counterhigh += 1

        n_plot_start = []
        for i in range(-counterlow, counterhigh):
            n_plot_start.append(i)

        n_plot = np.array(n_plot_start) + nhomo
        n_plot = n_plot.astype(int)  # recast each element of n_plot to integer

        print_interesting_quantities(ev, needed_evlist)
        self.plot_mol_wavefunctions(atom_mol, n_plot, V, ev, self.sp)

    def plot_mol_wavefunctions(self, atm_range, n_plot, V, ev, spin=0):
        nao = self.calc.wfs.setups.nao
        # nk = len(self.calc.wfs.ibzk_kc)
        nk = len(self.calc.wfs.kd.ibzk_kc)  # number of kpoints
        for ii in n_plot:
            p1 = V[:, ii]
            n1, cc = 0, 0
            psi = np.zeros([nk, nao])  # initialize psi matrix
            for i in range(len(self.atoms)):
                if i in atm_range:  # if i is atom list
                    no = self.calc.wfs.setups[i].nao  # get bfs on atom i
                    n2 = n1 + no  # max wfs in psi
                    psi[0, n1:n2] = p1[
                        cc : cc + no
                    ]  # add coefficients of molecular subspace to list
                    cc += no  # set start for next loop
                n1 += self.calc.wfs.setups[i].nao  # min wfs in psi (for next step)
            psi = psi.reshape(1, -1)  # reshape psi to get a vector
            psi_g = self.calc.wfs.gd.zeros(nk, dtype=self.calc.wfs.dtype)  # initialize
            ss = psi_g.shape  # get dimensions of psi_g
            psi_g = psi_g.reshape(1, -1)  # reshape psi_g to get a vector
            self.calc.wfs.basis_functions.lcao_to_grid(psi, psi_g, q=0)
            psi_g = psi_g.reshape(ss)  # resreo original shape

            # write output
            write(
                "orb_%1.4f_spin_%i.cube" % (ev[ii].real, spin),
                self.atoms,
                data=psi_g[0],
            )
            print("Cube files generated")
            print(".....done!")

    def plot_orbitals(self, plotorbindex, spin=0):
        for band in plotorbindex:
            wf = self.calc.get_pseudo_wave_function(band=band, spin=spin)
            fname = "orb_" + "%d" % (band) + ".cube"
            print("writing wf", band, "to file", fname)
            write(fname, self.atoms, data=wf)
