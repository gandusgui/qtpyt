import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist
from ase.io import write


def get_external_internal(atoms, symbols=None):

    # Atoms to include in external and internal indices
    if symbols is None:
        symbols = list(set(atoms.symbols))
        try:
            symbols.pop(symbols.index("H"))
        except ValueError as e:
            print(e)
    #
    symbols = list(symbols)

    # Define list of neighbors
    cov_radii = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(cov_radii, bothways=True, self_interaction=False)
    nl.update(atoms)

    external_i = []
    internal_i = []
    for a in atoms:
        if a.symbol not in symbols:
            continue
        nlist = nl.get_neighbors(a.index)[0]
        n_is_H = [True if atoms[n0].symbol == "H" else False for n0 in nlist]
        if any(n_is_H):
            external_i.append(a.index)
        else:
            internal_i.append(a.index)

    return external_i, internal_i


def plot_mol_wavefunctions(calc, atoms_i, ao_j, v_jj, e_j, spin=0):

    atoms = calc.atoms
    nao = calc.wfs.setups.nao
    nk = len(calc.wfs.kd.ibzk_kc)  # number of kpoints

    for ii in ao_j:
        p1 = v_jj[:, ii]
        n1, cc = 0, 0
        psi = np.zeros([nk, nao])  # initialize psi matrix
        for i in range(len(atoms)):
            if i in atoms_i:  # if i is atom list
                no = calc.wfs.setups[i].nao  # get bfs on atom i
                n2 = n1 + no  # max wfs in psi
                psi[0, n1:n2] = p1[
                    cc : cc + no
                ]  # add coefficients of molecular subspace to list
                cc += no  # set start for next loop
            n1 += calc.wfs.setups[i].nao  # min wfs in psi (for next step)
        psi = psi.reshape(1, -1)  # reshape psi to get a vector
        psi_g = calc.wfs.gd.zeros(nk, dtype=calc.wfs.dtype)  # initialize
        ss = psi_g.shape  # get dimensions of psi_g
        psi_g = psi_g.reshape(1, -1)  # reshape psi_g to get a vector
        calc.wfs.basis_functions.lcao_to_grid(psi, psi_g, q=0)
        psi_g = psi_g.reshape(ss)  # resreo original shape
        write("orb_%1.4f_spin_%i.cube" % (e_j[ii].real, spin), atoms, data=psi_g[0])
        print("Cube files generated")
        print(".....done!")


def display_blocks(h_mm, lines, precision=0.1, **kwargs):
    """Plot helper for subdiagonalized and ordered matrices."""
    from matplotlib import pyplot as plt

    size = h_mm.shape[0]
    lines = np.array(lines)
    figsize = kwargs.pop("figsize", None) or (10, 10)
    plt.figure(figsize=figsize, **kwargs)
    plt.spy(h_mm, precision=precision)
    plt.hlines(lines - 0.5, -0.5, size - 0.5, colors="r")
    plt.vlines(lines - 0.5, -0.5, size - 0.5, colors="r")


from ase import Atoms
from gpaw import GPAW
import PIL


def single_atom_calculation(element, basis="szp(dzp)", xc="PBE"):
    atoms = Atoms(element, [(0, 0, 0)])
    atoms.set_cell([7, 7, 7])
    atoms.center()
    calc = GPAW(mode="lcao", txt=None, basis=basis, xc=xc, maxiter=1)
    atoms.set_calculator(calc)
    try:
        atoms.get_potential_energy()
    except:
        pass
    return calc


def get_orbitals(calc, C_wM, indices=None, q=-1):
    """Get orbitals w from M ao coefficients to G grid"""
    if indices is None:
        indices = range(C_wM.shape[0])
    Ni = len(indices)
    w_wG = calc.wfs.gd.zeros(Ni, dtype=calc.wfs.dtype)
    calc.wfs.basis_functions.lcao_to_grid(C_wM, w_wG, q=q)
    return w_wG


from scipy.ndimage import rotate


def normalizer(psi, norm="standard"):
    """Valid norm are 'standard' and None."""
    if norm == "standard":

        def normalize(psi):
            return (psi - psi.mean()) / psi.std()

    else:
        raise NotImplementedError()
    return normalize


def contourer(contour, values):
    """Valid contours are 'constant' and 'std'."""
    from matplotlib import pyplot as plt

    values = np.array(values, ndmin=1)
    values = np.hstack((-values[::-1], values))
    colors = plt.cm.gray(np.linspace(0, 1, values.size))
    if contour == "std":

        def iter_contours(psi):
            std = psi[np.nonzero(psi)].std()
            for color, factor in zip(colors, values):
                yield factor * std, tuple(color[:-1])

    # Plot values contours
    elif contour == "constant":

        def iter_contours(psi):
            for color, factor in zip(colors, values):
                yield factor, tuple(color[:-1])

    else:
        raise NotImplementedError()
    return iter_contours


class OrbitalFactory:
    def __init__(self, rotation=None, norm=False, contour="std", values=4):
        self.rotation = rotation
        self.norm = norm
        self.iter_contours = contourer(contour, values)

    def rotate(self, psi):
        if self.rotation is None:
            return psi
        angle, axes = self.rotation
        return rotate(psi, angle, axes)

    def normalize(self, psi):
        if self.norm is False:
            return psi
        return normalize(psi, self.normalize)

    def prepare(self, psi):
        return self.rotate(self.normalize(psi))


def img_array_from_orbitals(
    mlab, w_wG, orb_i=None, rotation=None, norm=False, contour="std", values=4
):
    """rotation := (angle, axes=1,0)
       contours := 'std' or 'constant'
       values :=
                  if contours is 'std' return +- values * std
                  if contours is 'constant' return +- values
    """
    # Preprocess orbitals
    preprocessor = OrbitalFactory(
        rotation=rotation, norm=norm, contour=contour, values=values
    )
    indices = orb_i or range(w_wG.shape[0])
    images = []
    for i in indices:
        psi = preprocessor.prepare(w_wG[i])
        for isovalue, color in preprocessor.iter_contours(psi):
            contour = mlab.contour3d(psi, contours=[isovalue], color=color)
        mlab.outline(contour)
        mlab.axes(contour)
        img = mlab.screenshot()
        images.append(img)
        mlab.close()
    return np.array(images)


def img_array_from_coeff(mlab, calc, v_amm, atom_i=None, **kwargs):
    """Get screenshot orbitals v_amm[a,:,w] for atoms(a) orbitals(w)"""
    indices = atom_i or range(v_amm.shape[0])
    images = []
    # mlab.figure()
    for a in indices:
        c_mm = v_amm[a]
        w_wG = get_orbitals(calc, c_mm.T)
        images.append(img_array_from_orbitals(mlab, w_wG, **kwargs))
    return np.array(images)


# raddi and colors of atoms
from ase.data.colors import jmol_colors
from ase.data import covalent_radii


def plot_atoms(mlab, atoms):
    """Plot atoms by color and radii"""
    positions = atoms.positions
    for num in set(atoms.numbers):
        color = tuple(jmol_colors[num])
        radii = covalent_radii[num]
        points = positions[atoms.numbers == num]
        mlab.points3d(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            scale_factor=radii * 0.5,
            color=color,
        )
