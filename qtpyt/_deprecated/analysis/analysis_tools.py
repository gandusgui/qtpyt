from ase import *
import numpy as np
import pickle
from gpaw import *
from gpaw.lcao.tools import *
import time
import sys
from ase.io import read, write

"""Write out a pickle file"""


def write_pickle_file(fname, input):
    ff = open(fname, "wb")
    pickle.dump((input), ff)
    ff.close()


"""Normalize a vector assuming an orhtogonal basis"""


def normalize_vec(vec):
    sum = 0
    for i in range(len(vec)):
        sum += vec[i] ** 2
    return vec / sum ** 0.5


"""Get the hamiltonian of a converged DFT calculation using the cells Fermi energy as the reference"""


def get_hs_and_change_reference(calc, k=0, sp=0, transdir=2):
    atoms = calc.atoms
    atoms.calc = calc
    calc.initialize(atoms)
    calc.set_positions(atoms)
    #
    # get hamiltonian and overlap matrix
    # H_skMM, S_kMM,k_index,s_index = get_lcao_hamiltonian(calc,spin=sp,kpoint=k,originalversion=False)
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    #
    # we reduce the matrix shape for one kpoint and spin
    print("reduce the matrix shape for one kpoint and spin: k=%i s=%i" % (k, sp))
    H, S = H_skMM[k, sp], S_kMM[k]
    # H, S = H_skMM[0, 0], S_kMM[0]
    H_skMM = H
    S_kMM = S
    #
    Efs = calc.get_fermi_level()  # get fermi level
    remove_pbc(
        atoms, H_skMM[:, :], S_kMM[:, :], d=transdir
    )  # remove PBC in z-direction (d=2)
    H_skMM[:, :] -= Efs * S_kMM[:, :]  # use Fermi level as energy reference
    #
    return H_skMM, S_kMM  # ,k_index,s_index


"""Get the subspace in a hamiltonian, which belongs to the molecule in a cell containing a junction """


def find_mol_bf_indizes(atoms, calc):
    bfs_atoms = [
        "N",
        "C",
        "H",
        "Ru",
        "P",
        "Fe",
        "S",
        "Se",
        "Mo",
        "O",
    ]  # Atom types on the molecule
    numvalence = {
        "H": 1,
        "C": 4,
        "Au": 11,
        "N": 5,
        "Ru": 16,
        "P": 5,
        "Cl": 7,
        "O": 6,
        "Fe": 8,
        "S": 6,
        "Se": 6,
        "Mo": 14,
    }  # Number of valence electrons
    numElecMol = 0
    numElecOther = 0
    symbols = atoms.get_chemical_symbols()  # get chemical symbols
    bfs_mol = []
    atom_mol = []
    nuo1 = 1e4
    n1 = 0
    symbol_vec = []
    for i in range(len(atoms)):
        if symbols[i] in bfs_atoms:  # if atom type is one of the chosen
            n2 = n1 + calc.wfs.setups[i].nao
            for j in range(n1, n2):
                bfs_mol.append(j)  # append wf to molecule
            numElecMol += numvalence[symbols[i]]  # sum up electrons on molecule
            nuo1 = min(nuo1, n1)
            atom_mol.append(i)  # make list with indexes of atoms on molecule
            symbol_vec.append(symbols[i])  # make list of interested atoms
        else:
            numElecOther += numvalence[symbols[i]]  # sum up electrons in Au
        n1 += calc.wfs.setups[i].nao
    #
    print(
        "------------------------------------------------------------------------------------------"
    )
    print("number of atoms in cell:", len(atoms))
    print("atoms in molecule:", len(atom_mol))
    print(
        "------------------------------------------------------------------------------------------"
    )
    #
    nhomo = numElecMol / 2
    nuo_mol = len(bfs_mol)  # number of orbitals on molecule
    #
    return bfs_mol, nuo_mol, atom_mol, nhomo


"""Get the sub-hamiltonian containing only the molecular basis function"""


def get_molecular_hs(calc, atoms, k=0, spin=0):
    h, s = get_hs_and_change_reference(calc, k=k, sp=spin)
    bfs_mol, nuo_mol, atom_mol, nhomo = find_mol_bf_indizes(atoms, calc)
    H = np.empty(
        [nuo_mol, nuo_mol]
    )  # create empty hamiltonian of the molecular subspace
    S = np.empty(
        [nuo_mol, nuo_mol]
    )  # create empty scattering matrix of the molecular subspace
    #
    for ii, i in enumerate(bfs_mol):
        for jj, j in enumerate(bfs_mol):
            H[ii, jj] = h[i, j]  # write values into molecular hamiltonian
            S[ii, jj] = s[i, j]
    #
    return H, S, atom_mol, nhomo


"""Standard eigenvalue problem"""


def diagonalize_h(H, S=None):
    # find eigen-energies and -states of the molecular subspace

    # if S==None: S=np.eye(H.shape[1]) #replace with the following try/except
    try:
        S.shape
    except NameError:
        S = np.eye(H.shape[1])

    from scipy.linalg import eig

    ev, V = eig(H, b=S)  # calculate molecule eigenvalues and eigenvectors
    # print('V',V)
    nn = np.argsort(ev)  # sort MOs by energy
    # print('nn:',nn)
    ev = ev[nn]  # get eigenstates sorted by energy
    V = V[:, nn]
    return ev, V


"""Diagonalize the molecular subspace in a junction hamiltonian and transform the coupling accordingly"""


def subdiagonalize_h(h_ii, s_ii, index_j):
    nb = h_ii.shape[0]
    nb_sub = len(index_j)
    h_sub_jj = get_subspace(h_ii, index_j)
    s_sub_jj = get_subspace(s_ii, index_j)
    e_j, v_jj = np.linalg.eig(np.linalg.solve(s_sub_jj, h_sub_jj))
    normalize(v_jj, s_sub_jj)  # normalize: <v_j|s|v_j> = 1
    permute_list = np.argsort(e_j.real)
    e_j = np.take(e_j, permute_list)
    v_jj = np.take(v_jj, permute_list, axis=1)
    #
    # setup transformation matrix
    c_ii = np.identity(nb, complex)
    for i in xrange(nb_sub):
        for j in xrange(nb_sub):
            c_ii[index_j[i], index_j[j]] = v_jj[i, j]
    #
    h1_ii = unitary_trans(c_ii, h_ii)
    s1_ii = unitary_trans(c_ii, s_ii)
    #
    return h1_ii, s1_ii, c_ii, e_j


"""Get the subspace spanned by the basis function listed in index"""


def get_subspace(matrix, index):
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    return matrix.take(index, 0).take(index, 1)


"""Normalize column vectors.   <matrix[:,i]| S |matrix[:,i]> = 1    """


def normalize(matrix, S=None):
    for col in matrix.T:
        if S is None:
            col /= np.linalg.norm(col)
        else:
            col /= np.sqrt(np.dot(col.conj(), np.dot(S, col)))


""" Write some (or all) eigenstates and eigenvalues into a file"""


def write_eigenvalues_into_file(ev, k, spin):
    evlist = list(ev)
    evlist.sort()
    outeigen = open("eigenvals_k_%s_spin_%s.out" % (k, spin), "w")
    print("Eigenvalues of the molecular states (cap=+10ev):")
    for i in range(len(evlist)):
        if float(evlist[i].real) > 10.0:
            break
        outeigen.writelines("%s\n" % (evlist[i].real))
    outeigen.close()


"""Derivation of interesting quantities, such as gap size, homo and lumo energies. Just for nicer output"""


def print_interesting_quantities(ev, needed_evlist):
    for i in range(len(ev)):
        if float(ev[i].real) > 0:
            homo = float(ev[i - 1].real)
            lumo = float(ev[i].real)
            gap = lumo - homo
            lumoid = i
            break
    homoid = lumoid - 1
    #
    print(
        "----------------------------------------------------------------------------"
    )
    print("HOMO-LUMO GAP:", gap)
    print("HOMO:", homo)
    print("LUMO:", lumo)
    print(
        "----------------------------------------------------------------------------"
    )
    print("Eigenvalues of interested orbitals:")
    for i in range(len(needed_evlist)):
        print(needed_evlist[i])
    print(
        "----------------------------------------------------------------------------"
    )


"""Perform a matrix rotation <a_dag|b|a>"""


def unitary_trans(a, b):
    c = np.dot(np.transpose(np.conj(a)), np.dot(b, a))
    return c


"""lowdin orthogonalization. Not really working..."""


def lowdin(H, S):  # Not really working....
    from scipy.linalg import eig, sqrtm

    s, v = eig(S)
    for i in range(v.shape[1]):
        normalize_vec(v[:, i])
    X = sqrtm(np.linalg.inv(S))
    X2 = sqrtm(S)
    H2 = unitary_trans(X, H)
    S_test = unitary_trans(X, S)
    return H2, X2


"""Get the mulliken charges out of a converged DFT calculation. Has to be used either at the end of a DFT calculation or with a gpw file written with mode='all'"""


def get_mulliken_charges(atoms, calc):
    from gpaw.lcao.tools import get_mulliken

    char = get_mulliken(calc, range(len(atoms)))
    outfile = "mulliken_charges.out"

    numvalence = {
        "H": 1,
        "C": 4,
        "Au": 11,
        "N": 5,
        "Ru": 16,
        "P": 5,
        "Cl": 7,
        "O": 6,
        "Fe": 8,
        "S": 6,
        "Se": 6,
        "Mo": 14,
    }  # Number of valence electrons
    numElecMol = 0
    symbols = atoms.get_chemical_symbols()  # get chemical names

    valelec = []
    for i in range(len(atoms)):
        valelec.append(numvalence[symbols[i]])
        numElecMol += numvalence[symbols[i]]  # sum up electrons on mol

    sum, sumAu = 0, 0
    if world.rank == 0:
        out = open(outfile, "w")
        for i in range(len(atoms)):
            out.writelines("%s  %s  %s\n" % (i, symbols[i], -(char[i] - valelec[i])))
            if symbols[i] != "Au":
                sum += -(char[i] - valelec[i])
            else:
                sumAu += -(char[i] - valelec[i])
        out.writelines(
            "#charge on molecule:   %s\n #charge on Au:: %s\n" % (sum, sumAu)
        )
        out.close()
