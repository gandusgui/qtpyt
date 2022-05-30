import numpy as np
from scipy.sparse import coo_matrix
from ase.units import Hartree

import re
from pathlib import Path

from ..misc import tri2full


float_match = "([-+]?\d*\.\d+)"
int_match = "(\d+)"


def read_complex_bincsr(filename, dtype=np.float64):
    """Read complex matrix stored in .csr CP2K format.
    
    NOTE :: The returned matrix is in units of [eV].

    Args:
        filename : (str, file object)
        dtype : (np.dtype)
            data type of real and immaginary parts.
    """
    dt = np.dtype(
        [
            ("o1", np.uint32),
            ("x", np.uint32),
            ("y", np.uint32),
            ("real", dtype),
            ("imag", dtype),
            ("o2", np.uint32),
        ]
    )

    coo = np.fromfile(filename, dt, offset=0)
    data = coo["real"] + 1.0j * coo["imag"]
    mat = coo_matrix((data, (coo["x"] - 1, coo["y"] - 1))).toarray()
    mat *= Hartree
    tri2full(mat, UL="U")
    return mat


def read_real_bincsr(filename, dtype=np.float64):
    """Read real matrix stored in .csr CP2K format.
    
    NOTE :: The returned matrix is in units of [eV].

    Args:
        filename : (str, file object)
        dtype : (np.dtype)
            data type of real and immaginary parts.
    """
    dt = np.dtype(
        [
            ("o1", np.uint32),
            ("x", np.uint32),
            ("y", np.uint32),
            ("data", dtype),
            ("o2", np.uint32),
        ]
    )

    coo = np.fromfile(filename, dt, offset=0)
    mat = coo_matrix((coo["data"], (coo["x"] - 1, coo["y"] - 1))).toarray()
    mat *= Hartree
    tri2full(mat, UL="U")
    return mat


def read_fermi(filename):
    """Read Fermi level (in [eV]) from output file."""
    pattern = "Fermi energy:" + "\s+" + float_match
    matches = re.findall(pattern, open(filename, "r").read())
    fermi = np.unique(np.array(matches, dtype=float))
    assert (
        fermi.size == 1
    ), "Found multiple Fermi levels that don't match! {fermi*Hartree}"
    return fermi[0] * Hartree


def read_mp_kmesh(filename):
    """Read Monkhorst-Pack k-point grid from output file."""
    pattern = "K-Point grid" + 3 * ("\s+" + int_match)
    match = re.search(pattern, open(filename, "r").read())
    size = [int(match.group(i)) for i in range(1, 4)]
    return size


def read_kpts(filename):
    """Read list of k-points from output file."""
    fp = open(filename, "r").read()
    pattern = "List of Kpoints \[2 Pi/Bohr\]" + "\s+" + int_match
    num_kpts = int(re.search(pattern, fp).group(1))
    kpts = np.empty((num_kpts, 3))
    weights = np.empty(num_kpts)
    pattern = "BRILLOUIN\|.+?" + 4 * ("\s+" + float_match)
    for ik, kpt_info in enumerate(re.finditer(pattern, fp)):
        weights[ik] = float(kpt_info.group(1))
        kpts[ik] = [float(kpt) for kpt in kpt_info.group(2, 3, 4)]
        if ik == num_kpts - 1:
            break
    return kpts, weights


def read_basis_dict(filename):
    """Read basis function info as dictionary."""
    basis = {}
    fp = open(filename, "r").read()
    # for kind in re.finditer('ATOMIC KIND INFORMATION', fp):
    #     symbol = re.search('Atomic kind:\s+(\w)', fp).group(1)
    for kind in re.finditer("Atomic kind:\s+(\w+)", fp):
        symbol = kind.group(1)
        nao = re.search("Number of spherical basis functions:\s+(\d+)", fp).group(1)
        basis[symbol] = int(nao)
    return basis


def read_from_folder(path, project="*", reverse=True):
    """Read Hamilton and Overlap matrices.
    
    When `reverse` is True, read matrices in reversed order. 

    NOTE :: Use `reverse` flag for CP2K calculations that 
    use the default Monkhorst-Pack scheme.

    Both CP2K and PYQT apply inversion symmetry along the x-direction.
    However:
    e.g. with Monkhorst-Pack mesh = (2,2,2)
    PYQT considers the second half of the BZ:
       [ 0.25, -0.25, -0.25],
       [ 0.25, -0.25,  0.25],
       [ 0.25,  0.25, -0.25],
       [ 0.25,  0.25,  0.25]]
    CP2K considers the first half of the BZ:
       [ -0.25, -0.25, -0.25],
       [ -0.25, -0.25,  0.25],
       [ -0.25,  0.25, -0.25],
       [ -0.25,  0.25,  0.25]]

    Hence we 
        (i) read the k-point matrices from last k-point to first k-point
        (ii) transpose each matrix since A[k] = A[-k].T


    Args:
        path : (str, pathlib.Path object)
            path to folder
        project : (str), optional
            name of the project. 
            default is to read all Hamilton ans Overlap in folder.
    """
    path = Path(path)
    h_cc_k_files = list(path.glob(f"{project}-KS_SPIN_1_K_*-1_0.csr"))
    s_cc_k_files = list(path.glob(f"{project}-S_SPIN_1_K_*-1_0.csr"))

    pattern = ".+?_K_(\d+)-.+?"
    h_cc_k_files = sorted(
        h_cc_k_files,
        key=lambda p: int(re.search(pattern, p.stem).group(1)),
        reverse=reverse,
    )
    s_cc_k_files = sorted(
        s_cc_k_files,
        key=lambda p: int(re.search(pattern, p.stem).group(1)),
        reverse=reverse,
    )

    if reverse:
        read = lambda file: read_complex_bincsr(file).T
    else:
        read = lambda file: read_complex_bincsr(file)
    h_cc_k = np.stack(list(map(read, h_cc_k_files)))
    s_cc_k = np.stack(list(map(read, s_cc_k_files)))
    return h_cc_k, s_cc_k
