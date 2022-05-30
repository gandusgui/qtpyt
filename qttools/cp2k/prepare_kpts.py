import argparse
import sys
import os
import re
from pathlib import Path

import numpy as np

from ase.io import read

from qttools.cp2k.tools import (
    read_basis_dict,
    read_fermi,
    read_mp_kmesh,
    read_from_folder,
)
from qtpyt.surface.basis import Basis
from qtpyt.surface.tools import prepare_central_matrices, prepare_leads_matrices


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


def main():
    parser = argparse.ArgumentParser(argument_default=None)
    parser.add_argument(
        "--folder", help="Working directory", type=str, default=os.getcwd()
    )
    parser.add_argument(
        "--out", "-o", help="CP2K output file", type=str, default=r"ener[gy]*\.o[ut]*"
    )
    parser.add_argument(
        "--atoms", help="Atomic structure file", type=str, default=r".*\.xyz"
    )
    parser.add_argument("--fermi", type=float, help="Fermi level in eV")
    parser.add_argument("--kmesh", type=int, nargs=3, help="Monkhorst-Pack meshgird")
    parser.add_argument(
        "--basis", nargs="*", help="Basis dictionary", default=None, action=ParseKwargs
    )
    parser.add_argument(
        "--no-sym",
        help="Do not assume time-reversal symmetry. Read all k-points.",
        action="store_true",
    )
    parser.add_argument("--leads", help="Consider leads.", action="store_true")
    parser.add_argument(
        "--offset",
        type=int,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help="Monkhorst-Pack offset",
    )

    argv = parser.parse_args()

    path = Path(argv.folder)
    files = [p.name for p in path.glob("*") if p.is_file()]

    search_atoms = re.compile(argv.atoms)
    search_out = re.compile(argv.out)

    atoms = list(filter(search_atoms.match, files))
    out = list(filter(search_out.match, files))

    # _varname = lambda var: [ k for k,v in globals().items() if v == var][0]
    # _varname.__doc__ = "Get variable name as string."

    def _check(*args):
        """Check that parsed argument files are single and exist."""
        for arg, name in args:
            # name = _varname(arg)
            if not arg:
                raise RuntimeError(
                    f"""
                    File not found {name} {arg}. Use --{name} at input."""
                )
            if len(arg) > 1:
                raise RuntimeError(
                    f"""
                    Found multiple matches for {name} {arg}. Use --{name} at input."""
                )

    _check((out, "out"), (atoms, "atoms"))
    out_file = out[0]
    atoms_file = atoms[0]

    atoms = read(atoms_file)
    basis_dict = argv.basis or read_basis_dict(out_file)
    basis = Basis.from_dictionary(atoms, basis_dict)
    fermi = argv.fermi or read_fermi(out_file)
    kmesh = argv.kmesh or read_mp_kmesh(out_file)
    sym = bool(not argv.no_sym)
    offset = argv.offset

    print(
        f"""Input params:
        atoms : {atoms_file},
        out : {out_file},
        basis : {basis_dict},
        kmesh : {kmesh},
        symmetry : {sym},
        offset : {offset},
        fermi : {fermi}"""
    )

    h_k, s_k = read_from_folder(path, reverse=sym)
    h_k -= fermi * s_k

    if argv.leads:
        kpts_t, h_kii, s_kii, h_kij, s_kij = prepare_leads_matrices(
            basis, h_k, s_k, kmesh, offset
        )

        del h_k
        del s_k

        np.save(path / "hs_kii_kij", (h_kii, s_kii, h_kij, s_kij))
        np.savetxt(path / "kpts.txt", kpts_t, fmt="%6.3f")

    else:
        kpts, h_kii, s_kii = prepare_central_matrices(basis, h_k, s_k, kmesh, offset)

        del h_k
        del s_k

        np.save(path / "hs_kii", (h_kii, s_kii))
        np.save(path / "kpts.txt", kpts, fmt="%6.3f")


if __name__ == "__main__":
    sys.exit(main())
