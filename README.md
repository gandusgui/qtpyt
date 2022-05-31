# qtpyt

A quantum transport library based on the Non-equilibrium Green's function (NEGF) formalism written in Python.

## Features

* `base` : Base API for Green's function and Self-energies.
* `block_tridiag` : Block-tridiagonal Green's function and recursive algorithms.
* `surface` : Surface Green's function based on *Principal Layer*.
* `screening` : GW many-body screening.
* `parallel` : MPI API for parallel calculations.

## Dependencies


* ase
* scipy
* numpy >= 1.21, < 1.22
* numba >= 0.55
* mpi4py

## License

The qtpyt license is MIT, please see the LICENSE file.
