# GW calculation of CO Molecule with Au Leads.

> ### Workflow

* DFT calculation in LCAO basis with `GPAW` [1]. 
* Compute bare Coulomb AO matrix elements.
* GW calculation with `qtpyt`.

[1] Phys. Rev. B 71, 035109 (2005)

## DFT Calculation

```sh
mpiexec -n X gpaw python scatt.py
```

## Bare Coulomb Parameters

```sh
python makeAOs.py
mpiexec -n X -x OMP_NUM_THREADS=Y gpaw python makeU.py
mpiexec -n X -x OMP_NUM_THREADS=Y gpaw python makeV.py
```

## GW Calculation

```sh
mpiexec -n X -x OMP_NUM_THREADS=Y -x NUMBA_NUM_THREADS=Y python GW.py
```