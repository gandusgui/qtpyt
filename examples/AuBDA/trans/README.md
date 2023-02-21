# Transport calculation with RGF


How to run

```language=python
# Energy calculation of scattering region & dump H, S matrices
gpaw -P _1_ python scatt.py && python dump.py

# Energy calculation of leads.
gpaw -P _1_ python ../leads/leads.py

# Transport calculation
mpiexec -n _2_ -env OMP_NUM_THREADS=_3_ -env NUMBA_NUM_THREADS=_3_ python trans.py

```

Here,

- \_1_ is the number of processors for the GPAW calculation
- \_2_ is the number of processors for the `qtpyt` calculation
- \_3_ is the number of threads for the `qtpyt` calculation

In general, \_2_ + \_3_ = \_1_.

