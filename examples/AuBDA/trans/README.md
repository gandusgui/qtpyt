gpaw -P # python scatt.py && python dump.py
gpaw -P # python ../leads/leads.py
mpiexec -n # -env OMP_NUM_THREADS=# -env NUMBA_NUM_THREADS=# python trans.py &

