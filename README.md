# OpenMPISample_Jacobi
Runs jacobi iterations using both OpenMP and OpenMPI in combination. Written to run on a BCCD cluster.


## Run
Commands used to run on a BCCD cluster:

```bash
mpicxx Driver.cpp -openmp
bccd-syncdir ~ ~/machines-openmpi
mpirun -machinefile ~/machines-openmpi /tmp/node000-bccd/jacobi/a.out > Jacobi.ppm
```
