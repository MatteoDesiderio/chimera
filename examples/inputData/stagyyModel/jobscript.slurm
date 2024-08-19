#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --partition=standard
#SBATCH --qos=long
#SBATCH --account=n03-jpb-users
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block

export MPICH_NO_BUFFER_ALIAS_CHECK=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=1

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel metis petsc valgrind4hpc cmake cray-python

# srun launches the parallel program based on the SBATCH options

srun --cpu-bind=cores ./stagyympi 
