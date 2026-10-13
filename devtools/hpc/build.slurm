#!/bin/bash
#SBATCH --job-name=nxc-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=nxc_build_%j.out
# sbatch --partition=<partition> --export=ALL,PY=$PY devtools/hpc/build.slurm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "$SLURM_SUBMIT_DIR"            # submit from the repository root
devtools/hpc/build_and_test.sh
