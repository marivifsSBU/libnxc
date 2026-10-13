# Building and testing libnxc on a Linux cluster

`build_and_test.sh` builds the C++/Fortran library with CMake, runs the CTest suite (49
reference cases against pylibnxc, through the C++ and Fortran APIs, covering shipped
and exported models), and then a throughput benchmark. It uses conda-forge compilers, so no
cluster modules are needed.

```bash
conda env create -f devtools/hpc/environment.yml -p $SCRATCH/envs/nxc-build
export PY=$SCRATCH/envs/nxc-build/bin/python
sbatch --partition=<partition> --export=ALL,PY=$PY devtools/hpc/build.slurm   # from the repository root
```
The job log should end with `100% tests passed`, followed by the benchmark lines. The log is
also written to `devtools/hpc/results/build_and_test.log`.

On aarch64 Linux, remove `+cpu` from the torch line in `environment.yml`: those wheels are
CPU-only and carry no `+cpu` tag.

## Notes
- **Compilers.** libtorch 2.14 requires C++20, so GCC ≥ 11.
- **torch version.** The exported models in `models/` were written with torch 2.14.0
  (TorchScript archive format 10). An older libtorch may refuse to load them.
- **BLAS.** Use the pthreads build of OpenBLAS. The OpenMP build loads a second OpenMP runtime
  next to torch's, and the process aborts.
- **Threads.** The benchmark uses `OMP_NUM_THREADS` threads (the Slurm script sets it from
  `--cpus-per-task`).
- **GPU.** libnxc can move the model to CUDA (`func_param.cuda = 1`). That needs the CUDA torch
  wheel and a CUDA toolkit at build time; this recipe is CPU-only.
