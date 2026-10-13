#!/bin/bash
# Build libnxc with CMake, run the C++/Fortran tests and the benchmark (on a compute node):
#   PY=<env python> devtools/hpc/build_and_test.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:?set PY to the python of the nxc-build env}
NPROC=${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-8}}
ENVBIN=$(dirname "$PY")
export PATH=$ENVBIN:$PATH
export PYTHONDONTWRITEBYTECODE=1
LOG=$HERE/results/build_and_test.log
mkdir -p $HERE/results
{
  echo "== host $(hostname)  $(date)  nproc $NPROC"
  uname -a
  $PY -c "import torch, pyscf, numpy; print('torch', torch.__version__, 'pyscf', pyscf.__version__, 'numpy', numpy.__version__)"
  TP=$($PY -c "import torch; print(torch.utils.cmake_prefix_path)")
  cd $REPO
  TRIPLET=$(uname -m)-conda-linux-gnu               # x86_64 or aarch64
  cmake -S . -B build-hpc -G Ninja -DCMAKE_PREFIX_PATH="$TP" \
        -DCMAKE_CXX_COMPILER=$ENVBIN/$TRIPLET-c++ \
        -DCMAKE_Fortran_COMPILER=$ENVBIN/$TRIPLET-gfortran \
        -DNXC_TEST_PYTHON=$PY
  cmake --build build-hpc -j $NPROC
  ctest --test-dir build-hpc --output-on-failure
  echo "== benchmark (OMP_NUM_THREADS=${OMP_NUM_THREADS:-default})"
  for m in $REPO/models/GGA_XC_PBECLONE_TL $REPO/models/MGGA_XC_DEMO_RANDOM; do
    lvl=1; [[ $(basename $m) == MGGA* ]] && lvl=2
    for ns in 1 2; do ./build-hpc/test/revival/bench $m $lvl $ns 1000000 3; done
  done
  ./build-hpc/test/revival/bench MGGA_XC_SCAN 2 1 1000000 3
} 2>&1 | tee $LOG
echo "done: $LOG"
