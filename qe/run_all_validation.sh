#!/bin/bash
# libnxc test suites, then the QE twin validation. Run from anywhere:
#   PY=<python with torch, pyscf, jax, equinox> PW_X=<pw.x with NXC> QE_PSEUDO_DIR=<QE pseudo/> qe/run_all_validation.sh
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $R/qe/results
{
  echo "== libnxc ctest"
  ctest --test-dir $R/build-cmake 2>&1 | grep -E "tests passed|Failed"
  echo "== pylibnxc suite"
  (cd $R/pylibnxc/tests && PYTHONPATH=$R $PY -m pytest -q -p no:cacheprovider -W ignore::FutureWarning -W ignore::DeprecationWarning . 2>&1 | tail -1)
  echo "== exporter tests"
  (cd $R && $PY -m pytest -q -p no:cacheprovider -W ignore exporter/tests 2>&1 | tail -1)
} > $R/qe/results/libnxc_regression.log 2>&1
cd $R/qe && $PY run_twins.py > $R/qe/results/qe_twins.log 2>&1
