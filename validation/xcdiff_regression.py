"""Regression of the revived PySCF glue against numbers produced with the original
pylibnxc + PySCF 1.x in Sebastian's XC-diff notebooks
(~/RESEARCH/Sebastian/Sebastian/XC-diff/DataForPaper/{AtomicGrid,H2OGeomXCdiff}.ipynb).
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root (pylibnxc)

import numpy as np
from pyscf import gto, lib
import pylibnxc.pyscf as nxc_pyscf   # revival branch (PySCF >= 2)

lib.num_threads(4)

# He atom, def2-QZVPD, UKS, grid levels 3 and 9 (AtomicGrid.ipynb)
REF_HE = {3: -2.89919205352114, 9: -2.89919157689889}
he = gto.M(atom='He 0 0 0', basis='def2-QZVPD', verbose=0)
for lvl, ref in REF_HE.items():
    mf = nxc_pyscf.UKS(he, nxc='MGGA_XC_XCDIFF')
    mf.grids.level = lvl
    e = mf.kernel()
    print(f'He  level {lvl}: E = {e:.10f}  ref = {ref:.10f}  diff = {1e6*(e-ref):8.2f} uEh', flush=True)

# H2O, r = 0.95 A, theta = 103 deg, def2-QZVPD, UKS, grid level 6 (first point of the
# 20x20 scan in H2OGeomXCdiff.ipynb)
REF_H2O = -76.3378476780108
r, th = 0.96 - 0.01, np.deg2rad(104.0 - 1.0)
hx, hy = r * np.sin(th / 2), r * np.cos(th / 2)
h2o = gto.M(atom=[('O', (0, 0, 0)), ('H', (hx, hy, 0)), ('H', (-hx, hy, 0))],
            basis='def2-QZVPD', verbose=0)
mf = nxc_pyscf.UKS(h2o, nxc='MGGA_XC_XCDIFF')
mf.grids.level = 6
e = mf.kernel()
print(f'H2O level 6: E = {e:.10f}  ref = {REF_H2O:.10f}  diff = {1e6*(e-REF_H2O):8.2f} uEh')
