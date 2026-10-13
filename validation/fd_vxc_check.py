"""Finite-difference check that the libnxc glue's V_xc is the derivative of its E_xc
(tr[V_xc dD] vs (E_xc[D+h dD] - E_xc[D-h dD]) / 2h), RKS and UKS, GGA and meta-GGA."""
import os, sys, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root (pylibnxc)
import numpy as np
from pyscf import gto, dft
import pylibnxc.pyscf as nxc_pyscf   # revival branch (PySCF >= 2)

rng = np.random.default_rng(0)
for spin, atom in [(0, 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'), (1, 'O 0 0 0; H 0 0 0.9697')]:
    mol = gto.M(atom=atom, basis='def2-svp', spin=spin, verbose=0)
    ref = (dft.RKS if spin == 0 else dft.UKS)(mol); ref.xc = 'PBE'; ref.kernel()
    D = ref.make_rdm1()
    for nxc in ['GGA_XC_PBE', 'MGGA_XC_SCAN', 'MGGA_XC_XCDIFF']:
        mf = (nxc_pyscf.RKS if spin == 0 else nxc_pyscf.UKS)(mol, nxc=nxc); mf.grids.build()
        ni = mf._numint
        nr = (lambda d: ni.nr_rks(mol, mf.grids, mf.xc, d)) if spin == 0 else (lambda d: ni.nr_uks(mol, mf.grids, mf.xc, d))
        dD = rng.standard_normal(D.shape); dD = dD + dD.swapaxes(-1, -2); dD *= 1e-2 / np.abs(dD).max()
        _, e0, v0 = nr(D)
        h = 1e-3
        ep, em = nr(D + h * dD)[1], nr(D - h * dD)[1]
        fd = (ep - em) / (2 * h); an = np.sum(np.asarray(v0) * dD.swapaxes(-1, -2))
        print(f'spin={spin} {nxc:15s} FD = {fd: .10e}  tr(V dD) = {an: .10e}  rel.diff = {abs(fd-an)/abs(an):.1e}', flush=True)
