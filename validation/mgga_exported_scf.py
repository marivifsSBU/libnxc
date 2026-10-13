"""SmallData meta-GGA models exported to libnxc, self-consistently in PySCF:

  exported model via pylibnxc  vs  the same model via SmallData/mgga.py (JAX)  vs  Libxc r2SCAN

on H2O (RKS) and OH (UKS), def2-TZVP, grid level 3.  The pylibnxc/JAX difference must be
at the SCF-convergence level; the clone/r2SCAN difference is the cloning error.

    NXC_SMALLDATA=<dir with mgga.py> python validation/mgga_exported_scf.py
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALLDATA = os.environ['NXC_SMALLDATA']
sys.path.insert(0, ROOT)
sys.path.insert(0, SMALLDATA)

from pyscf import gto, dft, lib
import pylibnxc.pyscf as nxc
import mgga                                   # read-only use of the SmallData code

lib.num_threads(4)
MODELS = os.path.join(ROOT, 'models')
MOLS = {'H2O': ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 0),
        'OH': ('O 0 0 0; H 0 0 0.970', 1)}
EXPORTED = ['MGGA_XC_R2SCAN_CLONE', 'MGGA_XC_R2SCAN_CLONE_SC']


def setup(mf):
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    return mf


def main():
    rows = []
    for mname, (atom, spin) in MOLS.items():
        mol = gto.M(atom=atom, spin=spin, basis='def2-tzvp', verbose=0)
        KS = dft.RKS if spin == 0 else dft.UKS
        ref = setup(KS(mol))
        ref.xc = 'r2scan'
        e_r2 = ref.kernel()
        dm = ref.make_rdm1()
        for name in EXPORTED:
            man = json.load(open(os.path.join(MODELS, name, 'manifest.json')))
            t0 = time.time()
            mf = setup((nxc.RKS if spin == 0 else nxc.UKS)(mol, nxc=os.path.join(MODELS, name)))
            e_nxc = mf.kernel(dm0=dm)
            t_nxc = time.time() - t0
            t0 = time.time()
            jx = setup(mgga.attach(KS(mol), mgga.load(os.path.join(MODELS, name, man['sources']['xc']['path']))))
            e_jax = jx.kernel(dm0=dm)
            t_jax = time.time() - t0
            row = dict(mol=mname, model=name, E_nxc=float(e_nxc), E_jax=float(e_jax), E_r2scan=float(e_r2),
                       conv=bool(mf.converged and jx.converged), t_nxc=t_nxc, t_jax=t_jax)
            rows.append(row)
            print(f"{mname:4s} {name:26s} E = {e_nxc:.9f}  nxc-jax = {1e6*(e_nxc-e_jax):8.3f} uEh  "
                  f"nxc-r2SCAN = {1e3*(e_nxc-e_r2):8.3f} mEh  conv={row['conv']}  t={t_nxc:.1f}/{t_jax:.1f} s", flush=True)
    with open(os.path.join(ROOT, 'results', 'mgga_exported_scf.json'), 'w') as f:
        json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
