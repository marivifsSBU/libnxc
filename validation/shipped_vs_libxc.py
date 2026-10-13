"""Run shipped libnxc TorchScript models self-consistently in PySCF 2.x and
compare with the corresponding Libxc functionals.

For each (libnxc model, Libxc functional) pair and each molecule:
  * SCF total energy, HOMO, dipole with the NN model and with Libxc (same grid/basis);
  * non-SCF check on the converged Libxc density: E_xc and the AO-basis V_xc matrix
    from the NN model vs Libxc (isolates the model error from SCF relaxation).

Run from the repository root:
  python validation/shipped_vs_libxc.py
"""
import os
import sys
import time
import json
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root (pylibnxc)

import numpy as np
from pyscf import gto, dft, lib
import pylibnxc.pyscf as nxc_pyscf   # revival branch (PySCF >= 2)

lib.num_threads(4)

MOLS = {
    'H2O': dict(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='def2-tzvp', spin=0),
    'OH': dict(atom='O 0 0 0; H 0 0 0.9697', basis='def2-tzvp', spin=1),
}

PAIRS = [
    ('LDA_X_NUEG + LDA_C_NPW92', 'LDA,PW_MOD'),
    ('GGA_XC_PBE', 'PBE'),
    ('GGA_X_PBE, GGA_C_PBE', 'PBE'),
    ('MGGA_XC_SCAN', 'SCAN'),
    ('MGGA_X_SCAN, MGGA_C_SCAN', 'SCAN'),
    ('MGGA_XC_XCDIFF', 'SCAN'),   # no Libxc counterpart; SCAN shown for scale
]

GRID_LEVEL = 5


def build(molname, xc=None, nxc=None):
    m = MOLS[molname]
    mol = gto.M(atom=m['atom'], basis=m['basis'], spin=m['spin'], verbose=0)
    if nxc is None:
        mf = (dft.RKS if m['spin'] == 0 else dft.UKS)(mol)
        mf.xc = xc
    else:
        mf = (nxc_pyscf.RKS if m['spin'] == 0 else nxc_pyscf.UKS)(mol, nxc=nxc)
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = 1e-9
    mf.max_cycle = 100
    return mf


def homo(mf):
    if mf.mo_occ.ndim == 1:
        return mf.mo_energy[mf.mo_occ > 0].max()
    return max(mf.mo_energy[s][mf.mo_occ[s] > 0].max() for s in range(2))


def nr_vxc(mf, dm):
    ni, mol, grids = mf._numint, mf.mol, mf.grids
    if grids.coords is None:
        grids.build()
    if mf.mol.spin == 0:
        n, exc, vxc = ni.nr_rks(mol, grids, mf.xc, dm)
    else:
        n, exc, vxc = ni.nr_uks(mol, grids, mf.xc, dm)
    return exc, np.asarray(vxc)


def main():
    rows = []
    for molname in MOLS:
        ref_cache = {}
        for nxc, xc in PAIRS:
            if xc not in ref_cache:
                mf_ref = build(molname, xc=xc)
                t0 = time.time()
                e_ref = mf_ref.kernel()
                t_ref = time.time() - t0
                ref_cache[xc] = (mf_ref, e_ref, t_ref)
            mf_ref, e_ref, t_ref = ref_cache[xc]
            dm_ref = mf_ref.make_rdm1()

            mf = build(molname, nxc=nxc)
            t0 = time.time()
            try:
                e_nn = mf.kernel(dm0=dm_ref)
                err = None
            except Exception as ex:  # keep going, record the failure
                e_nn, err = float('nan'), f'{type(ex).__name__}: {ex}'
            t_nn = time.time() - t0

            exc_nn, vxc_nn = nr_vxc(mf, dm_ref)
            exc_ref, vxc_ref = nr_vxc(mf_ref, dm_ref)
            row = dict(
                mol=molname, nxc=nxc, libxc=xc,
                converged=bool(mf.converged), cycles=None,
                E_nn=float(e_nn), E_libxc=float(e_ref), dE_mEh=1e3 * float(e_nn - e_ref),
                dHOMO_mEh=1e3 * float(homo(mf) - homo(mf_ref)) if err is None else None,
                dDip_D=float(np.linalg.norm(mf.dip_moment(verbose=0) - mf_ref.dip_moment(verbose=0)))
                if err is None else None,
                nonSCF_dExc_mEh=1e3 * float(exc_nn - exc_ref),
                nonSCF_dVxc_max=float(np.abs(vxc_nn - vxc_ref).max()),
                t_nn_s=t_nn, t_libxc_s=t_ref, error=err)
            rows.append(row)
            print(json.dumps(row), flush=True)

    out = os.path.join(HERE, '..', 'results', 'shipped_vs_libxc.json')
    with open(out, 'w') as f:
        json.dump(dict(grid_level=GRID_LEVEL, torch=__import__('torch').__version__,
                       pyscf=__import__('pyscf').__version__, rows=rows), f, indent=1)

    print(f"\n{'mol':4s} {'libnxc model':26s} {'Libxc':10s} {'dE_SCF':>9s} {'dExc_nonSCF':>12s} "
          f"{'max|dVxc|':>10s} {'dHOMO':>8s} {'dDip':>7s} {'t_nn/t_lxc':>10s}")
    for r in rows:
        print(f"{r['mol']:4s} {r['nxc']:26s} {r['libxc']:10s} {r['dE_mEh']:9.3f} "
              f"{r['nonSCF_dExc_mEh']:12.3f} {r['nonSCF_dVxc_max']:10.2e} "
              f"{(r['dHOMO_mEh'] or float('nan')):8.3f} {(r['dDip_D'] or float('nan')):7.4f} "
              f"{r['t_nn_s']:5.1f}/{r['t_libxc_s']:4.1f}")
    print('(energies in mEh, dipole in Debye; converged:', all(r['converged'] for r in rows), ')')


if __name__ == '__main__':
    main()
