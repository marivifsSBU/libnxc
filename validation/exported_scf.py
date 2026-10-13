"""End to end: exported TorchScript models run self-consistently in PySCF
through pylibnxc (revival branch), compared with

  * an independent JAX/equinox SCF of the same functional (same grids/basis):
      - TL / UT spin-aware clones: SmallData/polclone.py (read-only import)
      - the unpolarized PBE clone and the meta-GGA demo: exporter.recipes_jax
  * Libxc PBE (for the PBE clones)

def2-TZVP, grid level 3, conv_tol 1e-10.  Energies in mEh.

    NXC_SMALLDATA=<dir with polclone.py> python validation/exported_scf.py
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pyscf import gto, dft, lib

import pylibnxc.pyscf as nxc
from pylibnxc.pyscf.pyscf import _unpack_rho
from exporter import eqx_classes
from exporter.recipes_jax import reference_outputs, RECIPES

lib.num_threads(4)
MODELS = os.path.join(ROOT, 'models')
SMALLDATA = os.environ['NXC_SMALLDATA']  # independent JAX implementation (polclone.py)

MOLS = {
    'H2O': ('O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', 0),
    'CH4': ('C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629', 0),
    'N2': ('N 0 0 0; N 0 0 1.098', 0),
    'H': ('H 0 0 0', 1),
    'OH': ('O 0 0 0; H 0 0 0.970', 1),
    'O2(triplet)': ('O 0 0 0; O 0 0 1.208', 2),
    'CH3': ('C 0 0 0; H 1.079 0 0; H -0.5395 0.9344 0; H -0.5395 -0.9344 0', 1),
}

RUNS = [  # model, molecules, libxc twin
    ('GGA_XC_PBECLONE_TL', ['H2O', 'H', 'OH', 'O2(triplet)', 'CH3'], 'PBE'),
    ('GGA_XC_PBECLONE_UT', ['H2O', 'OH'], 'PBE'),
    ('GGA_XC_PBECLONE_UNPOL_S05R1', ['H2O', 'CH4', 'N2', 'OH'], 'PBE'),
    ('MGGA_XC_DEMO_RANDOM', ['H2O', 'OH'], None),
]


def mol_of(name):
    atom, spin = MOLS[name]
    return gto.M(atom=atom, spin=spin, basis='def2-tzvp', verbose=0)


def setup(mf):
    mf.grids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    return mf


def jax_ks(mol, model):
    """Independent JAX SCF for an exported model."""
    man = json.load(open(os.path.join(MODELS, model, 'manifest.json')))
    base = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    if man['recipe'] == 'xcdiff_pol':
        sys.path.insert(0, SMALLDATA)
        import polclone
        sys.path.remove(SMALLDATA)
        xc = polclone.XC(fx=polclone.load_model(os.path.join(MODELS, man['name'], man['sources']['fx']['prefix'])),
                         fc=polclone.load_model(os.path.join(MODELS, man['name'], man['sources']['fc']['prefix'])))
        return polclone.attach(base, xc)
    models = {k: eqx_classes.load_model(man['sources'][k]['prefix'])[0] for k in ('fx', 'fc')}
    xctype = RECIPES[man['recipe']]['family']

    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        out = reference_outputs(man['recipe'], models, _unpack_rho(rho, spin, xctype))
        return out['zk'], (out['vrho'], out['vsigma'], None, out.get('vtau')), None, None
    base.xc = 'LDA,'
    return base.define_xc_(eval_xc, xctype)


def main():
    rows = []
    for model, mols, twin in RUNS:
        for mname in mols:
            mol = mol_of(mname)
            row = dict(model=model, mol=mname)
            mf = setup((nxc.RKS if mol.spin == 0 else nxc.UKS)(mol, nxc=os.path.join(MODELS, model)))
            t0 = time.time()
            try:
                row['E_nxc'] = float(mf.kernel())
                row['conv_nxc'] = bool(mf.converged)
                row['t_nxc'] = time.time() - t0
            except Exception as ex:
                row['error'] = str(ex).strip().splitlines()[-1][:160]
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            dm = mf.make_rdm1()
            ref = setup(jax_ks(mol, model))
            t0 = time.time()
            row['E_jax'] = float(ref.kernel(dm0=dm))
            row['conv_jax'] = bool(ref.converged)
            row['t_jax'] = time.time() - t0
            if twin:
                lx = setup(dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol))
                lx.xc = twin
                row['E_libxc'] = float(lx.kernel())
            rows.append(row)
            print(json.dumps(row), flush=True)

    with open(os.path.join(ROOT, 'results', 'exported_scf.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    print(f"\n{'model':28s} {'mol':12s} {'E (pylibnxc)':>15s} {'nxc-jax':>9s} {'nxc-PBE':>9s}  {'t_nxc/t_jax (s)'}")
    for r in rows:
        if 'error' in r:
            print(f"{r['model']:28s} {r['mol']:12s} refused: {r['error']}")
            continue
        d1 = 1e3 * (r['E_nxc'] - r['E_jax'])
        d2 = f"{1e3 * (r['E_nxc'] - r['E_libxc']):9.3f}" if 'E_libxc' in r else f"{'-':>9s}"
        flag = '' if r['conv_nxc'] and r['conv_jax'] else ' (not converged)'
        print(f"{r['model']:28s} {r['mol']:12s} {r['E_nxc']:15.8f} {d1:9.2e} {d2}  {r['t_nxc']:5.1f}/{r['t_jax']:5.1f}{flag}")


if __name__ == '__main__':
    main()
