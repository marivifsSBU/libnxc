"""Si band gaps in QE for the SG15 meta-GGA set (PBE, Libxc r2SCAN, the exported r2SCAN clones
and density-trained models).

For each functional and lattice constant (its own stress-based a0, and the experimental
5.431 A): SCF (8^3 shifted k-mesh) then an nscf on Gamma, L, X and a fine line along Gamma-X,
nbnd = 8.  Indirect gap = min CBM over those points - VBM(Gamma); direct gap at Gamma.
The FFT grid is forced to 45^3 (not divisible by 8) for every run, so QE's zero-gradient
grid-point issue (qe/README.md, Limitations) does not touch the Libxc reference.  Meta-GGA
eigenvalues in QE are generalized-KS eigenvalues (tau-dependent potential as an operator).

    python run_gaps.py [--workers 2]      (writes runs/gaps/* and results/qe_si_gaps.{json,log})
"""
import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PW = os.environ.get('PW_X', 'pw.x')                       # pw.x built with -DQE_ENABLE_NXC=ON
PSEUDO = os.environ.get('NXC_SG15_DIR', 'pseudos/sg15')       # SG15 ONCV 1.2 pseudopotentials
MODELS = os.path.join(HERE, '..', 'models')
RUNS = os.path.join(HERE, 'runs', 'gaps')
BOHR_A = 0.529177210903
A_EXP = 5.431
NR = 45

# stress-based equilibrium lattice constants (results/qe_eos_sg15_from_stress.log)
FUNCTIONALS = {  # label: (input_dft, NXC_MODEL or None, a0 in A)
    'PBE': ('PBE', None, 5.4773),
    'r2SCAN (libxc)': ('R2SCAN', None, 5.4589),
    'NN-r2SCAN clone (exported)': ('NXC', os.path.join(MODELS, 'MGGA_XC_R2SCAN_CLONE'), 5.4576),
    'NN-r2SCAN clone sc-gated (exported)': ('NXC', os.path.join(MODELS, 'MGGA_XC_R2SCAN_CLONE_SC'), 5.4581),
}
# k-points in 2pi/a units: Gamma, L, X and a line along Gamma-X around Si's CBM (~0.85 X)
KPTS = [(0, 0, 0), (0.5, 0.5, 0.5), (1, 0, 0)] + [(x, 0, 0) for x in (0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90)]
KNAMES = ['G', 'L', 'X'] + [f'{x:.2f}X' for x in (0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90)]


def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def pw_input(calc, a_bohr, dft, extra_sys='', kblock="K_POINTS automatic\n8 8 8 1 1 1\n"):
    return f"""&CONTROL
  calculation = '{calc}', prefix = 'si', outdir = './tmp', pseudo_dir = '{PSEUDO}'
/
&SYSTEM
  ibrav = 2, celldm(1) = {a_bohr:.6f}, nat = 2, ntyp = 1, ecutwfc = 60.0,
  nr1 = {NR}, nr2 = {NR}, nr3 = {NR},
{extra_sys}  input_dft = '{dft}'
/
&ELECTRONS
  conv_thr = 1.0d-10, mixing_beta = 0.5
/
ATOMIC_SPECIES
Si 28.086 Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS alat
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
{kblock}"""


def run_pw(d, inp, out, model):
    env = dict(os.environ, OMP_NUM_THREADS='4')
    if model:
        env['NXC_MODEL'] = model
    with open(os.path.join(d, inp)) as fin, open(os.path.join(d, out), 'w') as fout:
        subprocess.run([PW], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=d, env=env)
    return open(os.path.join(d, out)).read()


def parse_bands(txt):
    """{k-index: [eigenvalues (eV)]} from an nscf output."""
    bands = []
    for m in re.finditer(r'k =\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+)\s*\(\s*\d+ PWs\)\s+bands \(ev\):\s*\n\n((?:\s*[-\d.]+)+)', txt):
        bands.append([float(x) for x in m.group(4).split()])
    return bands


def job(label, a_label, a_A):
    dft, model, _ = FUNCTIONALS[label]
    d = os.path.join(RUNS, slug(label), a_label)
    os.makedirs(d, exist_ok=True)
    out_nscf = os.path.join(d, 'nscf.out')
    if not (os.path.exists(out_nscf) and 'JOB DONE' in open(out_nscf).read()):
        open(os.path.join(d, 'scf.in'), 'w').write(pw_input('scf', a_A / BOHR_A, dft))
        scf = run_pw(d, 'scf.in', 'scf.out', model)
        if 'convergence has been achieved' not in scf:
            return dict(functional=label, a_label=a_label, a_A=a_A, error='scf not converged')
        kblock = f"K_POINTS tpiba\n{len(KPTS)}\n" + '\n'.join(f'{k[0]:.4f} {k[1]:.4f} {k[2]:.4f} 1.0' for k in KPTS) + '\n'
        open(os.path.join(d, 'nscf.in'), 'w').write(pw_input('nscf', a_A / BOHR_A, dft, '  nbnd = 8, nosym = .true.,\n', kblock))
        run_pw(d, 'nscf.in', 'nscf.out', model)
        subprocess.run(['rm', '-rf', os.path.join(d, 'tmp')])
    txt = open(out_nscf).read()
    bands = parse_bands(txt)
    if len(bands) != len(KPTS):
        return dict(functional=label, a_label=a_label, a_A=a_A, error=f'parsed {len(bands)} k-points')
    e = np.array(bands)                      # (nk, nbnd); Si: 4 valence bands
    vbm = e[:, 3].max()
    ivb = int(e[:, 3].argmax())
    cbm = e[:, 4].min()
    icb = int(e[:, 4].argmin())
    return dict(functional=label, a_label=a_label, a_A=a_A,
                gap_indirect_eV=float(cbm - vbm), vbm_at=KNAMES[ivb], cbm_at=KNAMES[icb],
                gap_direct_G_eV=float(e[0, 4] - e[0, 3]), gap_X_eV=float(e[2, 4] - e[2, 3]),
                gap_L_eV=float(e[1, 4] - e[1, 3]),
                bands={KNAMES[i]: [float(x) for x in e[i]] for i in range(len(KPTS))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    jobs = []
    for label, (_, _, a0) in FUNCTIONALS.items():
        jobs.append((label, 'a0', a0))
        jobs.append((label, 'aexp', A_EXP))
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda j: job(*j), jobs))
    os.makedirs(os.path.join(HERE, '..', 'results'), exist_ok=True)
    with open(os.path.join(HERE, '..', 'results', 'qe_si_gaps.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    lines = [f"Si band gaps (eV), SG15, 60 Ry, FFT {NR}^3; experiment: indirect 1.17, direct(G) 3.4",
             f"{'functional':40s} {'a (A)':>7s} {'indirect':>9s} {'VBM->CBM':>12s} {'direct G':>9s} {'gap X':>7s} {'gap L':>7s}"]
    for r in rows:
        if 'error' in r:
            lines.append(f"{r['functional']:40s} {r['a_A']:7.4f}  {r['error']}")
            continue
        lines.append(f"{r['functional']:40s} {r['a_A']:7.4f} {r['gap_indirect_eV']:9.3f} {r['vbm_at']+'->'+r['cbm_at']:>12s} "
                     f"{r['gap_direct_G_eV']:9.3f} {r['gap_X_eV']:7.3f} {r['gap_L_eV']:7.3f}")
    print('\n'.join(lines))
    open(os.path.join(HERE, '..', 'results', 'qe_si_gaps.log'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
