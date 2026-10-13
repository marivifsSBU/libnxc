"""Equations of state in QE with NN functionals through libnxc vs QE's own functionals.

For each (system, functional): 7 lattice parameters (+-3 %), SCF with stress; 3rd-order
Birch-Murnaghan fit of E(V) -> a0, B0, B0'.  Also compares the pressure from the stress
tensor with -dE/dV of the fit (consistency of the NXC stress).

    python run_eos.py [--workers 2]     (writes runs/eos/* and results/qe_eos.json)
"""
import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
PW = os.environ.get('PW_X', 'pw.x')                       # pw.x built with -DQE_ENABLE_NXC=ON
PSEUDO = os.environ.get('QE_PSEUDO_DIR', 'pseudo')          # QE's test-suite pseudo/ directory
MODELS = os.path.join(HERE, '..', 'models')
SHIPPED = MODELS
RUNS = os.path.join(HERE, 'runs', 'eos')
RY_BOHR3_TO_GPA = 14710.507848260711
BOHR_A = 0.529177210903

SYSTEMS = {
    # Si_r.upf (ONCV) has a nonlinear core correction; Si.bhs (BHS, LDA-generated) has none,
    # which avoids QE's meta-GGA + NLCC inconsistency in the meta-GGA comparison
    'Si': dict(ibrav=2, a=10.33, nat=2, ecut=40, species="Si 28.086 Si_r.upf",
               positions="Si 0.00 0.00 0.00\nSi 0.25 0.25 0.25", k=8, extra="", vfac=0.25),
    'Si_bhs': dict(ibrav=2, a=10.20, nat=2, ecut=40, species="Si 28.086 Si.bhs",
                   positions="Si 0.00 0.00 0.00\nSi 0.25 0.25 0.25", k=8, extra="", vfac=0.25),
    # SG15 ONCV (no core corrections), the set for meta-GGA work; 60 Ry is converged for Si
    'Si_sg15': dict(ibrav=2, a=10.33, nat=2, ecut=60, species="Si 28.086 Si_ONCV_PBE-1.2.upf",
                    positions="Si 0.00 0.00 0.00\nSi 0.25 0.25 0.25", k=8, extra="", vfac=0.25,
                    pseudo_dir=os.environ.get('NXC_SG15_DIR', 'pseudos/sg15')),
    'BN': dict(ibrav=2, a=6.84, nat=2, ecut=80, species="B 10.811 B-PBE.upf\nN 14.007 N-PBE.upf",
               positions="B 0.00 0.00 0.00\nN 0.25 0.25 0.25", k=8, extra="", vfac=0.25),
    'Mo': dict(ibrav=3, a=5.97, nat=1, ecut=60, species="Mo 95.95 Mo-PBE.upf",
               positions="Mo 0.00 0.00 0.00", k=12,
               extra="  occupations = 'smearing', smearing = 'mv', degauss = 0.02,\n", vfac=0.5),
}

FUNCTIONALS = {  # label: (input_dft, NXC_MODEL or None)
    'PBE': ('PBE', None),
    'NN-PBE TL (exported)': ('NXC', os.path.join(MODELS, 'GGA_XC_PBECLONE_TL')),
    'NN-PBE Sara unpol (exported)': ('NXC', os.path.join(MODELS, 'GGA_XC_PBECLONE_UNPOL_S05R1')),
    'NN-PBE shipped': ('NXC', os.path.join(SHIPPED, 'GGA_XC_PBE')),
    'SCAN (libxc)': ('SCAN', None),
    'NN-SCAN shipped': ('NXC', os.path.join(SHIPPED, 'MGGA_XC_SCAN')),
    'XCDIFF shipped': ('NXC', os.path.join(SHIPPED, 'MGGA_XC_XCDIFF')),
    # SmallData meta-GGAs (recipe mgga_x3)
    'r2SCAN (libxc)': ('R2SCAN', None),
    'NN-r2SCAN clone (exported)': ('NXC', os.path.join(MODELS, 'MGGA_XC_R2SCAN_CLONE')),
    'NN-r2SCAN clone sc-gated (exported)': ('NXC', os.path.join(MODELS, 'MGGA_XC_R2SCAN_CLONE_SC')),
}
MGGA_X3_SET = ['PBE', 'r2SCAN (libxc)', 'NN-r2SCAN clone (exported)', 'NN-r2SCAN clone sc-gated (exported)',
               'density-trained d70_T1 (exported)', 'density-trained d70sc_T1 (exported)']
PLAN = [('Si', f) for f in FUNCTIONALS if f not in MGGA_X3_SET[1:]] + \
       [('Si_sg15', f) for f in MGGA_X3_SET] + \
       [('Si_bhs', f) for f in ('PBE', 'NN-PBE TL (exported)', 'SCAN (libxc)', 'NN-SCAN shipped',
                                'XCDIFF shipped')] + \
       [(s, f) for s in ('BN', 'Mo') for f in ('PBE', 'NN-PBE TL (exported)',
                                               'NN-PBE Sara unpol (exported)', 'NN-PBE shipped')]


def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def pw_input(sysname, a, dft, prefix):
    s = SYSTEMS[sysname]
    return f"""&CONTROL
  calculation = 'scf', prefix = '{prefix}', outdir = './tmp', pseudo_dir = '{s.get('pseudo_dir', PSEUDO)}',
  tstress = .true., tprnfor = .true.
/
&SYSTEM
  ibrav = {s['ibrav']}, celldm(1) = {a:.6f}, nat = {s['nat']}, ntyp = {len(s['species'].splitlines())},
  ecutwfc = {s['ecut']}.0,
{s['extra']}  input_dft = '{dft}'
/
&ELECTRONS
  conv_thr = 1.0d-10, mixing_beta = 0.5
/
ATOMIC_SPECIES
{s['species']}
ATOMIC_POSITIONS alat
{s['positions']}
K_POINTS automatic
{s['k']} {s['k']} {s['k']} 1 1 1
"""


def run_one(sysname, flabel, a):
    dft, model = FUNCTIONALS[flabel]
    d = os.path.join(RUNS, sysname, slug(flabel), f'a{a:.4f}')
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'pw.out')
    if not (os.path.exists(out) and '!    total energy' in open(out).read()):
        open(os.path.join(d, 'pw.in'), 'w').write(pw_input(sysname, a, dft, 'eos'))
        env = dict(os.environ, OMP_NUM_THREADS='4')
        if model:
            env['NXC_MODEL'] = model
        with open(os.path.join(d, 'pw.in')) as fin, open(out, 'w') as fout:
            subprocess.run([PW], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=d, env=env)
        subprocess.run(['rm', '-rf', os.path.join(d, 'tmp')])
    txt = open(out).read()
    E = re.findall(r'!\s+total energy\s+=\s+(-?\d+\.\d+)', txt)
    P = re.findall(r'total\s+stress.*P=\s*(-?\d+\.\d+)', txt)
    return dict(a=a, E=float(E[-1]) if E else None, P_kbar=float(P[-1]) if P else None,
                converged='convergence has been achieved' in txt)


def bm3(V, E0, V0, B0, Bp):
    eta = (V0 / V)**(2 / 3)
    return E0 + 9 * V0 * B0 / 16 * ((eta - 1)**3 * Bp + (eta - 1)**2 * (6 - 4 * eta))


def fit(points, vfac):
    a = np.array([p['a'] for p in points])
    E = np.array([p['E'] for p in points])
    V = vfac * a**3
    i = np.argmin(E)
    popt, _ = curve_fit(bm3, V, E, p0=[E[i], V[i], 0.01, 4.0], maxfev=20000)
    E0, V0, B0, Bp = popt
    # pressure from the fit: P = -dE/dV
    h = 1e-4 * V
    Pfit = -(bm3(V + h, *popt) - bm3(V - h, *popt)) / (2 * h) * RY_BOHR3_TO_GPA * 10  # kbar
    return dict(a0_bohr=float((V0 / vfac)**(1 / 3)), a0_A=float((V0 / vfac)**(1 / 3) * BOHR_A),
                B0_GPa=float(B0 * RY_BOHR3_TO_GPA), Bp=float(Bp), E0=float(E0),
                P_fit_kbar=[float(x) for x in Pfit],
                rms_fit_mRy=float(1e3 * np.sqrt(np.mean((bm3(V, *popt) - E)**2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    jobs = []
    for sysname, flabel in PLAN:
        a0 = SYSTEMS[sysname]['a']
        for f in np.linspace(0.97, 1.03, 7):
            jobs.append((sysname, flabel, round(a0 * f, 4)))
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda j: (j, run_one(*j)), jobs))
    table = {}
    for (sysname, flabel, a), r in res:
        table.setdefault((sysname, flabel), []).append(r)
    rows = []
    for (sysname, flabel), pts in table.items():
        pts = sorted(pts, key=lambda p: p['a'])
        ok = all(p['E'] is not None and p['converged'] for p in pts)
        row = dict(system=sysname, functional=flabel, points=pts, all_converged=ok)
        if ok:
            row.update(fit(pts, SYSTEMS[sysname]['vfac']))
            row['max_dP_stress_vs_fit_kbar'] = float(max(abs(p['P_kbar'] - q) for p, q in
                                                         zip(pts, row['P_fit_kbar'])))
        rows.append(row)
        print(f"{sysname:3s} {flabel:30s} a0 = {row.get('a0_A', float('nan')):.4f} A  "
              f"B0 = {row.get('B0_GPa', float('nan')):6.1f} GPa  B' = {row.get('Bp', float('nan')):.2f}  "
              f"|P_stress - P_fit| <= {row.get('max_dP_stress_vs_fit_kbar', float('nan')):.2f} kbar  conv={ok}",
              flush=True)
    with open(os.path.join(HERE, '..', 'results', 'qe_eos.json'), 'w') as f:
        json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
