"""Validate the QE <-> libnxc hook with analytic twins (qe/twins/make_twins.py).

Every system is run twice with the same (patched) pw.x: once with the QE-native or
QE+Libxc functional and once with input_dft='NXC' + NXC_MODEL=<twin>.  Total energies,
forces and stresses must agree (differences come only from QE's own density/gradient
thresholds and, for spin-polarized PBE, QE's two slightly different PBE-c gamma values).

    python run_twins.py            (writes runs/twins/* and results/qe_twins.json)
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PW = os.environ.get('PW_X', 'pw.x')                       # pw.x built with -DQE_ENABLE_NXC=ON
PSEUDO = os.environ.get('QE_PSEUDO_DIR', 'pseudo')          # QE's test-suite pseudo/ directory
TWINS = os.path.join(HERE, 'twins')
RUNS = os.path.join(HERE, 'runs', 'twins')
RESULTS = os.path.join(HERE, '..', 'results')

SYSTEMS = {
    'Si': dict(system="""  ibrav = 2, celldm(1) = 10.26, nat = 2, ntyp = 1, ecutwfc = 30.0""",
               species="Si 28.086 Si_r.upf",
               positions="Si 0.00 0.00 0.00\nSi 0.27 0.25 0.24", kpoints="4 4 4 1 1 1"),
    'BN': dict(system="""  ibrav = 2, celldm(1) = 6.83, nat = 2, ntyp = 2, ecutwfc = 50.0""",
               species="B 10.811 B-PBE.upf\nN 14.007 N-PBE.upf",
               positions="B 0.00 0.00 0.00\nN 0.25 0.25 0.25", kpoints="4 4 4 1 1 1"),
    'Mo': dict(system="""  ibrav = 3, celldm(1) = 5.95, nat = 1, ntyp = 1, ecutwfc = 45.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02""",
               species="Mo 95.95 Mo-PBE.upf",
               positions="Mo 0.00 0.00 0.00", kpoints="6 6 6 1 1 1"),
    'Fe': dict(system="""  ibrav = 3, celldm(1) = 5.42, nat = 1, ntyp = 1, ecutwfc = 45.0,
  nspin = 2, starting_magnetization(1) = 0.5,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02""",
               species="Fe 55.845 Fe.pz-n-nc.UPF",
               positions="Fe 0.00 0.00 0.00", kpoints="8 8 8 1 1 1"),
    'N_atom': dict(system="""  ibrav = 1, celldm(1) = 12.0, nat = 1, ntyp = 1, ecutwfc = 40.0,
  nspin = 2, tot_magnetization = 3, occupations = 'fixed'""",
                   species="N 14.007 N-PBE.upf",
                   positions="N 0.00 0.00 0.00", kpoints=None),
}

CASES = [  # (twin model, reference input_dft, systems)
    ('LDA_X_SLATERTWIN', 'SLA-NOC-NOGX-NOGC', ['Si', 'Fe', 'N_atom']),
    ('GGA_XC_PBETWIN', 'PBE', ['Si', 'BN', 'Mo', 'Fe', 'N_atom']),
    ('MGGA_X_PKZBTWIN', 'XC-000I-000I-000I-000I-213L-000I', ['Si', 'Fe']),
    ('MGGA_X_LTATWIN', 'XC-000I-000I-000I-000I-201L-000I', ['Si']),
]


def write_input(path, sysname, dft, prefix):
    s = SYSTEMS[sysname]
    kp = f"K_POINTS automatic\n{s['kpoints']}" if s['kpoints'] else "K_POINTS gamma"
    txt = f"""&CONTROL
  calculation = 'scf', prefix = '{prefix}', outdir = './tmp', pseudo_dir = '{PSEUDO}',
  tprnfor = .true., tstress = .true.
/
&SYSTEM
{s['system']},
  input_dft = '{dft}'
/
&ELECTRONS
  conv_thr = 1.0d-11, mixing_beta = 0.5
/
ATOMIC_SPECIES
{s['species']}
ATOMIC_POSITIONS alat
{s['positions']}
{kp}
"""
    open(path, 'w').write(txt)


def parse(out):
    txt = open(out).read()
    res = {'converged': 'convergence has been achieved' in txt}
    m = re.findall(r'!\s+total energy\s+=\s+(-?\d+\.\d+)', txt)
    res['E'] = float(m[-1]) if m else None
    forces = re.findall(r'atom\s+\d+\s+type\s+\d+\s+force =\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', txt)
    res['F'] = [[float(x) for x in f] for f in forces]
    m = re.search(r'total\s+stress.*\n(.*)\n(.*)\n(.*)\n', txt)
    res['S'] = [[float(x) for x in l.split()[:3]] for l in m.groups()] if m else None
    m = re.findall(r'convergence has been achieved in\s+(\d+) iterations', txt)
    res['iter'] = int(m[-1]) if m else None
    m = re.findall(r'xc contribution\s+=\s+(-?\d+\.\d+)', txt)
    res['Exc'] = float(m[-1]) if m else None
    m = re.findall(r'total magnetization\s+=\s+(-?\d+\.\d+)', txt)
    res['mag'] = float(m[-1]) if m else None
    return res


def run(sysname, dft, model, tag):
    d = os.path.join(RUNS, f'{sysname}__{tag}')
    os.makedirs(d, exist_ok=True)
    write_input(os.path.join(d, 'pw.in'), sysname, dft, tag)
    env = dict(os.environ, OMP_NUM_THREADS='4')
    if model:
        env['NXC_MODEL'] = os.path.join(TWINS, model)
    with open(os.path.join(d, 'pw.in')) as fin, open(os.path.join(d, 'pw.out'), 'w') as fout:
        subprocess.run([PW], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=d, env=env)
    subprocess.run(['rm', '-rf', os.path.join(d, 'tmp')])       # wavefunctions not needed
    return parse(os.path.join(d, 'pw.out'))


def maxdiff(a, b):
    if not a or not b:
        return None
    fa = [x for row in a for x in row]
    fb = [x for row in b for x in row]
    return max(abs(x - y) for x, y in zip(fa, fb))


def main():
    only = sys.argv[1:]
    rows = []
    for model, ref_dft, systems in CASES:
        for sysname in systems:
            if only and sysname not in only and model not in only:
                continue
            ref = run(sysname, ref_dft, None, 'ref_' + model)
            nxc = run(sysname, 'NXC', model, 'nxc_' + model)
            row = dict(system=sysname, twin=model, ref_dft=ref_dft, ref=ref, nxc=nxc,
                       dE_Ry=(nxc['E'] - ref['E']) if (nxc['E'] and ref['E']) else None,
                       dF=maxdiff(nxc['F'], ref['F']), dS_kbar=maxdiff(nxc['S'], ref['S']))
            rows.append(row)
            print(f"{sysname:7s} {model:18s} E_ref={ref['E']}  dE={row['dE_Ry']}  dF={row['dF']}  "
                  f"dS={row['dS_kbar']}  it={ref['iter']}/{nxc['iter']}  conv={ref['converged']}/{nxc['converged']}  "
                  f"mag={ref['mag']}/{nxc['mag']}",
                  flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'qe_twins.json'), 'w') as f:
        json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
