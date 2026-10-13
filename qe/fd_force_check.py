"""Finite-difference check of forces in QE, native vs libnxc twins (Si, one atom displaced).

F_x(atom 2) printed by QE vs -(E(x+h) - E(x-h)) / 2h.  With a meta-GGA and nonlinear core
correction, QE's force_cc uses v_xc (LDA/GGA part only), so its forces miss the NLCC
meta-GGA term; this shows that libnxc reproduces QE's behaviour in both cases.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_twins as rt

H = 0.001        # displacement in alat units
ALAT = 10.26


def energy_force(dft, model, x, tag):
    rt.SYSTEMS['Si_fd'] = dict(rt.SYSTEMS['Si'])
    rt.SYSTEMS['Si_fd']['positions'] = f"Si 0.00 0.00 0.00\nSi {x:.6f} 0.25 0.24"
    r = rt.run('Si_fd', dft, model, tag)
    return r['E'], r['F'][1][0] if r['F'] else None


rows = []
for label, dft, model in [('PBE native', 'PBE', None), ('PBE twin (NXC)', 'NXC', 'GGA_XC_PBETWIN'),
                          ('PKZB libxc', 'XC-000I-000I-000I-000I-213L-000I', None),
                          ('PKZB twin (NXC)', 'NXC', 'MGGA_X_PKZBTWIN')]:
    tag = re.sub(r'[^A-Za-z0-9]+', '_', label)
    e0, f0 = energy_force(dft, model, 0.27, tag + '_0')
    ep, _ = energy_force(dft, model, 0.27 + H, tag + '_p')
    em, _ = energy_force(dft, model, 0.27 - H, tag + '_m')
    ffd = -(ep - em) / (2 * H * ALAT)
    rows.append(dict(functional=label, F_printed=f0, F_fd=ffd, diff=f0 - ffd))
    print(f'{label:18s} F_x printed = {f0: .6f}  FD = {ffd: .6f}  diff = {f0 - ffd: .2e} Ry/bohr', flush=True)
json.dump(rows, open(os.path.join(rt.RESULTS, 'qe_fd_forces.json'), 'w'), indent=1)
