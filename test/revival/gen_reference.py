"""Reference cases for the C++/Fortran tests, computed with pylibnxc (Python).

    python gen_reference.py <outdir> [extra_models_dir]

Writes <outdir>/<case>.bin and <outdir>/cases.txt ("case model level nspin expect").
Binary layout (little endian): int32 magic, level, nspin, np, nsig; then float64 arrays in
libnxc's blocked layout (grid index fastest): rho[ns*np], sigma[nsig*np] (level >= 1),
tau[ns*np] (level 2), exc[np], vrho[ns*np], vsigma[nsig*np] (level >= 1), vtau[ns*np] (level 2).
Reference values follow the C++ convention: points with rho_a + rho_b <= 1e-10 have
exc = v = 0.  expect = ok | refuse (the model must reject the input).
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
from pylibnxc.functional import LibNXCFunctional   # noqa: E402

MAGIC = 20260928
THRESH = 1e-10
KF = (3 * np.pi**2)**(1 / 3)
CF = 0.3 * (3 * np.pi**2)**(2 / 3)
CFS = 0.3 * (6 * np.pi**2)**(2 / 3)
LEVEL = {'LDA': 0, 'GGA': 1, 'MGGA': 2}


def random_inputs(level, pol, n, seed, zeta_zero=False, zeta_edge=False):
    rng = np.random.default_rng(seed)
    rho = 10**rng.uniform(-6, 2, n)
    rho[:n // 20] = 10**rng.uniform(-14, -10, n // 20)          # below threshold
    if not pol:
        inp = {'rho': rho}
        if level >= 1:
            s = rng.uniform(0, 4, n)
            s[rng.random(n) < 0.1] = 0.0
            inp['sigma'] = (s * 2 * KF * rho**(4 / 3))**2
        if level >= 2:
            inp['tau'] = inp['sigma'] / (8 * rho) + rng.uniform(0, 3, n) * CF * rho**(5 / 3)
        return inp
    zeta = np.zeros(n) if zeta_zero else rng.uniform(-0.95, 0.95, n)
    if zeta_edge:                     # fully polarized points: one spin channel exactly empty
        zeta = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    ra, rb = rho * (1 + zeta) / 2, rho * (1 - zeta) / 2
    inp = {'rho': np.stack([ra, rb])}
    if level >= 1:
        sa = rng.uniform(0, 4, n)
        sb = sa if zeta_zero else rng.uniform(0, 4, n)
        ua = rng.normal(size=(n, 3)); ua /= np.linalg.norm(ua, axis=1, keepdims=True)
        ub = ua if zeta_zero else rng.normal(size=(n, 3))
        ub = ub / np.linalg.norm(ub, axis=1, keepdims=True)
        ga = sa * KF * (2 * ra)**(1 / 3) * 2 * ra
        gb = sb * KF * (2 * rb)**(1 / 3) * 2 * rb
        Ga, Gb = ga[:, None] * ua, gb[:, None] * ub
        inp['sigma'] = np.stack([np.sum(Ga * Ga, 1), np.sum(Ga * Gb, 1), np.sum(Gb * Gb, 1)])
    if level >= 2:
        aa = rng.uniform(0, 3, n)
        ab = aa if zeta_zero else rng.uniform(0, 3, n)
        with np.errstate(divide='ignore', invalid='ignore'):
            inp['tau'] = np.stack([np.where(ra > 0, ga**2 / (8 * ra), 0) + aa * CFS * ra**(5 / 3),
                                   np.where(rb > 0, gb**2 / (8 * rb), 0) + ab * CFS * rb**(5 / 3)])
    return inp


def molecular_inputs(level, pol):
    try:
        from pyscf import gto, dft
    except ImportError:
        return None
    atom = 'O 0 0 0; H 0 0 0.9697' if pol else 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
    mol = gto.M(atom=atom, basis='def2-svp', spin=int(pol), verbose=0)
    mf = (dft.UKS if pol else dft.RKS)(mol)
    mf.xc = 'PBE'
    mf.grids.level = 2
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    dms = mf.make_rdm1()
    blocks = [mf._numint.eval_rho(mol, ao, d, xctype='MGGA', with_lapl=False)
              for d in (dms if pol else [dms])]
    if not pol:
        r = blocks[0]
        inp = {'rho': r[0], 'sigma': np.sum(r[1:4]**2, 0), 'tau': r[4]}
    else:
        a, b = blocks
        inp = {'rho': np.stack([a[0], b[0]]),
               'sigma': np.stack([np.sum(a[1:4]**2, 0), np.sum(a[1:4] * b[1:4], 0), np.sum(b[1:4]**2, 0)]),
               'tau': np.stack([a[4], b[4]])}
    keep = {0: ['rho'], 1: ['rho', 'sigma'], 2: ['rho', 'sigma', 'tau']}[level]
    inp = {k: inp[k] for k in keep}
    # C++ clips negative spin densities before the model; do the same for the reference
    inp['rho'] = np.where(inp['rho'] >= 0, inp['rho'], 0.0)
    return inp


def reference(model, inp, level):
    func = LibNXCFunctional(model, kind='grid')
    out = func.compute({k: np.ascontiguousarray(v) for k, v in inp.items()})
    rho = inp['rho']
    rtot = rho if rho.ndim == 1 else rho.sum(0)
    off = rtot <= THRESH
    res = {'exc': np.where(off, 0.0, out['zk'])}
    for k, key in (('vrho', 'vrho'), ('vsigma', 'vsigma'), ('vtau', 'vtau')):
        if key in out:
            v = np.asarray(out[key])
            v = v.T if v.ndim == 2 else v          # pylibnxc returns (N, ns) -> (ns, N)
            res[k] = np.where(off, 0.0, v)
    return res


def write_case(path, level, pol, inp, ref):
    ns = 2 if pol else 1
    np_ = inp['rho'].shape[-1]
    nsig = (3 if pol else 1) if level >= 1 else 0
    with open(path, 'wb') as f:
        f.write(struct.pack('<5i', MAGIC, level, ns, np_, nsig))
        arrays = [inp['rho']] + ([inp['sigma']] if level >= 1 else []) + ([inp['tau']] if level >= 2 else [])
        arrays += [ref['exc'], ref['vrho']] + ([ref['vsigma']] if level >= 1 else []) + \
                  ([ref['vtau']] if level >= 2 else [])
        for a in arrays:
            f.write(np.ascontiguousarray(a, dtype='<f8').tobytes())


def main(outdir, extra=None):
    os.makedirs(outdir, exist_ok=True)
    shipped = os.path.join(HERE, '..', '..', 'models')
    models = [(os.path.join(shipped, m), True) for m in
              ('LDA_X_NUEG', 'GGA_XC_PBE', 'MGGA_XC_SCAN', 'MGGA_XC_XCDIFF')]
    if extra and os.path.isdir(extra):
        for m in sorted(os.listdir(extra)):
            if os.path.exists(os.path.join(extra, m, 'xc')):
                models.append((os.path.join(extra, m), not ('UNPOL' in m)))
    lines = []
    for mdir, polarized_ok in models:
        name = os.path.basename(mdir)
        level = LEVEL[name.split('_')[0]]
        cases = [('rand_unpol', False, random_inputs(level, False, 3000, 1)),
                 ('rand_pol', True, random_inputs(level, True, 3000, 2, zeta_zero=not polarized_ok))]
        if polarized_ok:
            cases.append(('rand_pol_edge', True, random_inputs(level, True, 1000, 4, zeta_edge=True)))
        for pol in (False, True):
            if pol and not polarized_ok:
                continue
            m = molecular_inputs(level, pol)
            if m is not None:
                cases.append(('mol_pol' if pol else 'mol_unpol', pol, m))
        for label, pol, inp in cases:
            case = f'{name}__{label}'
            write_case(os.path.join(outdir, case + '.bin'), level, pol, inp, reference(mdir, inp, level))
            lines.append(f'{case} {os.path.abspath(mdir)} {level} {2 if pol else 1} ok')
        if not polarized_ok:          # genuinely polarized input must be refused
            inp = random_inputs(level, True, 64, 3)
            ref = {'exc': np.zeros(64), 'vrho': np.zeros((2, 64)), 'vsigma': np.zeros((3, 64)),
                   'vtau': np.zeros((2, 64))}
            case = f'{name}__refuse_pol'
            write_case(os.path.join(outdir, case + '.bin'), level, True, inp, ref)
            lines.append(f'{case} {os.path.abspath(mdir)} {level} 2 refuse')
    with open(os.path.join(outdir, 'cases.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {len(lines)} cases to {outdir}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
