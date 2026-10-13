"""Numerical gate: exported TorchScript model (through pylibnxc.GridFunc.compute, the code
path used in SCF) vs the JAX/equinox reference (recipes_jax.reference_outputs).

Inputs
  random_*   log-uniform densities in the physical region: rho in [1e-8, 1e2] plus
             sub-threshold points, s in [0, 6] (incl. s = 0), zeta in [-1, 1]
             (incl. 0 and +-1), non-collinear spin gradients (sigma_ab < 0 occurs),
             tau = tau_W + alpha tau_unif with alpha in [0, 4] (incl. 0 and 1)
  mol_*      PySCF grid densities (PBE, def2-SVP, grid level 3): H2O unpolarized,
             OH polarized; all grid points, including tiny/negative densities
  batch_1    single-point calls (N = 1)
Metric: max|torch - jax| / max|jax| per output (zk, vrho, vsigma, vtau); gate < tol.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                          # repository root (pylibnxc)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from exporter.recipes_jax import RECIPES, reference_outputs   # noqa: E402

PI = np.pi
KF = (3 * PI**2)**(1 / 3)
CF = 0.3 * (3 * PI**2)**(2 / 3)


def _unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _rho(rng, n):
    rho = 10**rng.uniform(-8, 2, n)
    k = n // 20
    rho[:k] = 10**rng.uniform(-14, -10, k)            # below RHO_MIN
    return rho


def random_unpol(n, mgga, seed=0):
    rng = np.random.default_rng(seed)
    rho = _rho(rng, n)
    s = rng.uniform(0, 6, n)
    s[rng.random(n) < 0.1] = 0.0
    sigma = (s * 2 * KF * rho**(4 / 3))**2
    inp = {'rho': rho, 'sigma': sigma}
    if mgga:
        alpha = rng.uniform(0, 4, n)
        u = rng.random(n)
        alpha[u < 0.1] = 0.0
        alpha[(u >= 0.1) & (u < 0.2)] = 1.0
        inp['tau'] = sigma / (8 * rho) + alpha * CF * rho**(5 / 3)
    return inp


def random_pol(n, mgga, seed=1, zeta_zero=False):
    rng = np.random.default_rng(seed)
    rho = _rho(rng, n)
    zeta = rng.uniform(-1, 1, n)
    u = rng.random(n)
    zeta[u < 0.1] = 0.0
    zeta[(u >= 0.1) & (u < 0.15)] = 1.0
    zeta[(u >= 0.15) & (u < 0.2)] = -1.0
    if zeta_zero:
        zeta[:] = 0.0
    ra, rb = rho * (1 + zeta) / 2, rho * (1 - zeta) / 2
    sa, sb = rng.uniform(0, 6, n), rng.uniform(0, 6, n)
    sa[rng.random(n) < 0.1] = 0.0
    ua, ub = _unit(rng, n), _unit(rng, n)
    col = rng.random(n) < 0.3
    ub[col] = ua[col]
    if zeta_zero:
        sb, ub = sa, ua
    # per-channel reduced gradient of the spin-scaled density 2 rho_s
    ga = sa * KF * (2 * ra)**(1 / 3) * 2 * ra
    gb = sb * KF * (2 * rb)**(1 / 3) * 2 * rb
    Ga, Gb = ga[:, None] * ua, gb[:, None] * ub
    inp = {'rho': np.stack([ra, rb]),
           'sigma': np.stack([np.sum(Ga * Ga, 1), np.sum(Ga * Gb, 1), np.sum(Gb * Gb, 1)])}
    if mgga:
        cfs = 0.3 * (6 * PI**2)**(2 / 3)
        aa, ab = rng.uniform(0, 4, n), rng.uniform(0, 4, n)
        if zeta_zero:
            ab = aa
        with np.errstate(divide='ignore', invalid='ignore'):
            ta = np.where(ra > 0, ga**2 / (8 * ra), 0.0) + aa * cfs * ra**(5 / 3)
            tb = np.where(rb > 0, gb**2 / (8 * rb), 0.0) + ab * cfs * rb**(5 / 3)
        inp['tau'] = np.stack([ta, tb])
    return inp


def molecular(mgga):
    from pyscf import gto, dft
    out = {}
    for label, atom, spin in [('mol_H2O_unpol', 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', 0),
                              ('mol_OH_pol', 'O 0 0 0; H 0 0 0.9697', 1)]:
        mol = gto.M(atom=atom, basis='def2-svp', spin=spin, verbose=0)
        mf = (dft.RKS if spin == 0 else dft.UKS)(mol)
        mf.xc = 'PBE'
        mf.grids.level = 3
        mf.kernel()
        ni = mf._numint
        ao = ni.eval_ao(mol, mf.grids.coords, deriv=1)
        dm = mf.make_rdm1()
        if spin == 0:
            r = ni.eval_rho(mol, ao, dm, xctype='MGGA', with_lapl=False)
            inp = {'rho': r[0], 'sigma': np.sum(r[1:4]**2, 0)}
            if mgga:
                inp['tau'] = r[4]
        else:
            ra = ni.eval_rho(mol, ao, dm[0], xctype='MGGA', with_lapl=False)
            rb = ni.eval_rho(mol, ao, dm[1], xctype='MGGA', with_lapl=False)
            inp = {'rho': np.stack([ra[0], rb[0]]),
                   'sigma': np.stack([np.sum(ra[1:4]**2, 0), np.sum(ra[1:4] * rb[1:4], 0),
                                      np.sum(rb[1:4]**2, 0)])}
            if mgga:
                inp['tau'] = np.stack([ra[4], rb[4]])
        out[label] = inp
    return out


def compare(ref, out):
    res = {}
    for k, r in ref.items():
        r = np.asarray(r, dtype=float)
        o = np.asarray(out[k], dtype=float)
        finite = bool(np.all(np.isfinite(o)))
        scale = float(np.max(np.abs(r))) or 1.0
        err = float(np.max(np.abs(o - r))) if finite else float('inf')
        res[k] = dict(max_abs=err, max_ref=scale, rel=err / scale, finite=finite)
    return res


def _slice(inp, idx):
    return {k: (v[..., idx] if v.ndim > 1 else v[idx]) for k, v in inp.items()}


def run_check(recipe, models, model_dir, tol=1e-10, n=4000):
    from pylibnxc.functional import LibNXCFunctional
    func = LibNXCFunctional(os.path.abspath(model_dir), kind='grid')
    func.safe_mode = False                  # do not mask NaNs with nan_to_num
    mgga = RECIPES[recipe]['family'] == 'MGGA'
    polarized = RECIPES[recipe]['polarized']

    cases = {'random_unpol': random_unpol(n, mgga)}
    if polarized:
        cases['random_pol'] = random_pol(n, mgga)
    else:
        cases['random_pol_zeta0'] = random_pol(n, mgga, zeta_zero=True)
    for label, inp in molecular(mgga).items():
        if polarized or label.endswith('unpol'):
            cases[label] = inp
    idx = np.array([7])
    cases['batch_1_unpol'] = _slice(cases['random_unpol'], idx)
    pol_key = 'random_pol' if polarized else 'random_pol_zeta0'
    cases['batch_1_pol'] = _slice(cases[pol_key], idx)

    report = dict(recipe=recipe, model_dir=os.path.basename(os.path.abspath(model_dir)), tol=tol, cases={})
    passed = True
    for label, inp in cases.items():
        ref = reference_outputs(recipe, models, inp)
        out = func.compute({k: np.ascontiguousarray(v) for k, v in inp.items()})
        res = compare(ref, out)
        ok = all(v['finite'] and v['rel'] < tol for v in res.values())
        passed &= ok
        report['cases'][label] = dict(npoints=int(np.asarray(inp['rho']).shape[-1]), passed=ok, **res)

    if not polarized:                        # must refuse genuinely polarized input
        try:
            func.compute(random_pol(64, mgga))
            refused = False
        except Exception as ex:              # torch.jit raises a RuntimeError subclass
            refused = 'unpolarized' in str(ex)
        report['cases']['refuses_polarized'] = dict(passed=refused)
        passed &= refused
    report['passed'] = bool(passed)
    return report


def summarize(report):
    lines = [f"check {report['recipe']}  ->  {'PASSED' if report['passed'] else 'FAILED'} (tol {report['tol']:g})"]
    for label, c in report['cases'].items():
        errs = '  '.join(f"{k}={c[k]['rel']:.1e}" for k in ('zk', 'vrho', 'vsigma', 'vtau') if k in c)
        npt = f"N={c['npoints']:<6d}" if 'npoints' in c else ' ' * 8
        lines.append(f"  {label:18s} {npt} {'ok ' if c['passed'] else 'BAD'} {errs}")
    return '\n'.join(lines)
