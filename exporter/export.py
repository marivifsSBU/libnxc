"""equinox -> TorchScript exporter for libnxc grid functionals.

    # separate Fx / Fc checkpoints (PBE clones of arXiv:2605.10331)
    python -m exporter.export --recipe xcdiff_pol \
        --fx <dir>/GGA_Fx_xcdiff_d3_n16_s42_v_20000 --fc <dir>/GGA_Fc_xcdiff_d3_n16_s42_v_20000 \
        --name GGA_XC_PBECLONE_TL --outdir models
    # one MXC checkpoint (SmallData/mgga.py meta-GGAs: <dir>/xc.eqx + xc.eqx.json)
    python -m exporter.export --recipe mgga_x3 --xc <dir>/xc.eqx --name MGGA_XC_R2SCAN_CLONE

writes <outdir>/<name>/
    xc              TorchScript model (libnxc interface; manifest embedded as extra file)
    manifest.json   recipe, sources (+sha256), classes, activation, LOB limits, versions
    weights.npz     raw MLP weights (for native, libtorch-free evaluators)
    check.json      numerical gate vs the JAX/equinox model (exc, vrho, vsigma[, vtau])
The export fails (and the directory is marked FAILED) if the gate is not passed.
"""
import argparse
import datetime
import functools
import json
import os
import sys
import warnings

import numpy as np
import jax
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # repository root: exporter + pylibnxc

from exporter import eqx_classes, nxc_torch            # noqa: E402
from exporter.archive import bundle_sources, sha256, strip_archive   # noqa: E402
from exporter.recipes_jax import RECIPES              # noqa: E402

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# enhancement-factor form per eqx class ('lin': 1 + I_a(t * MLP), t given by the recipe)
MODES = {
    'GGA_Fx_G_transf_lin': 'lin', 'GGA_Fc_G_transf_lin': 'lin',
    'GGA_Fx_G_transf': 'tanh2', 'GGA_Fc_G_transf': 'tanh2',
    'GGA_Fx_xcdiff': 'lin', 'GGA_Fc_xcdiff': 'lin',
    'GGA_Fx_xcdiff_unct': 'unct', 'GGA_Fc_xcdiff_unct': 'unct',
    'MGGA_Fx_demo': 'lob', 'MGGA_Fc_demo': 'lob',
    'MGGA_Fx': 'lin', 'MGGA_Fc': 'lin',
}
RECIPE_CLASSES = {
    'sara_unpol_log10': ({'GGA_Fx_G_transf_lin', 'GGA_Fx_G_transf'},
                         {'GGA_Fc_G_transf_lin', 'GGA_Fc_G_transf'}),
    'xcdiff_pol': ({'GGA_Fx_xcdiff', 'GGA_Fx_xcdiff_unct'},
                   {'GGA_Fc_xcdiff', 'GGA_Fc_xcdiff_unct'}),
    'demo_mgga_alpha': ({'MGGA_Fx_demo'}, {'MGGA_Fc_demo'}),
    'mgga_x3': ({'MGGA_Fx'}, {'MGGA_Fc'}),
}
COMBINED_RECIPES = {'mgga_x3'}          # one MXC checkpoint instead of fx/fc prefixes


def gelu_mode(fn):
    """'tanh' for jax.nn.gelu (default approximate=True), 'none' for the erf form."""
    if fn is jax.nn.gelu:
        return 'tanh'
    if isinstance(fn, functools.partial) and fn.func is jax.nn.gelu and not fn.args:
        return 'tanh' if fn.keywords.get('approximate', True) else 'none'
    raise NotImplementedError(f'unsupported activation {fn!r}')


def extract_mlp(mlp):
    final = getattr(mlp, 'final_activation', None)
    if final is not None and getattr(final, '__name__', '') not in ('_identity', 'identity', '<lambda>'):
        # equinox uses an internal identity function; anything else is not supported
        x = np.linspace(-3, 3, 7)
        if not np.allclose(np.asarray(final(x)), x):
            raise NotImplementedError('MLP final_activation must be the identity')
    W = [np.asarray(l.weight, dtype=np.float64) for l in mlp.layers]
    b = [np.asarray(l.bias, dtype=np.float64) for l in mlp.layers]
    return W, b, gelu_mode(mlp.activation)


def enhancement(model, name):
    W, b, act = extract_mlp(model.net)
    lob = float(getattr(model, 'lob_lim', 0.0) or 0.0)
    return nxc_torch.Enhancement(nxc_torch.MLP(W, b, act), lob, MODES[name]), \
        dict(cls=name, mode=MODES[name], lob=lob, activation=f'gelu[{act}]',
             layers=[list(w.shape) for w in W]), (W, b)


def build(recipe, fx_model, fc_model):
    fx_name, fc_name = type(fx_model).__name__, type(fc_model).__name__
    okx, okc = RECIPE_CLASSES[recipe]
    if fx_name not in okx or fc_name not in okc:
        raise ValueError(f'recipe {recipe} expects Fx in {okx} and Fc in {okc}, '
                         f'got {fx_name}, {fc_name}')
    fx, fx_info, fx_w = enhancement(fx_model, fx_name)
    fc, fc_info, fc_w = enhancement(fc_model, fc_name)
    if recipe == 'sara_unpol_log10':
        module = nxc_torch.SaraUnpolGGA(fx, fc)
    elif recipe == 'xcdiff_pol':
        module = nxc_torch.XCDiffPolGGA(fx, fc)
    elif recipe == 'demo_mgga_alpha':
        module = nxc_torch.DemoMGGA(fx, fc)
    elif recipe == 'mgga_x3':
        fc_info['sc_gate'] = bool(fc_model.sc_gate)
        module = nxc_torch.MGGAX3(fx, fc, bool(fc_model.sc_gate))
    else:
        raise ValueError(recipe)
    module = module.double().eval()
    return torch.jit.script(module), dict(fx=fx_info, fc=fc_info), (fx_w, fc_w)




def export(recipe, name, outdir, fx_prefix=None, fc_prefix=None, xc_path=None, tol=1e-10, check=True):
    fam = RECIPES[recipe]['family']
    if name.split('_')[0] != fam:
        raise ValueError(f'model name must start with {fam}_ (libnxc reads the family from it)')
    if name.split('_')[-1] == 'HM':
        raise ValueError("names ending in '_HM' are reserved for Nagai's models in pylibnxc")
    if recipe in COMBINED_RECIPES:
        if not xc_path:
            raise ValueError(f'recipe {recipe} takes one combined checkpoint (--xc <dir>/xc.eqx)')
        xc, meta = eqx_classes.load_mxc(xc_path)
        fx_model, fc_model = xc.fx, xc.fc
        sources = {'xc': dict(path=os.path.abspath(xc_path), meta=meta, sha256=sha256(xc_path))}
    else:
        if not (fx_prefix and fc_prefix):
            raise ValueError(f'recipe {recipe} takes --fx and --fc checkpoint prefixes')
        fx_model, fx_meta = eqx_classes.load_model(fx_prefix)
        fc_model, fc_meta = eqx_classes.load_model(fc_prefix)
        sources = {'fx': dict(prefix=os.path.abspath(fx_prefix), meta=fx_meta, sha256=sha256(fx_prefix + '.eqx')),
                   'fc': dict(prefix=os.path.abspath(fc_prefix), meta=fc_meta, sha256=sha256(fc_prefix + '.eqx'))}
    scripted, info, (fx_w, fc_w) = build(recipe, fx_model, fc_model)

    mdir = os.path.join(outdir, name)
    os.makedirs(mdir, exist_ok=True)
    manifest = dict(
        name=name, recipe=recipe, family=fam, polarized=RECIPES[recipe]['polarized'],
        interface='libnxc grid: x (N,{}) float64 -> eps_xc (N,1)'.format(9 if fam == 'MGGA' else 5),
        # the model clamps densities and regularizes s itself (exact one-sided derivatives
        # at empty spin channels): libnxc skips its Libxc-style density/sigma floors
        thresholds='internal',
        sources=sources,
        networks=info, constants=dict(RHO_MIN=nxc_torch.RHO_MIN, EPS_LOG=nxc_torch.EPS_LOG),
        created=datetime.datetime.now().isoformat(timespec='seconds'),
        versions=dict(torch=torch.__version__, jax=jax.__version__,
                      equinox=__import__('equinox').__version__))
    with open(os.path.join(mdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    np.savez(os.path.join(mdir, 'weights.npz'),
             **{f'fx_W{i}': w for i, w in enumerate(fx_w[0])}, **{f'fx_b{i}': w for i, w in enumerate(fx_w[1])},
             **{f'fc_W{i}': w for i, w in enumerate(fc_w[0])}, **{f'fc_b{i}': w for i, w in enumerate(fc_w[1])})
    bundle_sources(mdir, manifest)       # checkpoints -> <mdir>/source, relative paths
    with open(os.path.join(mdir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    torch.jit.save(scripted, os.path.join(mdir, 'xc'),
                   _extra_files={'manifest.json': json.dumps(manifest)})
    strip_archive(os.path.join(mdir, 'xc'))   # drop debug pickles (absolute source paths)

    report = None
    if check:
        from exporter.check import run_check
        report = run_check(recipe, {'fx': fx_model, 'fc': fc_model}, mdir, tol=tol)
        with open(os.path.join(mdir, 'check.json'), 'w') as f:
            json.dump(report, f, indent=1)
        failed = os.path.join(mdir, 'FAILED')
        if not report['passed']:
            open(failed, 'w').write('numerical gate failed, see check.json\n')
            raise RuntimeError(f'export of {name} failed the numerical check (see {mdir}/check.json)')
        if os.path.exists(failed):
            os.remove(failed)
    return mdir, report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--recipe', required=True, choices=sorted(RECIPES))
    p.add_argument('--fx', help='Fx checkpoint prefix (without .eqx/.json)')
    p.add_argument('--fc', help='Fc checkpoint prefix (without .eqx/.json)')
    p.add_argument('--xc', help='combined MXC checkpoint (<dir>/xc.eqx; metadata in xc.eqx.json)')
    p.add_argument('--name', required=True, help='libnxc model name, e.g. GGA_XC_PBECLONE_TL')
    p.add_argument('--outdir', default=os.path.join(os.path.dirname(HERE), 'models'))
    p.add_argument('--tol', type=float, default=1e-10)
    p.add_argument('--no-check', action='store_true')
    a = p.parse_args()
    mdir, report = export(a.recipe, a.name, a.outdir, a.fx, a.fc, a.xc, a.tol, not a.no_check)
    print(f'exported {mdir}')
    if report:
        from exporter.check import summarize
        print(summarize(report))


if __name__ == '__main__':
    main()
