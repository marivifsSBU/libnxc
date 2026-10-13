"""Exporter tests (needs torch, jax, equinox, pyscf):

    PYTHONDONTWRITEBYTECODE=1 python -m pytest -q exporter/tests

1. Every exported model in models/ still passes its numerical gate against the equinox
   checkpoints bundled in models/<name>/source (always run).
2. Cross-checks against the independent training codes the models came from (skipped
   unless the paths are given):
   NXC_SARA_REPO  git checkout of the PBE-cloning code (class copies in eqx_classes.py);
   NXC_SMALLDATA  directory with polclone.py and mgga.py (independent JAX implementations).
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from exporter import eqx_classes                              # noqa: E402
from exporter.check import molecular, run_check, random_unpol  # noqa: E402

SARA_REPO = os.environ.get('NXC_SARA_REPO')
SMALLDATA = os.environ.get('NXC_SMALLDATA')
MODELS = os.path.join(ROOT, 'models')
needs_sara = pytest.mark.skipif(not SARA_REPO, reason='set NXC_SARA_REPO')
needs_smalldata = pytest.mark.skipif(not SMALLDATA, reason='set NXC_SMALLDATA')


def _exported(recipe=None):
    """Model directories written by the exporter (they carry a manifest with sources)."""
    out = []
    for mdir in sorted(glob.glob(os.path.join(MODELS, '*'))):
        mpath = os.path.join(mdir, 'manifest.json')
        if not os.path.exists(mpath):
            continue                         # shipped models of libnxc 0.x
        man = json.load(open(mpath))
        if 'sources' in man and (recipe is None or man['recipe'] == recipe):
            out.append(mdir)
    return out


@pytest.fixture(scope='module')
def sara_git_models():
    tmp = tempfile.mkdtemp()
    for m in ('models',):
        src = subprocess.check_output(['git', '-C', SARA_REPO, 'show', f'HEAD:sara/modules/{m}.py'], text=True)
        open(os.path.join(tmp, f'{m}.py'), 'w').write(src)
    sys.path.insert(0, tmp)
    import models as sara
    sys.path.remove(tmp)
    return sara


@needs_sara
@pytest.mark.parametrize('which', ['Fx', 'Fc'])
def test_sara_class_copies(sara_git_models, which):
    import jax
    src = os.path.join(MODELS, 'GGA_XC_PBECLONE_UNPOL_S05R1', 'source')
    prefix = f'{src}/GGA_{which}_G_transf_lin_d3_n16_s42_v_20000'
    theirs = sara_git_models.load_eqx_model(prefix)
    mine, _ = eqx_classes.load_model(prefix)
    inp = random_unpol(500, False, seed=3)
    x = np.stack([inp['rho'], np.sqrt(inp['sigma'])], axis=-1)[..., None]
    x = x[inp['rho'] > 1e-9]
    a = np.asarray(jax.vmap(theirs)(x))
    b = np.asarray(jax.vmap(mine)(x))
    assert np.array_equal(a, b)


def _polclone():
    sys.path.insert(0, SMALLDATA)
    import polclone
    sys.path.remove(SMALLDATA)
    return polclone


@needs_smalldata
@pytest.mark.parametrize('name,tail', [('GGA_XC_PBECLONE_TL', ''), ('GGA_XC_PBECLONE_UT', '_unct')])
def test_exported_vs_polclone(name, tail):
    import jax.numpy as jnp
    from pylibnxc.functional import LibNXCFunctional
    pc = _polclone()
    d = os.path.join(MODELS, name, 'source')
    xc = pc.XC(fx=pc.load_model(f'{d}/GGA_Fx_xcdiff{tail}_d3_n16_s42_v_20000'),
               fc=pc.load_model(f'{d}/GGA_Fc_xcdiff{tail}_d3_n16_s42_v_20000'))
    func = LibNXCFunctional(os.path.join(MODELS, name), kind='grid')
    func.safe_mode = False
    for label, inp in molecular(False).items():
        if inp['rho'].ndim == 2:             # polarized: polclone's spin-resolved pieces
            ra, rb = inp['rho']
            args = [jnp.asarray(a) for a in (ra, rb, *inp['sigma'])]
            e, vu, vd, vuu, vud, vdd = (np.asarray(a) for a in pc._pol_parts(xc, *args))
            rt = ra + rb
            ref = {'vrho': np.stack([vu, vd], 1), 'vsigma': np.stack([vuu, vud, vdd], 1)}
            mask = (ra > 1e-8) & (rb > 1e-8)  # away from clamps/guards (conventions differ there)
        else:
            e, vr, vs = (np.asarray(a) for a in
                         pc._unpol_parts(xc, jnp.asarray(inp['rho']), jnp.asarray(inp['sigma'])))
            rt = inp['rho']
            ref = {'vrho': vr, 'vsigma': vs}
            mask = rt > 1e-8
        ref['zk'] = np.where(rt > pc.RHO_MIN, e / np.where(rt > pc.RHO_MIN, rt, 1), 0)
        out = func.compute(inp)
        for k in ref:
            a, b = np.asarray(out[k])[mask], np.asarray(ref[k])[mask]
            rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
            assert rel < 1e-10, (label, k, rel)


@pytest.mark.parametrize('mdir', _exported(), ids=os.path.basename)
def test_exported_models_pass_gate(mdir):
    man = json.load(open(os.path.join(mdir, 'manifest.json')))
    rep = run_check(man['recipe'], eqx_classes.load_sources(man['sources'], mdir), mdir, n=500)
    assert rep['passed'], json.dumps(rep['cases'], indent=1)


@needs_smalldata
@pytest.mark.parametrize('mdir', _exported('mgga_x3'), ids=os.path.basename)
def test_exported_vs_mgga_py(mdir):
    """Exported mgga_x3 models vs SmallData/mgga.py itself (the code the models were trained
    and validated with), on H2O (closed shell) and OH (polarized) grids; generic points only."""
    import jax.numpy as jnp
    from pylibnxc.functional import LibNXCFunctional
    sys.path.insert(0, SMALLDATA)
    import mgga
    sys.path.remove(SMALLDATA)
    man = json.load(open(os.path.join(mdir, 'manifest.json')))
    xc = mgga.load(os.path.join(mdir, man['sources']['xc']['path']))
    func = LibNXCFunctional(mdir, kind='grid')
    func.safe_mode = False
    for label, inp in molecular(True).items():
        if inp['rho'].ndim == 1:          # closed shell: feed equal halves to both codes
            ra = rb = 0.5 * inp['rho']
            saa = sab = sbb = 0.25 * inp['sigma']
            ta = tb = 0.5 * inp['tau']
        else:
            ra, rb = inp['rho']
            saa, sab, sbb = inp['sigma']
            ta, tb = inp['tau']
        args = [jnp.asarray(a) for a in (ra, rb, saa, sab, sbb, ta, tb)]
        e, vu, vd, vuu, vud, vdd, vtu, vtd = (np.asarray(a) for a in mgga._pol_parts(xc, *args))
        rt = ra + rb
        ref = {'zk': np.where(rt > mgga.RHO_MIN, e / np.where(rt > mgga.RHO_MIN, rt, 1), 0),
               'vrho': np.stack([vu, vd], 1), 'vsigma': np.stack([vuu, vud, vdd], 1),
               'vtau': np.stack([vtu, vtd], 1)}
        out = func.compute({'rho': np.stack([ra, rb]), 'sigma': np.stack([saa, sab, sbb]),
                            'tau': np.stack([ta, tb])})
        mask = (ra > 1e-8) & (rb > 1e-8)
        for k in ref:
            a, b = np.asarray(out[k])[mask], np.asarray(ref[k])[mask]
            rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
            assert rel < 1e-10, (label, k, rel)
