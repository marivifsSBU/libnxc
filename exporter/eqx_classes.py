"""Equinox model classes known to the exporter (leaf-compatible copies).

Only the pytree layout (field order, MLP shapes) and the call semantics matter for
loading `.eqx` checkpoints with `eqx.tree_deserialise_leaves`.  Sources:

* GGA_F{x,c}_G_transf[_lin]: Sara/pretraining_pbe_SM `sara/modules/models.py` (git HEAD).
  Inputs [rho, |grad rho|]; features (log10) computed inside the class.
  Unpolarized training.  `tests/test_exporter.py` checks these copies against git.
* GGA_F{x,c}_xcdiff[_unct]: spin-aware PBE clones (SIESTA fork Util/NXC; the call
  semantics follow SmallData/polclone.py).  Inputs are the transformed features
  (x0, x1, x2) with natural logs, computed by the caller (see recipes).
* MGGA_F{x,c}_demo: untrained meta-GGA test models defined here, used only to
  validate the tau path (vtau) of the exporter.
"""
import json

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update('jax_enable_x64', True)     # checkpoints are float64


class LOB(eqx.Module):
    limit: float

    def __init__(self, limit: float):
        self.limit = limit

    def __call__(self, x):
        return self.limit * jax.nn.sigmoid(x - jnp.log(self.limit - 1)) - 1


def _mlp(in_size, depth, nodes, seed):
    return eqx.nn.MLP(in_size=in_size, out_size=1, depth=depth, width_size=nodes,
                      activation=jax.nn.gelu, key=jax.random.PRNGKey(seed))


# ------------------------------------------------ Sara, unpolarized (models.py)

class GGA_Fc_G_transf_lin(eqx.Module):
    name: str = 'GGA_Fc_G_transf_lin'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        self.depth, self.nodes, self.seed, self.lob_lim = depth, nodes, seed, lob_lim
        self.net = _mlp(2, depth, nodes, seed)
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        eps = 1e-5
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1 / 3)
        s = (grad_rho / (2 * k_F * rho)).flatten()
        x0 = jnp.log10(rho**(1 / 3) + eps)
        x1 = jnp.log10(s + 1) * (1 - jnp.exp(-s**2))
        netinp = jnp.stack([x0, x1], axis=0).flatten()
        return 1 + self.lobf(x1 * self.net(netinp)).squeeze()


class GGA_Fc_G_transf(eqx.Module):
    name: str = 'GGA_Fc_G_transf'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        self.depth, self.nodes, self.seed, self.lob_lim = depth, nodes, seed, lob_lim
        self.net = _mlp(2, depth, nodes, seed)
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        eps = 1e-5
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1 / 3)
        s = (grad_rho / (2 * k_F * rho)).flatten()
        x0 = jnp.log10(rho**(1 / 3) + eps)
        x1 = jnp.log10(s + 1) * (1 - jnp.exp(-s**2))
        netinp = jnp.stack([x0, x1], axis=0).flatten()
        return 1 + self.lobf(jnp.tanh(x1)**2 * self.net(netinp)).squeeze()


class GGA_Fx_G_transf_lin(eqx.Module):
    name: str = 'GGA_Fx_G_transf_lin'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804):
        self.depth, self.nodes, self.seed, self.lob_lim = depth, nodes, seed, lob_lim
        self.net = _mlp(1, depth, nodes, seed)
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1 / 3)
        s = (grad_rho / (2 * k_F * rho)).flatten()
        x1 = jnp.log10(s + 1) * (1 - jnp.exp(-s**2))
        return 1 + self.lobf(x1 * self.net(x1)).squeeze()


class GGA_Fx_G_transf(eqx.Module):
    name: str = 'GGA_Fx_G_transf'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804):
        self.depth, self.nodes, self.seed, self.lob_lim = depth, nodes, seed, lob_lim
        self.net = _mlp(1, depth, nodes, seed)
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1 / 3)
        s = (grad_rho / (2 * k_F * rho)).flatten()
        x1 = jnp.log10(s + 1) * (1 - jnp.exp(-s**2))
        return 1 + self.lobf(jnp.tanh(x1)**2 * self.net(x1)).squeeze()


# ------------------------------------------------ spin-aware PBE clones (xcdiff)

class GGA_Fc_xcdiff(eqx.Module):
    name: str = 'GGA_Fc_xcdiff'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        self.depth, self.nodes, self.seed = depth, nodes, seed
        self.lob_lim = 2.0 if lob_lim is None else lob_lim
        self.net = _mlp(3, depth, nodes, seed)
        self.lobf = LOB(limit=self.lob_lim)

    def __call__(self, x):                  # x = (x0, x1, x2)
        return 1 + self.lobf(x[2] * self.net(x)).squeeze()


class GGA_Fx_xcdiff(eqx.Module):
    name: str = 'GGA_Fx_xcdiff'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804):
        self.depth, self.nodes, self.seed = depth, nodes, seed
        self.lob_lim = 1.804 if lob_lim is None else lob_lim
        self.net = _mlp(1, depth, nodes, seed)
        self.lobf = LOB(limit=self.lob_lim)

    def __call__(self, x):                  # uses x2 only
        x2 = x[2:3]
        return 1 + self.lobf(x2 * self.net(x2)).squeeze()


class GGA_Fc_xcdiff_unct(eqx.Module):
    name: str = 'GGA_Fc_xcdiff_unct'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=None):
        self.depth, self.nodes, self.seed = depth, nodes, seed
        self.net = _mlp(3, depth, nodes, seed)

    def __call__(self, x):
        return self.net(x).squeeze()


class GGA_Fx_xcdiff_unct(eqx.Module):
    name: str = 'GGA_Fx_xcdiff_unct'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=None):
        self.depth, self.nodes, self.seed = depth, nodes, seed
        self.net = _mlp(1, depth, nodes, seed)

    def __call__(self, x):
        return self.net(x[2:3]).squeeze()


# ------------------------------------------------ meta-GGA demo (untrained)

class MGGA_Fx_demo(eqx.Module):
    """F_x(x2, xa) = 1 + I_a(MLP(x2, xa)); xa = ln(1 + alpha)."""
    name: str = 'MGGA_Fx_demo'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804):
        self.depth, self.nodes, self.seed = depth, nodes, seed
        self.lob_lim = 1.804 if lob_lim is None else lob_lim
        self.net = _mlp(2, depth, nodes, seed)
        self.lobf = LOB(limit=self.lob_lim)

    def __call__(self, x):                  # x = (x0, x1, x2, xa); uses (x2, xa)
        return 1 + self.lobf(self.net(x[2:4])).squeeze()


class MGGA_Fc_demo(eqx.Module):
    """F_c(x0, x1, x2, xa) = 1 + I_a(MLP(x0, x1, x2, xa))."""
    name: str = 'MGGA_Fc_demo'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        self.depth, self.nodes, self.seed = depth, nodes, seed
        self.lob_lim = 2.0 if lob_lim is None else lob_lim
        self.net = _mlp(4, depth, nodes, seed)
        self.lobf = LOB(limit=self.lob_lim)

    def __call__(self, x):
        return 1 + self.lobf(self.net(x)).squeeze()


# ------------------------------------------------ SmallData meta-GGA (mgga.py, 2026-09-30)
# r2SCAN clones and density-trained models.  Features (natural logs): x0 = ln(rho^(1/3)+1e-5),
# x1 = ln(zeta'+1e-5), x2 = (1-e^{-s^2}) ln(1+s), x3 = (1-alpha)/(1+alpha) with r2SCAN's
# regularized alpha; gate g = x2 + x3^2; bounds 1.174 (F_x) and 2 (F_c).  Saved as a single
# MXC pytree: <dir>/xc.eqx + <dir>/xc.eqx.json {depth, nodes, seed[, sc_gate]}.

A_X_R2SCAN, A_C = 1.174, 2.0


class MGGA_Fx(eqx.Module):
    name: str = 'MGGA_Fx'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth=3, nodes=32, seed=42, lob_lim=A_X_R2SCAN):
        self.depth, self.nodes, self.seed, self.lob_lim = depth, nodes, seed, lob_lim
        self.net = _mlp(2, depth, nodes, seed)
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, x):                  # x = (x2, x3)
        g = x[0] + x[1]**2
        return 1 + self.lobf(g * self.net(x)).squeeze()


def one_electron_factor(zeta_abs, x3):
    """r2SCAN's one-electron condition: 1 - |zeta|^12 f_c(alpha), f_c = exp(-0.64 alpha/(1-alpha))."""
    alpha = jnp.clip((1 - x3) / (1 + x3), 0.0, None)
    fc = jnp.where(alpha < 1.0, jnp.exp(-0.64 * alpha / jnp.clip(1 - alpha, 1e-6, None)), 0.0)
    return 1 - zeta_abs**12 * fc


class MGGA_Fc(eqx.Module):
    name: str = 'MGGA_Fc'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    sc_gate: bool = eqx.field(static=True)      # static (not a serialized leaf), as in mgga.py
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth=3, nodes=32, seed=42, lob_lim=A_C, sc_gate=False):
        self.depth, self.nodes, self.seed, self.lob_lim, self.sc_gate = depth, nodes, seed, lob_lim, sc_gate
        self.net = _mlp(4, depth, nodes, seed)
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, x):                  # x = (x0, x1, x2, x3, |zeta|); the network sees x0..x3
        g = x[2] + x[3]**2
        F = 1 + self.lobf(g * self.net(x[:4])).squeeze()
        if self.sc_gate:
            F = F * one_electron_factor(x[4], x[3])
        return F


class MXC(eqx.Module):
    fx: eqx.Module
    fc: eqx.Module


def load_mxc(path):
    """Load an mgga.py checkpoint: <path> is the .eqx file, metadata in <path>.json."""
    jax.config.update('jax_enable_x64', True)
    with open(path + '.json') as f:
        meta = json.load(f)
    like = MXC(fx=MGGA_Fx(meta['depth'], meta['nodes'], meta['seed']),
               fc=MGGA_Fc(meta['depth'], meta['nodes'], meta['seed'] + 1, sc_gate=meta.get('sc_gate', False)))
    meta = dict(meta, name='MXC')
    return eqx.tree_deserialise_leaves(path, like=like), meta


CLASSES = {c.__name__: c for c in (
    GGA_Fc_G_transf_lin, GGA_Fc_G_transf, GGA_Fx_G_transf_lin, GGA_Fx_G_transf,
    GGA_Fc_xcdiff, GGA_Fx_xcdiff, GGA_Fc_xcdiff_unct, GGA_Fx_xcdiff_unct,
    MGGA_Fx_demo, MGGA_Fc_demo)}


def load_model(prefix):
    """Load <prefix>.eqx using the metadata in <prefix>.json."""
    jax.config.update('jax_enable_x64', True)
    with open(prefix + '.json') as f:
        meta = json.load(f)
    cls = CLASSES[meta['name'].strip()]
    like = cls(depth=meta['depth'], nodes=meta['nodes'], seed=meta['seed'],
               **({'lob_lim': meta['lob_lim']} if 'lob_lim' in meta else {}))
    return eqx.tree_deserialise_leaves(prefix + '.eqx', like=like), meta


def load_sources(sources, mdir=None):
    """Models dict {'fx', 'fc'} from a manifest's 'sources' entry (fx/fc prefixes or one MXC).
    Relative paths are resolved against the model directory `mdir`."""
    if mdir is not None:
        from exporter.archive import resolve_sources
        sources = resolve_sources(sources, mdir)
    if 'xc' in sources:
        xc, _ = load_mxc(sources['xc']['path'])
        return {'fx': xc.fx, 'fc': xc.fc}
    return {k: load_model(sources[k]['prefix'])[0] for k in ('fx', 'fc')}


def save_model(model, prefix):
    meta = {'depth': model.depth, 'nodes': model.nodes, 'seed': model.seed, 'name': model.name}
    if hasattr(model, 'lob_lim'):
        meta['lob_lim'] = model.lob_lim
    eqx.tree_serialise_leaves(prefix + '.eqx', model)
    with open(prefix + '.json', 'w') as f:
        json.dump(meta, f)
