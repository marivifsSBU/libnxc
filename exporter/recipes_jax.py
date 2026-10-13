"""JAX reference implementations of the exported functionals.

Each recipe maps the libnxc spin-resolved variables (rho_a, rho_b, sigma_aa, sigma_ab,
sigma_bb, tau_a, tau_b) to the energy per particle eps_xc, evaluating the enhancement
factors with the original equinox models.  The torch side (`nxc_torch.py`) implements
the same maths independently; `check.py` compares the two including derivatives.

Conventions shared with nxc_torch.py (keep in sync):
  * negative spin densities are clamped to 0 with where(a >= 0, a, 0): gradient 1 at
    a == 0, so an exactly empty spin channel gets the one-sided derivative (a -> 0+);
  * points with rho <= RHO_MIN give eps = 0 (double-where, finite gradients);
  * eps is formed exactly as in the torch model (division by the clamped density);
    the energy is then sum(eps * (rho_a + rho_b)) with the raw inputs, as in pylibnxc.

Recipes
-------
sara_unpol_log10  unpolarized PBE clones (arXiv:2605.10331), GGA_F{x,c}_G_transf[_lin]:
                  eps = eps_x^LDA(rho) F_x(rho,|grad rho|) + eps_c^PW92(rho,0) F_c(rho,|grad rho|).
                  Unpolarized training -> refuses |zeta| > zeta_tol.
xcdiff_pol        spin-aware clones GGA_F{x,c}_xcdiff[_unct] (as SmallData/polclone.py):
                  x0 = ln(rho^(1/3)+1e-5), x1 = ln(zeta'+1e-5), x2 = (1-e^{-s^2}) ln(1+s);
                  exchange by spin scaling with F_x(x2); e_c = rho eps_c^PW92(rho,zeta) F_c(x0,x1,x2).
demo_mgga_alpha   untrained meta-GGA test model: xcdiff features + xa = ln(1+alpha),
                  exchange spin-scaled F_x(x2, xa), correlation F_c(x0,x1,x2,xa) with the
                  spin-scaled alpha of the total density.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

RHO_MIN = 1e-10
EPS_LOG = 1e-5
TINY_S2 = 1e-30        # s = sqrt(s^2 + TINY_S2)            (xcdiff, demo)
TINY_SIG = 1e-30       # |grad rho| = sqrt(sigma + TINY_SIG)  (sara_unpol)
PI = jnp.pi
CX = -0.75 * (3 / PI)**(1 / 3)          # eps_x^LDA = CX rho^(1/3)
S2_C = 4 * (3 * PI**2)**(2 / 3)         # s^2 = sigma / (S2_C rho^(8/3))
CF = 0.3 * (3 * PI**2)**(2 / 3)         # tau_unif = CF rho^(5/3)

PW92 = dict(A=(0.031090690869654895, 0.015545, 0.016887),
            a1=(0.21370, 0.20548, 0.11125),
            b1=(7.5957, 14.1189, 10.357),
            b2=(3.5876, 6.1977, 3.6231),
            b3=(1.6382, 3.3662, 0.88026),
            b4=(0.49294, 0.62517, 0.49671))


def _pw92_G(rs, k):
    A, a1, b1, b2, b3, b4 = (PW92[n][k] for n in ('A', 'a1', 'b1', 'b2', 'b3', 'b4'))
    B = b1 * jnp.sqrt(rs) + b2 * rs + b3 * rs**1.5 + b4 * rs**2
    return -2 * A * (1 + a1 * rs) * jnp.log1p(1 / (2 * A * B))


def eps_c_pw92(rho, zeta):
    rs = (3 / (4 * PI * rho))**(1 / 3)
    G0, G1, G2 = (_pw92_G(rs, k) for k in range(3))
    c0 = 1 / (2**(4 / 3) - 2)
    f = ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) * c0
    fpp0 = 8 / 9 * c0
    z4 = zeta**4
    return G0 - G2 * f / fpp0 * (1 - z4) + (G1 - G0) * f * z4


def eps_c_pw92_unpol(rho):
    return _pw92_G((3 / (4 * PI * rho))**(1 / 3), 0)


def _pos(a):
    return jnp.where(a >= 0, a, 0.0)


def _clamp_zeta(z):
    z = jnp.where(z > 1, 1.0, z)
    return jnp.where(z < -1, -1.0, z)


def xcdiff_features(rho, zeta, s2):
    s = jnp.sqrt(s2 + TINY_S2)
    zp = 0.5 * ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3))
    return jnp.stack([jnp.log(rho**(1 / 3) + EPS_LOG), jnp.log(zp + EPS_LOG),
                      (1 - jnp.exp(-s2)) * jnp.log1p(s)], axis=-1)


def _alpha_feature(rho, sigma, tau, ds=1.0):
    alpha = (tau - sigma / (8 * rho)) / (CF * rho**(5 / 3) * ds)
    return jnp.log1p(jnp.where(alpha > 0, alpha, 0.0))


# ------------------------------------------------------------------ recipes
# eps_*(models, ra, rb, saa, sab, sbb, ta, tb) -> energy per particle (N,)

def eps_sara_unpol(models, ra, rb, saa, sab, sbb, ta=None, tb=None):
    fx, fc = models['fx'], models['fc']
    ra, rb = _pos(ra), _pos(rb)
    rho = ra + rb
    ok = rho > RHO_MIN
    rho_ = jnp.where(ok, rho, 1.0)
    sig = _pos(saa + 2 * sab + sbb)
    g = jnp.sqrt(jnp.where(ok, sig, 0.0) + TINY_SIG)
    inp = jnp.stack([rho_, g], axis=-1)[..., None]          # (N, 2, 1) as in RXCModel
    Fx = jax.vmap(fx)(inp)
    Fc = jax.vmap(fc)(inp)
    lda_x = -3 / 4 * (3 / PI)**(1 / 3) * rho_**(1 / 3)       # LDA exchange as in the cloning code
    return jnp.where(ok, lda_x * Fx + eps_c_pw92_unpol(rho_) * Fc, 0.0)


def _ex_xcdiff_channel(fx, r, sg):
    ok = r > RHO_MIN / 2
    r_ = jnp.where(ok, r, 1.0)
    sg_ = jnp.where(ok, sg, 0.0)
    r2 = 2 * r_
    X = xcdiff_features(r2, jnp.zeros_like(r2), 4 * sg_ / (S2_C * r2**(8 / 3)))
    return jnp.where(ok, r_ * CX * r2**(1 / 3) * jax.vmap(fx)(X), 0.0)


def _ec_common(rho, ra, rb, saa, sab, sbb):
    ok = rho > RHO_MIN
    rho_ = jnp.where(ok, rho, 1.0)
    zeta = _clamp_zeta(jnp.where(ok, (ra - rb) / rho_, 0.0))
    sig = jnp.where(ok, _pos(saa + 2 * sab + sbb), 0.0)
    return ok, rho_, zeta, sig


def eps_xcdiff_pol(models, ra, rb, saa, sab, sbb, ta=None, tb=None):
    fx, fc = models['fx'], models['fc']
    ra, rb = _pos(ra), _pos(rb)
    ex = _ex_xcdiff_channel(fx, ra, saa) + _ex_xcdiff_channel(fx, rb, sbb)
    ok, rho_, zeta, sig = _ec_common(ra + rb, ra, rb, saa, sab, sbb)
    X = xcdiff_features(rho_, zeta, sig / (S2_C * rho_**(8 / 3)))
    ec = jnp.where(ok, rho_ * eps_c_pw92(rho_, zeta) * jax.vmap(fc)(X), 0.0)
    return jnp.where(ok, (ex + ec) / rho_, 0.0)


def _ex_demo_channel(fx, r, sg, t):
    ok = r > RHO_MIN / 2
    r_ = jnp.where(ok, r, 1.0)
    sg_ = jnp.where(ok, sg, 0.0)
    t_ = jnp.where(ok, t, 1.0)
    r2, s4, t2 = 2 * r_, 4 * sg_, 2 * t_
    X = xcdiff_features(r2, jnp.zeros_like(r2), s4 / (S2_C * r2**(8 / 3)))
    xa = _alpha_feature(r2, s4, t2)
    X = jnp.concatenate([X, xa[:, None]], axis=-1)
    return jnp.where(ok, r_ * CX * r2**(1 / 3) * jax.vmap(fx)(X), 0.0)


def eps_demo_mgga(models, ra, rb, saa, sab, sbb, ta, tb):
    fx, fc = models['fx'], models['fc']
    ra, rb = _pos(ra), _pos(rb)
    ex = _ex_demo_channel(fx, ra, saa, ta) + _ex_demo_channel(fx, rb, sbb, tb)
    ok, rho_, zeta, sig = _ec_common(ra + rb, ra, rb, saa, sab, sbb)
    tau = jnp.where(ok, ta + tb, 1.0)
    ds = 0.5 * ((1 + zeta)**(5 / 3) + (1 - zeta)**(5 / 3))
    X = xcdiff_features(rho_, zeta, sig / (S2_C * rho_**(8 / 3)))
    X = jnp.concatenate([X, _alpha_feature(rho_, sig, tau, ds)[:, None]], axis=-1)
    ec = jnp.where(ok, rho_ * eps_c_pw92(rho_, zeta) * jax.vmap(fc)(X), 0.0)
    return jnp.where(ok, (ex + ec) / rho_, 0.0)


# ---- mgga_x3: SmallData/mgga.py meta-GGA (r2SCAN clones, density-trained models).
# Written to follow mgga.py line by line, including its jnp.maximum / jnp.clip conventions
# (gradient 1/2 at ties), which nxc_torch.MGGAX3 emulates.
ETA_ALPHA = 1e-3
C_TAU = CF                                 # tau_unif = C_TAU rho^(5/3)


def _alpha_r2scan(rho, sigma, tau, zeta=None):
    tw = sigma / (8 * rho)
    tu = C_TAU * rho**(5 / 3)
    if zeta is not None:
        tu = tu * 0.5 * ((1 + zeta)**(5 / 3) + (1 - zeta)**(5 / 3))
    return jnp.maximum(tau - tw, 0.0) / (tu + ETA_ALPHA * tw)


def _x3_of(alpha):
    return (1 - alpha) / (1 + alpha)


def _features_x_mgga(rho2, sigma2, tau2):
    s2 = sigma2 / (S2_C * rho2**(8 / 3))
    s = jnp.sqrt(s2 + TINY_S2)
    return jnp.stack([(1 - jnp.exp(-s2)) * jnp.log1p(s), _x3_of(_alpha_r2scan(rho2, sigma2, tau2))], axis=-1)


def _features_c_mgga(rho, zeta, sigma, tau):
    s2 = sigma / (S2_C * rho**(8 / 3))
    s = jnp.sqrt(s2 + TINY_S2)
    zp = 0.5 * ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3))
    return jnp.stack([jnp.log(rho**(1 / 3) + EPS_LOG), jnp.log(zp + EPS_LOG), (1 - jnp.exp(-s2)) * jnp.log1p(s),
                      _x3_of(_alpha_r2scan(rho, sigma, tau, zeta)), jnp.abs(zeta)], axis=-1)


def eps_mgga_x3(models, ra, rb, saa, sab, sbb, ta, tb):
    fx, fc = models['fx'], models['fc']
    ra, rb = _pos(ra), _pos(rb)
    ex = 0.0
    for r, sg, t in ((ra, saa, ta), (rb, sbb, tb)):
        ok = r > RHO_MIN / 2
        r_ = jnp.where(ok, r, 1.0)
        sg_ = jnp.where(ok, sg, 0.0)
        t_ = jnp.where(ok, t, 0.0)
        X = _features_x_mgga(2 * r_, 4 * sg_, 2 * t_)
        ex = ex + jnp.where(ok, r_ * CX * (2 * r_)**(1 / 3) * jax.vmap(fx)(X), 0.0)
    rho = ra + rb
    ok = rho > RHO_MIN
    rho_ = jnp.where(ok, rho, 1.0)
    zeta = jnp.clip(jnp.where(ok, (ra - rb) / rho_, 0.0), -1.0, 1.0)
    sig = jnp.where(ok, saa + 2 * sab + sbb, 0.0)
    tau = jnp.where(ok, ta + tb, 0.0)
    X = _features_c_mgga(rho_, zeta, sig, tau)
    ec = jnp.where(ok, rho_ * eps_c_pw92(rho_, zeta) * jax.vmap(fc)(X), 0.0)
    return jnp.where(ok, (ex + ec) / rho_, 0.0)


RECIPES = {
    'sara_unpol_log10': dict(eps=eps_sara_unpol, family='GGA', polarized=False),
    'xcdiff_pol': dict(eps=eps_xcdiff_pol, family='GGA', polarized=True),
    'demo_mgga_alpha': dict(eps=eps_demo_mgga, family='MGGA', polarized=True),
    'mgga_x3': dict(eps=eps_mgga_x3, family='MGGA', polarized=True),
}


# ------------------------------------------------------------------ Libxc-convention outputs

_COMPILED = {}
PAD = 4096


def _compiled(recipe, polarized):
    """filter_jit-compiled (energy, exc, grads) for a recipe; models are an argument."""
    key = (recipe, polarized)
    if key not in _COMPILED:
        import equinox as eqx
        eps = RECIPES[recipe]['eps']
        mgga = RECIPES[recipe]['family'] == 'MGGA'

        def energy(r, s, t, models):
            if not polarized:    # libnxc splits the totals evenly
                args = (r / 2, r / 2, s / 4, s / 4, s / 4) + ((t / 2, t / 2) if mgga else ())
                rtot = r
            else:
                args = (r[0], r[1], s[0], s[1], s[2]) + ((t[0], t[1]) if mgga else ())
                rtot = r[0] + r[1]
            exc = eps(models, *args)
            return jnp.sum(exc * rtot), exc

        argnums = (0, 1, 2) if mgga else (0, 1)
        _COMPILED[key] = eqx.filter_jit(jax.value_and_grad(energy, argnums=argnums, has_aux=True))
    return _COMPILED[key]


def reference_outputs(recipe, models, inp):
    """exc, vrho, vsigma[, vtau] in the pylibnxc/Libxc layout for the input dict
    (keys rho, sigma[, tau]; unpolarized 1-d arrays or polarized (2|3, N)).
    Arrays are zero-padded to a multiple of PAD to limit recompilation."""
    mgga = RECIPES[recipe]['family'] == 'MGGA'
    rho = np.asarray(inp['rho'], dtype=float)
    polarized = rho.ndim == 2
    n = rho.shape[-1]
    npad = -(-n // PAD) * PAD

    def pad(a):
        a = np.asarray(a, dtype=float)
        out = np.zeros(a.shape[:-1] + (npad,))
        out[..., :n] = a
        return jnp.asarray(out)

    t = pad(inp['tau']) if mgga else jnp.zeros(1)
    (_, exc), grads = _compiled(recipe, polarized)(pad(rho), pad(inp['sigma']), t, models)
    out = {'zk': exc, 'vrho': grads[0], 'vsigma': grads[1]}
    if mgga:
        out['vtau'] = grads[2]
    return {k: np.asarray(v)[..., :n].T for k, v in out.items()}
