"""TorchScript modules with the libnxc grid-functional interface.

forward(x) with x a float64 tensor (N, k):
    columns rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb[, lapl_a, lapl_b, tau_a, tau_b]
    (k = 5 for GGA, 9 for MGGA); returns (N, 1) energy per particle eps_xc.
Derivatives are left to autograd (pylibnxc / libnxc back-propagate dot(eps, rho)).

The maths mirrors recipes_jax.py (keep in sync); see that module for the recipes.
No squeeze(): N = 1 works.  Points with rho <= RHO_MIN give eps = 0 with finite
gradients (double where).  TorchScript does not accept module-level float globals,
so constants live as attributes of the recipe modules (class _Physics).
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RHO_MIN = 1e-10
EPS_LOG = 1e-5


class MLP(nn.Module):
    """equinox.nn.MLP: Linear layers, activation after all but the last."""

    def __init__(self, weights: List[np.ndarray], biases: List[np.ndarray], approximate: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for W, b in zip(weights, biases):
            lin = nn.Linear(W.shape[1], W.shape[0]).double()
            with torch.no_grad():
                lin.weight.copy_(torch.as_tensor(np.asarray(W), dtype=torch.float64))
                lin.bias.copy_(torch.as_tensor(np.asarray(b), dtype=torch.float64))
            lin.weight.requires_grad_(False)
            lin.bias.requires_grad_(False)
            self.layers.append(lin)
        self.nlayers = len(weights)
        self.approximate = approximate        # 'tanh' (jax.nn.gelu default) or 'none' (erf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = 0
        for layer in self.layers:
            x = layer(x)
            if i < self.nlayers - 1:
                x = F.gelu(x, approximate=self.approximate)
            i += 1
        return x[:, 0]


class Enhancement(nn.Module):
    """Full enhancement factor F:  1 + I_a(t * MLP(features)) (mode 'lin'),
    1 + I_a(tanh(t)^2 * MLP) ('tanh2'), 1 + I_a(MLP) ('lob'), or MLP ('unct');
    I_a(y) = a sigmoid(y - ln(a-1)) - 1  (same expression order as the eqx LOB)."""

    def __init__(self, mlp: MLP, lob: float, mode: str):
        super().__init__()
        self.mlp = mlp
        self.lob = float(lob)
        self.shift = math.log(self.lob - 1) if self.lob > 1 else 0.0
        self.mode = mode

    def forward(self, feats: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        y = self.mlp(feats)
        if self.mode == 'unct':
            return y
        if self.mode == 'lin':
            y = t * y
        elif self.mode == 'tanh2':
            y = torch.tanh(t)**2 * y
        return 1 + (self.lob * torch.sigmoid(y - self.shift) - 1)


class _Physics(nn.Module):
    """Constants and shared density-functional pieces (LDA x, PW92 c, features)."""

    def __init__(self):
        super().__init__()
        pi = math.pi
        self.rho_min = RHO_MIN
        self.eps_log = EPS_LOG
        self.tiny_s2 = 1e-30
        self.tiny_sig = 1e-30
        self.pi = pi
        self.cx = -0.75 * (3 / pi)**(1 / 3)          # eps_x^LDA = cx rho^(1/3)
        self.s2c = 4 * (3 * pi**2)**(2 / 3)          # s^2 = sigma / (s2c rho^(8/3))
        self.cf = 0.3 * (3 * pi**2)**(2 / 3)         # tau_unif = cf rho^(5/3)
        self.ln10 = math.log(10.0)
        self.pw_A = [0.031090690869654895, 0.015545, 0.016887]
        self.pw_a1 = [0.21370, 0.20548, 0.11125]
        self.pw_b1 = [7.5957, 14.1189, 10.357]
        self.pw_b2 = [3.5876, 6.1977, 3.6231]
        self.pw_b3 = [1.6382, 3.3662, 0.88026]
        self.pw_b4 = [0.49294, 0.62517, 0.49671]
        self.pw_c0 = 1 / (2**(4 / 3) - 2)
        self.pw_fpp0 = 8 / 9 * self.pw_c0

    def pos(self, a: torch.Tensor) -> torch.Tensor:
        # '>=': gradient 1 at a == 0, i.e. the one-sided derivative from a -> 0+
        # (a spin channel that is exactly empty still gets its v_rho, as in Libxc)
        return torch.where(a >= 0, a, torch.zeros_like(a))

    def clamp_zeta(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.where(z > 1, torch.ones_like(z), z)
        return torch.where(z < -1, -torch.ones_like(z), z)

    def pw92_G(self, rs: torch.Tensor, k: int) -> torch.Tensor:
        A = self.pw_A[k]
        B = self.pw_b1[k] * torch.sqrt(rs) + self.pw_b2[k] * rs + self.pw_b3[k] * rs**1.5 \
            + self.pw_b4[k] * rs**2
        return -2 * A * (1 + self.pw_a1[k] * rs) * torch.log1p(1 / (2 * A * B))

    def rs_of(self, rho: torch.Tensor) -> torch.Tensor:
        return (3 / (4 * self.pi * rho))**(1 / 3)

    def eps_c_pw92(self, rho: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        rs = self.rs_of(rho)
        G0 = self.pw92_G(rs, 0)
        G1 = self.pw92_G(rs, 1)
        G2 = self.pw92_G(rs, 2)
        f = ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) * self.pw_c0
        z4 = zeta**4
        return G0 - G2 * f / self.pw_fpp0 * (1 - z4) + (G1 - G0) * f * z4

    def xcdiff_features(self, rho: torch.Tensor, zeta: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        s = torch.sqrt(s2 + self.tiny_s2)
        zp = 0.5 * ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3))
        return torch.stack([torch.log(rho**(1 / 3) + self.eps_log), torch.log(zp + self.eps_log),
                            (1 - torch.exp(-s2)) * torch.log1p(s)], dim=1)

    def alpha_feature(self, rho: torch.Tensor, sigma: torch.Tensor, tau: torch.Tensor,
                      ds: torch.Tensor) -> torch.Tensor:
        alpha = (tau - sigma / (8 * rho)) / (self.cf * rho**(5 / 3) * ds)
        return torch.log1p(torch.where(alpha > 0, alpha, torch.zeros_like(alpha)))


class SaraUnpolGGA(_Physics):
    """recipe sara_unpol_log10 (unpolarized training; refuses spin polarization)."""

    def __init__(self, fx: Enhancement, fc: Enhancement, zeta_tol: float = 1e-6):
        super().__init__()
        self.fx = fx
        self.fc = fc
        self.zeta_tol = zeta_tol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        zeta = torch.where(ok, (ra - rb) / rho_, torch.zeros_like(rho))
        if bool(torch.any(torch.abs(zeta) > self.zeta_tol)):
            raise RuntimeError('libnxc model trained on unpolarized densities: '
                               'spin-polarized input (|zeta| > zeta_tol) is not supported')
        sig = self.pos(x[:, 2] + 2 * x[:, 3] + x[:, 4])
        g = torch.sqrt(torch.where(ok, sig, torch.zeros_like(sig)) + self.tiny_sig)
        kf = (3 * self.pi**2 * rho_)**(1 / 3)
        s = g / (2 * kf * rho_)
        x0 = torch.log(rho_**(1 / 3) + self.eps_log) / self.ln10
        x1 = torch.log(s + 1) / self.ln10 * (1 - torch.exp(-s**2))
        Fx = self.fx(x1.unsqueeze(1), x1)
        Fc = self.fc(torch.stack([x0, x1], dim=1), x1)
        eps = self.cx * rho_**(1 / 3) * Fx + self.pw92_G(self.rs_of(rho_), 0) * Fc
        return torch.where(ok, eps, torch.zeros_like(eps)).unsqueeze(1)


class XCDiffPolGGA(_Physics):
    """recipe xcdiff_pol (spin-aware PBE clones)."""

    def __init__(self, fx: Enhancement, fc: Enhancement):
        super().__init__()
        self.fx = fx
        self.fc = fc

    def _ex(self, r: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
        ok = r > self.rho_min / 2
        r_ = torch.where(ok, r, torch.ones_like(r))
        sg_ = torch.where(ok, sg, torch.zeros_like(sg))
        r2 = 2 * r_
        X = self.xcdiff_features(r2, torch.zeros_like(r2), 4 * sg_ / (self.s2c * r2**(8 / 3)))
        Fx = self.fx(X[:, 2:3], X[:, 2])
        e = r_ * self.cx * r2**(1 / 3) * Fx
        return torch.where(ok, e, torch.zeros_like(e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        saa = x[:, 2]
        sab = x[:, 3]
        sbb = x[:, 4]
        ex = self._ex(ra, saa) + self._ex(rb, sbb)
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        zeta = self.clamp_zeta(torch.where(ok, (ra - rb) / rho_, torch.zeros_like(rho)))
        sig = torch.where(ok, self.pos(saa + 2 * sab + sbb), torch.zeros_like(rho))
        X = self.xcdiff_features(rho_, zeta, sig / (self.s2c * rho_**(8 / 3)))
        Fc = self.fc(X, X[:, 2])
        ec = torch.where(ok, rho_ * self.eps_c_pw92(rho_, zeta) * Fc, torch.zeros_like(rho))
        eps = torch.where(ok, (ex + ec) / rho_, torch.zeros_like(rho))
        return eps.unsqueeze(1)


class DemoMGGA(_Physics):
    """recipe demo_mgga_alpha (untrained meta-GGA used to validate vtau)."""

    def __init__(self, fx: Enhancement, fc: Enhancement):
        super().__init__()
        self.fx = fx
        self.fc = fc

    def _ex(self, r: torch.Tensor, sg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ok = r > self.rho_min / 2
        r_ = torch.where(ok, r, torch.ones_like(r))
        sg_ = torch.where(ok, sg, torch.zeros_like(sg))
        t_ = torch.where(ok, t, torch.ones_like(t))
        r2 = 2 * r_
        s4 = 4 * sg_
        X = self.xcdiff_features(r2, torch.zeros_like(r2), s4 / (self.s2c * r2**(8 / 3)))
        xa = self.alpha_feature(r2, s4, 2 * t_, torch.ones_like(r2))
        X = torch.cat([X, xa.unsqueeze(1)], dim=1)
        Fx = self.fx(X[:, 2:4], X[:, 2])
        e = r_ * self.cx * r2**(1 / 3) * Fx
        return torch.where(ok, e, torch.zeros_like(e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        saa = x[:, 2]
        sab = x[:, 3]
        sbb = x[:, 4]
        ta = x[:, 7]
        tb = x[:, 8]
        ex = self._ex(ra, saa, ta) + self._ex(rb, sbb, tb)
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        zeta = self.clamp_zeta(torch.where(ok, (ra - rb) / rho_, torch.zeros_like(rho)))
        sig = torch.where(ok, self.pos(saa + 2 * sab + sbb), torch.zeros_like(rho))
        tau = torch.where(ok, ta + tb, torch.ones_like(rho))
        ds = 0.5 * ((1 + zeta)**(5 / 3) + (1 - zeta)**(5 / 3))
        X = self.xcdiff_features(rho_, zeta, sig / (self.s2c * rho_**(8 / 3)))
        X = torch.cat([X, self.alpha_feature(rho_, sig, tau, ds).unsqueeze(1)], dim=1)
        Fc = self.fc(X, X[:, 2])
        ec = torch.where(ok, rho_ * self.eps_c_pw92(rho_, zeta) * Fc, torch.zeros_like(rho))
        eps = torch.where(ok, (ex + ec) / rho_, torch.zeros_like(rho))
        return eps.unsqueeze(1)


class MGGAX3(_Physics):
    """recipe mgga_x3: SmallData/mgga.py meta-GGA (r2SCAN clones, density-trained models).

    x3 = (1 - alpha)/(1 + alpha) with r2SCAN's regularized alpha = max(tau - tau_W, 0) /
    (tau_unif + 1e-3 tau_W); gate g = x2 + x3^2; optional one-electron factor on F_c.
    jnp.maximum / jnp.clip give gradient 1/2 at exact ties (e.g. tau == tau_W, |zeta| == 1);
    jmax0 / jclip11 reproduce that so the export matches the JAX model everywhere.
    """

    def __init__(self, fx: Enhancement, fc: Enhancement, sc_gate: bool):
        super().__init__()
        self.fx = fx
        self.fc = fc
        self.sc_gate = sc_gate
        self.eta = 1e-3

    def jmax0(self, d: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(d)
        return 0.5 * (torch.where(d >= 0, d, z) + torch.where(d > 0, d, z))

    def jclip11(self, z: torch.Tensor) -> torch.Tensor:
        one = torch.ones_like(z)
        lo = 0.5 * (torch.where(z >= -1, z, -one) + torch.where(z > -1, z, -one))
        return 0.5 * (torch.where(lo <= 1, lo, one) + torch.where(lo < 1, lo, one))

    def alpha_r2scan(self, rho: torch.Tensor, sigma: torch.Tensor, tau: torch.Tensor,
                     ds: torch.Tensor) -> torch.Tensor:
        tw = sigma / (8 * rho)
        tu = self.cf * rho**(5 / 3) * ds
        return self.jmax0(tau - tw) / (tu + self.eta * tw)

    def one_electron_factor(self, zeta_abs: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        alpha = self.jmax0((1 - x3) / (1 + x3))
        denom = self.jmax0(1 - alpha - 1e-6) + 1e-6          # jnp.clip(1 - alpha, 1e-6)
        fc = torch.where(alpha < 1.0, torch.exp(-0.64 * alpha / denom), torch.zeros_like(alpha))
        return 1 - zeta_abs**12 * fc

    def _ex(self, r: torch.Tensor, sg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ok = r > self.rho_min / 2
        r_ = torch.where(ok, r, torch.ones_like(r))
        sg_ = torch.where(ok, sg, torch.zeros_like(sg))
        t_ = torch.where(ok, t, torch.zeros_like(t))
        r2 = 2 * r_
        s4 = 4 * sg_
        t2 = 2 * t_
        s2 = s4 / (self.s2c * r2**(8 / 3))
        s = torch.sqrt(s2 + self.tiny_s2)
        x2 = (1 - torch.exp(-s2)) * torch.log1p(s)
        alpha = self.alpha_r2scan(r2, s4, t2, torch.ones_like(r2))
        x3 = (1 - alpha) / (1 + alpha)
        Fx = self.fx(torch.stack([x2, x3], dim=1), x2 + x3**2)
        e = r_ * self.cx * r2**(1 / 3) * Fx
        return torch.where(ok, e, torch.zeros_like(e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        saa = x[:, 2]
        sab = x[:, 3]
        sbb = x[:, 4]
        ta = x[:, 7]
        tb = x[:, 8]
        ex = self._ex(ra, saa, ta) + self._ex(rb, sbb, tb)
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        zeta = self.jclip11(torch.where(ok, (ra - rb) / rho_, torch.zeros_like(rho)))
        sig = torch.where(ok, saa + 2 * sab + sbb, torch.zeros_like(rho))
        tau = torch.where(ok, ta + tb, torch.zeros_like(rho))
        s2 = sig / (self.s2c * rho_**(8 / 3))
        s = torch.sqrt(s2 + self.tiny_s2)
        zp = 0.5 * ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3))
        ds = 0.5 * ((1 + zeta)**(5 / 3) + (1 - zeta)**(5 / 3))
        x0 = torch.log(rho_**(1 / 3) + self.eps_log)
        x1 = torch.log(zp + self.eps_log)
        x2 = (1 - torch.exp(-s2)) * torch.log1p(s)
        alpha = self.alpha_r2scan(rho_, sig, tau, ds)
        x3 = (1 - alpha) / (1 + alpha)
        Fc = self.fc(torch.stack([x0, x1, x2, x3], dim=1), x2 + x3**2)
        if self.sc_gate:
            Fc = Fc * self.one_electron_factor(torch.abs(zeta), x3)
        ec = torch.where(ok, rho_ * self.eps_c_pw92(rho_, zeta) * Fc, torch.zeros_like(rho))
        eps = torch.where(ok, (ex + ec) / rho_, torch.zeros_like(rho))
        return eps.unsqueeze(1)
