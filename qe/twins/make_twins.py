"""Analytic 'twin' functionals as libnxc TorchScript models, for validating the QE hook.

Each twin reproduces a functional that QE evaluates natively (or through Libxc), with QE's
own constants, so QE + libnxc(twin) must agree with QE alone to high precision.  Any
difference is a bug in the hook (units, conventions, spin, thresholds, stress/forces):

  LDA_X_SLATERTWIN   Slater exchange            vs QE input_dft='SLA-NOC-NOGX-NOGC'
  GGA_XC_PBETWIN     PBE (QE's pw/pw_spin, pbex, pbec)   vs input_dft='PBE'
  MGGA_X_LTATWIN     Libxc MGGA_X_LTA (tau only) vs input_dft='XC-000I-000I-000I-000I-201L-000I'
  MGGA_X_PKZBTWIN    Libxc MGGA_X_PKZB (rho, sigma, tau) vs 'XC-000I-000I-000I-000I-213L-000I'

    python make_twins.py [outdir]      (nxc env)

Conventions as libnxc models: forward(x (N,5|9)) -> eps_xc (N,1); tau in Libxc units.
"""
import json
import math
import os
import sys

import torch
import torch.nn as nn

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.rho_min = 1e-10
        self.pi = math.pi
        self.cx = -0.75 * (3 / math.pi)**(1 / 3)            # eps_x^unif = cx rho^(1/3)
        self.pi34 = 0.6203504908994                         # QE: (3/4pi)^(1/3), truncated
        self.kfac = 3.093667726280136                       # QE: (3 pi^2)^(1/3)

    def pos(self, a: torch.Tensor) -> torch.Tensor:
        return torch.where(a >= 0, a, torch.zeros_like(a))

    def ex_channel_pbe(self, r: torch.Tensor, sg: torch.Tensor, gga: bool) -> torch.Tensor:
        """Spin-scaled exchange energy per volume of one channel: E_x[2 r]/2."""
        ok = r > self.rho_min / 2
        r_ = torch.where(ok, r, torch.ones_like(r))
        r2 = 2 * r_
        e = r_ * self.cx * r2**(1 / 3)
        if gga:
            s4 = 4 * torch.where(ok, sg, torch.zeros_like(sg))
            kf = self.kfac * r2**(1 / 3)
            s2 = s4 / (2 * kf * r2)**2
            k = 0.804
            mu = 0.2195149727645171
            e = e * (1 + k - k / (1 + mu * s2 / k))
        return torch.where(ok, e, torch.zeros_like(e))


class SlaterTwin(_Base):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        z = torch.zeros_like(ra)
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        e = self.ex_channel_pbe(ra, z, False) + self.ex_channel_pbe(rb, z, False)
        return torch.where(ok, e / rho_, torch.zeros_like(rho)).unsqueeze(1)


class PBETwin(_Base):
    """QE's PBE: pw_spin (A = 0.031091, fz0 = 1.709921) + pbex + pbec (gamma of pbec)."""

    def __init__(self):
        super().__init__()
        self.ga = 0.0310906908696548950
        self.be = 0.06672455060314922
        self.xkf = 1.919158292677513
        self.xks = 1.128379167095513

    def G(self, rs: torch.Tensor, a: float, a1: float, b1: float, b2: float, b3: float,
          b4: float) -> torch.Tensor:
        rs12 = torch.sqrt(rs)
        om = 2 * a * (b1 * rs12 + b2 * rs + b3 * rs * rs12 + b4 * rs**2)
        return -2 * a * (1 + a1 * rs) * torch.log(1 + 1 / om)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        saa = self.pos(x[:, 2])
        sab = x[:, 3]
        sbb = self.pos(x[:, 4])
        ex = self.ex_channel_pbe(ra, saa, True) + self.ex_channel_pbe(rb, sbb, True)
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        zeta = torch.where(ok, (ra - rb) / rho_, torch.zeros_like(rho))
        zmax = 1.0 - 1e-6                     # QE gcc_spin: |zeta| <= 1 - rho_threshold_gga
        zeta = torch.where(zeta > zmax, torch.full_like(zeta, zmax), zeta)
        zeta = torch.where(zeta < -zmax, torch.full_like(zeta, -zmax), zeta)
        sig = torch.where(ok, self.pos(saa + 2 * sab + sbb), torch.zeros_like(rho))
        rs = self.pi34 / rho_**(1 / 3)
        epwc = self.G(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
        epwcp = self.G(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
        alpha = -self.G(rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
        fz = ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) / (2**(4 / 3) - 2)
        z4 = zeta**4
        ec = epwc + alpha * fz * (1 - z4) / 1.709921 + (epwcp - epwc) * fz * z4
        # PBE gradient correction H
        phi = 0.5 * ((1 + zeta)**(2 / 3) + (1 - zeta)**(2 / 3))
        kf = self.xkf / rs
        ks = self.xks * torch.sqrt(kf)
        t2 = sig / (2 * phi * ks * rho_)**2
        phi3 = phi**3
        expe = torch.exp(-ec / (phi3 * self.ga))
        af = self.be / self.ga / (expe - 1)
        y = af * t2
        xy = (1 + y) / (1 + y + y * y)
        H = phi3 * self.ga * torch.log(1 + self.be / self.ga * t2 * xy)
        ecorr = torch.where(ok, rho_ * (ec + H), torch.zeros_like(rho))
        return torch.where(ok, (ex + ecorr) / rho_, torch.zeros_like(rho)).unsqueeze(1)


class LTATwin(_Base):
    """Libxc MGGA_X_LTA: spin-scaled LDA exchange times (t/C_F)^(4/5), t = tau_s/rho_s^(5/3)
    per spin-scaled channel (tau in Libxc units)."""

    def __init__(self):
        super().__init__()
        self.cf = 0.3 * (3 * math.pi**2)**(2 / 3)

    def channel(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ok = r > self.rho_min / 2
        r_ = torch.where(ok, r, torch.ones_like(r))
        t_ = torch.where(ok, self.pos(t), torch.ones_like(t))
        r2 = 2 * r_
        t2 = 2 * t_
        f = (t2 / (self.cf * r2**(5 / 3)))**0.8
        e = r_ * self.cx * r2**(1 / 3) * f
        return torch.where(ok, e, torch.zeros_like(e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        e = self.channel(ra, x[:, 7]) + self.channel(rb, x[:, 8])
        return torch.where(ok, e / rho_, torch.zeros_like(rho)).unsqueeze(1)


class PKZBTwin(_Base):
    """Libxc MGGA_X_PKZB (Perdew-Kurth-Zupan-Blaha exchange), spin-scaled:
    F = 1 + k - k/(1 + x/k), x = 10/81 p + 146/2025 qb^2 - 73/405 qb p + (D + (10/81)^2/k) p^2,
    p = s^2, qb = 3 tau/(2 (3 pi^2)^(2/3) rho^(5/3)) - 9/20 - p/12, k = 0.804, D = 0.113."""

    def __init__(self):
        super().__init__()
        self.cf = 0.3 * (3 * math.pi**2)**(2 / 3)

    def channel(self, r: torch.Tensor, sg: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ok = r > self.rho_min / 2
        r_ = torch.where(ok, r, torch.ones_like(r))
        sg_ = torch.where(ok, self.pos(sg), torch.zeros_like(sg))
        t_ = torch.where(ok, self.pos(t), torch.ones_like(t))
        # Libxc (>= 5): sigma_s <= 8 rho_s tau_s, derivatives evaluated at the capped point
        sg_ = sg_ + (torch.minimum(sg_, 8 * r_ * t_) - sg_).detach()
        r2 = 2 * r_
        p = 4 * sg_ / (2 * self.kfac * r2**(1 / 3) * r2)**2
        # qb = 3 tau / (2 (3 pi^2)^(2/3) rho^(5/3)) - 9/20 - p/12   (= 0 for the uniform gas)
        qb = 1.5 * (2 * t_) / (self.kfac**2 * r2**(5 / 3)) - 9.0 / 20.0 - p / 12
        k = 0.804
        xx = 10 / 81 * p + 146 / 2025 * qb**2 - 73 / 405 * qb * p + (0.113 + (10 / 81)**2 / k) * p**2
        f = 1 + k - k / (1 + xx / k)
        e = r_ * self.cx * r2**(1 / 3) * f
        return torch.where(ok, e, torch.zeros_like(e))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0])
        rb = self.pos(x[:, 1])
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        e = self.channel(ra, x[:, 2], x[:, 7]) + self.channel(rb, x[:, 4], x[:, 8])
        return torch.where(ok, e / rho_, torch.zeros_like(rho)).unsqueeze(1)


def save(module, name, qe_twin):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    manifest = {'name': name, 'thresholds': 'internal', 'twin_of': qe_twin,
                'note': 'analytic twin for validating the QE-libnxc hook'}
    torch.jit.save(torch.jit.script(module.double().eval()), os.path.join(d, 'xc'),
                   _extra_files={'manifest.json': json.dumps(manifest)})
    with open(os.path.join(d, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print('wrote', d)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    save(SlaterTwin(), 'LDA_X_SLATERTWIN', "SLA-NOC-NOGX-NOGC")
    save(PBETwin(), 'GGA_XC_PBETWIN', "PBE")
    save(LTATwin(), 'MGGA_X_LTATWIN', "XC-000I-000I-000I-000I-201L-000I")
    save(PKZBTwin(), 'MGGA_X_PKZBTWIN', "XC-000I-000I-000I-000I-213L-000I")
