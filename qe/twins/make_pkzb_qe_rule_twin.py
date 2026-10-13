"""Diagnostic: PKZB twin that mimics QE's spin-polarized Libxc meta-GGA threshold rule
(potential of channel s zeroed, energy kept, where rho_s<=1e-12, sigma_ss<=1e-24 or tau_s<=1e-12).
With it, ferromagnetic Fe agrees with QE+Libxc PKZB to 1e-8 Ry (see qe/README.md)."""
import sys, os, json, math, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from make_twins import PKZBTwin, save

class PKZBTwinQERule(PKZBTwin):
    """PKZB twin + QE's spin-polarized Libxc meta rule: potential of channel s zeroed
    (energy kept) where rho_s <= 1e-12 or sigma_ss <= 1e-24 or tau_s <= 1e-12."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ra = self.pos(x[:, 0]); rb = self.pos(x[:, 1])
        sa = x[:, 2]; sb = x[:, 4]; ta = x[:, 7]; tb = x[:, 8]
        ka = (ra > 1e-12) & (sa > 1e-24) & (ta > 1e-12)
        kb = (rb > 1e-12) & (sb > 1e-24) & (tb > 1e-12)
        ra = torch.where(ka, ra, ra.detach()); sa = torch.where(ka, sa, sa.detach()); ta = torch.where(ka, ta, ta.detach())
        rb = torch.where(kb, rb, rb.detach()); sb = torch.where(kb, sb, sb.detach()); tb = torch.where(kb, tb, tb.detach())
        rho = ra + rb
        ok = rho > self.rho_min
        rho_ = torch.where(ok, rho, torch.ones_like(rho))
        e = self.channel(ra, sa, ta) + self.channel(rb, sb, tb)
        rho_raw = self.pos(x[:, 0]) + self.pos(x[:, 1])
        # exc such that exc * (raw rho) keeps the energy and the gradient pattern of QE
        # exc * rho_raw = e, so the gradients are those of e with the masked inputs detached
        return torch.where(ok, e / torch.where(ok, rho_raw, torch.ones_like(rho)),
                           torch.zeros_like(rho)).unsqueeze(1)

save(PKZBTwinQERule(), 'MGGA_X_PKZBTWINQERULE', 'PKZB twin with QE polarized threshold rule (diagnostic)')
