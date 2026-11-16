"""Probe: is the single-density Boussinesq RT model invariant under
(phi, y) -> (-phi, -y) (bubble <-> spike exchange)?  Machine-precision check."""
import sys, torch
sys.path.insert(0, "src")
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.rayleigh_taylor import RTConfig, rt_step

torch.set_num_threads(4)
torch.manual_seed(0)
NY, NX, LAM = 24, 16, 16
PERM = [0, 1, 4, 3, 2, 8, 7, 6, 5]   # c_y -> -c_y

cfg = RTConfig(lam=LAM, layers=NY / LAM, g=2e-4, atwood=1 / 3, tau=0.515, tau_phi=0.501,
               n_steps=200, sample_every=10, phase_scheme=__import__("os").environ.get("SCH","bgk"))
cfg.validate()

# a deliberately asymmetric, smooth, bounded state
y = torch.arange(NY).view(-1, 1).float()
x = torch.arange(NX).view(1, -1).float()
phi = 0.8 * torch.tanh((y - 0.35 * NY) / 2.0) + 0.15 * torch.sin(2 * torch.pi * x / NX + 0.3)
ux = 0.01 * torch.sin(2 * torch.pi * x / NX) * torch.cos(torch.pi * y / NY)
uy = 0.01 * torch.cos(2 * torch.pi * x / NX + 0.7) + 0.0 * y
rho = 1.0 + 0.02 * torch.sin(torch.pi * y / NY) + 0.0 * x
f = equilibrium(rho, ux, uy)

# mirrored twin: row j' = NY-1-j, populations permuted, phi and uy negated
def mirror(t8):
    return t8[PERM].flip(1)

def mirror2(t2):
    return t2.flip(0)

f2 = mirror(f).contiguous()
phi2 = -phi.flip(0).contiguous()
g = torch.zeros(9, NY, NX)
g2 = torch.zeros(9, NY, NX)
from tensorlbm.d2q9 import W
g = W.view(9, 1, 1) * phi.unsqueeze(0)
g2 = W.view(9, 1, 1) * phi2.unsqueeze(0)

for step in range(1, 201):
    f, g = rt_step(f, g, cfg, phi=phi)
    f2, g2 = rt_step(f2, g2, cfg, phi=phi2)
    phi = g.sum(0)
    phi2 = g2.sum(0)
    if step in (1, 2, 5, 50, 200):
        # compare only what the scheme must preserve
        dphi = (phi2 + phi.flip(0)).abs().max().item()
        rho_a, ux_a, uy_a = macroscopic(f)
        rho_b, ux_b, uy_b = macroscopic(f2)
        duy = (uy_b + uy_a.flip(0)).abs().max().item()
        drho = (rho_b - rho_a.flip(0)).abs().max().item()
        print(f"step {step:4d}  dphi={dphi:.3e}  duy={duy:.3e}  drho={drho:.3e}  "
              f"max|phi|={phi.abs().max():.3f} fin_a={torch.isfinite(f).all().item()} fin_b={torch.isfinite(f2).all().item()} maxu_a={float((ux_a**2+uy_a**2).sqrt().max()):.4f} maxu_b={float((ux_b**2+uy_b**2).sqrt().max()):.4f}")