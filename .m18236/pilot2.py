"""Pilot 2: timing + single-rung behaviour."""
import math, time, sys
import torch
from tensorlbm import cac_lbm2d as cac


def throated_channel(ny, nx, wall_t=2, throat_half=6, throat_span=(70, 78)):
    mask = torch.zeros((ny, nx), dtype=torch.bool)
    mask[:wall_t, :] = True
    mask[ny - wall_t:, :] = True
    R = (ny - 2 * wall_t) / 2.0
    protrude = int(round(R - throat_half))
    x0, x1 = throat_span
    mask[wall_t:wall_t + protrude, x0:x1] = True
    mask[ny - wall_t - protrude:ny - wall_t, x0:x1] = True
    return mask


def init_plug(shape, width, x_lo, x_hi):
    ny, nx = shape
    x = torch.arange(nx, dtype=torch.float32).view(1, -1)
    lo = 0.5 * (1.0 + torch.tanh(2.0 * (x - x_lo) / width))
    hi = 0.5 * (1.0 + torch.tanh(2.0 * (x - x_hi) / width))
    return (lo * (1.0 - hi)).expand(*shape).contiguous()


CFG = dict(rho_l=1.0, rho_g=0.1, mu_l=0.5, mu_g=0.05, sigma=0.05, width=4.0,
           mobility=0.05, mu_law="harmonic", p0=0.01, gravity=(0.0, 0.0),
           gravity_mode="rho", axes="pp")
ny, nx = 44, 128
shape = (ny, nx)
mask = throated_channel(ny, nx)
g = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
sim = cac.CAC2D(shape, wall_mask=mask, wall_phi_policy="contact", contact_angle_deg=60.0, **CFG)
sim.initialize(init_plug(shape, CFG["width"], 20.0, 50.0))
sim.gravity = (g, 0.0)
t0 = time.time()
for k in range(steps // 100):
    cac.run(sim, 100, sample_every=0)
    d = [cac.diagnostics(sim)]
    m = cac.macro(sim)
    phi = m["phi"]
    row = phi[ny // 2]
    liq = row > 0.5
    idx = torch.nonzero(liq).flatten()
    lead, trail = (int(idx.max()), int(idx.min())) if idx.numel() else (-1, -1)
    print(f"step {sim.step_count:5d} u_max={d[-1]['u_max_fluid']:.3e} lead={lead:3d} trail={trail:3d} "
          f"t={time.time()-t0:.1f}s", flush=True)
print("ms/step", (time.time() - t0) / steps * 1e3)