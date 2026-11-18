"""Pilot 4: burst scan (neumann walls) -- run in background, poll the log."""
import math, time, sys
import torch
torch.set_num_threads(1)
from tensorlbm import cac_lbm2d as cac


def throated_channel(ny, nx, wall_t=2, throat_half=8, throat_span=(70, 80)):
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


CFG = dict(rho_l=1.0, rho_g=0.1, mu_l=0.5, mu_g=0.05, sigma=0.05, width=3.0,
           mobility=0.05, mu_law="harmonic", p0=0.01, gravity=(0.0, 0.0),
           gravity_mode="rho", axes="pp")
ny, nx = 44, 128
shape = (ny, nx)
mask = throated_channel(ny, nx)
Lp = 30.0
sig, drho, rt = CFG["sigma"], CFG["rho_l"] - CFG["rho_g"], 8.0
g_burst = sig / (rt * drho * Lp)
print(f"g_burst_pred={g_burst:.4e}", flush=True)

def front(sim):
    m = cac.macro(sim)
    phi = m["phi"]
    row = phi[ny // 2]
    idx = torch.nonzero(row > 0.5).flatten()
    if idx.numel() == 0:
        return -1, -1
    return int(idx.max()), int(idx.min())

for mult in [0.5, 0.9, 1.05, 1.5]:
    g = mult * g_burst
    sim = cac.CAC2D(shape, wall_mask=mask, wall_phi_policy="neumann", **CFG)
    sim.initialize(init_plug(shape, CFG["width"], 20.0, 50.0))
    sim.gravity = (g, 0.0)
    t0 = time.time()
    rec = []
    for k in range(12):
        cac.run(sim, 250, sample_every=0)
        d = cac.diagnostics(sim)
        lead, trail = front(sim)
        rec.append(f"{sim.step_count}:{lead}/{trail}/{d['u_max_fluid']:.1e}")
        if d["nan_f"]:
            rec.append("NAN")
            break
    print(f"g={mult:.2f}*gb ({g:.2e}) {rec} ({time.time()-t0:.0f}s)", flush=True)