"""Pilot: throated channel + liquid plug + body-force drive (CAC2D)."""
import math, time
import torch
from tensorlbm import cac_lbm2d as cac

torch.manual_seed(0)


def throated_channel(ny, nx, wall_t=2, throat_half=6, throat_span=(70, 78)):
    mask = torch.zeros((ny, nx), dtype=torch.bool)
    mask[:wall_t, :] = True
    mask[ny - wall_t:, :] = True
    H = ny - 2 * wall_t
    R = H / 2.0
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
sim = cac.CAC2D(shape, wall_mask=mask, wall_phi_policy="contact", contact_angle_deg=60.0, **CFG)
sim.initialize(init_plug(shape, CFG["width"], 20.0, 50.0))
print("fluid cells", int((~mask).sum()), "R_wide", (ny - 4) / 2.0)

sig, d_rho = CFG["sigma"], CFG["rho_l"] - CFG["rho_g"]
Lp = 30.0
g_burst_pred = sig * (1.0 / 6.0 + math.cos(math.radians(60)) / 22.0) / (d_rho * Lp)
print("g_burst_pred", g_burst_pred)

for g in [0.0, 0.5 * g_burst_pred, 1.5 * g_burst_pred]:
    sim2 = cac.CAC2D(shape, wall_mask=mask, wall_phi_policy="contact", contact_angle_deg=60.0, **CFG)
    sim2.initialize(init_plug(shape, CFG["width"], 20.0, 50.0))
    sim2.gravity = (g, 0.0)
    t0 = time.time()
    hist = []
    for k in range(6):
        diags = cac.run(sim2, 500, sample_every=250)
        m = cac.macro(sim2)
        phi = m["phi"]
        row = phi[ny // 2]
        # front position: liquid where phi>0.5
        liq = row > 0.5
        idx = torch.nonzero(liq).flatten()
        lead = int(idx.max()) if idx.numel() else -1
        trail = int(idx.min()) if idx.numel() else -1
        hist.append((sim2.step_count, round(float(diags[-1]["u_max_fluid"]), 6), lead, trail))
    print("g=", g, "time", round(time.time() - t0, 1), hist)