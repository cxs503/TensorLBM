"""Pilot 3: stability variants."""
import math, time, sys
import torch
torch.set_num_threads(1)
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


CASES = {
    "throat_contact": dict(throat=True, policy="contact"),
    "throat_freeze": dict(throat=True, policy="freeze"),
    "throat_neumann": dict(throat=True, policy="neumann"),
    "flat_contact": dict(throat=False, policy="contact"),
    "flat_neumann": dict(throat=False, policy="neumann"),
}
name = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
g = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
c = CASES[name]
for mu_l, mu_g in ((0.5, 0.05), (0.2, 0.02)):
    CFG = dict(rho_l=1.0, rho_g=0.1, mu_l=mu_l, mu_g=mu_g, sigma=0.05, width=4.0,
               mobility=0.05, mu_law="harmonic", p0=0.01, gravity=(0.0, 0.0),
               gravity_mode="rho", axes="pp")
    ny, nx = 44, 128
    shape = (ny, nx)
    mask = throated_channel(ny, nx) if c["throat"] else None
    if mask is None:
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[:2, :] = True
        mask[ny - 2:, :] = True
    kw = {}
    if c["policy"] == "contact":
        kw = dict(wall_phi_policy="contact", contact_angle_deg=60.0)
    else:
        kw = dict(wall_phi_policy=c["policy"])
    sim = cac.CAC2D(shape, wall_mask=mask, **kw, **CFG)
    sim.initialize(init_plug(shape, CFG["width"], 20.0, 50.0))
    sim.gravity = (g, 0.0)
    out = []
    for k in range(steps // 100):
        cac.run(sim, 100, sample_every=0)
        d = cac.diagnostics(sim)
        out.append(f"{sim.step_count}:{d['u_max_fluid']:.2e}{'/NAN' if d['nan_f'] else ''}")
        if d["nan_f"]:
            break
    print(f"{name} mu={mu_l} g={g}: " + " ".join(out), flush=True)