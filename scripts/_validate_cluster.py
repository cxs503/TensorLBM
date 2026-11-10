"""Quick validation battery for the NACA0012 polar machinery (cluster smoke).

Runs a few short cases that expose the physics-level bugs that unit tests do
not catch: force symmetry at zero incidence, positive lift at positive
incidence, and the interpolated vs half-way bounce-back comparison.
"""
import math
import sys
import time

import torch

sys.path.insert(0, "src")

from tensorlbm.airfoil_polar import (  # noqa: E402
    AirfoilPolarConfig,
    apply_far_field,
    apply_wall,
    build_airfoil_grid,
    potential_flow_field,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402


def probe(alpha, chord=48, re_=1000.0, u=0.12, nsteps=3000, init="uniform",
          wall="interpolated"):
    torch.set_num_threads(8)
    cfg = AirfoilPolarConfig(chord=chord, alpha_deg=alpha, re=re_, u_in=u,
                             upstream_c=2.5, downstream_c=6.0, halfheight_c=3.0,
                             sponge_c=3.0, init=init, wall=wall)
    g = build_airfoil_grid(cfg)
    a = math.radians(alpha)
    xx = torch.arange(g.nx, dtype=torch.float32).view(1, -1).expand(g.ny, g.nx)
    x0 = int(g.nx - cfg.sponge_c * cfg.chord)
    wd = max(1, int(cfg.sponge_c * cfg.chord))
    sigma = ((xx - x0) / wd).clamp(0, 1) ** 2
    tau_field = cfg.tau * (1 + cfg.sponge_alpha * sigma)
    t0 = time.time()
    if init == "potential":
        ux0, uy0, _ = potential_flow_field(cfg, g)
        rho0 = (1.0 + 1.5 * (u**2 - (ux0**2 + uy0**2))).clamp(0.7, 1.4)
        f = equilibrium(rho0, ux0, uy0)
    else:
        rho0 = torch.ones((g.ny, g.nx))
        f = equilibrium(rho0, torch.full_like(rho0, u * math.cos(a)),
                        torch.full_like(rho0, u * math.sin(a)))
    tinit = time.time() - t0
    f = f.to(torch.float32)
    dpS = 0.5 * u * u * cfg.chord
    hist = []
    for step in range(1, nsteps + 1):
        fp = collide_mrt(f, tau=cfg.tau, tau_field=tau_field)
        fn = stream(fp)
        apply_far_field(fn, u * math.cos(a), u * math.sin(a), g.solid)
        fx, fy = apply_wall(fp, fn, g, scheme=wall)
        f = fn
        if step % 500 == 0:
            hist.append((step, (fy * math.cos(a) - fx * math.sin(a)) / dpS,
                         (fx * math.cos(a) + fy * math.sin(a)) / dpS))
    rho, ux, uy = macroscopic(f)
    print(f"  c={chord} Re={re_} a={alpha:+.1f} init={init} wall={wall} "
          f"tau={cfg.tau:.5f} cells={g.nx * g.ny} tinit={tinit:.1f}s "
          f"{(time.time() - t0) / nsteps:.4f}s/step "
          f"umax={float(torch.sqrt(ux**2 + uy**2).max()):.3f} "
          f"rhomax={float(rho.max()):.4f}", flush=True)
    for s, cl, cd in hist:
        print(f"     step {s}: cl={cl:+.4f} cd={cd:+.4f}", flush=True)
    return hist


if __name__ == "__main__":
    print("=== validation battery", flush=True)
    probe(0.0)
    probe(3.0)
    probe(3.0, init="potential")
    probe(6.0)
    probe(3.0, wall="staircase")