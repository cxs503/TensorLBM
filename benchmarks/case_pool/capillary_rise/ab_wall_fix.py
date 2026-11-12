#!/usr/bin/env python3
"""A/B of the two CAC wall treatment defects on the neutral-wall control.

Configuration: the CAPILLARY-RISE channel with a **neutral** wall
(``theta = 90 deg``), where Jurin's law predicts exactly zero rise.  Any rise is
therefore pure wall-treatment artefact, i.e. a free negative control for the two
defects fixed in this mission:

======================================  =================================
variant                                 what it disables
======================================  =================================
``old_both``                            both fixes (pre-mission ``main``)
``axes_fix``                            periodic-x policy restored only
``ghost_fix``                           normal-aware wetting ghost only
``fixed``                               both (= branch HEAD)
======================================  =================================

Reported per variant: ``h`` trace, liquid-mass drift, fluid-only ``u_max``.

    PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/ab_wall_fix.py \
        --steps 1500 --out artifacts/m15917/ab_wall_fix.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm import cac_lbm2d as cac
from tensorlbm.wetting import metrics as M

SHAPE = (120, 64)
GAP, BAFFLE, POOL, GRAVITY, SIGMA = 20, 2, 20, 1.0e-4, 0.03


def _old_shift(f, dy, dx, axes="pp"):
    """Pre-mission ``_shift``: only ``axes[0]`` is honoured ('wp' -> 'ww')."""
    if dy == 0 and dx == 0:
        return f
    if axes[0] == "p":
        return torch.roll(f, shifts=(-dy, -dx), dims=(0, 1))
    ny, nx = f.shape
    out = torch.zeros_like(f)
    ys = slice(max(dy, 0), ny + min(dy, 0))
    yd = slice(max(-dy, 0), ny + min(-dy, 0))
    xs = slice(max(dx, 0), nx + min(dx, 0))
    xd = slice(max(-dx, 0), nx + min(-dx, 0))
    if ys.stop - ys.start <= 0 or xs.stop - xs.start <= 0:
        return out
    out[yd, xd] = f[ys, xs]
    return out


def _old_ghost(self):
    """Pre-mission isotropic (orientation-blind) wetting ghost."""
    if self.wall_phi_policy == "freeze" or self.wall_mask is None:
        return
    target = self.wall_mask
    if not bool(target.any()):
        return
    phi = self.f.sum(dim=0)
    zero = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
    pf = torch.where(self.fluid_mask, phi, zero)
    cnt = cac.shifted_neighbors(self.fluid_mask.to(self.dtype), axes=self.axes).sum(dim=0)
    has = target & (cnt > 0)
    if not bool(has.any()):
        return
    base = cac.shifted_neighbors(pf, axes=self.axes).sum(dim=0) / cnt.clamp(min=1.0)
    ghost = cac.wall_contact_ghost(base, self.wall_contact_cos, self.width)
    if self.wall_contact_zero_net:
        ghost = ghost - (ghost - base)[has].mean()
    phi_new = torch.where(has, ghost, phi)
    feq = cac.f_equilibrium(phi_new, zero, self.wv, self.cf)
    self.f = torch.where(has.unsqueeze(0), feq, self.f)


def _build(theta_deg: float):
    ny, nx = SHAPE
    mask = cac.solid_plate(SHAPE, 2, "bottom") | cac.solid_plate(SHAPE, 2, "top")
    x_c = nx // 2
    x_left = x_c - GAP // 2 - BAFFLE
    x_right = x_c + (GAP - GAP // 2)
    mask[POOL : ny - 2, x_left : x_left + BAFFLE] = True
    mask[POOL : ny - 2, x_right : x_right + BAFFLE] = True
    sim = cac.CAC2D(
        SHAPE, rho_l=1.0, rho_g=0.1, mu_l=0.05, mu_g=0.005, sigma=SIGMA, width=4.0,
        mobility=0.05, p0=0.01, axes="wp", wall_mask=mask, wall_phi_policy="contact",
        contact_angle_deg=theta_deg, gravity=(0.0, -GRAVITY), gravity_mode="rho",
        dtype=torch.float32,
    )
    sim.initialize(cac.init_phi_layers(SHAPE, 4.0, y0=float(POOL)))
    return sim, x_left, x_right


def _sample(sim, x_left, x_right):
    phi = cac.macro(sim)["phi"].numpy()
    h_ch, h_ch_std = M.meniscus_height(phi, x_left + BAFFLE + 1, x_right - 1)
    h_res, h_res_std = M.meniscus_height(phi, 4, x_left - 2)
    d = cac.diagnostics(sim)
    mass = float(sim.f.sum(dim=0)[sim.fluid_mask].sum())
    return {"step": int(sim.step_count), "y_channel": float(h_ch), "y_reservoir": float(h_res),
            "h": float(h_ch - h_res), "u_max_fluid": float(d["u_max_fluid"]),
            "phi_min_fluid": float(d["phi_min_fluid"]), "phi_integral_fluid": mass}


def variant(name: str, steps: int, theta_deg: float = 90.0, sample_every: int = 250) -> dict:
    orig_shift, orig_ghost = cac._shift, cac.CAC2D.refresh_solid_ghosts
    if name in ("old_both", "ghost_fix"):      # periodic-x policy broken
        cac._shift = _old_shift
    if name in ("old_both", "axes_fix"):       # orientation-blind ghost
        cac.CAC2D.refresh_solid_ghosts = _old_ghost
    try:
        sim, x_left, x_right = _build(theta_deg)
        m0 = _sample(sim, x_left, x_right)["phi_integral_fluid"]
        traj = []
        for _ in range(steps // sample_every):
            cac.run(sim, sample_every, sample_every=0)
            traj.append(_sample(sim, x_left, x_right))
        traj = [t for t in traj if math.isfinite(t["h"])]
        tail = traj[-max(1, len(traj) // 4):]
        return {
            "variant": name, "steps": sim.step_count, "theta_deg": theta_deg,
            "h_equilibrium": float(np.mean([t["h"] for t in tail])),
            "h_spread_pct": float(100 * (max(t["h"] for t in tail) - min(t["h"] for t in tail))
                                  / abs(np.mean([t["h"] for t in tail]))),
            "mass_drift_pct": float(100 * (traj[-1]["phi_integral_fluid"] - m0) / m0),
            "u_max_fluid": float(np.mean([t["u_max_fluid"] for t in tail])),
            "phi_min_fluid": float(min(t["phi_min_fluid"] for t in traj)),
            "finite": len(traj) == steps // sample_every,
            "traj": traj,
        }
    finally:
        cac._shift, cac.CAC2D.refresh_solid_ghosts = orig_shift, orig_ghost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--theta", type=float, default=90.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "artifacts/ab_wall_fix.json")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = []
    for name in ("old_both", "axes_fix", "ghost_fix", "fixed"):
        t0 = time.time()
        r = variant(name, args.steps, theta_deg=args.theta)
        r["wall_time_s"] = time.time() - t0
        out.append(r)
        print(f"{name:10s} theta={args.theta:g} h={r['h_equilibrium']:+.3f} "
              f"mass_drift={r['mass_drift_pct']:+.2f}% u_max={r['u_max_fluid']:.4f} "
              f"phi_min={r['phi_min_fluid']:+.3f} finite={r['finite']} ({r['wall_time_s']:.0f}s)",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"shape": list(SHAPE), "gap": GAP, "gravity": GRAVITY,
                                    "sigma": SIGMA, "theta_deg": args.theta,
                                    "variants": out}, indent=2))
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())