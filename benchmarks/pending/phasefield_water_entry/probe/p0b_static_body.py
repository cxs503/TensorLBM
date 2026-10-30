"""P0b probe — static solid straddling the air/water interface (contact-line gate).

Question this probe answers
---------------------------
Water entry forces a *solid boundary to cross a diffuse interface*.  The
library CAC model (``tensorlbm.cac_lbm``) has no wetting / contact-angle
boundary condition and no capillary force on the fluid-solid contact line, so
before any entry case is attempted we must know whether the simplest possible
configuration survives:

    closed walled tank + per-mass gravity + a static disk whose centre sits
    exactly on the flat interface.

Measured here (prerequisite gate, NOT a benchmark):

* NaN-free run,
* phase-field boundedness ``phi in [0, 1]`` in the fluid near the solid,
* spurious velocity ``max|u|`` (the contact line is the usual suspect),
* phase-field integral drift,
* interface height far from the body versus through the body (a static
  no-slip body must not pump the interface).

Nothing here is compared with experiment; the case is deliberately static.

Run
---
    PYTHONPATH=src python benchmarks/pending/phasefield_water_entry/probe/p0b_static_body.py \
        --out /tmp/p0b_static_body.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorlbm import cac_lbm as m


def wall_mask(shape: tuple[int, int, int]) -> torch.Tensor:
    w = torch.zeros(shape, dtype=torch.bool)
    w[0, :, :] = w[-1, :, :] = True
    w[:, 0, :] = w[:, -1, :] = True
    w[:, :, 0] = w[:, :, -1] = True
    return w


def disk_mask(shape: tuple[int, int, int], cy: float, cx: float, radius: float) -> torch.Tensor:
    nz, ny, nx = shape
    y = torch.arange(ny, dtype=torch.float32).view(1, -1, 1)
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, -1)
    disk2d = ((y - cy) ** 2 + (x - cx) ** 2) <= radius**2
    return disk2d.expand(nz, ny, nx).clone()


def interface_height(phi_row: torch.Tensor) -> float:
    for y in range(phi_row.numel() - 1):
        a, b = float(phi_row[y]), float(phi_row[y + 1])
        if (a - 0.5) * (b - 0.5) <= 0.0 and a != b:
            return y + (0.5 - a) / (b - a)
    return float("nan")


def run_arm(
    *,
    rho_l: float,
    rho_g: float,
    nu: float,
    g: float,
    sigma: float,
    width: float,
    mobility: float,
    n_steps: int,
    sample_every: int,
    shape: tuple[int, int, int],
    radius: float,
    label: str,
) -> dict:
    wall = wall_mask(shape)
    nz, ny, nx = shape
    cy, cx = ny / 2.0, nx / 2.0
    body = disk_mask(shape, cy, cx, radius)
    solid = wall | body

    sim = m.CACSim(
        shape,
        rho_l=rho_l,
        rho_g=rho_g,
        mu_l=nu * rho_l,
        mu_g=nu * rho_g,
        sigma=sigma,
        width=width,
        mobility=mobility,
        p0=0.01,
        gravity=(0.0, -g, 0.0),
        gravity_mode="rho",
        axes="www",
        wall_mask=solid,
    )
    sim.initialize(m.init_phi_layers(shape, width=width, y0=cy))

    mid_z = nz // 2
    far_x = int(max(2, cx - 2.5 * radius))
    near_x = int(min(nx - 3, cx + radius + 3))
    phi0 = float(sim.f.sum(dim=0)[~solid].double().sum())

    t0 = time.time()
    samples = []
    nan_seen = False
    for step in range(1, n_steps + 1):
        sim.step()
        if step % sample_every == 0 or step == n_steps:
            diag = m.diagnostics(sim)
            phi, rho, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
            h_far = interface_height(phi[mid_z, :, far_x])
            h_near = interface_height(phi[mid_z, :, near_x])
            nan_seen = nan_seen or bool(diag["nan_f"] or diag["nan_g"])
            samples.append(
                {
                    "step": step,
                    "interface_y_far": h_far,
                    "interface_y_near": h_near,
                    "interface_jump": h_near - h_far,
                    "u_max": diag["u_max_fluid"],
                    "phi_min": diag["phi_min_fluid"],
                    "phi_max": diag["phi_max_fluid"],
                    "phi_integral_drift_rel": (diag["phi_integral_fluid"] - phi0) / phi0,
                    "rho_min": diag["rho_min_fluid"],
                    "rho_max": diag["rho_max_fluid"],
                }
            )

    return {
        "label": label,
        "shape": list(shape),
        "disk_radius": radius,
        "density_ratio": rho_l / rho_g,
        "g": g,
        "sigma": sigma,
        "width": width,
        "nu": nu,
        "mobility": mobility,
        "n_steps": n_steps,
        "nan_seen": nan_seen,
        "wall_time_s": time.time() - t0,
        "samples": samples,
        "final": samples[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/p0b_static_body.json"))
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    shape = (8, 96, 64)
    common = dict(
        rho_l=1.0,
        rho_g=0.01,
        nu=0.1,
        g=1e-4,
        sigma=0.01,
        width=4.0,
        mobility=0.05,
        n_steps=args.n_steps,
        sample_every=args.sample_every,
        shape=shape,
    )
    arms = [
        dict(label="disk_R8_interface", radius=8.0),
        dict(label="disk_R16_interface", radius=16.0),
    ]

    out = {"probe": "p0b_static_body", "model": "cac_lbm.CACSim", "arms": []}
    for arm in arms:
        print(f"[p0b] running {arm['label']} ...", flush=True)
        res = run_arm(**common, **arm)
        f = res["final"]
        print(
            f"[p0b]  max|u| {f['u_max']:.3e} | phi[{f['phi_min']:.3f},{f['phi_max']:.3f}] | "
            f"h_far {f['interface_y_far']:.2f} h_near {f['interface_y_near']:.2f} "
            f"jump {f['interface_jump']:+.2f} | phi drift {f['phi_integral_drift_rel']:+.2e} | "
            f"nan={res['nan_seen']} | {res['wall_time_s']:.1f}s",
            flush=True,
        )
        out["arms"].append(res)

    out["notes"] = (
        "Prerequisite probe only: static body crossing a diffuse interface in a walled "
        "gravity tank. No wetting/contact-angle BC and no contact-line capillary force "
        "exist in the library model, so this probe measures stability and artifact size, "
        "not physical contact-line behaviour."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[p0b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
