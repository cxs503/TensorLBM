#!/usr/bin/env python3
"""Stokes sphere (Re<<1) drag probe — surface-only Ladd MEM.

Reuses the verified cylinder_3d / sphere_re100 force idiom
(``2*sum_surface c_ix f_i``, post-stream pre-bounce-back) to measure the
Stokes drag F = Cd * 0.5 rho u^2 pi R^2 and back out the *hydrodynamic*
radius a_eff = F / (6 pi mu U).  For Re->0 the exact result is
F = 6 pi mu a U, so a_eff reveals the numerical wall radius of the
staircase (half-way bounce-back) sphere.

Outputs JSON with cd_mem_surface (vs 24/Re), a_eff, and a_eff/R.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src", _REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d, sphere_mask  # noqa: E402
from tensorlbm.d3q19 import C, equilibrium3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402
from tensorlbm.drag_pressure import (  # noqa: E402
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)


def _surface_mask(solid: torch.Tensor) -> torch.Tensor:
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def mem_force_x(f: torch.Tensor, mask: torch.Tensor) -> float:
    c = C.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    return float((2.0 * (cx * (f * mask.unsqueeze(0))).sum()).item())


def run_case(
    D_cells: int,
    device: str = "sdaa:0",
    n_steps: int = 2000,
    lateral_D: float = 16.0,
    up_D: float = 4.0,
    down_D: float = 12.0,
    u_in: float = 0.02,
    re: float = 0.1,
    r_off: float = 0.0,
    sample_interval: int = 100,
    compile_mode: str | None = "eager",
    report_every: int = 500,
) -> dict:
    dev = torch.device(device)
    R = D_cells / 2.0
    r_mask = R + r_off
    nx = int(round((up_D + down_D + 1.0) * D_cells))
    ny = nz = int(round(lateral_D * D_cells))
    cx = up_D * D_cells + R
    cy, cz = ny / 2.0, nz / 2.0
    nu = u_in * D_cells / re
    tau = 0.5 + 3.0 * nu
    mu = nu  # rho=1
    dpS = 0.5 * u_in**2 * math.pi * R**2
    cd_ref = 24.0 / re

    tag = f"[stk D={D_cells} off={r_off:+.1f} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={re} u={u_in} nu={nu:.5f} tau={tau:.4f} R={R} rmask={r_mask}", flush=True)
    t0 = time.time()

    solid = sphere_mask(nx, ny, nz, cx, cy, cz, r_mask, device=dev)
    surf = _surface_mask(solid)
    near = get_near_wall_3d(solid)
    try:
        mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, r_mask)
        pf_ok = True
    except Exception as exc:  # pragma: no cover
        print(f"{tag} PF instrument unavailable: {exc!r}", flush=True)
        mesh = None
        pf_ok = False
    print(f"{tag} solid={int(solid.sum().item())} surf={int(surf.sum().item())} "
          f"blockage={100.0*D_cells/ny:.2f}%", flush=True)

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0, ux0

    def _step(f):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d(f)
        return far_field_bc_3d(f, u_in)

    step_fn = route_step(_step, compile_mode, name=f"stokes_sph[D{D_cells}]")

    surf_hist = []
    cdp_hist, cdf_hist = [], []
    step = 0
    for step in range(1, n_steps + 1):
        f = step_fn(f)
        if step % sample_interval == 0:
            surf_hist.append(mem_force_x(f, surf) / dpS)
            if pf_ok:
                fxp, _, _ = drag_pressure_integration(f, mesh, dpS)
                fxf, _, _ = drag_friction_integration(f, mesh, dpS, mu)
                cdp_hist.append(float(fxp))
                cdf_hist.append(float(fxf))
        if step % report_every == 0:
            n = min(100, len(surf_hist))
            if n:
                cd = sum(surf_hist[-n:]) / n
                Feff = cd * dpS
                a_eff = Feff / (6.0 * math.pi * mu * u_in)
                pf = ""
                if n and cdp_hist:
                    cdp = sum(cdp_hist[-n:]) / len(cdp_hist[-n:])
                    cdf = sum(cdf_hist[-n:]) / len(cdf_hist[-n:])
                    pf = f" Cd_p={cdp:.3f} Cd_f={cdf:.3f} Cd_pf={cdp+cdf:.3f}"
                print(f"{tag} step={step} Cd_mem={cd:.4f} err={100*(cd-cd_ref)/cd_ref:+.2f}% "
                      f"a_eff={a_eff:.4f}{pf} ({time.time()-t0:.0f}s)", flush=True)
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n = len(surf_hist)
    win = min(200, n)
    cd = sum(surf_hist[-win:]) / win if win else float("nan")
    Feff = cd * dpS
    a_eff = Feff / (6.0 * math.pi * mu * u_in)
    err = 100.0 * (cd - cd_ref) / cd_ref
    print(f"{tag} === FINAL cd={cd:.4f} err={err:+.3f}% a_eff={a_eff:.5f} "
          f"a_eff-R={a_eff-R:.5f} ({elapsed:.0f}s) ===", flush=True)
    return {
        "D_cells": D_cells, "nx": nx, "ny": ny, "nz": nz, "lateral_D": lateral_D,
        "blockage_pct": 100.0 * D_cells / ny, "Re": re, "u_in": u_in, "nu": nu, "tau": tau,
        "r_off": r_off, "R": R, "r_mask": r_mask, "n_steps": n_steps, "n_finished": step,
        "n_solid": int(solid.sum().item()), "n_surface": int(surf.sum().item()),
        "cd_mem_surface": cd, "cd_ref": cd_ref, "err_pct": err,
        "a_eff": a_eff, "a_eff_minus_R": a_eff - R, "a_eff_over_R": a_eff / R,
        "finite": bool(torch.isfinite(f).all().item()), "elapsed_s": elapsed,
        "compile_mode_effective": getattr(step_fn, "compile_mode_effective", None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "scan"])
    ap.add_argument("arg")
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lateral", type=float, default=16.0)
    ap.add_argument("--up", type=float, default=4.0)
    ap.add_argument("--down", type=float, default=12.0)
    ap.add_argument("--u-in", type=float, default=0.02)
    ap.add_argument("--re", type=float, default=0.1)
    ap.add_argument("--r-off", type=float, default=0.0)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--grids", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0])
    ap.add_argument("--out", default=None)
    add_compile_mode_arg(ap, default="eager")
    a = ap.parse_args()
    cm = compile_mode_from_args(a)

    if a.mode == "single":
        r = run_case(int(a.arg), a.device, a.steps, a.lateral, a.up, a.down,
                     a.u_in, a.re, a.r_off, a.sample, cm)
        out = a.out or f"/tmp/stk_D{a.arg}.json"
        Path(out).write_text(json.dumps(r, indent=2))
        return 0

    out_dir = Path(a.arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for off in a.offsets:
        for D in a.grids:
            r = run_case(D, a.device, a.steps, a.lateral, a.up, a.down,
                         a.u_in, a.re, off, a.sample, cm)
            rows.append(r)
            (out_dir / f"case_D{D}_off{off:+.1f}.json").write_text(json.dumps(r, indent=2))
    (out_dir / "scan.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"RESULT D={r['D_cells']} off={r['r_off']:+.1f} cd={r['cd_mem_surface']:.4f} "
              f"err={r['err_pct']:+.2f}% a_eff-R={r['a_eff_minus_R']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())