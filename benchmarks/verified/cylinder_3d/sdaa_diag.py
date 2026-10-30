#!/usr/bin/env python3
"""SDAA cylinder_3d diagnostic runner.

Runs the cylinder_3d benchmark on SDAA (eager mode) and produces
a diagnostic JSON with per-criterion PASS/FAIL verdicts.

Usage:
    PYTHONPATH=src python3 benchmarks/verified/cylinder_3d/sdaa_diag.py \
        --device sdaa:0 --steps 30000 --out artifacts/sdaa_cylinder_3d_diag.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for _p in (_REPO / "src", _REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

# Reference values
CD_REF = 1.50
CD_CLUSTER = (1.48, 1.52)
ERR_BUDGET = 3.0  # percent
SPAN_BUDGET = 3.0  # percent

# Import benchmark modules
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll  # noqa: E402
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d  # noqa: E402
from tensorlbm.obstacles import compute_obstacle_forces_3d  # noqa: E402
from tensorlbm.drag_pressure import get_near_wall_2d  # noqa: E402


def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def surface_mask(solid: torch.Tensor) -> torch.Tensor:
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def run_diag(
    D_cells: int,
    device: str = "sdaa:0",
    n_steps: int = 30000,
    lateral_D: float = 40.0,
    nz: int = 1,
    u_in: float = 0.08,
    sample_interval: int = 100,
) -> dict:
    dev = torch.device(device)
    R = D_cells / 2.0
    nx = ny = int(round(lateral_D * D_cells))
    cx = cy = nx // 2
    Re = 40.0
    nu = u_in * D_cells / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in**2 * (D_cells * nz)

    tag = f"[diag cyl3d D={D_cells} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={Re} u_in={u_in} nu={nu:.6f} tau={tau:.6f}", flush=True)
    t0 = time.time()

    solid = cylinder3d_mask(nx, ny, nz, cx, cy, R, dev)
    surf = surface_mask(solid)
    interior = solid & ~surf
    n_solid = int(solid.sum().item())
    n_surf = int(surf.sum().item())
    n_int = int(interior.sum().item())
    print(f"{tag} solid={n_solid} surface={n_surf} interior={n_int}", flush=True)

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    im0 = float(f.sum().item())

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}

    cd_surf_hist, cd_all_hist, cd_int_hist = [], [], []
    mass_hist = []

    for step in range(1, n_steps + 1):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d_roll(f)
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % sample_interval == 0:
            fx_surf = compute_obstacle_forces_3d(f, surf)[0].item()
            fx_all = compute_obstacle_forces_3d(f, solid)[0].item()
            fx_int = compute_obstacle_forces_3d(f, interior)[0].item()
            cd_surf_hist.append(fx_surf / dpS)
            cd_all_hist.append(fx_all / dpS)
            cd_int_hist.append(fx_int / dpS)
            mass_hist.append(float(f.sum().item()))

        if step % 5000 == 0:
            n = min(200, len(cd_surf_hist))
            if n:
                avg_surf = sum(cd_surf_hist[-n:]) / n
                avg_all = sum(cd_all_hist[-n:]) / n
                print(
                    f"{tag} step={step} Cd_surf={avg_surf:.4f} Cd_all={avg_all:.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(cd_surf_hist)
    win = min(n_tot, 200)
    cd_surf = sum(cd_surf_hist[-win:]) / win
    cd_all = sum(cd_all_hist[-win:]) / win
    cd_int = sum(cd_int_hist[-win:]) / win
    mass_drift = (mass_hist[-1] - im0) / im0 * 100.0 if mass_hist else float("nan")

    err_surf = (cd_surf - CD_REF) / CD_REF * 100.0
    err_all = (cd_all - CD_REF) / CD_REF * 100.0

    return {
        "D_cells": D_cells,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lateral_D": lateral_D,
        "steps": n_steps,
        "tau": tau,
        "cd_mem_surface": cd_surf,
        "err_surf_pct": err_surf,
        "cd_mem_all": cd_all,
        "err_mem_all_pct": err_all,
        "cd_mem_interior": cd_int,
        "mass_drift_pct": mass_drift,
        "wall_s": elapsed,
        "finite": True,
        "n_samples": n_tot,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lateral", type=float, default=40.0)
    ap.add_argument("--nz", type=int, default=1)
    ap.add_argument("--out", default="artifacts/sdaa_cylinder_3d_diag.json")
    a = ap.parse_args()

    grids = [20, 40]
    results = []
    for D in grids:
        r = run_diag(D, a.device, a.steps, a.lateral, a.nz)
        results.append(r)

    cd_surf = [r["cd_mem_surface"] for r in results]
    cd_all = [r["cd_mem_all"] for r in results]
    span_surf = abs(cd_surf[-1] - cd_surf[0]) / CD_REF * 100.0 if len(cd_surf) > 1 else float("nan")
    span_all = abs(cd_all[-1] - cd_all[0]) / CD_REF * 100.0 if len(cd_all) > 1 else float("nan")

    # Verdicts
    g1_pass = abs(results[0]["err_surf_pct"]) <= ERR_BUDGET
    g2_pass = abs(results[1]["err_surf_pct"]) <= ERR_BUDGET
    g3_pass = span_surf <= SPAN_BUDGET

    diag = {
        "case": "cylinder_3d_re40",
        "device": a.device,
        "torch_sdaa_version": "3.1.1a3+gite3971b9",
        "compile_mode": "eager",
        "reference": {"cd_ref": CD_REF, "cd_cluster": list(CD_CLUSTER)},
        "grids": {
            f"D{results[0]['D_cells']}_L{int(results[0]['lateral_D'])}": {
                **results[0],
                "verdict": "PASS" if g1_pass else "FAIL",
            },
            f"D{results[1]['D_cells']}_L{int(results[1]['lateral_D'])}": {
                **results[1],
                "verdict": "PASS" if g2_pass else "FAIL",
            },
        },
        "convergence": {
            "cd_mem_surface_by_grid": cd_surf,
            "cd_span_pct": span_surf,
            "verdict": "PASS" if g3_pass else "FAIL",
        },
        "verdicts": {
            "G1": {"err_pct": results[0]["err_surf_pct"], "verdict": "PASS" if g1_pass else "FAIL"},
            "G2": {"err_pct": results[1]["err_surf_pct"], "verdict": "PASS" if g2_pass else "FAIL"},
            "G3": {"span_pct": span_surf, "verdict": "PASS" if g3_pass else "FAIL"},
        },
        "overall": "VERIFIED" if (g1_pass and g2_pass and g3_pass) else "NOT_VERIFIED",
    }

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2))
    print(f"\nSaved diagnostic to {out_path}", flush=True)
    print(json.dumps(diag["verdicts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
