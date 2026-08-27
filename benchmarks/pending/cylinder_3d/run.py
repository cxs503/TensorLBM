#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""3D cylinder Re=40 steady, infinite span (D3Q19, spanwise periodic) — benchmark.

Physics:
  - 3D extruded circular cylinder, axis along z (span), infinite span via
    periodic z-boundaries (Lz = 2D, periodicity handled by stream3d_roll).
  - Domain 30D x 30D x 2D targeted; on a 24 GB GPU the practical grid is
    16D x 16D x 2D (D=40) and 12D x 12D x 2D (D=60), far-field (free-stream
    Dirichlet) BC on x- / y- / y+, zero-gradient outlet at x+, periodic z.
  - Re = U_in * D / nu = 40 (steady, no vortex shedding; 2D-equivalent flow,
    no spanwise variation at Re=40 -> 3D result must equal the 2D value).
  - Reference (infinite-span 2D/3D values):
        Dennis & Chang (1970): Cd = 1.522
        Fornberg (1985):       Cd = 1.498
        Tritton (1959, exp):   Cd ~ 1.54
    Task reference window: Cd = 1.52 - 1.55.  Primary ref: 1.54.
  - Normalisation: Cd = Fx / (0.5 * rho * U_in^2 * D * Lz)  (frontal area D*Lz).

Method:
  - Library primitives only: solver3d.collide_bgk3d + stream3d_roll,
    boundaries3d.bounce_back_cells_3d + far_field_bc_3d (bc_config with
    periodic z faces), drag_pressure.SurfaceMesh.from_cylinder +
    get_near_wall_2d(axis='z') + drag_pressure_integration (extrap='none',
    p0_method='far_field') + drag_friction_integration (formula='standard').
  - Main loop (half-way bounce-back, verified library pattern from
    lbm_step_correct): collide -> NoDynamics (solid restored to pre-collision)
    -> bounce-back BEFORE streaming -> stream -> far-field BC.
  - No mass renormalisation (2D cylinder_re20_st verified lesson: mass
    rescaling excites a slowly decaying Cd oscillation and biases averages).
  - No extrapolation / correction factors.

Force methods:
  - PRIMARY: pressure + friction integration (drag_pressure), extrap='none'.
  - DIAGNOSTIC: Ladd MEM variants (standard / galilean / bg_sub) are
    computed once at the end to check the G15 curved-surface issue on the
    staircase cylinder (expected: background does not cancel -> spurious
    force, as on the sphere).

Usage:
    run.py single D_cells out.json [--steps 50000] [--device cuda:2]
    run.py verify out_dir   [--steps 50000] [--device cuda:2]   # D=40 + D=60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    get_near_wall_2d,
)
from tensorlbm.momentum_exchange import (
    momentum_exchange_background_subtracted,
    momentum_exchange_galilean,
    momentum_exchange_standard,
)
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll

REF_CD = 1.54  # Tritton 1959 (task window 1.52-1.55)
REF_CD_DC = 1.522  # Dennis & Chang 1970


def face_counts(solid):
    """Per-cell wall-face counts (nfx, nfy, nfz) with verified slicing."""
    fluid = ~solid
    nfx = torch.zeros_like(solid, dtype=torch.float32)
    nfy = torch.zeros_like(solid, dtype=torch.float32)
    nfz = torch.zeros_like(solid, dtype=torch.float32)
    nfx[:, :, 1:-1] += (solid[:, :, 2:] & fluid[:, :, 1:-1]).float()
    nfx[:, :, 1:-1] += (solid[:, :, :-2] & fluid[:, :, 1:-1]).float()
    nfy[:, 1:-1, :] += (solid[:, 2:, :] & fluid[:, 1:-1, :]).float()
    nfy[:, 1:-1, :] += (solid[:, :-2, :] & fluid[:, 1:-1, :]).float()
    nfz[1:-1, :, :] += (solid[2:, :, :] & fluid[1:-1, :, :]).float()
    nfz[1:-1, :, :] += (solid[:-2, :, :] & fluid[1:-1, :, :]).float()
    return nfx, nfy, nfz


def smooth_q(solid, cx, cy, R, near):
    """q_smooth = distance from near-wall cell center to the smooth
    cylinder surface along the radial, clamped to [0.05, 1.0]."""
    nz, ny, nx = solid.shape
    dev = solid.device
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    r_c = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    q = (r_c - R).clamp(0.05, 1.0)
    return q * near.float()


def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean mask for a z-axis extruded circular cylinder, shape (nz,ny,nx).

    NOTE: no library implementation exists in boundaries3d.py (only
    sphere_mask) -> this local helper is the gap to be recorded.  Mask
    radius = R (NOT R-0.5): verified 2D lesson (cylinder_re20_st) — the
    hydrodynamic radius of the staircase circle is ~R, and R-0.5 overshoots
    inward and lowers the drag.
    """
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def run_case(
    D_cells: int,
    device: str = "cuda:2",
    n_steps: int = 50000,
    lateral_D: float | None = None,
    sample_interval: int = 100,
    formula: str = "standard",
) -> dict:
    dev = torch.device(device)
    R = D_cells / 2.0
    nz = 2 * D_cells  # span Lz = 2D, periodic
    if lateral_D is None:
        lateral_D = 16.0 if D_cells <= 40 else 12.0
    nx = ny = int(round(lateral_D * D_cells))
    cx = nx // 2  # cylinder axis at domain centre
    cy = ny // 2

    Re = 40.0
    u_in = 0.08
    nu = u_in * D_cells / Re  # Re = U*D/nu
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in**2 * (D_cells * nz)  # 0.5*rho*U^2 * frontal area (D*Lz)

    tag = f"[cyl3d D={D_cells} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={Re} u_in={u_in} nu={nu:.6f} tau={tau:.6f} dpS={dpS:.4f}", flush=True)
    t0 = time.time()

    # --- geometry -------------------------------------------------------
    solid = cylinder3d_mask(nx, ny, nz, cx, cy, R, dev)
    n_solid = int(solid.sum().item())
    near = get_near_wall_2d(solid, axis="z")
    n_near = int(near.sum().item())
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    # friction-formula machinery (q_smooth for BFL variants, dA_scale ratio)
    q_wall = (
        smooth_q(solid, cx, cy, R, near)
        if formula in ("bfl", "bfl_lagrange", "bfl_smooth")
        else None
    )
    nfx, nfy, nfz = face_counts(solid)
    dA_ratio = float((nfx + nfy + nfz).sum().item()) / float(near.sum().item())
    print(f"{tag} formula={formula} faces/near ratio={dA_ratio:.4f}", flush=True)
    print(
        f"{tag} solid={n_solid} near={n_near} "
        f"(blockage {100.0 * D_cells / nx:.1f}% lateral, "
        f"Lz={nz} cells = {nz / D_cells:.1f}D)",
        flush=True,
    )

    # --- init: uniform free stream, solid cells at rest -----------------
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    del rho0, ux0, uy0, uz0
    im0 = float(f.sum().item())
    print(f"{tag} init done ({time.time() - t0:.0f}s)", flush=True)

    bc_config = {
        "far_field_faces": ["y-", "y+"],
        "periodic_faces": ["z-", "z+"],
    }

    # --- main loop: half-way BB pattern (lbm_step_correct 'bb' order) ---
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    mass_hist, umax_hist = [], []
    step = 0
    for step in range(1, n_steps + 1):
        f_pre_solid = f[:, solid].clone()  # NoDynamics restore set
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid  # NoDynamics
        f = bounce_back_cells_3d(f, solid)  # BB BEFORE stream (half-way)
        f = stream3d_roll(f)
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % sample_interval == 0:
            fx_p, fy_p, _ = drag_pressure_integration(
                f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid
            )
            if formula == "dA_scale":
                fx_f, fy_f, _ = drag_friction_integration(f, mesh, dpS, nu)
                fx_f *= dA_ratio
                fy_f *= dA_ratio
            else:
                # 'bfl_smooth' = library formula='bfl' with analytic q_wall
                # (q_smooth = r_c - R, clamped [0.05,1]); see run_compare_d20.py
                friction_formula = "bfl" if formula == "bfl_smooth" else formula
                fx_f, fy_f, _ = drag_friction_integration(
                    f, mesh, dpS, nu, q_wall=q_wall, formula=friction_formula, solid=solid
                )
            cd_p_hist.append(fx_p)
            cd_f_hist.append(fx_f)
            cd_tot_hist.append(fx_p + fx_f)
            cl_hist.append(fy_p + fy_f)
            mass_hist.append(float(f.sum().item()))
            umax_hist.append(float(f.abs().max().item()))

        if step % 5000 == 0:
            n_avg = min(200, len(cd_tot_hist))
            if n_avg:
                cd = sum(cd_tot_hist[-n_avg:]) / n_avg
                cd_p = sum(cd_p_hist[-n_avg:]) / n_avg
                cd_f = sum(cd_f_hist[-n_avg:]) / n_avg
                cl = sum(cl_hist[-n_avg:]) / n_avg
                dm = (mass_hist[-1] - im0) / im0 * 100
                print(
                    f"{tag} step={step} Cd_p={cd_p:.4f} Cd_f={cd_f:.4f} "
                    f"Cd={cd:.4f} Cl={cl:.6f} dmass={dm:+.4f}% "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(cd_tot_hist)
    win = min(n_tot, 200)  # last 200 samples = 20000 steps
    cd_p = sum(cd_p_hist[-win:]) / win
    cd_f = sum(cd_f_hist[-win:]) / win
    cd = cd_p + cd_f
    cl = sum(cl_hist[-win:]) / win
    # convergence diagnostics: last 100 vs previous 100 samples
    half = win // 2
    cd_a = sum(cd_tot_hist[-half:]) / half
    cd_b = sum(cd_tot_hist[-2 * half : -half]) / half
    drift_pct = (cd_a - cd_b) / REF_CD * 100.0

    # --- diagnostic: MEM variants on the staircase cylinder (G15 check) ---
    mem = {}
    try:
        mem["standard"] = momentum_exchange_standard(f, solid, near)
        mem["galilean"] = momentum_exchange_galilean(f, solid, near, tau)
        mem["bg_sub"] = momentum_exchange_background_subtracted(
            f, solid, near, rho0=1.0, u0=(u_in, 0.0, 0.0)
        )
        mem["cd"] = {k: v[0] / dpS for k, v in mem.items() if k != "cd"}
    except Exception as exc:  # pragma: no cover
        mem["error"] = str(exc)

    print(
        f"{tag} === FINAL === Cd_p={cd_p:.4f} Cd_f={cd_f:.4f} Cd={cd:.4f} "
        f"(ref {REF_CD}) err={(cd - REF_CD) / REF_CD * 100:+.2f}% "
        f"Cl={cl:.6f} drift={drift_pct:+.3f}% ({elapsed:.0f}s)",
        flush=True,
    )
    if mem.get("cd"):
        print(
            f"{tag} MEM diag: " + " ".join(f"{k}={v:.4f}" for k, v in mem["cd"].items()), flush=True
        )

    return {
        "case": "cylinder_3d_re40",
        "lattice": "D3Q19",
        "collision": "bgk",
        "geometry": "z-axis extruded cylinder (infinite span, z periodic)",
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "D_cells": D_cells,
        "R_cells": R,
        "mask_radius": R,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "Lz_D": nz / D_cells,
        "lateral_D": nx / D_cells,
        "blockage_pct": 100.0 * D_cells / nx,
        "cx": cx,
        "cy": cy,
        "n_solid_cells": n_solid,
        "n_near_cells": n_near,
        "n_wall_faces": int((nfx + nfy + nfz).sum().item()),
        "face_cell_ratio": dA_ratio,
        "friction_formula": formula,
        "n_steps": n_steps,
        "n_finished": step,
        "sample_interval": sample_interval,
        "avg_window_samples": win,
        "cd_pressure": cd_p,
        "cd_friction": cd_f,
        "cd_total": cd,
        "cl": cl,
        "err_cd_pct": (cd - REF_CD) / REF_CD * 100.0,
        "err_cd_dc_pct": (cd - REF_CD_DC) / REF_CD_DC * 100.0,
        "ref_cd": REF_CD,
        "ref_cd_dc": REF_CD_DC,
        "drift_cd_pct": drift_pct,
        "mass_drift_pct": (mass_hist[-1] - im0) / im0 * 100.0 if mass_hist else float("nan"),
        "finite": bool(torch.isfinite(f).all().item()),
        "mem_diag": mem.get("cd", mem),
        "wall_s": elapsed,
        "modules_used": [
            "solver3d.collide_bgk3d",
            "solver3d.stream3d_roll",
            "boundaries3d.far_field_bc_3d (bc_config periodic z)",
            "boundaries3d.bounce_back_cells_3d (half-way BB pre-stream)",
            "drag_pressure.SurfaceMesh.from_cylinder",
            "drag_pressure.get_near_wall_2d",
            "drag_pressure.drag_pressure_integration (extrap=none, p0=far_field)",
            f"drag_pressure.drag_friction_integration ({formula})",
            "momentum_exchange.* (diagnostic only)",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["single", "verify"])
    ap.add_argument("arg", help="D_cells (single) or output dir (verify)")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument(
        "--lateral",
        type=float,
        default=None,
        help="lateral domain in diameters (default 16 for D=40, 12 for D=60)",
    )
    ap.add_argument(
        "--formula",
        default="standard",
        choices=["standard", "lagrange", "bfl", "bfl_lagrange", "bfl_smooth", "faces", "dA_scale"],
        help="friction formula (default standard); 'bfl_smooth' = "
        "formula='bfl' with analytic q_smooth=r_c-R",
    )
    a = ap.parse_args()

    if a.mode == "single":
        r = run_case(int(a.arg), a.device, a.steps, lateral_D=a.lateral, formula=a.formula)
        print(json.dumps(r, indent=2))
        out_path = a.arg if a.arg.endswith(".json") else f"/tmp/cyl3d_d{a.arg}.json"
        Path(out_path).write_text(json.dumps(r, indent=2))
        return 0

    out_dir = Path(a.arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_grid = []
    for D in (40, 60):
        r = run_case(D, a.device, a.steps, lateral_D=a.lateral, formula=a.formula)
        per_grid.append(r)
        print(json.dumps(r, indent=2))
        (out_dir / f"case_D{D}.json").write_text(json.dumps(r, indent=2))
    ok = all(abs(r["err_cd_pct"]) <= 3.0 for r in per_grid)
    conv = (
        len(per_grid) >= 2
        and abs(per_grid[1]["cd_total"] - per_grid[0]["cd_total"]) / REF_CD * 100.0 <= 3.0
    )
    res = {
        "case": "cylinder_3d_re40",
        "lattice": "D3Q19",
        "collision": "bgk",
        "boundary": "far_field_bc_3d (x- inlet eq, x+ zero-gradient, "
        "y+- far-field, z+- periodic) + half-way bounce-back "
        "(NoDynamics + BB pre-stream)",
        "force": "drag_pressure_integration(extrap=none, p0=far_field) + "
        f"drag_friction_integration({a.formula}); MEM diagnostic only",
        "friction_formula": a.formula,
        "extrap": "none",
        "ref_cd": REF_CD,
        "ref_cd_window": "1.52-1.55",
        "per_grid": per_grid,
        "converged": ok and conv,
        "verdict": "verified" if (ok and conv) else "not_verified",
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
