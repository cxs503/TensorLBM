#!/usr/bin/env python3
"""SDAA sweep for cylinder_3d benchmark — independent driver with mid-flight checkpoint.

This script bypasses the SDAA agent workflow and runs the cylinder_3d benchmark
directly, saving intermediate results to survive interruptions. It addresses the
failures observed in m192/m188 (agent didn't call finish) and m53 (honest-FAIL).

Design:
- Independent execution (no agent dependency)
- Mid-flight checkpoint: saves state every N steps
- Resume capability: can restart from checkpoint
- Final result JSON with full metadata
- Explicit compile_mode tracking (eager fallback on SDAA)

Usage:
    python scripts/sdaa_sweep_cylinder_3d_v3.py --device sdaa:0 --grids 20 40 \\
        --steps 30000 --lateral 40 --checkpoint-interval 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
for _p in (_REPO / "src", _REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    route_step,
)
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.drag_pressure import (  # noqa: E402
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    get_near_wall_2d,
)
from tensorlbm.obstacles import compute_obstacle_forces_3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll  # noqa: E402

# Reference convention (from cylinder_3d/run.py)
CD_REF = 1.50
CD_CLUSTER = (1.48, 1.52)
REF_NOTE = (
    "2D free-stream NUMERICAL cluster Cd≈1.50 [1.48,1.52]: "
    "Dennis&Chang 1970 1.522 / Fornberg 1985 1.498 / Takami&Keller 1969 1.48. "
    "NOT Tritton 1959 ≈1.54 (finite-span experiment)."
)


def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def surface_mask(solid: torch.Tensor) -> torch.Tensor:
    """Wall-adjacent solid cells: solid with >=1 fluid 6-neighbour."""
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def run_single_grid(
    D_cells: int,
    device: str,
    n_steps: int,
    lateral_D: float,
    nz: int,
    u_in: float,
    sample_interval: int,
    compile_mode: str | None,
    checkpoint_path: Path | None,
    resume_from: Path | None,
) -> dict:
    """Run one grid and return result dict."""
    dev = torch.device(device)
    R = D_cells / 2.0
    nx = ny = int(round(lateral_D * D_cells))
    cx = cy = nx // 2
    Re = 40.0
    nu = u_in * D_cells / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in**2 * (D_cells * nz)

    tag = f"[cyl3d D={D_cells} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={Re} u_in={u_in} nu={nu:.6f} tau={tau:.6f} dpS={dpS:.4f}", flush=True)
    t0 = time.time()

    # Geometry setup
    solid = cylinder3d_mask(nx, ny, nz, cx, cy, R, dev)
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    surf = surface_mask(solid)
    interior = solid & ~surf

    n_solid = int(solid.sum().item())
    n_surf = int(surf.sum().item())
    n_int = int(interior.sum().item())
    print(
        f"{tag} solid={n_solid} surface={n_surf} interior={n_int} "
        f"(blockage {100.0 * D_cells / nx:.2f}%, span {nz} cells = {nz / D_cells:.2f}D)",
        flush=True,
    )

    # Resume or initialize
    start_step = 0
    if resume_from and resume_from.exists():
        print(f"{tag} Resuming from checkpoint: {resume_from}", flush=True)
        ckpt = torch.load(resume_from, map_location=dev)
        f = ckpt["f"]
        start_step = ckpt["step"]
        mem_surf_hist = ckpt["mem_surf_hist"]
        mem_hist = ckpt["mem_hist"]
        mem_int_hist = ckpt["mem_int_hist"]
        cdp_hist = ckpt["cdp_hist"]
        print(f"{tag} Resumed at step {start_step}, history length {len(mem_surf_hist)}", flush=True)
    else:
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
        ux0 = torch.full_like(rho0, u_in)
        ux0[solid] = 0.0
        f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
        del rho0, ux0
        mem_surf_hist = []
        mem_hist = []
        mem_int_hist = []
        cdp_hist = []

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}

    def _step(f):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d_roll(f)
        return far_field_bc_3d(f, u_in, bc_config=bc_config)

    step_fn = route_step(_step, compile_mode, name=f"cylinder_3d_re40[D{D_cells}]")
    compile_status = getattr(step_fn, "compile_status", "unknown")

    # Main loop
    for step in range(start_step + 1, n_steps + 1):
        f = step_fn(f)

        if step % sample_interval == 0:
            mem_hist.append(float(compute_obstacle_forces_3d(f, solid)[0].item()) / dpS)
            mem_surf_hist.append(float(compute_obstacle_forces_3d(f, surf)[0].item()) / dpS)
            mem_int_hist.append(float(compute_obstacle_forces_3d(f, interior)[0].item()) / dpS)
            cdp_hist.append(
                drag_pressure_integration(
                    f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid
                )[0]
            )

        # Checkpoint
        if checkpoint_path and step % (n_steps // 6) == 0:
            ckpt = {
                "step": step,
                "f": f,
                "mem_surf_hist": mem_surf_hist,
                "mem_hist": mem_hist,
                "mem_int_hist": mem_int_hist,
                "cdp_hist": cdp_hist,
            }
            torch.save(ckpt, checkpoint_path)
            print(f"{tag} Checkpoint saved at step {step}", flush=True)

        if step % 5000 == 0:
            n = min(200, len(mem_surf_hist))
            if n:
                cd_surf_avg = sum(mem_surf_hist[-n:]) / n
                cd_all_avg = sum(mem_hist[-n:]) / n
                print(
                    f"{tag} step={step} Cd_mem_surf={cd_surf_avg:.4f} "
                    f"Cd_mem_all={cd_all_avg:.4f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            return {
                "D": D_cells,
                "status": "diverged",
                "step": step,
                "elapsed_s": time.time() - t0,
            }

    elapsed = time.time() - t0
    n_tot = len(mem_surf_hist)
    win = min(n_tot, 200)
    cd_surf = sum(mem_surf_hist[-win:]) / win
    cd_all = sum(mem_hist[-win:]) / win
    cd_int = sum(mem_int_hist[-win:]) / win
    cdp = sum(cdp_hist[-win:]) / win

    err_pct = 100.0 * (cd_surf - CD_REF) / CD_REF
    within_3pct = abs(err_pct) <= 3.0

    result = {
        "D": D_cells,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lateral_D": lateral_D,
        "blockage_pct": 100.0 * D_cells / nx,
        "steps": n_steps,
        "u_in": u_in,
        "tau": tau,
        "cd_mem_surface": round(cd_surf, 4),
        "cd_mem_all": round(cd_all, 4),
        "cd_mem_interior": round(cd_int, 4),
        "cd_pressure": round(cdp, 4),
        "err_pct": round(err_pct, 2),
        "within_3pct": within_3pct,
        "elapsed_s": round(elapsed, 1),
        "compile_status": compile_status,
        "status": "completed",
    }

    print(
        f"{tag} FINAL Cd_mem_surface={cd_surf:.4f} (err={err_pct:+.2f}%) "
        f"Cd_mem_all={cd_all:.4f} Cd_mem_interior={cd_int:.4f} "
        f"compile={compile_status} elapsed={elapsed:.0f}s",
        flush=True,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="SDAA sweep for cylinder_3d benchmark")
    parser.add_argument("--device", default="sdaa:0", help="Device (e.g., sdaa:0, cuda:0, cpu)")
    parser.add_argument("--grids", nargs="+", type=int, default=[20, 40], help="Grid diameters")
    parser.add_argument("--steps", type=int, default=30000, help="Number of steps per grid")
    parser.add_argument("--lateral", type=float, default=40.0, help="Lateral domain in D")
    parser.add_argument("--nz", type=int, default=1, help="Span cells")
    parser.add_argument("--u-in", type=float, default=0.08, help="Inlet velocity")
    parser.add_argument("--sample-interval", type=int, default=100, help="Sample every N steps")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cylinder_3d_sdaa_sweep_v3.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints if available")
    add_compile_mode_arg(parser)
    args = parser.parse_args()

    compile_mode = compile_mode_from_args(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SDAA SWEEP: cylinder_3d benchmark (independent driver v3)")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Grids: {args.grids}")
    print(f"Steps: {args.steps}")
    print(f"Lateral: {args.lateral}D")
    print(f"nz: {args.nz}")
    print(f"Compile mode: {compile_mode}")
    print(f"Output: {args.output}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 80)

    results = {
        "case": "cylinder_3d_re40",
        "description": "3D z-axis extruded circular cylinder, infinite span (z-periodic), Re=40",
        "lattice": "D3Q19",
        "collision": "bgk",
        "reference": {
            "cd_ref": CD_REF,
            "cluster": list(CD_CLUSTER),
            "convention": REF_NOTE,
        },
        "force_method": "Ladd MEM, surface-only (wall-adjacent cells)",
        "device": args.device,
        "compile_mode_requested": compile_mode,
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "grids": {},
    }

    all_passed = True
    for D in args.grids:
        print(f"\n{'=' * 80}")
        print(f"Running grid D={D}")
        print(f"{'=' * 80}")

        ckpt_path = args.checkpoint_dir / f"cylinder_3d_D{D}.pt"
        resume_path = ckpt_path if args.resume and ckpt_path.exists() else None

        result = run_single_grid(
            D_cells=D,
            device=args.device,
            n_steps=args.steps,
            lateral_D=args.lateral,
            nz=args.nz,
            u_in=args.u_in,
            sample_interval=args.sample_interval,
            compile_mode=compile_mode,
            checkpoint_path=ckpt_path,
            resume_from=resume_path,
        )

        grid_key = f"D{D}_L{int(args.lateral)}"
        results["grids"][grid_key] = result

        if result["status"] != "completed" or not result.get("within_3pct", False):
            all_passed = False

    results["utc_end"] = datetime.now(timezone.utc).isoformat()
    results["all_passed"] = all_passed
    results["verdict"] = "PASS" if all_passed else "FAIL"

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Grid':<15} {'Cd_surf':<10} {'Cd_all':<10} {'err%':<10} {'within 3%':<10} {'status':<10}")
    print("-" * 80)
    for grid_key, grid_result in results["grids"].items():
        if grid_result["status"] == "completed":
            print(
                f"{grid_key:<15} {grid_result['cd_mem_surface']:<10.4f} "
                f"{grid_result['cd_mem_all']:<10.4f} {grid_result['err_pct']:<+10.2f} "
                f"{str(grid_result['within_3pct']):<10} {grid_result['status']:<10}"
            )
        else:
            print(f"{grid_key:<15} {'DIVERGED':<10} {'-':<10} {'-':<10} {'-':<10} {grid_result['status']:<10}")
    print("=" * 80)
    print(f"Verdict: {results['verdict']}")
    print("=" * 80)

    # Save result
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResult saved to: {args.output}")

    # Clean up checkpoints
    for ckpt in args.checkpoint_dir.glob("cylinder_3d_D*.pt"):
        ckpt.unlink()
    print("Checkpoints cleaned up")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
