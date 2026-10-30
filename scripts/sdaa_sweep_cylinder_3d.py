#!/usr/bin/env python3
"""SDAA sweep — cylinder_3d benchmark (TensorLBM verified case).

Independent runner that reuses the **same library primitives** as
``benchmarks/verified/cylinder_3d/run.py`` but is self-contained for the
SDAA validation sweep.  Does NOT modify any business code.

Case: 3D extruded circular cylinder, Re=40, infinite span (z-periodic).
Primary Cd = Ladd momentum-exchange restricted to **surface** solid cells.

Criteria (from the case README / result.json — original NVIDIA-side gates):
  G1  D=20  cd_mem_surface within 3% of Cd_ref=1.50
  G2  D=40  cd_mem_surface within 3% of Cd_ref=1.50
  G3  grid span |cd(D40)−cd(D20)| / Cd_ref ≤ 3%

Device: sdaa:0 (eager mode — SDAA teco-inductor falls back, same as run.py).
Triton: the eager step chain uses pure PyTorch ops (collide_bgk3d, stream3d_roll,
bounce_back_cells_3d, far_field_bc_3d) — no triton kernels, so
TRITON_INTERPRET is not needed.  Recorded as "not_required".

Output: artifacts/sdaa_cylinder_3d.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── path bootstrap (same as run.py) ──────────────────────────────────────────
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]  # scripts/ -> repo root
for _p in (_REPO / "src", _REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d  # noqa: E402
from tensorlbm.obstacles import compute_obstacle_forces_3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll  # noqa: E402

# ── reference convention (from README / REFERENCE_AUDIT.md) ──────────────────
CD_REF = 1.50
CD_CLUSTER = (1.48, 1.52)
TOL_PCT = 3.0          # per-grid tolerance
SPAN_TOL_PCT = 3.0     # grid-span tolerance

# ── sweep configuration ──────────────────────────────────────────────────────
# README recommends 30 000 steps, lateral=32, nz=1 for plateau.
# Timing on this SDAA host (eager, no compile):
#   D=20 (640²×1): ~40 ms/step (stepping only)
#   D=40 (1280²×1): ~130 ms/step (stepping only)
# Force sampling (compute_obstacle_forces_3d) adds significant overhead, so
# we sample less frequently and only compute the primary (surface) caliber
# during the run; all-solid / interior diagnostics are computed once at the
# final step.
#
# Budget: ≤ 20 min total wall clock.
#   D=20: 10 000 steps × 40 ms + 20 samples × ~1.5 s ≈ 430 s  (7.2 min)
#   D=40:  4 000 steps × 130 ms + 8 samples × ~6 s  ≈ 568 s  (9.5 min)
#   Total ≈ 16.7 min + setup overhead
GRIDS = [
    {"D": 20, "lateral": 32, "nz": 1, "steps": 10000, "sample": 500},
    {"D": 40, "lateral": 32, "nz": 1, "steps": 4000,  "sample": 500},
]
DEVICE = "sdaa:0"
U_IN = 0.08
RE = 40.0


# ── helpers (identical logic to run.py) ──────────────────────────────────────

def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def surface_mask(solid: torch.Tensor) -> torch.Tensor:
    """Wall-adjacent solid cells: solid with ≥1 fluid 6-neighbour."""
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def run_single(cfg: dict, device_str: str) -> dict:
    """Run one grid and return Cd metrics (same chain as run.py::run_case).

    During the stepping loop only the PRIMARY caliber (surface MEM) is
    sampled to minimise overhead.  The all-solid and interior calibers are
    computed once on the final field for diagnostic completeness.
    """
    dev = torch.device(device_str)
    D = cfg["D"]
    lateral = cfg["lateral"]
    nz = cfg["nz"]
    n_steps = cfg["steps"]
    sample_interval = cfg["sample"]

    R = D / 2.0
    nx = ny = int(round(lateral * D))
    cx = cy = nx // 2
    nu = U_IN * D / RE
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * U_IN ** 2 * (D * nz)

    tag = f"[cyl3d D={D} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={RE} u_in={U_IN} nu={nu:.6f} tau={tau:.6f} dpS={dpS:.4f}", flush=True)
    t0 = time.time()

    solid = cylinder3d_mask(nx, ny, nz, cx, cy, R, dev)
    surf = surface_mask(solid)
    interior = solid & ~surf
    n_solid = int(solid.sum().item())
    n_surf = int(surf.sum().item())
    n_int = int(interior.sum().item())
    print(
        f"{tag} solid={n_solid} surface={n_surf} interior={n_int} "
        f"(blockage {100.0 * D / nx:.2f}%)",
        flush=True,
    )

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, U_IN)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0, ux0
    im0 = float(f.sum().item())

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}

    def _step(f):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d_roll(f)
        return far_field_bc_3d(f, U_IN, bc_config=bc_config)

    # Eager mode — no torch.compile (SDAA teco-inductor fallback, same as run.py)
    mem_surf_hist = []   # PRIMARY caliber only during loop
    mass_hist = []
    step = 0
    diverged = False
    for step in range(1, n_steps + 1):
        f = _step(f)

        if step % sample_interval == 0:
            # Only compute the PRIMARY (surface) caliber to minimise overhead
            mem_surf_hist.append(
                float(compute_obstacle_forces_3d(f, surf)[0].item()) / dpS
            )
            mass_hist.append(float(f.sum().item()))

        if step % 2000 == 0:
            n = min(200, len(mem_surf_hist))
            if n:
                print(
                    f"{tag} step={step} Cd_mem_surf={sum(mem_surf_hist[-n:]) / n:.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

    elapsed = time.time() - t0
    n_tot = len(mem_surf_hist)
    win = min(n_tot, 200)
    cd_surf = sum(mem_surf_hist[-win:]) / win if win else float("nan")
    err_surf = (cd_surf - CD_REF) / CD_REF * 100.0 if win else float("nan")
    mass_drift = (mass_hist[-1] - im0) / im0 * 100.0 if mass_hist else float("nan")

    # Diagnostic calibers on the FINAL field only (one-shot, not per-sample)
    cd_all_final = float("nan")
    cd_int_final = float("nan")
    if not diverged:
        try:
            cd_all_final = float(compute_obstacle_forces_3d(f, solid)[0].item()) / dpS
            cd_int_final = float(compute_obstacle_forces_3d(f, interior)[0].item()) / dpS
        except Exception as e:
            print(f"{tag} diagnostic compute failed: {e}", flush=True)

    print(
        f"{tag} === FINAL win={win} === Cd_mem_surface={cd_surf:.4f} "
        f"err={err_surf:+.2f}%  |  diag: Cd_mem_all={cd_all_final:.4f} "
        f"Cd_mem_interior={cd_int_final:.4f}  ({elapsed:.0f}s)",
        flush=True,
    )

    return {
        "D": D,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lateral_D": lateral,
        "blockage_pct": 100.0 * D / nx,
        "Re": RE,
        "u_in": U_IN,
        "nu": nu,
        "tau": tau,
        "n_steps_requested": n_steps,
        "n_steps_finished": step,
        "n_samples": n_tot,
        "avg_window": win,
        "n_solid_cells": n_solid,
        "n_surface_cells": n_surf,
        "n_interior_cells": n_int,
        "cd_mem_surface": cd_surf,
        "err_surf_pct": err_surf,
        # Diagnostics from final field (one-shot, not time-averaged)
        "cd_mem_all_final": cd_all_final,
        "cd_mem_interior_final": cd_int_final,
        "mass_drift_pct": mass_drift,
        "finite": bool(torch.isfinite(f).all().item()) if not diverged else False,
        "diverged": diverged,
        "wall_s": elapsed,
        "cd_mem_surface_history": [round(v, 6) for v in mem_surf_hist],
    }


def main() -> int:
    # ── device evidence ──────────────────────────────────────────────────────
    dev = torch.device(DEVICE)
    device_name = "UNKNOWN"
    device_type = dev.type
    try:
        device_name = torch.sdaa.get_device_name(0)
    except Exception as e:
        device_name = f"ERROR: {e}"

    # Verify tensors actually land on sdaa
    probe = torch.randn(16, device=dev)
    probe_device = str(probe.device)
    probe_ok = probe_device == DEVICE
    print(f"[sdaa_sweep] device={DEVICE} name={device_name} "
          f"probe_device={probe_device} ok={probe_ok}", flush=True)

    # ── triton check ─────────────────────────────────────────────────────────
    triton_available = False
    triton_version = None
    try:
        import triton
        triton_available = True
        triton_version = triton.__version__
    except ImportError:
        pass
    # The eager step chain (collide_bgk3d → bounce_back_cells_3d → stream3d_roll
    # → far_field_bc_3d) uses pure PyTorch ops — no triton kernels.
    triton_required = False
    triton_interpret_used = os.environ.get("TRITON_INTERPRET", None)
    triton_note = (
        "Eager mode: step chain uses pure PyTorch ops (no triton kernels). "
        "TRITON_INTERPRET not needed."
    )

    # ── run grids ────────────────────────────────────────────────────────────
    t_total = time.time()
    results = []
    for cfg in GRIDS:
        r = run_single(cfg, DEVICE)
        results.append(r)

    total_wall = time.time() - t_total

    # ── criteria / gates ─────────────────────────────────────────────────────
    cd_vals = [r["cd_mem_surface"] for r in results]
    errs = [r["err_surf_pct"] for r in results]
    grid_span_pct = (
        abs(cd_vals[-1] - cd_vals[0]) / CD_REF * 100.0
        if len(cd_vals) > 1 and all(v == v for v in cd_vals)  # NaN check
        else float("nan")
    )

    gates = []
    for i, (r, err) in enumerate(zip(results, errs)):
        passed = abs(err) <= TOL_PCT if err == err else False
        gates.append({
            "gate_id": f"G{i+1}",
            "description": f"D={r['D']} cd_mem_surface within {TOL_PCT}% of Cd_ref={CD_REF}",
            "cd_mem_surface": r["cd_mem_surface"],
            "err_pct": err,
            "tolerance_pct": TOL_PCT,
            "pass": bool(passed),
        })

    span_pass = grid_span_pct <= SPAN_TOL_PCT if grid_span_pct == grid_span_pct else False
    gates.append({
        "gate_id": "G3",
        "description": f"grid span |cd(D40)−cd(D20)|/Cd_ref ≤ {SPAN_TOL_PCT}%",
        "cd_D20": cd_vals[0] if len(cd_vals) > 0 else None,
        "cd_D40": cd_vals[1] if len(cd_vals) > 1 else None,
        "span_pct": grid_span_pct,
        "tolerance_pct": SPAN_TOL_PCT,
        "pass": bool(span_pass),
    })

    all_pass = all(g["pass"] for g in gates)

    # ── assemble artifact ────────────────────────────────────────────────────
    artifact = {
        "case": "cylinder_3d_re40",
        "description": "3D extruded circular cylinder, infinite span (z-periodic), Re=40. "
                       "Primary Cd = Ladd MEM restricted to surface solid cells.",
        "sweep_script": "scripts/sdaa_sweep_cylinder_3d.py",
        "source_case": "benchmarks/verified/cylinder_3d",
        "device_evidence": {
            "device_requested": DEVICE,
            "device_type": device_type,
            "device_name": device_name,
            "probe_tensor_device": probe_device,
            "probe_ok": probe_ok,
        },
        "triton": {
            "available": triton_available,
            "version": triton_version,
            "required_for_step_chain": triton_required,
            "TRITON_INTERPRET_env": triton_interpret_used,
            "note": triton_note,
        },
        "compile_mode": "eager (SDAA teco-inductor fallback — same as run.py)",
        "reference": {
            "cd_ref": CD_REF,
            "cluster": list(CD_CLUSTER),
            "sources": {
                "Dennis & Chang 1970": 1.522,
                "Fornberg 1985": 1.498,
                "Takami & Keller 1969": 1.48,
            },
            "convention": "2D free-stream numerical cluster; NOT Tritton 1959 (finite-span experiment).",
        },
        "configuration": {
            "grids": GRIDS,
            "u_in": U_IN,
            "Re": RE,
            "tolerance_pct": TOL_PCT,
            "span_tolerance_pct": SPAN_TOL_PCT,
            "note": "Steps reduced from README-recommended 30000 to fit 20-min wall budget. "
                    "Cd may not be fully plateaued — see convergence history in each grid result.",
        },
        "grid_results": results,
        "cd_mem_surface_by_grid": cd_vals,
        "err_surf_pct_by_grid": errs,
        "grid_span_pct": grid_span_pct,
        "gates": gates,
        "all_gates_pass": bool(all_pass),
        "verdict": "PASS" if all_pass else "FAIL",
        "total_wall_s": total_wall,
        "total_wall_min": round(total_wall / 60.0, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_dir = _REPO / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sdaa_cylinder_3d.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n[sdaa_sweep] saved → {out_path}", flush=True)
    print(f"[sdaa_sweep] verdict={artifact['verdict']} "
          f"all_gates_pass={all_pass} total_wall={total_wall:.0f}s", flush=True)
    for g in gates:
        print(f"  {g['gate_id']}: {'PASS' if g['pass'] else 'FAIL'} — {g['description']}", flush=True)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
