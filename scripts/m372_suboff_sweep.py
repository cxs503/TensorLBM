#!/usr/bin/env python3
"""m372: SUBOFF Re=1000 resolution sweep on SDAA.

Runs L=16, 24, 32, 48 with enough steps for force-window stability.
Results saved to artifacts/m372_suboff_sweep.json.
"""
import sys, json, math, time, os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from tensorlbm.general_sim import (
    GeneralSimEngine, GeneralSimConfig, GeometryConfig, GeometrySource,
    PhysicsConfig, SolverConfig, OutputConfig, CollisionModel, LatticeModel,
    WallTreatment, ForceMethod,
)

SUBOFF_LENGTH_M = 4.356
SUBOFF_RADIUS_M = 0.254
U_PHYS = 1.0e-3
VISCOSITY = U_PHYS * SUBOFF_LENGTH_M / 1000.0  # Re=1000

# Steps per resolution: more steps for coarser grids to reach quasi-steady
# (coarser grids converge faster in physical time but need more steps for
# the force window to stabilise).
STEPS = {16: 8000, 24: 8000, 32: 8000, 48: 12000}
DEVICE = "sdaa:0"


def run_one(L: int, steps: int, device: str = DEVICE) -> dict:
    """Run one resolution and return result dict."""
    config = GeneralSimConfig(
        name=f"m372_suboff_L{L}",
        geometry=GeometryConfig(
            source=GeometrySource.PARAMETRIC_SUBOFF,
            suboff_length=SUBOFF_LENGTH_M,
            suboff_radius=SUBOFF_RADIUS_M,
        ),
        physics=PhysicsConfig(
            density=1000.0,
            viscosity=VISCOSITY,
            inlet_velocity=U_PHYS,
            reference_length=SUBOFF_LENGTH_M,
        ),
        solver=SolverConfig(
            lattice=LatticeModel.D3Q19,
            collision=CollisionModel.MRT,
            resolution=L,
            domain_padding=(1.0, 4.0, 1.0, 1.0, 1.0, 1.0),
            max_steps=steps,
            warmup_steps=None,
            snapshot_interval=10_000_000,
            force_sample_interval=10,
            device=device,
            wall_treatment=WallTreatment.AUTO,
            force_method=ForceMethod.PRESSURE_FRICTION,
            pressure_extrap="none",
            p0_method="near_wall",
            friction_formula="mix50",
            mass_correction=True,
            mass_correction_interval=200,
            smagorinsky_cs=0.05,
        ),
        output=OutputConfig(
            directory=str(_REPO_ROOT / f"artifacts/m372_suboff_L{L}"),
            formats=[],
            save_macroscopic=False,
            save_forces=True,
        ),
    )

    print(f"\n=== L={L}: {steps} steps on {device} ===", flush=True)
    engine = GeneralSimConfig
    engine = GeneralSimEngine(config)
    setup_info = engine.setup()

    R_cells = SUBOFF_RADIUS_M / SUBOFF_LENGTH_M * L
    D_cells = 2 * R_cells

    t0 = time.time()
    info = engine.run()
    elapsed = time.time() - t0

    forces = engine.forces_log
    n_samples = len(forces)

    # Compute window averages at different checkpoints
    conv = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(n_samples * frac)
        n_win = min(500, k)
        if n_win > 0 and k > 0:
            seg = forces[max(0, k - n_win) : k]
            if seg:
                conv[f"{int(frac * 100)}%"] = round(
                    sum(f["cd_total"] for f in seg) / len(seg), 6
                )

    # Last-window average (last 500 samples or all if fewer)
    if forces:
        n_win = min(500, len(forces))
        last_forces = forces[-n_win:]
        cd_p = sum(f["cd_pressure"] for f in last_forces) / len(last_forces)
        cd_f = sum(f["cd_friction"] for f in last_forces) / len(last_forces)
        cd_tot = sum(f["cd_total"] for f in last_forces) / len(last_forces)
    else:
        cd_p = cd_f = cd_tot = 0.0

    cf_ref = 1.328 / math.sqrt(1000.0)
    err_pct = (cd_tot - cf_ref) / cf_ref * 100.0

    result = {
        "L": L,
        "device": device,
        "domain_lu": list(setup_info["domain_lu"]),
        "total_cells": setup_info["total_cells"],
        "solid_cells": setup_info["obstacle_cells"],
        "near_wall_cells": setup_info["near_wall_cells"],
        "R_cells": round(R_cells, 1),
        "D_cells": round(D_cells, 1),
        "steps": info["steps"],
        "elapsed_s": round(elapsed, 1),
        "ms_per_step": round(elapsed / max(info["steps"], 1) * 1000, 2),
        "force_samples": n_samples,
        "Cd_pressure": round(cd_p, 6),
        "Cd_friction": round(cd_f, 6),
        "Cd_total": round(cd_tot, 6),
        "Cf_ref": cf_ref,
        "error_pct": round(err_pct, 2),
        "compile_status": info.get("compile_status", "N/A"),
        "compile_mode_effective": info.get("compile_mode_effective", "N/A"),
        "convergence_windows": conv,
        "diverged": info.get("diverged", False),
        "finite": info.get("finite", True),
    }

    verdict = "PASS (<3%)" if abs(err_pct) < 3.0 else "FAIL"
    result["verdict"] = verdict

    print(
        f"  L={L}: Cd_tot={cd_tot:.6f} (ref {cf_ref:.6f}) err={err_pct:+.2f}% "
        f"| {verdict} | {elapsed:.1f}s ({result['ms_per_step']} ms/step)",
        flush=True,
    )
    return result


def main():
    resolutions = [16, 24, 32, 48]
    results = []
    for L in resolutions:
        steps = STEPS.get(L, 8000)
        r = run_one(L, steps)
        results.append(r)
        # Save intermediate
        out_path = _REPO_ROOT / "artifacts/m372_suboff_sweep.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary
    print("\n=== SWEEP SUMMARY ===")
    cf_ref = 1.328 / math.sqrt(1000.0)
    for r in results:
        print(
            f"  L={r['L']:2d}: R={r['R_cells']:.1f} cells, "
            f"Cd_tot={r['Cd_total']:.6f}, err={r['error_pct']:+.2f}%, "
            f"{r['verdict']}, {r['ms_per_step']} ms/step"
        )

    # Check monotonicity
    errors = [r["error_pct"] for r in results]
    # For convergence, we expect error to decrease (in magnitude) as L increases
    # Since all errors are positive at low L and negative at high L,
    # we check if Cd_tot approaches Cf_ref monotonically
    cds = [r["Cd_total"] for r in results]
    monotone = all(cds[i] >= cds[i + 1] for i in range(len(cds) - 1))
    print(f"\n  Cd monotone decreasing: {monotone}")
    print(f"  Cf_ref (Blasius): {cf_ref:.6f}")

    # Save final
    summary = {
        "mission": "m372",
        "case": "suboff_re1000_resolution_sweep",
        "device": DEVICE,
        "cf_ref": cf_ref,
        "cf_ref_formula": "1.328/sqrt(Re)",
        "Re": 1000,
        "normalization": "wetted area S=pi*D*L",
        "results": results,
        "monotone_decreasing": monotone,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = _REPO_ROOT / "artifacts/m372_suboff_sweep.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
