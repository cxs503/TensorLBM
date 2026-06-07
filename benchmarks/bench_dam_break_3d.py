"""TensorLBM – 3D Dam-Break Benchmark (Koshizuka & Oka 1996).

Compares the leading-edge front position against experimental data from
Koshizuka & Oka (1996) "Moving-Particle Semi-Implicit Method for
Fragmentation of Incompressible Fluid".

Usage::

    PYTHONPATH=src python benchmarks/bench_dam_break_3d.py
    PYTHONPATH=src python benchmarks/bench_dam_break_3d.py --device cuda
    PYTHONPATH=src python benchmarks/bench_dam_break_3d.py --fast
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from tensorlbm.dam_break_3d import (
    DamBreak3DConfig,
    _KOSHIZUKA_FRONT,
    run_dam_break_3d,
)


def _compute_front_rmse(
    front_series: list[dict[str, float]],
    ref_data: list[tuple[float, float]],
) -> dict[str, float]:
    """Compute RMSE vs reference data by interpolating at ref T* points."""
    if len(front_series) < 2:
        return {"rmse": float("nan"), "mae": float("nan")}

    t_sim = [f["t_star"] for f in front_series]
    x_sim = [f["x_star"] for f in front_series]

    errors: list[float] = []
    for t_ref, x_ref in ref_data:
        if t_ref < t_sim[0] or t_ref > t_sim[-1]:
            continue
        # Linear interpolation
        for i in range(len(t_sim) - 1):
            if t_sim[i] <= t_ref <= t_sim[i + 1]:
                w = (t_ref - t_sim[i]) / (t_sim[i + 1] - t_sim[i] + 1e-12)
                x_interp = x_sim[i] + w * (x_sim[i + 1] - x_sim[i])
                errors.append(x_interp - x_ref)
                break

    if not errors:
        return {"rmse": float("nan"), "mae": float("nan")}

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mae = sum(abs(e) for e in errors) / len(errors)
    return {"rmse": rmse, "mae": mae, "n_points": len(errors)}


def _sc_density() -> dict:
    """SC-optimal densities matching the 2D dam-break stability rules."""
    return {"rho_heavy": 0.8, "rho_light": 0.4, "tau": 1.5}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorLBM 3D dam-break benchmark (Koshizuka & Oka 1996)"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced resolution for quick validation")
    parser.add_argument("--model", default="sc", choices=["sc", "cg", "fe"],
                        help="Multiphase model (default: cg)")
    parser.add_argument("--device", default="cpu",
                        help="Compute device (default: cpu)")
    parser.add_argument("--output-root", default="/tmp/tensorlbm_dam3d",
                        help="Output directory")
    args = parser.parse_args()

    if args.fast:
        cfg = DamBreak3DConfig(
            nx=80, ny=30, nz=30,
            dam_width=30, fill_height=29,
            model=args.model, gravity=5e-5,
            n_steps=3000, output_interval=300,
            device=args.device,
            output_root=Path(args.output_root),
            run_name="bench_fast",
            overwrite=True,
            **(_sc_density() if args.model == "sc" else {}),
        )
    else:
        cfg = DamBreak3DConfig(
            nx=161, ny=50, nz=50,
            dam_width=61, fill_height=49,
            model=args.model, gravity=5e-5,
            n_steps=6000, output_interval=500,
            device=args.device,
            output_root=Path(args.output_root),
            run_name="bench_full",
            overwrite=True,
            **(_sc_density() if args.model == "sc" else {}),
        )

    print("=" * 70)
    print("  3D DAM-BREAK BENCHMARK — Koshizuka & Oka (1996)")
    print("=" * 70)
    print(f"  model={cfg.model}  device={cfg.device}  "
          f"grid={cfg.nx}×{cfg.ny}×{cfg.nz}  "
          f"steps={cfg.n_steps}  fast={args.fast}")
    print(f"  Water: {cfg.dam_width}×{cfg.fill_height}×{cfg.nz}")

    t0 = time.perf_counter()
    run_dir = run_dam_break_3d(cfg)
    elapsed = time.perf_counter() - t0

    # Load results
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    front_series = meta.get("front_series", [])

    # Compare with Koshizuka & Oka
    metrics = _compute_front_rmse(front_series, _KOSHIZUKA_FRONT)
    rmse = metrics.get("rmse", float("nan"))
    mae = metrics.get("mae", float("nan"))
    n_pts = metrics.get("n_points", 0)

    # Tolerance: RMSE < 1.5 for full, < 2.5 for fast
    tol = 2.5 if args.fast else 1.5
    ok = math.isfinite(rmse) and rmse < tol

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    if front_series:
        final = front_series[-1]
        print(f"  Final: T*={final['t_star']:.3f}  X*={final['x_star']:.3f}")
    print(f"  RMSE vs Koshizuka & Oka:  {rmse:.4f}  (tol<{tol:.1f})")
    print(f"  MAE:                      {mae:.4f}")
    print(f"  Reference points matched: {n_pts}")
    print(f"  Elapsed: {elapsed:.1f}s")
    flag = "✓ PASS" if ok else "✗ FAIL"
    print(f"  Status: {flag}")
    print(f"{'='*70}")

    # Save report
    report = {
        "benchmark": "dam_break_3d_koshizuka_oka_1996",
        "model": cfg.model,
        "device": cfg.device,
        "fast": args.fast,
        "grid": f"{cfg.nx}x{cfg.ny}x{cfg.nz}",
        "rmse": rmse,
        "mae": mae,
        "tolerance": tol,
        "n_ref_points": n_pts,
        "ok": ok,
        "elapsed_s": elapsed,
        "run_dir": str(run_dir),
    }
    report_path = Path(args.output_root) / "dam_break_3d_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Report: {report_path}")

    if not ok:
        exit(1)


if __name__ == "__main__":
    main()
