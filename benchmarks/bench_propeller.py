"""TensorLBM – Propeller Open-Water Benchmark.

Runs open-water propeller simulations at multiple inflow speeds and reports
thrust/torque coefficients.  KP505 reference rows are emitted as context only:
they are not a geometry-proven validation result and never produce a pass/fail
verdict.

Usage::

    PYTHONPATH=src python benchmarks/bench_propeller.py --fast
    PYTHONPATH=src python benchmarks/bench_propeller.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from tensorlbm.propeller_benchmark import (
    PropellerBenchmarkConfig,
    run_propeller_benchmark,
)
from tensorlbm.propeller_cad import PropellerGeometryConfig

# KP505 reference rows (Fujisawa et al. 2000, SIMMAN 2008); context only.
_KP505_REFERENCE: dict[float, tuple[float, float]] = {
    0.1: (0.450, 0.065),
    0.3: (0.420, 0.061),
    0.5: (0.370, 0.055),
    0.7: (0.290, 0.047),
    0.9: (0.170, 0.033),
    1.1: (0.040, 0.015),
}

def _summarize_kp505_context(
    results: list[dict[str, object]],
    rpm: float,
    diameter: float,
) -> dict[str, object]:
    """Report nearest KP505 rows without claiming validation or a verdict."""
    matches: list[dict[str, float]] = []
    kt_errs: list[float] = []
    kq_errs: list[float] = []

    for r in results:
        u_in = float(r["u_in"])
        j_sim = u_in / (rpm * diameter)
        kt_sim = float(r["kt"])
        kq_sim = float(r["kq"])
        j_closest = min(_KP505_REFERENCE.keys(), key=lambda j: abs(j - j_sim))
        kt_ref, kq_ref = _KP505_REFERENCE[j_closest]
        kt_err = abs(kt_sim - kt_ref) / max(abs(kt_ref), 1e-10) * 100
        kq_err = abs(kq_sim - kq_ref) / max(abs(kq_ref), 1e-10) * 100
        matches.append({
            "j_sim": j_sim, "j_ref": j_closest,
            "kt_sim": kt_sim, "kt_ref": kt_ref, "kt_err_pct": kt_err,
            "kq_sim": kq_sim, "kq_ref": kq_ref, "kq_err_pct": kq_err,
        })
        kt_errs.append(kt_err)
        kq_errs.append(kq_err)

    n = len(matches)
    return {
        "claim_status": "context_only_not_validation",
        "matches": matches,
        "n_samples": n,
        "kt_rmse_pct": (sum(e**2 for e in kt_errs) / max(n, 1)) ** 0.5,
        "kq_rmse_pct": (sum(e**2 for e in kq_errs) / max(n, 1)) ** 0.5,
    }


def _header(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TensorLBM Propeller Open-Water Benchmark")
    p.add_argument("--fast", action="store_true", help="Quick mode (coarse grid, 1 rev)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--blades", type=int, default=5)
    p.add_argument("--diameter", type=float, default=48.0)
    args = p.parse_args(argv)

    geo = PropellerGeometryConfig(n_blades=args.blades, diameter=args.diameter)

    if args.fast:
        cfg = PropellerBenchmarkConfig(
            geometry=PropellerGeometryConfig(n_blades=args.blades, diameter=32.0),
            inflow_velocities=(0.005, 0.010),
            rpm=0.00001,
            nx=60, ny=30, nz=30,
            tau=0.8,
            smagorinsky_cs=0.0,
            n_revolutions=1,
            warmup_steps=200,
            device=args.device,
            output_root=Path(args.output_root),
            run_name=args.run_name,
            overwrite=True,
        )
    else:
        cfg = PropellerBenchmarkConfig(
            geometry=geo,
            device=args.device,
            output_root=Path(args.output_root),
            run_name=args.run_name,
        )

    _header(f"Propeller OWT :: {'FAST' if args.fast else 'MEDIUM'} mode, device={args.device}")
    j_vals = [v / (cfg.rpm * geo.diameter) for v in cfg.inflow_velocities]
    print(f"  Blades={geo.n_blades}, D={geo.diameter:.1f} lu, RPM={cfg.rpm:.4f}")
    print(f"  J ≈ {j_vals}")
    print()

    result = run_propeller_benchmark(cfg)

    results_list = result.get("results", [])
    if isinstance(results_list, list) and results_list:
        context = _summarize_kp505_context(results_list, cfg.rpm, geo.diameter)
        _header("KP505 Reference Context (not validation)")
        matches = context["matches"]
        print("  Informational nearest-reference comparison only; no pass/fail claim.")
        print(f"\n  {'J':>6s}  {'KT_sim':>8s}  {'KT_ref':>8s}  {'err%':>7s}  "
              f"{'KQ_sim':>8s}  {'KQ_ref':>8s}  {'err%':>7s}")
        print(f"  {'-'*62}")
        for m in matches:  # type: ignore[assignment]
            d = dict(m)  # type: ignore[arg-type]
            print(f"  {float(d['j_sim']):6.3f}  {float(d['kt_sim']):8.4f}  "
                  f"{float(d['kt_ref']):8.3f}  {float(d['kt_err_pct']):6.1f}%  "
                  f"{float(d['kq_sim']):8.4f}  {float(d['kq_ref']):8.3f}  "
                  f"{float(d['kq_err_pct']):6.1f}%")
        print(f"  {'='*62}")
        print(f"  KT context RMSE={float(context['kt_rmse_pct']):.1f}%  "
              f"KQ context RMSE={float(context['kq_rmse_pct']):.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
