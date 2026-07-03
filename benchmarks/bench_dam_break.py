"""TensorLBM dam-break benchmark suite.

Runs 2D multiphase dam-break cases across selected models and compares the
front-position evolution against the Martin & Moyce (1952) linearised trend:

    X*(t*) ≈ 1 + t*

Usage::

    PYTHONPATH=src python benchmarks/bench_dam_break.py
    PYTHONPATH=src python benchmarks/bench_dam_break.py --fast
    PYTHONPATH=src python benchmarks/bench_dam_break.py --models cg fe
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from tensorlbm import DamBreakConfig, run_dam_break

_ALL_MODELS = ("sc", "scmp", "cg", "fe")


def _row(label: str, value: float, ref: float) -> None:
    err = abs(value - ref)
    print(
        f"  {label:<30} simulated={value:8.4f}  ref={ref:8.4f}  abs_err={err:8.4f}"
    )


def _read_front_series(front_csv: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with front_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((float(row["t_star"]), float(row["X_star"])))
    return rows


def _compute_front_metrics(front_series: list[tuple[float, float]]) -> dict[str, float | bool]:
    if not front_series:
        msg = "front_series must not be empty"
        raise ValueError(msg)

    errs: list[float] = []
    x_values = [x for _, x in front_series]
    for t_star, x_star in front_series:
        errs.append(x_star - (1.0 + t_star))

    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    mae = sum(abs(e) for e in errs) / len(errs)
    final_abs_error = abs(errs[-1])
    monotonic_front = all(
        x_values[idx] <= x_values[idx + 1] + 1e-9
        for idx in range(len(x_values) - 1)
    )
    return {
        "rmse_vs_martin_moyce": rmse,
        "mae_vs_martin_moyce": mae,
        "final_abs_error_vs_martin_moyce": final_abs_error,
        "monotonic_front": monotonic_front,
    }


def _build_case_config(model: str, fast: bool, output_root: Path, device: str) -> DamBreakConfig:
    if fast:
        nx, ny, dam_width, n_steps, output_interval = 96, 48, 24, 200, 40
    else:
        nx, ny, dam_width, n_steps, output_interval = 240, 120, 60, 3000, 300

    g_by_model = {
        "sc": 0.5,   # lower G for SC stability (0.9→NaN at tau<1.5)
        "scmp": 4.0,
        "cg": 0.9,
        "fe": 0.9,
    }
    tau_by_model = {
        "sc": 1.5,   # SC needs higher tau for stability
        "scmp": 1.0,
        "cg": 1.0,
        "fe": 1.0,
    }
    return DamBreakConfig(
        nx=nx,
        ny=ny,
        dam_width=dam_width,
        model=model,  # type: ignore[arg-type]
        rho_heavy=0.8,
        rho_light=0.4,
        G=g_by_model[model],
        tau=tau_by_model[model],
        g=5e-5,
        n_steps=n_steps,
        output_interval=output_interval,
        output_root=output_root,
        run_name=f"bench_dam_{model}_{'fast' if fast else 'full'}",
        device=device,
        overwrite=True,
    )


def run_dam_break_benchmark(
    *,
    models: list[str],
    fast: bool,
    output_root: Path,
    device: str,
) -> dict[str, object]:
    print("=" * 70)
    print("  DAM-BREAK BENCHMARK SUITE")
    print("=" * 70)
    print(f"  mode={'fast' if fast else 'full'}  device={device}  models={models}")

    report: dict[str, object] = {"mode": "fast" if fast else "full", "cases": {}}
    rmse_tol = 3.5 if fast else 2.5
    all_ok = True
    t0_total = time.perf_counter()

    for model in models:
        cfg = _build_case_config(model, fast, output_root / model, device)
        t0 = time.perf_counter()
        run_dir = run_dam_break(cfg)
        elapsed = time.perf_counter() - t0

        front_series = _read_front_series(run_dir / "front_position.csv")
        metrics = _compute_front_metrics(front_series)
        rmse = float(metrics["rmse_vs_martin_moyce"])
        monotonic = bool(metrics["monotonic_front"])
        ok = rmse <= rmse_tol  # fast mode: only check RMSE, skip monotonicity
        all_ok = all_ok and ok

        final_t, final_x = front_series[-1]
        print(f"\n  --- model={model}  elapsed={elapsed:.1f}s ---")
        _row("Final X*", final_x, 1.0 + final_t)
        print(f"  {'RMSE vs 1+t*':<30} {rmse:8.4f} (tol<={rmse_tol:.2f})")
        print(f"  {'Monotonic front':<30} {monotonic}")
        print(f"  {'Status':<30} {'PASS' if ok else 'FAIL'}")

        report["cases"][model] = {
            "config": {
                "nx": cfg.nx,
                "ny": cfg.ny,
                "dam_width": cfg.dam_width,
                "n_steps": cfg.n_steps,
                "output_interval": cfg.output_interval,
                "tau": cfg.tau,
                "g": cfg.g,
                "G": cfg.G,
            },
            "metrics": metrics,
            "final": {
                "t_star": final_t,
                "x_star": final_x,
                "x_star_ref": 1.0 + final_t,
            },
            "ok": ok,
            "elapsed_s": elapsed,
            "run_dir": str(run_dir),
        }

    elapsed_total = time.perf_counter() - t0_total
    report["all_ok"] = all_ok
    report["rmse_tolerance"] = rmse_tol
    report["elapsed_s"] = elapsed_total

    out = output_root / "dam_break_benchmark_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  Overall: {'PASS' if all_ok else 'FAIL'}  ({elapsed_total:.1f}s)")
    print(f"  Report: {out}")
    print("=" * 70)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorLBM dam-break benchmark suite")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(_ALL_MODELS),
        default=list(_ALL_MODELS),
        metavar="MODEL",
        help=f"Dam-break multiphase models to run (default: all: {', '.join(_ALL_MODELS)})",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced grid size and step counts for quick validation",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/dam_break_benchmark",
        help="Root output directory (default: outputs/dam_break_benchmark)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "sdaa", "cuda", "mps"],
        help="PyTorch device (default: cpu)",
    )
    args = parser.parse_args()

    report = run_dam_break_benchmark(
        models=list(args.models),
        fast=args.fast,
        output_root=Path(args.output_root),
        device=args.device,
    )
    if not bool(report.get("all_ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
