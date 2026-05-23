#!/usr/bin/env python3
"""Batch parameter scan for the 2D cylinder-flow benchmark.

Runs :func:`~tensorlbm.run_cylinder_flow` for each Reynolds number in a
user-specified list and collects the time-averaged drag coefficient Cd,
lift-coefficient amplitude Cl_rms, and Strouhal number St into a summary
CSV and a comparison bar chart.

Usage example::

    PYTHONPATH=src python examples/param_scan.py \\
        --re 20 40 80 100 \\
        --nx 160 --ny 60 --n-steps 2000 --output-interval 100 \\
        --output-root outputs/scan

"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# Ensure the package is importable when run directly from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm import CylinderFlowConfig, run_cylinder_flow


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Re-scan for D2Q9 cylinder flow")
    p.add_argument("--re", type=float, nargs="+", default=[20.0, 40.0, 80.0, 100.0],
                   help="List of Reynolds numbers to simulate")
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=60)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--u-in", type=float, default=0.05)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--output-interval", type=int, default=100)
    p.add_argument("--output-root", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _summarise_run(run_dir: Path) -> dict[str, float | int | None]:
    """Extract summary statistics from a completed run directory."""
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    diagnostics = meta.get("diagnostics", [])
    re = meta["config"]["re"]
    strouhal = meta.get("strouhal")

    # Use second half of diagnostics to avoid transient
    half = max(1, len(diagnostics) // 2)
    late = diagnostics[half:]

    cd_values = [d["cd"] for d in late if isinstance(d.get("cd"), float) and math.isfinite(d["cd"])]
    cl_values = [d["cl"] for d in late if isinstance(d.get("cl"), float) and math.isfinite(d["cl"])]

    cd_mean = statistics.mean(cd_values) if cd_values else float("nan")
    cl_rms = math.sqrt(statistics.mean(v * v for v in cl_values)) if cl_values else float("nan")

    return {"re": re, "cd_mean": cd_mean, "cl_rms": cl_rms, "strouhal": strouhal}


def _save_summary(rows: list[dict], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "scan_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["re", "cd_mean", "cl_rms", "strouhal"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary CSV: {csv_path}")
    return csv_path


def _plot_summary(rows: list[dict], output_root: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plot.")
        return

    re_vals = [r["re"] for r in rows]
    cd_vals = [r["cd_mean"] for r in rows]
    cl_vals = [r["cl_rms"] for r in rows]
    st_vals = [r["strouhal"] if r["strouhal"] is not None else float("nan") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    axes[0].bar([str(r) for r in re_vals], cd_vals)
    axes[0].set_xlabel("Re")
    axes[0].set_ylabel("Cd (time-averaged)")
    axes[0].set_title("Drag coefficient")

    axes[1].bar([str(r) for r in re_vals], cl_vals)
    axes[1].set_xlabel("Re")
    axes[1].set_ylabel("Cl rms")
    axes[1].set_title("Lift coefficient (rms)")

    axes[2].bar([str(r) for r in re_vals], st_vals)
    axes[2].set_xlabel("Re")
    axes[2].set_ylabel("St")
    axes[2].set_title("Strouhal number")

    plot_path = output_root / "scan_summary.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot: {plot_path}")


def main() -> None:
    args = _parse_args()
    summary_rows: list[dict] = []

    for re in args.re:
        print(f"\n{'='*60}")
        print(f"  Re = {re}")
        print(f"{'='*60}")
        cfg = CylinderFlowConfig(
            nx=args.nx,
            ny=args.ny,
            u_in=args.u_in,
            re=re,
            radius=args.radius,
            n_steps=args.n_steps,
            output_interval=args.output_interval,
            output_root=args.output_root / "cylinder_flow_scan",
            device=args.device,
            overwrite=args.overwrite,
        )
        try:
            run_dir = run_cylinder_flow(cfg)
            row = _summarise_run(run_dir)
            summary_rows.append(row)
            print(f"  → Cd={row['cd_mean']:.4f}  Cl_rms={row['cl_rms']:.4f}  St={row['strouhal']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR for Re={re}: {exc}")
            summary_rows.append(
                {
                    "re": re,
                    "cd_mean": float("nan"),
                    "cl_rms": float("nan"),
                    "strouhal": None,
                }
            )

    if summary_rows:
        _save_summary(summary_rows, args.output_root)
        _plot_summary(summary_rows, args.output_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
