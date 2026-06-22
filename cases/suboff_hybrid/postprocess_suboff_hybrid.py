"""Post-processing script for the DG-LBM hybrid SUBOFF case.

Reads the ``run_metadata.json`` produced by ``run_suboff_hybrid.py`` and
generates:

* A diagnostics time-series plot (mass drift and max velocity vs step).
* A summary table printed to stdout.

Usage
-----
Run from the repository root::

    PYTHONPATH=src python cases/suboff_hybrid/postprocess_suboff_hybrid.py \\
        --run-dir outputs/dg_lbm_suboff/suboff_re200

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process a DG-LBM SUBOFF run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", dest="run_dir", required=True,
        help="Path to the run output directory (contains run_metadata.json)",
    )
    parser.add_argument(
        "--no-plot", dest="no_plot", action="store_true",
        help="Skip matplotlib plot (useful for headless environments)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    diagnostics = meta.get("diagnostics", [])
    cfg = meta.get("config", {})
    derived = meta.get("derived", {})
    hull_stats = meta.get("hull_stats", {})

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 60)
    print("DG-LBM SUBOFF – post-processing summary")
    print("=" * 60)
    print(f"  Run directory : {run_dir}")
    print(f"  Hull type     : {cfg.get('hull_type', 'N/A')}")
    print(f"  Grid          : {cfg.get('nx')} × {cfg.get('ny')} × {cfg.get('nz')}")
    print(f"  Hull length   : {cfg.get('hull_length')} lu")
    print(f"  Re            : {cfg.get('re')}")
    print(f"  u_in          : {cfg.get('u_in')}")
    print(f"  tau           : {derived.get('tau', 'N/A'):.6f}")
    print(f"  nu            : {derived.get('nu', 'N/A'):.6f}")
    print(f"  n_steps       : {cfg.get('n_steps')}")
    if hull_stats:
        print(f"  Hull L/D      : {hull_stats.get('L_D_ratio', 'N/A'):.3f}")
        print(f"  Hull cells    : {hull_stats.get('solid_cells', 'N/A')}")
    print("-" * 60)

    if diagnostics:
        last = diagnostics[-1]
        print(f"  Final step    : {last['step']}")
        print(f"  Final mass    : {last['mass']:.6f}")
        print(f"  Mass drift    : {last['mass_drift']:+.6f}")
        print(f"  Max |u|       : {last['max_speed']:.6f}")
        print(f"  Mean rho      : {last['mean_rho']:.6f}")
    else:
        print("  No diagnostic data found.")

    print("=" * 60)

    # ------------------------------------------------------------------
    # Diagnostics plot
    # ------------------------------------------------------------------
    if args.no_plot or not diagnostics:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [d["step"] for d in diagnostics]
        mass_drifts = [d["mass_drift"] for d in diagnostics]
        max_speeds = [d["max_speed"] for d in diagnostics]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

        axes[0].plot(steps, mass_drifts, marker="o", markersize=3)
        axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Mass drift")
        axes[0].set_title("Mass conservation")
        axes[0].grid(True, alpha=0.4)

        axes[1].plot(steps, max_speeds, marker="o", markersize=3, color="tab:orange")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Max velocity magnitude")
        axes[1].set_title("Max |u| over time")
        axes[1].grid(True, alpha=0.4)

        fig.suptitle(
            f"DG-LBM SUBOFF diagnostics  "
            f"(Re={cfg.get('re')}, hull={cfg.get('hull_type')})"
        )

        out = run_dir / "diagnostics_plot.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Diagnostics plot saved: {out}")
    except ImportError:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
