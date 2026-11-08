#!/usr/bin/env python3
"""AMR lid-driven cavity (Re=1000) with the 2:1 reflux ledger, Ghia comparison.

Example
-------
    PYTHONPATH=src python benchmarks/amr_reflux/run_cavity_amr.py \
        --coarse 128 --strip 32 --steps 60000 --out artifacts/amr_reflux_ladder

Writes ``cavity_amr.json`` (configuration + conservation metrics + Ghia errors),
``cavity_amr_metrics.jsonl`` (one record per log interval, crash safe) and
``cavity_amr.png`` (centreline profiles vs Ghia 1982).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorlbm.amr_cavity2d import CavityAMR2D, CavityAMR2DConfig, ghia_comparison
from tensorlbm.lid_driven_cavity import GHIA_RE1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=int, default=128)
    parser.add_argument("--strip", type=int, default=32)
    parser.add_argument("--re", type=float, default=1000.0)
    parser.add_argument("--u-lid", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--no-reflux", action="store_true")
    parser.add_argument("--save-every", type=int, default=0, help="write a restart snapshot every N steps")
    parser.add_argument("--resume", action="store_true", help="resume from cavity_state.npz")
    parser.add_argument("--out", type=str, default="artifacts/amr_reflux_ladder")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CavityAMR2DConfig(
        coarse=args.coarse,
        strip_coarse=args.strip,
        re=args.re,
        u_lid=args.u_lid,
        reflux=not args.no_reflux,
    )
    solver = CavityAMR2D(cfg)
    state_path = out / "cavity_state.npz"
    if args.resume and state_path.exists():
        state = np.load(state_path)
        solver.f = state["f"].copy()
        solver.f_top = state["f_top"].copy()
        solver.f_bottom = state["f_bottom"].copy()
        solver.step_count = int(state["steps"])
        print(f"[cavity-amr] resumed from step {solver.step_count}", flush=True)
    inv0 = solver.inventory()["total"]
    metrics: list[dict[str, float]] = []
    t0 = time.time()
    with (out / "cavity_amr_metrics.jsonl").open("w") as fh:
        for step in range(1, args.steps + 1):
            diag = solver.step()
            if args.save_every and (step % args.save_every == 0):
                np.savez_compressed(
                    state_path,
                    f=solver.f,
                    f_top=solver.f_top,
                    f_bottom=solver.f_bottom,
                    steps=solver.step_count,
                )
            if step % args.log_every == 0 or step == 1:
                inv = solver.inventory()
                rec = {
                    "step": step,
                    "elapsed_s": time.time() - t0,
                    "interface_mismatch_rel": diag["interface_mismatch_rel"],
                    "interface_balance_rel": diag["interface_balance_rel"],
                    "interface_flux_scale": diag["interface_flux_scale"],
                    "inventory_total": inv["total"],
                    "mass_drift_rel": abs(inv["total"] - inv0) / inv0,
                }
                if not np.isfinite(solver.f).all() or not np.isfinite(solver.f_top).all():
                    rec["status"] = "non_finite"
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"[cavity-amr] NON-FINITE at step {step}", flush=True)
                    break
                metrics.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                prof = solver.centreline_profiles()
                partial = {
                    "case": "cavity_amr_reflux_ghia",
                    "config": {
                        "coarse": cfg.coarse,
                        "strip_coarse": cfg.strip_coarse,
                        "fine_shape": [2 * cfg.strip_coarse, 2 * cfg.coarse],
                        "re": cfg.re,
                        "u_lid": cfg.u_lid,
                        "tau_coarse": cfg.tau_coarse,
                        "tau_fine": cfg.tau_fine,
                        "reflux": cfg.reflux,
                        "steps": args.steps,
                        "collision": "rlbm",
                    },
                    "steps_done": step,
                    "partial": step < args.steps,
                    "conservation": {
                        "max_interface_mismatch_rel": max(
                            m["interface_mismatch_rel"] for m in metrics
                        ),
                        "max_interface_balance_rel": max(
                            m["interface_balance_rel"] for m in metrics
                        ),
                        "mass_drift_rel": rec["mass_drift_rel"],
                        "elapsed_s": rec["elapsed_s"],
                    },
                    "ghia": ghia_comparison(prof, GHIA_RE1000, cfg.u_lid),
                    "series": {k: v.tolist() for k, v in prof.items()},
                    "metrics_log": metrics,
                }
                (out / "cavity_amr.json").write_text(json.dumps(partial, indent=1))
                print(
                    f"[cavity-amr] step {step}/{args.steps} "
                    f"mismatch={rec['interface_mismatch_rel']:.3e} "
                    f"balance={rec['interface_balance_rel']:.3e} "
                    f"drift={rec['mass_drift_rel']:.3e} "
                    f"t={rec['elapsed_s']:.0f}s",
                    flush=True,
                )

    profiles = solver.centreline_profiles()
    ghia = ghia_comparison(profiles, GHIA_RE1000, cfg.u_lid)
    inv1 = solver.inventory()["total"]
    summary = {
        "case": "cavity_amr_reflux_ghia",
        "config": {
            "coarse": cfg.coarse,
            "strip_coarse": cfg.strip_coarse,
            "fine_shape": [2 * cfg.strip_coarse, 2 * cfg.coarse],
            "re": cfg.re,
            "u_lid": cfg.u_lid,
            "tau_coarse": cfg.tau_coarse,
            "tau_fine": cfg.tau_fine,
            "reflux": cfg.reflux,
            "steps": args.steps,
            "log_every": args.log_every,
            "collision": "rlbm",
            "boundary": "pre-streaming bounce-back (3 walls) + Zou/He moving lid",
        },
        "conservation": {
            "max_interface_mismatch_rel": max((m["interface_mismatch_rel"] for m in metrics), default=0.0),
            "max_interface_balance_rel": max((m["interface_balance_rel"] for m in metrics), default=0.0),
            "mass_start": inv0,
            "mass_end": inv1,
            "mass_drift_rel": abs(inv1 - inv0) / inv0,
            "elapsed_s": time.time() - t0,
        },
        "ghia": ghia,
        "reference": "Ghia, Ghia & Shin 1982, J. Comput. Phys. 48:387, Re=1000",
        "series": {
            "y": profiles["y"].tolist(),
            "u": profiles["u"].tolist(),
            "x": profiles["x"].tolist(),
            "v": profiles["v"].tolist(),
        },
        "metrics_log": metrics,
    }
    (out / "cavity_amr.json").write_text(json.dumps(summary, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(profiles["u"] / cfg.u_lid, profiles["y"], "-", label="AMR (2:1 strips)")
    axes[0].plot(GHIA_RE1000["u"], GHIA_RE1000["y"], "o", mfc="none", label="Ghia 1982")
    axes[0].set_xlabel("u / U_lid")
    axes[0].set_ylabel("y / H")
    axes[0].legend(fontsize=8)
    axes[0].set_title("vertical centreline")
    axes[1].plot(profiles["x"], profiles["v"] / cfg.u_lid, "-", label="AMR (coarse core)")
    axes[1].plot(GHIA_RE1000["x"], GHIA_RE1000["v"], "o", mfc="none", label="Ghia 1982")
    axes[1].set_xlabel("x / H")
    axes[1].set_ylabel("v / U_lid")
    axes[1].legend(fontsize=8)
    axes[1].set_title("horizontal centreline")
    if metrics:
        axes[2].semilogy([m["step"] for m in metrics], [m["interface_mismatch_rel"] for m in metrics], "-o", label="mismatch (pre)")
        axes[2].semilogy([m["step"] for m in metrics], [max(m["interface_balance_rel"], 1e-18) for m in metrics], "-s", label="residual (post)")
        axes[2].axhline(1e-3, color="k", ls="--", lw=0.8, label="0.1% criterion")
        axes[2].set_xlabel("coarse step")
        axes[2].set_ylabel("relative interface imbalance")
        axes[2].legend(fontsize=8)
        axes[2].set_title("coarse/fine interface ledger")
    fig.tight_layout()
    fig.savefig(out / "cavity_amr.png", dpi=140)
    print(
        f"[cavity-amr] done: max|du|={ghia['max_abs_dev_u']:.4f} "
        f"max|dv|={ghia['max_abs_dev_v']:.4f} "
        f"max_dev={ghia['max_abs_dev']:.4f} ({ghia['max_dev_pct_of_u_lid']:.2f}% of U_lid)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())