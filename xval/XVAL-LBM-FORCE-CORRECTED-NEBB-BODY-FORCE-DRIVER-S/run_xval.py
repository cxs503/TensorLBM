#!/usr/bin/env python3
"""XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S — run script.

Runs the force-corrected NEBB body-force channel cross-validation
and writes the run artifacts + evidence bundle.

Arms:
  A — D2Q9, repo NEBB module (tensorlbm.nebb_bodyforce.BodyForceChannel)
  B — D3Q19, main solver (GPUCorePathSolver)

Usage:
    python3 run_xval.py --out artifacts/m19695/xval_run.json --figure artifacts/m19695/fig_profile.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.gpu_core_path import (
    GPUCorePathConfig,
    GPUCorePathSolver,
    channel_analytic_velocity,
)
from tensorlbm.nebb_bodyforce import (
    BodyForceChannel,
    ChannelConfig,
    analytic_poiseuille,
    profile_errors,
)

ANCHOR = "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S"

# Pre-declared thresholds
OWNER_THRESHOLD = 0.15
ECOSYSTEM_THRESHOLD = 0.03
NO_SLIP_TOL = 0.01  # of u_max


def _pointwise_max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    interior = slice(1, -1)
    denom = np.maximum(np.abs(ref[interior]), 1e-12)
    return float((np.abs(got[interior] - ref[interior]) / denom).max())


def run_arm_d2q9(ny: int, nu: float, u_max: float, steps: int, scheme: str = "edm") -> dict:
    """Arm A: D2Q9 NEBB module."""
    h = ny - 1
    y_c = (h - 1) / 2.0
    gx = 2.0 * nu * u_max / (y_c * (h - y_c))
    cfg = ChannelConfig(
        nx=8, ny=ny, nu=nu, gx=gx, scheme=scheme,
        wall="force_corrected_nebb", steps=steps, steady_tol=1e-13,
    )
    res = BodyForceChannel(cfg).run()
    ref = analytic_poiseuille(ny, gx, nu, 0.0)
    got = res.ux_corrected.mean(axis=0)
    err = profile_errors(got, ref)
    u_max_ref = float(ref.max())
    return {
        "arm": "D2Q9_NEBB",
        "scheme": scheme,
        "ny": ny,
        "pointwise_err": float(_pointwise_max_rel_err(got, ref)),
        "u_max_got": float(got.max()),
        "u_max_ref": u_max_ref,
        "u_max_rel_err": float(err["umax_rel_err"]),
        "no_slip_ok": bool(abs(float(got[0]) - float(ref[0])) / max(u_max_ref, 1e-12) < NO_SLIP_TOL),
        "ux_profile": got.tolist(),
        "ref_profile": ref.tolist(),
        "y": np.arange(ny, dtype=np.float64).tolist(),
    }


def run_arm_d3q19(ny: int, nu: float, u_max: float, steps: int) -> dict:
    """Arm B: D3Q19 main solver."""
    cfg = GPUCorePathConfig(
        nx=8, ny=ny, nz=8, tau=nu * 3.0 + 0.5, u_max_target=u_max, init="rest",
        precision="FP64FP64", backend="eager", steps=steps,
        label=f"xval_ny{ny}",
    )
    solver = GPUCorePathSolver(cfg)
    solver.run(cfg.steps)
    y, u_got = solver.channel_profile()
    _, u_ref_all = channel_analytic_velocity(cfg)
    u_ref = u_ref_all[1 : cfg.ny - 1].to("cpu").numpy()
    u_got_np = u_got.numpy()
    u_max_ref = float(u_ref.max())
    return {
        "arm": "D3Q19_main",
        "ny": ny,
        "pointwise_err": float(_pointwise_max_rel_err(u_got_np, u_ref)),
        "u_max_got": float(u_got_np.max()),
        "u_max_ref": u_max_ref,
        "u_max_rel_err": float(abs(float(u_got_np.max()) - u_max_ref) / max(u_max_ref, 1e-12)),
        "no_slip_ok": bool(abs(float(u_got_np[0]) - float(u_ref[0])) / max(u_max_ref, 1e-12) < NO_SLIP_TOL),
        "ux_profile": u_got_np.tolist(),
        "ref_profile": u_ref.tolist(),
        "y": y.numpy().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure")
    args = parser.parse_args()

    nu, u_max = 0.2, 0.05
    results = {
        "anchor": ANCHOR,
        "predeclared_metric": "max pointwise relative error (interior fluid rows)",
        "owner_threshold": OWNER_THRESHOLD,
        "ecosystem_threshold": ECOSYSTEM_THRESHOLD,
        "arms": {},
    }

    for ny in [10, 20]:
        for scheme in ["edm", "guo"]:
            key = f"D2Q9_{scheme}_ny{ny}"
            results["arms"][key] = run_arm_d2q9(ny, nu, u_max, 200_000, scheme)
        key = f"D3Q19_ny{ny}"
        results["arms"][key] = run_arm_d3q19(ny, nu, u_max, 8000)

    # Summary
    d2q9_err = results["arms"]["D2Q9_edm_ny10"]["pointwise_err"]
    d3q19_err = results["arms"]["D3Q19_ny10"]["pointwise_err"]
    results["summary"] = {
        "tensorlbm_d2q9_pointwise_ny10": d2q9_err,
        "tensorlbm_d2q9_pointwise_ny20": results["arms"]["D2Q9_edm_ny20"]["pointwise_err"],
        "tensorlbm_d3q19_pointwise_ny10": d3q19_err,
        "tensorlbm_d3q19_pointwise_ny20": results["arms"]["D3Q19_ny20"]["pointwise_err"],
        "pass_owner_15pct": bool(d2q9_err < OWNER_THRESHOLD and d3q19_err < OWNER_THRESHOLD),
        "pass_ecosystem_3pct": bool(d2q9_err < ECOSYSTEM_THRESHOLD and d3q19_err < ECOSYSTEM_THRESHOLD),
        "no_slip_d2q9": results["arms"]["D2Q9_edm_ny10"]["no_slip_ok"],
        "no_slip_d3q19": results["arms"]["D3Q19_ny10"]["no_slip_ok"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for key, color, ax_idx in [
            ("D2Q9_edm_ny10", "tab:blue", 0),
            ("D2Q9_edm_ny20", "tab:cyan", 0),
            ("D3Q19_ny10", "tab:red", 1),
            ("D3Q19_ny20", "tab:orange", 1),
        ]:
            r = results["arms"][key]
            ax = axes[ax_idx]
            y = np.array(r["y"])
            ax.plot(y, r["ref_profile"], "k--", lw=1.0, label="analytic" if "ny10" in key else None)
            ax.plot(y, r["ux_profile"], "o-", ms=3, color=color,
                    label=f"{key} (err {r['pointwise_err'] * 100:.3f} %)")
        axes[0].set_title("D2Q9 NEBB module")
        axes[1].set_title("D3Q19 main solver")
        for ax in axes:
            ax.set_xlabel("y (lattice)")
            ax.set_ylabel("u_x")
            ax.legend(fontsize=8)
        fig.suptitle("Poiseuille channel, body force in x — steady profile vs analytic")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=140)
        print(f"wrote {args.figure}")
        plt.close(fig)


if __name__ == "__main__":
    main()
