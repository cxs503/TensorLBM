#!/usr/bin/env python3
"""XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S — Round 3 (machine-gate evidence).

CLI entry that re-runs the force-corrected NEBB body-force channel
cross-validation end to end and writes the run artifacts + evidence bundle
that the maturity gate re-computes (no self-declared grades).

Arms (all executed in this process, zero hand-copied numbers)
--------------------------------------------------------------
* **A — D2Q9, repo NEBB module** (``tensorlbm.nebb_bodyforce``):
  ``BodyForceChannel`` with ``scheme=edm`` and
  ``wall=force_corrected_nebb`` (the OSS vivsim recipe: BGK + EDM/Guo
  forcing + force-corrected NEBB wall, rows 0/ny-1 are wall nodes).
  Reference: closed-form Poiseuille ``analytic_poiseuille(ny, gx, nu, 0.0)``.
* **B — D3Q19, main solver** (``tensorlbm.gpu_core_path``):
  ``GPUCorePathSolver`` (``collide_guo_into`` + ``apply_channel_walls``
  half-way bounce-back).  Reference: ``channel_analytic_velocity``.
* **OSS reference (haimingz/vivsim D2Q9)**: numbers from the board-audit
  rerun ``docs/board_audit/m17963/rerun/xval_nebb_bodyforce_driver.json``
  (independently reproduced, bit-identical to the Round-1 delivery).  The
  OSS arm itself is NOT re-executed here (``jax`` absent => NOT_RUN,
  honest scope).

Pre-declared decision (unchanged since Round 1, per audit m17963 F2)
--------------------------------------------------------------------
* Primary metric: **max pointwise relative error** of the steady ``ux(y)``
  profile vs the closed-form Poiseuille solution (interior fluid rows).
* PASS if ``pointwise_max_rel_err < 0.15`` (owner standard);
  ecosystem stretch ``< 0.03``.
* Sensitivity: grid refinement NY = 10 vs NY = 20 (>= 2 points).
* No-slip check: first fluid row equals the analytic parabola value at that
  row within 1 % of u_max (no spurious wall slip).

Usage
-----
    python3 tools/nebb_bodyforce_xval.py \
        --out artifacts/m18744/xval_nebb_bodyforce_r3.json \
        --figure artifacts/m18744/fig_nebb_bodyforce_r3_profile.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.gpu_core_path import (  # noqa: E402
    GPUCorePathConfig,
    GPUCorePathSolver,
    channel_analytic_velocity,
)
from tensorlbm.nebb_bodyforce import (  # noqa: E402
    BodyForceChannel,
    ChannelConfig,
    analytic_poiseuille,
    profile_errors,
)

ANCHOR = "XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S"
OSS_AUDIT_JSON = _REPO / "xval" / "oss_vivsim_d2q9_reference_m17963.json"

# Pre-declared thresholds (NOT tuned after seeing results)
OWNER_THRESHOLD = 0.15
ECOSYSTEM_THRESHOLD = 0.03
NO_SLIP_TOL = 0.01  # of u_max


def _pointwise_max_rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    """Pre-declared metric: max pointwise relative error (interior rows)."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    interior = slice(1, -1)
    denom = np.maximum(np.abs(ref[interior]), 1e-12)
    return float((np.abs(got[interior] - ref[interior]) / denom).max())


def run_d2q9_nebb(ny: int, nu: float, u_max: float, steps: int) -> dict:
    """Arm A: repo D2Q9 NEBB module vs analytic Poiseuille (offset-0)."""
    h = ny - 1  # offset-0 convention: no-slip plane on the boundary nodes
    y_c = (h - 1) / 2.0  # centre row (rows 0..ny-1, walls at 0 and ny-1)
    gx = 2.0 * nu * u_max / (y_c * (h - y_c))
    cfg = ChannelConfig(
        nx=8, ny=ny, nu=nu, gx=gx, scheme="edm",
        wall="force_corrected_nebb", steps=steps, steady_tol=1e-13,
    )
    res = BodyForceChannel(cfg).run()
    ref = analytic_poiseuille(ny, gx, nu, 0.0)
    got = res.ux_corrected.mean(axis=0)
    err = profile_errors(got, ref)
    return {
        "arm": "tensorlbm_d2q9_nebb_module",
        "scheme": "BGK+EDM, force-corrected NEBB wall (rows 0/ny-1)",
        "ny": ny, "nu": nu, "gx": gx, "u_max_target": u_max, "steps": steps,
        "steps_used": res.steps_used,
        "steady_residual": res.steady_residual,
        "mass_drift": res.mass_drift,
        "u_max_ref": float(ref.max()),
        "u_max_got": float(got.max()),
        "u_max_rel_err": err["umax_rel_err"],
        "pointwise_max_rel_err": _pointwise_max_rel_err(got, ref),
        "profile_rel_l2": err["l2_rel_global"],
        "wall_row_velocity_got": float(got[0]),
        "wall_row_velocity_ref": float(ref[0]),
        "y": np.arange(ny, dtype=np.float64).tolist(),
        "u_got": got.tolist(),
        "u_ref": ref.tolist(),
    }


def run_d3q19_solver(ny: int, nu: float, u_max: float, steps: int) -> dict:
    """Arm B: REAL main solver (GPUCorePathSolver, D3Q19) vs analytic."""
    cfg = GPUCorePathConfig(
        nx=8, ny=ny, nz=8, tau=nu * 3.0 + 0.5, u_max_target=u_max, init="rest",
        precision="FP64FP64", backend="eager", steps=steps,
        label=f"xval_r3_ny{ny}",
    )
    solver = GPUCorePathSolver(cfg)
    solver.run(cfg.steps)
    y, u_got = solver.channel_profile()
    _, u_ref_all = channel_analytic_velocity(cfg)
    u_ref = u_ref_all[1 : cfg.ny - 1].to("cpu").numpy()
    u_got_np = u_got.numpy()
    u_max_ref = float(u_ref.max())
    u_max_got = float(u_got_np.max())
    return {
        "arm": "tensorlbm_d3q19_main_solver",
        "scheme": "BGK+Guo, half-way bounce-back (walls at 0.5/ny-1.5)",
        "ny": ny, "nu": nu, "u_max_target": u_max, "steps": steps,
        "u_max_ref": u_max_ref,
        "u_max_got": u_max_got,
        "u_max_rel_err": abs(u_max_got - u_max_ref) / u_max_ref,
        "pointwise_max_rel_err": _pointwise_max_rel_err(u_got_np, u_ref),
        "profile_rel_l2": float(np.sqrt(np.mean((u_got_np - u_ref) ** 2)) / u_max_ref),
        "wall_row_velocity_got": float(u_got_np[0]),
        "wall_row_velocity_ref": float(u_ref[0]),
        "y": y.numpy().tolist(),
        "u_got": u_got_np.tolist(),
        "u_ref": u_ref.tolist(),
    }


def _no_slip(run: dict) -> bool:
    return (
        abs(run["wall_row_velocity_got"] - run["wall_row_velocity_ref"])
        / run["u_max_ref"] < NO_SLIP_TOL
    )


def make_figure(out_png: Path, runs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, color in (
        ("d2q9_ny10", "tab:blue"),
        ("d2q9_ny20", "tab:cyan"),
        ("d3q19_ny10", "tab:red"),
        ("d3q19_ny20", "tab:orange"),
    ):
        r = runs[key]
        ax = axes[0 if key.startswith("d2q9") else 1]
        ax.plot(r["y"], r["u_ref"], "k--", lw=1.0, label="analytic" if key.endswith("ny10") else None)
        ax.plot(r["y"], r["u_got"], "o-", ms=3, color=color,
                label=f"{key} (err {r['pointwise_max_rel_err'] * 100:.3f} %)")
    axes[0].set_title("D2Q9 NEBB module (repo)")
    axes[1].set_title("D3Q19 main solver (repo)")
    for ax in axes:
        ax.set_xlabel("y (lattice)")
        ax.set_ylabel("u_x")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Poiseuille channel, body force in x — steady profile vs analytic "
        "(pre-declared pointwise metric)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--figure", default="", help="optional PNG output path")
    ap.add_argument("--steps-d2q9", type=int, default=200_000)
    ap.add_argument("--steps-d3q19", type=int, default=8000)
    args = ap.parse_args(argv)

    nu, u_max = 0.2, 0.05
    runs = {
        "d2q9_ny10": run_d2q9_nebb(10, nu, u_max, args.steps_d2q9),
        "d2q9_ny20": run_d2q9_nebb(20, nu, u_max, args.steps_d2q9),
        "d3q19_ny10": run_d3q19_solver(10, nu, u_max, args.steps_d3q19),
        "d3q19_ny20": run_d3q19_solver(20, nu, u_max, args.steps_d3q19),
    }

    # OSS reference arm (board-audit rerun; jax NOT_RUN on this host)
    oss = None
    if OSS_AUDIT_JSON.exists():
        oss_raw = json.loads(OSS_AUDIT_JSON.read_text())
        oss_ny10 = oss_raw["runs"]["vivsim_bgk_edm"]
        oss = {
            "arm": "oss_vivsim_d2q9",
            "source": "board-audit rerun m17963, in-repo copy xval/oss_vivsim_d2q9_reference_m17963.json (sha256 c62ff2bd...f2c197; jax NOT_RUN here)",
            "u_max": oss_ny10["umax"],
            "u_max_analytic": oss_ny10["umax_analytic"],
            "u_max_rel_err": oss_ny10["umax_rel_err"],
            "pointwise_max_rel_err": oss_ny10["max_rel_err_vs_analytic_interior"],
            "profile_rel_l2": oss_ny10["err_global_vs_analytic"],
        }

    primary = runs["d3q19_ny10"]["pointwise_max_rel_err"]
    verdict = {
        "predeclared_metric": "max pointwise relative error (interior fluid rows)",
        "owner_threshold": OWNER_THRESHOLD,
        "ecosystem_threshold": ECOSYSTEM_THRESHOLD,
        "tensorlbm_d3q19_pointwise_ny10": primary,
        "tensorlbm_d3q19_pointwise_ny20": runs["d3q19_ny20"]["pointwise_max_rel_err"],
        "tensorlbm_d2q9_pointwise_ny10": runs["d2q9_ny10"]["pointwise_max_rel_err"],
        "tensorlbm_d2q9_pointwise_ny20": runs["d2q9_ny20"]["pointwise_max_rel_err"],
        "pass_owner_15pct": bool(primary < OWNER_THRESHOLD),
        "pass_ecosystem_3pct": bool(primary < ECOSYSTEM_THRESHOLD),
        "no_slip_d2q9": _no_slip(runs["d2q9_ny10"]),
        "no_slip_d3q19": _no_slip(runs["d3q19_ny10"]),
    }

    out = {
        "anchor": ANCHOR,
        "mission_id": 18744,
        "round": 3,
        "problem": "2D Poiseuille channel, periodic x, no-slip top/bottom, body force in x",
        "params": {"nu": nu, "u_max_target": u_max, "nx": 8, "nz": 8,
                   "backend": "eager", "precision": "FP64FP64"},
        "runs": runs,
        "oss_reference": oss,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(verdict, indent=2))
    print(f"wrote {out_path}")
    if args.figure:
        fig_path = Path(args.figure)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        make_figure(fig_path, runs)
        print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
