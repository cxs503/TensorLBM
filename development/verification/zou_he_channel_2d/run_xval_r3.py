#!/usr/bin/env python3
"""XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION — round 3 (review) driver.

Round 2 (``m18793``) issued PASS on a worst ``dp_max`` of 4.15 %, but every
physics arm of that run violated the *pre-registered* steady-state gate
``drift < 1e-6`` (PREREG.md, metric table, last row): the reported drifts were
1.6e-4 … 2.6e-3, i.e. 160×–2 600× over the declared bound, and the arms were
stopped at 5 000 / 8 000 steps instead of the declared "≤ 40 000, steady-stop".

This driver re-executes the **same frozen code, same cases, same parameters**
and enforces the declared gate:

* ``drift`` still means ``max_y |ux(t) - ux(t-1000)| / max ux`` at the primary
  measurement column (``nx//2``), sampled every 1 000 steps;
* the march stops on ``drift < --drift-tol`` (declared default 1e-6) **or** at
  ``--steps``; ``steady_ok`` records which of the two happened;
* the whole drift / dp_max history is kept, so the convergence of the primary
  metric is a measured curve instead of a single point estimate;
* ``dp_max`` is recomputed from the artifact JSON by the unchanged
  ``compare_profile`` of round 2 (imported, not copied).

Nothing about the metric, the column, the threshold or the analytic reference is
changed here. The only new outputs are the gate status and the convergence
curve, i.e. the evidence round 2 lacked.

Usage
-----
    python3 run_xval_r3.py --out artifacts/r3_steady_gate.json
    python3 run_xval_r3.py --cases C1,C3 --steps 60000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location("rx", HERE / "run_xval.py")
rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rx)  # noqa: E402  (round-2 driver, frozen)

from tensorlbm.d2q9 import macroscopic  # noqa: E402

DTYPE = torch.float64

# pre-registered cases (PREREG.md §"Cases (fixed now)"), unchanged
CASES = {
    "C1": dict(nx=120, ny=18, mode="velocity", steps=40000, u_in=0.05),
    "C2": dict(nx=220, ny=34, mode="velocity", steps=40000, u_in=0.05),
    "C3": dict(nx=120, ny=18, mode="pressure", steps=40000, d_rho=0.004),
    "C4": dict(nx=220, ny=34, mode="pressure", steps=40000, d_rho=0.004),
    "PR": dict(nx=120, ny=18, mode="velocity-profile", steps=40000, u_max_in=0.06),
}
TAU = 0.8
DRIFT_TOL = 1e-6
SAMPLE_EVERY = 1000


def _profile_inlet(ny: int, u_max_in: float) -> torch.Tensor:
    h = ny - 1
    y = torch.arange(1, ny - 1, dtype=DTYPE)
    prof = torch.zeros(ny, dtype=DTYPE)
    prof[1:-1] = rx.u_poiseuille(y, h, u_max_in)
    return prof


def run_case(tag: str, *, steps: int, drift_tol: float, sample_every: int) -> dict:
    cfg = CASES[tag]
    nx, ny = cfg["nx"], cfg["ny"]
    cols = (nx // 2, nx - 4)
    kwargs: dict = dict(
        nx=nx, ny=ny, tau=TAU, steps=steps, measure_cols=cols,
        sample_every=sample_every, drift_tol=drift_tol,
    )
    nu = (TAU - 0.5) / 3.0
    if cfg["mode"] == "velocity":
        run = rx.run_channel(mode="velocity", u_in=cfg["u_in"], rho_out=1.0, **kwargs)
    elif cfg["mode"] == "velocity-profile":
        run = rx.run_channel(
            mode="velocity", u_in_profile=_profile_inlet(ny, cfg["u_max_in"]),
            rho_out=1.0, **kwargs,
        )
    else:
        d_rho = cfg["d_rho"]
        run = rx.run_channel(
            mode="pressure", rho_in=1.0 + d_rho / 2, rho_out=1.0 - d_rho / 2, **kwargs
        )

    h = ny - 1
    L = nx - 1
    if cfg["mode"] == "pressure":
        G = cfg["d_rho"] / 3.0 / L
        u_max_ana = G * h * h / (8.0 * nu)
        cmp_ = rx.compare_profile(run, col=nx // 2, u_max_ana=u_max_ana)
        G_out = G
    else:
        # C1/C2 (uniform velocity inlet): the amplitude is not predicted a
        # priori -> flux-matched shape test, exactly as round 2 did.
        # PR (parabolic inlet): amplitude *is* predicted a priori (u_max_in),
        # exactly as round 2 did -- mixing the two normalisations would make the
        # PR arm look better than round 2 reported, i.e. not a re-execution.
        u_max_ana = cfg.get("u_max_in", cfg.get("u_in"))
        cmp_ = rx.compare_profile(
            run, col=nx // 2, fit_amplitude=(cfg["mode"] == "velocity"),
            u_max_ana=u_max_ana,
        )
        G_out = None

    # convergence curve of the two metrics that matter (primary + gate)
    curve = [{"step": hh["step"], "drift": hh["drift"]} for hh in run["history"]]
    # dp_max as a function of time: rebuild the mid-column profile each sample
    # is not stored, so the curve of dp_max is only available at the end; we
    # instead report the per-1000-step amplitude, which is what dp_max tracks.
    curve_amp = [
        {"step": hh["step"], "umax": hh["umax"]} for hh in run["history"]
    ]
    return {
        "case": tag,
        "mode": cfg["mode"],
        "nx": nx,
        "ny": ny,
        "h": h,
        "nu": nu,
        "Re": (u_max_ana * h / nu) if u_max_ana else None,
        "Ma": (u_max_ana * math.sqrt(3.0)) if u_max_ana else None,
        "G": G_out,
        "u_max_ana": u_max_ana,
        "steps_requested": steps,
        "steps_used": run["steps"],
        "drift_tol": drift_tol,
        "drift": run["drift"],
        "steady_ok": bool(run["drift"] < drift_tol),
        "flux_err": rx.flux_error(run),
        "uy_max": run["uy_max"],
        "dp_max": cmp_["dp_max"],
        "dp_l2": cmp_["dp_l2"],
        "amplitude_err": cmp_["amplitude_err"],
        "amplitude_source": cmp_["amplitude_source"],
        "u_max_got": cmp_["u_max_got"],
        "wall_offset": cmp_["wall_offset"],
        "ux_from_lbm": cmp_["ux_from_lbm"],
        "ux_analytic": cmp_["ux_analytic"],
        "curve_drift": curve,
        "curve_umax": curve_amp,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/r3_steady_gate.json")
    ap.add_argument("--cases", default="C1,C2,C3,C4,PR")
    ap.add_argument("--steps", type=int, default=0, help="0 = per-case declared budget")
    ap.add_argument("--drift-tol", type=float, default=DRIFT_TOL)
    ap.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    outdir = (HERE / args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cases: dict = {}
    for tag in [t.strip() for t in args.cases.split(",") if t.strip()]:
        steps = args.steps or CASES[tag]["steps"]
        t1 = time.time()
        res = run_case(tag, steps=steps, drift_tol=args.drift_tol,
                       sample_every=args.sample_every)
        res["wall_time_s"] = time.time() - t1
        cases[tag] = res
        print(
            f"{tag:4s} steps={res['steps_used']:6d} drift={res['drift']:.3e} "
            f"steady={'YES' if res['steady_ok'] else 'NO ':3s} "
            f"dp_max={res['dp_max']*100:6.2f}% amp_err={res['amplitude_err']*100:+6.2f}% "
            f"({res['wall_time_s']:.0f}s)",
            flush=True,
        )
        Path(args.out if Path(args.out).is_absolute() else HERE / args.out).write_text(
            json.dumps({"cases": cases, "elapsed_s": time.time() - t0}, indent=1)
        )

    gating = [c for c in cases.values() if c["mode"] != "velocity-profile"]
    worst = max(c["dp_max"] for c in cases.values())
    worst_gating = max(c["dp_max"] for c in gating) if gating else None
    report = {
        "anchor": "XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION",
        "round": 3,
        "purpose": "re-execute round-2 arms under the pre-registered steady gate",
        "drift_tol": args.drift_tol,
        "sample_every": args.sample_every,
        "cases": cases,
        "verdict": {
            "primary_metric": "dp_max",
            "threshold": 0.15,
            "stretch": 0.03,
            "dp_max_worst_all": worst,
            "dp_max_worst_gating": worst_gating,
            "all_steady": all(c["steady_ok"] for c in cases.values()),
            "not_steady": [t for t, c in cases.items() if not c["steady_ok"]],
            "PASS_dp_max": worst < 0.15,
            "STRETCH_dp_max": worst < 0.03,
            "PASS_under_steady_gate": bool(worst < 0.15 and all(
                c["steady_ok"] for c in cases.values())),
        },
        "env": {
            "torch": torch.__version__,
            "threads": args.threads,
            "dtype": "float64",
        },
        "elapsed_s": time.time() - t0,
    }
    path = args.out if Path(args.out).is_absolute() else HERE / args.out
    Path(path).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdict"], indent=1))
    print(f"wrote {path}  ({report['elapsed_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())