#!/usr/bin/env python3
"""Re-execute the round-2 configurations and diff them against the committed
artifact, programmatically (no number is copied by hand).

Round 2 (``m18793``, merge ``60ac3409e``) published
``artifacts/zou_he_channel_results.json``.  A verification claim is only worth
what its re-execution is worth, so this script rebuilds every physics arm with
round 2's own call arguments -- including ``sample_every=500``, which is what
round 2 used and which (see ``REVIEW_ROUND3.md`` F3) is *not* the pre-registered
metric's 1 000-step window -- and reports the deviation of ``dp_max`` and of the
whole measured profile for each case.

Usage
-----
    python3 repro_round2.py --out artifacts/repro_round2.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("rx", HERE / "run_xval.py")
rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rx)

DTYPE = torch.float64

# exactly round 2's call arguments (see run_xval.py main(); steps as run)
ROUND2 = {
    "C1_coarse": dict(nx=120, ny=18, mode="velocity", steps=5000, u_in=0.05),
    "C2_fine": dict(nx=220, ny=34, mode="velocity", steps=8000, u_in=0.05),
    "C3_coarse": dict(nx=120, ny=18, mode="pressure", steps=5000, d_rho=0.004),
    "C4_fine": dict(nx=220, ny=34, mode="pressure", steps=8000, d_rho=0.004),
    "P_inlet_profile": dict(nx=120, ny=18, mode="velocity-profile", steps=5000,
                            u_max_in=0.06),
}
SAMPLE_EVERY = 500      # round-2 default
DRIFT_TOL = 1e-7        # round-2 default


def _run(tag: str) -> dict:
    cfg = ROUND2[tag]
    nx, ny = cfg["nx"], cfg["ny"]
    kw = dict(nx=nx, ny=ny, tau=0.8, steps=cfg["steps"],
              measure_cols=(nx // 2, nx - 4),
              sample_every=SAMPLE_EVERY, drift_tol=DRIFT_TOL)
    nu = 0.1
    if cfg["mode"] == "pressure":
        d = cfg["d_rho"]
        run = rx.run_channel(mode="pressure", rho_in=1.0 + d / 2,
                             rho_out=1.0 - d / 2, **kw)
        G = d / 3.0 / (nx - 1)
        u_max = G * (ny - 1) ** 2 / (8.0 * nu)
        return {"run": run, "cmp": rx.compare_profile(run, col=nx // 2, u_max_ana=u_max)}
    if cfg["mode"] == "velocity":
        run = rx.run_channel(mode="velocity", u_in=cfg["u_in"], rho_out=1.0, **kw)
        return {"run": run, "cmp": rx.compare_profile(run, col=nx // 2, fit_amplitude=True)}
    h = ny - 1
    y = torch.arange(1, ny - 1, dtype=DTYPE)
    prof = torch.zeros(ny, dtype=DTYPE)
    prof[1:-1] = rx.u_poiseuille(y, h, cfg["u_max_in"])
    run = rx.run_channel(mode="velocity", u_in_profile=prof, rho_out=1.0, **kw)
    return {"run": run,
            "cmp": rx.compare_profile(run, col=nx // 2, u_max_ana=cfg["u_max_in"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/repro_round2.json")
    ap.add_argument("--cases", default=",".join(ROUND2))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    ref = json.loads((HERE / "artifacts" / "zou_he_channel_results.json").read_text())
    out: dict = {}
    for tag in [t.strip() for t in args.cases.split(",") if t.strip()]:
        got = _run(tag)
        # round 2 stored the mid-column entry under "<tag>_col<nx//2>"
        key = f"{tag}_col{ROUND2[tag]['nx'] // 2}" if tag != "P_inlet_profile" else tag
        prior = ref["results"][key]
        prof_now = np.array(got["cmp"]["ux_from_lbm"])
        prof_then = np.array(prior["ux_from_lbm"])
        out[tag] = {
            "key_round2": key,
            "dp_max_round2": prior["dp_max"],
            "dp_max_reproduced": got["cmp"]["dp_max"],
            "dp_max_abs_delta": abs(got["cmp"]["dp_max"] - prior["dp_max"]),
            "profile_max_abs_delta": float(np.abs(prof_now - prof_then).max()),
            "profile_bitwise_equal": bool(np.array_equal(prof_now, prof_then)),
            "amplitude_err_round2": prior.get("amplitude_err"),
            "amplitude_err_reproduced": got["cmp"]["amplitude_err"],
            "drift_round2": prior.get("drift"),
            "drift_reproduced": got["run"]["drift"],
            "steps": got["run"]["steps"],
        }
        print(
            f"{tag:16s} dp_max {prior['dp_max']*100:6.3f}% -> "
            f"{got['cmp']['dp_max']*100:6.3f}%  |d|={out[tag]['dp_max_abs_delta']:.2e}  "
            f"profile_bitwise={out[tag]['profile_bitwise_equal']}",
            flush=True,
        )

    path = HERE / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())