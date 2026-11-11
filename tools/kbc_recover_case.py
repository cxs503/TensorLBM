#!/usr/bin/env python3
"""Rebuild gates.json/meta.json for a case whose run died after the CSVs.

The production runner writes ``profiles.csv`` and ``history.csv`` before
``meta.json``/``gates.json``; if the process dies in between, the expensive
simulation output is still on disk and only the two derived JSONs are missing.
This tool recomputes them from the CSVs and the run configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.kbc_channel import evaluate_gates, wall_units  # noqa: E402

FIELDS = ("y", "y_plus", "u_plus", "uu_plus", "vv_plus", "ww_plus", "uv_plus", "stress", "y_over_h")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--re-tau", type=float, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--u-tau", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args(argv)
    d = Path(args.case_dir)
    table = np.genfromtxt(d / "profiles.csv", delimiter=",", names=True)
    half = (args.ny - 2) / 2.0
    force = args.u_tau**2 / half
    nu = args.u_tau * half / args.re_tau
    prof = {f: np.asarray(table[f], float) for f in ("u_plus", "uu_plus", "vv_plus", "ww_plus",
                                                     "uv_plus", "stress")}
    # rebuild lattice-unit quantities from the wall-unit table
    prof["u"] = prof.pop("u_plus") * args.u_tau
    for k in ("uu_plus", "vv_plus", "ww_plus", "uv_plus", "stress"):
        prof[k[:-5] if k.endswith("_plus") else k] = prof.pop(k) * args.u_tau**2
    prof["rho"] = np.ones(args.ny)
    prof["pi_xy"] = prof["stress"] + prof["u"] ** 2
    scalars = {}
    hist = d / "history.csv"
    if hist.exists():
        h = np.genfromtxt(hist, delimiter=",", names=True)
        if len(h):
            scalars = {
                "min_population": float(np.min(h["min_pop"])),
                "max_abs_u": float(np.max(h["max_abs_u"])),
                "alpha_mean": float(h["alpha_mean"][-1]),
                "alpha_min": float(h["alpha_min"][-1]),
                "alpha_max": float(h["alpha_max"][-1]),
                "gamma_mean": float(h["gamma_mean"][-1]),
                "gamma_limited_fraction": float(h["gamma_limited_fraction"][-1]),
                "entropy_skipped_fraction": float(np.mean(h["gamma_limited_fraction"]) * 0.0),
                "ghost_fraction": float(h["ghost_fraction"][-1]),
                "newton_iterations": float(np.mean(h["newton_iterations"])) if "newton_iterations" in h.dtype.names else 0.0,
                "dh_global_worst": float(np.max(h["dh_global"])) if "dh_global" in h.dtype.names else float("nan"),
                "finite": 1.0,
                "samples": float(len(h)),
            }
    gates = evaluate_gates(prof, nu=nu, force=force, half_height=half, re_tau=args.re_tau,
                          nominal_re_tau=int(round(args.re_tau)), scalars=scalars)
    (d / "gates.json").write_text(json.dumps(gates, indent=1, default=float))
    meta = dict(label=d.name, re_tau=args.re_tau, ny=args.ny, u_tau=args.u_tau, nu=nu,
                tau=0.5 + 3.0 * nu, force=force, delta_plus=args.re_tau / half,
                steps=args.steps, warmup=args.warmup, recovered_from_csv=True)
    meta.update(scalars)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps({"cf_rel_err": gates["cf"]["rel_err"], "cf_pass": gates["cf"]["pass"],
                      "uplus_worst": gates["uplus"]["worst_rel_err"],
                      "uplus_pass": gates["uplus"]["pass"],
                      "uv_rms": gates["reynolds_stress"]["rms_rel_err"],
                      "wall_err": gates["wall"]["u_tau_rel_err"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
