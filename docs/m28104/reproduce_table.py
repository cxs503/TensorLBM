#!/usr/bin/env python3
"""m28104 — DFG 2D-2 (Re=100) lift deviation reproduction: Cl/Cd vs grid/τ/resolution.

Reproduction script for the LBM-P0-DFG root-cause investigation.  It re-scores
the frozen series of the existing LBM cylinder case
(``tensorlbm.validation.cylinder_dfg``) with both the *legacy* (peak-pick over
the whole record) and the *fixed* (fundamental at the shedding frequency,
transient excluded) estimators, and emits a machine-readable Cl/Cd numerical
table with the full grid/τ/resolution provenance.

The legacy column reproduces the +35 % headline deviation; the fixed column is
the post-m16827/m23014 estimator.  New grid levels (D48/D56) are appended when
their artifacts exist.

Usage
-----
    python docs/m28104/reproduce_table.py --levels 20 28 40 48 56
    python docs/m28104/reproduce_table.py --rejudge-only

The table is written to ``docs/m28104/cl_cd_table.json`` (and printed as text).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402

ART = REPO / "benchmarks/verified/cylinder/re100_dfg/artifacts"


def load_frozen(D: int, tag: str = "u0833") -> dict | None:
    """Load a frozen level artifact (no re-run)."""
    p = ART / f"level_D{D}_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def rejudge_level(D: int, tag: str = "u0833", transient_periods: float = 5.0) -> dict | None:
    """Re-score one frozen series with the fixed estimator (no re-run)."""
    spath = ART / f"level_D{D}_{tag}.series.npz"
    jpath = ART / f"level_D{D}_{tag}.json"
    if not (spath.exists() and jpath.exists()):
        return None
    old = json.loads(jpath.read_text())
    z = np.load(spath)
    T = cdfg.CaseConfig(D_cells=D, u_mean_lb=0.0833, n_steps=10**9,
                        transient_periods=transient_periods,
                        avg_periods=transient_periods).period_steps
    skip = int(round(transient_periods * T))
    cd = z["cd"].astype("float64")[skip:]
    cl = z["cl"].astype("float64")[skip:]
    dp = z["dp"].astype("float64")[skip:]
    amp = cdfg.limit_cycle_amplitude(torch.tensor(cl), T, d_cells=D, u_mean_lb=0.0833)
    st = cdfg.strouhal_from_series(torch.tensor(cl, dtype=torch.float32), T,
                                   cdfg.band_mid(cdfg.REFERENCE_BANDS["2D-2"]["st"]),
                                   D, 0.0833)
    return {
        "D_cells": D,
        "nx": old["nx"], "ny": old["ny"], "cells": old["cells"],
        "dx": old["dx"], "tau": old["tau"], "nu_lb": old["nu_lb"],
        "u_mean_lb": old["u_mean_lb"], "n_steps": old["n_steps"],
        "avg_window": old["avg_window"],
        "cd_legacy": old["cd"], "cl_amp_legacy": old["cl_amp"],
        "st_legacy": old["st"], "dp_legacy": old["dp"],
        "cd_fixed": float(cd.mean()), "cl_amp_fixed": amp["amp"],
        "st_fixed": st["st_fft"], "dp_fixed": float(dp.mean()),
        "n_periods_fixed": round(len(cd) / T, 2),
    }


def main(argv: list[str] | None = None) -> int:
    import torch  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=int, nargs="+", default=[20, 28, 40])
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "cl_cd_table.json"))
    a = ap.parse_args(argv)

    bands = cdfg.REFERENCE_BANDS["2D-2"]
    rows = []
    for D in a.levels:
        r = rejudge_level(D)
        if r is None:
            print(f"[skip] no frozen series for D={D}", flush=True)
            continue
        rows.append(r)

    out = {
        "case": "CASE-CYLINDER-DRAG-RE100 (DFG 2D-2, Re=100)",
        "reference": {
            "cd_band": list(bands["cd"]),
            "cl_amp_band": list(bands["cl_amp"]),
            "st_band": list(bands["st"]),
            "dp_half_rho_u2_band": list(bands["dp_half_rho_u2"]),
        },
        "rows": rows,
    }
    path = pathlib.Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    hdr = ("D  nx  ny  cells  tau   cd_leg  cd%_leg  cl_leg  cl%_leg  "
           "cd_fix  cd%_fix  cl_fix  cl%_fix  st_fix  st%_fix  dp_fix  dp%_fix")
    print(hdr)
    for r in rows:
        cd_le = cdfg.band_error(r["cd_legacy"], bands["cd"])["err_pct_vs_mid"]
        cl_le = cdfg.band_error(r["cl_amp_legacy"], bands["cl_amp"])["err_pct_vs_mid"]
        cd_fx = cdfg.band_error(r["cd_fixed"], bands["cd"])["err_pct_vs_mid"]
        cl_fx = cdfg.band_error(r["cl_amp_fixed"], bands["cl_amp"])["err_pct_vs_mid"]
        st_fx = cdfg.band_error(r["st_fixed"], bands["st"])["err_pct_vs_mid"]
        dp_fx = cdfg.band_error(r["dp_fixed"], bands["dp_half_rho_u2"])["err_pct_vs_mid"]
        print(f"{r['D_cells']:>2} {r['nx']:>4} {r['ny']:>3} {r['cells']:>6} "
              f"{r['tau']:.4f} {r['cd_legacy']:.4f} {cd_le:>+7.2f} "
              f"{r['cl_amp_legacy']:.4f} {cl_le:>+7.2f} "
              f"{r['cd_fixed']:.4f} {cd_fx:>+7.2f} "
              f"{r['cl_amp_fixed']:.4f} {cl_fx:>+7.2f} "
              f"{r['st_fixed']:.4f} {st_fx:>+6.2f} "
              f"{r['dp_fixed']:.4f} {dp_fx:>+6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
