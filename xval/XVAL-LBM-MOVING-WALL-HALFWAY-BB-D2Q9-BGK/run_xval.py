#!/usr/bin/env python3
"""Cross-validation driver — XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK.

Staged so that every invocation stays a short (<2 min) local run:

    python run_xval.py --stage chunk1      # couette + reference + delta=4
    python run_xval.py --stage chunk2      # ablation + parameter points
    python run_xval.py --stage finalize    # assemble artifacts/summary.json + gate
    python run_xval.py --stage quick       # tiny smoke subset

Every per-case number lands in ``artifacts/case_*.json``; ``finalize`` only
aggregates those files (the report is generated from ``summary.json`` only, so
no number is copied by hand).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import mwbb

REF = {"tau": 0.8, "delta_bl": 8.0, "gap_over_delta": 6.0, "U0": 0.02,
       "n_periods": 6.0, "sample_stride": 4, "nx": 4}


def stokes2_cfg(delta_bl: float = 8.0, tau: float = 0.8, U0: float = 0.02,
                gap_over_delta: float = 6.0, variant: str = "ladd_pre_local",
                n_periods: float = 6.0, sample_stride: int = 4, nx: int = 4,
                tag: str | None = None) -> dict:
    ny = int(round(gap_over_delta * delta_bl)) + 2
    r = mwbb.run_stokes2(ny=ny, nx=nx, tau=tau, u0=U0, delta_bl=delta_bl,
                         variant=variant, n_periods=n_periods,
                         sample_stride=sample_stride)
    r["tag"] = tag or (f"stokes2_d{delta_bl:g}_tau{tau:g}_U{U0:g}"
                       f"_gap{gap_over_delta:g}_{variant}")
    r["probes"] = [mwbb.harmonic_fit(r, e) for e in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
    return r


def couette_cfg(H: int = 32, tau: float = 0.8, U0: float = 0.05,
                variant: str = "ladd_pre_local", nx: int = 4,
                n_steps: int = 60000) -> dict:
    r = mwbb.run_couette(ny=H + 2, nx=nx, tau=tau, u0=U0, variant=variant,
                         n_steps=n_steps)
    r["tag"] = f"couette_H{H}_tau{tau:g}_U{U0:g}_{variant}"
    return r


KEEP = ("tag", "case", "variant", "ny", "nx", "tau", "nu", "H_eff", "U0", "Ma",
        "Re", "omega", "period_steps", "delta_bl", "gap_over_delta", "n_steps",
        "steady", "l2_rel", "l2_rel_pct", "max_rel_pct_cf_umax",
        "max_err_pct_U0_full", "max_err_pct_U0_resolved", "argmax_row_resolved",
        "far_field_floor_pct", "mass_drift_pct", "slope_err_pct", "y_wall_fit",
        "probes")


def _slim(r: dict) -> dict:
    return {k: r[k] for k in KEEP if k in r}


def _write(out: Path, name: str, r: dict, full: bool = True) -> None:
    (out / f"case_{name}.json").write_text(json.dumps(r, indent=1))
    if full:
        (out / f"slim_{name}.json").write_text(json.dumps(_slim(r), indent=1))


def _logf(msg: str, t0: float) -> None:
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def stage_chunk1(out: Path) -> None:
    t0 = time.time()
    for H in (16, 32):
        r = couette_cfg(H=H)
        _write(out, r["tag"], r)
        _logf(f"{r['tag']:36s} max={r['max_rel_pct_cf_umax']:.3e}% "
              f"y_wall={r['y_wall_fit']:+.2e} steady={r['steady']} n={r['n_steps']}", t0)
    r = couette_cfg(H=32, tau=1.2)
    _write(out, r["tag"], r)
    _logf(f"{r['tag']:36s} max={r['max_rel_pct_cf_umax']:.3e}% n={r['n_steps']}", t0)
    # ablation on Couette: the static wall must not produce the Couette profile
    r = couette_cfg(H=32, variant="static_bb")
    _write(out, r["tag"], r)
    _logf(f"{r['tag']:36s} max={r['max_rel_pct_cf_umax']:.3e}%", t0)

    r = stokes2_cfg(**REF, tag="stokes2_REF")
    _write(out, "stokes2_REF", r)
    _logf(f"stokes2 REF max_resolved={r['max_err_pct_U0_resolved']:.4f}% "
          f"max_full={r['max_err_pct_U0_full']:.4f}% n={r['n_steps']}", t0)

    r = stokes2_cfg(**{**REF, "delta_bl": 4.0}, tag="stokes2_res_d4")
    _write(out, "stokes2_res_d4", r)
    _logf(f"resolution d=4 max_resolved={r['max_err_pct_U0_resolved']:.4f}%", t0)


def stage_chunk2(out: Path) -> None:
    t0 = time.time()
    for v in ("static_bb", "eq_wall_node", "ladd_sign_flip", "ladd_precollision",
              "ladd_pre_unitrho"):
        r = stokes2_cfg(**{**REF, "n_periods": 4.0, "sample_stride": 8},
                        variant=v, tag=f"stokes2_abl_{v}")
        _write(out, f"stokes2_abl_{v}", r)
        _logf(f"ablation {v:18s} max_resolved={r['max_err_pct_U0_resolved']:9.4f}% "
              f"amp@1d={r['probes'][2]['amp_err_pct']:+9.3f}%", t0)
    for tau in (1.8,):
        r = stokes2_cfg(**{**REF, "tau": tau}, tag=f"stokes2_tau{tau:g}")
        _write(out, f"stokes2_tau{tau:g}", r)
        _logf(f"tau={tau:4g} max_resolved={r['max_err_pct_U0_resolved']:9.4f}%", t0)
    for U0 in (0.1, 0.005):
        r = stokes2_cfg(**{**REF, "U0": U0}, tag=f"stokes2_U{U0:g}")
        _write(out, f"stokes2_U{U0:g}", r)
        _logf(f"U0={U0:6g} Ma={U0/math.sqrt(1/3):.4f} "
              f"max_resolved={r['max_err_pct_U0_resolved']:9.4f}%", t0)


def stage_finalize(out: Path) -> None:
    files = sorted(out.glob("case_*.json"))
    cases = {}
    for f in files:
        r = json.loads(f.read_text())
        cases[f.stem.replace("case_", "")] = r
    cou = [c for c in cases.values() if c.get("case") == "couette"]
    st2 = [c for c in cases.values() if c.get("case") == "stokes2"]
    ref = cases["stokes2_REF"]
    results = {
        "couette": [_slim(c) for c in sorted(cou, key=lambda x: x["tag"])],
        "stokes2_reference": _slim(ref),
        "stokes2_ablation": [_slim(c) for c in sorted(st2, key=lambda x: x["tag"])
                             if c["variant"] != "ladd_pre_local"],
        "stokes2_resolution": [_slim(cases[k]) for k in ("stokes2_res_d4", "stokes2_REF")
                              if k in cases],
        "stokes2_parameters": [_slim(cases[k]) for k in
                               ("stokes2_tau1.8", "stokes2_U0.1", "stokes2_U0.005")
                               if k in cases],
    }
    gate = _evaluate(results, ref)
    summary = {
        "anchor": "XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK",
        "mission_id": 18794,
        "lattice": "D2Q9", "collision": "BGK", "dtype": "float64",
        "wall_closure": "pre-streaming half-way bounce-back + Ladd momentum term "
                        "2*w*rho_w*(c.u_w)/cs^2, rho_w from the pre-collision wall populations",
        "reference_config": REF,
        "n_cases": len(cases),
        "results": results,
        "gate": gate,
        "artifact_files": [f.name for f in files],
        "elapsed_s": None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("gate ->", json.dumps({k: (v["ok"], v["value"]) for k, v in gate["checks"].items()},
                                indent=0))
    print("verdict =", gate["verdict"], "failed:", gate["failed"])


def _evaluate(results: dict, ref: dict) -> dict:
    """Apply the predeclared thresholds (PREREG.md M1..M8) mechanically."""
    checks: dict[str, dict] = {}

    def chk(cid, desc, value, threshold, ok, unit=""):
        checks[cid] = {"desc": desc, "value": value, "threshold": threshold,
                       "ok": bool(ok), "unit": unit}

    m1 = ref["max_err_pct_U0_resolved"]
    chk("M1", "Stokes-2nd max |du|/U0 over eta<=3 delta_bl (reference config)",
        m1, 15.0, m1 <= 15.0, "%")
    p1 = min(ref["probes"], key=lambda p: abs(p["eta_over_delta"] - 1.0))
    p2 = min(ref["probes"], key=lambda p: abs(p["eta_over_delta"] - 2.0))
    amp = max(abs(p1["amp_err_pct"]), abs(p2["amp_err_pct"]))
    chk("M2", "Stokes-2nd amplitude error at eta=delta and 2delta", amp, 5.0,
        amp <= 5.0, "%")
    ph = max(abs(p1["phase_err_deg"]), abs(p2["phase_err_deg"]))
    chk("M3", "Stokes-2nd phase error at eta=delta and 2delta", ph, 5.0, ph <= 5.0, "deg")
    cou = [c for c in results["couette"]
           if c["H_eff"] == 32 and c["tau"] == 0.8 and c["variant"] == "ladd_pre_local"]
    if cou:
        c0 = cou[0]
        chk("M4", "Couette pointwise max |du|/umax (H=32, tau=0.8, U0=0.05)",
            c0["max_rel_pct_cf_umax"], 2.0, c0["max_rel_pct_cf_umax"] <= 2.0, "%")
        chk("M5", "Couette fitted effective wall position |y_wall|", c0["y_wall_fit"],
            0.01, abs(c0["y_wall_fit"]) <= 0.01, "lattice")
    allc = ([*results["couette"], ref, *results["stokes2_ablation"],
             *results["stokes2_resolution"], *results["stokes2_parameters"]])
    worst_mass = max(abs(c["mass_drift_pct"]) for c in allc)
    chk("M6", "max |mass drift| over all cases", worst_mass, 1e-6,
        worst_mass <= 1e-6, "%")
    by_var = {c["variant"]: c["max_err_pct_U0_resolved"] for c in results["stokes2_ablation"]}
    by_var["ladd_pre_local"] = ref["max_err_pct_U0_resolved"]
    ok7 = by_var["ladd_pre_local"] < by_var["eq_wall_node"] < by_var["static_bb"]
    chk("M7", "ablation ranking err(ladd) < err(eq_wall_node) < err(static_bb)",
        {k: by_var[k] for k in ("ladd_pre_local", "eq_wall_node", "static_bb")},
        "strict", ok7)
    res = {c["delta_bl"]: c["max_err_pct_U0_resolved"] for c in results["stokes2_resolution"]}
    if 4.0 in res and 8.0 in res:
        chk("M8", "resolution err(delta=8) < err(delta=4)",
            {"d4": res[4.0], "d8": res[8.0]}, "strict", res[8.0] < res[4.0])
    cou_static = [c for c in results["couette"] if c["variant"] == "static_bb"]
    if cou_static and cou:
        chk("M9", "Couette ablation: static wall cannot reproduce u(y)=U0(y-.5)/H",
            {"ladd": cou[0]["max_rel_pct_cf_umax"], "static": cou_static[0]["max_rel_pct_cf_umax"]},
            "strict", cou_static[0]["max_rel_pct_cf_umax"] > cou[0]["max_rel_pct_cf_umax"])
    verdict = "PASS" if all(v["ok"] for v in checks.values()) else "FAIL"
    return {"checks": checks, "verdict": verdict,
            "failed": [k for k, v in checks.items() if not v["ok"]]}


def stage_quick(out: Path) -> None:
    t0 = time.time()
    r = couette_cfg(H=16, n_steps=6000)
    _write(out, r["tag"], r)
    _logf(f"{r['tag']} max={r['max_rel_pct_cf_umax']:.3e}%", t0)
    r = stokes2_cfg(**{**REF, "delta_bl": 6.0, "n_periods": 3.0, "sample_stride": 8},
                    tag="stokes2_quick")
    _write(out, "stokes2_quick", r)
    _logf(f"stokes2 quick max_resolved={r['max_err_pct_U0_resolved']:.4f}%", t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="finalize",
                    choices=("chunk1", "chunk2", "finalize", "quick"))
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    {"chunk1": stage_chunk1, "chunk2": stage_chunk2,
     "finalize": stage_finalize, "quick": stage_quick}[args.stage](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())