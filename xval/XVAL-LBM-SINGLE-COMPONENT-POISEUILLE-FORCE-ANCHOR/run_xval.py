#!/usr/bin/env python3
"""XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR driver.

Cross-validation of TensorLBM's body-force convention against
ProjectPhysX/FluidX3D's single-component force-driven Poiseuille anchor
(``#define VOLUME_FORCE`` + ``#define SRT``, kernel.cpp:1589-1648).

2x2 factorial (path x forcing convention) + predeclared sweeps; see
``predeclared.json`` for the decision, hypotheses, thresholds and stopping
rule (written before any result was inspected).

Usage
-----
    python run_xval.py --shard 0 --nshards 4      # one shard (cluster)
    python run_xval.py --merge                    # merge shards -> results.json
    python run_xval.py --figures                  # results.json -> figures/*.png
    python run_xval.py --quick                    # single base case smoke test
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import poiseuille_kernels as K  # noqa: E402

NZ = 1  # periodic spanwise extent (degenerate: exactly one-dimensional channel)
NX = 1  # periodic streamwise extent (1 cell: the profile cannot develop x structure)
NX_OSS = 4  # D2Q9 path: 4 periodic x cells (cheaper per cell than the D3Q19 path)

MAX_STEPS = 200000
TOL = 1e-9
CHECK = 500


def force_for(ny: int, tau: float, u_target: float) -> float:
    """Body force that would give ``u_target`` in the continuum channel."""
    nu = (tau - 0.5) / 3.0
    return 8.0 * nu * u_target / (ny - 2) ** 2


# arm -> (path, forcing, collision)
ARMS = {
    "AO_OS": ("oss", "oss", "bgk"),   # OSS anchor: OSS path + OSS forcing
    "AO_RS": ("oss", "repo", "bgk"),  # crossed cell: OSS path + repo forcing
    "AR_RS": ("repo", "repo", "bgk"),  # repo baseline (its own convention)
    "AR_OS": ("repo", "oss", "bgk"),  # ported cell: repo path + OSS forcing
    "AR_RS_TRT": ("repo", "repo", "trt"),
    "AR_OS_TRT": ("repo", "oss", "trt"),
}


def build_cases() -> list[dict]:
    """Predeclared case matrix (identical for every shard)."""
    cases: list[dict] = []

    def add(arm, ny, tau, u_target, tag):
        path, forcing, collision = ARMS[arm]
        cases.append(
            {
                "case_id": f"{arm}|ny{ny}|tau{tau}|ut{u_target}|{tag}",
                "arm": arm,
                "path": path,
                "forcing": forcing,
                "collision": collision,
                "ny": ny,
                "tau": tau,
                "u_target": u_target,
                "force": force_for(ny, tau, u_target),
                "tag": tag,
            }
        )

    base = dict(ny=32, tau=0.8, ut=0.02)
    # (a) factorial base case: every arm at the base configuration
    for arm in ("AO_OS", "AO_RS", "AR_RS", "AR_OS", "AR_RS_TRT", "AR_OS_TRT"):
        add(arm, base["ny"], base["tau"], base["ut"], "factorial_base")
    # (b) grid convergence at fixed tau/u_target
    for ny in (16, 64):
        for arm in ("AO_OS", "AR_RS", "AR_OS"):
            add(arm, ny, base["tau"], base["ut"], "grid_sweep")
    # (c) relaxation-rate sensitivity (omega = 1.818 ... 0.667)
    for tau in (0.55, 0.60, 1.00, 1.50):
        for arm in ("AO_OS", "AR_RS", "AR_OS"):
            add(arm, base["ny"], tau, base["ut"], "tau_sweep")
    # (d) force / Mach sensitivity -> exposes the order of the forcing-convention effect
    for ut in (0.01, 0.05):
        for arm in ("AO_OS", "AO_RS", "AR_RS", "AR_OS"):
            add(arm, base["ny"], base["tau"], ut, "force_sweep")
    return cases


def run_case(case: dict) -> dict:
    t0 = time.time()
    if case["path"] == "oss":
        rec = K.simulate_oss(
            case["ny"], case["tau"], case["force"], nx=NX_OSS,
            steps=MAX_STEPS, tol=TOL, check_every=CHECK, forcing=case["forcing"],
        )
    else:
        rec = K.simulate_repo(
            case["ny"], case["tau"], case["force"], nx=NX, nz=NZ,
            collision=case["collision"], forcing=case["forcing"],
            steps=MAX_STEPS, tol=TOL, check_every=CHECK,
        )
    met = K.evaluate(rec)
    out = dict(case)
    out.update({"metrics": met, "wall_clock_s": round(time.time() - t0, 2)})
    out["profile"] = {"y": rec["y"], "u": rec["u_stored"], "u_phys": rec["u_phys"],
                      "rho": rec["rho_profile"]}
    return out


def analyse(results: list[dict]) -> dict:
    """Predeclared analysis: factorial effects, convergence order, Ma scaling."""
    by_id = {r["case_id"]: r for r in results}

    def err(case_id):
        return by_id[case_id]["metrics"]["rel_err_u_max_pct"]

    def diff_pct(a, b):
        ua = np.asarray(by_id[a]["profile"]["u_phys"])
        ub = np.asarray(by_id[b]["profile"]["u_phys"])
        umax = np.max(np.abs(np.asarray(by_id[a]["profile"]["u_phys"])))
        return 100.0 * float(np.max(np.abs(ua - ub))) / umax

    def cid(arm, ny=32, tau=0.8, ut=0.02, tag=None):
        hits = [
            r["case_id"]
            for r in results
            if r["arm"] == arm and r["ny"] == ny and abs(r["tau"] - tau) < 1e-12
            and abs(r["u_target"] - ut) < 1e-15
        ]
        return hits[0] if hits else None

    out: dict = {"arms": {}, "factorial": {}, "convergence": {}, "ma_scaling": {},
                 "tau_sensitivity": {}, "pass_fail": {}}

    for r in results:
        out["arms"].setdefault(r["arm"], []).append(
            {"case_id": r["case_id"], "ny": r["ny"], "tau": r["tau"],
             "u_target": r["u_target"], **r["metrics"]}
        )

    # --- factorial base case -------------------------------------------------
    ids = {a: cid(a) for a in ARMS}
    for a, i in ids.items():
        if i:
            out["factorial"][a] = {
                "rel_err_u_max_pct": err(i),
                "profile_rel_l2_pct": by_id[i]["metrics"]["profile_rel_l2_pct"],
                "umax_over_umean": by_id[i]["metrics"]["umax_over_umean"],
                "rho_deviation_max": by_id[i]["metrics"]["rho_deviation_max"],
                "steps": by_id[i]["metrics"]["steps"],
            }
    if ids["AO_OS"] and ids["AO_RS"] and ids["AR_OS"] and ids["AR_RS"]:
        out["factorial"]["forcing_effect_oss_path_pct"] = diff_pct(ids["AO_OS"], ids["AO_RS"])
        out["factorial"]["forcing_effect_repo_path_pct"] = diff_pct(ids["AR_OS"], ids["AR_RS"])
        out["factorial"]["path_effect_oss_forcing_pct"] = diff_pct(ids["AO_OS"], ids["AR_OS"])
        out["factorial"]["path_effect_repo_forcing_pct"] = diff_pct(ids["AR_RS"], ids["AO_RS"])
        e = {a: err(ids[a]) for a in ("AO_OS", "AO_RS", "AR_RS", "AR_OS")}
        out["factorial"]["interaction_pct"] = (
            (e["AO_OS"] - e["AO_RS"]) - (e["AR_OS"] - e["AR_RS"])
        )

    # --- grid convergence ----------------------------------------------------
    for arm in ("AO_OS", "AR_RS", "AR_OS"):
        series = sorted(
            [
                (r["ny"], r["metrics"]["rel_err_u_max_pct"])
                for r in results
                if r["arm"] == arm and r["tag"] in ("grid_sweep", "factorial_base")
            ]
        )
        if len(series) >= 3:
            (n1, e1), (n2, e2), (n3, e3) = series[-3], series[-2], series[-1]
            out["convergence"][arm] = {
                "series": series,
                "order_16_to_32": math.log(e1 / e2) / math.log(n2 / n1) if e2 > 0 else None,
                "order_32_to_64": math.log(e2 / e3) / math.log(n3 / n2) if e3 > 0 else None,
                "err_ratio_32_to_64": e2 / e3 if e3 > 0 else None,
            }

    # --- tau sensitivity -----------------------------------------------------
    for arm in ("AO_OS", "AR_RS", "AR_OS"):
        pts = sorted(
            [(r["tau"], r["metrics"]["rel_err_u_max_pct"]) for r in results if r["arm"] == arm]
        )
        out["tau_sensitivity"][arm] = pts

    # --- Ma / force scaling of the forcing-convention difference -------------
    ma = {}
    for ut, arm_a, arm_b in (
        (0.01, "AO_OS", "AO_RS"), (0.05, "AO_OS", "AO_RS"),
    ):
        a, b = cid(arm_a, ut=ut), cid(arm_b, ut=ut)
        if a and b:
            ma[f"oss_path_ut{ut}"] = diff_pct(a, b)
    for ut in (0.01, 0.02, 0.05):
        a, b = cid("AR_OS", ut=ut), cid("AR_RS", ut=ut)
        if a and b:
            ma[f"repo_path_ut{ut}"] = diff_pct(a, b)
    if "repo_path_ut0.05" in ma and "repo_path_ut0.01" in ma and ma["repo_path_ut0.01"] > 0:
        ma["repo_path_ratio_005_over_001"] = ma["repo_path_ut0.05"] / ma["repo_path_ut0.01"]
    if "oss_path_ut0.05" in ma and "oss_path_ut0.01" in ma and ma["oss_path_ut0.01"] > 0:
        ma["oss_path_ratio_005_over_001"] = ma["oss_path_ut0.05"] / ma["oss_path_ut0.01"]
    out["ma_scaling"] = ma

    # --- predeclared pass / fail --------------------------------------------
    pf = out["pass_fail"]
    base = out["factorial"]
    if "AO_OS" in base:
        pf["H1_base_lt_3pct"] = base["AO_OS"]["rel_err_u_max_pct"] < 3.0
        pf["H1_finest_lt_1pct"] = (
            out["convergence"].get("AO_OS", {}).get("series", [[0, 1e9]])[-1][1] < 1.0
        )
        pf["H1_convergence_ratio_in_band"] = (
            2.0 <= out["convergence"].get("AO_OS", {}).get("err_ratio_32_to_64", 0.0) <= 8.0
        )
    if "AR_RS" in base:
        pf["H2_repo_lt_15pct"] = base["AR_RS"]["rel_err_u_max_pct"] < 15.0
    tau_errs = [e for _, e in out["tau_sensitivity"].get("AR_RS", [])]
    if tau_errs:
        pf["H2_repo_all_tau_lt_15pct"] = max(tau_errs) < 15.0
    diffs = [v for k, v in base.items() if k.startswith("forcing_effect")]
    if diffs:
        pf["forcing_effect_lt_1pct_all_paths"] = max(diffs) < 1.0
    ratio = ma.get("repo_path_ratio_005_over_001")
    if ratio is not None:
        pf["H3_ma2_ratio_in_band"] = 10.0 <= ratio <= 60.0
    if "AO_OS" in out["arms"]:
        bias = [r["stored_velocity_bias_pct"] for r in out["arms"]["AO_OS"]]
        pf["H4_stored_velocity_bias_lt_0.1pct"] = max(bias) < 0.1
    cores = [out["factorial"].get(a, {}).get("rel_err_u_max_pct") for a in ARMS
             if a in out["factorial"]]
    if cores:
        pf["all_arms_within_owner_15pct"] = max(cores) < 15.0
    out["decision"] = decide(pf)
    return out


def decide(pf: dict) -> dict:
    """Predeclared decision rule (see predeclared.json -> decision_rule)."""
    keep = (
        pf.get("H2_repo_all_tau_lt_15pct", False)
        and pf.get("forcing_effect_lt_1pct_all_paths", False)
        and pf.get("H3_ma2_ratio_in_band", False)
    )
    if keep:
        verdict = "NO_PORT_REQUIRED"
        reason = ("repo full-impulse convention stays inside the owner 15% band at every tau and "
                  "the forcing-convention difference is < 1% of u_max and scales like Ma^2 "
                  "(second order) -> document the OSS difference, keep the repo operator.")
    else:
        verdict = "PORT_OSS_CONVENTION"
        reason = ("at least one predeclared keep-condition failed (see pass_fail); the OSS "
                  "half-force convention is then recommended and the measured gain is in "
                  "factorial/tau_sensitivity.")
    return {"verdict": verdict, "reason": reason, "pass_fail": pf}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.merge:
        rows: list[dict] = []
        for path in sorted(glob.glob(str(HERE / "results_shard_*.json"))):
            rows.extend(json.loads(Path(path).read_text())["cases"])
        uniq = {}
        for r in rows:
            uniq[r["case_id"]] = r
        rows = [uniq[k] for k in sorted(uniq)]
        payload = {
            "schema": "xval-lbm-single-component-poiseuille-force-anchor-v1",
            "mission": 18372,
            "oss": "ProjectPhysX/FluidX3D master: kernel.cpp:1124-1135,1589-1648,1640-1648,"
                   "1036-1059,1360-1373; lbm.cpp:207-210 (update_force_field); defines.hpp:19,80",
            "reference": "closed-form plane Poiseuille at lattice nodes (analytic_profile)",
            "cases": rows,
        }
        payload["analysis"] = analyse(rows)
        (HERE / "results.json").write_text(json.dumps(payload, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(json.dumps(payload["analysis"], indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:4000])
        return 0

    if args.figures:
        import make_figures

        make_figures.main()
        return 0

    cases = build_cases()
    if args.quick:
        cases = [c for c in cases if "factorial_base" in c["tag"]][:2]
    else:
        cases = [c for i, c in enumerate(cases) if i % args.nshards == args.shard]

    out = []
    for case in cases:
        print(f"[run] {case['case_id']}", flush=True)
        rec = run_case(case)
        print(f"      err_u_max={rec['metrics']['rel_err_u_max_pct']:.4f}%  "
              f"steps={rec['metrics']['steps']}  {rec['wall_clock_s']}s", flush=True)
        out.append(rec)
        (HERE / f"results_shard_{args.shard}.json").write_text(
            json.dumps({"shard": args.shard, "nshards": args.nshards, "cases": out}, indent=1)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())