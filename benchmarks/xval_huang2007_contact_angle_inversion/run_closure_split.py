#!/usr/bin/env python3
"""Chunked driver for the Huang-2007 closure audit.

``run_closure.py`` does the whole audit in one process (~14 min of SC-LBM at the
production grids).  This wrapper slices it into two independent chunks so each
one fits a short local slot (<=10 min, <=8 cores) while the cluster queue is
saturated; the chunks are then merged into the same ``closure_result.json``
schema that ``run_closure.py`` writes, so every downstream consumer
(figures, markdown report, tests) is identical.

    python .../run_closure_split.py --chunk a     # grid-A calibration + in-grid closure + convergence
    python .../run_closure_split.py --chunk b     # transfer on grid B + recalibration on grid B
    python .../run_closure_split.py --merge       # -> closure_result.json (+ verdict)

Every number is produced by the solver; nothing is transcribed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "xval_huang2007_contact_angle_inversion"))

from run_closure import run_case  # noqa: E402
from tensorlbm.contact_angle_inversion import calibrate, invert  # noqa: E402
from tensorlbm.contact_angle_inversion.closure import (  # noqa: E402
    closure_metrics,
    estimator_disagreement,
    self_interpolation_error,
    stage_verdict,
)

ART = REPO / "artifacts" / "xval_huang2007_contact_angle_inversion"
PART_A = ART / "closure_partA.json"
PART_B = ART / "closure_partB.json"
RESULT = ART / "closure_result.json"

GRID_A = dict(nx=128, ny=64, wall_rows=2, radius=15.0)
GRID_B = dict(nx=160, ny=84, wall_rows=2, radius=19.0)
STEPS, CONV_STEPS = 8000, 16000
FW_A = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
FW_B = [0.35, 0.42, 0.50, 0.58, 0.65, 0.70]
TARGETS = [70.0, 100.0, 130.0]
THREADS = 6

ROUND1_COARSE = [
    (0.40, 132.03), (0.45, 118.92), (0.50, 105.73),
    (0.55, 91.76), (0.65, 61.31), (0.70, 41.35),
]


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def cal_table(cal) -> list[dict]:
    return [
        {"fw": p.f_w, "theta_measured": p.theta_measured, "status": p.status, "drift_deg": p.drift_deg}
        for p in cal.points
    ]


def run_and_measure(label: str, fw: float, grid: dict, steps: int, extra: dict | None = None) -> dict:
    res = run_case(fw=fw, steps=steps, threads=THREADS, **grid)
    dis = estimator_disagreement(res["theta_measured"], res["theta_measured_alt"])
    row = {
        "label": label,
        "theta_measured": res["theta_measured"],
        "theta_measured_alt": res["theta_measured_alt"],
        "status": "ok" if math.isfinite(res["theta_measured"]) else "no_measurement",
        "fw": fw,
        "estimator_disagreement": dis,
        "run": res,
    }
    if extra:
        row.update(extra)
    return row


def chunk_a(t0: float) -> int:
    out: dict = {"stages": {}, "notes": []}
    out["round1_tautology"] = {
        "source": "benchmarks/xval_huang2007_contact_angle_inversion/results.json (round 1)",
        "rows": self_interpolation_error(ROUND1_COARSE, [60, 75, 90, 105, 120, 30, 150]),
        "in_range_max_err_pct": max(
            r["err_pct"] for r in self_interpolation_error(ROUND1_COARSE, [60, 75, 90, 105, 120])
        ),
        "verdict": "round-1 '8/9 within 15%' measures interpolation identity, not physics",
    }
    log(f"round-1 tautology in-range max err {out['round1_tautology']['in_range_max_err_pct']:.2e} %", t0)

    log(f"calibrating grid A {GRID_A}, {len(FW_A)} x {STEPS} steps", t0)
    cal_a = calibrate(fw_values=FW_A, steps=STEPS, threads=THREADS, **GRID_A)
    meta = cal_a.to_dict()
    meta.pop("points", None)
    meta["table"] = cal_table(cal_a)
    valid = [p for p in cal_a.points if p.status == "ok"]
    lo = min(p.theta_measured for p in valid)
    hi = max(p.theta_measured for p in valid)
    meta["measured_angle_range_deg"] = [lo, hi]
    out["calibration_gridA"] = meta
    log(f"grid A calibration {len(valid)}/{len(cal_a.points)} valid, theta {lo:.1f}..{hi:.1f} deg", t0)

    rows = []
    for tgt in TARGETS:
        fw = invert(cal_a, tgt, on_ambiguous="nan")
        if not math.isfinite(fw):
            rows.append({"label": f"in_grid_{tgt:g}", "theta_target": tgt,
                         "theta_measured": float("nan"), "status": "inversion_failed"})
            continue
        log(f"in-grid closure target {tgt:g} deg -> fw {fw:.4f}", t0)
        row = run_and_measure(f"in_grid_{tgt:g}", fw, GRID_A, STEPS,
                              {"theta_target": tgt, "target_in_cal_range": bool(lo < tgt < hi)})
        rows.append(row)
        log(f"   measured {row['theta_measured']:.2f} deg (alt {row['theta_measured_alt']:.2f})", t0)
    out["stages"]["in_grid_gridA"] = {"rows": rows, "metrics": closure_metrics(rows)}

    first = next((r for r in rows if r.get("status") == "ok"), None)
    if first:
        log(f"convergence check: {first['theta_target']:g} deg at {CONV_STEPS} steps", t0)
        long_run = run_case(fw=first["fw"], steps=CONV_STEPS, threads=THREADS, **GRID_A)
        delta = abs(long_run["theta_measured"] - first["theta_measured"])
        out["convergence"] = {
            "theta_target": first["theta_target"],
            "fw": first["fw"],
            "steps_short": STEPS,
            "steps_long": CONV_STEPS,
            "theta_short": first["theta_measured"],
            "theta_long": long_run["theta_measured"],
            "delta_deg": delta,
            "time_step_effect_pct": delta / first["theta_target"] * 100.0,
            "long_run": long_run,
            "relaxed": bool(delta < 2.0),
        }
        log(f"   theta {first['theta_measured']:.2f} -> {long_run['theta_measured']:.2f} deg", t0)

    PART_A.write_text(json.dumps(out, indent=2, default=str))
    log(f"wrote {PART_A}", t0)
    return 0


def chunk_b(t0: float) -> int:
    part_a = json.loads(PART_A.read_text())
    out: dict = {"stages": {}, "notes": []}

    transfer_rows = []
    for row_a in part_a["stages"]["in_grid_gridA"]["rows"]:
        fw = row_a.get("fw", float("nan"))
        tgt = row_a["theta_target"]
        if not math.isfinite(fw):
            transfer_rows.append({"label": f"transfer_{tgt:g}", "theta_target": tgt,
                                  "theta_measured": float("nan"), "status": "inversion_failed"})
            continue
        log(f"transfer target {tgt:g} deg, fw {fw:.4f} on grid B {GRID_B}", t0)
        row = run_and_measure(f"transfer_{tgt:g}", fw, GRID_B, STEPS, {"theta_target": tgt})
        transfer_rows.append(row)
        log(f"   measured {row['theta_measured']:.2f} deg", t0)
    out["stages"]["transfer_gridB_with_calA"] = {"rows": transfer_rows,
                                                 "metrics": closure_metrics(transfer_rows)}

    log(f"recalibrating grid B {GRID_B}, {len(FW_B)} x {STEPS} steps", t0)
    cal_b = calibrate(fw_values=FW_B, steps=STEPS, threads=THREADS, **GRID_B)
    meta = cal_b.to_dict()
    meta.pop("points", None)
    meta["table"] = cal_table(cal_b)
    out["calibration_gridB"] = meta

    recal_rows = []
    for tgt in TARGETS:
        fw = invert(cal_b, tgt, on_ambiguous="nan")
        if not math.isfinite(fw):
            recal_rows.append({"label": f"recal_{tgt:g}", "theta_target": tgt,
                               "theta_measured": float("nan"), "status": "inversion_failed"})
            continue
        log(f"recal closure target {tgt:g} deg -> fw {fw:.4f} on grid B", t0)
        row = run_and_measure(f"recal_{tgt:g}", fw, GRID_B, STEPS, {"theta_target": tgt})
        recal_rows.append(row)
        log(f"   measured {row['theta_measured']:.2f} deg", t0)
    out["stages"]["recal_gridB"] = {"rows": recal_rows, "metrics": closure_metrics(recal_rows)}

    PART_B.write_text(json.dumps(out, indent=2, default=str))
    log(f"wrote {PART_B}", t0)
    return 0


def merge(t0: float) -> int:
    part_a = json.loads(PART_A.read_text())
    part_b = json.loads(PART_B.read_text())
    out = {
        "anchor": "XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION",
        "round": 2,
        "purpose": "replace round-1 self-interpolation 'validation' with a solver closure test",
        "criterion_pct": 15.0,
        "config": {
            "grid_A": GRID_A,
            "grid_B": GRID_B,
            "steps": STEPS,
            "convergence_steps": CONV_STEPS,
            "fw_sweep_A": FW_A,
            "fw_sweep_B": FW_B,
            "holdout_targets_deg": TARGETS,
            "threads": THREADS,
            "light_mode": False,
            "execution": "two chunks (run_closure_split.py a/b) merged, each <=10 min local slot",
        },
        "round1_tautology": part_a["round1_tautology"],
        "calibration_gridA": part_a["calibration_gridA"],
        "calibration_gridB": part_b["calibration_gridB"],
        "convergence": part_a.get("convergence"),
        "stages": {
            "in_grid_gridA": part_a["stages"]["in_grid_gridA"],
            "transfer_gridB_with_calA": part_b["stages"]["transfer_gridB_with_calA"],
            "recal_gridB": part_b["stages"]["recal_gridB"],
        },
        "notes": part_a.get("notes", []) + part_b.get("notes", []),
    }
    out["verdict"] = stage_verdict({k: v["metrics"] for k, v in out["stages"].items()})
    out["honest_notes"] = [
        "Round-1 '8/9 within 15%' is an interpolation identity; re-derived here from the published "
        "calibration table so the tautology is visible rather than argued.",
        "The alternate estimator (theta_fit.contact_angle_from_circle) returns the supplement of the "
        "liquid-side angle by documented convention (contact_line/microtexture.py); it is normalised "
        "with 180-theta before comparison.",
        "A transfer failure would not be a defect of the inversion method: Huang-2007 calibration is "
        "grid-specific, so a per-grid calibration is mandatory (stage recal_gridB).",
        "Step count, hold-out targets and grids are recorded in config; nothing is re-entered by hand.",
    ]
    RESULT.write_text(json.dumps(out, indent=2, default=str))
    log(f"wrote {RESULT}", t0)
    log(f"verdict {out['verdict']['verdict']} -> {out['verdict']['stages']}", t0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", choices=["a", "b"])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    if args.chunk == "a":
        return chunk_a(t0)
    if args.chunk == "b":
        return chunk_b(t0)
    if args.merge:
        return merge(t0)
    ap.error("pick --chunk a|b or --merge")


if __name__ == "__main__":
    sys.exit(main())