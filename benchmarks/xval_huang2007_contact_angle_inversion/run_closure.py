#!/usr/bin/env python3
"""Round-2 closure ("round-trip") acceptance for the Huang-2007 contact-angle inversion.

What round 1 did
----------------
``run_xval.py`` (round 1, merged) inverted the measured calibration curve to a
wall fraction and then evaluated *the same fitted curve* forward.  Inside the
measured range that composition is the identity, so "8/9 targets within 15%"
is a property of linear interpolation, not of the LBM wall condition.  See
:func:`tensorlbm.contact_angle_inversion.closure.self_interpolation_error`,
which reproduces the round-1 pass count from the published table alone.

What this driver does instead
-----------------------------
It breaks the loop with the solver:

``calibration``  sweep ``f_w`` on grid A with a real SC-LBM sessile drop
                 (``WallStepper``), measure ``theta`` from the relaxed field.
``in_grid``      invert at hold-out targets that are *not* calibration knots,
                 re-run the **simulation** with the inverted ``f_w``, measure
                 ``theta`` again from the new field, compare with the target.
``transfer``     apply grid A's calibration to a **different grid** (finer,
                 larger drop) without recalibrating -- does the curve transfer?
``recal``        recalibrate on that grid and repeat the closure test, to
                 separate "inversion method is wrong" from "calibration is
                 grid-specific".
``convergence``  repeat one in-grid case with 2x the steps to show the measured
                 angle is a relaxed equilibrium, not a transient.

Every reported angle comes from two independent estimators
(``contact_angle_from_mask``: linear LS on cap half-widths;
``theta_fit.angle_from_density``: contour cloud + nonlinear circle fit), so a
result that depends on one estimator's bias is visible.

Usage::

    python benchmarks/xval_huang2007_contact_angle_inversion/run_closure.py --light   # smoke
    python benchmarks/xval_huang2007_contact_angle_inversion/run_closure.py           # full
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

from tensorlbm.contact_angle_inversion import calibrate, invert  # noqa: E402
from tensorlbm.contact_angle_inversion.closure import (  # noqa: E402
    closure_metrics,
    estimator_disagreement,
    self_interpolation_error,
    stage_verdict,
)
from tensorlbm.contact_line.engine import (  # noqa: E402
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    Channel,
    WallStepper,
    sessile_drop_state,
)
from tensorlbm.contact_line.theta_fit import angle_from_density  # noqa: E402
from tensorlbm.sc_surface_tension import contact_angle_from_mask, psi_exp  # noqa: E402

OUT_DIR = REPO / "artifacts" / "xval_huang2007_contact_angle_inversion"


# ---------------------------------------------------------------------------
# One relaxation run + two independent measurements
# ---------------------------------------------------------------------------


def run_case(
    *,
    nx: int,
    ny: int,
    wall_rows: int,
    radius: float,
    fw: float,
    steps: int,
    sample_interval: int | None = None,
    threads: int = 4,
) -> dict:
    """Relax a sessile drop at wall fraction ``fw`` and measure its angle twice."""
    torch.set_num_threads(threads)
    channel = Channel(nx=nx, ny=ny, wall_rows=wall_rows, device="cpu")
    solid = channel.solid_mask()
    y_wall = channel.y_wall()
    mid = 0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT)

    f, _ = sessile_drop_state(channel, radius=radius, height=0.0)
    stepper = WallStepper(
        channel,
        G=5.0,
        tau=1.0,
        rho_l=RHO_L_DEFAULT,
        rho_v=RHO_V_DEFAULT,
        psi_fn=psi_exp,
        wall_fraction=fw,
    )

    if sample_interval is None:
        sample_interval = max(25, steps // 8)

    t0 = time.time()
    history: list[tuple[int, float]] = []
    for step in range(1, steps + 1):
        f = stepper.step(f)
        if step % sample_interval == 0:
            rho = stepper.density(f)
            m = contact_angle_from_mask(rho, solid, mid=mid)
            if m.get("status") == "ok":
                history.append((step, float(m["theta_deg"])))
    runtime = time.time() - t0

    rho = stepper.density(f)
    prim = contact_angle_from_mask(rho, solid, mid=mid)
    theta_prim = float(prim.get("theta_deg", float("nan"))) if prim.get("status") == "ok" else float("nan")

    theta_alt = float("nan")
    alt_status = "not_run"
    try:
        est = angle_from_density(rho, mid, y_wall, solid=solid)
        raw = float(getattr(est, "theta_deg", float("nan")))
        # Documented convention (theta_fit.contact_angle_from_circle, see
        # contact_line/microtexture.py): that estimator returns the *supplement*
        # of the angle measured through the liquid, so normalise before comparing.
        theta_alt = 180.0 - raw if math.isfinite(raw) else float("nan")
        alt_status = "ok" if math.isfinite(theta_alt) else "no_measurement"
    except Exception as exc:  # estimator is optional diagnostics, never fatal
        alt_status = f"error:{type(exc).__name__}"

    drift = float("nan")
    if len(history) >= 4:
        drift = abs(history[-1][1] - history[-4][1])

    return {
        "fw": float(fw),
        "steps": int(steps),
        "grid": [nx, ny],
        "radius": float(radius),
        "wall_rows": int(wall_rows),
        "theta_measured": theta_prim,
        "theta_measured_alt": theta_alt,
        "primary_status": prim.get("status", "unknown"),
        "alt_status": alt_status,
        "n_contour_rows": int(prim.get("n_rows", 0) or 0),
        "drift_deg_last_3samples": drift,
        "history": [[int(s), round(float(t), 4)] for s, t in history],
        "runtime_s": round(runtime, 2),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true", help="tiny grids/steps for a plumbing smoke test")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    if args.light:
        grid_a = dict(nx=64, ny=40, wall_rows=2, radius=8.0)
        grid_b = dict(nx=80, ny=48, wall_rows=2, radius=10.0)
        steps, conv_steps = 1200, 2400
        fw_a = [0.30, 0.40, 0.50, 0.60, 0.70]
        fw_b = [0.40, 0.50, 0.60]
        targets = [90.0, 110.0]
        tag = "closure_light"
    else:
        grid_a = dict(nx=128, ny=64, wall_rows=2, radius=15.0)
        grid_b = dict(nx=160, ny=84, wall_rows=2, radius=19.0)
        steps, conv_steps = 8000, 16000
        fw_a = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        fw_b = [0.35, 0.42, 0.50, 0.58, 0.65, 0.70]
        targets = [70.0, 100.0, 130.0]
        tag = "closure"

    t_start = time.time()
    out: dict = {
        "anchor": "XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION",
        "round": 2,
        "purpose": "replace round-1 self-interpolation 'validation' with a solver closure test",
        "criterion_pct": 15.0,
        "config": {
            "grid_A": grid_a,
            "grid_B": grid_b,
            "steps": steps,
            "convergence_steps": conv_steps,
            "fw_sweep_A": fw_a,
            "fw_sweep_B": fw_b,
            "holdout_targets_deg": targets,
            "light_mode": bool(args.light),
            "rho_l": RHO_L_DEFAULT,
            "rho_v": RHO_V_DEFAULT,
            "G": 5.0,
            "tau": 1.0,
        },
        "stages": {},
        "notes": [],
    }

    def log(msg: str) -> None:
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    # -- Stage 0: show that round-1 arithmetic is a tautology -----------------
    round1_coarse = [
        (0.40, 132.03), (0.45, 118.92), (0.50, 105.73),
        (0.55, 91.76), (0.65, 61.31), (0.70, 41.35),
    ]
    taut = self_interpolation_error(round1_coarse, [60, 75, 90, 105, 120, 30, 150])
    out["round1_tautology"] = {
        "source": "benchmarks/xval_huang2007_contact_angle_inversion/results.json (round 1)",
        "rows": taut,
        "in_range_max_err_pct": max(r["err_pct"] for r in taut if not r["clamped"]),
        "verdict": "round-1 '8/9 within 15%' measures interpolation identity, not physics",
    }
    log(f"round-1 tautology: in-range max err = {out['round1_tautology']['in_range_max_err_pct']:.3e} %")

    # -- Stage 1: calibration on grid A --------------------------------------
    log(f"calibrating grid A {grid_a} with {len(fw_a)} wall fractions x {steps} steps")
    cal_a = calibrate(fw_values=fw_a, steps=steps, threads=args.threads, **grid_a)
    out["calibration_gridA"] = cal_a.to_dict()
    out["calibration_gridA"].pop("points", None)
    valid_a = [p for p in cal_a.points if p.status == "ok"]
    log(f"grid A calibration: {len(valid_a)}/{len(cal_a.points)} valid, "
        f"theta range {min(p.theta_measured for p in valid_a):.1f}..{max(p.theta_measured for p in valid_a):.1f} deg")

    def cal_table(cal) -> list[dict]:
        return [
            {"fw": p.f_w, "theta_measured": p.theta_measured, "status": p.status, "drift_deg": p.drift_deg}
            for p in cal.points
        ]

    out["calibration_gridA"]["table"] = cal_table(cal_a)
    theta_hi_a = max(p.theta_measured for p in valid_a)
    theta_lo_a = min(p.theta_measured for p in valid_a)
    out["calibration_gridA"]["measured_angle_range_deg"] = [theta_lo_a, theta_hi_a]

    def in_range(tgt: float, lo: float, hi: float) -> bool:
        return lo < tgt < hi

    for tgt in targets:
        if not in_range(tgt, theta_lo_a, theta_hi_a):
            out["notes"].append(
                f"target {tgt:g} deg is OUTSIDE the grid-A measured range "
                f"[{theta_lo_a:.1f}, {theta_hi_a:.1f}] -> its inversion is a clamped end knot, "
                "so the row is reported as extrapolated and cannot count as a closure pass."
            )

    # -- Stage 2: in-grid closure on grid A ---------------------------------
    rows, detail = [], []
    for tgt in targets:
        fw = invert(cal_a, tgt, on_ambiguous="nan")
        if not math.isfinite(fw):
            rows.append({"label": f"in_grid_{tgt:g}", "theta_target": tgt,
                         "theta_measured": float("nan"), "status": "inversion_failed"})
            detail.append({"theta_target": tgt, "fw": fw, "status": "inversion_failed"})
            continue
        log(f"in-grid closure: target {tgt:g} deg -> f_w = {fw:.4f}, running {steps} steps")
        res = run_case(fw=fw, steps=steps, threads=args.threads, **grid_a)
        dis = estimator_disagreement(res["theta_measured"], res["theta_measured_alt"])
        row = {
            "label": f"in_grid_{tgt:g}",
            "target_in_cal_range": bool(in_range(tgt, theta_lo_a, theta_hi_a)),
            "theta_target": tgt,
            "theta_measured": res["theta_measured"],
            "theta_measured_alt": res["theta_measured_alt"],
            "status": "ok" if math.isfinite(res["theta_measured"]) else "no_measurement",
            "fw": fw,
            "estimator_disagreement": dis,
            "run": res,
        }
        rows.append(row)
        detail.append(row)
        log(f"    measured {res['theta_measured']:.2f} deg (alt {res['theta_measured_alt']:.2f} deg, "
            f"delta {dis['delta_deg']:.2f}), drift {res['drift_deg_last_3samples']:.3f} deg")
    out["stages"]["in_grid_gridA"] = {"rows": rows, "metrics": closure_metrics(rows)}

    # -- Stage 3: transfer grid-A calibration to grid B ---------------------
    rows_t = []
    for tgt, row_a in zip(targets, detail):
        fw = row_a.get("fw", float("nan"))
        if not math.isfinite(fw):
            rows_t.append({"label": f"transfer_{tgt:g}", "theta_target": tgt,
                           "theta_measured": float("nan"), "status": "inversion_failed"})
            continue
        log(f"transfer: target {tgt:g} deg (f_w {fw:.4f} from grid A) on grid B {grid_b}")
        res = run_case(fw=fw, steps=steps, threads=args.threads, **grid_b)
        dis = estimator_disagreement(res["theta_measured"], res["theta_measured_alt"])
        rows_t.append({
            "label": f"transfer_{tgt:g}",
            "theta_target": tgt,
            "theta_measured": res["theta_measured"],
            "theta_measured_alt": res["theta_measured_alt"],
            "status": "ok" if math.isfinite(res["theta_measured"]) else "no_measurement",
            "fw": fw,
            "estimator_disagreement": dis,
            "run": res,
        })
        log(f"    measured {res['theta_measured']:.2f} deg (target {tgt:g})")
    out["stages"]["transfer_gridB_with_calA"] = {"rows": rows_t, "metrics": closure_metrics(rows_t)}

    # -- Stage 4: recalibrate on grid B, close the loop there ---------------
    log(f"recalibrating grid B {grid_b} with {len(fw_b)} wall fractions x {steps} steps")
    cal_b = calibrate(fw_values=fw_b, steps=steps, threads=args.threads, **grid_b)
    out["calibration_gridB"] = cal_b.to_dict()
    out["calibration_gridB"].pop("points", None)
    out["calibration_gridB"]["table"] = cal_table(cal_b)

    rows_r = []
    for tgt in targets:
        fw = invert(cal_b, tgt, on_ambiguous="nan")
        if not math.isfinite(fw):
            rows_r.append({"label": f"recal_{tgt:g}", "theta_target": tgt,
                           "theta_measured": float("nan"), "status": "inversion_failed"})
            continue
        log(f"recal closure: target {tgt:g} deg -> f_w = {fw:.4f} on grid B")
        res = run_case(fw=fw, steps=steps, threads=args.threads, **grid_b)
        dis = estimator_disagreement(res["theta_measured"], res["theta_measured_alt"])
        rows_r.append({
            "label": f"recal_{tgt:g}",
            "theta_target": tgt,
            "theta_measured": res["theta_measured"],
            "theta_measured_alt": res["theta_measured_alt"],
            "status": "ok" if math.isfinite(res["theta_measured"]) else "no_measurement",
            "fw": fw,
            "estimator_disagreement": dis,
            "run": res,
        })
        log(f"    measured {res['theta_measured']:.2f} deg (target {tgt:g})")
    out["stages"]["recal_gridB"] = {"rows": rows_r, "metrics": closure_metrics(rows_r)}

    # -- Stage 5: is 8000 steps a relaxed state? ---------------------------
    conv_row = detail[0] if detail else None
    if conv_row and math.isfinite(conv_row.get("fw", float("nan"))):
        log(f"convergence check: re-running target {conv_row['theta_target']:g} deg with {conv_steps} steps")
        res_long = run_case(fw=conv_row["fw"], steps=conv_steps, threads=args.threads, **grid_a)
        short = conv_row["theta_measured"]
        long_ = res_long["theta_measured"]
        delta = abs(long_ - short)
        out["convergence"] = {
            "theta_target": conv_row["theta_target"],
            "fw": conv_row["fw"],
            "steps_short": steps,
            "steps_long": conv_steps,
            "theta_short": short,
            "theta_long": long_,
            "delta_deg": delta,
            "time_step_effect_pct": delta / conv_row["theta_target"] * 100.0,
            "long_run": res_long,
            "relaxed": bool(delta < 2.0),
        }
        log(f"    theta {short:.2f} -> {long_:.2f} deg (delta {delta:.2f})")

    # -- Verdict ------------------------------------------------------------
    out["verdict"] = stage_verdict({k: v["metrics"] for k, v in out["stages"].items()})
    out["honest_notes"] = [
        "Round-1 '8/9 within 15%' is an interpolation identity; it is re-derived here from the "
        "published calibration table so the tautology is visible rather than argued.",
        "A transfer failure is not a defect of the inversion method: Huang-2007 calibration is "
        "grid-specific, so a per-grid calibration is mandatory (stage recal_gridB).",
        "Step count, hold-out targets and grids are recorded in config; nothing is re-entered by hand.",
    ]
    out["runtime_total_s"] = round(time.time() - t_start, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{tag}_result.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    log(f"wrote {path}")
    log(f"verdict {out['verdict']['verdict']} -> {out['verdict']['stages']}")
    return 0 if out["verdict"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())