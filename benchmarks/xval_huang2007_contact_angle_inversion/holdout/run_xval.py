#!/usr/bin/env python3
"""XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION — hold-out + transferability round.

Anchor
------
``XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION`` (je-santos/MPLBM-UT
``huang2007-contact-angle-inversion``: sweep ``theta(f_w)``, then invert).

What this round adds to the merged m18090 deliverable
-----------------------------------------------------
m18090 validated ``calibrate``/``invert`` at **one** configuration
(nx=128, ny=64, R=15, G=5, tau=1.0) on 3 hold-out angles.  Its own coarse
sweep is non-monotone in ``cos(f_w)`` near ``f_w = 0.60-0.65``, and nothing
in the repository tests what ``invert`` does there or whether a calibration
built on one grid / one viscosity can be used on another.  This script does
exactly those two things, with the gates predeclared in ``predeclared.json``:

* Arm A — target angles placed inside / beside the non-monotone band, measured
  on the calibration grid.
* Arm B — the coarse calibration applied on the production grid (resolution
  transfer).
* Arm C — the ``tau=1.0`` calibration applied at ``tau=1.2`` (a static contact
  angle must not depend on viscosity).
* H1 — when a target has more than one inversion branch, **both** branches are
  run; the spread is the inversion ambiguity of the delivered pair.

Every number in ``REPORT.md`` is rendered from ``summary.json``, which is
aggregated from the per-row ``evidence/*.json`` written by this script.

Usage
-----
    python run_xval.py calibrate --tag cal_coarse
    python run_xval.py row --tag A_th090 --arm A_inband_coarse --theta 90
    python run_xval.py matrix --workers 4
    python run_xval.py report
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.contact_angle_inversion import (  # noqa: E402
    CalibrationResult,
    calibrate,
    inversion_branches,
    is_monotone,
    young_prediction,
)
from tensorlbm.contact_line.engine import (  # noqa: E402
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    Channel,
    WallStepper,
    sessile_drop_state,
)
from tensorlbm.sc_surface_tension import contact_angle_from_mask, psi_exp  # noqa: E402

RUNS = HERE / "evidence"
GATE = 15.0
CAL_TAG = "cal_coarse"

CAL_SETTINGS = dict(nx=64, ny=32, radius=8.0, steps=5000, sample_interval=250)
FW_VALUES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

ARMS = {
    "A_inband_coarse": dict(nx=64, ny=32, radius=8.0, steps=5000, tau=1.0),
    "B_fine_transfer": dict(nx=128, ny=64, radius=15.0, steps=6000, tau=1.0),
    "C_tau_transfer": dict(nx=64, ny=32, radius=8.0, steps=5000, tau=1.2),
}

# arm -> (target angle, branch selector)
MATRIX = [
    ("A_th090", "A_inband_coarse", 90.0, "first"),
    ("A_th096", "A_inband_coarse", 96.0, "first"),
    ("A_th108", "A_inband_coarse", 108.0, "first"),
    ("A_th0955_first", "A_inband_coarse", 95.5, "first"),
    ("A_th0955_last", "A_inband_coarse", 95.5, "last"),
    ("B_th090", "B_fine_transfer", 90.0, "first"),
    ("B_th108", "B_fine_transfer", 108.0, "first"),
    ("B_th120", "B_fine_transfer", 120.0, "first"),
    ("C_th090", "C_tau_transfer", 90.0, "first"),
    ("C_th108", "C_tau_transfer", 108.0, "first"),
]


# ---------------------------------------------------------------------------
# calibration (de)serialisation
# ---------------------------------------------------------------------------


def save_cal(cal: CalibrationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"calibration": cal.to_dict(), "monotone": is_monotone(cal)}
    path.write_text(json.dumps(payload, indent=1))


def load_cal(path: Path) -> CalibrationResult:
    d = json.loads(Path(path).read_text())["calibration"]
    from tensorlbm.contact_angle_inversion.calibration import CalibrationPoint

    cal = CalibrationResult(
        nx=d["nx"],
        ny=d["ny"],
        radius=d["radius"],
        steps=d["steps"],
        wall_rows=d["wall_rows"],
        psi_l=d["psi_l"],
        psi_v=d["psi_v"],
    )
    cal.points = [CalibrationPoint(**p) for p in d["points"]]
    cal.fw_values = d["fw_values"]
    cal.cos_values = d["cos_values"]
    cal.valid_range = tuple(d["valid_range"])
    cal.young_slope = d["young_slope"]
    cal.measured_slope = d["measured_slope"]
    cal.slope_ratio = d["slope_ratio"]
    cal.neutral_fw_young = d["neutral_fw_young"]
    cal.neutral_fw_measured = d["neutral_fw_measured"]
    return cal


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_calibrate(args) -> int:
    torch.set_num_threads(args.threads)
    t0 = time.time()
    cal = calibrate(fw_values=FW_VALUES, **{**CAL_SETTINGS, "device": "cpu"})
    dt = time.time() - t0
    out = RUNS / f"{args.tag}.json"
    save_cal(cal, out)
    print(f"[calibrate] {args.tag}: {len(cal.points)} points in {dt:.1f}s -> {out}")
    for p in cal.points:
        print(
            f"  f_w={p.f_w:.2f} theta={p.theta_measured:8.2f} cos={p.cos_measured:+.4f} "
            f"status={p.status} drift={p.drift_deg:.2f}"
        )
    print(f"  monotone={is_monotone(cal)}")
    return 0


def cmd_row(args) -> int:
    torch.set_num_threads(args.threads)
    arm = ARMS[args.arm]
    cal = load_cal(RUNS / f"{args.cal}.json")
    branches = inversion_branches(cal, args.theta)
    if args.branch == "first":
        f_w = branches[0] if branches else float("nan")
    elif args.branch == "last":
        f_w = branches[-1] if branches else float("nan")
    else:
        raise SystemExit(f"unknown branch selector {args.branch}")

    rec = {
        "row_id": args.tag,
        "arm": args.arm,
        "theta_target": args.theta,
        "branch": args.branch,
        "n_branches": len(branches),
        "branches": branches,
        "f_w": f_w,
        "psi_w_over_psi_l_young": None,
        "young_theta_at_fw": None,
        "config": dict(arm),
        "calibration_tag": args.cal,
        "gate_pct": GATE,
    }
    if not math.isfinite(f_w):
        rec.update(
            status="no_branch",
            theta_measured=float("nan"),
            err_pct=float("nan"),
            error_deg=float("nan"),
            seconds=0.0,
        )
        (RUNS / f"{args.tag}.json").write_text(json.dumps(rec, indent=1))
        print(f"[row] {args.tag}: no inversion branch for {args.theta} deg")
        return 0

    rec["young_theta_at_fw"] = young_prediction(f_w, cal.psi_l, cal.psi_v)

    channel = Channel(nx=arm["nx"], ny=arm["ny"], wall_rows=2, device="cpu")
    solid = channel.solid_mask()
    y_wall = channel.y_wall()
    f, _ = sessile_drop_state(channel, radius=arm["radius"], height=0.0)
    mass0 = float(f.sum())
    stepper = WallStepper(
        channel,
        G=5.0,
        tau=arm["tau"],
        rho_l=RHO_L_DEFAULT,
        rho_v=RHO_V_DEFAULT,
        psi_fn=psi_exp,
        wall_fraction=f_w,
    )

    history = []
    t0 = time.time()
    for step in range(1, arm["steps"] + 1):
        f = stepper.step(f)
        if step % CAL_SETTINGS["sample_interval"] == 0:
            rho = stepper.density(f)
            if not torch.isfinite(rho).all():
                rec.update(
                    status="diverged",
                    theta_measured=float("nan"),
                    err_pct=float("nan"),
                    error_deg=float("nan"),
                    seconds=time.time() - t0,
                    step=step,
                )
                (RUNS / f"{args.tag}.json").write_text(json.dumps(rec, indent=1))
                print(f"[row] {args.tag}: diverged at step {step}")
                return 0
            m = contact_angle_from_mask(rho, solid, mid=0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT))
            if m.get("status") == "ok":
                history.append([step, m["theta_deg"]])
    dt = time.time() - t0

    rho = stepper.density(f)
    meas = contact_angle_from_mask(rho, solid, mid=0.5 * (RHO_L_DEFAULT + RHO_V_DEFAULT))
    drift = abs(history[-1][1] - history[-4][1]) if len(history) >= 4 else float("nan")
    theta = float(meas.get("theta_deg", float("nan")))
    ok = meas.get("status") == "ok" and math.isfinite(theta)
    err_deg = abs(theta - args.theta) if ok else float("nan")
    err_pct = err_deg / args.theta * 100.0 if ok else float("nan")

    status = "ok" if ok else f"not_measured:{meas.get('status')}"
    if ok and math.isfinite(drift) and drift > 1.0:
        status = "unconverged"

    rec.update(
        {
            "theta_measured": theta,
            "cos_measured": float(meas.get("cos_theta", float("nan"))),
            "R_fit": float(meas.get("R_fit", float("nan"))),
            "apex": float(meas.get("apex", float("nan"))),
            "n_rows": int(meas.get("n_rows", 0)),
            "status": status,
            "err_deg": err_deg,
            "err_pct": err_pct,
            "drift_deg": drift,
            "within_gate": bool(ok and err_pct < GATE),
            "mass_drift": abs(float(f.sum()) - mass0) / mass0 if mass0 > 0 else float("nan"),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "seconds": dt,
            "history": history,
        }
    )
    (RUNS / f"{args.tag}.json").write_text(json.dumps(rec, indent=1))
    print(
        f"[row] {args.tag}: target={args.theta:.1f} f_w={f_w:.4f} "
        f"theta={theta:.2f} err={err_pct:.2f}% ({status}) {dt:.1f}s"
    )
    return 0


def cmd_matrix(args) -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    if not (RUNS / f"{CAL_TAG}.json").exists():
        raise SystemExit(f"missing calibration {RUNS / (CAL_TAG + '.json')}; run `calibrate` first")
    procs, running = [], []
    queue = list(MATRIX)
    while queue or running:
        while queue and len(running) < args.workers:
            tag, arm, theta, branch = queue.pop(0)
            out = RUNS / f"{tag}.json"
            if out.exists() and not args.force:
                print(f"[matrix] skip {tag} (exists)")
                continue
            cmd = [
                sys.executable,
                str(Path(__file__)),
                "row",
                "--tag",
                tag,
                "--arm",
                arm,
                "--theta",
                str(theta),
                "--branch",
                branch,
                "--cal",
                CAL_TAG,
                "--threads",
                str(args.threads),
            ]
            running.append(subprocess.Popen(cmd, cwd=str(HERE)))
        time.sleep(0.5)
        running = [p for p in running if p.poll() is None]
    print("[matrix] done")
    return 0


def cmd_analyse(args) -> int:
    """Pure-math ambiguity analysis of the calibration curve (no LBM)."""
    cal = load_cal(RUNS / f"{args.cal}.json")
    fw, cos = cal.fw_values, cal.cos_values
    d = [cos[i + 1] - cos[i] for i in range(len(cos) - 1)]
    # dominant trend of cos(f_w): a wetting sweep normally has cos monotonically
    # increasing in f_w (theta decreasing); a segment with the opposite sign is a fold.
    trend = 1.0 if sum(d) > 0 else -1.0
    fold_segments = [
        {"segment": [fw[i], fw[i + 1]], "d_cos": d[i]} for i in range(len(d)) if d[i] * trend < 0
    ]
    # ambiguous target window: cos values reachable by two distinct f_w branches
    amb = []
    for i in range(1, len(cos) - 1):
        is_min = cos[i] < cos[i - 1] and cos[i] < cos[i + 1]
        is_max = cos[i] > cos[i - 1] and cos[i] > cos[i + 1]
        if is_min or is_max:
            nb = [cos[i - 1], cos[i + 1]]
            # two branches sandwich the fold only where BOTH neighbours stay on the
            # far side of the fold value
            outer = min(nb) if is_min else max(nb)
            lo, hi = sorted([cos[i], outer])
            amb.append(
                {
                    "node_fw": fw[i],
                    "fold": "cos-min" if is_min else "cos-max",
                    "cos_at_node": cos[i],
                    "theta_at_node_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos[i])))),
                    "ambiguous_cos_window": [lo, hi],
                    "ambiguous_theta_window_deg": [
                        math.degrees(math.acos(max(-1.0, min(1.0, hi)))),
                        math.degrees(math.acos(max(-1.0, min(1.0, lo)))),
                    ],
                }
            )
    out = {
        "calibration_tag": args.cal,
        "fw_values": fw,
        "cos_values": cos,
        "theta_values": [math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in cos],
        "monotone": is_monotone(cal),
        "fold_segments": fold_segments,
        "fold_points": amb,
        "young_theta_at_same_fw": [young_prediction(f, cal.psi_l, cal.psi_v) for f in fw],
    }
    (RUNS / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def cmd_report(args) -> int:
    cal_d = json.loads((RUNS / f"{CAL_TAG}.json").read_text())
    rows = []
    for p in sorted(RUNS.glob("*.json")):
        if p.stem == CAL_TAG:
            continue
        d = json.loads(p.read_text())
        if "theta_target" in d and "arm" in d:
            rows.append(d)
    measured = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "anchor": "XVAL-LBM-HUANG2007-CONTACT-ANGLE-INVERSION",
        "round": "holdout+transfer",
        "gate_pct": GATE,
        "calibration": cal_d["calibration"],
        "calibration_monotone": cal_d["monotone"],
        "rows": rows,
        "gate_counts": {},
        "arms": {},
    }
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        m = [r for r in arm_rows if r.get("status") == "ok"]
        summary["arms"][arm] = {
            "n_rows": len(arm_rows),
            "n_measured": len(m),
            "n_within_gate": sum(1 for r in m if r.get("within_gate")),
            "max_err_pct": max((r["err_pct"] for r in m), default=float("nan")),
            "mean_err_pct": (sum(r["err_pct"] for r in m) / len(m)) if m else float("nan"),
            "rows": [r["row_id"] for r in arm_rows],
        }
        if m:
            summary["arms"][arm]["mean_err_pct"] = sum(r["err_pct"] for r in m) / len(m)
    summary["gate_counts"] = {
        "measured_rows": len(measured),
        "within_gate": sum(1 for r in measured if r.get("within_gate")),
        "outside_gate": sum(1 for r in measured if not r.get("within_gate")),
        "not_measured": len(rows) - len(measured),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["gate_counts"], indent=1))
    for arm, a in summary["arms"].items():
        print(
            f"{arm}: {a['n_within_gate']}/{a['n_measured']} within gate; max err {a['max_err_pct']:.2f}%"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate")
    c.add_argument("--tag", default=CAL_TAG)
    c.add_argument("--threads", type=int, default=2)
    c.set_defaults(fn=cmd_calibrate)

    r = sub.add_parser("row")
    r.add_argument("--tag", required=True)
    r.add_argument("--arm", required=True, choices=list(ARMS))
    r.add_argument("--theta", type=float, required=True)
    r.add_argument("--branch", default="first", choices=["first", "last"])
    r.add_argument("--cal", default=CAL_TAG)
    r.add_argument("--threads", type=int, default=2)
    r.set_defaults(fn=cmd_row)

    m = sub.add_parser("matrix")
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--threads", type=int, default=2)
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_matrix)

    a = sub.add_parser("analyse")
    a.add_argument("--cal", default=CAL_TAG)
    a.set_defaults(fn=cmd_analyse)

    rp = sub.add_parser("report")
    rp.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
