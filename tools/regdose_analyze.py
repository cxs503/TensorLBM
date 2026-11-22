#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-COL2D-REGDOSE: programmatic extraction of the measured triple.

No number in the card is hand-copied: every value is read from
``runs/<arm>/result.json`` (produced by the SDAA lane) or from the cited
1x reference file ``artifacts/COL2D-REGDOSE/col2_summary_cited.json`` (which is
a byte copy of the DE-X-COL2-REG card summary, commit da0ffa73b).

Outputs
-------
artifacts/COL2D-REGDOSE/summary.json   machine-readable triple + verdicts
artifacts/COL2D-REGDOSE/triple.md      markdown table (injected into the card)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Campaign anchors (DE_A1_FACTS / DE-X-COL2-REG card).
EXP_PEAK_D = 1.857434
EXP_PEAK_MS_PC = 10.39446
TAU_1X = 0.5003877  # 1x dose = COL2 rr_base relaxation time

# Dose arms: (arm name, dose multiplier).  The 1x arm is *cited*, never re-run.
DOSE_ARMS = [
    ("reg_d050", 0.5),
    ("reg_d200", 2.0),
    ("reg_d400", 4.0),
]


def load_json(path: Path):
    return json.loads(path.read_text())


def arm_metrics(res: dict) -> dict:
    hist = res.get("history") or []
    if not hist:
        return {}
    contact = res.get("contact_step")
    dt_ms = 1000.0 * res["scales"]["dt_s"]
    # Pre-contact deaths have no depth_over_D field at all; treat the cavity as
    # zero and report the peak time only when the tracker produced one.
    with_depth = [r for r in hist if r.get("depth_over_D") is not None]
    if with_depth:
        peak = max(with_depth, key=lambda r: float(r.get("depth_over_D")))
        peak_d = float(peak.get("depth_over_D"))
        peak_abs = float(peak.get("time_ms"))
        peak_step = peak.get("step")
    else:
        peak = hist[-1]
        peak_d = 0.0
        peak_abs = float(peak.get("time_ms"))
        peak_step = peak.get("step")
    peak_pc = peak_abs - contact * dt_ms if contact is not None else None
    return {
        "stop_reason": res.get("stop_reason"),
        "rr_tau": res.get("rr_tau"),
        "contact_step": contact,
        "steps": len(hist),
        "peak_depth_D": peak_d,
        "peak_time_ms_abs": peak_abs,
        "peak_time_ms_pc": peak_pc,
        "peak_step": peak_step,
        "peak_err_vs_exp_pct": 100.0 * (peak_d - EXP_PEAK_D) / EXP_PEAK_D,
        "mach_max": max(float(r.get("mach_max", 0.0)) for r in hist),
        "volume_drift_final": float(hist[-1].get("volume_drift")),
        "window_complete": res.get("stop_reason") == "window_complete",
        "postcontact_healthy_ms": (float(hist[-1].get("time_ms")) - contact * dt_ms)
        if contact is not None else 0.0,
    }


def cited_1x(cited: dict) -> dict:
    for entry in cited["arms"]:
        if entry["arm"] == "rr_base":
            return {
                "arm": "rr_base",
                "rr_tau": entry["rr_tau"],
                "peak_depth_D": entry["peak"]["depth_over_D"],
                "peak_time_ms_abs": entry["peak"]["time_ms_abs"],
                "peak_time_ms_pc": entry["peak_time_ms_pc"],
                "peak_err_vs_exp_pct": entry["peak_err_vs_exp_pct"],
                "mach_max": entry["mach_max"],
                "volume_drift_final": entry["volume_drift_final"],
                "stop_reason": entry["stop_reason"],
                "steps": entry["steps"],
                "contact_step": entry["contact_step"],
                "window_complete": entry["stop_reason"] == "window_complete",
                "source": "DE-X-COL2-REG card summary (cited, not re-run)",
            }
    raise SystemExit("cited 1x arm rr_base not found")


def cited_ctrl(cited: dict) -> dict:
    for entry in cited["arms"]:
        if entry["arm"] == "ctrl_bgk":
            return {
                "arm": "ctrl_bgk",
                "rr_tau": None,
                "peak_depth_D": entry["peak"]["depth_over_D"],
                "peak_time_ms_abs": entry["peak"]["time_ms_abs"],
                "peak_time_ms_pc": entry["peak_time_ms_pc"],
                "peak_err_vs_exp_pct": entry["peak_err_vs_exp_pct"],
                "mach_max": entry["mach_max"],
                "volume_drift_final": entry["volume_drift_final"],
                "stop_reason": entry["stop_reason"],
                "steps": entry["steps"],
                "contact_step": entry["contact_step"],
                "window_complete": entry["stop_reason"] == "window_complete",
                "source": "DE-X-COL2-REG card summary (cited, not re-run)",
            }
    raise SystemExit("cited ctrl arm not found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--cited", default="artifacts/COL2D-REGDOSE/col2_summary_cited.json")
    ap.add_argument("--out-dir", default="artifacts/COL2D-REGDOSE")
    a = ap.parse_args()

    runs = Path(a.runs_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cited = load_json(Path(a.cited))

    one_x = cited_1x(cited)
    ctrl = cited_ctrl(cited)

    # The campaign time step is taken from the measured ctrl re-run (bit-level
    # anchor check) and applied to the *cited* arms, whose history is not
    # archived in the repository.
    dt_ms = None
    ctrl_run = runs / "ctrl_pl" / "result.json"
    if ctrl_run.exists():
        dt_ms = 1000.0 * load_json(ctrl_run)["scales"]["dt_s"]
    if dt_ms is None:
        for name, _ in DOSE_ARMS:
            p = runs / name / "result.json"
            if p.exists():
                dt_ms = 1000.0 * load_json(p)["scales"]["dt_s"]
                break
    if dt_ms is not None:
        for entry in (one_x, ctrl):
            entry["dt_ms"] = dt_ms
            entry["postcontact_healthy_ms"] = (entry["steps"] - 1) * dt_ms - entry["contact_step"] * dt_ms

    arms = []
    for name, dose in DOSE_ARMS:
        p = runs / name / "result.json"
        if not p.exists():
            arms.append({"arm": name, "dose": dose, "missing": True})
            continue
        res = load_json(p)
        met = arm_metrics(res)
        nu1 = TAU_1X - 0.5
        met.update({
            "arm": name,
            "dose": dose,
            "dose_tau": TAU_1X * dose,
            "rr_tau_in_run": res.get("rr_tau"),
            # LBM lattice viscosity nu = cs^2 (tau - 1/2): the RR branch inherits
            # the BGK tau->nu mapping (same second-order stress relaxation
            # factor 1-1/tau), so a dose multiplier is also a viscosity
            # multiplier.  Negative for tau < 0.5 (unphysical).
            "nu_ratio_vs_1x": (float(res.get("rr_tau")) - 0.5) / nu1 if res.get("rr_tau") else None,
            "delta_vs_1x_D": met["peak_depth_D"] - one_x["peak_depth_D"],
            "delta_vs_1x_pct": 100.0 * (met["peak_depth_D"] - one_x["peak_depth_D"]) / one_x["peak_depth_D"],
            "delta_vs_ctrl_D": met["peak_depth_D"] - ctrl["peak_depth_D"],
            "delta_vs_ctrl_pct": 100.0 * (met["peak_depth_D"] - ctrl["peak_depth_D"]) / ctrl["peak_depth_D"],
        })
        arms.append(met)

    # ---- pre-registered verdicts -------------------------------------------
    complete = [m for m in arms if m.get("window_complete")]
    peak_1x = one_x["peak_depth_D"]
    high = [m for m in complete if m["dose"] > 1.0]
    non_monotonic = [m["arm"] for m in high if m["peak_depth_D"] < peak_1x]
    sat = None
    if len(complete) == len(arms) and len(arms) >= 2:
        by_dose = sorted(arms, key=lambda m: m["dose"])
        sat = by_dose[-1]["peak_depth_D"] - by_dose[-2]["peak_depth_D"]
    gap_4x_2x = None
    by_name = {m["arm"]: m for m in arms if not m.get("missing")}
    if "reg_d400" in by_name and "reg_d200" in by_name:
        gap_4x_2x = by_name["reg_d400"]["peak_depth_D"] - by_name["reg_d200"]["peak_depth_D"]
    boundary = [m["arm"] for m in arms if not m.get("window_complete")]

    if sat is not None and sat < 0.02:
        verdict = "SATURATED"
    elif non_monotonic:
        verdict = "NON_MONOTONIC"
    elif boundary:
        verdict = "BOUNDARY_HIT"
    else:
        verdict = "MONOTONIC_INCREASING"

    summary = {
        "mission": "DE-X-COL2D-REGDOSE",
        "exp_peak_D": EXP_PEAK_D,
        "exp_peak_ms_pc": EXP_PEAK_MS_PC,
        "tau_1x": TAU_1X,
        "cited_1x": one_x,
        "cited_ctrl": ctrl,
        "arms": arms,
        "verdict": verdict,
        "nu_ratio_note": "nu = cs^2 (tau - 0.5); the RR branch keeps the BGK tau->nu mapping, "
                         "so the dose multiplier is also a viscosity multiplier (tau<0.5 => negative nu)",
        "non_monotonic_arms": non_monotonic,
        "saturation_4x_minus_2x_D": sat,
        "peak_4x_minus_2x_D": gap_4x_2x,
        "boundary_arms": boundary,
        "prereg_prediction": {
            "monotonic_increase": True,
            "peak_2x_interval_D": [1.60, 1.75],
            "boundary_expected_at": "4x (early death / drift over limit)",
            "falsify_non_monotonic": "any higher-dose arm with peak < 1x peak",
            "falsify_saturation": "all doses stable but 4x - 2x < 0.02 D",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")

    # ---- markdown triple (injected into the card, zero hand-copy) ----------
    lines = []
    lines.append("| 臂 | 剂量 | rr_tau | stop | peak D | peak ms(abs) | peak ms(pc) | 峰深误差(vs 1.857434D) | Δpeak vs 1× (D) | mach_max | 体积漂移(end) | 接触后健康(ms) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in arms:
        if m.get("missing"):
            lines.append(f"| {m['arm']} | {m['dose']}× | — | MISSING | — | — | — | — | — | — | — |")
            continue
        pc = f"{m['peak_time_ms_pc']:.4f}" if m["peak_time_ms_pc"] is not None else "n/a(未接触)"
        lines.append(
            f"| {m['arm']} | {m['dose']}× | {m['rr_tau_in_run']} | {m['stop_reason']} | {m['peak_depth_D']:.6f} | "
            f"{m['peak_time_ms_abs']:.4f} | {pc} | {m['peak_err_vs_exp_pct']:+.3f}% | "
            f"{m['delta_vs_1x_D']:+.6f} | {m['mach_max']:.4f} | {m['volume_drift_final']:.3e} | "
            f"{m['postcontact_healthy_ms']:.3f} |")
    lines.append(
        f"| rr_base (引 1×) | 1× | {one_x['rr_tau']} | {one_x['stop_reason']} | {one_x['peak_depth_D']:.6f} | "
        f"{one_x['peak_time_ms_abs']:.4f} | {one_x['peak_time_ms_pc']:.4f} | {one_x['peak_err_vs_exp_pct']:+.3f}% | "
        f"+0.000000 | {one_x['mach_max']:.4f} | {one_x['volume_drift_final']:.3e} | "
        f"{one_x.get('postcontact_healthy_ms', float('nan')):.3f} |")
    lines.append(
        f"| ctrl_bgk (引) | 0× | — | {ctrl['stop_reason']} | {ctrl['peak_depth_D']:.6f} | "
        f"{ctrl['peak_time_ms_abs']:.4f} | {ctrl['peak_time_ms_pc']:.4f} | {ctrl['peak_err_vs_exp_pct']:+.3f}% | "
        f"{ctrl['peak_depth_D'] - one_x['peak_depth_D']:+.6f} | {ctrl['mach_max']:.4f} | {ctrl['volume_drift_final']:.3e} | "
        f"{ctrl.get('postcontact_healthy_ms', float('nan')):.3f} |")
    (out_dir / "triple.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nverdict={verdict} non_monotonic={non_monotonic} saturation={sat} boundary={boundary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
