"""RELX1-PHASETAU-R2 dose-matrix analyzer: programmatic triple extraction.

Reads runs/<arm>/result.json for each arm, extracts the pre-registered triple
(peak depth / peak time absolute ms / postcontact healthy window ms), the
peak error vs experiment 1.857434D, and deltas vs ctrl.  Emits a markdown
table + JSON summary.  No hand-copied numbers.
"""
import json
import sys
from pathlib import Path

EXP_DEPTH = 1.857434  # Santini case 3 experimental peak depth (D)
ARMS = ["pt_ctrl", "pt_p05", "pt_p10", "pt_p20", "pt_m05"]
DOSES = {"pt_ctrl": 0.0, "pt_p05": 0.5, "pt_p10": 1.0, "pt_p20": 2.0, "pt_m05": -0.5}


def analyze(runs_dir: Path, arms=None):
    arms = arms or ARMS
    rows = []
    for arm in arms:
        p = runs_dir / arm / "result.json"
        if not p.exists():
            rows.append({"arm": arm, "status": "MISSING"})
            continue
        r = json.loads(p.read_text())
        h = r.get("history") or []
        if not h:
            rows.append({"arm": arm, "status": "NO_HISTORY"})
            continue
        pk = max(h, key=lambda x: x.get("depth_over_D", -1))
        contact = r.get("contact_step")
        dt_ms = 1000 * r["scales"]["dt_s"]
        # postcontact healthy window: last history row time - contact time
        last = h[-1]
        window_ms = (last["step"] - contact) * dt_ms if contact is not None else None
        peak_d = pk.get("depth_over_D")
        peak_t = pk.get("time_ms")
        rows.append({
            "arm": arm,
            "dose": DOSES.get(arm),
            "status": "ok",
            "stop_reason": r.get("stop_reason"),
            "peak_depth_D": peak_d,
            "peak_time_ms": peak_t,
            "window_ms": window_ms,
            "peak_error_pct": 100 * (peak_d - EXP_DEPTH) / EXP_DEPTH if peak_d is not None else None,
            "mach_max": max(x.get("mach_max", 0) for x in h),
            "vol_drift_end": last.get("volume_drift"),
            "contact_step": contact,
            "iface_tau_dose": r.get("iface_tau_dose"),
            "n_steps": len(h),
        })
    return rows


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float) and abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{nd}f}"


def main():
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    rows = analyze(runs_dir)
    ctrl = next((x for x in rows if x["arm"] == "pt_ctrl" and x.get("status") == "ok"), None)
    lines = ["| 臂 | dose | stop | 峰值D | 峰时刻(绝对ms) | 窗口(ms) | 峰深误差(vs 1.857434D) | Δpeak vs ctrl (D) | mach_max | 体积漂移(end) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        if x.get("status") != "ok":
            lines.append(f"| {x['arm']} | {DOSES.get(x['arm'])} | {x.get('status')} | — | — | — | — | — | — | — |")
            continue
        dpeak = (x["peak_depth_D"] - ctrl["peak_depth_D"]) if ctrl else None
        lines.append(
            f"| {x['arm']} | {x['dose']:+.1f} | {x['stop_reason']} | {fmt(x['peak_depth_D'])} | "
            f"{fmt(x['peak_time_ms'], 3)} | {fmt(x['window_ms'], 3)} | {fmt(x['peak_error_pct'], 3)}% | "
            f"{dpeak:+.5f} | {fmt(x['mach_max'])} | {fmt(x['vol_drift_end'], 3)} |"
        )
    print("\n".join(lines))
    out = runs_dir / "phasetau_summary.json"
    out.write_text(json.dumps({"experiment_peak_D": EXP_DEPTH, "rows": rows}, indent=1))
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
