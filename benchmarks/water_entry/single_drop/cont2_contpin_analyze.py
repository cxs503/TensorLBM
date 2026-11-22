#!/usr/bin/env python3
"""DE-X-CONT2-CONT-PIN: extract the decision triple from cluster probe outputs.

Reads runs/cont2/{contpin,ctrl}.json (free_surface_probe result JSONs) and
prints the predeclared triple (peak_depth / peak_error / postcontact_healthy_ms)
plus the preregistration verdict.  All numbers are read from the artifacts;
nothing is hand-entered.
"""
import json
import sys
from pathlib import Path

EXP_PEAK_D = 1.857434   # Santini case-3 experimental peak depth (D)
EXP_PEAK_MS = 10.394    # experimental peak time (ms after contact)
CTRL_ANCHOR_PEAK_D = 1.4352  # established ctrl anchor (CAVSWEEP/KERN-B)


def triple(result: dict) -> dict:
    hist = result.get("history") or []
    rows = [r for r in hist if r.get("depth_over_D") is not None]
    if not rows:
        return {"peak_depth_D": None, "peak_ms": None, "peak_error_pct": None,
                "healthy_ms": None, "stop": result.get("stop_reason")}
    pk = max(rows, key=lambda r: r["depth_over_D"])
    last = hist[-1]
    return {
        "peak_depth_D": round(pk["depth_over_D"], 4),
        "peak_ms": round(pk.get("time_ms") or 0.0, 3),
        "peak_error_pct": round(100.0 * (EXP_PEAK_D - pk["depth_over_D"]) / EXP_PEAK_D, 3),
        "healthy_ms": round(last.get("time_ms") or 0.0, 3),
        "stop": result.get("stop_reason"),
        "contact_step": result.get("contact_step"),
        "mach_max": last.get("mach_max"),
        "drift_end": last.get("geometric_volume_drift"),
        "impact_m_s": result.get("impact_speed_m_s"),
    }


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/cont2")
    out = {}
    for name in ("ctrl", "contpin"):
        p = base / f"{name}.json"
        if not p.exists():
            print(f"MISSING {p}")
            continue
        r = json.loads(p.read_text())
        t = triple(r)
        t["contact_pin"] = r.get("contact_pin")
        out[name] = t
        print(f"{name}: " + json.dumps(t, ensure_ascii=False))
    if "ctrl" in out and "contpin" in out:
        c, a = out["ctrl"], out["contpin"]
        if c["peak_depth_D"] is not None and a["peak_depth_D"] is not None:
            print(f"delta_peak_vs_ctrl = {a['peak_depth_D'] - c['peak_depth_D']:+.4f} D")
            print(f"delta_peak_vs_anchor = {a['peak_depth_D'] - CTRL_ANCHOR_PEAK_D:+.4f} D")
            in_band = 1.50 <= a["peak_depth_D"] <= 1.70
            falsified = abs(a["peak_depth_D"] - CTRL_ANCHOR_PEAK_D) < 0.05
            early = a["stop"] != "window_complete" and (a["peak_ms"] or 0) < EXP_PEAK_MS
            print(f"prereg band [1.50,1.70]D hit: {in_band}")
            print(f"falsify |peak-1.4352|<0.05D: {falsified}")
            print(f"early death (<{EXP_PEAK_MS}ms, non-window): {early}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
