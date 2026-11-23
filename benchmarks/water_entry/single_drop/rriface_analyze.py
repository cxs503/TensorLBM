"""DE-X-RELX3-RRIFACE: programmatic triplet extraction from runs/<arm>/result.json.

peak_depth / peak_error (vs experiment 1.857434D) / postcontact_healthy_ms,
plus the anchor-gate fields (peak pc time, window, mach_max, drift_end,
contact_step).  Every number printed is read from the JSON, never typed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP_PEAK_D = 1.857434
EXP_PEAK_MS = 10.39446  # post-contact ms (Santini case-3)


def load(tree: Path, arm: str) -> dict:
    return json.loads((tree / "runs" / arm / "result.json").read_text())


def triplet(res: dict, arm: str) -> dict:
    h = res["history"]
    pk = max(h, key=lambda r: r["depth_over_D"])
    contact = res.get("contact_step")
    dt_ms = h[1]["time_ms"] - h[0]["time_ms"] if len(h) > 1 else None
    t_abs = pk["time_ms"]
    t_pc = t_abs - contact * dt_ms if (contact is not None and dt_ms) else None
    last = h[-1]
    window_pc_ms = (last["time_ms"] - contact * dt_ms) if (contact is not None and dt_ms) else None
    return {
        "arm": arm,
        "stop": res["stop_reason"],
        "steps": len(h),
        "t_end_ms": last["time_ms"],
        "window_pc_ms": window_pc_ms,
        "peak_D": pk["depth_over_D"],
        "peak_t_abs_ms": t_abs,
        "peak_t_pc_ms": t_pc,
        "peak_err_pct": 100.0 * (pk["depth_over_D"] - EXP_PEAK_D) / EXP_PEAK_D,
        "contact_step": contact,
        "mach_max": max(r["mach_max"] for r in h),
        "drift_end": last["volume_drift"],
        "geo_drift_end": last["geometric_volume_drift"],
    }


def main() -> int:
    tree = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    arms = sys.argv[2:] or ["relx3_rriface_ctrl", "relx3_rriface"]
    rows = [triplet(load(tree, a), a) for a in arms]
    hdr = ("arm", "stop", "peak_D", "peak_t_abs_ms", "peak_t_pc_ms", "peak_err_pct",
           "contact", "mach_max", "drift_end", "t_end_ms")
    for r in rows:
        print(
            f"{r['arm']:<20} {r['stop']:<16} {r['peak_D']:.6f} {r['peak_t_abs_ms']:.6f} "
            f"{r['peak_t_pc_ms']:.6f} {r['peak_err_pct']:+.3f}% {r['contact_step']} "
            f"{r['mach_max']:.4f} {r['drift_end']:.3e} {r['t_end_ms']:.3f}"
        )
    print("cols:", " ".join(hdr))
    # anchor gate on the ctrl arm
    if "relx3_rriface_ctrl" in arms:
        c = rows[0]
        checks = {
            "peak_D": abs(c["peak_D"] - 1.435220) < 5e-5,
            "peak_pc_ms": abs(c["peak_t_pc_ms"] - 10.543339) < 0.001,
            "window_ms": abs(c["window_pc_ms"] - 24.005) < 0.01,
            "mach": abs(c["mach_max"] - 0.2598) < 0.001,
            "drift": abs(c["drift_end"] - 4.81e-6) < 1e-7,
            "contact": c["contact_step"] == 45,
        }
        print("ANCHOR_GATE:", "PASS" if all(checks.values()) else f"FAIL {checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
