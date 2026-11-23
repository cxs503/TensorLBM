#!/usr/bin/env python3
"""DE-X-CAVSWEEP: per-arm extraction. Reads local copies of /tmp/cavsweep/<arm>.json,
emits one summary row per arm. No interpolation, no resurrection: dead arms
report their death facts as-is."""

import json
import glob
import os
import sys


def extract(path):
    d = json.load(open(path))
    h = d.get("history") or []
    row = {
        "arm": os.path.basename(path)[:-5],
        "stop": d.get("stop_reason"),
        "contact_step": d.get("contact_step"),
        "u_limit": d.get("u_limit"),
        "gap": d.get("gap_over_D"),
        "pool": d.get("pool_depth_over_D"),
        "speed": d.get("lattice_speed"),
        "rs": d.get("release_speed_m_s"),
        "cap": d.get("capacity_redistribution"),
        "sme": d.get("simple_mass_exchange"),
        "ns": d.get("nearest_surface_velocity"),
        "n_steps": len(h),
        "t_end_ms": h[-1]["time_ms"] if h else None,
    }
    if h:
        depths = [
            (x["depth_over_D"], x["time_ms"], x["step"])
            for x in h
            if x.get("depth_over_D") is not None
        ]
        if depths:
            pk = max(depths, key=lambda t: t[0])
            row["peak_D"] = pk[0]
            row["peak_t_ms"] = pk[1]
            row["peak_step"] = pk[2]
        row["mach_max"] = max(x["mach_max"] for x in h)
        row["drift_end"] = h[-1]["volume_drift"]
        row["drift_max"] = max(abs(x["volume_drift"]) for x in h)
        row["rho_min"] = min(x["rho_min"] for x in h)
    svm = d.get("side_view_measurement")
    if svm:
        row["svm_keys"] = list(svm.keys()) if isinstance(svm, dict) else type(svm).__name__
    return row


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pat = sys.argv[1]
    else:
        d = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runs", "cavsweep"
        )
        pat = os.path.join(os.path.abspath(d), "*.json")
    rows = [extract(f) for f in sorted(glob.glob(pat))]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
