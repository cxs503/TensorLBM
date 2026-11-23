"""DE-X-CMB9-IFACETUNE: programmatic triple extraction + gate verdict.

Reads runs/cmb9_*/result.json, extracts the measurement triple
(peak_depth / peak_error / peak_time_pc) plus postcontact_healthy_ms,
drift_end, mach_max, and applies the pre-registered dual gates
(unchanged from CMB8).  All numbers are injected from the JSON (no
hand-copy).  Exchange rate is reported vs the rr_iface base
(1.6216 D @ 10.227 ms pc).
"""
from __future__ import annotations

import json
from pathlib import Path

EXP_DEPTH = 1.857434  # D, fig5_experiment.csv peak
EXP_TIME = 10.39446  # ms pc, fig5_experiment.csv peak
DEPTH_GATE = 1.8017  # D  (= EXP_DEPTH * 0.97)
TIME_GATE = 11.4334  # ms pc (= EXP_TIME * 1.10)
DEPTH_FAIL = 1.70  # D
DRIFT_GATE = 1e-4
MACH_GATE = 0.3

# rr_iface base (RELX3 / CMB8 re-check) for exchange-rate accounting
BASE_DEPTH = 1.6216
BASE_TIME = 10.227

ARMS = [
    ("cmb9_a_z27", "rr_iface+SMAG0.075+HYDRO27 (cells=8)", "main"),
    ("cmb9_b_cs0_z245", "rr_iface+HYDRO24.5, Cs=0 (cells=8)", "main"),
    ("cmb9_c12", "rr_iface+SMAG0.075+HYDRO24.5 (cells=12)", "main"),
    ("cmb9_c12_ctrl", "BGK ctrl (cells=12)", "anchor"),
    ("cmb9_d_cs05_z245", "rr_iface+SMAG0.05+HYDRO24.5 (cells=8)", "main"),
    ("cmb9_base8", "rr_iface base (cells=8)", "anchor"),
]


def analyze(rj: dict) -> dict:
    hist = rj["history"]
    contact = rj.get("contact_step")
    peak_row = max(hist, key=lambda r: r["depth_over_D"])
    peak_depth = peak_row["depth_over_D"]
    peak_t_abs = peak_row["time_ms"]
    if contact is not None:
        contact_t = next(r["time_ms"] for r in hist if r["step"] == contact)
    else:
        contact_t = None
    peak_t_pc = (peak_t_abs - contact_t) if contact_t is not None else None
    last = hist[-1]
    last_pc = (last["time_ms"] - contact_t) if contact_t is not None else None
    return {
        "stop_reason": rj["stop_reason"],
        "contact_step": contact,
        "peak_depth": peak_depth,
        "peak_t_abs_ms": peak_t_abs,
        "peak_t_pc_ms": peak_t_pc,
        "peak_err_pct": 100.0 * (peak_depth - EXP_DEPTH) / EXP_DEPTH,
        "time_err_pct": (
            100.0 * (peak_t_pc - EXP_TIME) / EXP_TIME if peak_t_pc is not None else None
        ),
        "postcontact_ms": last_pc,
        "drift_end": last["volume_drift"],
        "mach_max": max(r["mach_max"] for r in hist),
        "cells_per_D": rj.get("cells_per_D"),
        "collision_mode": rj.get("collision_mode"),
        "C_s": rj.get("C_s"),
        "rr_tau": rj.get("rr_tau"),
    }


def verdict(a: dict) -> str:
    if a["stop_reason"] != "window_complete":
        return "EARLY_DEATH"
    if a["drift_end"] > DRIFT_GATE or a["mach_max"] > MACH_GATE:
        return "UNSTABLE"
    d, t = a["peak_depth"], a["peak_t_pc_ms"]
    if d > DEPTH_GATE and t < TIME_GATE:
        return "DUAL_PASS"
    if d > DEPTH_GATE:
        return "DEPTH_ONLY"
    if t < TIME_GATE and DEPTH_FAIL <= d <= DEPTH_GATE:
        return "TIME_HOLD"
    if d < DEPTH_FAIL:
        return "FAIL"
    return "NEAR_MISS"


def main() -> None:
    out = {}
    for name, label, role in ARMS:
        p = Path(f"runs/{name}/result.json")
        if not p.exists():
            out[name] = {"label": label, "role": role, "missing": True}
            continue
        a = analyze(json.loads(p.read_text()))
        a["label"] = label
        a["role"] = role
        a["verdict"] = verdict(a)
        if a["peak_t_pc_ms"] is not None and a["peak_depth"] != BASE_DEPTH:
            a["delta_D"] = a["peak_depth"] - BASE_DEPTH
            a["delta_t_ms"] = a["peak_t_pc_ms"] - BASE_TIME
            a["exchange_ms_per_D"] = a["delta_t_ms"] / a["delta_D"]
        out[name] = a
    Path("runs/cmb9_summary.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    for name, a in out.items():
        if a.get("missing"):
            print(f"{name}: MISSING")
            continue
        print(
            f"{name} [{a['role']}] {a['label']}: stop={a['stop_reason']} "
            f"peak={a['peak_depth']:.6f}D ({a['peak_err_pct']:+.2f}%) "
            f"t_pc={a['peak_t_pc_ms']:.4f}ms ({a['time_err_pct']:+.2f}%) "
            f"drift={a['drift_end']:.2e} mach={a['mach_max']:.4f} -> {a['verdict']}"
        )


if __name__ == "__main__":
    main()
