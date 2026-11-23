"""DE-X-CMB8-IFACECOMBO: programmatic triple extraction + gate verdict.

Reads runs/cmb8_*/result.json, extracts the measurement triple
(peak_depth / peak_error / peak_time_pc) plus postcontact_healthy_ms,
drift_end, mach_max, and applies the pre-registered dual gates.  All numbers
are injected from the JSON (no hand-copy).
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

ARMS = [
    ("cmb8_a_cs075_z22", "rr_iface+SMAG0.075+HYDRO22", "main"),
    ("cmb8_b_cs075_z245", "rr_iface+SMAG0.075+HYDRO24.5", "main"),
    ("cmb8_c_cs01_z22", "rr_iface+SMAG0.1+HYDRO22", "main"),
    ("cmb8_ctrl", "BGK ctrl", "anchor"),
    ("cmb8_base", "rr_iface base", "anchor"),
    ("cmb8_diag_smag", "rr_iface+SMAG0.075 (300s)", "diag"),
    ("cmb8_diag_hydro", "rr_iface+HYDRO22 (300s)", "diag"),
]


def analyze(rj: dict) -> dict:
    hist = rj["history"]
    contact = rj.get("contact_step")
    # peak depth and its absolute time
    peak_row = max(hist, key=lambda r: r["depth_over_D"])
    peak_depth = peak_row["depth_over_D"]
    peak_t_abs = peak_row["time_ms"]
    # contact absolute time
    if contact is not None:
        contact_t = next(r["time_ms"] for r in hist if r["step"] == contact)
    else:
        contact_t = None
    peak_t_pc = (peak_t_abs - contact_t) if contact_t is not None else None
    last = hist[-1]
    # postcontact healthy window: last healthy time minus contact time
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
        "drift_max": max(r["volume_drift"] for r in hist),
        "n_steps": len(hist),
        "collision_mode": rj.get("collision_mode"),
        "C_s": rj.get("C_s"),
        "rr_tau": rj.get("rr_tau"),
    }


def verdict(a: dict) -> str:
    if a["stop_reason"] != "window_complete":
        return "FAIL(early-death)"
    d, t = a["peak_depth"], a["peak_t_pc_ms"]
    if d > DEPTH_GATE and t < TIME_GATE:
        return "DUAL_PASS"
    if d > DEPTH_GATE:
        return "DEPTH_ONLY"
    if t < TIME_GATE and DEPTH_FAIL <= d < DEPTH_GATE:
        return "TIME_HOLD"
    if d < DEPTH_FAIL:
        return "FAIL"
    # gap: depth in [1.70, 1.8017) but time >= 11.4334 (both marginal)
    return "NEAR_MISS"


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "runs"
    out = {}
    for name, label, role in ARMS:
        p = root / name / "result.json"
        if not p.exists():
            out[name] = {"missing": True, "label": label, "role": role}
            continue
        a = analyze(json.loads(p.read_text()))
        a["label"] = label
        a["role"] = role
        a["verdict"] = verdict(a)
        out[name] = a
    (root / "cmb8_summary.json").write_text(
        json.dumps(out, indent=1, allow_nan=False) + "\n"
    )
    # console table
    hdr = (
        f"{'arm':<18}{'role':<8}{'stop':<16}{'peak_D':>9}{'err%':>8}"
        f"{'t_pc_ms':>10}{'t_err%':>9}{'pc_ms':>8}{'drift':>10}{'mach':>8}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, label, role in ARMS:
        a = out[name]
        if a.get("missing"):
            print(f"{name:<18}{role:<8}{'(not run)':<16}")
            continue
        t_pc = f"{a['peak_t_pc_ms']:.4f}" if a["peak_t_pc_ms"] is not None else "n/a"
        t_err = f"{a['time_err_pct']:+.2f}" if a["time_err_pct"] is not None else "n/a"
        pc = f"{a['postcontact_ms']:.3f}" if a["postcontact_ms"] is not None else "n/a"
        print(
            f"{name:<18}{role:<8}{a['stop_reason']:<16}{a['peak_depth']:>9.6f}"
            f"{a['peak_err_pct']:>8.2f}{t_pc:>10}{t_err:>9}{pc:>8}"
            f"{a['drift_end']:>10.2e}{a['mach_max']:>8.4f}  {a['verdict']}"
        )
    # exchange rate table (main arms vs base)
    base = out.get("cmb8_base")
    if base and not base.get("missing"):
        print("\nExchange rate (vs rr_iface base):")
        for name, label, role in ARMS:
            if role != "main":
                continue
            a = out[name]
            if a.get("missing") or a["stop_reason"] != "window_complete":
                continue
            dd = a["peak_depth"] - base["peak_depth"]
            dt = (a["peak_t_pc_ms"] - base["peak_t_pc_ms"]) if (
                a["peak_t_pc_ms"] is not None and base["peak_t_pc_ms"] is not None
            ) else None
            rate = (dt / dd) if (dd and dt is not None) else None
            print(
                f"  {name}: dD={dd:+.4f} dt={dt if dt is None else format(dt,'+.4f')}ms"
                f" rate={rate if rate is None else format(rate,'+.3f')}ms/D"
            )


if __name__ == "__main__":
    main()
