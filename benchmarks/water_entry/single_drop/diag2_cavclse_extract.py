#!/usr/bin/env python3
"""DE-X-DIAG2-CAVCLSE: extract the crater-closure point kinematics.

Reads the ctrl and cavclse arm results (SDAA full window, cells=8, CAVSWEEP
ctrl anchor flags) and emits, into <out_dir>:

- anchor_check.json   bit-level anchor check of the ctrl arm
- history_diff.json   zero-behavior-change check (ctrl vs cavclse)
- verdict.json        pre-registered verdict on t_close / U_close
- trajectory.csv      full closure-point trajectory (step, t, x, z, depth)
- trajectory_table.md downsampled markdown table for the card
- experiment.json     the 18-point experimental cavity-depth curve

Usage: python3 diag2_cavclse_extract.py <ctrl.json> <cavclse.json> <out_dir>
"""
import csv
import json
import os
import sys

ANCHOR = {
    "peak_depth_over_D": 1.4352204748231814,
    "peak_time_ms": 11.090571428571428,
    "stop_reason": "window_complete",
    "contact_step": 45,
    "mach_max": 0.25980762113533173,
    "impact_speed_m_s": 1.4091844081433575,
    "final_volume_drift": 4.8058856125798854e-06,
    "final_time_ms": 24.55248214285714,
}
PREREG_T_CLOSE_MS = (12.0, 20.0)  # post-contact, locked pre-run


def peak_of(history):
    return max(history, key=lambda r: r["depth_over_D"])


def anchor_check(ctrl):
    h = ctrl["history"]
    pk = peak_of(h)
    out = {
        "peak_depth_over_D": pk["depth_over_D"],
        "peak_time_ms": pk["time_ms"],
        "stop_reason": ctrl.get("stop_reason"),
        "contact_step": ctrl.get("contact_step"),
        "mach_max": max(r["mach_max"] for r in h),
        "impact_speed_m_s": ctrl.get("impact_speed_m_s"),
        "final_volume_drift": h[-1]["volume_drift"],
        "final_time_ms": h[-1]["time_ms"],
    }
    out["bit_identical"] = all(out[k] == ANCHOR[k] for k in ANCHOR if k in out)
    out["per_field"] = {k: (out[k] == ANCHOR[k]) for k in ANCHOR if k in out}
    return out


def history_diff(a, b):
    ha, hb = a["history"], b["history"]
    if len(ha) != len(hb):
        return {"n_a": len(ha), "n_b": len(hb), "max_abs_delta": None,
                "identical": False}
    max_delta, worst = 0.0, None
    for i, (ra, rb) in enumerate(zip(ha, hb)):
        for k in set(ra) | set(rb):
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, float) and isinstance(vb, float):
                d = abs(va - vb)
                if d > max_delta:
                    max_delta, worst = d, (i, k)
            elif va != vb:
                return {"n_a": len(ha), "n_b": len(hb),
                        "max_abs_delta": None, "identical": False,
                        "first_mismatch": (i, k, va, vb)}
    return {"n_a": len(ha), "n_b": len(hb), "max_abs_delta": max_delta,
            "worst": worst, "identical": max_delta == 0.0}


def main():
    ctrl_p, arm_p, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    ctrl = json.load(open(ctrl_p))
    arm = json.load(open(arm_p))
    dt_s = (arm["history"][1]["time_ms"] - arm["history"][0]["time_ms"]) / 1000.0
    contact = arm.get("contact_step")

    anchor = anchor_check(ctrl)
    diff = history_diff(ctrl, arm)
    cc = arm.get("cav_closure") or {}
    rows = cc.get("rows") or []
    summ = cc.get("summary") or {}

    t_close_pc = summ.get("t_close_after_contact_ms")
    lo, hi = PREREG_T_CLOSE_MS
    in_range = t_close_pc is not None and lo <= t_close_pc <= hi
    reopens = False
    z_at_close = None
    if t_close_pc is not None and rows:
        t_close_abs = summ.get("t_close")
        after = [r for r in rows if r["time_ms"] > t_close_abs]
        z_at_close = next(r["z"] for r in rows if r["time_ms"] == t_close_abs)
        reopens = any(r["z"] < z_at_close - 0.5 for r in after)
    verdict = {
        "t_close_ms_abs": summ.get("t_close"),
        "t_close_ms_post_contact": t_close_pc,
        "U_close_m_s": summ.get("U_close"),
        "peak_step": summ.get("peak_step"),
        "peak_depth_D": summ.get("peak_depth"),
        "prereg_range_ms_pc": list(PREREG_T_CLOSE_MS),
        "t_close_in_range": bool(in_range),
        "reopens_after_t_close": bool(reopens),
        "falsified": bool(t_close_pc is None or not in_range or reopens),
    }

    # full trajectory csv
    with open(os.path.join(out_dir, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_ms_abs", "t_ms_pc", "x", "y", "z", "depth_D"])
        for r in rows:
            w.writerow([r["step"], f"{r['time_ms']:.6f}",
                        f"{r['time_ms'] - contact * 1000 * dt_s:.6f}",
                        f"{r['x']:.3f}", f"{r['y']:.3f}",
                        f"{r['z']:.4f}", f"{r['depth']:.6f}"])

    # downsampled markdown table: every 25th step + all rows within
    # [peak-25, t_close+50] steps
    keep = set()
    for i, r in enumerate(rows):
        if i % 25 == 0:
            keep.add(i)
    peak_step = summ.get("peak_step")
    t_close_abs = summ.get("t_close")
    for i, r in enumerate(rows):
        if peak_step is not None and abs(r["step"] - peak_step) <= 25:
            keep.add(i)
        if t_close_abs is not None:
            s = r["step"]
            sc = next((q["step"] for q in rows if q["time_ms"] == t_close_abs), None)
            if sc is not None and 0 <= s - sc <= 50:
                keep.add(i)
    lines = ["| step | t_ms(abs) | t_ms(pc) | x | z | depth_D |",
             "|---|---|---|---|---|---|"]
    for i in sorted(keep):
        r = rows[i]
        t_pc = r["time_ms"] - contact * 1000 * dt_s
        lines.append(f"| {r['step']} | {r['time_ms']:.4f} | {t_pc:.4f} "
                     f"| {r['x']:.2f} | {r['z']:.3f} | {r['depth']:.4f} |")
    open(os.path.join(out_dir, "trajectory_table.md"), "w").write(
        "\n".join(lines) + f"\n\n(full table: trajectory.csv, {len(rows)} rows)\n")

    exp = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fig5_experiment.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            exp = [{"t_ms": float(r["time_ms"]),
                    "depth_D": float(r["depth_over_D"])}
                  for r in csv.DictReader(f)]
    json.dump(exp, open(os.path.join(out_dir, "experiment.json"), "w"), indent=1)

    for name, obj in [("anchor_check", anchor), ("history_diff", diff),
                     ("verdict", verdict), ("closure_summary", summ)]:
        json.dump(obj, open(os.path.join(out_dir, name + ".json"), "w"), indent=1)

    print(json.dumps({"anchor_bit_identical": anchor["bit_identical"],
                      "history_identical": diff["identical"],
                      "max_abs_delta": diff["max_abs_delta"],
                      "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
