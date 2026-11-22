#!/usr/bin/env python3
"""K1-CAVCLOS-R2 verdict analysis: programmatic triple extraction + pre-registration check.

Usage: python3 tools/k1r2_analyze.py <result.json> [--alpha 0.5] [--out verdict.json]
Zero hand-copy: every number printed comes from the result.json file.
"""
import json
import sys

EXP_PEAK_D = 1.857434          # experiment peak depth (D)
EXP_PEAK_MS_PC = 10.394        # experiment peak time (ms, post-contact)
CTRL_PEAK_D = 1.435220         # campaign ctrl anchor (D)
CTRL_PEAK_MS_ABS = 11.091      # ctrl peak time (ms, absolute)
WINDOW_MS_PC = 24.005          # full window (ms, post-contact)
CONTACT_STEP = 45
PREREG_LO, PREREG_HI = 1.50, 1.70
FALSIFY_NEAR_CTRL = 0.05       # |peak - ctrl| < 0.05 D => no effect
EARLY_DEATH_MS_PC = 10.394     # death before this => falsified


def load(path):
    with open(path) as fh:
        return json.load(fh)


def triple(res):
    h = res["history"]
    last = h[-1]
    pk = max(h, key=lambda r: r.get("depth_over_D", -1.0))
    peak_d = float(pk["depth_over_D"])
    sc = res["scales"]
    dt_ms = 1000.0 * sc["dt_s"]
    contact = res.get("contact_step") or 0
    pk_abs = float(pk["time_ms"])
    pk_pc = pk_abs - contact * dt_ms
    last_pc = float(last["time_ms"]) - contact * dt_ms
    peak_err = (peak_d - EXP_PEAK_D) / EXP_PEAK_D * 100.0
    return {
        "stop_reason": res["stop_reason"],
        "contact_step": contact,
        "n_history": len(h),
        "last_step_index": len(h) - 1,
        "last_time_ms_abs": float(last["time_ms"]),
        "last_time_ms_pc": last_pc,
        "peak_depth_D": peak_d,
        "peak_time_ms_abs": pk_abs,
        "peak_time_ms_pc": pk_pc,
        "peak_error_vs_exp_pct": peak_err,
        "postcontact_healthy_ms": last_pc,
        "mach_max": max(float(r.get("mach_max", 0.0)) for r in h),
    }


def verdict(t, alpha):
    checks = {}
    in_band = PREREG_LO <= t["peak_depth_D"] <= PREREG_HI
    near_ctrl = abs(t["peak_depth_D"] - CTRL_PEAK_D) < FALSIFY_NEAR_CTRL
    early_death = t["postcontact_healthy_ms"] < EARLY_DEATH_MS_PC
    checks["prereg_band_hit"] = in_band
    checks["falsify_near_ctrl"] = near_ctrl
    checks["falsify_early_death"] = early_death
    if near_ctrl or early_death:
        checks["verdict"] = "FAIL"
    elif in_band:
        checks["verdict"] = "PASS_BAND"
    else:
        checks["verdict"] = "OUT_OF_BAND"
    checks["alpha"] = alpha
    return checks


def main():
    argv = sys.argv[1:]
    path = argv[0]
    alpha = 0.5
    if "--alpha" in argv:
        alpha = float(argv[argv.index("--alpha") + 1])
    res = load(path)
    t = triple(res)
    v = verdict(t, alpha)
    out = {"triple": t, "verdict": v}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if "--out" in argv:
        with open(argv[argv.index("--out") + 1], "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
