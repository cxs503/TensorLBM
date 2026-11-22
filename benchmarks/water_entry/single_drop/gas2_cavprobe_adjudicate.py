"""DE-X-GAS2-CAVPROBE-R1 adjudication (predeclared logic; run after both arms land).

Predeclared (task prescription, locked before results):
  P1: growth-phase cavity-wall pressure hugs the constant-pressure closure
      p_ref within +/-5%  (K1-DIAG-B ventilation mechanism)
  P2: Rayleigh-Plesset deviation grows monotonically with closure
  FALSIFY: growth-phase |p_cav - p_ref|/p_ref > 15%  OR  non-physical
      oscillation of p_cav in growth phase  ->  honest-FAIL
  Zero-behavior-change: arm history must be bit-identical to ctrl history.
  Anchor: ctrl must reproduce 1.4352204748 D @ 11.0906 ms, window_complete,
      contact=45, impact=1.409184408143358 m/s, mach_max=0.2598076211,
      drift_end=4.805886e-06.
"""
import json
import math
import sys
from pathlib import Path

EXP_PEAK_D = 1.857434
ANCHOR = {
    "peak_depth": 1.4352204748,
    "peak_time_ms": 11.0906,
    "stop_reason": "window_complete",
    "contact_step": 45,
    "impact_speed_m_s": 1.409184408143358,
    "mach_max": 0.2598076211,
    "drift_end": 4.805886e-06,
}


def load(p):
    return json.loads(Path(p).read_text())


def triple(res):
    h = res["history"]
    pk = max(h, key=lambda r: r["depth_over_D"])
    return {
        "peak_depth_D": pk["depth_over_D"],
        "peak_time_ms_abs": pk["time_ms"],
        "peak_error_pct": (EXP_PEAK_D - pk["depth_over_D"]) / EXP_PEAK_D * 100.0,
        "postcontact_healthy_ms": (h[-1]["time_ms"] - res["contact_step"] * (h[1]["time_ms"] - h[0]["time_ms"])) if len(h) > 1 else None,
        "stop_reason": res["stop_reason"],
        "contact_step": res["contact_step"],
        "impact_speed_m_s": res["impact_speed_m_s"],
        "mach_max": max(r["mach_max"] for r in h),
        "drift_end": h[-1]["volume_drift"],
    }


def main():
    ctrl = load(sys.argv[1])
    arm = load(sys.argv[2])
    out = {}

    # --- anchor + triple ---
    out["ctrl_triple"] = triple(ctrl)
    out["arm_triple"] = triple(arm)
    a, c = out["ctrl_triple"], out["arm_triple"]
    out["anchor"] = {
        "peak_depth": abs(a["peak_depth_D"] - ANCHOR["peak_depth"]) < 1e-9,
        "peak_time": abs(a["peak_time_ms_abs"] - ANCHOR["peak_time_ms"]) < 1e-3,
        "stop": c["stop_reason"] == ANCHOR["stop_reason"],
        "contact": c["contact_step"] == ANCHOR["contact_step"],
        "impact": abs(c["impact_speed_m_s"] - ANCHOR["impact_speed_m_s"]) < 1e-9,
        "mach_max": abs(c["mach_max"] - ANCHOR["mach_max"]) < 1e-9,
        "drift_end": abs(c["drift_end"] - ANCHOR["drift_end"]) < 1e-12,
    }

    # --- zero behavior change: ctrl vs arm history field-wise ---
    hc, ha = ctrl["history"], arm["history"]
    n = min(len(hc), len(ha))
    maxdiff = 0.0
    for i in range(n):
        for k in hc[i]:
            if k in ha[i] and isinstance(hc[i][k], (int, float)) and isinstance(ha[i][k], (int, float)):
                maxdiff = max(maxdiff, abs(hc[i][k] - ha[i][k]))
    out["zero_behavior_change"] = {"n_rows": n, "max_abs_diff": maxdiff, "identical": maxdiff == 0.0 and len(hc) == len(ha)}

    # --- probe data ---
    cp = arm.get("cav_probe")
    if not cp or not cp.get("data"):
        out["verdict"] = "NO_PROBE_DATA"
        print(json.dumps(out, indent=1))
        return
    data = cp["data"]
    p_ref = cp["vapor_pressure_lattice"]  # 1.0 lattice units (constant-pressure closure)
    out["probe_meta"] = {
        "n_samples": len(data),
        "contact_step": cp["contact_step"],
        "T_rayleigh_s": cp["T_rayleigh_s"],
        "R0_m": cp["cavity_center_radius_m"],
        "p_ref_lattice": p_ref,
        "p_inf_lattice": cp["ambient_pressure_lattice"],
    }

    # growth phase = first sample .. peak-depth sample; closure = peak .. last
    pk_i = max(range(len(data)), key=lambda i: data[i]["depth_over_D"])
    growth = data[: pk_i + 1]
    closure = data[pk_i:]
    out["phases"] = {
        "growth_steps": [growth[0]["step"], growth[-1]["step"]],
        "closure_steps": [closure[0]["step"], closure[-1]["step"]],
        "peak_depth_D": data[pk_i]["depth_over_D"],
        "peak_time_ms": data[pk_i]["time_ms"],
    }

    # P1: growth-phase deviation from p_ref
    dev_g = [abs(d["p_cav_lattice"] - p_ref) / p_ref for d in growth]
    out["P1_growth"] = {
        "max_dev_pct": max(dev_g) * 100,
        "mean_dev_pct": sum(dev_g) / len(dev_g) * 100,
        "min_p_cav": min(d["p_cav_lattice"] for d in growth),
        "max_p_cav": max(d["p_cav_lattice"] for d in growth),
        "hit_5pct": max(dev_g) <= 0.05,
        "falsified_15pct": max(dev_g) > 0.15,
    }

    # non-physical oscillation: sign flips of (p_cav - p_ref) in growth phase
    sgn = [1 if d["p_cav_lattice"] - p_ref >= 0 else -1 for d in growth]
    flips = sum(1 for i in range(1, len(sgn)) if sgn[i] != sgn[i - 1])
    amp = max(abs(d["p_cav_lattice"] - p_ref) for d in growth)
    out["oscillation_growth"] = {"sign_flips": flips, "max_amp": amp,
                                "nonphysical": flips >= 3 and amp > 0.05}

    # P2: RP deviation monotone growth in closure phase
    dev_c = [abs(d["p_cav_lattice"] - d["p_rp_lattice"]) for d in closure]
    half = len(dev_c) // 2
    h1, h2 = dev_c[:half], dev_c[half:]
    # spearman-ish: fraction of pairs (i<j) with dev_i <= dev_j
    pairs, ok = 0, 0
    for i in range(len(dev_c)):
        for j in range(i + 1, len(dev_c)):
            pairs += 1
            ok += dev_c[i] <= dev_c[j]
    mono_frac = ok / pairs if pairs else 0.0
    out["P2_closure"] = {
        "n": len(dev_c),
        "dev_first_half_mean": sum(h1) / len(h1) if h1 else None,
        "dev_second_half_mean": sum(h2) / len(h2) if h2 else None,
        "monotone_pair_fraction": mono_frac,
        "hit": mono_frac >= 0.9 and (not h2 or not h1 or sum(h2) / len(h2) > sum(h1) / len(h1)),
    }

    # RP line values at key times (for the card table)
    out["rp_at_peak"] = {k: data[pk_i][k] for k in ("p_cav_lattice", "p_rp_lattice", "R_t", "T_collapse_s")}
    out["rp_at_end"] = {k: data[-1][k] for k in ("p_cav_lattice", "p_rp_lattice", "R_t", "T_collapse_s", "time_ms")}

    # verdict
    p1 = out["P1_growth"]
    p2 = out["P2_closure"]
    if p1["falsified_15pct"] or out["oscillation_growth"]["nonphysical"]:
        out["verdict"] = "FALSIFIED (constant-pressure closure distorted -> missing term)"
    elif p1["hit_5pct"] and p2["hit"]:
        out["verdict"] = "HIT (constant-pressure closure consistent; RP line diverges with closure)"
    else:
        out["verdict"] = "PARTIAL (see P1/P2 detail)"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
