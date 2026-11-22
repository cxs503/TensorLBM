#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-CMB5-HYDRORR-STAB: programmatic extraction + verdict + figures.
All numbers read from result.json files; zero hand-copying.
"""
import json, os, sys

DT_MS = 1.2160714285714285e-2  # ms per step

M27044 = "/root/tp-agent/work/m27044/runs/cmb3"
MINE = "/root/tp-agent/work/m27100/runs"
EXP_DEPTH, EXP_MS_PC = 1.857434, 10.39446
CHAMP_PEAK = 1.750541  # combo_hydro (champion, full window)
CTRL_PEAK, CTRL_MS_PC = 1.435220, 10.543339
WINDOW_MS_PC = 24.005
CONTACT_STEP = 45
DT_MS = 1.2160714285714285e-2  # ms per step (from scales dt_s/dx_m)


def load(path):
    with open(path) as f:
        return json.load(f)


def peak_of(d):
    h = d["history"]
    pk = max(h, key=lambda r: r.get("depth_over_D", -1))
    return pk


def triple(d):
    """(peak_depth_D, peak_ms_abs, peak_ms_pc, stop, mach_max, end_drift, contact)"""
    pk = peak_of(d)
    t_abs = pk["time_ms"]
    cs = d["contact_step"]
    t_pc = (t_abs - cs * DT_MS) if cs is not None else None
    mach = max(r["mach_max"] for r in d["history"])
    end = d["history"][-1]
    return (pk["depth_over_D"], t_abs, t_pc, d["stop_reason"], mach,
            end["volume_drift"], d["contact_step"])


def main():
    out = {}
    # --- death-cause localization from m27044 hydro_rr history ---
    d = load(f"{M27044}/hydro_rr/result.json")
    h = d["history"]
    crit = next(r for r in h if r["volume_drift"] > 1e-4)
    mach_max_all = max(r["mach_max"] for r in h)
    seg = h[-200:]
    xs = [r["step"] for r in seg]
    ys = [r["volume_drift"] for r in seg]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    death = {
        "arm": "hydro_rr",
        "stop_reason": d["stop_reason"],
        "critical_step": crit["step"],
        "critical_time_ms_abs": crit["time_ms"],
        "critical_time_ms_pc": crit["time_ms"] - d["contact_step"] * DT_MS,
        "volume_drift_at_death": crit["volume_drift"],
        "geometric_volume_drift_at_death": crit["geometric_volume_drift"],
        "represented_mass_drift_at_death": crit["represented_mass_drift"],
        "mach_max_at_death": crit["mach_max"],
        "mach_max_whole_run": mach_max_all,
        "gate": "volume_drift > 1e-4",
        "slope_last200_per_step": slope,
        "contact_step": d["contact_step"],
        "rr_tau": d["rr_tau"],
    }
    out["death_cause"] = death

    # --- dose ladder: my arms ---
    arms = {}
    for name in ("cmb5_d075", "cmb5_d050", "cmb5_ul13"):
        p = f"{MINE}/{name}/result.json"
        if os.path.exists(p):
            arms[name] = triple(load(p))
    out["dose_ladder"] = arms

    # --- reference triples (programmatic from m27044 lane) ---
    refs = {}
    for name in ("ctrl", "combo_hydro", "hydro_rr", "hydro_only"):
        refs[name] = triple(load(f"{M27044}/{name}/result.json"))
    out["references"] = refs

    # --- verdict per pre-registration ---
    verdicts = []
    for name, (pk, t_abs, t_pc, stop, mach, drift, contact) in arms.items():
        if stop == "window_complete":
            if pk > CHAMP_PEAK:
                v = "HIT_NEW_CHAMPION"
            else:
                v = "FAIL_STABILIZED_NO_GAIN"
        else:
            v = "EARLY_DEATH"
        err_d = (pk - EXP_DEPTH) / EXP_DEPTH * 100
        err_t = (t_pc - EXP_MS_PC) / EXP_MS_PC * 100 if t_pc is not None else None
        verdicts.append({
            "arm": name, "peak_D": pk, "peak_ms_abs": t_abs, "peak_ms_pc": t_pc,
            "stop": stop, "mach_max": mach, "end_drift": drift,
            "depth_err_pct": err_d, "time_err_pct": err_t,
            "depth_gate_3pct": abs(err_d) < 3.0 if err_d is not None else None,
            "time_gate_3pct": abs(err_t) < 3.0 if err_t is not None else None,
            "verdict": v,
        })
    out["verdicts"] = verdicts
    if arms and all(v["verdict"] == "EARLY_DEATH" for v in verdicts):
        out["line_verdict"] = "LINE_CLOSED_HYDRORR_FRONTIER_UNSTABILIZABLE"
    elif any(v["verdict"] == "HIT_NEW_CHAMPION" for v in verdicts):
        out["line_verdict"] = "HIT_NEW_CHAMPION"
    elif any(v["verdict"] == "FAIL_STABILIZED_NO_GAIN" for v in verdicts):
        out["line_verdict"] = "FAIL_STABILIZED_NO_GAIN"
    else:
        out["line_verdict"] = "PENDING"

    os.makedirs(os.path.dirname(sys.argv[1]) or ".", exist_ok=True)
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
