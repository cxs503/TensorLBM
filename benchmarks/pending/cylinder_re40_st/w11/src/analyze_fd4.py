"""FD4 own-solver ladder analysis: discretization/rmax/theta tables + Richardson
extrapolation in rmax, vs locked reference. Zero hand-typed digits (refs read
from ref/ref_lock.json)."""

import glob
import json

BASE = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady"
ref = json.load(open(f"{BASE}/ref/ref_lock.json"))
REF = {20: ref["locked"]["re20"]["primary"]["value"], 40: ref["locked"]["re40"]["primary"]["value"]}

cases = {}
for p in sorted(glob.glob(f"{BASE}/diag/fd4_*.json")):
    if p.endswith("fd4_analysis.json"):
        continue
    d = json.load(open(p))
    cases[d["re"]] = cases.get(d["re"], [])
    cases[d["re"]].append(d)

out = {"per_case": {}, "ladders": {}}
for p in sorted(glob.glob(f"{BASE}/diag/fd4_*.json")):
    if p.endswith("fd4_analysis.json"):
        continue
    d = json.load(open(p))
    e = d["re"]
    err = (d["cd"] / REF[e] - 1.0) * 100.0 if e in REF else None
    out["per_case"][p.split("/")[-1]] = {
        "re": e,
        "n_r": d["n_r"],
        "n_th": d["n_th"],
        "r_max": d["r_max"],
        "beta": d["beta"],
        "cd": d["cd"],
        "cd_p": d["cd_p"],
        "cd_f": d["cd_f"],
        "err_pct_vs_primary": err,
        "stop_reason": d["stop_reason"],
        "iters": d["iters"],
        "res": d["res"],
        "theta_s_deg": d["theta_s_deg"],
        "lw_over_d": d["lw_over_d"],
        "lw_over_d_corrected": d["lw_over_d"] / 2.0,
    }

# rmax ladder at Re=40 (n_r=240, n_th=128 fixed)
lad = sorted(
    [d for d in cases.get(40.0, []) if d["n_r"] == 240 and d["n_th"] == 128],
    key=lambda d: d["r_max"],
)
if len(lad) >= 3:
    cd = [d["cd"] for d in lad]
    rm = [d["r_max"] for d in lad]
    d1, d2 = cd[1] - cd[0], cd[2] - cd[1]
    ratio = d2 / d1 if d1 != 0 else None
    rich = cd[2] + d2 * d2 / (d1 - d2) if d1 != d2 else None  # Aitken/Richardson
    out["ladders"]["rmax_re40"] = {
        "r_max": rm,
        "cd": cd,
        "diff_first": d1,
        "diff_second": d2,
        "ratio": ratio,
        "richardson_limit": rich,
        "richardson_err_pct": (rich / REF[40] - 1.0) * 100.0,
    }

json.dump(out, open(f"{BASE}/diag/fd4_analysis.json", "w"), indent=1)
print(json.dumps(out["ladders"], indent=1))
for k, v in out["per_case"].items():
    print(
        f"{k:22s} Re={v['re']:5.1f} rmax={v['r_max']:5.1f} nr={v['n_r']:3d} "
        f"nth={v['n_th']:3d} Cd={v['cd']:.5f} err={v['err_pct_vs_primary']}% "
        f"stop={v['stop_reason']} th_s={v['theta_s_deg']} Lw={v['lw_over_d']}"
    )
