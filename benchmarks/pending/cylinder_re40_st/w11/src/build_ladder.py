"""Assemble formal/ladder_summary.json: G1/G2/G3 gate evaluation (frozen §7).

Reads formal/re*_D*.json (written by run_formal.py) + ref/ref_lock.json.
Zero hand-typed digits.
"""

import glob
import json

BASE = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady"
lock = json.load(open(f"{BASE}/ref/ref_lock.json"))["locked"]

summary = {}
for re_s in ("20", "40"):
    key = f"re{re_s}"
    rows = []
    for p in sorted(glob.glob(f"{BASE}/formal/re{re_s}_D*.json")):
        rows.append(json.load(open(p)))
    if not rows:
        continue
    rows.sort(key=lambda r: r["D"])
    errs = [r["err_pct"] for r in rows]
    g1 = all(abs(e) <= 3.0 for e in errs)
    diffs = [errs[i + 1] - errs[i] for i in range(len(errs) - 1)]
    g2_strict = all(d < 0 for d in diffs)  # |err| strictly decreasing
    g2_noninc = all(d <= 0 for d in diffs)
    g3 = all(
        abs(r["g3_relock"]["Re_check"] - r["re"]) < 1e-9 * r["re"] and r["R_area"] == r["R_eff"]
        for r in rows
    )
    complete = all(r["stop_reason"] in ("plateau", "cap_150tc") for r in rows)
    summary[key] = {
        "rows": [
            {
                k: r[k]
                for k in (
                    "D",
                    "cd",
                    "err_pct",
                    "tau",
                    "nu_lb",
                    "R_area",
                    "D_eff_cells",
                    "t_c_steps",
                    "steps",
                    "plateau_at",
                    "stop_reason",
                    "wall_s",
                )
            }
            for r in rows
        ],
        "cd_ref_primary": lock[key]["primary"]["value"],
        "g1_all_within_3pct": g1,
        "g2_strict_monotone": g2_strict,
        "g2_nonincreasing": g2_noninc,
        "err_diffs_pct": diffs,
        "g3_relock_chain_ok": g3,
        "ladder_complete": complete,
        "verdict": ("PASS" if (g1 and g2_strict and g3 and complete) else "FAIL"),
    }

json.dump(summary, open(f"{BASE}/formal/ladder_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
