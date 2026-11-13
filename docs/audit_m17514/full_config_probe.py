#!/usr/bin/env python3
"""Auditor probe F5: does the delivered criterion hold at the *declared full*
configuration (N=64/128, Re=100 + N=128, Re=320, 1000 steps), which the shipped
artifacts never exercised (they are the --quick CI config N=32/48, 400 steps)?

Only the AA arm is run (the AA==AB bitwise equality is already verified at two
resolutions by the shipped bench and by my own reference), so 3 cases * 1000
steps fit in one short shell call.

Run: PYTHONPATH=/root/tp-agent/work/m17514/src python3 docs/audit_m17514/full_config_probe.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SRC_TREE = Path("/root/tp-agent/work/m17514")
sys.path.insert(0, str(SRC_TREE / "src"))
sys.path.insert(0, str(SRC_TREE))

spec_path = SRC_TREE / "benchmarks/xval_aa_pattern_even_odd.py"
mod_spec = importlib.util.spec_from_file_location("xval_bench", spec_path)
bench = importlib.util.module_from_spec(mod_spec)
sys.modules["xval_bench"] = bench  # dataclasses need a registered module
mod_spec.loader.exec_module(bench)

# exactly the full-config specs from benchmarks/xval_aa_pattern_even_odd.py:634-643
SPECS = [
    bench.TGSpec("tg_n64_re100", 64, 0.05, 100.0, 1000, 25, 251),
    bench.TGSpec("tg_n128_re100", 128, 0.05, 100.0, 1000, 25, 251),
    bench.TGSpec("tg_n128_re320", 128, 0.05, 320.0, 1000, 25, 251),
]

rows = {}
for s in SPECS:
    m = bench.run_arm(s, "aa")
    rows[s.name] = {
        "n": s.n, "re": s.re, "nu": round(s.nu, 6), "tau": round(s.tau, 4),
        "ma": round(s.ma, 5), "steps": s.steps,
        "gamma_K": m["gamma_K"], "gamma_K_exact": m["gamma_K_exact"],
        "gamma_K_err_pct": m["gamma_K_err_pct"], "gamma_K_r2": m["gamma_K_r2"],
        "field_l2_pct": m["field_l2_pct"],
        "mass_drift_rel": m["mass_drift_rel"], "mom_drift_rel": m["mom_drift_rel"],
        "wall_s": round(m["wall_s"], 2), "s_per_step": round(m["s_per_step"], 6),
        "gate_15pct": abs(m["gamma_K_err_pct"]) < 15.0 and m["field_l2_pct"] < 15.0,
        "under_3pct": abs(m["gamma_K_err_pct"]) < 3.0 and m["field_l2_pct"] < 3.0,
    }
    print(s.name, json.dumps(rows[s.name], ensure_ascii=False), flush=True)

out = Path(__file__).with_name("full_config_probe.json")
out.write_text(json.dumps(rows, indent=1))
print("worst gamma_K %", max(abs(v["gamma_K_err_pct"]) for v in rows.values()))
print("worst field L2 %", max(v["field_l2_pct"] for v in rows.values()))
print("field L2 vs N (Re=100):", [round(rows[k]["field_l2_pct"], 4) for k in ("tg_n64_re100", "tg_n128_re100")])