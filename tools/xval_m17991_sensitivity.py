#!/usr/bin/env python3
"""Sensitivity ladders for the m17991 model units (predeclared in predeclared.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import torch

from tensorlbm.model_units import discover

torch.set_num_threads(1)
OUT = REPO / "artifacts" / "xval_m17991" / "sensitivity.json"
units = {u.name: u for u in discover(REPO / "models" / "lbm", repo_root=REPO)}


def run(name, **over):
    unit = units[name]
    mod = unit.case_path
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"case_{name}", mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    params = dict(unit.params)
    params.update(over)
    return m.run(params)


out = {
    "poiseuille2d_resolution": [],
    "poiseuille2d_steps_rest": [],
    "poiseuille2d_tau": [],
    "tgv2d_resolution": [],
}
for ny in (16, 32, 64):
    r = run("poiseuille2d_d2q9_guo", ny=ny)
    out["poiseuille2d_resolution"].append(
        {
            "ny": ny,
            "kpi": r["kpis"]["u_profile_rel_sup_err"],
            "u_max_rel_err": r["metrics"]["u_max_rel_err"],
        }
    )
for steps in (500, 1500, 3000):
    r = run("poiseuille2d_d2q9_guo", steps=steps, init="rest")
    out["poiseuille2d_steps_rest"].append(
        {
            "steps": steps,
            "kpi": r["kpis"]["u_profile_rel_sup_err"],
            "u_max": r["metrics"]["u_max_numeric"],
        }
    )
for tau in (0.6, 0.8, 1.2):
    r = run("poiseuille2d_d2q9_guo", tau=tau)
    out["poiseuille2d_tau"].append(
        {"tau": tau, "nu": r["metrics"]["nu"], "kpi": r["kpis"]["u_profile_rel_sup_err"]}
    )
for n in (16, 32, 64):
    r = run("tgv2d_d2q9_bgk", n=n)
    out["tgv2d_resolution"].append(
        {
            "n": n,
            "kpi": r["kpis"]["decay_rate_rel_err"],
            "field_l2": r["metrics"]["field_l2_rel_err"],
        }
    )
OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2)[:4000])
