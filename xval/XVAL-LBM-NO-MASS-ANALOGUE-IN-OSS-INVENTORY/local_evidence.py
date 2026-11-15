"""Light local re-run of the minimum evidence set (same JSON schema as run_xval.py).

Used when the cluster queue is deep: every arm here is < 20 s on one core, i.e.
inside the local short-test envelope.  Writes ``runs/<arm>/result.json`` so that
``analyze.py`` / ``make_figures.py`` consume it unchanged, plus
``reference/potential_grid.json`` (grid convergence of the potential reference).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_xval import BASE, diagnostic_arm  # noqa: E402

from tensorlbm.oscillatory_cylinder_inertia import (  # noqa: E402
    ArmCase,
    potential_reference,
    run_arm,
)

ARMS = [
    ("rep", "rep_d8", dict(BASE, D=8, transient=1.5, measure=3.0)),
    ("rep", "rep_d12", dict(BASE, D=12, transient=1.5, measure=3.0)),
    ("rep", "rep_d16", dict(BASE, D=16, transient=1.5, measure=3.0)),
    ("protocol", "proto_t1p5_m3", dict(BASE, transient=1.5, measure=3.0)),
    ("protocol", "proto_t3_m3", dict(BASE, transient=3.0, measure=3.0)),
    ("protocol", "proto_t8_m5", dict(BASE, transient=8.0, measure=5.0)),
]


def main() -> int:
    runs = HERE / "runs"
    manifest = []
    for group, name, cfg in ARMS:
        cfg = dict(cfg)
        cfg["name"] = name
        case = ArmCase(**cfg)
        res, _trace = run_arm(case, threads=4)
        res.update({"status": "ok", "arm": name, "group": group})
        (runs / name).mkdir(parents=True, exist_ok=True)
        (runs / name / "result.json").write_text(
            json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")
        manifest.append({"arm": name, "group": group, "status": "ok", "cfg": cfg})
        k = res["kpi"]["E1_mem_area"]
        print(f"{name}: C_I={k['C_I']:.4f} C_Q={k['C_Q']:.4f} "
              f"resid={res['harmonic']['E1_mem']['resid_rel']:.3f} "
              f"c_a_pot={res['reference']['potential_same_mask_area']:.4f}", flush=True)

    # diagnostic arm (surface pressure dipole + global momentum balance)
    dcfg = dict(BASE, transient=3.0, measure=3.0)
    diag = diagnostic_arm(ArmCase(name="e3_beta_50", **dcfg), threads=4)
    diag.update({"status": "ok", "arm": "e3_beta_50", "group": "e3"})
    (runs / "e3_beta_50").mkdir(parents=True, exist_ok=True)
    (runs / "e3_beta_50" / "result.json").write_text(
        json.dumps(diag, indent=2, sort_keys=True, default=str) + "\n")
    manifest.append({"arm": "e3_beta_50", "group": "e3", "status": "ok",
                     "cfg": dict(dcfg, name="e3_beta_50")})
    print("e3_beta_50 diagnostics:", json.dumps(diag["diagnostics"], indent=1), flush=True)
    (runs / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # grid convergence of the same-mask potential reference (box = 4D fixed)
    grid = []
    for D in (16, 32, 64):
        pot = potential_reference(4 * D, 4 * D, float(D))
        grid.append({"D": D, "nx": 4 * D, "c_a_raw": pot["c_a"],
                     "c_a_area": pot["c_a_area"], "c_a_halfway": pot["c_a_halfway"],
                     "body_area": pot["area_solid"],
                     "area_over_analytic": pot["area_solid"] / (3.141592653589793 * (D / 2) ** 2)})
        print("potential grid", grid[-1], flush=True)
    (HERE / "reference").mkdir(exist_ok=True)
    (HERE / "reference" / "potential_grid.json").write_text(json.dumps(grid, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())