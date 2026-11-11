"""Consolidate the MCMP stratified-Poiseuille records into one analysis table.

Reads every record under ``artifacts/mcmp_stratified_poiseuille/records`` and
recomputes the flow-rate errors against the closed-form reference under **all
four** (wall convention x viscosity rule) combinations, so the reported verdict
cannot depend on a silently chosen convention:

* ``halfway``    -- no-slip planes at y = 0.5 and y = ny - 1.5 (``H = ny - 2``,
  interface at the volume-conserving ``h = half - 0.5``);
* ``cellcentre`` -- no-slip planes on the wall-cell centres (``H = ny - 1``,
  interface at ``h = half``) -- the convention asserted by the QR audit
  ``docs/REPORT_qr_pilot_a.md``.

Which one is real is decided by the control experiment (``--mode walls``), not by
preference: ``wall_calibration`` in the same records measures the effective
no-slip plane through the identical collision/streaming/bounce-back path.

The derived ``analytical`` block of every record is **rewritten in place** with
these four references (the raw simulation outputs are untouched), so a record
and this table can never disagree.  Writes
``artifacts/mcmp_stratified_poiseuille/analysis.json``.

Usage::

    PYTHONPATH=src:. python cases/mcmp_stratified_poiseuille/analyze.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cases.mcmp_stratified_poiseuille.analytic import (  # noqa: E402
    reference_block,
    wall_conventions,
)

RECORDS = os.path.join("artifacts", "mcmp_stratified_poiseuille", "records")
OUT = os.path.join("artifacts", "mcmp_stratified_poiseuille", "analysis.json")
G_X = 5e-5
KEYS = ("halfway/component", "halfway/mixture", "cellcentre/component", "cellcentre/mixture")


def main() -> int:
    files = sorted(glob.glob(os.path.join(RECORDS, "*.json")))
    walls, flow = [], []
    for path in files:
        rec = json.load(open(path))
        tag = os.path.basename(path)[:-5]
        ny = rec["config"]["ny"]
        if rec["mode"] == "walls":
            walls.append(dict(tag=tag, ny=ny, tau=rec["config"]["tau_lower"],
                              u_max_sim=rec["u_max"], **rec["wall_calibration"]))
            continue
        if not rec.get("finite", False):
            flow.append(dict(tag=tag, ny=ny, G_12=rec["config"]["G_12"],
                             u_eq=rec["config"]["u_eq"],
                             force_scheme=rec["config"]["force_scheme"],
                             finite=False, steady=False, steps=rec["config"]["steps"],
                             tau_upper=rec["config"]["tau_upper"]))
            continue
        rec["analytical"] = reference_block(rec, G_X)
        rec["analytical_recomputed_by"] = "cases/mcmp_stratified_poiseuille/analyze.py"
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        row = dict(
            tag=tag, ny=ny, G_12=rec["config"]["G_12"],
            u_eq=rec["config"]["u_eq"], force_scheme=rec["config"]["force_scheme"],
            tau_upper=rec["config"]["tau_upper"],
            steps=rec.get("steps_total")
            or (rec["history"][-1]["step"] if rec.get("history") else rec["config"]["steps"]),
            finite=rec["finite"], steady=rec["steady"],
            phi_contrast=rec["phi_contrast"], stratified=rec["stratified"],
            interface_row=rec["interface_crossing"],
            Q_mix=rec["Q_mix"], Q_mix_lower=rec["layer_flux"]["Q_mix_lower"],
            Q_mix_upper=rec["layer_flux"]["Q_mix_upper"],
            Q1=rec["Q1"], Q2=rec["Q2"],
            u_max_mix=rec["u_max_mix"],
            spurious_ratio=rec["spurious_mix"] / max(rec["u_max_mix"], 1e-30),
            references={k: {kk: v[kk] for kk in
                            ("Q_total", "Q_lower", "Q_upper", "nu_lower", "nu_upper",
                             "err_Q_mix", "err_Q_mix_lower", "err_Q_mix_upper",
                             "err_profile_L2", "profile")}
                        for k, v in rec["analytical"].items()},
        )
        flow.append(row)

    analysis = dict(
        wall_calibration=walls,
        flow=flow,
        conventions=wall_conventions(25),
        note=("err_* are relative to the closed-form two-layer solution; the wall convention "
              "is decided by wall_calibration, not by preference.  Keys are "
              "'<wall convention>/<viscosity rule>'."),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(analysis, fh, indent=1)

    print("== wall calibration (effective no-slip plane) ==")
    for w in sorted(walls, key=lambda w: (w["ny"], w["tau"])):
        h1, h2 = w["H_ny_minus_1"], w["H_ny_minus_2"]
        print(f"  ny={w['ny']:2d} tau={w['tau']:.2f}  H=ny-1: u_max {100*h1['u_max_error_rel']:+6.2f}% "
              f"Q {100*h1['Q_error_rel']:+6.2f}% | H=ny-2: u_max {100*h2['u_max_error_rel']:+6.2f}% "
              f"Q {100*h2['Q_error_rel']:+6.2f}%")
    print("== two-phase mixture flow rate ==")
    print(f"  {'tag':30s} {'ny':>2s} {'g':>4s} {'u_eq':8s} {'force':5s} "
          f"{'phi':>4s} {'steady':6s} {'steps':>6s} {'Qmix':>7s} "
          f"{'hw/comp':>8s} {'hw/mixt':>8s} {'cc/comp':>8s} {'cc/mixt':>8s}")
    for row in sorted(flow, key=lambda r: (r["ny"], r["u_eq"], r["force_scheme"], r.get("steps", 0))):
        if not row.get("finite", False):
            print(f"  {row['tag']:30s} {row['ny']:2d} {row['G_12']:+4.1f} {row['u_eq']:8s} "
                  f"{row['force_scheme']:5s} DIVERGED")
            continue
        r = row["references"]
        print(f"  {row['tag']:30s} {row['ny']:2d} {row['G_12']:+4.1f} {row['u_eq']:8s} "
              f"{row['force_scheme']:5s} {row['phi_contrast']:4.2f} {str(row['steady']):6s} "
              f"{row['steps']:6d} {row['Q_mix']:7.4f} "
              + " ".join(f"{100*r[k]['err_Q_mix']:+7.1f}%" for k in KEYS))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())