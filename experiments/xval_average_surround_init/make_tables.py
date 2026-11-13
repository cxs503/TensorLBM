#!/usr/bin/env python
"""Render ``artifacts/xval_results.json`` into the report tables (zero hand-copy).

    PYTHONPATH=src python experiments/xval_average_surround_init/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "artifacts" / "xval_results.json"
OUT = HERE / "artifacts" / "xval_tables.md"
THRESHOLD = 0.15


def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}e}" if 0 < abs(v) < 1e-2 else f"{v:.{nd}f}"
    return str(v)


def main() -> str:
    d = json.loads(RES.read_text())
    L: list[str] = []
    L.append(f"### Operator table (ref = OSS literal `average_surround`, float64)\n")
    L.append("| case | active nbrs | ref rho | ref ux | got rho (avg) | rel_f (avg) | got rho (fixed) | rel_f (fixed) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for row in d["operator_cases"]:
        avg = row["modes"]["average_surround"]
        fix = row["modes"]["fixed_rho"]
        L.append(
            f"| {row['case']} | {row['active_neighbours']} | {fmt(row['ref']['rho'])} | "
            f"{fmt(row['ref']['ux'])} | {fmt(avg['rho'])} | {fmt(avg['rel_f_linf'], 2)} | "
            f"{fmt(fix['rho'])} | {fmt(fix['rel_f_linf'], 2)} |"
        )
    iso = d["isolated_cell_deviation"]
    L.append("")
    L.append(
        f"### Isolated-cell case (documented deviation)\n\n"
        f"active neighbours = {iso['active_neighbours']}; OSS ref rho = "
        f"{fmt(iso['ref_oss_rho'])} (set_equi -> empty population); "
        f"TensorLBM `average_surround` rho = {fmt(iso['modes']['average_surround']['rho'])} "
        f"(population {fmt(iso['modes']['average_surround']['population'])}); "
        f"TensorLBM `fixed_rho` rho = {fmt(iso['modes']['fixed_rho']['rho'])}; "
        f"historical zero fallback (`rho_init=None`) rho = "
        f"{fmt(iso['modes']['legacy_zero_fallback']['rho'])} "
        f"(population {fmt(iso['modes']['legacy_zero_fallback']['population'])}) "
        f"= OSS state reproduced."
    )
    L.append("")
    L.append("### Sensitivity tables")
    L.append("")
    L.append("| u amplitude | rel_f (average_surround) | rel_f (fixed_rho) |")
    L.append("|---|---|---|")
    for r in d["sensitivity_velocity"]:
        L.append(f"| {r['u_amp']} | {fmt(r['avg_rel_f'], 2)} | {fmt(r['fixed_rel_f'], 2)} |")
    L.append("")
    L.append("| d(rho)/dx per cell | rel_rho (average_surround) | rel_rho (fixed_rho) |")
    L.append("|---|---|---|")
    for r in d["sensitivity_density_contrast"]:
        L.append(
            f"| {r['rho_slope_per_cell']} | {fmt(r['avg_rel_rho'], 2)} | "
            f"{fmt(r['fixed_rel_rho'], 2)} |"
        )
    L.append("")
    L.append("### System table (2D dam break)")
    L.append("")
    L.append("| grid | steps | mode | births (cells) | inventory | rel drift | front [cells] | iface | finite |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in d["solver_runs"]:
        L.append(
            f"| {r['grid']} | {r['steps']} | {r['mode']} | {r['births']['cells']} | "
            f"{r['inventory']:.6f} | {fmt(r['inventory_drift_rel'], 2)} | {r['front']:.0f} | "
            f"{r['n_interface']} | {r['finite']} |"
        )
    L.append("")
    L.append("### Mode divergence (legacy baseline vs selected mode)")
    L.append("")
    L.append("| grid | mode | rel L-inf df | front legacy | front mode | front rel diff | inv rel diff | <15 % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in d["solver_mode_divergence"]:
        L.append(
            f"| {r['grid']} | {r['mode']} | {fmt(r['rel_linf_f_state_diff'], 2)} | "
            f"{r['front_legacy']:.0f} | {r['front_mode']:.0f} | "
            f"{r['front_rel_diff']*100:.2f}% | {r['inventory_rel_diff']*100:.3f}% | "
            f"{'yes' if r['under_15pct'] else 'no'} |"
        )
    v = d["verdict"]
    L.append("")
    L.append("### Verdict")
    L.append("")
    L.append(f"- operator max rel_f, `average_surround`: **{fmt(v['operator_average_surround_max_rel_f'], 2)}** "
             f"({'PASS' if v['operator_average_surround_pass'] else 'FAIL'} vs {THRESHOLD})")
    L.append(f"- operator max rel_f, `fixed_rho`: **{fmt(v['operator_fixed_rho_max_rel_f'], 2)}** "
             f"({'PASS' if v['operator_fixed_rho_pass'] else 'FAIL'} vs {THRESHOLD})")
    L.append(f"- system front max rel diff across modes: **{v['system_front_max_rel_diff']*100:.2f}%**")
    L.append(f"- system inventory max rel drift: **{fmt(v['system_inventory_max_drift_rel'], 2)}**")
    L.append(f"- all runs finite: **{v['all_finite']}**")
    L.append(f"- runtime: {v['runtime_s']} s")
    text = "\n".join(L) + "\n"
    OUT.write_text(text)
    print(text)
    return text


if __name__ == "__main__":
    main()