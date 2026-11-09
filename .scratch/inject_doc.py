"""Inject the evidence JSON into the D3Q27 slip-wall note (no hand-copied numbers)."""
import json
from pathlib import Path

J = Path("artifacts/P106W1-N04/d3q27_slip_wall.json")
D = Path("docs/P106W1_N04_SLIP_WALL_3D_D3Q27.md")

d = json.loads(J.read_text())
cases, laws, chk = d["cases"], d["laws"], d["checks"]


def sci(x, nd=2):
    return f"{x:.{nd}e}"


def pct(x, nd=3):
    return f"{100.0 * x:.{nd}f}%"


# ------------------------------------------------------------------ tables
rows = ["| lattice | ny | H | sigma | steps | u_max | b_model | b_bulk | b_quadratic |",
        "|---|---|---|---|---|---|---|---|---|"]
for c in cases:
    rows.append(
        f"| {c['lattice']} | {c['ny']} | {c['H']:.1f} | {c['sigma']:.2f} | {c['steps']} | "
        f"{c['u_max']:.6f} | {c['b_model']:.6f} | {c['b_bulk']:.6f} | "
        f"{c['b_quadratic']:.6f} |"
    )
table_cases = "\n".join(rows)

rows = ["| fit | a | c | c*H | R^2 | monotone |", "|---|---|---|---|---|---|"]
for k, v in laws.items():
    rows.append(f"| {k} | {v['a']:.8f} | {v['c']:.8f} | {v['c_times_H']:.7f} | "
                f"{v['r2']:.12f} | {v['monotone']} |")
table_laws = "\n".join(rows)

wave = chk["wave_start"]
q = chk["quadratic_fit_vs_model"]
rows = [
    "| check | value |",
    "|---|---|",
    f"| collapse_vs_1d_reduction.rel_profile_deviation | {sci(chk['collapse_vs_1d_reduction']['rel_profile_deviation'])} |",
    f"| collapse_vs_1d_reduction.plane_uniformity | {sci(chk['collapse_vs_1d_reduction']['plane_uniformity'])} |",
    f"| cross_lattice_profile_dev | {sci(chk['cross_lattice_profile_dev'])} |",
    f"| sigma0_vs_analytic_noslip_dev | {pct(chk['sigma0_vs_analytic_noslip_dev'])} |",
    f"| sigma0_slip_length_over_H | {sci(chk['sigma0_slip_length_over_H'])} |",
    f"| wave_start.profile_dev_vs_uniform_run | {sci(wave['profile_dev_vs_uniform_run'])} |",
    f"| wave_start.plane_spread | {sci(wave['plane_spread'])} |",
    f"| wave_start.steps | {wave['steps']} |",
    f"| closed_loop_inversion_rel_error | {sci(chk['closed_loop_inversion_rel_error'])} |",
    f"| quadratic_fit_vs_model.max_abs_error_lu | {sci(q['max_abs_error_lu'])} |",
    f"| quadratic_fit_vs_model.max_rel_error | {sci(q['max_rel_error'])} |",
]
table_checks = "\n".join(rows)

law_a_diff = max(abs(laws["D3Q27_ny17"]["a"] - laws["D3Q19_ny17"]["a"]) / laws["D3Q19_ny17"]["a"],
                  abs(laws["D3Q27_ny33"]["a"] - laws["D3Q19_ny33"]["a"]) / laws["D3Q19_ny33"]["a"])
law_c_diff = max(abs(laws["D3Q27_ny17"]["c_times_H"] - laws["D3Q19_ny17"]["c_times_H"]),
                 abs(laws["D3Q27_ny33"]["c_times_H"] - laws["D3Q19_ny33"]["c_times_H"]))

subs = {
    "{{tau}}": f"{d['tau']}",
    "{{nu}}": f"{d['nu']:.10f}",
    "{{g}}": f"{d['g']:g}",
    "{{TABLE_CASES}}": table_cases,
    "{{TABLE_LAWS}}": table_laws,
    "{{TABLE_CHECKS}}": table_checks,
    "{{collapse}}": sci(chk["collapse_vs_1d_reduction"]["rel_profile_deviation"]),
    "{{cross}}": sci(chk["cross_lattice_profile_dev"]),
    "{{sigma0_dev}}": pct(chk["sigma0_vs_analytic_noslip_dev"]),
    "{{sigma0_dev_pct}}": pct(chk["sigma0_vs_analytic_noslip_dev"]),
    "{{law_a_diff}}": sci(law_a_diff),
    "{{law_c_diff}}": sci(law_c_diff),
    "{{quad_abs}}": sci(q["max_abs_error_lu"]),
    "{{quad_rel}}": sci(q["max_rel_error"]),
    "{{loop}}": sci(chk["closed_loop_inversion_rel_error"]),
    "{{wave}}": sci(wave["profile_dev_vs_uniform_run"]),
}

md = D.read_text()
for k, v in subs.items():
    assert k in md, k
    md = md.replace(k, v)
D.write_text(md)
print("injected", len(subs), "values")