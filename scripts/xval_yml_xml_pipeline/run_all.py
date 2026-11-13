#!/usr/bin/env python3
"""XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE acceptance driver.

Full pipeline per case, mirroring the MPLBM-UT workflow shape:

    input.yml --load/validate--> spec --render--> solver.xml
              --parse back--> spec' (must equal spec)
              --solver--> fixed-text log --parse--> metrics
              --compare--> exact plane-Poiseuille K = H^2/12

Acceptance criterion (feature ledger): relative error < 15 % on permeability
and on the L2 velocity-profile error, for every case in the table.

Usage::

    python scripts/xval_yml_xml_pipeline/run_all.py [--out DIR]

Writes ``report.json``, ``cases/*.yml``, ``cases/*.xml``, ``cases/*.log`` and
``figure.png`` under ``artifacts/xval_yml_xml_pipeline/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm import yml_xml_pipeline as pipe  # noqa: E402

TOL = 0.15  # acceptance: relative error < 15 %
U_TARGET = 0.03  # target peak velocity (Ma ~ 0.052), keeps compressibility < 0.3 %


def make_case(ny: int, omega: float, force_scale: float = 1.0, name: str = "") -> dict:
    """Build a channel case at a prescribed peak-velocity Mach number."""
    nu = (1.0 / omega - 0.5) * pipe.LATTICE_CS2
    h = ny - 2
    g = force_scale * U_TARGET * 8.0 * nu / (h * h)
    return {
        "geometry": {"size": [4, ny], "period": [True, False], "solid_rows": [0, ny - 1]},
        "init": {"rho": 1.0},
        "fluids": {"omega": omega, "force_x": g},
        "output": {"out_folder": name or f"channel_ny{ny}_w{omega}", "it_max": 300000, "convergence": 1e-9},
    }


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def run_case(spec: dict, out_dir: Path) -> dict:
    """yml -> xml -> solver -> text log -> parsed metrics -> reference compare."""
    spec = pipe.validate_spec(spec)
    q = pipe.derived_quantities(spec)

    # 1) render the solver XML document from the YAML spec
    xml_text = pipe.render_xml(spec)
    xml_path = out_dir / "cases" / f"{spec['output']['out_folder']}.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_text, encoding="utf-8")

    # 2) round-trip invariant: the solver must consume exactly what YAML said
    recovered = pipe.parse_xml(xml_path.read_text(encoding="utf-8"))
    roundtrip_ok = recovered == spec

    # 3) drive the solver from the *parsed* document (not from the yaml dict)
    res = pipe.run_channel(recovered)

    # 4) fixed text markers -> parser (the MPLBM-UT post-processing contract)
    log_text = pipe.format_run_log(res)
    log_path = out_dir / "cases" / f"{spec['output']['out_folder']}.log"
    log_path.write_text(log_text + f"# residual = {res['residual']:.6e}\n", encoding="utf-8")
    parsed = pipe.parse_run_log(log_text)

    # 5) reference: exact plane Poiseuille
    h = res["H"]
    k_ref = pipe.analytic_permeability(h)
    s, u_ref = pipe.analytic_profile(h, q["nu"], q["force_x"])
    u = res["u_profile"][1 : res["ny"] - 1]
    k_err = abs(res["permeability"] - k_ref) / k_ref
    l2_err = float(np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref))

    return {
        "case": spec["output"]["out_folder"],
        "ny": res["ny"],
        "H": h,
        "omega": spec["fluids"]["omega"],
        "tau": q["tau"],
        "nu": q["nu"],
        "force_x": q["force_x"],
        "u_peak": res["u_peak"],
        "mach": q["mach"],
        "iters": res["iters"],
        "residual": res["residual"],
        "perm_lbm": res["permeability"],
        "perm_ref": k_ref,
        "perm_rel_err": k_err,
        "l2_profile_rel_err": l2_err,
        "log_parsed_perm": parsed["permeability"],
        "log_marker_consistent": math.isclose(parsed["permeability"], res["permeability"], rel_tol=1e-12),
        "roundtrip_ok": roundtrip_ok,
        "xml_path": _rel(xml_path),
        "log_path": _rel(log_path),
        "pass": bool(roundtrip_ok and k_err < TOL and l2_err < TOL and math.isclose(parsed["permeability"], res["permeability"], rel_tol=1e-11)),
        "u_profile": u.tolist(),
        "u_ref": u_ref.tolist(),
        "y": s.tolist(),
    }


def escaping_probe(out_dir: Path) -> dict:
    """A value with XML metacharacters must survive the render/parse round trip."""
    spec = make_case(24, 1.7, name='weird_&_<value>"x\'')
    xml_text = pipe.render_xml(spec)
    recovered = pipe.parse_xml(xml_text)
    return {
        "case": "escaping_probe",
        "out_folder": spec["output"]["out_folder"],
        "roundtrip_ok": recovered["output"]["out_folder"] == spec["output"]["out_folder"],
        "xml_snippet": [ln for ln in xml_text.splitlines() if "out_folder" in ln][0].strip(),
    }


def make_figure(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for row in rows[:2]:
        ax.plot(row["u_ref"], row["y"], "k-", lw=1.4, label=f"analytic H={row['H']:.0f}")
        ax.plot(row["u_profile"], row["y"], "o", ms=3.2, label=f"LBM H={row['H']:.0f} (L2 {row['l2_profile_rel_err']:.2%})")
    ax.set_xlabel("$u_x$ [lattice units]")
    ax.set_ylabel("y [lattice]")
    ax.set_title("YAML->XML driven channel: profile vs exact Poiseuille")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [f"H={r['H']:.0f}\n$\\omega$={r['omega']}\nF x{r['force_x']/rows[0]['force_x']:.1f}" for r in rows]
    errs = [100 * r["perm_rel_err"] for r in rows]
    bars = ax.bar(range(len(rows)), errs, color="#3d7ab8")
    ax.axhline(15.0, color="r", ls="--", lw=1, label="acceptance 15%")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("|K_LBM - K_ref| / K_ref  [%]")
    ax.set_title("permeability error vs $K=H^2/12$")
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, e, f"{e:.2f}%", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "artifacts" / "xval_yml_xml_pipeline"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)

    cases = [
        make_case(24, 1.7, 1.0, "ch_ny24_w1p7"),
        make_case(36, 1.7, 1.0, "ch_ny36_w1p7"),
        make_case(24, 1.7, 0.5, "ch_ny24_w1p7_halfF"),
        make_case(24, 0.9, 1.0, "ch_ny24_w0p9"),
    ]
    rows = []
    for spec in cases:
        row = run_case(spec, out_dir)
        rows.append(row)
        print(
            f"{row['case']:<22} H={row['H']:.0f} it={row['iters']:>6d} "
            f"K={row['perm_lbm']:.5f} ref={row['perm_ref']:.5f} "
            f"errK={row['perm_rel_err']:.3%} L2={row['l2_profile_rel_err']:.3%} "
            f"{'PASS' if row['pass'] else 'FAIL'}"
        )

    esc = escaping_probe(out_dir)
    fig_path = out_dir / "figure.png"
    make_figure(rows, fig_path)

    worst_k = max(r["perm_rel_err"] for r in rows)
    worst_l2 = max(r["l2_profile_rel_err"] for r in rows)
    report = {
        "anchor": "XVAL-LBM-YML-DRIVEN-XML-TEMPLATE-PIPELINE",
        "mission": 18791,
        "reference": {
            "quantity": "permeability K (lattice units), plane Poiseuille",
            "formula": "K = H^2 / 12",
            "profile": "u(s) = g/(2 nu) s (H - s), s = y - 1/2",
            "source": "Poiseuille (1840) exact solution; lattice form: Succi, The Lattice Boltzmann Equation (2001) ch. 5; "
            "Krueger et al., Lattice Boltzmann Method: Principles and Practice (2017) §5.3 (halfway bounce-back wall at mid-link)",
        },
        "acceptance": {"tol": TOL, "criterion": "perm + L2 profile relative error < 15 % on every case"},
        "cases": rows,
        "escaping_probe": esc,
        "summary": {
            "worst_perm_rel_err": worst_k,
            "worst_l2_rel_err": worst_l2,
            "all_pass": all(r["pass"] for r in rows) and esc["roundtrip_ok"],
            "n_cases": len(rows),
        },
        "artifacts": {"figure": _rel(fig_path)},
        "convergence": [
            {"H": rows[0]["H"], "perm_rel_err": rows[0]["perm_rel_err"]},
            {"H": rows[1]["H"], "perm_rel_err": rows[1]["perm_rel_err"]},
        ],
        "sensitivity": {
            "force_doubling_perm_delta": abs(rows[0]["perm_lbm"] - rows[2]["perm_lbm"]) / rows[0]["perm_lbm"],
            "omega_0p9_vs_1p7_perm_delta": abs(rows[0]["perm_lbm"] - rows[3]["perm_lbm"]) / rows[0]["perm_lbm"],
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nworst perm err = {worst_k:.3%}   worst L2 = {worst_l2:.3%}   all_pass={report['summary']['all_pass']}")
    print(f"report -> {report_path}")
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())