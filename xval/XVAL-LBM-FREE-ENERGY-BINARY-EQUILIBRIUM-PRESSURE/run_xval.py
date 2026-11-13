#!/usr/bin/env python3
"""XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE — run entry.

Measures the equilibrium pressure of the symmetric free-energy (binary-fluid)
LBM model and grades it against closed forms, using the criteria pre-declared in
``predeclared.json`` (read at run time; never tuned after the fact).

    python xval/XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE/run_xval.py \
        --out xval/XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE/artifacts

Artifacts (written into ``--out``):
    xval_binary_equilibrium_pressure.json   full numeric report (all rows)
    fig_normal_pressure_profile.png         P_nn(x)/p_coex at 3 interface resolutions
    fig_convergence_and_lbm.png             sigma error vs xi/h + LBM residual history
Exit code 0 on PASS, 1 on FAIL (per-criterion table is printed either way).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.free_energy_stress import FreeEnergyParams, pressure_tensor  # noqa: E402
from tensorlbm.phasefield.binary_equilibrium_pressure import (  # noqa: E402
    EquilibriumPressureConfig,
    equilibrium_pressure_report,
    periodic_planar_profile,
    run_planar_equilibrium_lbm,
)


def load_criteria() -> dict:
    return json.loads((HERE / "predeclared.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "artifacts"))
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="short sweep for CI smoke runs")
    args = ap.parse_args()

    torch.set_num_threads(1)
    spec = load_criteria()
    case = spec["cases"]["planar_periodic_two_interface"]
    nx = args.nx or case["nx"]
    ny = args.ny or case["ny"]
    steps = args.steps or case["steps"]
    resolutions = tuple(float(v) for v in case["resolutions_interface_cells"])
    if args.quick:
        nx, ny, steps, resolutions = 64, 8, 300, (2.0, 3.0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = equilibrium_pressure_report(
        resolutions=resolutions,
        nx=nx,
        ny=ny,
        steps=steps,
        schemes=tuple(case["gradient_schemes"]),
    )

    crit = spec["criteria"]
    tol_coex = crit["C1_coexistence_pressure"]["tolerance_relative"]
    tol_uniform = crit["C2_normal_pressure_uniformity"]["tolerance_relative_to_p_coex"]
    tol_sigma = crit["C3_surface_tension_from_pressure_tensor"]["tolerance_relative"]
    tol_lbm_p = crit["C4_lbm_mechanical_equilibrium"]["tolerance_relative_to_p_coex"]
    tol_lbm_u = crit["C4_lbm_mechanical_equilibrium"]["tolerance_max_velocity_in_cs"]

    checks: list[dict] = []
    for row in report["rows"]:
        iso = row["schemes"]["iso9"]
        checks.append(
            {
                "criterion": "C1_coexistence_pressure",
                "resolution_interface_cells": row["interface_cells"],
                "reference": iso["p_coex_closed_form"],
                "measured": iso["p0_bulk"],
                "metric": max(abs(v - iso["p_coex_closed_form"]) for v in iso["p0_bulk"])
                / iso["p_coex_closed_form"],
                "tolerance": tol_coex,
            }
        )
        checks.append(
            {
                "criterion": "C2_normal_pressure_uniformity",
                "resolution_interface_cells": row["interface_cells"],
                "reference": 0.0,
                "measured": iso["p_nn_max_deviation_relative"],
                "tolerance": tol_uniform,
            }
        )
        for scheme, diag in row["schemes"].items():
            checks.append(
                {
                    "criterion": "C3_surface_tension_from_pressure_tensor",
                    "resolution_interface_cells": row["interface_cells"],
                    "scheme": scheme,
                    "reference": diag["sigma_analytic"],
                    "measured": diag["sigma_integral"],
                    "metric": abs(diag["sigma_relative_error"]),
                    "tolerance": tol_sigma,
                }
            )
        lbm = row.get("lbm_run")
        if lbm:
            checks.append(
                {
                    "criterion": "C4_lbm_mechanical_equilibrium_pressure",
                    "resolution_interface_cells": row["interface_cells"],
                    "measured": lbm["final_p_nn_total_deviation_relative"],
                    "tolerance": tol_lbm_p,
                }
            )
            checks.append(
                {
                    "criterion": "C4_lbm_residual_velocity",
                    "resolution_interface_cells": row["interface_cells"],
                    "measured": lbm["final_max_velocity"],
                    "tolerance": tol_lbm_u * math.sqrt(1.0 / 3.0),
                }
            )
    for c in checks:
        value = c["metric"] if "metric" in c else c["measured"]
        c["passed"] = bool(value <= c["tolerance"])
    passed = all(c["passed"] for c in checks)

    # ---- figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for cells in resolutions:
        cfg = EquilibriumPressureConfig(
            A=case["A"], B=case["B"], interface_cells=float(cells), nx=nx, ny=ny, steps=1
        )
        field = periodic_planar_profile(ny, nx, cfg.params)
        pxx, _pyy, _ = pressure_tensor(field.phi, cfg.params, "iso9")
        p_nn = pxx[0].numpy() / field.p_coex_closed_form
        ax.plot(range(nx), p_nn, label=f"$\\xi/h$ = {cells:g}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="continuum: $P_{nn}=p_{coex}$")
    ax.set_xlabel("x (lattice)")
    ax.set_ylabel("$P_{nn} / p_{coex}$")
    ax.set_title("Normal pressure across two flat interfaces")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = [row["interface_cells"] for row in report["rows"]]
    for scheme, marker in (("iso9", "o"), ("5pt", "s")):
        errs = [100.0 * abs(row["schemes"][scheme]["sigma_relative_error"]) for row in report["rows"]]
        ax.plot(xs, errs, marker + "-", label=f"$\\sigma$ error ({scheme})")
    ax.axhline(15.0, color="r", ls="--", lw=1, label="15% acceptance")
    ax2 = ax.twinx()
    for row in report["rows"]:
        hist = row["lbm_run"]["max_velocity_history"]
        ax2.semilogy(
            [row["interface_cells"]] * len(hist), hist, "x", color="0.4"
        )
    ax2.set_ylabel("residual $\\max|u|$ (LBM, grey x)")
    ax.set_xlabel("interface resolution $\\xi/h$ (cells)")
    ax.set_ylabel("$|\\sigma_{measured}/\\sigma_{analytic}-1|$ (%)")
    ax.set_title("Capillary integral convergence + LBM residual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_convergence_and_lbm.png", dpi=130)
    plt.close(fig)

    results = {
        "anchor": report["anchor"],
        "settings": report["settings"],
        "criteria_source": str(HERE / "predeclared.json"),
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
        "report": report,
    }
    (out / "xval_binary_equilibrium_pressure.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )

    print(f"{'criterion':42s} {'xi/h':>5s} {'measured':>12s} {'tol':>8s}  pass")
    for c in checks:
        m = c["metric"] if "metric" in c else c["measured"]
        m_str = f"{m:.3e}" if isinstance(m, float) else str(m)
        print(f"{c['criterion']:42s} {c['resolution_interface_cells']:5g} {m_str:>12s} {c['tolerance']:8.3g}  {c['passed']}")
    print(f"\nVERDICT: {results['verdict']}  (artifacts: {out})")
    print("CH phase-flux gap diagnostic:", json.dumps(report.get("ch_phase_flux_gap", {}), sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())