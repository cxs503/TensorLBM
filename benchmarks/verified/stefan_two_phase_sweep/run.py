#!/usr/bin/env python
"""P272W2-N05: Stefan-number sweep of the enthalpy thermal LBM (TensorLBM).

Verification surface
--------------------
The enthalpy-form lattice-Boltzmann phase-change solver
(``tensorlbm.phase_change.stefan_verification.EnthalpyThermalLBM``) is driven on
the **two-phase** one-dimensional Stefan problem -- cold wall ``T_0 < T_m``
freezing a *superheated* melt ``T_i > T_m`` -- and compared against the coupled
similarity solution in ``tensorlbm.phase_change.stefan_two_phase``:

    s(t)  = 2*lam*sqrt(alpha*t),
    lam   = Ste_s*exp(-lam^2)/(sqrt(pi)*erf(lam))
            - Ste_l*exp(-lam^2)/(sqrt(pi)*erfc(lam))          (equal properties)

Acceptance (predeclared, see ``stefan_sweep.FRONT_TOL`` etc.)
-------------------------------------------------------------
  C1  front position vs similarity solution, 3 Stefan numbers   < 5 %
  C2  temperature profile vs similarity solution (max, whole domain) < 3 %
  C3  energy closure: heat input vs sensible + latent on the same
      control volume, plus an independent wall-flux operator     < 2 %
  C4  0 <= f_l <= 1 over the whole run (no overshoot)            hard bound
  C5  grid convergence of the front error (>= 3 resolutions)     order >= 1
  C6  machine-readable JSON with per-array sha256 provenance      present

Every number in ``artifacts/stefan_two_phase_sweep/results.json`` is written by
this script -- nothing is hand-copied; figures land in
``artifacts/stefan_two_phase_sweep/figures``.

Usage
-----
    PYTHONPATH=src python benchmarks/verified/stefan_two_phase_sweep/run.py \
        [--quick] [--out artifacts/stefan_two_phase_sweep]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

from tensorlbm.phase_change.stefan_exact import planar_lambda
from tensorlbm.phase_change.stefan_sweep import (
    ENERGY_TOL,
    FRONT_TOL,
    PROFILE_TOL,
    run_two_phase_grid_convergence,
    run_two_phase_stefan,
)
from tensorlbm.phase_change.stefan_two_phase import (
    TwoPhaseSimilarity,
    two_phase_lambda,
    two_phase_residual,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEFAN_NUMBERS = (0.1, 0.5, 1.0)
STEFAN_LIQUID = 0.2
FRONT_CELLS = 33.0
CONVERGENCE_CELLS = (17.0, 33.0, 66.0)


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------
def array_sha256(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a, dtype=float))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# exact-solution self-audit (the reference must be exact before it is a reference)
# ---------------------------------------------------------------------------
def exact_solution_audit() -> dict:
    out: dict = {"one_phase_limit": [], "energy_identity": [], "pde_residual": []}

    for ste in STEFAN_NUMBERS + (2.0,):
        lam_two = two_phase_lambda(ste, 0.0)
        lam_one = planar_lambda(ste)
        out["one_phase_limit"].append(
            {
                "stefan_number": ste,
                "lambda_two_phase_with_Ste_l_zero": lam_two,
                "lambda_one_phase_neumann": lam_one,
                "abs_diff": abs(lam_two - lam_one),
                "residual_at_root": abs(two_phase_residual(lam_two, ste, 0.0)),
            }
        )

    for ste_s, ste_l in ((0.1, 0.2), (0.5, 0.2), (1.0, 0.2), (0.5, 0.5), (1.0, 1.0)):
        model = TwoPhaseSimilarity(
            stefan_solid=ste_s, stefan_liquid=ste_l, alpha_solid=1.0 / 6.0, alpha_liquid=1.0 / 6.0
        )
        t = 5000.0
        budget = model.energy_budget(t)
        s = model.front(t)
        out["energy_identity"].append(
            {
                "stefan_solid": ste_s,
                "stefan_liquid": ste_l,
                "lambda": model.lam,
                "front": s,
                "analytic_total_enthalpy_change": budget["total"],
                "analytic_wall_input": budget["wall_input"],
                "abs_diff": abs(budget["wall_input_minus_total"]),
                "rel_diff": abs(budget["wall_input_minus_total"]) / abs(budget["total"]),
                "stefan_condition_residual": model.stefan_condition_residual(t),
                "T_wall": float(model.temperature(np.array([0.0]), t)[0]),
                "T_wall_expected": model.wall_temperature,
                "T_front": float(model.temperature(np.array([s]), t)[0]),
                "T_far": float(model.temperature(np.array([1.0e4]), t)[0]),
                "T_far_expected": model.initial_temperature,
            }
        )

    for ste_s, ste_l, alpha_l in ((0.5, 0.2, 1.0 / 6.0), (0.5, 0.2, 1.0 / 3.0)):
        model = TwoPhaseSimilarity(
            stefan_solid=ste_s, stefan_liquid=ste_l, alpha_solid=1.0 / 6.0, alpha_liquid=alpha_l
        )
        out["pde_residual"].append(
            {
                "stefan_solid": ste_s,
                "stefan_liquid": ste_l,
                "alpha_liquid_over_solid": alpha_l / (1.0 / 6.0),
                "lambda": model.lam,
                "lambda_liquid": model.lam_liquid,
                **model.pde_residual(5000.0),
                "energy_rel_diff": abs(model.energy_budget(5000.0)["wall_input_minus_total"])
                / abs(model.energy_budget(5000.0)["total"]),
            }
        )

    out["max_one_phase_limit_abs_diff"] = max(r["abs_diff"] for r in out["one_phase_limit"])
    out["max_energy_identity_rel_diff"] = max(r["rel_diff"] for r in out["energy_identity"])
    out["max_stefan_condition_residual"] = max(
        abs(r["stefan_condition_residual"]) for r in out["energy_identity"]
    )
    out["max_pde_residual_rel"] = max(
        max(r["solid_ode_residual_rel"], r["liquid_ode_residual_rel"]) for r in out["pde_residual"]
    )
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_fronts(sweep: list[dict], out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sweep)))
    for res, c in zip(sweep, colors):
        p = res["parameters"]
        model = TwoPhaseSimilarity(
            stefan_solid=p["stefan_solid"], stefan_liquid=p["stefan_liquid"],
            alpha_solid=p["alpha"], alpha_liquid=p["alpha"],
        )
        t = np.array([s["time"] for s in res["series"]])
        s_num = np.array([s["front_phase_integral"] for s in res["series"]])
        s_exa = np.array([s["front_exact"] for s in res["series"]])
        tt = np.linspace(1.0, res["parameters"]["time_end"], 200)
        label = f"Ste_s={p['stefan_solid']:.1f}, Ste_l={p['stefan_liquid']:.1f}"
        ax.plot(tt, model.front(tt), "-", color=c, lw=1.4,
                label=fr"{label} exact $2\lambda\sqrt{{\alpha t}}$")
        ax.plot(t, s_num, "o", ms=4.5, color=c, mfc="none",
                label=f"{label} LBM (max err {res['criteria']['C1_front_error']*100:.2f}%)")
    ax.set_xlabel("time (lattice steps)")
    ax.set_ylabel("front position s (cells)")
    ax.set_title("Stefan sweep: front position vs two-phase similarity solution")
    ax.legend(fontsize=6.6, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig1_front_vs_exact.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fig_profiles(sweep: list[dict], out_dir: Path) -> str:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sweep)))
    for res, c in zip(sweep, colors):
        prof = res["profile"]
        x = np.array(prof["x"])
        T = np.array(prof["T_numerical"])
        Te = np.array(prof["T_exact"])
        p = res["parameters"]
        lab = f"Ste_s={p['stefan_solid']:.1f}"
        ax.plot(x, Te, "-", color=c, lw=1.3, label=f"{lab} exact")
        ax.plot(x[::4], T[::4], "o", ms=2.6, color=c, mfc="none", label=f"{lab} LBM")
        ax2.plot(x, (T - Te) / prof["metrics"]["dT_reference"], "-", color=c, lw=1.2, label=lab)
    ax.set_xlabel("x (cells)")
    ax.set_ylabel(r"T ($T_m=0$, lattice units)")
    ax.set_title(f"Temperature profile at t={sweep[0]['parameters']['steps']} steps")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax2.set_xlabel("x (cells)")
    ax2.set_ylabel(r"$(T_{{num}}-T_{{exact}})/\Delta T_{{ref}}$")
    ax2.set_title("Profile error (whole domain, incl. smeared interface)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "fig2_temperature_profiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fig_convergence(conv: dict, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    markers = {"0.5": "o", "1.0": "s"}
    for key, data in conv.items():
        xs = np.array(data["front_cells"])
        ys = np.array([r["final"]["front_window_rel_error"] for r in data["cases"]])
        ax.loglog(xs, ys, markers.get(key, "o") + "-", label=(
            f"Ste_s={key}: front error, observed order p={data['front_order']['order']:.2f}"))
    xs = np.array(conv[list(conv)[0]]["front_cells"])
    ax.loglog(xs, ys[0] * (xs / xs[0]) ** -1.0, "k--", lw=1.0, label="slope -1 reference")
    ax.set_xlabel("front resolution (cells spanned by s)")
    ax.set_ylabel(r"relative front error $|s_{num}-s_{exact}|/s_{exact}$")
    ax.set_title("Grid convergence of the enthalpy-method front")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    path = out_dir / "fig3_grid_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fig_energy(sweep: list[dict], out_dir: Path) -> str:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    labels = [f"Ste_s={r['parameters']['stefan_solid']:.1f}" for r in sweep]
    lat = [abs(r["final"]["latent_change"]) for r in sweep]
    sen = [abs(r["final"]["sensible_change"]) for r in sweep]
    x = np.arange(len(sweep))
    ax.bar(x - 0.18, lat, 0.36, label="latent |L*s| (LBM)", color="#3b6ea5")
    ax.bar(x + 0.18, sen, 0.36, label="sensible (LBM)", color="#c46a3a")
    ax.plot(x - 0.18, [abs(r["final"]["analytic_latent"]) for r in sweep], "k_", ms=18,
            label="analytic latent")
    ax.plot(x + 0.18, [abs(r["final"]["analytic_sensible"]) for r in sweep], "k_", ms=18,
            label="analytic sensible")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("enthalpy change (per unit depth)")
    ax.set_title("Energy decomposition: latent vs sensible")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    for i, r in enumerate(sweep):
        ax2.scatter([i], [r["criteria"]["C3_energy_error"]], marker="o", color="#2b7a4b",
                    label="C3 energy closure" if i == 0 else None)
    ax2.axhline(ENERGY_TOL, color="r", ls="--", lw=1.0, label=f"tolerance {ENERGY_TOL:.0%}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("max relative energy-closure error")
    ax2.set_title("C3: heat input vs sensible + latent")
    ax2.set_ylim(0, max(ENERGY_TOL * 1.5, 5e-3))
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "fig4_energy_closure.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/stefan_two_phase_sweep")
    ap.add_argument("--quick", action="store_true", help="reduced resolutions for CI")
    ap.add_argument("--json", default=None, help="also write the payload here")
    args = ap.parse_args()

    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    front_cells = 17.0 if args.quick else FRONT_CELLS
    conv_cells = (9.0, 17.0, 33.0) if args.quick else CONVERGENCE_CELLS
    t_start = time.time()

    print("[1/5] exact-solution self-audit ...", flush=True)
    audit = exact_solution_audit()

    print(f"[2/5] two-phase Stefan sweep (front_cells={front_cells}) ...", flush=True)
    sweep = [
        run_two_phase_stefan(ste, STEFAN_LIQUID, front_cells=front_cells)
        for ste in STEFAN_NUMBERS
    ]

    print("[3/5] one-phase limit check (Ste_l = 0) ...", flush=True)
    one_phase = [
        run_two_phase_stefan(ste, 0.0, front_cells=front_cells) for ste in STEFAN_NUMBERS
    ]

    print("[4/5] grid convergence ...", flush=True)
    conv = {}
    for ste in ((1.0,) if args.quick else (0.5, 1.0)):
        conv[str(ste)] = run_two_phase_grid_convergence(
            ste, STEFAN_LIQUID, front_cells=conv_cells
        )

    print("[5/5] figures + provenance ...", flush=True)
    figs_written = [
        fig_fronts(sweep, figs),
        fig_profiles(sweep, figs),
        fig_convergence(conv, figs),
        fig_energy(sweep, figs),
    ]

    # -- criteria ----------------------------------------------------------
    c1 = {str(r["parameters"]["stefan_solid"]): r["criteria"]["C1_front_error"] for r in sweep}
    c2 = {
        str(r["parameters"]["stefan_solid"]): r["criteria"]["C2_profile_max_rel_error"]
        for r in sweep
    }
    c3 = {str(r["parameters"]["stefan_solid"]): r["criteria"]["C3_energy_error"] for r in sweep}
    c4_bounds = [r["liquid_fraction_bounds"] for r in sweep]
    criteria = {
        "C1_front_error_max": max(c1.values()),
        "C1_per_stefan": c1,
        "C1_tol": FRONT_TOL,
        "C1_pass": all(v < FRONT_TOL for v in c1.values()) and len(c1) == 3,
        "C2_profile_max_error": max(c2.values()),
        "C2_per_stefan": c2,
        "C2_tol": PROFILE_TOL,
        "C2_pass": all(v < PROFILE_TOL for v in c2.values()),
        "C3_energy_error_max": max(c3.values()),
        "C3_per_stefan": c3,
        "C3_tol": ENERGY_TOL,
        "C3_pass": all(v < ENERGY_TOL for v in c3.values()),
        "C4_liquid_fraction_min": min(b["min"] for b in c4_bounds),
        "C4_liquid_fraction_max": max(b["max"] for b in c4_bounds),
        "C4_pass": all(
            b["min"] >= -1e-12 and b["max"] <= 1.0 + 1e-12 for b in c4_bounds
        ),
        "C5_front_order": {k: v["front_order"]["order"] for k, v in conv.items()},
        "C5_monotone": {k: v["monotone"] for k, v in conv.items()},
        "C5_pass": all(v["C5_pass"] for v in conv.values()),
        "C6_present": True,
    }
    # one-phase limit must also reproduce the reference front law
    criteria["C1b_one_phase_limit_max_error"] = max(
        r["criteria"]["C1_front_error"] for r in one_phase
    )
    criteria["C1b_one_phase_limit_pass"] = all(
        r["criteria"]["C1_pass"] for r in one_phase
    )
    criteria["exact_audit_pass"] = bool(
        audit["max_one_phase_limit_abs_diff"] < 1e-12
        and audit["max_energy_identity_rel_diff"] < 1e-10
        and audit["max_stefan_condition_residual"] < 1e-15
        and audit["max_pde_residual_rel"] < 1e-4
    )
    criteria["all_pass"] = bool(
        criteria["C1_pass"]
        and criteria["C2_pass"]
        and criteria["C3_pass"]
        and criteria["C4_pass"]
        and criteria["C5_pass"]
        and criteria["C1b_one_phase_limit_pass"]
        and criteria["exact_audit_pass"]
    )

    payload = {
        "node": "P272W2-N05",
        "repo": "TensorLBM",
        "title": "Stefan sweep of the enthalpy thermal LBM vs the two-phase similarity solution",
        "generated_by": "benchmarks/verified/stefan_two_phase_sweep/run.py",
        "code_sha256": file_sha256(Path(__file__)),
        "wall_seconds": time.time() - t_start,
        "units": "lattice (dx=dt=1, rho=1, cp=1, L=1, T_freeze=0)",
        "setup": {
            "stefan_solid_sweep": list(STEFAN_NUMBERS),
            "stefan_liquid": STEFAN_LIQUID,
            "front_cells": front_cells,
            "convergence_front_cells": list(conv_cells),
            "tau": 1.0,
            "alpha": 1.0 / 6.0,
            "lattice": "D2Q5",
            "far_eta": 2.5,
        },
        "exact_solution_audit": audit,
        "sweep": sweep,
        "one_phase_limit": one_phase,
        "grid_convergence": conv,
        "criteria": criteria,
        "figures": [f"figures/{f}" for f in figs_written],
    }

    # per-array digests so any consumer can re-derive the numbers from the arrays
    digests = {}
    for res in payload["sweep"] + payload["one_phase_limit"]:
        key = f"Ste_{res['parameters']['stefan_solid']}_{res['parameters']['stefan_liquid']}"
        digests[key] = {
            "T_numerical": array_sha256(res["profile"]["T_numerical"]),
            "T_exact": array_sha256(res["profile"]["T_exact"]),
            "f_l": array_sha256(res["profile"]["f_l"]),
            "x": array_sha256(res["profile"]["x"]),
        }
    payload["array_sha256"] = digests

    payload_path = out / "results.json"
    text = json.dumps(payload, indent=2, default=float, ensure_ascii=False)
    payload_path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (out / "results.json.sha256").write_text(digest + "  results.json\n")
    if args.json:
        Path(args.json).write_text(text)

    lineage = {
        "node": "P272W2-N05",
        "repo": "TensorLBM",
        "campaign_theme": "phase-change heat transfer / Stefan problem / Stefan-number sweep",
        "module": [
            "tensorlbm.phase_change.stefan_two_phase",
            "tensorlbm.phase_change.stefan_sweep",
        ],
        "builds_on": [
            "tensorlbm.phase_change.stefan_verification (enthalpy thermal LBE, P155W1-N01)",
            "tensorlbm.phase_change.stefan_exact (one-phase Neumann branches)",
        ],
        "new_verified_capability": (
            "two-phase (superheated-melt) Stefan similarity solution with coupled dual-lambda "
            "coefficient + enthalpy-LBM Stefan-number sweep (Ste_s = 0.1/0.5/1.0) with "
            "front/profile/energy/front-order acceptance"
        ),
        "reference_solutions": [
            "coupled two-phase planar Stefan similarity solution (solid erf / liquid erfc branches)",
            "one-phase Neumann branch recovered exactly as the Ste_l -> 0 limit",
            "analytic enthalpy decomposition (latent + solid/liquid sensible) on a finite window",
        ],
        "explicitly_not_used": [
            "quasi-steady (Planck) closed form as a <=3% reference (see P155W1-N01: up to 20% off)",
            "unequal phase diffusivities in the LBM run (single sensible-enthalpy field); the closed "
            "form supports it and is audited only analytically",
        ],
        "artifacts": {
            "results": str(payload_path),
            "results_sha256": digest,
            "figures": str(figs) + "/",
            "report": "docs/P272W2_N05_STEFAN_TWO_PHASE_SWEEP.md",
            "tests": "tests/test_stefan_two_phase_sweep.py",
        },
    }
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2, ensure_ascii=False))

    print(
        json.dumps(
            {
                "all_pass": criteria["all_pass"],
                "wall_seconds": round(payload["wall_seconds"], 1),
                "C1": criteria["C1_front_error_max"],
                "C2": criteria["C2_profile_max_error"],
                "C3": criteria["C3_energy_error_max"],
                "C5_orders": criteria["C5_front_order"],
                "results": str(payload_path),
                "sha256": digest,
            },
            indent=1,
            default=float,
        )
    )
    return 0 if criteria["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())