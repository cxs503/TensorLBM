#!/usr/bin/env python3
"""P33-N3 scaling study: partitioned-FSI added-mass instability, accelerators,
and VIV validation.

Usage
-----
    python experiments/fsi_partitioned_scaling.py oned      # 1-D analytic + numeric
    python experiments/fsi_partitioned_scaling.py viv --case 0   # one VIV case
    python experiments/fsi_partitioned_scaling.py figures   # PNG figures + report JSON

Artifacts are written under ``artifacts/fsi_partitioned/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.fsi_partitioned import (  # noqa: E402
    AddedMassModel1D,
    AitkenRatio,
    AitkenRelaxation,
    IQNILS,
    analytic_convergence_factor,
    beta_over_one_plus_beta,
    detect_beta_critical,
    make_accelerator,
    measure_contraction_1d,
    optimal_relaxation,
    solve_partitioned_1d,
    theta_split_rho,
)

OUT = ROOT / "artifacts" / "fsi_partitioned"
DT = 0.01
BETAS = (0.25, 1.0, 2.5, 10.0)


# ---------------------------------------------------------------------------
# 1-D stage
# ---------------------------------------------------------------------------


def run_oned() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"dt": DT, "betas": list(BETAS)}

    # --- C1: analytic vs measured contraction -------------------------
    rows = []
    for beta in BETAS:
        model = AddedMassModel1D(m_struct=1.0, m_added=beta)
        rho_wet, res_wet, _ = measure_contraction_1d(
            model, dt=DT, variant="wet_mass", n_iter=60, skip=3, window=10
        )
        rho_exp, res_exp, _ = measure_contraction_1d(
            model, dt=DT, variant="explicit", n_iter=60, skip=3, window=10
        )
        ana_wet = beta_over_one_plus_beta(beta)
        rows.append(
            {
                "beta": beta,
                "rho_analytic_explicit": beta,
                "rho_measured_explicit": rho_exp,
                "rel_err_explicit": abs(rho_exp - beta) / beta,
                "rho_analytic_wet_mass": ana_wet,
                "rho_measured_wet_mass": rho_wet,
                "rel_err_wet_mass": abs(rho_wet - ana_wet) / ana_wet,
                "final_residual_wet_mass": res_wet[-1],
                "initial_residual_wet_mass": res_wet[0],
            }
        )
    result["c1_contraction"] = rows
    result["c1_max_rel_err"] = max(
        max(r["rel_err_wet_mass"], r["rel_err_explicit"]) for r in rows
    )

    # --- C1b: relaxed convergence-factor formula rho(omega) -----------
    relax_rows = []
    model = AddedMassModel1D(m_struct=1.0, m_added=1.0)
    for omega in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0):
        rho_num, _, _ = measure_contraction_1d(
            model, dt=DT, variant="explicit",
            accelerator=make_accelerator("constant", omega=omega), n_iter=60, skip=3, window=10,
        )
        relax_rows.append(
            {
                "omega": omega,
                "rho_analytic": analytic_convergence_factor(1.0, omega),
                "rho_measured": rho_num,
            }
        )
    result["c1b_relaxation_formula"] = relax_rows

    # --- C2: critical added-mass ratio --------------------------------
    crit = detect_beta_critical(dt=DT, m_struct=1.0, variant="explicit")
    criteria = []
    for beta in (0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 2.5, 5.0):
        model = AddedMassModel1D(m_struct=1.0, m_added=beta)
        rho, _, _ = measure_contraction_1d(model, dt=DT, variant="explicit", n_iter=40, skip=3, window=10)
        criteria.append({"beta": beta, "rho_measured": rho, "divergent": rho >= 1.0})
    crit["rel_err_vs_analytic"] = abs(crit["beta_crit"] - 1.0) / 1.0
    crit["sweep"] = criteria
    result["c2_beta_critical"] = crit

    # --- C3: stabilisation at beta = 10 -------------------------------
    c3 = []
    model = AddedMassModel1D(m_struct=1.0, m_added=10.0)
    forcing = lambda t: 0.1  # noqa: E731
    for variant in ("wet_mass", "explicit"):
        for name, acc in (
            ("none", make_accelerator("none")),
            ("aitken", AitkenRelaxation()),
            ("iqn_ils", IQNILS()),
        ):
            run = solve_partitioned_1d(
                model, n_steps=1, dt=DT, variant=variant, tol=1e-6,
                max_iter=400, forcing=forcing, accelerator=acc,
            )
            c3.append(
                {
                    "variant": variant,
                    "accelerator": name,
                    "iterations": int(run.iterations[0]),
                    "final_residual": float(run.step_residuals[0]),
                    "converged": bool(run.converged),
                    "residual_history": [float(v) for v in run.residual_history[0]],
                }
            )
    result["c3_stabilisation"] = c3

    # --- C4: Aitken relaxation factor --------------------------------
    c4 = []
    for beta in np.round(np.linspace(0.25, 2.0, 8), 4):
        model = AddedMassModel1D(m_struct=1.0, m_added=float(beta))
        rho, _, oms = measure_contraction_1d(
            model, dt=DT, variant="explicit", accelerator=AitkenRelaxation(),
            n_iter=30, skip=1, window=5,
        )
        rho_ratio, _, oms_ratio = measure_contraction_1d(
            model, dt=DT, variant="explicit", accelerator=AitkenRatio(),
            n_iter=30, skip=1, window=5,
        )
        c4.append(
            {
                "beta": float(beta),
                "omega_analytic_opt": optimal_relaxation(float(beta)),
                "omega_aitken": float(oms[1]),
                "omega_aitken_ratio_form": float(oms_ratio[-1]),
                "rho_after_aitken": rho,
                "rho_after_ratio_form": rho_ratio,
            }
        )
    omegas = [r["omega_aitken"] for r in c4]
    result["c4_aitken"] = {
        "sweep": c4,
        "omega_min": min(omegas),
        "omega_max": max(omegas),
        "inside_band_0p3_0p8": bool(min(omegas) >= 0.3 and max(omegas) <= 0.8),
        "max_rel_err_vs_optimum": max(
            abs(r["omega_aitken"] - r["omega_analytic_opt"]) / r["omega_analytic_opt"] for r in c4
        ),
    }

    # --- theta-split family (consistency preserving) ------------------
    theta_rows = []
    for beta in (0.5, 1.0, 3.0, 10.0):
        theta_rows.append(
            {
                "beta": beta,
                "rho": {str(t): theta_split_rho(beta, t) for t in (0.0, 0.25, 0.5, 0.75, 1.0)},
            }
        )
    result["theta_split"] = theta_rows

    path = OUT / "oned_scaling.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[oned] wrote {path}")
    print(f"[oned] C1 max rel err = {result['c1_max_rel_err']:.3e}")
    print(f"[oned] C2 beta_crit   = {crit['beta_crit']:.6f} (rel err {crit['rel_err_vs_analytic']:.3e})")
    print(f"[oned] C4 omega band  = [{result['c4_aitken']['omega_min']:.4f}, {result['c4_aitken']['omega_max']:.4f}]")
    return result


# ---------------------------------------------------------------------------
# VIV stage
# ---------------------------------------------------------------------------

VIV_CASES = [
    {"u_star": 3.0, "diameter": 16, "mass_ratio": 10.0, "n_steps": 60000, "warmup": 24000},
    {"u_star": 4.0, "diameter": 16, "mass_ratio": 10.0, "n_steps": 60000, "warmup": 24000},
    {"u_star": 5.0, "diameter": 16, "mass_ratio": 10.0, "n_steps": 60000, "warmup": 24000},
    {"u_star": 6.0, "diameter": 16, "mass_ratio": 10.0, "n_steps": 60000, "warmup": 24000},
    {"u_star": 7.5, "diameter": 16, "mass_ratio": 10.0, "n_steps": 60000, "warmup": 24000},
]


def run_viv(index: int, *, extra_steps: int = 0, coupling: str = "iqn_ils") -> dict:
    from tensorlbm.fsi_viv_ibm import VIVConfig, run_viv_case

    OUT.mkdir(parents=True, exist_ok=True)
    spec = dict(VIV_CASES[index])
    u_star = spec.pop("u_star")
    cfg = VIVConfig(
        reduced_velocity=u_star,
        coupling=coupling,
        **spec,
    )
    res = run_viv_case(cfg)
    stride = 10
    payload = {
        "index": index,
        "u_star": u_star,
        "reynolds": cfg.reynolds,
        "mass_ratio": cfg.mass_ratio,
        "zeta": cfg.zeta,
        "diameter": cfg.diameter,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "n_steps": cfg.n_steps,
        "warmup": cfg.warmup,
        "coupling": coupling,
        "amplitude_ratio": res.amplitude_ratio,
        "strouhal": res.strouhal,
        "mean_drag_coefficient": res.mean_drag_coefficient,
        "rms_lift_coefficient": res.rms_lift_coefficient,
        "response_frequency": res.extra.get("response_frequency"),
        "natural_frequency": cfg.natural_frequency,
        "frequency_ratio": (
            res.extra.get("response_frequency", float("nan")) / cfg.natural_frequency
            if res.extra.get("response_frequency")
            else None
        ),
        "converged": res.converged,
        "mean_sub_iterations": float(res.iterations.mean()),
        "max_sub_iterations": int(res.iterations.max()),
        "time": res.time[::stride].tolist(),
        "displacement": res.displacement[::stride].tolist(),
        "lift": res.lift[::stride].tolist(),
        "drag": res.drag[::stride].tolist(),
        "iterations": res.iterations[::stride].tolist(),
    }
    path = OUT / f"viv_U{u_star:.1f}.json"
    path.write_text(json.dumps(payload))
    print(f"[viv {index}] U*={u_star} A/D={res.amplitude_ratio:.3f} St={res.strouhal:.3f} "
          f"Cd={res.mean_drag_coefficient:.3f} conv={res.converged} iters={res.iterations.mean():.2f}")
    return payload


def run_fixed_cylinder_check(diameter: int = 16, n_steps: int = 40000, warmup: int = 20000) -> dict:
    from tensorlbm.fsi_viv_ibm import VIVConfig, run_fixed_cylinder

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = VIVConfig(diameter=diameter, n_steps=n_steps, warmup=warmup)
    out = run_fixed_cylinder(cfg, n_steps=n_steps, warmup=warmup)
    payload = {
        "reynolds": cfg.reynolds,
        "diameter": diameter,
        "n_steps": n_steps,
        "warmup": warmup,
        "cd": out["cd"],
        "strouhal": out["strouhal"],
        "cl_rms": out["cl_rms"],
        "reference_cd": 1.34,
        "reference_strouhal": 0.164,
        "rel_err_cd": abs(out["cd"] - 1.34) / 1.34,
        "rel_err_strouhal": abs(out["strouhal"] - 0.164) / 0.164,
        "drag_history": np.asarray(out["drag"])[::50].tolist(),
        "lift_history": np.asarray(out["lift"])[::50].tolist(),
    }
    (OUT / "fixed_cylinder_re100.json").write_text(json.dumps(payload))
    print(f"[fixed] Cd={out['cd']:.3f} (ref 1.34, err {payload['rel_err_cd']*100:.1f}%) "
          f"St={out['strouhal']:.3f} (ref 0.164, err {payload['rel_err_strouhal']*100:.1f}%)")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["oned", "viv", "fixed", "figures", "all"])
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--coupling", default="iqn_ils")
    args = ap.parse_args()
    if args.stage == "oned":
        run_oned()
    elif args.stage == "viv":
        run_viv(args.case, coupling=args.coupling)
    elif args.stage == "fixed":
        run_fixed_cylinder_check()
    elif args.stage == "figures":
        from tensorlbm.fsi_partitioned_report import build_figures  # noqa: F401
        build_figures()
    else:
        run_oned()
        run_fixed_cylinder_check()


if __name__ == "__main__":
    main()