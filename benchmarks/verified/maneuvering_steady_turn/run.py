#!/usr/bin/env python3
"""BMAN-1: ship maneuvering steady turn -- linear sway-yaw analytic closure.

Verifies the tensorlbm maneuvering stack (``tensorlbm.maneuvering``) for the
horizontal-plane problem, mirroring the verified ``floating_body_free_decay``
seakeeping closure:

  * exact steady-turn fixed point of the linear sway-yaw model
    (``ManeuveringModel.steady_turn`` -- an exact 2x2 solve);
  * independent algebraic cross-routes for the same fixed point
    (NumPy dense solve, and the closed-form Nomoto gain ``K = r/delta``);
  * time-domain integration of the sway-yaw ODE + kinematics
    (``simulate_turn``, semi-implicit Euler and RK4);
  * the exact analytic modal transient
    (``analytic_transient``) as the reference for the whole r(t), v(t) curves,
    not just the fixed point.

Two rudder angles (10 deg, 20 deg) x two time-step levels (coarse, fine) give
the multi-case / two-level evidence required by the repo standard.  The measured
steady turn radius ``R`` and drift angle ``beta`` must match the analytic values
within 3% at every level, and the two-level span must be <= 3%.

Evidence boundary (NOT covered): the hydrodynamic derivatives are a
caller-supplied representative KVLCC2-class *linear* set (not extracted from an
LBM flow, not from a PMM test).  There is no free surface, no rudder/propeller
slipstream, no nonlinear (cross-flow / Abkowitz higher-order) hull terms, and no
LBM end-to-end hull-in-flow coupling -- that end-to-end path is a structural
blocker (free surface; see benchmarks/STATUS.md).  See README.md / NOTES.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from tensorlbm.maneuvering import (  # noqa: E402
    NOMINAL_TANKER_PRIME,
    ManeuveringModel,
    analytic_transient,
    prime_to_dimensional,
    simulate_turn,
)

# --------------------------------------------------------------------------- #
# ship (KVLCC2-class large tanker, full scale)
# --------------------------------------------------------------------------- #
L = 320.0        # length between perpendiculars [m]
B = 58.0         # beam [m]
D = 20.8         # draft [m]
CB = 0.81        # block coefficient
RHO = 1000.0     # water density [kg/m^3]
U = 7.97         # forward speed [m/s] (~15.5 kn)
MASS = RHO * CB * L * B * D
IZ = MASS * (0.25 * L) ** 2

RUDDERS_DEG = (10.0, 20.0)
# two time-step levels (fine = coarse / 4)
DT_LEVELS = {"coarse": 1.0, "fine": 0.25}
T_END = 600.0            # enough for ~14 slow time constants


def build_model() -> ManeuveringModel:
    dim = prime_to_dimensional(NOMINAL_TANKER_PRIME, L, D, U, RHO)
    return ManeuveringModel(
        L=L, d=D, U=U, mass=MASS, inertia_z=IZ, x_G=0.0, rho=RHO,
        name="kvlcc2-class tanker", **dim,
    )


def circle_fit_radius(x: np.ndarray, y: np.ndarray) -> float:
    """Kasa algebraic circle fit; returns the fitted radius."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx**2 + cy**2
    return float(math.sqrt(max(r2, 0.0)))


def steady_measure(sim: dict, t_frac: float = 0.5) -> dict:
    """Average v, r over the final (1 - t_frac) of the run."""
    n = sim["n_steps"]
    i0 = int(t_frac * n)
    v = np.asarray(sim["v"][i0:], dtype=float)
    r = np.asarray(sim["r"][i0:], dtype=float)
    x = np.asarray(sim["x"][i0:], dtype=float)
    y = np.asarray(sim["y"][i0:], dtype=float)
    return {
        "v": float(v.mean()),
        "r": float(r.mean()),
        "v_std": float(v.std()),
        "r_std": float(r.std()),
        "R_circle": circle_fit_radius(x, y),
        "x": x, "y": y,
    }


def run_case(model: ManeuveringModel, delta_deg: float, level: str, dt: float,
             integrator: str) -> dict:
    delta = math.radians(delta_deg)
    n_steps = int(round(T_END / dt))

    ana = model.steady_turn(delta)
    R_geo_ana = ana.speed / abs(ana.r)      # geometric turning radius V/|r|

    t0 = time.time()
    sim = simulate_turn(model, delta, dt, n_steps, integrator=integrator)
    wall = time.time() - t0

    m = steady_measure(sim)
    R_geo_meas = math.hypot(U, m["v"]) / abs(m["r"])
    beta_meas = math.degrees(math.atan2(-m["v"], U))

    # transient: analytic modal solution vs the numerical r(t), v(t)
    t = np.asarray(sim["t"], dtype=float)
    v_a, r_a = analytic_transient(model, delta, t)
    v_n = np.asarray(sim["v"], dtype=float)
    r_n = np.asarray(sim["r"], dtype=float)
    r_scale = abs(ana.r)
    trans_r_err = float(np.max(np.abs(r_n - r_a)) / r_scale) * 100.0
    trans_v_err = float(np.max(np.abs(v_n - v_a)) / abs(ana.v)) * 100.0

    return {
        "level": level,
        "delta_deg": delta_deg,
        "integrator": integrator,
        "dt": dt,
        "n_steps": n_steps,
        "wall_s": wall,
        # analytic
        "v_ana": ana.v,
        "r_ana": ana.r,
        "R_ana": R_geo_ana,
        "R_over_L_ana": R_geo_ana / L,
        "beta_ana_deg": ana.beta_deg,
        "K_ana": ana.K,
        # measured
        "v_meas": m["v"],
        "r_meas": m["r"],
        "R_meas": R_geo_meas,
        "R_over_L_meas": R_geo_meas / L,
        "beta_meas_deg": beta_meas,
        "R_circle_fit": m["R_circle"],
        # errors
        "R_err_pct": (R_geo_meas - R_geo_ana) / R_geo_ana * 100.0,
        "beta_err_pct": (beta_meas - ana.beta_deg) / ana.beta_deg * 100.0,
        "R_circle_err_pct": (m["R_circle"] - R_geo_ana) / R_geo_ana * 100.0,
        "r_err_pct": (m["r"] - ana.r) / ana.r * 100.0,
        "steady_r_std": m["r_std"],
        "transient_r_err_pct": trans_r_err,
        "transient_v_err_pct": trans_v_err,
    }


def run_nonlinear_diagnostic(base: ManeuveringModel, delta_deg: float = 20.0) -> dict:
    """Diagnostic: add cross-flow (|v|v) hull terms and verify the nonlinear
    equilibrium is reproduced by the time-domain integrator.

    The cross-flow coefficients are a representative magnitude (~10-20% of the
    linear term at the steady sway speed).  This exercises the nonlinear branch
    of the module; it is NOT part of the linear acceptance criteria.
    """
    import dataclasses

    delta = math.radians(delta_deg)
    Y_vv = -1.2e6    # reduces |v| (drag opposes sway)
    N_vv = -8.0e7    # cross-flow yaw damping
    model = dataclasses.replace(base, Y_vv=Y_vv, N_vv=N_vv)

    ana = model.steady_turn(delta)               # nonlinear bisection equilibrium
    dt = DT_LEVELS["fine"]
    n_steps = int(round(T_END / dt))
    sim = simulate_turn(model, delta, dt, n_steps, integrator="rk4")
    m = steady_measure(sim)
    R_meas = math.hypot(U, m["v"]) / abs(m["r"])
    R_ana = ana.speed / abs(ana.r)

    lin = base.steady_turn(delta)
    return {
        "note": "diagnostic-only (nonlinear cross-flow terms; not in the acceptance set)",
        "Y_vv": Y_vv,
        "N_vv": N_vv,
        "delta_deg": delta_deg,
        "linear_r": lin.r,
        "nonlinear_r_ana": ana.r,
        "nonlinear_r_meas": m["r"],
        "r_shift_vs_linear_pct": (ana.r - lin.r) / lin.r * 100.0,
        "beta_ana_deg": ana.beta_deg,
        "beta_meas_deg": math.degrees(math.atan2(-m["v"], U)),
        "R_ana": R_ana,
        "R_meas": R_meas,
        "integrator_vs_rootfind_R_pct": (R_meas - R_ana) / R_ana * 100.0,
        "integrator_vs_rootfind_beta_pct": (math.degrees(math.atan2(-m["v"], U))
                                            - ana.beta_deg) / ana.beta_deg * 100.0,
        "criterion": "diagnostic-only",
    }


def physical_plausibility(model: ManeuveringModel) -> dict:
    """Corroboration only (NOT an acceptance criterion): are the computed turn
    characteristics inside the published range for large tankers?"""
    d10 = model.steady_turn(math.radians(10.0))
    d20 = model.steady_turn(math.radians(20.0))
    return {
        "note": "corroboration only, NOT part of the acceptance criteria",
        "computed": {
            "R_over_L_at_10deg": d10.R_over_L,
            "R_over_L_at_20deg": d20.R_over_L,
            "tactical_diameter_over_L_at_20deg": 2.0 * d20.R_over_L,
            "drift_angle_deg_at_20deg": d20.beta_deg,
            "nomoto_K_prime": d20.K * model.L / model.U,
        },
        "published_ranges": {
            "tactical_diameter_over_L": "~2.5-4.5 (large tanker, 20-35 deg rudder; "
                                        "KVLCC2 SIMMAN sim D_T/L ~3)",
            "steady_drift_angle_deg": "~10-25 (large tanker steady turn)",
            "nomoto_K_prime": "~0.5-3 (large tanker)",
        },
        "sources": [
            "SIMMAN 2008 workshop turning-circle submissions (KVLCC2)",
            "Yasukawa & Yoshimura (2015) Introduction of MMG standard method, "
            "J. Marine Sci. Tech. 20:37-52",
            "Fossen (2011) ch.7; Lewis (ed.) (1989) PNA Vol.III ch.9",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_HERE)
    args = ap.parse_args()

    model = build_model()
    dim = prime_to_dimensional(NOMINAL_TANKER_PRIME, L, D, U, RHO)

    # ---- exact analytic fixed point + independent cross-routes ------------- #
    delta_ref = math.radians(20.0)
    A = model.damping_matrix().numpy()
    b = model.input_vector(delta_ref).numpy()
    q_lib = np.array([model.steady_turn(delta_ref).v, model.steady_turn(delta_ref).r])
    q_np = np.linalg.solve(A, -b)                       # independent route
    K_r = model.nomoto_K() * delta_ref                  # independent route
    route_agree_pct = {
        "numpy_vs_library_r": abs(q_np[1] - q_lib[1]) / abs(q_lib[1]) * 100.0,
        "nomoto_vs_library_r": abs(K_r - q_lib[1]) / abs(q_lib[1]) * 100.0,
    }

    modes = model.modes()
    nonlinear = run_nonlinear_diagnostic(model)
    plausibility = physical_plausibility(model)

    # ---- cases: 2 rudders x 2 dt levels x 2 integrators ------------------- #
    cases = []
    for ddeg in RUDDERS_DEG:
        for level, dt in DT_LEVELS.items():
            cases.append(run_case(model, ddeg, level, dt, "semi_implicit_euler"))
    # RK4 as an independent integrator at the fine level
    for ddeg in RUDDERS_DEG:
        cases.append(run_case(model, ddeg, "fine_rk4", DT_LEVELS["fine"], "rk4"))

    # ---- per-rudder two-level span ---------------------------------------- #
    spans = {}
    for ddeg in RUDDERS_DEG:
        rs = [c for c in cases if c["delta_deg"] == ddeg
              and c["integrator"] == "semi_implicit_euler"]
        Rs = [c["R_meas"] for c in rs]
        betas = [c["beta_meas_deg"] for c in rs]
        # transient error per level (this IS dt-dependent, unlike the fixed point)
        te = {c["level"]: c["transient_r_err_pct"] for c in rs}
        ratio = (te["coarse"] / te["fine"]) if te.get("fine", 0) else float("nan")
        spans[f"{ddeg:g}deg"] = {
            "R_span_pct": (max(Rs) - min(Rs)) / np.mean(Rs) * 100.0,
            "beta_span_pct": (max(betas) - min(betas)) / abs(np.mean(betas)) * 100.0,
            "max_R_err_pct": max(abs(c["R_err_pct"]) for c in rs),
            "max_beta_err_pct": max(abs(c["beta_err_pct"]) for c in rs),
            "transient_r_err_coarse_pct": te.get("coarse"),
            "transient_r_err_fine_pct": te.get("fine"),
            "transient_refinement_ratio": ratio,
        }

    TOL = 3.0
    euler = [c for c in cases if c["integrator"] == "semi_implicit_euler"]
    rk4 = [c for c in cases if c["integrator"] == "rk4"]
    verdict = {
        "tolerance_pct": TOL,
        "each_level_R_within_tol": all(abs(c["R_err_pct"]) <= TOL for c in cases),
        "each_level_beta_within_tol": all(abs(c["beta_err_pct"]) <= TOL for c in cases),
        "two_level_span_within_tol": all(v["R_span_pct"] <= TOL and v["beta_span_pct"] <= TOL
                                         for v in spans.values()),
        "analytic_routes_agree": all(v < 1e-6 for v in route_agree_pct.values()),
        "transient_rk4_fine_within_tol": all(c["transient_r_err_pct"] <= TOL for c in rk4),
        "transient_euler_fine_within_tol": all(c["transient_r_err_pct"] <= TOL for c in euler),
        "transient_refinement_ratio_gt_3": all(v["transient_refinement_ratio"] > 3.0
                                               for v in spans.values()),
        "directionally_stable": all(model.steady_turn(math.radians(d)).stable
                                    for d in RUDDERS_DEG),
    }
    verdict["pass"] = all(verdict[k] for k in (
        "each_level_R_within_tol", "each_level_beta_within_tol",
        "two_level_span_within_tol", "analytic_routes_agree",
        "transient_rk4_fine_within_tol", "transient_euler_fine_within_tol",
        "transient_refinement_ratio_gt_3", "directionally_stable"))

    result = {
        "benchmark": "maneuvering_steady_turn",
        "scope": "horizontal-plane linear sway-yaw steady turn + modal transient (analytic closure)",
        "library": [
            "tensorlbm.maneuvering.ManeuveringModel.steady_turn",
            "tensorlbm.maneuvering.simulate_turn",
            "tensorlbm.maneuvering.analytic_transient",
        ],
        "device": "cpu",
        "dtype": "float64 (library + metrics)",
        "ship": {
            "class": "KVLCC2-class large tanker (full scale)",
            "L_m": L, "B_m": B, "draft_m": D, "Cb": CB, "U_m_s": U,
            "mass_kg": MASS, "Iz_kg_m2": IZ,
        },
        "derivatives": {
            "prime_input": NOMINAL_TANKER_PRIME,
            "dimensional": dim,
            "note": ("representative KVLCC2-class linear derivative set "
                     "(order-of-magnitude consistent with published PMM/MMG data, "
                     "Yasukawa & Yoshimura 2015 / SIMMAN 2008); caller-supplied, "
                     "NOT a digit-exact reprint and NOT extracted from an LBM flow."),
        },
        "reference": {
            "analytic_fixed_point": "solve A q = -b for q=[v,r]; R=V/|r|, beta=atan2(-v,U)",
            "nomoto": "K = r/delta = (N_v Y_d - Y_v N_d)/(Y_v N_r - N_v(Y_r - mU))",
            "transient": "q(t)=q_ss + sum_i c_i phi_i exp(lambda_i t), lambda=eig(M^-1 A)",
            "sources": [
                "Abkowitz (1964) Lectures on Ship Hydrodynamics and Stability",
                "Nomoto et al. (1957) J. Zosen Kiokai 101:41-52",
                "Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion Control, ch.7",
                "Lewis (ed.) (1989) Principles of Naval Architecture Vol.III ch.9",
            ],
        },
        "modes": {"eigenvalues": modes["eigenvalues"],
                  "time_constants": modes["time_constants"]},
        "acceptance": (
            "steady R and drift angle beta within 3% of the analytic fixed point at "
            "every (rudder, dt) level; two-level span <=3%; independent algebraic "
            "routes agree <1e-6%; Euler transient r(t) within 3% at fine level and "
            "refines by >3x per dt reduction; RK4 transient within 3%"
        ),
        "cases": cases,
        "spans": spans,
        "nonlinear_crossflow_diagnostic": nonlinear,
        "physical_plausibility": plausibility,
        "analytic_route_agreement_pct": route_agree_pct,
        "verdict": verdict,
        "verified": verdict["pass"],
        "evidence_boundary": (
            "Caller-supplied representative KVLCC2-class LINEAR derivatives. No LBM "
            "flow, no free surface, no rudder/propeller slipstream, no nonlinear "
            "cross-flow/higher-order Abkowitz terms, no LBM end-to-end hull-in-flow "
            "maneuvering. End-to-end fluid-maneuvering coupling is a structural "
            "blocker (free surface)."
        ),
    }

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True),
                                     encoding="utf-8")
    print(json.dumps({"verdict": verdict, "spans": spans,
                      "route_agree": route_agree_pct,
                      "nonlinear_diagnostic": nonlinear,
                      "cases": [{k: c[k] for k in
                                 ("delta_deg", "level", "integrator", "dt",
                                  "R_over_L_meas", "beta_meas_deg",
                                  "R_err_pct", "beta_err_pct", "transient_r_err_pct")}
                                for c in cases]},
                     indent=2, sort_keys=True))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())