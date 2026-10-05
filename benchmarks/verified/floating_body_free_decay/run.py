#!/usr/bin/env python3
"""B6DOF-1: still-water free-decay eigenfrequency of a floating body.

Analytic closure benchmark for the tensorlbm 6DOF stack:

  * ``tensorlbm.rao_analysis.compute_natural_frequencies`` -- solves the
    generalized eigenproblem ``(M + A)^-1 C`` for ``omega_n``;
  * ``tensorlbm.rigid_body_6dof.cummins_time_integration`` -- time-domain
    Cummins equation
        (M + A_inf) xi_dd + int K(t-tau) xi_d dtau + C xi = F_exc + F_ext.

For an UNDAMPED, unforced single-DOF oscillator the Cummins system reduces to
the harmonic oscillator with the exact analytic eigenfrequency

        omega_n = sqrt( C_dof,dof / ( M_dof,dof + A_inf_dof,dof ) ).

The benchmark excites that mode with a unit impulse, records the free
oscillation, measures its frequency from the evenly spaced zero crossings, and
compares against (a) the closed-form analytic ``omega_n`` and (b) the exact
symplectic-Euler discrete-dispersion frequency

        omega_h = arccos(1 - (omega_n dt)^2 / 2) / dt,

which is what the integrator's update (explicit velocity + position update)
must reproduce to machine precision.

Two DOFs (heave, roll) x two time-step levels (coarse / fine) give the
mesh-convergence evidence required by the repo standard.

Evidence boundary (NOT covered): caller-synthesised body/hydrodynamic
coefficients; no LBM flow, no hull coupling, no free-surface, no radiation
memory / damping validation.  The damped memory path is reported as a
diagnostic only.  See README.md / NOTES.md.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.rao_analysis import compute_natural_frequencies
from tensorlbm.rigid_body_6dof import (
    BodyProperties6DOF,
    HydrostaticMatrix,
    RadiationData,
    cummins_time_integration,
)

RHO = 1000.0
G = 9.81


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def measure_omega_zero_crossing(t: np.ndarray, x: np.ndarray) -> dict:
    """Frequency from evenly spaced zero crossings (linear interpolation).

    Successive crossings are half periods; a linear fit of crossing time
    versus crossing index gives the half period.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    sign_change = np.where(np.diff(np.sign(x)) != 0)[0]
    crossings = []
    for i in sign_change:
        x0, x1 = x[i], x[i + 1]
        if x1 == x0:
            continue
        frac = -x0 / (x1 - x0)
        crossings.append(t[i] + frac * (t[i + 1] - t[i]))
    crossings = np.asarray(crossings)
    k = np.arange(crossings.size)
    A = np.vstack([np.ones_like(k, dtype=np.float64), k.astype(np.float64)]).T
    coef, *_ = np.linalg.lstsq(A, crossings, rcond=None)
    half_period = coef[1]
    omega = math.pi / half_period
    # residual of the linear crossing-time fit (quality indicator)
    resid = crossings - A @ coef
    return {
        "omega_meas": omega,
        "half_period": half_period,
        "n_crossings": int(crossings.size),
        "crossing_fit_rms": float(np.sqrt(np.mean(resid**2))),
    }


def measure_omega_lsq_sine(t: np.ndarray, x: np.ndarray, omega_guess: float) -> dict:
    """Cross-check: 2-parameter linear LSQ sine fit at a fixed frequency grid.

    x(t) ~ a cos(w t) + b sin(w t); minimises ||x - model||^2 over a fine w grid.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    ws = omega_guess * np.linspace(0.95, 1.05, 4001)
    best = None
    for w in ws:
        c = np.cos(w * t)
        s = np.sin(w * t)
        A = np.vstack([c, s]).T
        coef, res, *_ = np.linalg.lstsq(A, x, rcond=None)
        r2 = 1.0 - res[0] / np.sum((x - x.mean()) ** 2) if res.size else 1.0
        if best is None or r2 > best[0]:
            best = (r2, w)
    return {"omega_meas_lsq": float(best[1]), "fit_r2": float(best[0])}


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def build_case(dof: int, name: str, mass: float, added_mass: float,
               stiffness: float, inertia_diag: tuple, waterplane_area: float):
    """Assemble body/hydrostatics/radiation for a single-DOF mode."""
    inertia = torch.diag(torch.tensor(inertia_diag, dtype=torch.float32))
    body = BodyProperties6DOF(
        mass=mass,
        inertia_matrix=inertia,
        displacement_volume=mass / RHO,
        waterplane_area=waterplane_area,
        water_density=RHO,
        gravity=G,
    )
    C = torch.zeros(6, 6)
    C[dof, dof] = stiffness
    hydro = HydrostaticMatrix(c_matrix=C)

    n_freq = 16
    omega_f = torch.linspace(0.02, 5.0, n_freq)
    added = torch.zeros(n_freq, 6, 6)
    damping = torch.zeros(n_freq, 6, 6)  # undamped
    added[:, dof, dof] = added_mass
    radiation = RadiationData(omega_f, added, damping, added[0].clone())
    return body, hydro, radiation


def run_undamped(dof, name, mass, added_mass, stiffness, inertia_diag,
                 waterplane_area, dt, n_periods):
    body, hydro, radiation = build_case(
        dof, name, mass, added_mass, stiffness, inertia_diag, waterplane_area
    )
    M = body.build_mass_matrix()
    m_total = M[dof, dof].item() + added_mass
    omega_n = math.sqrt(stiffness / m_total)
    T = 2.0 * math.pi / omega_n

    # eigen solver cross-check
    nat = compute_natural_frequencies(M, radiation.added_mass_inf, hydro.c_matrix)
    omega_n_eig = nat[dof].item()

    n_steps = int(round(n_periods * T / dt))
    exc = torch.zeros(n_steps, 6)
    exc[0, dof] = 1.0  # unit impulse

    t0 = time.time()
    state = cummins_time_integration(body, hydro, radiation, exc, dt, n_steps)
    wall_s = time.time() - t0

    pos = np.array([p.numpy()[dof] for p in state.position_history], dtype=np.float64)
    t = np.arange(pos.size, dtype=np.float64) * dt

    zc = measure_omega_zero_crossing(t, pos)
    lsq = measure_omega_lsq_sine(t, pos, omega_n)

    omega_h = math.acos(max(-1.0, min(1.0, 1.0 - 0.5 * (omega_n * dt) ** 2))) / dt

    return {
        "case": name,
        "dof": dof,
        "dt": dt,
        "n_steps": n_steps,
        "n_periods_requested": n_periods,
        "m_eff": m_total,
        "stiffness": stiffness,
        "omega_n_analytic": omega_n,
        "omega_n_eig": omega_n_eig,
        "eig_vs_analytic_pct": (omega_n_eig - omega_n) / omega_n * 100.0,
        "omega_n_dt": omega_n * dt,
        "omega_h_symplectic": omega_h,
        "omega_h_vs_analytic_pct": (omega_h - omega_n) / omega_n * 100.0,
        "omega_meas": zc["omega_meas"],
        "omega_meas_pct": (zc["omega_meas"] - omega_n) / omega_n * 100.0,
        "omega_meas_vs_discrete_pct": (zc["omega_meas"] - omega_h) / omega_h * 100.0,
        "omega_meas_lsq": lsq["omega_meas_lsq"],
        "omega_lsq_pct": (lsq["omega_meas_lsq"] - omega_n) / omega_n * 100.0,
        "omega_lsq_vs_discrete_pct": (lsq["omega_meas_lsq"] - omega_h) / omega_h * 100.0,
        "fit_r2": lsq["fit_r2"],
        "n_crossings": zc["n_crossings"],
        "crossing_fit_rms": zc["crossing_fit_rms"],
        "amplitude": float(np.abs(pos).max()),
        "wall_s": wall_s,
        "finite": bool(np.all(np.isfinite(pos))),
    }


def run_damped_diagnostic(dof, name, mass, added_mass, stiffness, inertia_diag,
                          waterplane_area, B0, dt, n_periods, wmax):
    """Diagnostic only: constant B(w)=B0 over [0, wmax].

    Analytic viscous reference: zeta = B0 / (2 sqrt((M+A) C)),
    omega_d = omega_n sqrt(1 - zeta^2).  The finite-band K(t) is only an
    approximation of the 2 B0 delta(t) kernel, so this is NOT a criterion.
    """
    body, hydro, radiation = build_case(
        dof, name, mass, added_mass, stiffness, inertia_diag, waterplane_area
    )
    n_freq = 256
    omega_f = torch.linspace(0.0, wmax, n_freq)
    added = torch.zeros(n_freq, 6, 6)
    damping = torch.zeros(n_freq, 6, 6)
    added[:, dof, dof] = added_mass
    damping[:, dof, dof] = B0
    radiation = RadiationData(omega_f, added, damping, added[0].clone())

    M = body.build_mass_matrix()
    m_total = M[dof, dof].item() + added_mass
    omega_n = math.sqrt(stiffness / m_total)
    zeta = B0 / (2.0 * math.sqrt(m_total * stiffness))
    omega_d = omega_n * math.sqrt(max(0.0, 1.0 - zeta**2))
    T = 2.0 * math.pi / omega_n
    n_steps = int(round(n_periods * T / dt))
    exc = torch.zeros(n_steps, 6)
    exc[0, dof] = 1.0

    t0 = time.time()
    state = cummins_time_integration(body, hydro, radiation, exc, dt, n_steps)
    wall_s = time.time() - t0
    pos = np.array([p.numpy()[dof] for p in state.position_history], dtype=np.float64)
    t = np.arange(pos.size, dtype=np.float64) * dt
    zc = measure_omega_zero_crossing(t, pos)
    return {
        "case": name,
        "B0": B0,
        "zeta": zeta,
        "dt": dt,
        "omega_n_analytic": omega_n,
        "omega_d_analytic": omega_d,
        "omega_meas": zc["omega_meas"],
        "omega_meas_vs_d_analytic_pct": (zc["omega_meas"] - omega_d) / omega_d * 100.0,
        "amp_first_period": float(np.abs(pos[: max(1, int(T / dt))]).max()),
        "amp_last_period": float(np.abs(pos[-max(1, int(T / dt)):]).max()),
        "wall_s": wall_s,
        "criterion": "diagnostic-only",
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--periods", type=int, default=16)
    ap.add_argument("--skip-damped", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)

    # Case definitions.  omega_n is set by the chosen (stiffness, m_eff).
    #   heave: C = rho g A_wp = 2000,  m_eff = 1000 + 1000 = 2000  -> wn = 1.000
    #   roll : C = rho g grad GM_T = 4905, m_eff = 800 + 800 = 1600 -> wn = 1.751
    heave = dict(dof=2, name="heave", mass=1000.0, added_mass=1000.0,
                 stiffness=2000.0, inertia_diag=(1.0, 1.0, 1.0),
                 waterplane_area=2000.0 / (RHO * G))
    roll = dict(dof=3, name="roll", mass=1000.0, added_mass=800.0,
                stiffness=4905.0, inertia_diag=(800.0, 1.0, 1.0),
                waterplane_area=1.0)

    cases = []
    for spec in (heave, roll):
        wn = math.sqrt(spec["stiffness"] / (spec["mass"] + spec["added_mass"]))
        dt_coarse = 0.5 / wn        # omega_n dt = 0.5
        for tag, dt in (("coarse", dt_coarse), ("fine", dt_coarse / 2.0)):
            spec_run = {k: v for k, v in spec.items()}
            r = run_undamped(**spec_run, dt=dt, n_periods=args.periods)
            r["level"] = tag
            cases.append(r)

    # per-case two-level span
    by_case: dict[str, list] = {}
    for r in cases:
        by_case.setdefault(r["case"], []).append(r)
    spans = {}
    for name, rs in by_case.items():
        errs = [abs(r["omega_meas_pct"]) for r in rs]
        meas = [r["omega_meas"] for r in rs]
        spans[name] = {
            "max_abs_err_pct": max(errs),
            "meas_span_pct": (max(meas) - min(meas)) / np.mean(meas) * 100.0,
            "eig_err_pct": max(abs(r["eig_vs_analytic_pct"]) for r in rs),
            "discrete_meas_err_pct": max(abs(r["omega_meas_vs_discrete_pct"]) for r in rs),
        }

    damped = []
    if not args.skip_damped:
        # Diagnostic sweep exposing the setting-sensitivity / instability of the
        # finite-band delayed-kernel convolution.  NOT part of the criteria.
        for dt in (0.02, 0.05):
            for B0 in (50.0, 200.0):
                damped.append(run_damped_diagnostic(
                    heave["dof"], heave["name"], heave["mass"], heave["added_mass"],
                    heave["stiffness"], heave["inertia_diag"], heave["waterplane_area"],
                    B0=B0, dt=dt, n_periods=40, wmax=math.pi / dt,
                ))

    TOL = 3.0
    verdict = {
        "tolerance_pct": TOL,
        "each_level_within_tol": all(abs(r["omega_meas_pct"]) <= TOL for r in cases),
        "each_case_span_within_tol": all(v["meas_span_pct"] <= TOL for v in spans.values()),
        "eig_exact": all(v["eig_err_pct"] < 1e-3 for v in spans.values()),
        "integrator_matches_discrete_theory": all(
            v["discrete_meas_err_pct"] < 0.5 for v in spans.values()
        ),
    }
    verdict["pass"] = all(verdict[k] for k in
                          ("each_level_within_tol", "each_case_span_within_tol",
                           "eig_exact", "integrator_matches_discrete_theory"))

    result = {
        "benchmark": "floating_body_free_decay",
        "scope": "still-water free-decay eigenfrequency (undamped analytic closure)",
        "library": [
            "tensorlbm.rao_analysis.compute_natural_frequencies",
            "tensorlbm.rigid_body_6dof.cummins_time_integration",
        ],
        "dtype": "float32 (json metrics in float64)",
        "device": "cpu",
        "reference": {
            "analytic": "omega_n = sqrt(C_dof,dof / (M_dof,dof + A_inf_dof,dof))",
            "discrete": "omega_h = arccos(1 - (omega_n dt)^2/2)/dt  (symplectic Euler)",
            "sources": [
                "Cummins (1962) impulse-response formulation",
                "Faltinsen (1990) Sea Loads on Ships and Offshore Structures, ch.5",
                "standard harmonic-oscillator / symplectic-Euler dispersion",
            ],
        },
        "acceptance": (
            "|omega_meas - omega_n| <= 3% at every level; two-level span <= 3%; "
            "eigen-solver exact; integrator reproduces discrete-dispersion theory <0.5%"
        ),
        "cases": cases,
        "spans": spans,
        "verdict": verdict,
        "verified": verdict["pass"],
        "damped_memory_diagnostic": damped,
        "evidence_boundary": (
            "Caller-synthesised body and hydrodynamic coefficients. No LBM flow, "
            "hull coupling, free surface, or radiation-memory (damping) validation. "
            "The damped memory path is a reported diagnostic only, NOT part of the "
            "acceptance criteria."
        ),
    }

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps({"verdict": verdict, "spans": spans,
                      "damped": [{k: v[k] for k in
                                  ("B0", "zeta", "omega_meas_vs_d_analytic_pct",
                                   "amp_first_period", "amp_last_period")}
                                 for v in damped]},
                     indent=2, sort_keys=True))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())