"""Stefan-number sweep of the enthalpy thermal LBM against the two-phase exact solution.

This module is the numerical counterpart of :mod:`stefan_two_phase`: it drives
:class:`~tensorlbm.phase_change.stefan_verification.EnthalpyThermalLBM` on a
one-dimensional planar Stefan case whose *initial* liquid is superheated
(``T_i > T_freeze``) and whose wall is held below the freezing temperature
(``T_0 < T_freeze``) -- i.e. the classical two-phase Stefan configuration -- and
measures the acceptance quantities of the campaign node:

===========  ==========================================================  =======
criterion    quantity                                                    bound
===========  ==========================================================  =======
C1           front position vs ``s(t) = 2*lam*sqrt(alpha*t)``            < 5 %
C2           temperature profile vs the similarity solution (max)        < 3 %
C3           energy closure: wall heat input vs sensible + latent        < 2 %
C4           liquid fraction bounds ``0 <= f_l <= 1``                    hard
C5           grid convergence of the front error (3 resolutions)         order >= 1
C6           JSON artefact + sha256 provenance                           present
===========  ==========================================================  =======

All quantities are lattice units (``dx = dt = 1``, ``rho = 1``).  Nothing is
hard-coded to a particular run: the front, the profiles and the energy
decomposition all come from the exact solution for the *same* parameters the
solver is given.

Numerical setup
---------------
``nx`` is chosen from the similarity solution itself so that the (truncated)
far field sits at ``eta_l >= far_eta`` -- i.e. the Dirichlet far boundary is
thermally untouched to ``erfc(far_eta)/erfc(lam_l)``.  ``front_cells`` is the
number of lattice cells spanned by the front at the final time, which is the
resolution parameter of the grid-convergence study.

Known limitations
-----------------
* single diffusivity for both phases (the thermal LBE diffuses one
  sensible-enthalpy field); the exact solution supports ``alpha_s != alpha_l``
  but the LBM verification does not exercise it,
* first-order (operator-splitting) coupling of the latent/sensible parts, so
  the observed front order is expected to be ~1,
* equilibrium (single-valued) enthalpy: an undercooled / metastable liquid
  cannot be represented.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .stefan_two_phase import TwoPhaseSimilarity
from .stefan_verification import EnthalpyThermalLBM

__all__ = [
    "FRONT_TOL",
    "PROFILE_TOL",
    "ENERGY_TOL",
    "run_two_phase_stefan",
    "run_two_phase_sweep",
    "run_two_phase_grid_convergence",
    "case_criteria",
]

#: predeclared acceptance thresholds (campaign node C1/C2/C3)
FRONT_TOL = 0.05
PROFILE_TOL = 0.03
ENERGY_TOL = 0.02


# ---------------------------------------------------------------------------
# single run
# ---------------------------------------------------------------------------
def run_two_phase_stefan(
    stefan_solid: float,
    stefan_liquid: float = 0.0,
    *,
    front_cells: float = 33.0,
    far_eta: float = 2.5,
    tau: float = 1.0,
    cp: float = 1.0,
    latent: float = 1.0,
    t_freeze: float = 0.0,
    lattice: str = "D2Q5",
    sample_fractions: Sequence[float] = (0.0625, 0.25, 1.0),
    energy_audit: bool = True,
    profile_band: float = 4.0,
) -> dict[str, Any]:
    """Run one 1-D two-phase Stefan case and return the verification record.

    Parameters
    ----------
    stefan_solid, stefan_liquid :
        ``Ste_s = cp*(T_freeze - T_0)/L``, ``Ste_l = cp*(T_i - T_freeze)/L``.
    front_cells :
        Solid thickness (in cells) reached at the final time -- the resolution
        knob of the grid-convergence study.
    far_eta :
        Required Boltzmann-variable distance of the far Dirichlet node,
        ``eta_l(nx) = nx/(2*sqrt(alpha*t_end)) >= far_eta``.
    sample_fractions :
        Fractions of the final time at which the front / energy series is
        recorded (measured in *front travel*, i.e. ``t/t_end``).
    """
    if front_cells <= 0:
        raise ValueError("front_cells must be positive")
    alpha = (tau - 0.5) / 3.0
    model = TwoPhaseSimilarity(
        stefan_solid=stefan_solid,
        stefan_liquid=stefan_liquid,
        alpha_solid=alpha,
        alpha_liquid=alpha,
        t_freeze=t_freeze,
        cp_solid=cp,
        cp_liquid=cp,
        latent=latent,
    )

    t_target = model.time_for_front(float(front_cells))
    n_steps = max(1, int(round(t_target)))
    t_end = float(n_steps)
    s_end = model.front(t_end)

    # far field: eta_l(nx) = nx/(2*sqrt(alpha*t_end)) >= far_eta
    nx = int(math.ceil(far_eta * 2.0 * math.sqrt(alpha * t_end)))
    nx = max(32, int(math.ceil(nx / 8.0) * 8))  # round up, multiple of 8

    sim = EnthalpyThermalLBM(
        nx=nx,
        ny=1,
        tau=tau,
        cp=cp,
        latent=latent,
        t_freeze=t_freeze,
        lattice=lattice,
        periodic=(False, True),
    )
    T_i = model.initial_temperature
    sim.set_field(np.full((nx, 1), T_i), np.ones((nx, 1)))  # superheated liquid

    interior = np.zeros((nx, 1), dtype=bool)
    interior[1:-1, :] = True
    wall_mask = np.zeros((nx, 1), dtype=bool)
    wall_mask[0, 0] = True
    far_mask = np.zeros((nx, 1), dtype=bool)
    far_mask[nx - 1, 0] = True
    sim.add_dirichlet(wall_mask, model.wall_temperature, 0.0)  # cold wall -> solid
    sim.add_dirichlet(far_mask, T_i, 1.0)  # far liquid

    n_interior = int(np.count_nonzero(interior))
    h_init = sim.total_enthalpy(mask=interior)

    sample_steps = sorted({max(1, int(round(f * n_steps))) for f in sample_fractions})
    if sample_steps[-1] < n_steps:
        sample_steps.append(n_steps)

    q_1st = 0.0
    q_2nd = 0.0
    q_1st_right = 0.0
    q_2nd_right = 0.0
    fl_min = float(np.min(sim.f_l[interior]))
    fl_max = float(np.max(sim.f_l[interior]))
    done = 0
    series: list[dict[str, Any]] = []
    last: dict[str, Any] = {}

    x = np.arange(nx, dtype=float)
    dT_ref = model.dT_reference

    for target in sample_steps:
        while done < target:
            if energy_audit:
                q_1st += sim.wall_heat_rate("left", order=1)
                q_2nd += sim.wall_heat_rate("left", order=2)
                # far boundary: only a leakage diagnostic (the exact solution has
                # no flux there; it measures how well the domain is truncated)
                q_1st_right += sim.wall_heat_rate("right", order=1)
                q_2nd_right += sim.wall_heat_rate("right", order=2)
            sim.step(1)
            done += 1
            fl_min = min(fl_min, float(np.min(sim.f_l[interior])))
            fl_max = max(fl_max, float(np.max(sim.f_l[interior])))
        t = float(done)
        s_exact = model.front(t)
        s_integral = sim.front_x(solid_behind_left=True, mask=interior)
        s_local = sim.front_half_crossing(solid_behind_left=True)
        # The lattice nodes 1..nx-2 own the control volume [0.5, nx-1.5]; the wall
        # node owns only its [0, 0.5] half-cell.  Comparing the raw node sum with
        # s_exact would therefore charge the solver for a half-cell it does not
        # own, so the primary front metric uses the same window on both sides.
        s_exact_window = max(min(s_exact, nx - 1.5) - 0.5, 1e-12)
        T_exact_now = model.temperature(x, t)
        prof_err_now = float(np.max(np.abs(sim.T[:, 0] - T_exact_now)) / dT_ref)

        entry: dict[str, Any] = {
            "step": int(done),
            "time": t,
            "front_exact": s_exact,
            "front_exact_window": float(s_exact_window),
            "front_phase_integral": s_integral,
            "front_half_crossing": s_local,
            "front_window_rel_error": abs(s_integral - s_exact_window) / s_exact_window,
            "front_crossing_rel_error": abs(s_local - s_exact) / s_exact,
            # transparency: raw node sum vs the front measured from the wall face
            # (carries a systematic -0.5 dx bookkeeping bias)
            "front_rel_error_raw": abs(s_integral - s_exact) / s_exact,
            "profile_max_rel_error": prof_err_now,
            "liquid_fraction_min": fl_min,
            "liquid_fraction_max": fl_max,
            "temperature_min": float(np.min(sim.T[interior])),
            "temperature_max": float(np.max(sim.T[interior])),
        }
        if energy_audit:
            sensible_change = cp * float(np.sum((sim.T - T_i)[interior]))
            latent_change = latent * float(np.sum((sim.f_l - 1.0)[interior]))
            d_h = sensible_change + latent_change
            budget = model.energy_budget(t)
            # same control volume as the lattice nodes 1..nx-2 -> [0.5, nx-1.5]
            window = model.energy_budget_window(t, 0.5, nx - 1.5)
            entry.update(
                {
                    "wall_heat_input_first_order": q_1st,
                    "wall_heat_input_second_order": q_2nd,
                    "wall_heat_input_right_first_order": q_1st_right,
                    "wall_heat_input_right_second_order": q_2nd_right,
                    "far_field_leakage_share": abs(q_2nd_right) / abs(q_2nd),
                    "wall_heat_input_exact": model.wall_heat_input(t),
                    "enthalpy_change": d_h,
                    "sensible_change": sensible_change,
                    "latent_change": latent_change,
                    "analytic_sensible": window["sensible"],
                    "analytic_latent": window["latent"],
                    "analytic_total": window["total"],
                    "analytic_total_semi_infinite": budget["total"],
                    # C3 (a): LBM enthalpy vs analytic enthalpy on the SAME window
                    "closure_window_total": abs(d_h - window["total"]) / abs(window["total"]),
                    "closure_window_latent": abs(latent_change - window["latent"])
                    / abs(window["latent"]),
                    "closure_window_sensible": abs(sensible_change - window["sensible"])
                    / abs(window["sensible"]),
                    # C3 (b): independent flux operator at the wall vs analytic input
                    "closure_flux_vs_exact": abs(abs(q_2nd) - abs(budget["wall_input"]))
                    / abs(budget["wall_input"]),
                    # transparency only: the wall half-cell is not part of the
                    # interior control volume, so this carries an O(dx)/s artifact
                    "closure_flux_vs_enthalpy": abs(abs(q_2nd) - abs(d_h)) / abs(d_h),
                    "latent_share_of_input": abs(latent_change) / abs(q_2nd),
                    "sensible_share_of_input": abs(sensible_change) / abs(q_2nd),
                }
            )
        series.append(entry)
        last = entry

    t = float(n_steps)
    T_exact = model.temperature(x, t)
    s_exact = model.front(t)
    diff = sim.T[:, 0] - T_exact
    band = np.abs(x - s_exact) <= profile_band
    profile = {
        "max_abs_error": float(np.max(np.abs(diff))),
        "rms_abs_error": float(np.sqrt(np.mean(diff**2))),
        "max_rel_error": float(np.max(np.abs(diff)) / dT_ref),
        "rms_rel_error": float(np.sqrt(np.mean(diff**2)) / dT_ref),
        "max_rel_error_band_excluded": float(np.max(np.abs(diff[~band])) / dT_ref),
        "n_nodes_excluded": int(np.count_nonzero(band)),
        "dT_reference": dT_ref,
        "band_cells": profile_band,
    }

    u_wall = model.wall_heat_rate(t)
    result: dict[str, Any] = {
        "case": "planar_two_phase_stefan",
        "mode": "freezing_superheated_liquid",
        "lattice": lattice,
        "grid": {
            "nx": nx,
            "ny": 1,
            "dx": 1.0,
            "dt": 1.0,
            "front_cells": float(s_end),
            "far_eta": float(nx / (2.0 * math.sqrt(alpha * t))),
        },
        "parameters": {
            "stefan_solid": stefan_solid,
            "stefan_liquid": stefan_liquid,
            "alpha": alpha,
            "tau": tau,
            "cp": cp,
            "latent": latent,
            "t_freeze": t_freeze,
            "wall_temperature": model.wall_temperature,
            "initial_temperature": T_i,
            "lambda": model.lam,
            "lambda_liquid": model.lam_liquid,
            "front_coefficient": 2.0 * model.lam,
            "steps": n_steps,
            "time_end": t_end,
            "wall_heat_rate_exact": u_wall,
            "interface_flux_solid": model.interface_flux_solid(t),
            "interface_flux_liquid": model.interface_flux_liquid(t),
        },
        "series": series,
        "final": dict(last),
        "profile": {
            "x": x.tolist(),
            "T_numerical": sim.T[:, 0].tolist(),
            "T_exact": T_exact.tolist(),
            "f_l": sim.f_l[:, 0].tolist(),
            "front_exact": s_exact,
            "metrics": profile,
        },
        "liquid_fraction_bounds": {"min": fl_min, "max": fl_max},
        "enthalpy_initial_interior": h_init,
        "interior_cells": n_interior,
    }
    result["criteria"] = case_criteria(result)
    return result


# ---------------------------------------------------------------------------
# acceptance
# ---------------------------------------------------------------------------
def case_criteria(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate C1/C2/C3/C4 of a single case against the predeclared bounds."""
    final = result["final"]
    profile = result["profile"]["metrics"]
    bounds = result["liquid_fraction_bounds"]
    series = result["series"]

    # C1/C2 are evaluated over *every* sampled time (not just the final one):
    # early times have a smaller front and therefore a relatively coarser
    # smeared interface, so the worst case is genuinely informative.
    front_err = max(
        max(e["front_window_rel_error"], e["front_crossing_rel_error"]) for e in series
    )
    profile_err = max(e["profile_max_rel_error"] for e in series)
    energy_err = None
    if "closure_window_total" in final:
        energy_err = max(
            max(e["closure_window_total"], e["closure_flux_vs_exact"]) for e in series
        )
    out = {
        "C1_front_error": float(front_err),
        "C1_front_error_final": float(
            max(final["front_window_rel_error"], final["front_crossing_rel_error"])
        ),
        "C1_front_error_phase_integral_window": float(final["front_window_rel_error"]),
        "C1_front_error_half_crossing": float(final["front_crossing_rel_error"]),
        "C1_front_error_raw": float(final["front_rel_error_raw"]),
        "C1_n_samples": len(series),
        "C1_pass": bool(front_err < FRONT_TOL),
        "C2_profile_max_rel_error": float(profile_err),
        "C2_profile_max_rel_error_final": float(profile["max_rel_error"]),
        "C2_profile_max_rel_error_band_excluded": float(profile["max_rel_error_band_excluded"]),
        "C2_pass": bool(profile_err < PROFILE_TOL),
        "C4_liquid_fraction_min": float(bounds["min"]),
        "C4_liquid_fraction_max": float(bounds["max"]),
        "C4_pass": bool(bounds["min"] >= -1e-12 and bounds["max"] <= 1.0 + 1e-12),
    }
    if energy_err is not None:
        out.update(
            {
                "C3_energy_error": float(energy_err),
                "C3_energy_error_final": float(
                    max(final["closure_window_total"], final["closure_flux_vs_exact"])
                ),
                "C3_energy_error_window_total": float(final["closure_window_total"]),
                "C3_energy_error_window_latent": float(final["closure_window_latent"]),
                "C3_energy_error_window_sensible": float(final["closure_window_sensible"]),
                "C3_energy_error_flux_vs_exact": float(final["closure_flux_vs_exact"]),
                "C3_transparency_flux_vs_enthalpy": float(final["closure_flux_vs_enthalpy"]),
                "C3_pass": bool(energy_err < ENERGY_TOL),
            }
        )
    out["tol"] = {"C1": FRONT_TOL, "C2": PROFILE_TOL, "C3": ENERGY_TOL}
    return out


# ---------------------------------------------------------------------------
# sweep / convergence drivers
# ---------------------------------------------------------------------------
def run_two_phase_sweep(
    stefan_numbers: Sequence[float] = (0.1, 0.5, 1.0),
    stefan_liquid: float = 0.2,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run the two-phase case for each ``Ste_s`` in ``stefan_numbers``."""
    return [
        run_two_phase_stefan(ste, stefan_liquid, **kwargs) for ste in stefan_numbers
    ]


def run_two_phase_grid_convergence(
    stefan_solid: float,
    stefan_liquid: float = 0.2,
    front_cells: Sequence[float] = (17.0, 33.0, 66.0),
    **kwargs: Any,
) -> dict[str, Any]:
    """Front/profile error versus resolution plus the fitted observed order."""
    results = [
        run_two_phase_stefan(stefan_solid, stefan_liquid, front_cells=s, **kwargs)
        for s in front_cells
    ]
    for s, res in zip(front_cells, results):
        res["requested_front_cells"] = float(s)

    xs = np.asarray([r["grid"]["front_cells"] for r in results], dtype=float)
    out: dict[str, Any] = {"cases": results, "front_cells": xs.tolist()}

    def fit(ys: np.ndarray, key: str) -> dict[str, float]:
        ys = np.asarray(ys, dtype=float)
        good = ys > 0.0
        if np.count_nonzero(good) < 2:
            return {"order": float("nan"), "n_points": int(np.count_nonzero(good))}
        p, log_c = np.polyfit(np.log(xs[good]), np.log(ys[good]), 1)
        return {
            "order": float(-p),
            "constant": float(math.exp(log_c)),
            "n_points": int(np.count_nonzero(good)),
            "errors": ys.tolist(),
        }

    out["front_order"] = fit(
        np.asarray([r["final"]["front_window_rel_error"] for r in results]), "front"
    )
    out["front_order_raw"] = fit(
        np.asarray([r["final"]["front_rel_error_raw"] for r in results]), "front-raw"
    )
    out["front_order_crossing"] = fit(
        np.asarray([r["final"]["front_crossing_rel_error"] for r in results]), "crossing"
    )
    out["profile_order"] = fit(
        np.asarray([r["profile"]["metrics"]["max_rel_error"] for r in results]), "profile"
    )
    # worst case over *all* sampled times (the C1/C2 acceptance quantity)
    out["front_order_series_max"] = fit(
        np.asarray([r["criteria"]["C1_front_error"] for r in results]), "front-series-max"
    )
    out["profile_order_series_max"] = fit(
        np.asarray([r["criteria"]["C2_profile_max_rel_error"] for r in results]),
        "profile-series-max",
    )
    errs = [r["final"]["front_window_rel_error"] for r in results]
    out["monotone"] = bool(all(b < a for a, b in zip(errs[:-1], errs[1:])))
    out["C5_pass"] = bool(
        out["monotone"]
        and out["front_order"]["order"] >= 1.0
        and len(results) >= 3
    )
    out["tol"] = {"C5_min_order": 1.0}
    return out