"""Acceptance metrics for the 2-D numerical wave tank (NWT).

Every metric in :func:`evaluate_case` is a *number computed from recorded probe
time series*, compared against a closed-form reference
(:class:`tensorlbm.nwt.stokes2.Stokes2Wave`).  No fitted constant and no manual
value is involved.

Criteria (mission IN33-N02):

===========  =============================================================
C1           dispersion: measured phase speed vs ``sqrt((g/k) tanh(kh))``
             at the *measured* ``k``; |error| < 1 % for kh in {0.7, 1.5, 3.0}
C2           waveform: first-harmonic amplitude |error| < 3 %, second
             harmonic |error| < 10 %, at ``eps = ka`` in {0.02, 0.08}
C3           absorption: Goda-Suzuki two-point reflection coefficient < 5 %
             with a >= 1-wavelength relaxation zone; negative control with
             the absorber disabled must exceed 30 %
C4           mass: relative liquid-inventory drift < 0.5 % per period over
             5 periods
C5           grid: halving ``dx`` changes the first-harmonic amplitude by
             < 3 %
C6           indexed in benchmarks/verified + existing pytest green
C7           one case < 20 min on loong64
===========  =============================================================
"""

from __future__ import annotations

import math

import tensorlbm.nwt as nwt

__all__ = ["evaluate_case", "harmonic_row"]

SPEED_OF_SOUND_SQ = 1.0 / 3.0


def harmonic_row(result, probe: int) -> dict:
    h = nwt.fit_harmonics(result.eta_probes[probe], 1.0, result.config.omega)
    h["probe_x"] = result.probe_x[probe]
    return h


def _attenuation(rows, omega: float, cfg) -> dict:
    """Log-linear fit of the raw fundamental amplitude along the probe array.

    ``A1(x) = A1(0) exp(-x / L_d)`` gives the decay length in cells; the
    equivalent numerical viscosity follows from the linear damping rate
    ``2 nu k^2 = 1 / L_d`` and is reported both in lattice units and as the ratio
    to the molecular value implied by ``tau``.  This is the headline number for
    how well the free-surface model transports a gravity wave.
    """
    import numpy as _np

    xs = _np.asarray([r["probe_x"] for r in rows], dtype=float)
    a = _np.asarray([r["A1"] for r in rows], dtype=float)
    good = a > 1e-6
    out = {"probe_x": xs.tolist(), "A1_raw": a.tolist()}
    if good.sum() < 2:
        out["note"] = "insufficient usable probes"
        return out
    slope, intercept = _np.polyfit(xs[good], _np.log(a[good]), 1)
    decay_length = -1.0 / slope if slope < 0 else float("inf")
    k = cfg.wavenumber
    nu_eff = (1.0 / decay_length) / (2.0 * k * k) if decay_length != float("inf") else float("inf")
    nu_mol = (cfg.tau - 0.5) / 3.0
    out.update(
        {
            "log_slope_per_cell": float(slope),
            "decay_length_cells": float(decay_length),
            "decay_per_wavelength_neper": float(-slope * cfg.wavelength),
            "equivalent_numerical_viscosity_lattice": float(nu_eff),
            "molecular_viscosity_lattice": float(nu_mol),
            "viscosity_ratio": float(nu_eff / nu_mol) if nu_mol > 0 else float("inf"),
            "A1_nearest_probe": float(a[0]),
            "A1_nearest_probe_error": float(abs(a[0] - cfg.wave.amplitude) / cfg.wave.amplitude),
        }
    )
    return out


def evaluate_case(result) -> dict:
    """Compute all C1-C4/C7 observables for one finished NWT run."""
    cfg = result.config
    wave = result.wave
    omega = cfg.omega
    eta = result.eta_probes
    xs = result.probe_x
    n_probe = eta.shape[0]

    # ---- C1 dispersion ---------------------------------------------------
    # Two independent estimators.  Primary: fit the incident+reflected model to
    # ALL probes and minimize the residual over k (immune to the standing-wave
    # distortion that biases a raw phase regression).  Secondary: the phase
    # regression over the probe array, reported for cross-validation.
    phase_est = nwt.wavenumber_from_probes([eta[j] for j in range(n_probe)], xs, 1.0, omega)
    fit1 = nwt.wavenumber_by_model_fit(
        [eta[j] for j in range(n_probe)], xs, 1.0, omega, k_guess=abs(phase_est["k_measured"])
    )
    k_meas = abs(fit1["k_measured"])
    c_meas = omega / k_meas
    c_closed = math.sqrt(cfg.gravity / k_meas * math.tanh(k_meas * cfg.depth))
    cc = nwt.lag_from_cross_correlation(eta[0], eta[-1], 1.0)
    c_cc = (xs[-1] - xs[0]) / cc["lag_time"] if cc["lag_time"] else float("inf")

    # ---- C2 waveform -----------------------------------------------------
    # The amplitude compared with the analytic incident wave is the *separated
    # incident* amplitude (raw per-probe amplitudes carry a +-10 % standing-wave
    # modulation for a 5 % reflection), at omega (k) and at 2 omega (wave number
    # 2k: the second-order Stokes harmonic is a bound wave with spatial structure
    # cos(2kx - 2 omega t)).
    rows = [harmonic_row(result, j) for j in range(n_probe)]
    a1_raw = sum(r["A1"] for r in rows) / n_probe
    a2_raw = sum(r["A2"] for r in rows) / n_probe
    a2_ref = abs(wave.second_harmonic_amplitude)

    fit2 = nwt.wavenumber_by_model_fit(
        [eta[j] for j in range(n_probe)], xs, 1.0, 2.0 * omega, k_guess=2.0 * k_meas
    )
    a1 = fit1["a_incident"]
    a2 = fit2["a_incident"]

    # ---- C3 reflection ---------------------------------------------------
    tp = nwt.two_point_reflection(eta[0], eta[1], xs[0], xs[1], 1.0, omega, k_meas)
    lsq = nwt.least_squares_reflection([eta[j] for j in range(n_probe)], xs, 1.0, omega, k_meas)
    tp2 = nwt.two_point_reflection(eta[0], eta[1], xs[0], xs[1], 1.0, 2.0 * omega, 2.0 * k_meas)

    # ---- C4 mass ---------------------------------------------------------
    drift = nwt.inventory_drift(result.inventory, int(cfg.steps_per_period))

    return {
        "case": {
            "kh": wave.kh,
            "steepness": cfg.steepness,
            "wavelength": cfg.wavelength,
            "depth": cfg.depth,
            "amplitude_a": wave.amplitude,
            "second_harmonic_amplitude": wave.second_harmonic_amplitude,
            "period_steps": cfg.steps_per_period,
            "tau": cfg.tau,
            "sponge_enabled": cfg.sponge_enabled,
            "generator_wavelengths": cfg.generator_wavelengths,
            "sponge_wavelengths": cfg.sponge_wavelengths,
            "nx": result.geometry["nx"],
            "ny": result.geometry["ny"],
            "cells": result.geometry["nx"] * result.geometry["ny"],
            "total_steps": result.steps,
            "hydrostatic_compressibility_gh_cs2": cfg.gravity * cfg.depth / SPEED_OF_SOUND_SQ,
            "max_surface_velocity_ma": wave.horizontal_velocity_amplitude_at_swl(),
        },
        # C1
        "dispersion": {
            "k_measured": k_meas,
            "k_analytic": cfg.wavenumber,
            "k_from_phase_regression": abs(phase_est["k_measured"]),
            "model_fit_residual": fit1["residual"],
            "model_fit_residual_at_phase_guess": fit1["residual_at_guess"],
            "k_spread_over_probe_pairs": phase_est["k_spread"],
            "wavelength_measured": 2.0 * math.pi / k_meas,
            "phase_speed_measured": c_meas,
            "phase_speed_closed_form_at_k_measured": c_closed,
            "relative_error": abs(c_meas - c_closed) / c_closed,
            "cross_correlation_lag_steps": cc["lag_samples"],
            "cross_correlation_peak": cc["peak_correlation"],
            "phase_speed_cross_correlation": c_cc,
            "phase_speed_analytic_target": wave.phase_speed,
        },
        # derived: spatial attenuation of the fundamental along the probe array,
        # expressed as an equivalent numerical viscosity 2 nu k^2 = -d(ln A1)/dx
        "attenuation": _attenuation(rows, omega, cfg),
        # C2
        "waveform": {
            "A1_measured_mean": a1,
            "A1_analytic": wave.amplitude,
            "A1_relative_error": abs(a1 - wave.amplitude) / wave.amplitude,
            "A2_measured_mean": a2,
            "A2_analytic": a2_ref,
            "A2_relative_error": abs(a2 - a2_ref) / a2_ref if a2_ref > 0 else float("inf"),
            "A1_raw_probe_mean": a1_raw,
            "A2_raw_probe_mean": a2_raw,
            "A1_incident_two_point": tp["a_incident"],
            "A1_source": "least-squares incident+reflected model fit over all probes at omega",
            "A2_source": "least-squares incident+reflected model fit over all probes at 2*omega",
            "A2_model_fit_residual": fit2["residual"],
            "A2_incident_two_point": tp2["a_incident"],
            "A2_separation_conditions": {"two_point": tp2["condition"]},
            "A1_per_probe": [r["A1"] for r in rows],
            "A2_per_probe": [r["A2"] for r in rows],
            "non_harmonic_rms": [r["residual_rms"] for r in rows],
            "mean_level_offset": [r["mean_level"] for r in rows],
        },
        # C3
        "reflection": {
            "KR_two_point": tp["KR"],
            "KR_two_point_condition": tp["condition"],
            "a_incident_two_point": tp["a_incident"],
            "a_reflected_two_point": tp["a_reflected"],
            "KR_least_squares": lsq["KR"],
            "KR_model_fit": fit1["KR"],
            "lsq_max_residual": lsq["max_residual"],
        },
        # C4
        "mass": drift,
        "runtime": {
            "wall_seconds": result.wall_seconds,
            "wall_minutes": result.wall_seconds / 60.0,
            "steps": result.steps,
            "million_cell_updates_per_second": (
                result.steps * result.geometry["nx"] * result.geometry["ny"] / result.wall_seconds / 1e6
                if result.wall_seconds > 0
                else None
            ),
        },
        "probe_x": xs,
        "diagnostics": result.diagnostics,
    }