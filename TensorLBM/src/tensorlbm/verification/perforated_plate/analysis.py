"""Analysis of perforated-plate oscillatory-flow signals.

Conventions
-----------
All quantities are in lattice units (rho = 1, dt = 1).

* ``Q(t)``  : volume flow rate per unit depth, ``Q = int u_x dy`` [L^2/T]
* ``u_h(t)``: mean velocity through the holes, ``u_h = Q / (A_ch * beta)`` [L/T]
* ``A_ch``  : channel cross-section, ``A_ch = ny_fluid`` (2-D, unit depth)

The unsteady pressure drop across a sharp-edged perforated plate is modelled as

    dp(t) = rho * K_m * du_h/dt + 0.5 * rho * K_q * u_h |u_h|                (1)

with ``K_m`` an inertial length [L] (``C_m = K_m / t_plate`` the added-mass
coefficient) and ``K_q = 1 / C_d**2`` the quadratic loss coefficient.  For a
harmonic flow ``u_h = u0 sin(theta)`` the two basis functions of (1) are
orthogonal over a period, so ``K_m`` and ``K_q`` follow from a linear
least-squares projection of the phase-averaged signals.

Energy dissipation per cycle is

    Phi = -oint dp dQ = oint dp * Q dt      (with the sign convention above dp>0 for Q>0)

and the equivalent sloshing damping ratio

    zeta_eff = Phi / (2 pi omega * 2 E_kin).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Phase averaging and harmonic decomposition
# ---------------------------------------------------------------------------

def phase_average(time: np.ndarray, signal: np.ndarray, omega: float,
                  n_bins: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Bin a periodic signal on the oscillation phase.

    Parameters
    ----------
    time : array_like
        Lattice time of the samples.
    signal : array_like
        Sampled quantity.
    omega : float
        Angular frequency.
    n_bins : int
        Number of phase bins.

    Returns
    -------
    theta : (n_bins,) array
        Bin-centre phases in [0, 2 pi).
    avg : (n_bins,) array
        Phase-averaged signal (NaN-safe mean, empty bins are interpolated).
    """
    time = np.asarray(time, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if time.size == 0:
        return np.zeros(n_bins), np.zeros(n_bins)

    theta = np.mod(omega * time, 2.0 * math.pi)
    edges = np.linspace(0.0, 2.0 * math.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)

    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(sums, idx, signal)
    np.add.at(counts, idx, 1.0)
    avg = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)

    if np.isnan(avg).any():
        good = ~np.isnan(avg)
        if good.sum() == 0:
            avg = np.zeros(n_bins)
        else:
            avg = np.interp(np.arange(n_bins), np.flatnonzero(good), avg[good],
                            period=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, avg


def first_harmonic(theta: np.ndarray, signal: np.ndarray) -> complex:
    """Complex amplitude of the fundamental (``signal ~ Re{A exp(i theta)}``)."""
    theta = np.asarray(theta, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    n = theta.size
    if n == 0:
        return 0.0 + 0.0j
    return (2.0 / n) * np.sum(signal * np.exp(-1j * theta))


def harmonic_amplitude(signal: np.ndarray) -> float:
    """Amplitude of the fundamental of a uniformly sampled periodic record."""
    s = np.asarray(signal, dtype=np.float64)
    n = s.size
    if n == 0:
        return 0.0
    spec = np.fft.rfft(s - s.mean())
    return float(2.0 * np.abs(spec[1]) / n) if spec.size > 1 else 0.0


# ---------------------------------------------------------------------------
# Coefficient extraction
# ---------------------------------------------------------------------------

def extract_cd_cm(
    phase_q: np.ndarray,
    phase_dp: np.ndarray,
    omega: float,
    beta: float,
    ny: int,
    rho0: float = 1.0,
    plate_thickness: float = 2.0,
) -> dict:
    """Least-squares fit of ``dp = rho K_m du_h/dt + 0.5 rho K_q u_h|u_h|``.

    Returns
    -------
    dict with ``C_d``, ``C_m``, ``K_m``, ``K_q``, ``r_squared``,
    ``u_h_amp`` and the fitted amplitudes of both terms.
    """
    q = np.asarray(phase_q, dtype=np.float64)
    dp = np.asarray(phase_dp, dtype=np.float64)
    n = q.size
    if n < 4:
        return {"C_d": float("nan"), "C_m": float("nan"), "K_m": float("nan"),
                "K_q": float("nan"), "r_squared": 0.0, "u_h_amp": 0.0,
                "ddp_inertial_amp": 0.0, "ddp_loss_amp": 0.0}

    area = max(ny - 2, 1)                      # fluid channel height
    # the model has no DC component (both basis functions are mean free), so any
    # residual start-up mean flow is removed before the projection
    q = q - q.mean()
    dp = dp - dp.mean()
    u_h = q / (area * max(beta, 1e-12))

    # d/dt with the period known analytically
    dt = 2.0 * math.pi / omega / n
    du_h_dt = np.gradient(u_h, dt)

    inertia = rho0 * du_h_dt                   # rho K_m du_h/dt  (K_m = coeff)
    loss = 0.5 * rho0 * u_h * np.abs(u_h)      # 0.5 rho K_q u|u|  (K_q = coeff)

    X = np.column_stack([inertia, loss])
    try:
        coeffs, *_ = np.linalg.lstsq(X, dp, rcond=None)
    except np.linalg.LinAlgError:              # pragma: no cover
        return {"C_d": float("nan"), "C_m": float("nan"), "K_m": float("nan"),
                "K_q": float("nan"), "r_squared": 0.0, "u_h_amp": 0.0,
                "ddp_inertial_amp": 0.0, "ddp_loss_amp": 0.0}

    K_m, K_q = float(coeffs[0]), float(coeffs[1])
    cd = 1.0 / math.sqrt(K_q) if K_q > 0 else float("nan")
    cm = K_m / plate_thickness if plate_thickness > 0 else float("nan")

    pred = X @ coeffs
    ss_res = float(np.sum((dp - pred) ** 2))
    ss_tot = float(np.sum((dp - dp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "C_d": cd,
        "C_m": cm,
        "K_m": K_m,
        "K_q": K_q,
        "r_squared": r2,
        "u_h_amp": float(np.max(np.abs(u_h))),
        "ddp_inertial_amp": float(np.max(np.abs(K_m * inertia))),
        "ddp_loss_amp": float(np.max(np.abs(K_q * loss))),
    }


def compute_phase_lag(phase_q: np.ndarray, phase_dp: np.ndarray,
                      omega: float) -> dict:
    """Phase by which ``dp`` leads ``Q`` (degrees), from the fundamental harmonic.

    Low KC  -> inertia dominated -> ~ +90 deg
    High KC -> loss dominated    -> ~   0 deg
    """
    q = np.asarray(phase_q, dtype=np.float64)
    dp = np.asarray(phase_dp, dtype=np.float64)
    n = q.size
    if n < 4:
        return {"phase_lag_deg": float("nan"), "correlation": 0.0,
                "q_amp": 0.0, "dp_amp": 0.0}

    theta = 2.0 * math.pi * np.arange(n) / n
    a_q = first_harmonic(theta, q - q.mean())
    a_dp = first_harmonic(theta, dp - dp.mean())
    lag = math.degrees(math.atan2((a_dp * np.conj(a_q)).imag,
                                  (a_dp * np.conj(a_q)).real))
    if abs(a_q) > 0 and abs(a_dp) > 0:
        corr = float((a_dp * np.conj(a_q)).real / (abs(a_dp) * abs(a_q)))
    else:
        corr = 0.0
    return {
        "phase_lag_deg": lag,
        "correlation": corr,
        "q_amp": float(abs(a_q)),
        "dp_amp": float(abs(a_dp)),
    }


def compute_energy_dissipation(phase_q: np.ndarray, phase_dp: np.ndarray,
                               omega: float) -> float:
    """Energy dissipated per cycle ``Phi = oint dp * Q dt`` (should be > 0)."""
    q = np.asarray(phase_q, dtype=np.float64)
    dp = np.asarray(phase_dp, dtype=np.float64)
    n = q.size
    if n < 2:
        return float("nan")
    dt = 2.0 * math.pi / omega / n
    phi = float(np.sum(dp * q) * dt)
    return phi


def compute_zeta_eff(phi: float, omega: float, e_kin: float) -> float:
    """Equivalent damping ratio ``zeta_eff = Phi / (2 pi omega * 2 E_kin)``."""
    denom = 2.0 * math.pi * omega * 2.0 * max(abs(e_kin), 1e-30)
    return float(phi / denom)


def kinetic_energy(snapshots: list[dict], rho0: float = 1.0) -> float:
    """Cycle-mean kinetic energy per unit depth from stored velocity snapshots."""
    if not snapshots:
        return 0.0
    vals = []
    for snap in snapshots:
        vals.append(0.5 * rho0 * float(np.sum(snap["ux"] ** 2 + snap["uy"] ** 2)))
    return float(np.mean(vals))


def analytic_dissipation(omega: float, area: float, beta: float, C_d: float,
                         u_h_amp: float) -> float:
    """Quasi-steady (quadratic-damping) cycle dissipation for a harmonic flow.

    ``Phi = 0.5 rho K_q / (A beta)^2 * int |Q|^3 dt`` with
    ``int_0^T |sin|^3 dt = 4T/(3 pi)``.
    """
    if not (C_d > 0) or u_h_amp <= 0:
        return float("nan")
    T = 2.0 * math.pi / omega
    K_q = 1.0 / C_d ** 2
    q_amp = u_h_amp * area * beta
    return 0.5 * K_q / (area * beta) ** 2 * q_amp ** 3 * (4.0 * T / (3.0 * math.pi))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze_simulation(result: dict, omega: float, beta: float, ny: int,
                       n_phase_bins: int = 48,
                       plate_thickness: float = 2.0) -> dict:
    """Run the full analysis chain on a solver result dictionary."""
    ts = result["time_series"]
    time = ts["time"]
    q = ts["flow_rate"]
    dp_force = ts["dp_plate"]
    dp_stat = ts["dp_plate_stations"]
    ny_fluid = result.get("ny_fluid", ny - 2)
    area = max(ny_fluid, 1)

    theta, q_ph = phase_average(time, q, omega, n_phase_bins)
    _, dpf_ph = phase_average(time, dp_force, omega, n_phase_bins)
    _, dps_ph = phase_average(time, dp_stat, omega, n_phase_bins)

    fit_force = extract_cd_cm(q_ph, dpf_ph, omega, beta, ny, 1.0, plate_thickness)
    fit_stat = extract_cd_cm(q_ph, dps_ph, omega, beta, ny, 1.0, plate_thickness)
    lag_force = compute_phase_lag(q_ph, dpf_ph, omega)
    lag_stat = compute_phase_lag(q_ph, dps_ph, omega)

    phi_force = compute_energy_dissipation(q_ph, dpf_ph, omega)
    phi_stat = compute_energy_dissipation(q_ph, dps_ph, omega)
    phi_model = analytic_dissipation(omega, area, beta, fit_force["C_d"],
                                     fit_force["u_h_amp"])

    e_kin = kinetic_energy(result.get("snapshots", []))

    # stationarity: relative change of the |Q| amplitude over the last cycles
    cyc = np.asarray(result.get("cycle_amplitudes", []), dtype=np.float64)
    pre = None
    if cyc.size >= 2 and cyc[-2] > 0:
        pre = float(abs(cyc[-1] - cyc[-2]) / cyc[-2])

    return {
        "theta": theta,
        "phase_q": q_ph,
        "phase_dp": dpf_ph,
        "phase_dp_stations": dps_ph,
        "C_d": fit_force["C_d"],
        "C_m": fit_force["C_m"],
        "K_m": fit_force["K_m"],
        "K_q": fit_force["K_q"],
        "r_squared": fit_force["r_squared"],
        "C_d_stations": fit_stat["C_d"],
        "C_m_stations": fit_stat["C_m"],
        "r_squared_stations": fit_stat["r_squared"],
        "u_h_amp": fit_force["u_h_amp"],
        "q_amp": float(np.max(np.abs(q_ph))),
        "phase_lag_deg": lag_force["phase_lag_deg"],
        "phase_lag_deg_stations": lag_stat["phase_lag_deg"],
        "dp_inertial_amp": fit_force["ddp_inertial_amp"],
        "dp_loss_amp": fit_force["ddp_loss_amp"],
        "phi": phi_force,
        "phi_stations": phi_stat,
        "phi_model": phi_model,
        "e_kin": e_kin,
        "zeta_eff": compute_zeta_eff(phi_force, omega, e_kin),
        "zeta_eff_stations": compute_zeta_eff(phi_stat, omega, e_kin),
        "zeta_model": compute_zeta_eff(phi_model, omega, e_kin),
        "cycle_amplitudes": cyc.tolist(),
        "stationarity": pre,
    }