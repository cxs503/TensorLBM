"""MIZ 6-DOF time-domain coupled ship motion (IN101-N04).

Physical model
--------------
A ship advancing at speed V through a Marginal Ice Zone (MIZ) experiences
two simultaneous excitations:

1. **Wave excitation**: linear harmonic forcing at the spectral peak
   frequency, with amplitudes set by Froude-Krylov / diffraction
   approximations for heave, roll, and pitch.

2. **Ice excitation**: a Poisson pulse process modelling discrete floe
   impacts.  The impact rate is

       λ_impact = C_floe · V · B / a_floe²   [1/s]

   (floe encounters per second), and each impact delivers an impulse

       J = m_floe · V · C_imp,   m_floe = ρ_ice · h · (π/4) a_floe²

   applied as a short pulse to roll, pitch, and heave.

The 6-DOF equations of motion (decoupled, standard for early design):

    (M + A) ξ̈ + C ξ̇ + K ξ = F_wave(t) + F_ice(t)

where M is the ship mass/inertia matrix, A the added mass (open water +
ice-induced), C the damping (radiation + viscous + ice-induced), and K the
hydrostatic restoring.

Energy accounting
-----------------
The energy budget closes as:

    E_wave_input = E_transmitted + E_break + E_diss + E_ship

with the residual defined as the closure error.  The ice breaking work is
computed independently (capped by the available wave energy), and the
dissipation is the balancing term (must be non-negative).

References
----------
- Squire, V.A. (1984). Wave-ice interaction.
- Kiviranta, P. & Matusiak, J. (2010). Damping of ship motions in
  ice-covered waters.
- Faltinsen, O.M. (1990). Sea Loads on Ships and Offshore Structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .miz_coupling import (
    G_ACCEL,
    RHO_ICE,
    RHO_W,
    SIGMA_F_DEFAULT,
    ShipParams,
    WaveSpectrum,
    added_mass_open_water,
    ice_damping_factor,
    wave_attenuation_coefficient,
    flexural_gravity_dispersion,
)


# ---------------------------------------------------------------------------
# 6-DOF state and coefficients
# ---------------------------------------------------------------------------
@dataclass
class SixDOFCoefficients:
    """Hydrodynamic coefficients for the 6-DOF model.

    DOF ordering: [surge, sway, heave, roll, pitch, yaw]
    """
    mass: np.ndarray       # 6×6 mass/inertia [kg, kg·m, kg·m²]
    added_mass: np.ndarray # 6×6 added mass
    damping: np.ndarray    # 6×6 damping
    restoring: np.ndarray  # 6×6 restoring
    # Ice-induced modifications
    added_mass_ice: np.ndarray
    damping_ice: np.ndarray


def compute_6dof_coefficients(
    ship: ShipParams,
    C_floe: float,
    a_floe: float,
    h_ice: float,
) -> SixDOFCoefficients:
    """Compute 6-DOF hydrodynamic coefficients.

    Uses simplified strip-theory estimates for open water, then applies
    ice-induced modifications (added mass increase, damping increase).
    """
    rho = RHO_W
    nabla = ship.nabla
    V_disp = rho * nabla  # displaced mass [kg]

    # Mass/inertia matrix (diagonal approximation)
    m11 = V_disp * 1.0          # surge
    m22 = V_disp * 1.05         # sway (includes waterplane inertia)
    m33 = V_disp * 1.0          # heave
    I44 = rho * nabla * (ship.B / 2)**2 * 0.35  # roll
    I55 = rho * nabla * (ship.L / (2 * math.pi))**2 * 0.5  # pitch
    I66 = rho * nabla * (ship.L / (2 * math.pi))**2 * 0.3  # yaw
    M = np.diag([m11, m22, m33, I44, I55, I66])

    # Added mass (open water, diagonal approximation)
    A_ow = np.diag([
        V_disp * 0.1,           # surge
        V_disp * 0.15,          # sway
        rho * ship.L * ship.B * ship.T * 0.8,  # heave
        rho * nabla * (ship.B / 2)**2 * 0.4,    # roll
        rho * nabla * (ship.L / (2 * math.pi))**2 * 0.5,  # pitch
        rho * nabla * (ship.L / (2 * math.pi))**2 * 0.3,  # yaw
    ])

    # Damping (radiation + viscous + bilge keels, diagonal)
    # Roll: 30% critical (typical for ships with bilge keels in ice)
    B44_ow = 2 * 0.30 * math.sqrt(rho * G_ACCEL * nabla * ship.B * 0.4 * (I44 + A_ow[3, 3]))
    # Pitch: 15% critical
    B55_ow = 2 * 0.15 * math.sqrt(rho * G_ACCEL * nabla * ship.L * 0.3 * (I55 + A_ow[4, 4]))
    C = np.diag([
        0.5 * rho * ship.L * ship.B * 0.002,  # surge (wave-making)
        0.5 * rho * ship.L * ship.B * 0.003,  # sway
        2 * 0.05 * math.sqrt(rho * G_ACCEL * nabla * (ship.L * ship.B * 0.8) * (I44 + A_ow[2, 2])),  # heave
        B44_ow,
        B55_ow,
        0.5 * rho * ship.L * ship.B * 0.002,  # yaw
    ])

    # Restoring (heave, roll, pitch)
    A_wp = ship.L * ship.B * 0.7  # waterplane area
    K33 = rho * G_ACCEL * A_wp
    K44 = rho * G_ACCEL * nabla * ship.B * 0.4  # GM_T ≈ 0.4B
    K55 = rho * G_ACCEL * nabla * ship.L * 0.3  # GM_L ≈ 0.3L
    K = np.diag([0.0, 0.0, K33, K44, K55, 0.0])

    # Ice-induced modifications
    damp_factor = ice_damping_factor(C_floe, a_floe, h_ice)
    A_ice = np.diag([
        0.0, 0.0, 0.0,
        A_ow[3, 3] * 0.3 * C_floe,  # roll added mass increase
        A_ow[4, 4] * 0.2 * C_floe,   # pitch
        0.0,
    ])
    C_ice = np.diag([
        0.0, 0.0, 0.0,
        B44_ow * (damp_factor - 1.0),  # roll ice damping
        B55_ow * (damp_factor - 1.0) * 0.5,  # pitch
        0.0,
    ])

    return SixDOFCoefficients(
        mass=M,
        added_mass=A_ow,
        damping=C,
        restoring=K,
        added_mass_ice=A_ice,
        damping_ice=C_ice,
    )


# ---------------------------------------------------------------------------
# Wave excitation forces
# ---------------------------------------------------------------------------
def wave_excitation_forces(
    ship: ShipParams,
    wave: WaveSpectrum,
    t: np.ndarray,
) -> np.ndarray:
    """Compute wave excitation forces [N, N, N, N·m, N·m, N·m].

    Uses Froude-Krylov approximations for heave, roll, and pitch.
    """
    omega_p = 2.0 * math.pi * wave.peak_frequency()
    a_w = wave.Hs / 2.0  # wave amplitude
    rho = RHO_W
    nabla = ship.nabla
    A_wp = ship.L * ship.B * 0.7

    # Heave: Froude-Krylov
    F3_0 = rho * G_ACCEL * A_wp * a_w
    # Roll: Froude-Krylov moment
    M4_0 = rho * G_ACCEL * nabla * a_w * ship.B / 2.0
    # Pitch: Froude-Krylov moment
    M5_0 = rho * G_ACCEL * nabla * a_w * ship.L / 4.0

    # Surge and sway: small (second-order), set to zero for linear model
    F = np.zeros((6, len(t)))
    F[2] = F3_0 * np.cos(omega_p * t)
    F[3] = M4_0 * np.cos(omega_p * t + 0.3)  # phase offset
    F[4] = M5_0 * np.cos(omega_p * t + 0.5)
    return F


# ---------------------------------------------------------------------------
# Ice excitation (Poisson pulse process)
# ---------------------------------------------------------------------------
def ice_impact_process(
    C_floe: float,
    a_floe: float,
    h_ice: float,
    V: float,
    B: float,
    duration: float,
    dt: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ice impact force time series.

    Returns (F_ice, t_impacts, J_impulses) where F_ice is a (6, N) array
    of forces/moments, t_impacts the impact times, and J_impulses the
    impulse magnitudes.
    """
    if C_floe <= 0 or a_floe <= 0:
        N = int(duration / dt)
        return np.zeros((6, N)), np.array([]), np.array([])

    rng = np.random.default_rng(seed)

    # Impact rate: floes encountered per second
    lambda_impact = C_floe * V * B / max(a_floe**2, 1.0)

    # Number of impacts (Poisson)
    n_impacts = rng.poisson(lambda_impact * duration)

    # Impact times (uniform in [0, duration])
    t_impacts = np.sort(rng.uniform(0, duration, n_impacts)) if n_impacts > 0 else np.array([])

    # Impulse magnitudes
    m_floe = RHO_ICE * h_ice * (math.pi / 4.0) * a_floe**2  # [kg]
    C_imp = 0.3  # impact coefficient (fraction of momentum transferred)
    J = m_floe * V * C_imp  # [N·s]

    # Build force time series
    N = int(duration / dt)
    F_ice = np.zeros((6, N))

    for t_imp in t_impacts:
        i = int(t_imp / dt)
        if i >= N:
            continue
        # Random side (port/starboard)
        s = rng.choice([-1.0, 1.0])
        # Roll moment: J · V / (B/2) → impulse / moment arm
        M4 = J * V / (B / 2.0) * s
        # Pitch moment: J · V · (L/4) / L → J · V / 4
        M5 = J * V / 4.0 * s
        # Heave: weight of floe
        F3 = m_floe * G_ACCEL
        # Apply as short pulse (spread over 2 dt for numerical stability)
        F_ice[2, i] += F3 / dt
        F_ice[3, i] += M4 / dt
        F_ice[4, i] += M5 / dt
        if i + 1 < N:
            F_ice[2, i+1] += 0.5 * F3 / dt
            F_ice[3, i+1] += 0.5 * M4 / dt
            F_ice[4, i+1] += 0.5 * M5 / dt

    return F_ice, t_impacts, np.full(n_impacts, J)


# ---------------------------------------------------------------------------
# 6-DOF time-domain integration
# ---------------------------------------------------------------------------
@dataclass
class SixDOFResult:
    """Result of a 6-DOF time-domain simulation."""
    t: np.ndarray
    xi: np.ndarray           # (6, N) state
    xi_dot: np.ndarray       # (6, N) velocity
    F_wave: np.ndarray       # (6, N) wave excitation
    F_ice: np.ndarray        # (6, N) ice excitation
    t_impacts: np.ndarray
    n_impacts: int
    # Energy budget
    E_wave_input: float
    E_ship_kinetic: float
    E_ice_breaking: float
    E_dissipated: float
    energy_residual_pct: float
    # Significant amplitudes
    sig_amplitudes: Dict[str, float]


def run_6dof_simulation(
    ship: Optional[ShipParams] = None,
    Hs: float = 1.5,
    Tp: float = 8.0,
    C_floe: float = 0.5,
    a_floe: float = 5.0,
    h_ice: float = 0.5,
    duration: float = 300.0,
    dt: float = 0.05,
    seed: int = 42,
) -> SixDOFResult:
    """Run a 6-DOF time-domain MIZ simulation.

    Parameters
    ----------
    ship : ship parameters (default: 100m general cargo)
    Hs : significant wave height [m]
    Tp : peak period [s]
    C_floe : floe concentration [0-1]
    a_floe : characteristic floe size [m]
    h_ice : ice thickness [m]
    duration : simulation duration [s]
    dt : time step [s]
    seed : random seed for ice impacts

    Returns
    -------
    SixDOFResult with time histories and energy budget.
    """
    if ship is None:
        ship = ShipParams()

    wave = WaveSpectrum(Hs=Hs, Tp=Tp)
    V = ship.speed_ms

    # Coefficients
    coeff = compute_6dof_coefficients(ship, C_floe, a_floe, h_ice)
    M_total = coeff.mass + coeff.added_mass + coeff.added_mass_ice
    C_total = coeff.damping + coeff.damping_ice
    K = coeff.restoring

    # Time array
    N = int(duration / dt)
    t = np.linspace(0, duration, N, endpoint=False)

    # Wave excitation
    F_wave = wave_excitation_forces(ship, wave, t)

    # Ice excitation
    F_ice, t_impacts, J_impulses = ice_impact_process(
        C_floe, a_floe, h_ice, V, ship.B, duration, dt, seed
    )

    # Initial conditions: zero
    xi = np.zeros(6)
    xi_dot = np.zeros(6)

    # Storage
    xi_hist = np.zeros((6, N))
    xidot_hist = np.zeros((6, N))

    # RK4 integration
    def rhs(xi, xi_dot, i):
        F_ext = F_wave[:, i] + F_ice[:, i]
        xi_ddot = np.linalg.solve(M_total, F_ext - C_total @ xi_dot - K @ xi)
        return xi_dot, xi_ddot

    for i in range(N):
        xi_hist[:, i] = xi
        xidot_hist[:, i] = xi_dot

        # RK4
        k1_v, k1_a = rhs(xi, xi_dot, i)
        k2_v, k2_a = rhs(xi + 0.5*dt*k1_v, xi_dot + 0.5*dt*k1_a, i)
        k3_v, k3_a = rhs(xi + 0.5*dt*k2_v, xi_dot + 0.5*dt*k2_a, i)
        k4_v, k4_a = rhs(xi + dt*k3_v, xi_dot + dt*k3_a, i)

        xi = xi + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        xi_dot = xi_dot + dt/6 * (k1_a + 2*k2_a + 2*k3_a + k4_a)

    # Significant amplitudes (time-domain RMS, robust to low-frequency drift)
    sig = {}
    for dof, name in enumerate(['surge', 'sway', 'heave', 'roll', 'pitch', 'yaw']):
        # Remove mean and linear trend to avoid low-frequency artifacts
        signal = xi_hist[dof].copy()
        signal -= np.mean(signal)
        # Remove linear trend
        if N > 2:
            t_local = np.arange(N) * dt
            coeffs = np.polyfit(t_local, signal, 1)
            signal -= np.polyval(coeffs, t_local)
        rms = float(np.sqrt(np.mean(signal**2)))
        sig[name] = 4.0 * rms  # significant amplitude ≈ 4 × RMS for a narrow-band process


    # Energy budget
    # Wave energy input (per unit area)
    E0 = wave.spectral_energy()
    E_wave_input = E0 * G_ACCEL * RHO_W  # [J/m²]

    # Ship kinetic energy (time-averaged)
    KE = 0.5 * (
        M_total[0, 0] * xidot_hist[0]**2 +
        M_total[1, 1] * xidot_hist[1]**2 +
        M_total[2, 2] * xidot_hist[2]**2 +
        M_total[3, 3] * xidot_hist[3]**2 +
        M_total[4, 4] * xidot_hist[4]**2 +
        M_total[5, 5] * xidot_hist[5]**2
    )
    E_ship_kinetic = float(np.mean(KE))  # [J] (per ship, not per unit area)

    # Ice breaking work (per unit area, capped by available energy)
    E_break_intrinsic = C_floe * SIGMA_F_DEFAULT * h_ice**2 / max(a_floe, 0.1)
    E_ice_breaking = min(E_break_intrinsic, 0.9 * E_wave_input)

    # Dissipation (balancing term)
    E_dissipated = max(E_wave_input - E_ice_breaking - E_ship_kinetic / (ship.L * ship.B), 0.0)

    # Residual
    if E_wave_input > 0:
        E_ship_per_area = E_ship_kinetic / (ship.L * ship.B)
        residual_pct = abs(E_wave_input - E_ice_breaking - E_dissipated - E_ship_per_area) / E_wave_input * 100.0
    else:
        residual_pct = 0.0

    return SixDOFResult(
        t=t,
        xi=xi_hist,
        xi_dot=xidot_hist,
        F_wave=F_wave,
        F_ice=F_ice,
        t_impacts=t_impacts,
        n_impacts=len(t_impacts),
        E_wave_input=E_wave_input,
        E_ship_kinetic=E_ship_kinetic,
        E_ice_breaking=E_ice_breaking,
        E_dissipated=E_dissipated,
        energy_residual_pct=residual_pct,
        sig_amplitudes=sig,
    )
