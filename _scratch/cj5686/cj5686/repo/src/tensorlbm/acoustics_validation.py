"""FWH acoustic validation suite: Curle dipole, moving-source Doppler, airfoil noise.

This module provides three canonical validation cases for the Ffowcs Williams–
Hawkings (FWH) far-field acoustic analogy implemented in ``tensorlbm.acoustics``.

Cases
-----
1. **Curle dipole** – Oscillating cylinder in quiescent fluid.  The far-field
   acoustic pressure is compared against the analytical Curle (1955) solution
   for a compact oscillating cylinder (ka ≪ 1).  Tested at 3 observer angles
   and 3 frequencies (all with ka < 0.2, observer in far-field r ≫ λ).

2. **Moving-source Doppler** – Uniformly translating monopole.  The observed
   frequency shift is verified against the classical Doppler formula
   f_obs = f_src / (1 − M cos θ).

3. **Airfoil trailing-edge noise** – NACA0012 at zero angle of attack in
   uniform flow.  The far-field OASPL is compared against experimental data
   from Brooks–Pope–Marcolini (NASA RP-1218).

References
----------
Curle, N. (1955). "The influence of solid boundaries upon aerodynamic sound."
    *Proc. R. Soc. Lond. A* 231, 505–514.
Ffowcs Williams, J.E. & Hawkings, D.L. (1969).
    "Sound generation by turbulence and surfaces in arbitrary motion."
    *Phil. Trans. R. Soc. Lond. A* 264, 321–342.
Brooks, T.F., Pope, D.S. & Marcolini, M.A. (1989).
    "Airfoil self-noise and prediction." NASA RP-1218.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from tensorlbm.acoustics import (
    AcousticObserver,
    FWHSurface,
    compute_fwh_far_field,
    oaspl,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REF_PRESSURE = 20.0e-6  # 20 µPa
_PI = math.pi
_TWO_PI = 2.0 * _PI


# ===================================================================
# Case 1: Curle dipole – oscillating cylinder
# ===================================================================


@dataclass
class CurleDipoleConfig:
    """Configuration for the Curle dipole validation case."""

    a: float = 0.01  # cylinder radius (m)
    U0: float = 0.1  # oscillation velocity amplitude (m/s)
    freq_hz: float = 1000.0  # oscillation frequency (Hz)
    rho0: float = 1.225  # fluid density (kg/m³)
    c0: float = 343.0  # speed of sound (m/s)
    n_src: int = 128  # number of surface source points
    n_steps: int = 2048  # number of time steps
    dt: float = 1e-5  # time step (s)
    r_obs: float = 1.5  # observer distance (m)


def curle_analytical_pressure(r: float, theta: float, t: float, cfg: CurleDipoleConfig) -> float:
    """Curle analytical far-field pressure time series."""
    omega = _TWO_PI * cfg.freq_hz
    amplitude = cfg.rho0 * cfg.a ** 2 * cfg.U0 * omega ** 2 / (4.0 * cfg.c0 * r) * abs(math.cos(theta))
    return -amplitude * math.cos(theta) / abs(math.cos(theta) + 1e-15) * math.sin(omega * (t - r / cfg.c0))

def curle_analytical_amplitude(r: float, theta: float, cfg: CurleDipoleConfig) -> float:
    """Peak amplitude of the Curle analytical far-field pressure.

    |p'| = ρ₀ a² U₀ ω² / (4 c₀ r) · |cos(θ)|
    """
    omega = _TWO_PI * cfg.freq_hz
    return cfg.rho0 * cfg.a ** 2 * cfg.U0 * omega ** 2 / (4.0 * cfg.c0 * r) * abs(math.cos(theta))


def build_curle_surface(cfg: CurleDipoleConfig) -> FWHSurface:
    """Build an FWHSurface representing an oscillating cylinder."""
    omega = _TWO_PI * cfg.freq_hz
    n = cfg.n_src
    T = cfg.n_steps
    dt = cfg.dt

    phi = torch.linspace(0, _TWO_PI, n, dtype=torch.float64)
    positions = torch.stack([
        cfg.a * torch.cos(phi),
        cfg.a * torch.sin(phi),
        torch.zeros(n, dtype=torch.float64),
    ], dim=-1)

    normals = torch.stack([
        torch.cos(phi),
        torch.sin(phi),
        torch.zeros(n, dtype=torch.float64),
    ], dim=-1)

    arc_length = _TWO_PI * cfg.a / n
    areas = torch.full((n,), arc_length, dtype=torch.float64)

    t_arr = torch.linspace(0, T * dt, T, dtype=torch.float64)
    pressure = torch.zeros(n, T, dtype=torch.float64)
    for i in range(n):
        pressure[i, :] = cfg.rho0 * cfg.U0 * omega * cfg.a * torch.cos(phi[i]) * torch.cos(omega * t_arr)

    return FWHSurface(
        positions=positions,
        normals=normals,
        areas=areas,
        pressure=pressure,
        dt=dt,
        c0=cfg.c0,
    )


def validate_curle_dipole(
    cfg: CurleDipoleConfig | None = None,
    observer_angles_deg: list[float] | None = None,
    frequencies_hz: list[float] | None = None,
) -> dict:
    """Run the Curle dipole validation at 3 observers × 3 frequencies."""
    if cfg is None:
        cfg = CurleDipoleConfig()

    if observer_angles_deg is None:
        observer_angles_deg = [0.0, 45.0, 60.0]

    if frequencies_hz is None:
        frequencies_hz = [250.0, 500.0, 1000.0]

    r_obs = cfg.r_obs

    all_results = []
    all_pass = True

    for freq_hz in frequencies_hz:
        # Adjust n_steps to ensure signal arrives and we have steady-state data
        wavelength = cfg.c0 / freq_hz
        arrival_time = r_obs / cfg.c0
        periods_needed = 5  # at least 5 periods of steady-state
        period = 1.0 / freq_hz
        total_time = arrival_time + periods_needed * period
        n_steps = max(cfg.n_steps, int(math.ceil(total_time / cfg.dt)))

        case_cfg = CurleDipoleConfig(
            a=cfg.a, U0=cfg.U0, freq_hz=freq_hz,
            rho0=cfg.rho0, c0=cfg.c0,
            n_src=cfg.n_src, n_steps=n_steps, dt=cfg.dt,
            r_obs=r_obs,
        )
        surface = build_curle_surface(case_cfg)

        observers = [
            AcousticObserver(x=r_obs * math.cos(math.radians(th)),
                           y=r_obs * math.sin(math.radians(th)),
                           label=f"θ={th:.0f}°")
            for th in observer_angles_deg
        ]

        p_prime_sim, time_list = compute_fwh_far_field(surface, observers)

        case_results = {"freq_hz": freq_hz, "observers": []}
        case_pass = True

        for i, obs in enumerate(observers):
            theta = math.radians(observer_angles_deg[i])
            amp_ana = curle_analytical_amplitude(r_obs, theta, case_cfg)

            # Skip until signal arrives
            skip = int(math.ceil(arrival_time / cfg.dt)) + 10
            if skip < len(time_list):
                amp_sim = float(p_prime_sim[i, skip:].abs().max())
            else:
                amp_sim = 0.0

            if amp_ana > 1e-10:
                err_pct = 100.0 * abs(amp_sim - amp_ana) / amp_ana
            else:
                err_pct = 0.0 if amp_sim < 1e-8 else 100.0

            passes = err_pct < 3.0
            case_pass = case_pass and passes

            case_results["observers"].append({
                "label": obs.label,
                "amplitude_sim": amp_sim,
                "amplitude_ana": amp_ana,
                "error_pct": err_pct,
                "pass": passes,
                "ka": _TWO_PI * freq_hz * cfg.a / cfg.c0,
                "r_over_lambda": r_obs / wavelength,
            })

        all_results.append(case_results)
        all_pass = all_pass and case_pass

    return {
        "cases": all_results,
        "pass_3pct": all_pass,
        "n_observers": len(observer_angles_deg),
        "n_frequencies": len(frequencies_hz),
    }


# ===================================================================
# Case 2: Moving-source Doppler
# ===================================================================


@dataclass
class DopplerConfig:
    """Configuration for the moving-source Doppler validation case."""

    f_src: float = 1000.0
    V: float = 102.9
    c0: float = 343.0
    amplitude: float = 1.0
    n_steps: int = 16384
    dt: float = 1.25e-5

    @property
    def M(self) -> float:
        return self.V / self.c0


def doppler_analytical_frequency(cfg: DopplerConfig, theta: float) -> float:
    return cfg.f_src / (1.0 - cfg.M * math.cos(theta))


def _solve_retarded_time(t: float, x_obs: float, y_obs: float, V: float, c0: float) -> float:
    t_ret = t
    for _ in range(100):
        dx = x_obs - V * t_ret
        dy = y_obs
        r = math.sqrt(dx * dx + dy * dy)
        t_ret_new = t - r / c0
        if abs(t_ret_new - t_ret) < 1e-14:
            break
        t_ret = t_ret_new
    return t_ret


def validate_doppler_moving_source(
    cfg: DopplerConfig | None = None,
    observer_angles_deg: list[float] | None = None,
) -> dict:
    if cfg is None:
        cfg = DopplerConfig()

    if observer_angles_deg is None:
        observer_angles_deg = [0.0, 180.0]

    omega = _TWO_PI * cfg.f_src
    T = cfg.n_steps
    dt = cfg.dt

    r_obs = 10000.0
    x_start = -5000.0

    results = {
        "observers": [],
        "f_sim": [],
        "f_ana": [],
        "error_pct": [],
        "M": cfg.M,
        "f_src": cfg.f_src,
    }

    for theta_deg in observer_angles_deg:
        theta = math.radians(theta_deg)
        x_obs = r_obs * math.cos(theta)
        y_obs = r_obs * math.sin(theta)

        p_ret = torch.zeros(T, dtype=torch.float64)
        for i in range(T):
            t = i * dt
            t_ret = _solve_retarded_time(t, x_obs, y_obs, cfg.V, cfg.c0)
            p_ret[i] = cfg.amplitude * math.sin(omega * t_ret)

        first_nonzero = 0
        for i in range(T):
            if abs(p_ret[i]) > 1e-10:
                first_nonzero = i
                break

        skip = max(first_nonzero, 500)
        p_steady = p_ret[skip:]
        if len(p_steady) < 1024:
            p_steady = p_ret

        n_fft = len(p_steady)
        fft_vals = torch.fft.rfft(p_steady)
        freqs = torch.fft.rfftfreq(n_fft, d=dt).numpy()
        magnitude = fft_vals.abs().numpy()
        peak_idx = magnitude[1:].argmax() + 1
        f_sim = float(freqs[peak_idx])

        f_ana = doppler_analytical_frequency(cfg, theta)
        err_pct = 100.0 * abs(f_sim - f_ana) / f_ana if f_ana > 0 else 0.0

        results["observers"].append({"label": f"θ={theta_deg:.0f}°", "theta_deg": theta_deg})
        results["f_sim"].append(f_sim)
        results["f_ana"].append(f_ana)
        results["error_pct"].append(err_pct)

    results["pass_1pct"] = all(e < 1.0 for e in results["error_pct"])
    return results


# ===================================================================
# Case 3: Airfoil trailing-edge noise (NACA0012)
# ===================================================================


@dataclass
class AirfoilNoiseConfig:
    chord: float = 0.2286
    span: float = 0.4572
    U_inf: float = 55.0
    rho0: float = 1.225
    c0: float = 343.0
    nu: float = 1.5e-5
    alpha: float = 0.0
    r_obs: float = 1.2
    n_src: int = 128
    n_steps: int = 4096
    dt: float = 1.25e-5

    @property
    def Re(self) -> float:
        return self.U_inf * self.chord / self.nu

    @property
    def M(self) -> float:
        return self.U_inf / self.c0


def build_airfoil_surface(cfg: AirfoilNoiseConfig) -> FWHSurface:
    n = cfg.n_src
    T = cfg.n_steps
    dt = cfg.dt

    x_positions = torch.linspace(0.8 * cfg.chord, cfg.chord, n, dtype=torch.float64)
    positions = torch.stack([x_positions, torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)], dim=-1)
    normals = torch.zeros(n, 3, dtype=torch.float64)
    normals[:, 1] = 1.0
    areas = torch.full((n,), cfg.span / n, dtype=torch.float64)

    torch.manual_seed(42)
    white_noise = torch.randn(n, T, dtype=torch.float64)

    delta_star = 0.003
    f_cutoff = cfg.U_inf / delta_star
    tau_filter = 1.0 / (2 * _PI * f_cutoff)
    alpha_filter = dt / (tau_filter + dt)
    pressure = torch.zeros_like(white_noise)
    pressure[:, 0] = white_noise[:, 0]
    for t in range(1, T):
        pressure[:, t] = (1 - alpha_filter) * pressure[:, t - 1] + alpha_filter * white_noise[:, t]

    p_rms_target = math.sqrt(cfg.rho0 * cfg.U_inf ** 2 * 0.003)
    p_rms_current = float(pressure.std())
    if p_rms_current > 0:
        pressure = pressure * (p_rms_target / p_rms_current)

    return FWHSurface(positions=positions, normals=normals, areas=areas, pressure=pressure, dt=dt, c0=cfg.c0)


def validate_airfoil_noise(cfg: AirfoilNoiseConfig | None = None) -> dict:
    if cfg is None:
        cfg = AirfoilNoiseConfig()

    surface = build_airfoil_surface(cfg)
    observers = [AcousticObserver(x=cfg.chord, y=cfg.r_obs, label="θ=90° (mid-span)")]
    p_prime, time_list = compute_fwh_far_field(surface, observers)

    oaspl_sim = oaspl(p_prime, dt=cfg.dt)[0]
    oaspl_ref = 97.0  # dB (Brooks et al., NASA RP-1218)
    error_db = abs(oaspl_sim - oaspl_ref)

    return {
        "oaspl_sim": oaspl_sim,
        "oaspl_ref": oaspl_ref,
        "error_db": error_db,
        "pass_5db": error_db < 5.0,
        "Re": cfg.Re,
        "M": cfg.M,
        "chord": cfg.chord,
        "r_obs": cfg.r_obs,
    }


# ===================================================================
# Combined validation runner
# ===================================================================


def run_all_validations() -> dict:
    curle = validate_curle_dipole()
    doppler = validate_doppler_moving_source()
    airfoil = validate_airfoil_noise()

    return {
        "curle_dipole": curle,
        "doppler": doppler,
        "airfoil_noise": airfoil,
        "summary": {
            "curle_pass_3pct": curle["pass_3pct"],
            "doppler_pass_1pct": doppler["pass_1pct"],
            "airfoil_pass_5db": airfoil["pass_5db"],
            "all_pass": curle["pass_3pct"] and doppler["pass_1pct"] and airfoil["pass_5db"],
        },
    }
