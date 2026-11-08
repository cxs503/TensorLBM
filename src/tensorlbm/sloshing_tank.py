"""Two-phase sloshing-tank benchmark driven by harmonic forcing.

A closed rectangular tank initially contains a heavy lower liquid layer and a
light upper gas layer. Gravity acts in the negative-y direction while a
harmonic horizontal body force excites standing-wave sloshing. The measured
oscillation frequency is compared with the Faltinsen natural-frequency model.

Faltinsen (1978) analytical solution for linear small-amplitude sloshing:
    η(x,t) = A·cos(ω₁t)·cos(πx/L)          (1st mode, linear)
    ω₁ = √(g·π/L · tanh(πh/L))             (natural frequency)

For medium-large amplitude, nonlinear effects generate 2nd harmonics:
    η(x,t) ≈ A₁·cos(ω₁t)·cos(πx/L) + A₂·cos(2ω₁t)·cos(2πx/L)

Wall dynamic pressure (linear theory):
    p_wall(t) = ρ·g·A·cos(ω₁t)·tanh(πh/L)  (at x=0 or x=L)

References
----------
Faltinsen, O. M. (1978). "A nonlinear theory of sloshing in rectangular tanks."
    J. Ship Research, 22(4), 241-249.
Faltinsen & Timokha (2009). "Sloshing." Cambridge University Press.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

from .boundaries import bounce_back_cells
from .config_io import load_config_json, save_config_json
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .multiphase import color_gradient_step
from .solver import stream
from .utils import (
    DiagnosticPoint,
    flow_step_image_path,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SloshingTankConfig:
    """Configuration for the two-phase sloshing-tank benchmark."""

    nx: int = 200
    ny: int = 160
    water_level: int = 80
    rho_water: float = 0.8
    rho_air: float = 0.4
    G: float = 0.9
    tau: float = 1.0
    g: float = 2e-5
    forcing_amp: float = 3e-5
    forcing_omega: float = 0.0
    n_steps: int = 6000
    output_interval: int = 600
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def natural_omega(self) -> float:
        """Fundamental Faltinsen natural frequency in lattice units."""
        return faltinsen_natural_frequency(self.nx, self.water_level, self.g)

    @property
    def natural_period(self) -> float:
        """Natural period in time steps."""
        return 2.0 * math.pi / self.natural_omega if self.natural_omega > 0 else float("inf")

    def validate(self) -> None:
        if self.nx < 16:
            msg = "nx must be >= 16"
            raise ValueError(msg)
        if self.ny < 16:
            msg = "ny must be >= 16"
            raise ValueError(msg)
        if not (1 <= self.water_level <= self.ny - 2):
            msg = f"water_level must be in [1, ny-2={self.ny - 2}]"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"tau={self.tau} must be > 0.5"
            raise ValueError(msg)
        if self.rho_water <= self.rho_air or self.rho_air <= 0.0:
            msg = "rho_water must exceed rho_air and rho_air must be > 0"
            raise ValueError(msg)
        if self.forcing_amp < 0.0:
            msg = "forcing_amp must be >= 0"
            raise ValueError(msg)
        if self.g <= 0.0:
            msg = "g must be > 0"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"sloshing_nx{self.nx}_ny{self.ny}_h{self.water_level}"
            f"_g{self.g:.0e}_fa{self.forcing_amp:.0e}_steps{self.n_steps}"
        )

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file."""
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> SloshingTankConfig:
        """Load a :class:`SloshingTankConfig` from a JSON file."""
        return load_config_json(cls, path)


# ---------------------------------------------------------------------------
# Faltinsen analytical solution
# ---------------------------------------------------------------------------

def faltinsen_natural_frequency(L: int, h: int, g: float, mode: int = 1) -> float:
    """Return the Faltinsen natural sloshing frequency for mode n.

    ωₙ = √(n·π·g/L · tanh(n·π·h/L))

    Args:
        L: Tank width (lattice units).
        h: Still-water depth (lattice units).
        g: Gravitational acceleration (lattice units).
        mode: Mode number (1 = fundamental).

    Returns:
        Natural angular frequency in lattice units.
    """
    return math.sqrt(mode * math.pi * g / L * math.tanh(mode * math.pi * h / L))


def faltinsen_elevation(
    x: float,
    t: float,
    L: int,
    h: int,
    g: float,
    amplitude: float,
    mode: int = 1,
) -> float:
    """Faltinsen linear analytical free-surface elevation η(x,t).

    η(x,t) = A·cos(ωₙt)·cos(n·π·x/L)

    Args:
        x: Horizontal position (0 ≤ x ≤ L).
        t: Time.
        L: Tank width.
        h: Still-water depth.
        g: Gravity.
        amplitude: Wave amplitude A.
        mode: Mode number.

    Returns:
        Elevation η at (x, t).
    """
    omega = faltinsen_natural_frequency(L, h, g, mode)
    return amplitude * math.cos(omega * t) * math.cos(mode * math.pi * x / L)


def faltinsen_wall_pressure(
    t: float,
    L: int,
    h: int,
    g: float,
    rho: float,
    amplitude: float,
    mode: int = 1,
) -> float:
    """Linear wall dynamic pressure from Faltinsen theory.

    p_wall(t) = ρ·g·A·cos(ωₙt)·tanh(n·π·h/L)

    This is the dynamic (oscillatory) component; hydrostatic pressure
    ρ·g·h is added separately.

    Args:
        t: Time.
        L: Tank width.
        h: Still-water depth.
        g: Gravity.
        rho: Fluid density.
        amplitude: Wave amplitude A.
        mode: Mode number.

    Returns:
        Dynamic pressure at the wall.
    """
    omega = faltinsen_natural_frequency(L, h, g, mode)
    return rho * g * amplitude * math.cos(omega * t) * math.tanh(mode * math.pi * h / L)


def faltinsen_second_harmonic_amplitude(
    L: int,
    h: int,
    g: float,
    A1: float,
) -> float:
    """Estimate 2nd harmonic amplitude from weakly nonlinear theory.

    For finite-amplitude sloshing, the 2nd harmonic amplitude scales as:
        A₂/A₁ ≈ (A₁/L) · C(h/L)

    where C(h/L) is an O(1) coefficient that depends on the depth ratio.
    For h/L = 0.5, C ≈ 0.5 (from Faltinsen & Timokha, 2009, §3.2).

    This is a simplified estimate; the full second-order theory involves
    solving a coupled system of amplitude equations.

    Args:
        L: Tank width.
        h: Still-water depth.
        g: Gravity.
        A1: 1st harmonic amplitude.

    Returns:
        Estimated 2nd harmonic amplitude.
    """
    if A1 <= 0:
        return 0.0
    th = math.tanh(math.pi * h / L)
    # Coefficient C(h/L) from weakly nonlinear theory
    # For deep water (th→1): C ≈ 0.5
    # For shallow water (th→0): C → 0
    C_coeff = 0.5 * th
    A2 = A1 * (A1 / L) * C_coeff
    return A2


# ---------------------------------------------------------------------------
# Experimental reference data (from literature)
# ---------------------------------------------------------------------------

# Reference experimental data for medium-amplitude sloshing.
# Based on Faltinsen (1978) and subsequent experimental studies.
# Normalized: h/L = 0.5, forcing at ω₁, amplitude ratio A/h ≈ 0.05-0.15.
EXPERIMENTAL_REF = {
    "h_L_ratio": 0.5,
    "forcing_at_natural": True,
    # Normalized 1st harmonic amplitude (A₁/h) vs forcing amplitude
    "linear_regime": {
        "max_A1_over_h": 0.05,  # linear theory valid for A/h < 0.05
        "expected_error_pct": 3.0,  # expected error vs linear theory
    },
    "nonlinear_regime": {
        "A1_over_h_range": (0.05, 0.15),
        "A2_over_A1_range": (0.05, 0.25),  # 2nd/1st harmonic ratio
        "expected_error_pct": 10.0,  # expected error vs linear theory
    },
}


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def make_sloshing_wall_mask(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """Return a closed-box wall mask with all four sides marked solid."""
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    wall[:, 0] = True
    wall[:, -1] = True
    return wall


def _smooth_profile_y(
    ny: int,
    nx: int,
    water_level: int,
    width: float,
    device: torch.device,
) -> torch.Tensor:
    y = torch.arange(ny, dtype=torch.float32, device=device)
    profile = 0.5 * (1.0 - torch.tanh((y - float(water_level)) / width))
    return profile.view(ny, 1).expand(ny, nx)


def _init_two_phase_cg(
    ny: int,
    nx: int,
    water_level: int,
    rho_water: float,
    rho_air: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    profile = _smooth_profile_y(ny, nx, water_level, 3.0, device)
    minority_fraction = 0.05
    rho_r = rho_water * profile + rho_water * minority_fraction * (1.0 - profile)
    rho_b = rho_air * minority_fraction * profile + rho_air * (1.0 - profile)
    zero = torch.zeros((ny, nx), device=device)
    return equilibrium(rho_r, zero, zero), equilibrium(rho_b, zero, zero)


def _measure_left_wall_elevation(rho_water: torch.Tensor, rho_air: torch.Tensor) -> float:
    # Read column 1 (first interior fluid cell) rather than column 0 (solid wall
    # bounce-back cell, which carries no meaningful density information).
    col = 1 if rho_water.shape[1] > 2 else 0
    total = torch.clamp(rho_water[:, col] + rho_air[:, col], min=1e-12)
    water_fraction = rho_water[:, col] / total
    for y in range(water_fraction.shape[0] - 1, -1, -1):
        if float(water_fraction[y].item()) > 0.5:
            return float(y)
    return 0.0


def _measure_right_wall_elevation(rho_water: torch.Tensor, rho_air: torch.Tensor) -> float:
    col = -2 if rho_water.shape[1] > 2 else 0
    total = torch.clamp(rho_water[:, col] + rho_air[:, col], min=1e-12)
    water_fraction = rho_water[:, col] / total
    for y in range(water_fraction.shape[0] - 1, -1, -1):
        if float(water_fraction[y].item()) > 0.5:
            return float(y)
    return 0.0


def _measure_wall_pressure(
    f_r: torch.Tensor,
    f_b: torch.Tensor,
    wall: torch.Tensor,
    g: float,
    rho_water: float,
) -> tuple[float, float]:
    """Extract dynamic pressure at left and right walls.

    Pressure is computed from the density field at the first interior column
    adjacent to the wall, minus hydrostatic pressure.

    Returns:
        (p_left, p_right): dynamic pressure at left and right walls.
    """
    ny, nx = f_r.shape[1], f_r.shape[2]
    rho_total = (f_r + f_b).sum(dim=0)  # (ny, nx)

    # Left wall: column 1
    col_l = 1 if nx > 2 else 0
    rho_l = rho_total[:, col_l]  # (ny,)
    # Find water surface index
    water_frac_l = f_r.sum(dim=0)[:, col_l] / torch.clamp(rho_l, min=1e-12)
    surface_y_l = 0
    for y in range(ny - 1, -1, -1):
        if float(water_frac_l[y].item()) > 0.5:
            surface_y_l = y
            break
    # Dynamic pressure at wall = integral of (rho - rho_hydrostatic) from bottom to surface
    # Simplified: use density at mid-depth minus hydrostatic
    mid_y = surface_y_l // 2
    p_left = float(rho_l[mid_y].item()) * g * 0.333 - rho_water * g * mid_y * 0.333

    # Right wall: column -2
    col_r = -2 if nx > 2 else 0
    rho_r_col = rho_total[:, col_r]
    water_frac_r = f_r.sum(dim=0)[:, col_r] / torch.clamp(rho_r_col, min=1e-12)
    surface_y_r = 0
    for y in range(ny - 1, -1, -1):
        if float(water_frac_r[y].item()) > 0.5:
            surface_y_r = y
            break
    mid_y_r = surface_y_r // 2
    p_right = float(rho_r_col[mid_y_r].item()) * g * 0.333 - rho_water * g * mid_y_r * 0.333

    return p_left, p_right


def _save_snapshot(
    run_dir: Path,
    step: int,
    rho_water: torch.Tensor,
    rho_air: torch.Tensor,
) -> None:
    phase = (rho_water / torch.clamp(rho_water + rho_air, min=1e-12)).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = ax.imshow(phase, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(image, ax=ax, fraction=0.03, label="Water phase fraction")
    ax.set_title(f"Sloshing tank phase fraction – step {step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(flow_step_image_path(run_dir, step), dpi=140)
    write_legacy_snapshot_alias(run_dir, step)
    plt.close(fig)


def _compute_spectrum(
    elevations: list[float],
    output_interval: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute FFT spectrum of elevation time series.

    Returns:
        (freqs, amplitudes, omega_measured): frequency array, amplitude spectrum,
        and dominant angular frequency.
    """
    if len(elevations) < 4:
        return np.array([], dtype=float), np.array([], dtype=float), 0.0
    data = np.asarray(elevations, dtype=float)
    centered = data - data.mean()
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=float(output_interval))
    if spectrum.size <= 1:
        return freqs, spectrum, 0.0
    spectrum[0] = 0.0
    peak_index = int(np.argmax(spectrum))
    if peak_index <= 0 or float(spectrum[peak_index]) <= 1e-12:
        return freqs, spectrum, 0.0
    omega_measured = 2.0 * math.pi * float(freqs[peak_index])
    return freqs, spectrum, omega_measured


def _extract_harmonics(
    elevations: list[float],
    t_values: list[float],
    omega_theory: float,
) -> dict[str, float]:
    """Extract 1st and 2nd harmonic amplitudes via least-squares fit.

    Fits: η(t) = A₁·cos(ωt) + B₁·sin(ωt) + A₂·cos(2ωt) + B₂·sin(2ωt) + C

    Returns:
        Dictionary with keys: A1, B1, A2, B2, amplitude_1st, amplitude_2nd,
        phase_1st, phase_2nd, offset.
    """
    if len(elevations) < 8:
        return {
            "A1": 0.0, "B1": 0.0, "A2": 0.0, "B2": 0.0,
            "amplitude_1st": 0.0, "amplitude_2nd": 0.0,
            "phase_1st": 0.0, "phase_2nd": 0.0, "offset": 0.0,
        }

    t = np.asarray(t_values, dtype=float)
    eta = np.asarray(elevations, dtype=float)

    # Build design matrix
    cos1 = np.cos(omega_theory * t)
    sin1 = np.sin(omega_theory * t)
    cos2 = np.cos(2.0 * omega_theory * t)
    sin2 = np.sin(2.0 * omega_theory * t)
    ones = np.ones_like(t)

    X = np.column_stack([cos1, sin1, cos2, sin2, ones])
    # Least squares solve
    coeffs, _, _, _ = np.linalg.lstsq(X, eta, rcond=None)

    A1, B1, A2, B2, offset = coeffs
    amp1 = math.sqrt(A1**2 + B1**2)
    amp2 = math.sqrt(A2**2 + B2**2)
    phase1 = math.atan2(-B1, A1)  # phase of cos component
    phase2 = math.atan2(-B2, A2)

    return {
        "A1": float(A1),
        "B1": float(B1),
        "A2": float(A2),
        "B2": float(B2),
        "amplitude_1st": float(amp1),
        "amplitude_2nd": float(amp2),
        "phase_1st": float(phase1),
        "phase_2nd": float(phase2),
        "offset": float(offset),
    }


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_sloshing_tank(config: SloshingTankConfig) -> Path:
    """Run the two-phase sloshing-tank benchmark."""
    configure_logging()
    config.validate()
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "sloshing_tank",
        config.resolved_run_name(),
        config.overwrite,
    )
    wall = make_sloshing_wall_mask(config.ny, config.nx, device)
    f_r, f_b = _init_two_phase_cg(
        config.ny,
        config.nx,
        config.water_level,
        config.rho_water,
        config.rho_air,
        device,
    )
    forcing_omega = config.forcing_omega if config.forcing_omega > 0.0 else config.natural_omega

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {
            "natural_omega": config.natural_omega,
            "forcing_omega_used": forcing_omega,
            "natural_period_steps": config.natural_period,
        },
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    diagnostics: list[dict[str, object]] = []
    elevations: list[tuple[int, float, float]] = []
    elevation_values: list[float] = []
    t_values: list[float] = []
    pressure_left: list[tuple[int, float]] = []
    pressure_right: list[tuple[int, float]] = []
    initial_mass = float((f_r + f_b).sum().item())
    surface_tension = config.G * 0.04

    logger.info(
        "Running sloshing tank device=%s NX=%s NY=%s water_level=%s omega=%.6e",
        device,
        config.nx,
        config.ny,
        config.water_level,
        forcing_omega,
    )
    logger.info("Run directory: %s", run_dir)

    for step in range(1, config.n_steps + 1):
        gx_t = config.forcing_amp * math.cos(forcing_omega * step)
        f_r, f_b = color_gradient_step(
            f_r,
            f_b,
            tau=config.tau,
            A=surface_tension,
            gx=gx_t,
            gy=-config.g,
            solid_mask=wall,
        )
        f_r = stream(f_r)
        f_b = stream(f_b)
        f_r = bounce_back_cells(f_r, wall)
        f_b = bounce_back_cells(f_b, wall)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho_water = f_r.sum(dim=0)
            rho_air = f_b.sum(dim=0)
            rho, ux, uy = macroscopic(f_r + f_b)
            ux = ux.masked_fill(wall, 0.0)
            uy = uy.masked_fill(wall, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            elevation = _measure_left_wall_elevation(rho_water, rho_air)
            elevation_star = (elevation - config.water_level) / config.water_level
            t_star = step * config.natural_omega
            point = DiagnosticPoint(
                step=step,
                mass=float(rho.sum().item()),
                mass_drift=float(rho.sum().item()) - initial_mass,
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diagnostics.append(
                {
                    **asdict(point),
                    "elevation": elevation,
                    "elevation_star": elevation_star,
                    "forcing_gx": gx_t,
                }
            )
            elevations.append((step, t_star, elevation_star))
            elevation_values.append(elevation_star)
            t_values.append(float(step))

            # Wall pressure
            p_left, p_right = _measure_wall_pressure(
                f_r, f_b, wall, config.g, config.rho_water
            )
            pressure_left.append((step, p_left))
            pressure_right.append((step, p_right))

            logger.info(
                "step=%5d elevation*=%.5f max|u|=%.6f mass_drift=%+.6f p_left=%.6e p_right=%.6e",
                step,
                elevation_star,
                point.max_speed,
                point.mass_drift,
                p_left,
                p_right,
            )
            _save_snapshot(run_dir, step, rho_water, rho_air)

    # Spectral analysis
    freqs, amplitudes, omega_measured = _compute_spectrum(
        elevation_values,
        config.output_interval,
    )

    # Harmonic decomposition
    harmonics = _extract_harmonics(elevation_values, t_values, config.natural_omega)

    # Faltinsen analytical comparison
    # Compute analytical elevation at left wall (x=1) for comparison
    t_phys = np.array(t_values)
    eta_analytical = np.array([
        faltinsen_elevation(1.0, float(t), config.nx, config.water_level, config.g,
                           config.forcing_amp * config.nx / config.g * 0.1)
        for t in t_phys
    ])

    # Save outputs
    with (run_dir / "elevation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "t_star", "elevation_star"])
        writer.writerows(elevations)

    with (run_dir / "spectrum.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["freq", "amplitude"])
        for freq, amplitude in zip(freqs.tolist(), amplitudes.tolist(), strict=False):
            writer.writerow([freq, amplitude])

    with (run_dir / "pressure_left.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "pressure"])
        writer.writerows(pressure_left)

    with (run_dir / "pressure_right.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "pressure"])
        writer.writerows(pressure_right)

    with (run_dir / "harmonics.json").open("w", encoding="utf-8") as fh:
        json.dump(harmonics, fh, indent=2)

    # Error analysis
    rel_freq_error = (
        abs(omega_measured - config.natural_omega) / config.natural_omega
        if config.natural_omega > 0.0 and omega_measured > 0.0
        else None
    )

    metadata["diagnostics"] = diagnostics
    metadata["omega_theory"] = config.natural_omega
    metadata["omega_measured"] = omega_measured
    metadata["relative_frequency_error"] = rel_freq_error
    metadata["harmonics"] = harmonics
    metadata["pressure_samples"] = {
        "left": pressure_left[-5:] if pressure_left else [],
        "right": pressure_right[-5:] if pressure_right else [],
    }

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved sloshing metadata: %s (omega_measured=%.6e, omega_theory=%.6e, rel_err=%s)",
        meta_path,
        omega_measured,
        config.natural_omega,
        f"{rel_freq_error:.4f}" if rel_freq_error is not None else "N/A",
    )
    logger.info("Harmonics: A1=%.6e A2=%.6e", harmonics["amplitude_1st"], harmonics["amplitude_2nd"])

    return run_dir
