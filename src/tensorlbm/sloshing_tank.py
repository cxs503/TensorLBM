"""Two-phase sloshing-tank benchmark driven by harmonic forcing.

A closed rectangular tank initially contains a heavy lower liquid layer and a
light upper gas layer. Gravity acts in the negative-y direction while a
harmonic horizontal body force excites standing-wave sloshing. The measured
oscillation frequency is compared with the Faltinsen natural-frequency model.
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
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
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


def faltinsen_natural_frequency(L: int, h: int, g: float, mode: int = 1) -> float:
    """Return the Faltinsen natural sloshing frequency."""
    return math.sqrt(mode * math.pi * g / L * math.tanh(mode * math.pi * h / L))


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
    col = 1 if rho_water.shape[1] > 2 else 0  # noqa: SIM210
    total = torch.clamp(rho_water[:, col] + rho_air[:, col], min=1e-12)
    water_fraction = rho_water[:, col] / total
    for y in range(water_fraction.shape[0] - 1, -1, -1):
        if float(water_fraction[y].item()) > 0.5:
            return float(y)
    return 0.0


def _save_snapshot(
    run_dir: Path,
    step: int,
    rho_water: torch.Tensor,
    rho_air: torch.Tensor,
) -> None:
    phase = (
        rho_water / torch.clamp(rho_water + rho_air, min=1e-12)
    ).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    image = ax.imshow(phase, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(image, ax=ax, fraction=0.03, label="Water phase fraction")
    ax.set_title(f"Sloshing tank phase fraction – step {step}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(run_dir / f"snapshot_{step:06d}.png", dpi=140)
    plt.close(fig)


def _compute_spectrum(
    elevations: list[float],
    output_interval: int,
) -> tuple[np.ndarray, np.ndarray, float]:
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
        },
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    diagnostics: list[dict[str, object]] = []
    elevations: list[tuple[int, float, float]] = []
    elevation_values: list[float] = []
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
            logger.info(
                "step=%5d elevation*=%.5f max|u|=%.6f mass_drift=%+.6f",
                step,
                elevation_star,
                point.max_speed,
                point.mass_drift,
            )
            _save_snapshot(run_dir, step, rho_water, rho_air)

    freqs, amplitudes, omega_measured = _compute_spectrum(
        elevation_values,
        config.output_interval,
    )

    with (run_dir / "elevation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "t_star", "elevation_star"])
        writer.writerows(elevations)

    with (run_dir / "spectrum.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["freq", "amplitude"])
        for freq, amplitude in zip(freqs.tolist(), amplitudes.tolist(), strict=False):
            writer.writerow([freq, amplitude])

    metadata["diagnostics"] = diagnostics
    metadata["omega_theory"] = config.natural_omega
    metadata["omega_measured"] = omega_measured
    metadata["relative_frequency_error"] = (
        abs(omega_measured - config.natural_omega) / config.natural_omega
        if config.natural_omega > 0.0 and omega_measured > 0.0
        else None
    )

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved sloshing metadata: %s (omega_measured=%.6e, omega_theory=%.6e)",
        meta_path,
        omega_measured,
        config.natural_omega,
    )
    return run_dir


__all__ = [
    "SloshingTankConfig",
    "faltinsen_natural_frequency",
    "make_sloshing_wall_mask",
    "run_sloshing_tank",
]
