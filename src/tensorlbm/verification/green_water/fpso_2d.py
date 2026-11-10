"""2D FPSO green water on deck benchmark.

Simulates a fixed FPSO in regular waves (λ/L=1.5, H/λ=0.05) and tracks
free surface evolution vs Greco 2012 experimental data.

Physics:
- Color-gradient multiphase LBM (D2Q9)
- Gravity in -y direction
- FPSO deck as bounce-back obstacle
- Free surface tracking via phase fraction

Outputs:
- Free surface profiles at selected time instances
- Impact pressure time history at deck edge
- Mass conservation diagnostics
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

from tensorlbm.boundaries import bounce_back_cells
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import color_gradient_step
from tensorlbm.solver import stream

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .reference_data import GRECO_2012_FREE_SURFACE, GRECO_2012_IMPACT_PRESSURE
from tensorlbm.utils import prepare_run_dir, resolve_device


@dataclass(frozen=True)
class FPSOGreenWaterConfig:
    """Configuration for 2D FPSO green water benchmark."""

    nx: int = 100
    ny: int = 40
    deck_y: int = 22
    deck_x0: int = 40
    deck_x1: int = 80
    deck_thickness: int = 2
    water_level: int = 18
    wave_amplitude: float = 1.0
    wave_length: float = 60.0
    rho_water: float = 1.0
    rho_air: float = 0.05
    tau: float = 1.0
    G_surface: float = 0.04
    g: float = 1e-5
    n_steps: int = 200
    output_interval: int = 50
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.nx < 30 or self.ny < 15:
            raise ValueError("nx >= 30, ny >= 15 required")
        if self.deck_y <= self.water_level:
            raise ValueError("deck_y must be above water_level")
        if self.deck_x0 <= 0 or self.deck_x1 >= self.nx:
            raise ValueError("deck must be within domain")
        if self.deck_x0 >= self.deck_x1:
            raise ValueError("deck_x0 must be < deck_x1")
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} must be > 0.5")

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"fpso_gw_nx{self.nx}_ny{self.ny}_dy{self.deck_y}"
            f"_wl{self.water_level}_A{self.wave_amplitude:.1f}"
            f"_lambda{self.wave_length:.0f}_steps{self.n_steps}"
        )


def _build_fpsobstacle(
    ny: int, nx: int, deck_y: int, deck_x0: int, deck_x1: int,
    deck_thickness: int, water_level: int, device: torch.device,
) -> torch.Tensor:
    """Build solid mask: walls + FPSO deck + hull below waterline."""
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, -1] = True
    mask[deck_y:deck_y + deck_thickness, deck_x0:deck_x1] = True
    hull_bottom = max(0, water_level - 10)
    mask[hull_bottom:deck_y, deck_x0:deck_x1] = True
    return mask


def _init_fpso_field(
    ny: int, nx: int, water_level: int, deck_y: int,
    deck_x0: int, deck_x1: int, rho_water: float, rho_air: float,
    wave_amplitude: float, wave_length: float, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize water/air fields with perturbed free surface."""
    y = torch.arange(ny, dtype=torch.float32, device=device).view(ny, 1).expand(ny, nx)
    x = torch.arange(nx, dtype=torch.float32, device=device).view(1, nx).expand(ny, nx)

    k = 2.0 * np.pi / wave_length
    eta = wave_amplitude * np.cos(k * x[0, :].cpu().numpy())
    eta_tensor = torch.tensor(eta, device=device).view(1, nx).expand(ny, nx)
    water_y = water_level + eta_tensor

    interface_width = 3.0
    frac_water = 0.5 * (1.0 - torch.tanh((y - water_y) / interface_width))

    frac = 0.03
    rho_r = rho_water * frac_water + rho_water * frac * (1.0 - frac_water)
    rho_b = rho_air * frac * frac_water + rho_air * (1.0 - frac_water)

    zero = torch.zeros((ny, nx), device=device)
    f_r = equilibrium(rho_r, zero, zero)
    f_b = equilibrium(rho_b, zero, zero)

    return f_r, f_b


def _extract_free_surface(
    rho_r: torch.Tensor, rho_b: torch.Tensor, threshold: float = 0.5,
) -> np.ndarray:
    """Extract free surface elevation η(x) from phase fraction field."""
    phi = rho_r / (rho_r + rho_b + 1e-12)
    ny, nx = phi.shape
    eta = np.zeros(nx, dtype=np.float64)
    for j in range(nx):
        col = phi[:, j]
        above = (col > threshold).nonzero(as_tuple=True)[0]
        if above.numel() > 0:
            eta[j] = float(above.max().item())
        else:
            eta[j] = 0.0
    return eta


def _compute_impact_pressure(
    rho_r: torch.Tensor, rho_b: torch.Tensor,
    ux: torch.Tensor, uy: torch.Tensor,
    deck_x: int, deck_y: int, rho_water: float,
) -> float:
    """Compute dynamic pressure at deck edge."""
    if deck_x < rho_r.shape[1] and deck_y < rho_r.shape[0]:
        rho = rho_r[deck_y, deck_x] + rho_b[deck_y, deck_x]
        u = torch.sqrt(ux[deck_y, deck_x]**2 + uy[deck_y, deck_x]**2)
        p_dyn = 0.5 * float(rho.item()) * float(u.item())**2
        return max(p_dyn, 0.0)
    return 0.0


def _save_snapshot(
    run_dir: Path, step: int, rho_r: torch.Tensor, rho_b: torch.Tensor,
    wall: torch.Tensor, config: FPSOGreenWaterConfig,
) -> None:
    """Save phase fraction snapshot."""
    phi = (rho_r / (rho_r + rho_b + 1e-12)).detach().cpu().numpy()
    wall_np = wall.detach().cpu().numpy()
    phi[wall_np] = 0.5

    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    im = ax.imshow(phi, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.02, label="Water phase fraction φ")
    ax.set_title(f"FPSO Green Water – step {step:d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axhline(y=config.deck_y, color="r", linestyle="--", alpha=0.5, label="Deck")
    ax.legend(loc="upper right")
    out = run_dir / f"fpso_step_{step:06d}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)


def run_fpso_green_water(config: FPSOGreenWaterConfig) -> Path:
    """Run the 2D FPSO green water benchmark."""
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "green_water", config.resolved_run_name(), config.overwrite
    )

    ny, nx = config.ny, config.nx
    wall = _build_fpsobstacle(
        ny, nx, config.deck_y, config.deck_x0, config.deck_x1,
        config.deck_thickness, config.water_level, device,
    )

    f_r, f_b = _init_fpso_field(
        ny, nx, config.water_level, config.deck_y,
        config.deck_x0, config.deck_x1, config.rho_water, config.rho_air,
        config.wave_amplitude, config.wave_length, device,
    )

    print(
        f"Running FPSO green water  device={device}  NX={nx}  NY={ny}  "
        f"deck_y={config.deck_y}  water_level={config.water_level}  "
        f"steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    gy = -config.g
    mass_history: list[dict] = []
    surface_profiles: list[dict] = []
    pressure_history: list[dict] = []

    m_r0 = float(f_r.sum().item())
    m_b0 = float(f_b.sum().item())

    for step in range(1, config.n_steps + 1):
        # Collision + streaming
        f_r, f_b = color_gradient_step(
            f_r, f_b,
            tau=config.tau,
            A=config.G_surface,
            gy=gy,
            solid_mask=wall,
        )
        f_r, f_b = stream(f_r), stream(f_b)
        f_r = bounce_back_cells(f_r, wall)
        f_b = bounce_back_cells(f_b, wall)

        # Diagnostics
        if step % config.output_interval == 0 or step == config.n_steps:
            rho_r = f_r.sum(dim=0)
            rho_b = f_b.sum(dim=0)
            _, ux, uy = macroscopic(f_r + f_b)

            eta = _extract_free_surface(rho_r, rho_b)
            surface_profiles.append({"step": step, "eta": eta.tolist()})

            p_impact = _compute_impact_pressure(
                rho_r, rho_b, ux, uy,
                config.deck_x0, config.deck_y, config.rho_water,
            )
            pressure_history.append({"step": step, "p_impact": p_impact})

            m_r = float(f_r.sum().item())
            m_b = float(f_b.sum().item())
            mass_drift_r = abs(m_r - m_r0) / max(m_r0, 1e-12)
            mass_drift_b = abs(m_b - m_b0) / max(m_b0, 1e-12)
            mass_history.append({
                "step": step,
                "mass_water": m_r,
                "mass_air": m_b,
                "drift_water": mass_drift_r,
                "drift_air": mass_drift_b,
            })

            print(
                f"step={step:5d}  drift_w={mass_drift_r:.2e}  "
                f"drift_a={mass_drift_b:.2e}  p_impact={p_impact:.2e}"
            )
            _save_snapshot(run_dir, step, rho_r, rho_b, wall, config)

    _write_outputs(run_dir, config, surface_profiles, pressure_history, mass_history)
    return run_dir


def _write_outputs(
    run_dir: Path, config: FPSOGreenWaterConfig,
    surface_profiles: list[dict], pressure_history: list[dict],
    mass_history: list[dict],
) -> None:
    """Write all output files."""
    with (run_dir / "free_surface_profiles.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step"] + [f"eta_{j}" for j in range(config.nx)])
        for prof in surface_profiles:
            writer.writerow([prof["step"]] + prof["eta"])

    with (run_dir / "impact_pressure.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "p_impact"])
        for rec in pressure_history:
            writer.writerow([rec["step"], rec["p_impact"]])

    with (run_dir / "mass_history.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "mass_water", "mass_air", "drift_water", "drift_air"])
        for rec in mass_history:
            writer.writerow([
                rec["step"], rec["mass_water"], rec["mass_air"],
                rec["drift_water"], rec["drift_air"],
            ])

    metadata = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "n_profiles": len(surface_profiles),
        "n_pressure_records": len(pressure_history),
        "max_mass_drift_water": max((r["drift_water"] for r in mass_history), default=0),
        "max_mass_drift_air": max((r["drift_air"] for r in mass_history), default=0),
        "max_impact_pressure": max((r["p_impact"] for r in pressure_history), default=0),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


__all__ = ["FPSOGreenWaterConfig", "run_fpso_green_water"]
