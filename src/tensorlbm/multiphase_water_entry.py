"""Multiphase sphere/cylinder water-entry benchmark.

Implements two variants of the classic sphere-water-entry problem:

* **2-D (default)** – A circular cylinder falling through a two-phase domain.
* **3-D** – A spherical body (Shan-Chen two-component, 3-D).

Supported multiphase models (2-D only):

* ``"cg"``   – Color-Gradient (default, most stable, recommended)
* ``"sc"``   – Shan-Chen two-component

Physical setup
--------------
Domain:
    Closed box with bounce-back walls on all sides.
Initial condition:
    Heavy fluid (water) fills the lower portion ``y < water_level``.
    Light fluid (air) fills the upper portion.
Sphere/cylinder:
    Rigid bounce-back obstacle, initially centred above the water surface.
Gravity:
    Acts in the −y (2-D) or −z (3-D) direction.

Diagnostics
-----------
The impact force on the sphere is computed at every diagnostic step via the
momentum-exchange method (Ladd 1994) and written to ``forces.csv``.
Snapshots of the density field are saved every ``output_interval`` steps.

References
----------
Worthington (1908) "A Study of Splashes"
Truscott, Epps & Belden (2014) Annu. Rev. Fluid Mech. 46 355
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import torch

from .boundaries import bounce_back_cells
from .boundaries3d import bounce_back_cells_3d
from .d2q9 import C as C2
from .d2q9 import equilibrium
from .d3q19 import C as C3
from .d3q19 import equilibrium3d
from .multiphase import (
    collide_sc_two_component,
    color_gradient_step,
)
from .multiphase3d import collide_sc_two_component_3d
from .solver import stream
from .solver3d import stream3d
from .utils import prepare_run_dir, resolve_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WaterEntryMode = Literal["2d", "3d"]
WaterEntryModel2D = Literal["cg", "sc"]


@dataclass(frozen=True)
class MultiphaseWaterEntryConfig:
    """Configuration for the sphere/cylinder water-entry benchmark."""

    mode: WaterEntryMode = "2d"
    model: WaterEntryModel2D = "cg"
    nx: int = 200
    ny: int = 160
    nz: int = 80   # only used in 3-D mode
    radius: float = 12.0
    water_level: int = 80
    clearance: int = 4
    rho_water: float = 0.8
    rho_air: float = 0.4
    G: float = 0.9
    tau: float = 1.0
    g: float = 5e-5
    n_steps: int = 3000
    output_interval: int = 300
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        if self.mode not in ("2d", "3d"):
            msg = f"mode must be '2d' or '3d', got {self.mode!r}"
            raise ValueError(msg)
        if self.nx < 16 or self.ny < 16:
            msg = "nx and ny must be >= 16"
            raise ValueError(msg)
        if self.radius <= 0:
            msg = "radius must be > 0"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"tau={self.tau} must be > 0.5"
            raise ValueError(msg)
        if self.water_level <= 0 or self.water_level >= self.ny:
            msg = "water_level must be in (0, ny)"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"water_entry_{self.mode}_nx{self.nx}_ny{self.ny}"
            f"_r{self.radius:.0f}_G{self.G:.2f}_g{self.g:.0e}_steps{self.n_steps}"
        )

    @property
    def sphere_center_2d(self) -> tuple[float, float]:
        cx = float(self.nx // 2)
        cy = float(self.water_level + int(self.clearance) + int(self.radius) + 1)
        return cx, cy

    @property
    def sphere_center_3d(self) -> tuple[float, float, float]:
        cx = float(self.nx // 2)
        cy = float(self.ny // 2)
        cz = float(self.water_level + int(self.clearance) + int(self.radius) + 1)
        return cx, cy, cz


# ---------------------------------------------------------------------------
# 2-D geometry helpers
# ---------------------------------------------------------------------------

def _circle_mask(
    ny: int, nx: int, cx: float, cy: float, r: float, device: torch.device
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _wall_mask_2d(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


# ---------------------------------------------------------------------------
# 2-D initialisation helpers
# ---------------------------------------------------------------------------

def _smooth_profile_y(
    ny: int, nx: int, water_level: float, width: float, device: torch.device
) -> torch.Tensor:
    """tanh profile: 1.0 in water (y < water_level), 0.0 in air."""
    y = torch.arange(ny, dtype=torch.float32, device=device)
    prof = 0.5 * (1.0 - torch.tanh((y - water_level) / width))
    return prof.view(ny, 1).expand(ny, nx)


def _init_two_phase_cg_2d(
    ny: int, nx: int, water_level: float, rho_water: float, rho_air: float,
    obstacle: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CG initialisation: smooth tanh water/air split with 5% minority fraction."""
    prof = _smooth_profile_y(ny, nx, water_level, 3.0, device)
    frac = 0.05
    rho_r = rho_water * prof + rho_water * frac * (1.0 - prof)
    rho_b = rho_air * frac * prof + rho_air * (1.0 - prof)
    rho_r[obstacle] = 0.0
    rho_b[obstacle] = 0.0
    zero = torch.zeros((ny, nx), device=device)
    return equilibrium(rho_r, zero, zero), equilibrium(rho_b, zero, zero)


def _init_two_phase_sc_2d(
    ny: int, nx: int, water_level: float, rho_water: float, rho_air: float,
    obstacle: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SC two-component initialisation with smooth interface."""
    prof = _smooth_profile_y(ny, nx, water_level, 4.0, device)
    frac = 0.15
    rho2 = rho_water * prof + rho_water * frac * (1.0 - prof)
    rho1 = rho_air * frac * prof + rho_air * (1.0 - prof)
    rho1[obstacle] = 0.0
    rho2[obstacle] = 0.0
    zero = torch.zeros((ny, nx), device=device)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


# ---------------------------------------------------------------------------
# 3-D geometry helpers
# ---------------------------------------------------------------------------

def _sphere_mask_3d(
    nz: int, ny: int, nx: int, cx: float, cy: float, cz: float, r: float, device: torch.device
) -> torch.Tensor:
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r ** 2


def _wall_mask_3d(nz: int, ny: int, nx: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    mask[0] = mask[-1] = True
    mask[:, 0] = mask[:, -1] = True
    mask[:, :, 0] = mask[:, :, -1] = True
    return mask


def _init_two_phase_3d(
    nz: int, ny: int, nx: int, water_level: float, rho_water: float, rho_air: float,
    obstacle: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.arange(nz, dtype=torch.float32, device=device)
    prof = 0.5 * (1.0 - torch.tanh((z - water_level) / 3.0)).view(nz, 1, 1).expand(nz, ny, nx)
    frac = 0.05
    rho2 = rho_water * prof + rho_water * frac * (1.0 - prof)
    rho1 = rho_air * frac * prof + rho_air * (1.0 - prof)
    rho1[obstacle] = 0.0
    rho2[obstacle] = 0.0
    zero = torch.zeros((nz, ny, nx), device=device)
    f1 = equilibrium3d(rho1, zero, zero, zero)
    f2 = equilibrium3d(rho2, zero, zero, zero)
    return f1, f2


# ---------------------------------------------------------------------------
# Force diagnostics
# ---------------------------------------------------------------------------

def _momentum_exchange_2d(
    f1: torch.Tensor, f2: torch.Tensor, solid: torch.Tensor
) -> tuple[float, float]:
    """Ladd momentum-exchange impact force on a 2-D solid obstacle."""
    device = f1.device
    f_total = f1 + f2
    c = C2.to(device)
    cx = c[:, 0].float().view(9, 1, 1)
    cy = c[:, 1].float().view(9, 1, 1)
    mask = solid.unsqueeze(0)
    f_sol = f_total * mask
    fx = 2.0 * float((cx * f_sol).sum().item())
    fy = 2.0 * float((cy * f_sol).sum().item())
    return fx, fy


def _momentum_exchange_3d(
    f1: torch.Tensor, f2: torch.Tensor, solid: torch.Tensor
) -> tuple[float, float, float]:
    device = f1.device
    f_total = f1 + f2
    c = C3.to(device)
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    mask = solid.unsqueeze(0)
    f_sol = f_total * mask
    fx = 2.0 * float((cx * f_sol).sum().item())
    fy = 2.0 * float((cy * f_sol).sum().item())
    fz = 2.0 * float((cz * f_sol).sum().item())
    return fx, fy, fz


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _save_snapshot_2d(
    run_dir: Path, step: int, rho_water: torch.Tensor, rho_air: torch.Tensor, obstacle: torch.Tensor
) -> None:
    phi = (rho_water / (rho_water + rho_air + 1e-12)).detach().cpu().numpy()
    obs_np = obstacle.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(phi, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.contour(obs_np, levels=[0.5], colors="red", linewidths=1.5)
    plt.colorbar(im, ax=ax, fraction=0.03, label="Water phase fraction φ")
    ax.set_title(f"Water entry – step {step:d}")
    out = run_dir / f"snapshot_{step:06d}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_multiphase_water_entry(config: MultiphaseWaterEntryConfig) -> Path:
    """Run the sphere/cylinder water-entry benchmark.

    Args:
        config: Simulation configuration.

    Returns:
        Path of the output directory.
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "sphere_water_entry",
        config.resolved_run_name(), config.overwrite,
    )

    print(
        f"Running sphere water entry ({config.mode}, {config.model.upper()})  "
        f"device={device}  NX={config.nx}  NY={config.ny}"
        + (f"  NZ={config.nz}" if config.mode == "3d" else "")
        + f"  radius={config.radius}  G={config.G}  g={config.g:.1e}  steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    force_series: list[dict[str, object]] = []

    if config.mode == "2d":
        ny, nx = config.ny, config.nx
        cx, cy = config.sphere_center_2d
        wall = _wall_mask_2d(ny, nx, device)
        sphere = _circle_mask(ny, nx, cx, cy, config.radius, device)
        solid = wall | sphere

        if config.model == "cg":
            f1, f2 = _init_two_phase_cg_2d(
                ny, nx, config.water_level, config.rho_water, config.rho_air, sphere, device
            )
        else:  # sc
            f1, f2 = _init_two_phase_sc_2d(
                ny, nx, config.water_level, config.rho_water, config.rho_air, sphere, device
            )

        gy = -config.g

        for step in range(1, config.n_steps + 1):
            if config.model == "cg":
                A_surface = config.G * 0.04
                # f1=red=heavy(water), f2=blue=light(air)
                f1, f2 = color_gradient_step(
                    f1, f2, tau=config.tau, A=A_surface, gy=gy, solid_mask=solid
                )
            else:  # sc
                f1, f2 = collide_sc_two_component(
                    f1, f2, G_12=config.G, tau1=config.tau, tau2=config.tau,
                    gy=gy, solid_mask=solid,
                )

            f1 = stream(f1)
            f2 = stream(f2)

            if step % config.output_interval == 0 or step == config.n_steps:
                fx_s, fy_s = _momentum_exchange_2d(f1, f2, sphere)

            f1 = bounce_back_cells(f1, solid)
            f2 = bounce_back_cells(f2, solid)

            if step % config.output_interval == 0 or step == config.n_steps:
                if config.model == "cg":
                    rho_w = f1.sum(dim=0)  # RED = water = heavy
                    rho_a = f2.sum(dim=0)  # BLUE = air = light
                else:
                    rho_w = f2.sum(dim=0)  # component 2 = water
                    rho_a = f1.sum(dim=0)  # component 1 = air
                entry: dict[str, object] = {
                    "step": step,
                    "fx": round(fx_s, 8),
                    "fy": round(fy_s, 8),
                    "mean_rho_water": round(float(rho_w[~solid].mean().item()), 6),
                }
                force_series.append(entry)
                print(
                    f"step={step:5d}  Fx={fx_s:.4e}  Fy={fy_s:.4e}  "
                    f"mean_ρ_water={entry['mean_rho_water']:.4f}"
                )
                _save_snapshot_2d(run_dir, step, rho_w, rho_a, sphere)

    else:
        # ---- 3-D simulation (SC two-component) ----
        nz, ny, nx = config.nz, config.ny, config.nx
        cx, cy, cz = config.sphere_center_3d
        wall = _wall_mask_3d(nz, ny, nx, device)
        sphere = _sphere_mask_3d(nz, ny, nx, cx, cy, cz, config.radius, device)
        solid = wall | sphere

        f1, f2 = _init_two_phase_3d(
            nz, ny, nx, config.water_level, config.rho_water, config.rho_air, sphere, device
        )
        gz = -config.g

        for step in range(1, config.n_steps + 1):
            f1, f2 = collide_sc_two_component_3d(
                f1, f2, G_12=config.G, tau1=config.tau, tau2=config.tau,
                gz=gz, solid_mask=solid,
            )
            f1 = stream3d(f1)
            f2 = stream3d(f2)

            if step % config.output_interval == 0 or step == config.n_steps:
                fx_s, fy_s, fz_s = _momentum_exchange_3d(f1, f2, sphere)

            f1 = bounce_back_cells_3d(f1, solid)
            f2 = bounce_back_cells_3d(f2, solid)

            if step % config.output_interval == 0 or step == config.n_steps:
                entry = {
                    "step": step,
                    "fx": round(fx_s, 8),
                    "fy": round(fy_s, 8),
                    "fz": round(fz_s, 8),
                }
                force_series.append(entry)
                print(f"step={step:5d}  Fx={fx_s:.4e}  Fy={fy_s:.4e}  Fz={fz_s:.4e}")

    # ----- write outputs -----
    forces_csv = run_dir / "forces.csv"
    with forces_csv.open("w", newline="", encoding="utf-8") as fh:
        if force_series:
            writer = csv.DictWriter(fh, fieldnames=list(force_series[0].keys()))
            writer.writeheader()
            writer.writerows(force_series)

    metadata = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "forces": force_series,
        "note": (
            "Impact force computed via Ladd momentum-exchange method. "
            "Fy (2-D) / Fz (3-D) is the vertical force on the sphere."
        ),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved metadata → {run_dir / 'run_metadata.json'}")
    return run_dir


__all__ = ["MultiphaseWaterEntryConfig", "run_multiphase_water_entry"]
