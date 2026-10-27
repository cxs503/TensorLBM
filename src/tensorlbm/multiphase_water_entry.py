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
    Rigid obstacle with *prescribed constant downward velocity* (``v_entry``),
    initially centred above the water surface.  Moving bounce-back (Ladd 1994)
    with fresh/cleared-node treatment; impact force via the moving-wall
    momentum-exchange method.  ``v_entry=0`` recovers the static obstacle.
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
from .utils import (
    flow_step_image_path,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)

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
    nz: int = 80  # only used in 3-D mode
    radius: float = 12.0
    water_level: int = 80
    clearance: int = 4
    rho_water: float = 0.8
    rho_air: float = 0.4
    G: float = -0.9
    # CG surface-tension coefficient (positive; decoupled from the SC coupling
    # G — the legacy A = G*0.04 was negative, i.e. anti-tension).
    A_surface: float = 0.04
    # Prescribed downward entry speed (lattice units per step); 0 = static.
    v_entry: float = 0.02
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
        if self.v_entry < 0:
            msg = "v_entry must be >= 0"
            raise ValueError(msg)
        if self.A_surface < 0:
            msg = "A_surface must be >= 0 (negative A is anti-tension)"
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
        return float(self.nx // 2), float(
            self.water_level + int(self.clearance) + int(self.radius) + 1,
        )

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
    ny: int,
    nx: int,
    cx: float,
    cy: float,
    r: float,
    device: torch.device,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def _wall_mask_2d(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


# ---------------------------------------------------------------------------
# 2-D initialisation helpers
# ---------------------------------------------------------------------------


def _smooth_profile_y(
    ny: int,
    nx: int,
    water_level: float,
    width: float,
    device: torch.device,
) -> torch.Tensor:
    """tanh profile: 1.0 in water (y < water_level), 0.0 in air."""
    y = torch.arange(ny, dtype=torch.float32, device=device)
    prof = 0.5 * (1.0 - torch.tanh((y - water_level) / width))
    return prof.view(ny, 1).expand(ny, nx)


def _init_two_phase_cg_2d(
    ny: int,
    nx: int,
    water_level: float,
    rho_water: float,
    rho_air: float,
    obstacle: torch.Tensor,
    device: torch.device,
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
    ny: int,
    nx: int,
    water_level: float,
    rho_water: float,
    rho_air: float,
    obstacle: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SC two-component initialisation with smooth interface.

    CRITICAL for SCMC stability:
    1. Total density rho1+rho2 must be uniform.
    2. Sphere cells get LOCAL phase densities (not zeroed).
    3. solid_mask in collision must include the sphere.
    """
    prof = _smooth_profile_y(ny, nx, water_level, 4.0, device)
    # Component 2 = water (heavy), varies with profile
    rho2 = rho_water * prof + rho_air * (1.0 - prof)
    # Component 1 = air (light).  Uniform total density.
    rho_tot = rho_water + rho_air
    rho1 = rho_tot - rho2
    # NO zeroing at obstacle — let solid_mask handle it in collision
    zero = torch.zeros((ny, nx), device=device)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


# ---------------------------------------------------------------------------
# 3-D geometry helpers
# ---------------------------------------------------------------------------


def _sphere_mask_3d(
    nz: int,
    ny: int,
    nx: int,
    cx: float,
    cy: float,
    cz: float,
    r: float,
    device: torch.device,
) -> torch.Tensor:
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r**2


def _wall_mask_3d(nz: int, ny: int, nx: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    mask[0] = mask[-1] = True
    mask[:, 0] = mask[:, -1] = True
    mask[:, :, 0] = mask[:, :, -1] = True
    return mask


def _init_two_phase_3d(
    nz: int,
    ny: int,
    nx: int,
    water_level: float,
    rho_water: float,
    rho_air: float,
    obstacle: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SC two-component 3D initialisation.

    Same stability rules as 2D: constant total density, local phase densities
    at obstacle (not zeroed), solid_mask handles sphere in collision.
    """
    z = torch.arange(nz, dtype=torch.float32, device=device)
    prof = 0.5 * (1.0 - torch.tanh((z - water_level) / 3.0)).view(nz, 1, 1).expand(nz, ny, nx)
    # Component 2 = water (heavy)
    rho2 = rho_water * prof + rho_air * (1.0 - prof)
    # Component 1 = air.  Uniform total density.
    rho_tot = rho_water + rho_air
    rho1 = rho_tot - rho2
    # NO zeroing at obstacle
    zero = torch.zeros((nz, ny, nx), device=device)
    f1 = equilibrium3d(rho1, zero, zero, zero)
    f2 = equilibrium3d(rho2, zero, zero, zero)
    return f1, f2


# ---------------------------------------------------------------------------
# Force diagnostics
# ---------------------------------------------------------------------------


def _momentum_exchange_2d(
    f1: torch.Tensor,
    f2: torch.Tensor,
    solid: torch.Tensor,
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
    f1: torch.Tensor,
    f2: torch.Tensor,
    solid: torch.Tensor,
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
# Prescribed-velocity moving body (Ladd 1994)
# ---------------------------------------------------------------------------

_D2Q9_OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long)
_D3Q19_OPP = torch.tensor(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17],
    dtype=torch.long,
)
_D2Q9_W = torch.tensor(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=torch.float32,
)
_CS2 = 1.0 / 3.0


def _pull_source_in_2d(mask: torch.Tensor) -> torch.Tensor:
    """``(9, ny, nx)``: pull source ``x - c_q`` lies in ``mask`` (q=0: self)."""
    c = C2.to(mask.device)
    out = [mask]
    for q in range(1, 9):
        out.append(
            torch.roll(mask, shifts=(int(c[q, 1]), int(c[q, 0])), dims=(0, 1))
        )
    return torch.stack(out)


def _link_into_2d(mask: torch.Tensor) -> torch.Tensor:
    """``(9, ny, nx)``: downstream neighbour ``x + c_q`` lies in ``mask``."""
    c = C2.to(mask.device)
    out = [mask]
    for q in range(1, 9):
        out.append(
            torch.roll(mask, shifts=(-int(c[q, 1]), -int(c[q, 0])), dims=(0, 1))
        )
    return torch.stack(out)


def _pull_source_in_3d(mask: torch.Tensor) -> torch.Tensor:
    """``(19, nz, ny, nx)``: pull source ``x - c_q`` lies in ``mask``."""
    c = C3.to(mask.device)
    out = [mask]
    for q in range(1, 19):
        out.append(
            torch.roll(
                mask,
                shifts=(int(c[q, 2]), int(c[q, 1]), int(c[q, 0])),
                dims=(0, 1, 2),
            )
        )
    return torch.stack(out)


def _link_into_3d(mask: torch.Tensor) -> torch.Tensor:
    """``(19, nz, ny, nx)``: downstream neighbour ``x + c_q`` lies in ``mask``."""
    c = C3.to(mask.device)
    out = [mask]
    for q in range(1, 19):
        out.append(
            torch.roll(
                mask,
                shifts=(-int(c[q, 2]), -int(c[q, 1]), -int(c[q, 0])),
                dims=(0, 1, 2),
            )
        )
    return torch.stack(out)


def _moving_wall_bb_2d(f, f_pre, body, solid, uw_x, uw_y):
    """Fluid-side moving bounce-back (Ladd 1994) on body links, D2Q9.

    ``f`` is post-stream, ``f_pre`` post-collision (pre-stream) of the same
    component.  For a fluid cell whose pull source lies in the moving ``body``:
    ``f_q = f_pre[opp(q)] + 2 w_q rho (c_q . u_w) / cs²`` with the
    component's own pre-stream density.  Static walls are left to the
    runner's post-stream bounce-back.
    """
    device = f.device
    fluid = ~solid
    src = _pull_source_in_2d(body)
    link = fluid.unsqueeze(0) & src
    if not bool(link.any()):
        return f
    opp = _D2Q9_OPP.to(device)
    c = C2.to(device).float()
    w = _D2Q9_W.to(device).view(9, 1, 1)
    rho = f_pre.sum(dim=0)
    cuw = c[:, 0].view(9, 1, 1) * uw_x + c[:, 1].view(9, 1, 1) * uw_y
    f_refl = f_pre[opp] + 2.0 * w * rho.unsqueeze(0) * cuw / _CS2
    return torch.where(link, f_refl, f)


def _moving_wall_bb_3d(f, f_pre, body, solid, uw_x, uw_y, uw_z):
    """Fluid-side moving bounce-back (Ladd 1994) on body links, D3Q19."""
    from .d3q19 import W as W3

    device = f.device
    fluid = ~solid
    src = _pull_source_in_3d(body)
    link = fluid.unsqueeze(0) & src
    if not bool(link.any()):
        return f
    opp = _D3Q19_OPP.to(device)
    c = C3.to(device).float()
    w = W3.to(device).view(19, 1, 1, 1)
    rho = f_pre.sum(dim=0)
    cuw = (
        c[:, 0].view(19, 1, 1, 1) * uw_x
        + c[:, 1].view(19, 1, 1, 1) * uw_y
        + c[:, 2].view(19, 1, 1, 1) * uw_z
    )
    f_refl = f_pre[opp] + 2.0 * w * rho.unsqueeze(0) * cuw / _CS2
    return torch.where(link, f_refl, f)


def _refill_cleared_2d(f, cleared, fluid):
    """Refill cleared (solid→fluid) nodes with neighbour-averaged equilibrium."""
    if not bool(cleared.any()):
        return f
    device = f.device
    c = C2.to(device).float()
    rho = f.sum(dim=0)
    rho_safe = rho.clamp(min=1e-12)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    stacks = [(fluid, rho, ux, uy)]
    nb_f, nb_r, nb_x, nb_y = [], [], [], []
    for q in range(1, 9):
        shift = (int(c[q, 1]), int(c[q, 0]))
        nb_f.append(torch.roll(fluid, shifts=shift, dims=(0, 1)))
        nb_r.append(torch.roll(rho, shifts=shift, dims=(0, 1)))
        nb_x.append(torch.roll(ux, shifts=shift, dims=(0, 1)))
        nb_y.append(torch.roll(uy, shifts=shift, dims=(0, 1)))
    nb_f = torch.stack(nb_f).to(f.dtype)
    cnt = nb_f.sum(dim=0).clamp(min=1.0)
    rho_m = (torch.stack(nb_r).to(f.dtype) * nb_f).sum(dim=0) / cnt
    ux_m = (torch.stack(nb_x).to(f.dtype) * nb_f).sum(dim=0) / cnt
    uy_m = (torch.stack(nb_y).to(f.dtype) * nb_f).sum(dim=0) / cnt
    feq = equilibrium(rho_m, ux_m, uy_m)
    filled = (nb_f.sum(dim=0) > 0) & cleared
    return torch.where(filled.unsqueeze(0), feq, f)


def _refill_cleared_3d(f, cleared, fluid):
    """Refill cleared (solid→fluid) nodes with neighbour-averaged equilibrium."""
    if not bool(cleared.any()):
        return f
    device = f.device
    c = C3.to(device).float()
    rho = f.sum(dim=0)
    rho_safe = rho.clamp(min=1e-12)
    ux = (f * c[:, 0].view(19, 1, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(19, 1, 1, 1)).sum(dim=0) / rho_safe
    uz = (f * c[:, 2].view(19, 1, 1, 1)).sum(dim=0) / rho_safe
    nb_f, nb_r, nb_x, nb_y, nb_z = [], [], [], [], []
    for q in range(1, 19):
        shift = (int(c[q, 2]), int(c[q, 1]), int(c[q, 0]))
        nb_f.append(torch.roll(fluid, shifts=shift, dims=(0, 1, 2)))
        nb_r.append(torch.roll(rho, shifts=shift, dims=(0, 1, 2)))
        nb_x.append(torch.roll(ux, shifts=shift, dims=(0, 1, 2)))
        nb_y.append(torch.roll(uy, shifts=shift, dims=(0, 1, 2)))
        nb_z.append(torch.roll(uz, shifts=shift, dims=(0, 1, 2)))
    nb_f = torch.stack(nb_f).to(f.dtype)
    cnt = nb_f.sum(dim=0).clamp(min=1.0)
    rho_m = (torch.stack(nb_r).to(f.dtype) * nb_f).sum(dim=0) / cnt
    ux_m = (torch.stack(nb_x).to(f.dtype) * nb_f).sum(dim=0) / cnt
    uy_m = (torch.stack(nb_y).to(f.dtype) * nb_f).sum(dim=0) / cnt
    uz_m = (torch.stack(nb_z).to(f.dtype) * nb_f).sum(dim=0) / cnt
    feq = equilibrium3d(rho_m, ux_m, uy_m, uz_m)
    filled = (nb_f.sum(dim=0) > 0) & cleared
    return torch.where(filled.unsqueeze(0), feq, f)


def _momentum_exchange_moving_2d(f1_pre, f2_pre, body, fluid, uw_x, uw_y):
    """Moving-wall momentum-exchange force on the body (Ladd), D2Q9.

    ``F = Σ 2 c_i [f̃_i − w_i ρ (c_i . u_w)/cs²]`` over fluid cells and links
    entering the body, with post-collision (pre-stream) populations.  At
    ``u_w = 0`` this reduces to the static Ladd force.
    """
    device = f1_pre.device
    f_tot = f1_pre + f2_pre
    rho = f_tot.sum(dim=0)
    into = _link_into_2d(body)
    link = fluid.unsqueeze(0) & into
    c = C2.to(device).float()
    w = _D2Q9_W.to(device).view(9, 1, 1)
    cuw = c[:, 0].view(9, 1, 1) * uw_x + c[:, 1].view(9, 1, 1) * uw_y
    contrib = (f_tot - w * rho.unsqueeze(0) * cuw / _CS2) * link.to(f_tot.dtype)
    fx = 2.0 * float((c[:, 0].view(9, 1, 1) * contrib).sum().item())
    fy = 2.0 * float((c[:, 1].view(9, 1, 1) * contrib).sum().item())
    return fx, fy


def _momentum_exchange_moving_3d(f1_pre, f2_pre, body, fluid, uw_x, uw_y, uw_z):
    """Moving-wall momentum-exchange force on the body (Ladd), D3Q19."""
    from .d3q19 import W as W3

    device = f1_pre.device
    f_tot = f1_pre + f2_pre
    rho = f_tot.sum(dim=0)
    into = _link_into_3d(body)
    link = fluid.unsqueeze(0) & into
    c = C3.to(device).float()
    w = W3.to(device).view(19, 1, 1, 1)
    cuw = (
        c[:, 0].view(19, 1, 1, 1) * uw_x
        + c[:, 1].view(19, 1, 1, 1) * uw_y
        + c[:, 2].view(19, 1, 1, 1) * uw_z
    )
    contrib = (f_tot - w * rho.unsqueeze(0) * cuw / _CS2) * link.to(f_tot.dtype)
    fx = 2.0 * float((c[:, 0].view(19, 1, 1, 1) * contrib).sum().item())
    fy = 2.0 * float((c[:, 1].view(19, 1, 1, 1) * contrib).sum().item())
    fz = 2.0 * float((c[:, 2].view(19, 1, 1, 1) * contrib).sum().item())
    return fx, fy, fz


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _save_snapshot_2d(
    run_dir: Path,
    step: int,
    rho_water: torch.Tensor,
    rho_air: torch.Tensor,
    obstacle: torch.Tensor,
) -> None:
    phi = (rho_water / (rho_water + rho_air + 1e-12)).detach().cpu().numpy()
    obs_np = obstacle.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(phi, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.contour(obs_np, levels=[0.5], colors="red", linewidths=1.5)
    plt.colorbar(im, ax=ax, fraction=0.03, label="Water phase fraction φ")
    ax.set_title(f"Water entry – step {step:d}")
    out = flow_step_image_path(run_dir, step)
    fig.savefig(out, dpi=120)
    write_legacy_snapshot_alias(run_dir, step)
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
        config.output_root,
        "sphere_water_entry",
        config.resolved_run_name(),
        config.overwrite,
    )

    print(
        f"Running sphere water entry ({config.mode}, {config.model.upper()})  "
        f"device={device}  NX={config.nx}  NY={config.ny}"
        + (f"  NZ={config.nz}" if config.mode == "3d" else "")
        + f"  radius={config.radius}  G={config.G}  g={config.g:.1e}  steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    force_series: list[dict[str, object]] = []

    trajectory: list[dict[str, object]] = []
    if config.mode == "2d":
        ny, nx = config.ny, config.nx
        cx, cy = config.sphere_center_2d
        wall = _wall_mask_2d(ny, nx, device)
        sphere = _circle_mask(ny, nx, cx, cy, config.radius, device)
        solid = wall | sphere
        cy_float = float(cy)

        if config.model == "cg":
            f1, f2 = _init_two_phase_cg_2d(
                ny,
                nx,
                config.water_level,
                config.rho_water,
                config.rho_air,
                sphere,
                device,
            )
        else:  # sc
            f1, f2 = _init_two_phase_sc_2d(
                ny,
                nx,
                config.water_level,
                config.rho_water,
                config.rho_air,
                sphere,
                device,
            )

        gy = -config.g
        v_entry = float(config.v_entry)

        for step in range(1, config.n_steps + 1):
            # Prescribed rigid-body motion (down = -y), resting on the floor.
            cy_prev = cy_float
            cy_float = max(cy_float - v_entry, config.radius + 1.5)
            v_eff = cy_prev - cy_float
            sphere_new = _circle_mask(ny, nx, cx, cy_float, config.radius, device)
            if v_eff > 0.0:
                cleared = sphere & ~sphere_new & ~wall
                fresh = sphere_new & ~sphere
                f1 = _refill_cleared_2d(f1, cleared, ~(wall | sphere_new))
                f2 = _refill_cleared_2d(f2, cleared, ~(wall | sphere_new))
                if config.model == "cg":
                    # CG convention: no populations inside the obstacle.
                    f1 = torch.where(
                        fresh.unsqueeze(0), torch.zeros_like(f1), f1
                    )
                    f2 = torch.where(
                        fresh.unsqueeze(0), torch.zeros_like(f2), f2
                    )
                sphere = sphere_new
                solid = wall | sphere
            uw_y = -v_eff

            if config.model == "cg":
                # f1=red=heavy(water), f2=blue=light(air)
                f1, f2 = color_gradient_step(
                    f1,
                    f2,
                    tau=config.tau,
                    A=config.A_surface,
                    gy=gy,
                    solid_mask=solid,
                )
            else:  # sc
                f1, f2 = collide_sc_two_component(
                    f1,
                    f2,
                    G_12=config.G,
                    tau1=config.tau,
                    tau2=config.tau,
                    gy=gy,
                    solid_mask=solid,
                )

            f1_pre, f2_pre = f1, f2
            f1 = stream(f1)
            f2 = stream(f2)

            if v_eff > 0.0:
                f1 = _moving_wall_bb_2d(f1, f1_pre, sphere, solid, 0.0, uw_y)
                f2 = _moving_wall_bb_2d(f2, f2_pre, sphere, solid, 0.0, uw_y)

            if step % config.output_interval == 0 or step == config.n_steps:
                fx_s, fy_s = _momentum_exchange_moving_2d(
                    f1_pre, f2_pre, sphere, ~solid, 0.0, uw_y
                )

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
                trajectory.append({"step": step, "cy": round(cy_float, 4)})
                print(
                    f"step={step:5d}  Fx={fx_s:.4e}  Fy={fy_s:.4e}  "
                    f"mean_ρ_water={entry['mean_rho_water']:.4f}  cy={cy_float:.2f}"
                )
                _save_snapshot_2d(run_dir, step, rho_w, rho_a, sphere)

    else:
        # ---- 3-D simulation (SC two-component) ----
        nz, ny, nx = config.nz, config.ny, config.nx
        cx, cy, cz = config.sphere_center_3d
        wall = _wall_mask_3d(nz, ny, nx, device)
        sphere = _sphere_mask_3d(nz, ny, nx, cx, cy, cz, config.radius, device)
        solid = wall | sphere
        cz_float = float(cz)

        f1, f2 = _init_two_phase_3d(
            nz,
            ny,
            nx,
            config.water_level,
            config.rho_water,
            config.rho_air,
            sphere,
            device,
        )
        gz = -config.g
        v_entry = float(config.v_entry)

        for step in range(1, config.n_steps + 1):
            # Prescribed rigid-body motion (down = -z), resting on the floor.
            cz_prev = cz_float
            cz_float = max(cz_float - v_entry, config.radius + 1.5)
            v_eff = cz_prev - cz_float
            sphere_new = _sphere_mask_3d(
                nz, ny, nx, cx, cy, cz_float, config.radius, device
            )
            if v_eff > 0.0:
                cleared = sphere & ~sphere_new & ~wall
                f1 = _refill_cleared_3d(f1, cleared, ~(wall | sphere_new))
                f2 = _refill_cleared_3d(f2, cleared, ~(wall | sphere_new))
                sphere = sphere_new
                solid = wall | sphere
            uw_z = -v_eff

            f1, f2 = collide_sc_two_component_3d(
                f1,
                f2,
                G_12=config.G,
                tau1=config.tau,
                tau2=config.tau,
                gz=gz,
                solid_mask=solid,
            )
            f1_pre, f2_pre = f1, f2
            f1 = stream3d(f1)
            f2 = stream3d(f2)

            if v_eff > 0.0:
                f1 = _moving_wall_bb_3d(f1, f1_pre, sphere, solid, 0.0, 0.0, uw_z)
                f2 = _moving_wall_bb_3d(f2, f2_pre, sphere, solid, 0.0, 0.0, uw_z)

            if step % config.output_interval == 0 or step == config.n_steps:
                fx_s, fy_s, fz_s = _momentum_exchange_moving_3d(
                    f1_pre, f2_pre, sphere, ~solid, 0.0, 0.0, uw_z
                )

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
                trajectory.append({"step": step, "cz": round(cz_float, 4)})
                print(
                    f"step={step:5d}  Fx={fx_s:.4e}  Fy={fy_s:.4e}  Fz={fz_s:.4e}  "
                    f"cz={cz_float:.2f}"
                )

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
        "sphere_trajectory": trajectory,
        "note": (
            "Prescribed-velocity water entry: the sphere descends at v_entry "
            "(lattice units per step) with moving bounce-back. Impact force "
            "via the moving-wall Ladd momentum-exchange method. "
            "Fy (2-D) / Fz (3-D) is the vertical force on the sphere."
        ),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved metadata → {run_dir / 'run_metadata.json'}")
    return run_dir


__all__ = ["MultiphaseWaterEntryConfig", "run_multiphase_water_entry"]
