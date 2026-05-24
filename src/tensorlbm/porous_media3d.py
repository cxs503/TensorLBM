"""3D porous-media gas-water displacement benchmarks for D3Q19 LBM.

This module provides a 3D analogue of :mod:`porous_media`, offering:

1. **Geometry helpers** — :func:`make_random_sphere_medium` places randomly
   distributed non-overlapping spheres in a 3D domain to create a realistic
   porous-rock-like structure; :func:`make_tube_array_medium_3d` constructs
   a regular array of straight pore channels oriented in the x-direction.

2. **Drainage benchmark** (:class:`PorousDrainageConfig3D` /
   :func:`run_porous_drainage_3d`) — gas (non-wetting phase) is injected
   into a water-saturated 3D porous medium via an inlet pressure difference.
   Saturation and capillary number are tracked over time.

The SC two-component model is used for multiphase physics and the same
:func:`~tensorlbm.porous_media.apply_wall_wettability_sc` boundary condition
controls contact-angle via the adsorption parameter ``G_ads``.

References
----------
Shan & Chen (1993) Phys. Rev. E 47 1815
Pan et al. (2004) Phys. Rev. E 70 026702
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .boundaries3d import (
    bounce_back_cells_3d,
    zou_he_inlet_velocity_z,
    zou_he_outlet_pressure_z,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .multiphase3d import (
    collide_sc_two_component_3d,
)
from .porous_media import apply_wall_wettability_sc
from .solver3d import stream3d
from .utils import prepare_run_dir, resolve_device

__all__ = [
    "make_random_sphere_medium",
    "make_tube_array_medium_3d",
    "PorousDrainageConfig3D",
    "run_porous_drainage_3d",
]

_CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def make_random_sphere_medium(
    nz: int,
    ny: int,
    nx: int,
    n_spheres: int,
    r_min: float,
    r_max: float,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a 3D porous medium made of randomly placed non-overlapping spheres.

    The domain has solid walls at z = 0 and z = nz−1 (the inlet/outlet faces)
    and solid walls at all four lateral faces (±x, ±y).  Spheres are placed
    randomly within the interior such that they do not overlap each other and
    do not touch any wall.

    Args:
        nz:         Number of lattice nodes in z (flow direction).
        ny:         Number of lattice nodes in y.
        nx:         Number of lattice nodes in x.
        n_spheres:  Target number of sphere obstacles.
        r_min:      Minimum sphere radius (lattice units).
        r_max:      Maximum sphere radius (lattice units).
        seed:       Random seed for reproducibility.
        device:     PyTorch device.  Defaults to CPU.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)`` — *True* at solid nodes.
    """
    if device is None:
        device = torch.device("cpu")

    rng = torch.Generator()
    rng.manual_seed(seed)

    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    # Solid walls on all six faces
    solid[0, :, :] = True
    solid[-1, :, :] = True
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[:, :, 0] = True
    solid[:, :, -1] = True

    cz_list: list[float] = []
    cy_list: list[float] = []
    cx_list: list[float] = []
    r_list: list[float] = []

    attempts_per_sphere = 300
    for _ in range(n_spheres):
        placed = False
        for _ in range(attempts_per_sphere):
            r = float(r_min + (r_max - r_min) * torch.rand(1, generator=rng).item())
            margin = r + 2.0
            cz_c = float(margin + (nz - 2 * margin) * torch.rand(1, generator=rng).item())
            cy_c = float(margin + (ny - 2 * margin) * torch.rand(1, generator=rng).item())
            cx_c = float(margin + (nx - 2 * margin) * torch.rand(1, generator=rng).item())

            # Check overlap with existing spheres
            overlap = False
            for cx_e, cy_e, cz_e, r_e in zip(cx_list, cy_list, cz_list, r_list, strict=True):
                dx = cx_c - cx_e
                dy = cy_c - cy_e
                dz = cz_c - cz_e
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < r + r_e + 2.0:
                    overlap = True
                    break
            if overlap:
                continue

            # Stamp sphere onto solid mask
            zs = torch.arange(nz, dtype=torch.float32, device=device)
            ys = torch.arange(ny, dtype=torch.float32, device=device)
            xs = torch.arange(nx, dtype=torch.float32, device=device)
            zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
            inside = (
                (xx - cx_c) ** 2 + (yy - cy_c) ** 2 + (zz - cz_c) ** 2
            ) <= r ** 2
            solid = solid | inside

            cz_list.append(cz_c)
            cy_list.append(cy_c)
            cx_list.append(cx_c)
            r_list.append(r)
            placed = True
            break

        if not placed:
            break

    return solid


def make_tube_array_medium_3d(
    nz: int,
    ny: int,
    nx: int,
    n_tubes_y: int,
    n_tubes_x: int,
    tube_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a 3D porous medium with a regular array of straight pore tubes.

    Tubes are oriented in the z-direction (flow direction).  The domain has
    ``n_tubes_y × n_tubes_x`` pore channels arranged in a regular lattice.

    Args:
        nz:          Domain depth (flow direction, z).
        ny:          Domain height (y).
        nx:          Domain width (x).
        n_tubes_y:   Number of tube rows in the y-direction.
        n_tubes_x:   Number of tube columns in the x-direction.
        tube_width:  Diameter of each tube in lattice nodes.
        device:      PyTorch device.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)`` — *True* at solid nodes.
    """
    if device is None:
        device = torch.device("cpu")

    solid = torch.ones((nz, ny, nx), dtype=torch.bool, device=device)
    # Open a grid of circular tubes oriented in z
    pitch_y = ny // (n_tubes_y + 1)
    pitch_x = nx // (n_tubes_x + 1)
    radius = tube_width // 2

    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    for row in range(n_tubes_y):
        cy_c = float((row + 1) * pitch_y)
        for col in range(n_tubes_x):
            cx_c = float((col + 1) * pitch_x)
            inside = ((xx - cx_c) ** 2 + (yy - cy_c) ** 2) <= float(radius) ** 2
            # Open tube in all z-slices (excluding face walls)
            solid[1:-1, :, :] &= ~inside.unsqueeze(0)

    # Face walls
    solid[0, :, :] = True
    solid[-1, :, :] = True

    return solid


# ---------------------------------------------------------------------------
# Drainage benchmark
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PorousDrainageConfig3D:
    """Configuration for the 3D porous drainage benchmark.

    Gas (non-wetting phase) is injected from z = 0 into a water-saturated
    3D porous medium at a constant inlet velocity.  Saturation is tracked
    over time.

    Attributes
    ----------
    nz, ny, nx:      Domain dimensions.
    medium:          Pore geometry — ``"random_spheres"`` or ``"tube_array"``.
    n_spheres:       Target sphere count (``"random_spheres"`` only).
    r_min, r_max:    Sphere radius range (``"random_spheres"`` only).
    n_tubes_y, n_tubes_x, tube_width:
                     Tube array parameters (``"tube_array"`` only).
    G_12:            SC coupling constant.
    G_ads:           Wall adsorption parameter for water (≥ 0).
    tau_water:       Relaxation time for water.
    tau_gas:         Relaxation time for gas.
    rho_water:       Initial water density.
    rho_gas:         Initial gas density.
    u_inlet:         Gas injection velocity at z=0 (lattice units).
    n_steps:         Number of time steps.
    output_interval: Diagnostic sampling interval.
    seed:            Random seed for geometry generation.
    """

    nz: int = 40
    ny: int = 24
    nx: int = 24
    medium: str = "random_spheres"
    n_spheres: int = 8
    r_min: float = 2.0
    r_max: float = 4.0
    n_tubes_y: int = 2
    n_tubes_x: int = 2
    tube_width: int = 4
    G_12: float = 0.9
    G_ads: float = 0.3
    tau_water: float = 1.0
    tau_gas: float = 1.0
    rho_water: float = 0.7
    rho_gas: float = 0.3
    u_inlet: float = 0.005
    n_steps: int = 2000
    output_interval: int = 500
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.nz < 10 or self.ny < 8 or self.nx < 8:
            msg = "nz, ny, nx must be at least 10, 8, 8"
            raise ValueError(msg)
        if self.tau_water <= 0.5 or self.tau_gas <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.rho_water <= self.rho_gas:
            msg = "rho_water must exceed rho_gas"
            raise ValueError(msg)
        if self.G_12 <= 0:
            msg = "G_12 must be > 0"
            raise ValueError(msg)
        if self.medium not in {"random_spheres", "tube_array"}:
            msg = f"medium must be 'random_spheres' or 'tube_array', got {self.medium!r}"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"3d_{self.medium}_G{self.G_12:.2f}_nz{self.nz}"
            f"_ny{self.ny}_nx{self.nx}_steps{self.n_steps}"
        )


def _gas_saturation_3d(
    f_water: torch.Tensor,
    f_gas: torch.Tensor,
    solid_mask: torch.Tensor,
) -> float:
    """Compute gas saturation (fraction of pore volume occupied by gas)."""
    rho_w, _, _, _ = macroscopic3d(f_water)
    rho_g, _, _, _ = macroscopic3d(f_gas)
    fluid_mask = ~solid_mask
    denom = rho_w + rho_g
    phi = torch.where(denom > 1e-12, rho_g / denom, torch.zeros_like(rho_g))
    gas_vol = (phi * fluid_mask.float()).sum()
    total_vol = fluid_mask.float().sum().clamp(min=1)
    return float((gas_vol / total_vol).item())


def run_porous_drainage_3d(config: PorousDrainageConfig3D) -> dict[str, object]:
    """Run the 3D porous-media drainage benchmark.

    Gas (non-wetting phase) is injected from the z=0 face into a water-
    saturated 3D porous medium.  Saturation is tracked over time.

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with ``saturation_series`` (step, gas_saturation) and
        the final solid fraction (porosity = 1 − solid_fraction).
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "porous_drainage_3d",
        config.resolved_run_name(),
        config.overwrite,
    )

    nz, ny, nx = config.nz, config.ny, config.nx

    # Build solid mask
    if config.medium == "random_spheres":
        solid = make_random_sphere_medium(
            nz, ny, nx,
            config.n_spheres, config.r_min, config.r_max,
            seed=config.seed, device=device,
        )
    else:
        solid = make_tube_array_medium_3d(
            nz, ny, nx,
            config.n_tubes_y, config.n_tubes_x, config.tube_width,
            device=device,
        )

    porosity = float((~solid).float().mean().item())

    # Initial condition: domain filled with water
    zero3 = torch.zeros((nz, ny, nx), device=device)
    rho_w0 = torch.full((nz, ny, nx), config.rho_water, device=device)
    rho_g0 = torch.full((nz, ny, nx), config.rho_gas * 0.05, device=device)
    # Seed gas at the first 2 fluid z-layers
    rho_g0[1:3, :, :] = config.rho_gas
    rho_w0[1:3, :, :] = config.rho_water * 0.05

    f_water = equilibrium3d(rho_w0, zero3, zero3, zero3)
    f_gas = equilibrium3d(rho_g0, zero3, zero3, zero3)

    print(
        f"3D porous drainage  NZ={nz}  NY={ny}  NX={nx}  "
        f"medium={config.medium}  porosity={porosity:.3f}  "
        f"G={config.G_12}  steps={config.n_steps}"
    )

    saturation_series: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for step in range(1, config.n_steps + 1):
        # Wettability adsorption at solid nodes (water-wet)
        rho_w, _, _, _ = macroscopic3d(f_water)
        rho_g, _, _, _ = macroscopic3d(f_gas)

        # apply_wall_wettability_sc works on (ny, nx) slices; apply per z-slice
        for iz in range(nz):
            rw_s, rg_s = apply_wall_wettability_sc(
                rho_w[iz], rho_g[iz], solid[iz],
                G_ads1=config.G_ads, G_ads2=0.0,
            )
            rho_w[iz] = rw_s
            rho_g[iz] = rg_s

        # Inject modified densities into solid nodes as adsorption pseudo-distribution
        feq_w_ads = equilibrium3d(rho_w, zero3, zero3, zero3)
        feq_g_ads = equilibrium3d(rho_g, zero3, zero3, zero3)
        solid_5d = solid.unsqueeze(0)
        f_water = torch.where(solid_5d, feq_w_ads, f_water)
        f_gas = torch.where(solid_5d, feq_g_ads, f_gas)

        # SC collision
        f_water, f_gas = collide_sc_two_component_3d(
            f_water, f_gas,
            G_12=config.G_12,
            tau1=config.tau_water,
            tau2=config.tau_gas,
        )
        f_water = stream3d(f_water)
        f_gas = stream3d(f_gas)

        # Bounce-back at solid walls
        f_water = bounce_back_cells_3d(f_water, solid)
        f_gas = bounce_back_cells_3d(f_gas, solid)

        # Gas injection: Zou-He inlet at z=0 for gas (constant upward velocity),
        # outlet at z=nz-1 with prescribed low-pressure outlet
        f_gas = zou_he_inlet_velocity_z(f_gas, uz_in=config.u_inlet)
        f_gas = zou_he_outlet_pressure_z(f_gas, rho_out=config.rho_gas * 0.05)

        if step % config.output_interval == 0 or step == config.n_steps:
            sat = _gas_saturation_3d(f_water, f_gas, solid)
            saturation_series.append({"step": step, "gas_saturation": round(sat, 6)})
            diagnostics.append({"step": step, "gas_saturation": round(sat, 6)})
            print(f"step={step:5d}  gas_saturation={sat:.4f}")

    # Compute solid fraction and porosity
    solid_fraction = float(solid.float().mean().item())

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "porosity": porosity,
        "solid_fraction": solid_fraction,
        "saturation_series": saturation_series,
        "diagnostics": diagnostics,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Write CSV
    with (run_dir / "saturation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "gas_saturation"])
        for row in saturation_series:
            writer.writerow([row["step"], row["gas_saturation"]])

    print(f"Results saved → {run_dir / 'run_metadata.json'}")
    return metadata
