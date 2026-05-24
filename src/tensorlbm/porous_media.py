"""Porous-media gas-water displacement benchmarks for D2Q9 LBM.

Three canonical benchmarks are provided:

1. **Laplace pressure test** (:class:`LaplaceTestConfig` / :func:`run_laplace_test`):
   A circular gas bubble of radius *R* is placed inside a periodic water domain.
   At steady state the internal pressure exceeds the external pressure by
   σ/R (Young-Laplace equation in 2-D).  This benchmark validates the
   effective surface tension of the SC two-component model.

2. **Capillary invasion test** (:class:`CapillaryInvasionConfig` /
   :func:`run_capillary_invasion`):
   Gas invades a water-saturated tube driven by a small inlet pressure
   difference.  The invasion length follows the Washburn equation
   L(t) ∝ √t when capillary forces dominate viscous resistance.
   Contact-angle control is achieved via the SC wall-adsorption boundary
   condition (fluid-solid coupling :math:`G_{\\mathrm{ads}}`).

3. **Two-phase Poiseuille flow** (:class:`TwoPhasePoiseuilleConfig` /
   :func:`run_two_phase_poiseuille`):
   Two immiscible fluids occupy the lower and upper halves of a 2-D channel
   driven by a body force.  The steady-state velocity profile has an
   analytical solution involving the viscosity ratio; this benchmark
   validates momentum coupling between the two phases.

4. **Primary drainage in 2-D porous medium**
   (:class:`PorousDrainageConfig` / :func:`run_porous_drainage`):
   Gas (non-wetting phase) is injected into a water-saturated 2-D porous
   medium (random cylinder array).  The simulation tracks saturation and
   capillary number and compares the displacement front advance rate with
   the pore-volume-based estimate.

Geometry helpers
----------------
:func:`make_random_cylinder_medium`
    Generate a boolean solid mask for a 2-D domain packed with randomly
    placed non-overlapping cylinders.

:func:`make_tube_array_medium`
    Generate a boolean solid mask for a 2-D domain with a regular array of
    straight pore throats (channels).

Wettability
-----------
:func:`apply_wall_wettability_sc`
    Apply the SC fluid-solid interaction (adsorption BC) that sets the
    effective contact angle at solid walls.  A positive *G_ads* favours the
    first component (water) near the wall (water-wet), while a negative
    value makes the wall oil/gas-wet.

References
----------
Young (1805) Phil. Trans. R. Soc. 95 65
Laplace (1806) Mécanique Céleste, Supplément
Washburn (1921) Phys. Rev. 17 273
Shan & Chen (1993) Phys. Rev. E 47 1815
Pan et al. (2004) Phys. Rev. E 70 026702
Huang et al. (2007) J. Fluid Mech. 569 229
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import torch

from .boundaries import bounce_back_cells
from .d2q9 import C, W, equilibrium, macroscopic
from .multiphase import (
    collide_sc_two_component,
    color_gradient_step,
)
from .solver import stream
from .utils import prepare_run_dir, resolve_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CS2 = 1.0 / 3.0


def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


def _w_on(device: torch.device) -> torch.Tensor:
    return W.to(device)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def make_random_cylinder_medium(
    ny: int,
    nx: int,
    n_cylinders: int,
    r_min: float,
    r_max: float,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a 2-D porous medium made of randomly placed cylinders.

    The domain has periodic boundary conditions in x but solid walls at
    y = 0 and y = ny-1.  Cylinders are placed such that they do not
    overlap each other and do not touch the y-walls.

    Args:
        ny:          Number of lattice nodes in y.
        nx:          Number of lattice nodes in x.
        n_cylinders: Number of cylinder obstacles.
        r_min:       Minimum cylinder radius (lattice units).
        r_max:       Maximum cylinder radius (lattice units).
        seed:        Random seed for reproducibility.
        device:      PyTorch device.  Defaults to CPU.

    Returns:
        Boolean tensor of shape ``(ny, nx)`` — *True* at solid nodes.
    """
    if device is None:
        device = torch.device("cpu")

    rng = torch.Generator()
    rng.manual_seed(seed)

    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    # Horizontal walls
    solid[0, :] = True
    solid[-1, :] = True

    cy_list: list[float] = []
    cx_list: list[float] = []
    r_list: list[float] = []

    attempts_per_cylinder = 200
    for _ in range(n_cylinders):
        placed = False
        for _ in range(attempts_per_cylinder):
            r = float(r_min + (r_max - r_min) * torch.rand(1, generator=rng).item())
            margin = r + 2.0
            cx_c = float(margin + (nx - 2 * margin) * torch.rand(1, generator=rng).item())
            cy_c = float(margin + (ny - 2 * margin) * torch.rand(1, generator=rng).item())

            # Check overlap with existing cylinders (periodic in x)
            overlap = False
            for cx_e, cy_e, r_e in zip(cx_list, cy_list, r_list, strict=True):
                dx = abs(cx_c - cx_e)
                dx = min(dx, nx - dx)  # periodic wrap
                dy = abs(cy_c - cy_e)
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < r + r_e + 2.0:
                    overlap = True
                    break
            if overlap:
                continue

            # Stamp cylinder onto solid mask
            ys = torch.arange(ny, dtype=torch.float32, device=device)
            xs = torch.arange(nx, dtype=torch.float32, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            # Periodic wrap in x for cylinder membership
            dx_field = torch.abs(xx - cx_c)
            dx_field = torch.minimum(dx_field, torch.tensor(nx, dtype=torch.float32) - dx_field)
            inside = (dx_field ** 2 + (yy - cy_c) ** 2) <= r ** 2
            solid = solid | inside

            cx_list.append(cx_c)
            cy_list.append(cy_c)
            r_list.append(r)
            placed = True
            break

        if not placed:
            # Not enough space; stop adding cylinders
            break

    return solid


def make_tube_array_medium(
    ny: int,
    nx: int,
    n_tubes: int,
    tube_width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a 2-D domain with a regular array of straight pore tubes.

    The domain consists of *n_tubes* open channels separated by solid walls,
    with the tubes oriented in the x direction.  The domain has solid walls
    at y = 0 and y = ny-1.

    Args:
        ny:         Total domain height.
        nx:         Domain length.
        n_tubes:    Number of pore channels.
        tube_width: Width of each channel in lattice nodes.
        device:     PyTorch device.

    Returns:
        Boolean tensor of shape ``(ny, nx)`` — *True* at solid nodes.
    """
    if device is None:
        device = torch.device("cpu")

    solid = torch.ones((ny, nx), dtype=torch.bool, device=device)
    # Outer walls remain solid; open tubes periodically in y
    pitch = ny // (n_tubes + 1)
    for k in range(n_tubes):
        y_center = (k + 1) * pitch
        y_lo = max(1, y_center - tube_width // 2)
        y_hi = min(ny - 2, y_center + tube_width // 2)
        solid[y_lo : y_hi + 1, :] = False

    return solid


# ---------------------------------------------------------------------------
# Wall wettability (adsorption BC)
# ---------------------------------------------------------------------------

def apply_wall_wettability_sc(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    solid_mask: torch.Tensor,
    G_ads1: float = 0.3,
    G_ads2: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SC fluid-solid interaction to enforce wall wettability.

    The SC adsorption boundary condition modifies the density just
    outside solid walls to create an apparent contact angle.  When
    *G_ads1* > *G_ads2*, component 1 is preferred near the wall
    (component-1-wet wall, e.g., water-wet).

    For drainage simulations, component 1 is water (wetting phase) and
    component 2 is gas (non-wetting phase).  Setting ``G_ads1 > 0`` and
    ``G_ads2 = 0`` creates water-wet walls.

    **Note**: Both adsorption values must be non-negative.  Negative values
    create negative distributions at solid nodes which destabilise the
    simulation through streaming.  The asymmetry between the two components
    (not absolute magnitude) drives the apparent contact angle.

    Args:
        rho1:       Density of component 1 (wetting, water), shape ``(ny, nx)``.
        rho2:       Density of component 2 (non-wetting, gas), shape ``(ny, nx)``.
        solid_mask: Boolean solid mask ``(ny, nx)`` — *True* at solid nodes.
        G_ads1:     Adsorption pseudo-density for component 1 at solid nodes
                    (≥ 0; larger value → more water-wet).
        G_ads2:     Adsorption pseudo-density for component 2 at solid nodes
                    (≥ 0; default 0 → gas not adsorbed).

    Returns:
        Modified ``(rho1, rho2)`` with adsorption values set at solid nodes.
    """
    if G_ads1 < 0.0 or G_ads2 < 0.0:
        msg = "Adsorption values G_ads1 and G_ads2 must be non-negative"
        raise ValueError(msg)
    rho1 = rho1.clone()
    rho2 = rho2.clone()
    if G_ads1 != 0.0:
        rho1[solid_mask] = G_ads1
    if G_ads2 != 0.0:
        rho2[solid_mask] = G_ads2
    return rho1, rho2


# ---------------------------------------------------------------------------
# 1. Laplace pressure test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaplaceTestConfig:
    """Configuration for the Laplace pressure benchmark.

    A circular gas bubble of radius *bubble_radius* is placed at the centre
    of a periodic square domain.  The SC model is run until steady state and
    the pressure jump ΔP = P_inside − P_outside is compared with the
    Young-Laplace prediction σ/R.

    Attributes
    ----------
    nx, ny:          Domain size (square domain recommended).
    bubble_radius:   Gas-bubble radius in lattice units.
    G_12:            SC coupling constant (> 0 for phase separation).
    tau1, tau2:      Relaxation times for water and gas.
    rho_water:       Initial water density (outer phase).
    rho_gas:         Initial gas density (inner bubble).
    n_steps:         Number of time steps to reach steady state.
    output_interval: Diagnostic sampling interval.
    """

    nx: int = 100
    ny: int = 100
    bubble_radius: float = 20.0
    G_12: float = 0.9
    tau1: float = 1.0
    tau2: float = 1.0
    rho_water: float = 0.7
    rho_gas: float = 0.3
    n_steps: int = 5000
    output_interval: int = 1000
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.nx < 20 or self.ny < 20:
            msg = "nx and ny must be at least 20"
            raise ValueError(msg)
        if self.bubble_radius <= 0 or self.bubble_radius >= min(self.nx, self.ny) // 2:
            msg = "bubble_radius must be positive and smaller than half the domain"
            raise ValueError(msg)
        if self.tau1 <= 0.5 or self.tau2 <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.rho_water <= self.rho_gas:
            msg = "rho_water must exceed rho_gas"
            raise ValueError(msg)
        if self.G_12 <= 0:
            msg = "G_12 must be > 0 for SC phase separation"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"laplace_R{self.bubble_radius:.0f}_G{self.G_12:.2f}"
            f"_nx{self.nx}_steps{self.n_steps}"
        )


def _measure_laplace_pressure(
    f_water: torch.Tensor,
    f_gas: torch.Tensor,
    bubble_radius: float,
) -> tuple[float, float, float]:
    """Measure pressure inside and outside the bubble.

    Returns ``(p_inside, p_outside, delta_p)``.
    The lattice pressure is p = cs² ρ = ρ/3.
    """
    rho_w, _, _ = macroscopic(f_water)
    rho_g, _, _ = macroscopic(f_gas)

    ny, nx = rho_w.shape
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=rho_w.device),
        torch.arange(nx, dtype=torch.float32, device=rho_w.device),
        indexing="ij",
    )
    cy, cx = ny / 2.0, nx / 2.0
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    inside = r_field <= bubble_radius * 0.6
    outside = r_field >= bubble_radius * 1.4

    rho_total = rho_w + rho_g
    p_in = float((_CS2 * rho_total[inside]).mean().item())
    p_out = float((_CS2 * rho_total[outside]).mean().item())
    return p_in, p_out, p_in - p_out


def run_laplace_test(config: LaplaceTestConfig) -> dict[str, object]:
    """Run the Laplace pressure benchmark.

    At steady state the pressure jump across a circular gas bubble should
    satisfy the Young-Laplace equation:

        ΔP = σ / R   (2-D)

    where σ is the effective surface tension of the SC model.

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with keys ``delta_p``, ``bubble_radius``, ``sigma_eff``
        (= ΔP·R), and the full ``diagnostics`` list.
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "laplace_test", config.resolved_run_name(), config.overwrite
    )

    ny, nx = config.ny, config.nx
    cy, cx = ny / 2.0, nx / 2.0
    R = config.bubble_radius

    # Initial condition: gas bubble in water (periodic domain, no walls)
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float32, device=device),
        torch.arange(nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    r_field = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Smooth tanh interface to reduce initial transients
    width = 3.0
    alpha = 0.5 * (1.0 - torch.tanh((r_field - R) / width))  # 1 inside, 0 outside
    frac = 0.05

    rho_gas_field = config.rho_gas * alpha + config.rho_gas * frac * (1.0 - alpha)
    rho_water_field = config.rho_water * frac * alpha + config.rho_water * (1.0 - alpha)

    zero = torch.zeros((ny, nx), device=device)
    f_water = equilibrium(rho_water_field, zero, zero)
    f_gas = equilibrium(rho_gas_field, zero, zero)

    print(
        f"Laplace pressure test  NX={nx}  NY={ny}  R={R}  G={config.G_12}  "
        f"steps={config.n_steps}"
    )

    diagnostics: list[dict[str, object]] = []

    for step in range(1, config.n_steps + 1):
        # Periodic SC collision (no walls)
        f_water, f_gas = collide_sc_two_component(
            f_water, f_gas,
            G_12=config.G_12,
            tau1=config.tau1,
            tau2=config.tau2,
        )
        f_water = stream(f_water)
        f_gas = stream(f_gas)

        if step % config.output_interval == 0 or step == config.n_steps:
            p_in, p_out, dp = _measure_laplace_pressure(f_water, f_gas, R)
            sigma_eff = dp * R
            diag: dict[str, object] = {
                "step": step,
                "p_inside": round(p_in, 8),
                "p_outside": round(p_out, 8),
                "delta_p": round(dp, 8),
                "sigma_eff": round(sigma_eff, 8),
            }
            diagnostics.append(diag)
            print(
                f"step={step:5d}  ΔP={dp:.6f}  σ_eff={sigma_eff:.4f}"
            )

    final = diagnostics[-1]
    # Save metadata
    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "final_delta_p": final["delta_p"],
        "sigma_eff": final["sigma_eff"],
        "bubble_radius": R,
        "note": "ΔP = σ/R (Young-Laplace 2D). sigma_eff = ΔP * R.",
        "diagnostics": diagnostics,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Results saved → {run_dir / 'run_metadata.json'}")
    return metadata


# ---------------------------------------------------------------------------
# 2. Capillary invasion benchmark (Washburn)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapillaryInvasionConfig:
    """Configuration for the capillary invasion (Washburn) benchmark.

    Gas invades a water-filled straight tube of width *tube_width* driven
    by a small inlet pressure.  The Washburn equation predicts:

        L(t) = sqrt(R σ cos θ / (2 μ)) · √t

    where R is the tube half-width, σ is surface tension, θ is the contact
    angle, and μ is the dynamic viscosity of water.

    Attributes
    ----------
    nx, ny:          Domain size.
    tube_width:      Tube interior width in lattice nodes (interior cells).
    G_12:            SC coupling constant.
    G_ads_water:     Adsorption parameter for water at walls (> 0 → water-wet).
    G_ads_gas:       Adsorption parameter for gas at walls (< 0 → repelled).
    tau_water:       Relaxation time for water.
    tau_gas:         Relaxation time for gas.
    rho_water:       Water density.
    rho_gas:         Gas density.
    dp_inlet:        Pressure difference driving gas inlet (lattice units).
    n_steps:         Number of time steps.
    output_interval: Diagnostic sampling interval.
    """

    nx: int = 200
    ny: int = 30
    tube_width: int = 20
    G_12: float = 0.9
    G_ads_water: float = 0.3
    G_ads_gas: float = 0.0
    tau_water: float = 1.0
    tau_gas: float = 1.0
    rho_water: float = 0.7
    rho_gas: float = 0.3
    dp_inlet: float = 1e-4
    n_steps: int = 4000
    output_interval: int = 500
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.nx < 40 or self.ny < 10:
            msg = "nx must be ≥ 40 and ny ≥ 10"
            raise ValueError(msg)
        if self.tube_width < 4 or self.tube_width >= self.ny - 2:
            msg = "tube_width must be ≥ 4 and < ny - 2"
            raise ValueError(msg)
        if self.tau_water <= 0.5 or self.tau_gas <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.rho_water <= self.rho_gas:
            msg = "rho_water must exceed rho_gas"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"capillary_tw{self.tube_width}_G{self.G_12:.2f}"
            f"_Gads{self.G_ads_water:.2f}_nx{self.nx}_steps{self.n_steps}"
        )


def _capillary_wall_mask(ny: int, nx: int, tube_width: int, device: torch.device) -> torch.Tensor:
    """Solid mask for a single straight tube with top/bottom walls."""
    solid = torch.ones((ny, nx), dtype=torch.bool, device=device)
    y_center = ny // 2
    y_lo = y_center - tube_width // 2
    y_hi = y_center + tube_width // 2
    solid[y_lo : y_hi + 1, :] = False
    return solid


def _measure_invasion_front(
    f_water: torch.Tensor,
    f_gas: torch.Tensor,
    solid_mask: torch.Tensor,
) -> float:
    """Return the x-position of the gas-water invasion front."""
    rho_w, _, _ = macroscopic(f_water)
    rho_g, _, _ = macroscopic(f_gas)
    # Phase fraction of gas in each column (fluid nodes only)
    fluid_mask = ~solid_mask
    phi = rho_g / (rho_w + rho_g + 1e-12)
    # Average over interior y (fluid nodes)
    phi_col = (phi * fluid_mask.float()).sum(dim=0) / (
        fluid_mask.float().sum(dim=0).clamp(min=1)
    )
    gas_cols = (phi_col > 0.4).nonzero(as_tuple=True)[0]
    if gas_cols.numel() == 0:
        return 0.0
    return float(gas_cols.max().item())


def run_capillary_invasion(config: CapillaryInvasionConfig) -> dict[str, object]:
    """Run the capillary invasion (Washburn) benchmark.

    Gas is injected at the left boundary and invades a water-filled tube.
    The invasion length is tracked over time and compared with the
    Washburn √t prediction.

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with ``invasion_series`` (step, front_x, sqrt_t) and
        ``washburn_exponent`` estimated by linear regression on log-log data.
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "capillary_invasion", config.resolved_run_name(), config.overwrite
    )

    ny, nx = config.ny, config.nx
    solid = _capillary_wall_mask(ny, nx, config.tube_width, device)

    # Initial condition: tube filled with water, gas reservoir at left
    zero = torch.zeros((ny, nx), device=device)
    rho_w0 = torch.full((ny, nx), config.rho_water, device=device)
    rho_g0 = torch.full((ny, nx), config.rho_gas * 0.05, device=device)

    # Gas occupies the left 5 columns of the fluid zone (initial gas seed)
    gas_init_cols = max(2, nx // 20)
    rho_g0[:, :gas_init_cols] = config.rho_gas
    rho_w0[:, :gas_init_cols] = config.rho_water * 0.05

    f_water = equilibrium(rho_w0, zero, zero)
    f_gas = equilibrium(rho_g0, zero, zero)

    print(
        f"Capillary invasion  NX={nx}  NY={ny}  tube_width={config.tube_width}  "
        f"G={config.G_12}  G_ads_w={config.G_ads_water}  steps={config.n_steps}"
    )

    diagnostics: list[dict[str, object]] = []
    invasion_series: list[tuple[int, float, float]] = []

    for step in range(1, config.n_steps + 1):
        # Wettability: apply adsorption at solid nodes
        rho_w, _, _ = macroscopic(f_water)
        rho_g, _, _ = macroscopic(f_gas)
        rho_w, rho_g = apply_wall_wettability_sc(
            rho_w, rho_g, solid,
            G_ads1=config.G_ads_water, G_ads2=config.G_ads_gas,
        )
        # Write modified densities back into equilibrium at solid nodes so
        # SC force computation uses the adsorbed pseudo-density
        feq_w_wall = equilibrium(rho_w, zero, zero)
        feq_g_wall = equilibrium(rho_g, zero, zero)
        solid_4d = solid.unsqueeze(0)
        f_water = torch.where(solid_4d, feq_w_wall, f_water)
        f_gas = torch.where(solid_4d, feq_g_wall, f_gas)

        f_water, f_gas = collide_sc_two_component(
            f_water, f_gas,
            G_12=config.G_12,
            tau1=config.tau_water,
            tau2=config.tau_gas,
            solid_mask=solid,
        )
        f_water = stream(f_water)
        f_gas = stream(f_gas)
        f_water = bounce_back_cells(f_water, solid)
        f_gas = bounce_back_cells(f_gas, solid)

        # Re-inject gas at the left boundary (constant-pressure inlet)
        rho_inlet_w = config.rho_water * 0.05
        rho_inlet_g = config.rho_gas
        rho_w_col = torch.full((ny, 1), rho_inlet_w, device=device)
        rho_g_col = torch.full((ny, 1), rho_inlet_g, device=device)
        zero_2d = torch.zeros((ny, 1), device=device)
        f_water[:, :, 0:1] = equilibrium(rho_w_col, zero_2d, zero_2d)
        f_gas[:, :, 0:1] = equilibrium(rho_g_col, zero_2d, zero_2d)

        if step % config.output_interval == 0 or step == config.n_steps:
            x_front = _measure_invasion_front(f_water, f_gas, solid)
            sqrt_t = math.sqrt(float(step))
            invasion_series.append((step, x_front, sqrt_t))
            diag: dict[str, object] = {
                "step": step,
                "front_x": round(x_front, 3),
                "sqrt_t": round(sqrt_t, 4),
            }
            diagnostics.append(diag)
            print(f"step={step:5d}  front_x={x_front:.1f}  √t={sqrt_t:.2f}")

    # Estimate Washburn exponent by linear regression log(front) vs log(√t)
    washburn_exp = _estimate_washburn_exponent(invasion_series)

    # Save results
    front_csv = run_dir / "invasion_front.csv"
    with front_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "front_x", "sqrt_t"])
        writer.writerows(invasion_series)

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "invasion_series": diagnostics,
        "washburn_exponent": round(washburn_exp, 4),
        "note": (
            "Washburn equation: L ∝ √t → log-log slope ≈ 0.5. "
            "washburn_exponent is the measured log-log slope."
        ),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved → {run_dir / 'run_metadata.json'}  (Washburn exponent ≈ {washburn_exp:.3f})")
    return metadata


def _estimate_washburn_exponent(series: list[tuple[int, float, float]]) -> float:
    """Estimate Washburn exponent by linear regression on log-log data."""
    valid = [(s, x, sq) for s, x, sq in series if x > 2.0 and sq > 0.0]
    if len(valid) < 3:
        return float("nan")
    log_x = [math.log(x) for _, x, _ in valid]
    log_sq = [math.log(sq) for _, _, sq in valid]
    n = len(log_x)
    mean_lx = sum(log_x) / n
    mean_ls = sum(log_sq) / n
    num = sum((lx - mean_lx) * (ls - mean_ls) for lx, ls in zip(log_x, log_sq, strict=True))
    den = sum((ls - mean_ls) ** 2 for ls in log_sq)
    if abs(den) < 1e-15:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
# 3. Two-phase Poiseuille benchmark
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoPhasePoiseuilleConfig:
    """Configuration for the two-phase Poiseuille flow benchmark.

    Two immiscible fluids fill the lower half (water, y < ny/2) and upper
    half (gas/oil, y >= ny/2) of a 2-D channel driven by a body force G_x.
    At steady state the velocity profile is piecewise linear with a kink at
    the interface, and the two slopes satisfy the viscosity ratio:

        μ_water · du/dy|_water = μ_gas · du/dy|_gas  (stress continuity)

    Attributes
    ----------
    nx, ny:          Domain size.
    tau_water:       Relaxation time for water.
    tau_gas:         Relaxation time for gas.
    rho_water:       Water density.
    rho_gas:         Gas/oil density.
    G_x:             Body-force acceleration in x (lattice units).
    G_12:            SC coupling constant.
    n_steps:         Number of time steps to reach steady state.
    output_interval: Diagnostic sampling interval.
    """

    nx: int = 6
    ny: int = 40
    tau_water: float = 1.0
    tau_gas: float = 0.7
    rho_water: float = 0.7
    rho_gas: float = 0.3
    G_x: float = 5e-5
    G_12: float = 0.9
    n_steps: int = 8000
    output_interval: int = 2000
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.ny < 10:
            msg = "ny must be ≥ 10"
            raise ValueError(msg)
        if self.tau_water <= 0.5 or self.tau_gas <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.rho_water <= 0 or self.rho_gas <= 0:
            msg = "densities must be positive"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"two_phase_poiseuille_ny{self.ny}"
            f"_tau_w{self.tau_water:.2f}_tau_g{self.tau_gas:.2f}"
        )

    def nu_water(self) -> float:
        """Kinematic viscosity of water in lattice units."""
        return _CS2 * (self.tau_water - 0.5)

    def nu_gas(self) -> float:
        """Kinematic viscosity of gas in lattice units."""
        return _CS2 * (self.tau_gas - 0.5)


def run_two_phase_poiseuille(config: TwoPhasePoiseuilleConfig) -> dict[str, object]:
    """Run the two-phase Poiseuille benchmark.

    Simulates two immiscible fluids in a channel driven by a body force
    and compares the steady-state velocity profile with the analytical
    piecewise-linear solution.

    The analytical solution for a channel of total height H with the
    interface at y = H/2:

    * Lower half (water, 0 ≤ y ≤ H/2):
      u_water(y) = G_x / (2 μ_w) · y · (H - y · (1 + μ_w/μ_g))
    * Upper half (gas, H/2 ≤ y ≤ H):
      obtained by continuity and stress balance.

    The benchmark measures the relative L2 error between the simulated and
    analytical profiles (excluding wall nodes).

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with ``velocity_profile``, ``analytical_profile``, and
        ``l2_error_rel``.
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "two_phase_poiseuille",
        config.resolved_run_name(), config.overwrite,
    )

    ny, nx = config.ny, config.nx
    half = ny // 2

    # Walls at y=0 and y=ny-1; periodic in x
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    # Initial condition: water in lower half, gas in upper half
    rho_w0 = torch.zeros((ny, nx), device=device)
    rho_g0 = torch.zeros((ny, nx), device=device)
    rho_w0[:half, :] = config.rho_water
    rho_g0[half:, :] = config.rho_gas
    # Small minority fraction to avoid zero density
    frac = 0.05
    rho_w0[half:, :] = config.rho_water * frac
    rho_g0[:half, :] = config.rho_gas * frac

    zero = torch.zeros((ny, nx), device=device)
    f_water = equilibrium(rho_w0, zero, zero)
    f_gas = equilibrium(rho_g0, zero, zero)

    print(
        f"Two-phase Poiseuille  NY={ny}  NX={nx}  "
        f"τ_w={config.tau_water}  τ_g={config.tau_gas}  G_x={config.G_x}  "
        f"steps={config.n_steps}"
    )

    diagnostics: list[dict[str, object]] = []

    for step in range(1, config.n_steps + 1):
        f_water, f_gas = collide_sc_two_component(
            f_water, f_gas,
            G_12=config.G_12,
            tau1=config.tau_water,
            tau2=config.tau_gas,
            gx=config.G_x,
            solid_mask=wall,
        )
        f_water = stream(f_water)
        f_gas = stream(f_gas)
        f_water = bounce_back_cells(f_water, wall)
        f_gas = bounce_back_cells(f_gas, wall)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho_w, ux_w, _ = macroscopic(f_water)
            rho_g, ux_g, _ = macroscopic(f_gas)
            rho_tot = rho_w + rho_g
            rho_safe = rho_tot.clamp(min=1e-12)
            # Mixture velocity
            ux = (f_water.sum(dim=0) * ux_w + f_gas.sum(dim=0) * ux_g) / rho_safe
            ux_profile = ux[:, nx // 2].tolist()

            diag: dict[str, object] = {
                "step": step,
                "max_ux": round(float(ux[1:-1, :].max().item()), 8),
            }
            diagnostics.append(diag)
            print(f"step={step:5d}  max_ux={diag['max_ux']:.6f}")

    # Final velocity profile
    rho_w, ux_w, _ = macroscopic(f_water)
    rho_g, ux_g, _ = macroscopic(f_gas)
    rho_tot = (rho_w + rho_g).clamp(min=1e-12)
    ux_mix = (rho_w * ux_w + rho_g * ux_g) / rho_tot
    ux_profile = ux_mix[:, nx // 2].tolist()

    # Analytical solution (piecewise Poiseuille, Stokes + stress continuity)
    nu_w = config.nu_water()
    nu_g = config.nu_gas()
    mu_ratio = nu_w / nu_g  # μ_w / μ_g (same density → ν ratio = μ ratio)
    analytical = _two_phase_poiseuille_analytical(ny, half, config.G_x, nu_w, nu_g, mu_ratio)

    # L2 error (excluding wall nodes)
    sim_arr = torch.tensor(ux_profile[1:-1])
    ana_arr = torch.tensor(analytical[1:-1])
    l2_err = float((sim_arr - ana_arr).norm() / (ana_arr.norm().clamp(min=1e-15)))

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "velocity_profile": [round(v, 8) for v in ux_profile],
        "analytical_profile": [round(v, 8) for v in analytical],
        "l2_error_rel": round(l2_err, 6),
        "nu_water": round(nu_w, 6),
        "nu_gas": round(nu_g, 6),
        "diagnostics": diagnostics,
        "note": "l2_error_rel = ||sim - analytical|| / ||analytical||.",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved → {run_dir / 'run_metadata.json'}  (L2 error = {l2_err:.4f})")
    return metadata


def _two_phase_poiseuille_analytical(
    ny: int,
    half: int,
    G_x: float,
    nu_w: float,
    nu_g: float,
    mu_ratio: float,
) -> list[float]:
    """Compute the analytical two-phase Poiseuille velocity profile.

    The analytical solution assumes:
    * No-slip at walls (y=0 and y=ny-1).
    * Continuous velocity and shear stress at the interface (y = half).
    * Each phase occupies a half-channel of height H/2.

    The solution in each half is parabolic (Stokes flow driven by G_x):

    Lower half (water, 0 ≤ j ≤ half):
        u(j) = (G_x / (2 ν_w)) · j · (B - j)

    where B is found from the stress-balance condition.
    """
    # With Stokes: d²u/dy² = -G/ν in each layer.
    # u_w(y) = a_w y² + b_w y + c_w,  u_g(y) = a_g y² + b_g y + c_g
    # BCs:
    #   u_w(0) = 0                  (bottom wall)
    #   u_g(H) = 0                  (top wall, H = ny-1)
    #   u_w(h) = u_g(h)             (velocity continuity, h = half)
    #   ν_w du_w/dy(h) = ν_g du_g/dy(h)  (stress continuity)
    # where a_σ = -G_x / (2 ν_σ)

    H = float(ny - 1)
    h = float(half)
    aw = -G_x / (2.0 * nu_w) if nu_w > 0 else 0.0
    ag = -G_x / (2.0 * nu_g) if nu_g > 0 else 0.0

    # From BCs:
    # u_w(y) = aw y² + bw y         (c_w = 0)
    # u_g(y) = ag y² + bg y + cg
    # u_g(H) = 0 → cg = -ag H² - bg H
    # Stress: ν_w(2aw h + bw) = ν_g(2ag h + bg) → bw·ν_w - bg·ν_g = ν_g·2ag·h - ν_w·2aw·h
    # Velocity: aw h² + bw h = ag h² + bg h + cg = ag h² + bg h + (-ag H² - bg H)
    #         = ag(h² - H²) + bg(h - H)
    # => bw h - bg(h - H) = ag(h² - H²) - aw h²
    #    bw h + bg(H - h) = ag(h - H)(h + H) - aw h²
    # Two equations, two unknowns (bw, bg):
    # [ν_w, -ν_g] [bw]   [ν_g·2ag·h - ν_w·2aw·h   ]
    # [h,   H-h ] [bg] = [ag(h-H)(h+H) - aw·h²     ]

    # Solve 2×2 linear system
    A00, A01 = nu_w, -nu_g
    A10, A11 = h, H - h
    b0 = nu_g * 2.0 * ag * h - nu_w * 2.0 * aw * h
    b1 = ag * (h - H) * (h + H) - aw * h * h

    det = A00 * A11 - A01 * A10
    if abs(det) < 1e-20:
        # Degenerate; return zeros
        return [0.0] * ny
    bw = (b0 * A11 - b1 * A01) / det
    bg = (A00 * b1 - A10 * b0) / det
    cg = -ag * H * H - bg * H

    profile = []
    for j in range(ny):
        y = float(j)
        u = aw * y * y + bw * y if j <= half else ag * y * y + bg * y + cg
        profile.append(max(u, 0.0))  # clip negatives at walls
    return profile


# ---------------------------------------------------------------------------
# 4. Primary drainage in 2-D porous medium
# ---------------------------------------------------------------------------

MultiphaseModel2D = Literal["sc", "cg"]


@dataclass(frozen=True)
class PorousDrainageConfig:
    """Configuration for the 2-D porous medium primary drainage benchmark.

    Gas (non-wetting phase) is injected at the left boundary and displaces
    water (wetting phase) from a 2-D porous medium.  The medium is either:
    * ``geometry="random_cylinders"`` — randomly placed circular obstacles.
    * ``geometry="tube_array"`` — regular array of straight pore tubes.

    The benchmark tracks:
    * Water saturation S_w(t) = V_water / V_pore.
    * Gas saturation S_g(t) = 1 - S_w(t).
    * Breakthrough time t_BT (first step where gas reaches the outlet).

    Attributes
    ----------
    nx, ny:          Domain size.
    geometry:        Porous medium geometry (``"random_cylinders"`` or
                     ``"tube_array"``).
    n_cylinders:     Number of cylinders (for ``"random_cylinders"``).
    r_min, r_max:    Cylinder radii range (for ``"random_cylinders"``).
    n_tubes:         Number of tubes (for ``"tube_array"``).
    tube_width:      Tube width (for ``"tube_array"``).
    seed:            Random seed (for ``"random_cylinders"``).
    model:           Multiphase model (``"sc"`` or ``"cg"``).
    G_12:            SC coupling constant (or CG surface tension amplitude A).
    G_ads_water:     Wall adsorption for water (SC model, > 0 → water-wet).
    G_ads_gas:       Wall adsorption for gas (SC model, < 0 → gas repelled).
    tau_water:       Relaxation time for water.
    tau_gas:         Relaxation time for gas.
    rho_water:       Water density.
    rho_gas:         Gas density.
    n_steps:         Number of time steps.
    output_interval: Diagnostic sampling interval.
    """

    nx: int = 300
    ny: int = 100
    geometry: Literal["random_cylinders", "tube_array"] = "random_cylinders"
    n_cylinders: int = 20
    r_min: float = 4.0
    r_max: float = 8.0
    n_tubes: int = 4
    tube_width: int = 10
    seed: int = 42
    model: MultiphaseModel2D = "sc"
    G_12: float = 0.9
    G_ads_water: float = 0.3
    G_ads_gas: float = 0.0
    tau_water: float = 1.0
    tau_gas: float = 1.0
    rho_water: float = 0.7
    rho_gas: float = 0.3
    n_steps: int = 6000
    output_interval: int = 1000
    output_root: Path = Path("outputs")
    run_name: str | None = None
    device: str = "cpu"
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if self.nx < 40 or self.ny < 20:
            msg = "nx must be ≥ 40 and ny ≥ 20"
            raise ValueError(msg)
        if self.geometry not in ("random_cylinders", "tube_array"):
            msg = f"Unknown geometry: {self.geometry!r}"
            raise ValueError(msg)
        if self.model not in ("sc", "cg"):
            msg = f"Unknown model: {self.model!r}"
            raise ValueError(msg)
        if self.tau_water <= 0.5 or self.tau_gas <= 0.5:
            msg = "tau must be > 0.5"
            raise ValueError(msg)
        if self.rho_water <= self.rho_gas:
            msg = "rho_water must exceed rho_gas"
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (
            f"porous_drainage_{self.geometry}_{self.model}"
            f"_nx{self.nx}_ny{self.ny}_G{self.G_12:.2f}_steps{self.n_steps}"
        )


def _make_solid_mask(config: PorousDrainageConfig, device: torch.device) -> torch.Tensor:
    if config.geometry == "random_cylinders":
        return make_random_cylinder_medium(
            config.ny, config.nx,
            config.n_cylinders, config.r_min, config.r_max,
            config.seed, device,
        )
    # tube_array
    return make_tube_array_medium(config.ny, config.nx, config.n_tubes, config.tube_width, device)


def _measure_saturation(
    f_water: torch.Tensor,
    f_gas: torch.Tensor,
    solid_mask: torch.Tensor,
) -> tuple[float, float]:
    """Compute water and gas saturation in the pore space."""
    rho_w = f_water.sum(dim=0)
    rho_g = f_gas.sum(dim=0)
    fluid = ~solid_mask
    total_rho = (rho_w + rho_g)[fluid]
    water_rho = rho_w[fluid]
    S_w = float((water_rho / total_rho.clamp(min=1e-12)).mean().item())
    return S_w, 1.0 - S_w


def _gas_has_broken_through(
    f_gas: torch.Tensor,
    solid_mask: torch.Tensor,
    threshold: float = 0.3,
) -> bool:
    """Return True if gas has reached the right outlet."""
    rho_g = f_gas.sum(dim=0)
    rho_g_col = rho_g[:, -2]  # last fluid column
    fluid_col = (~solid_mask)[:, -2]
    if fluid_col.sum() == 0:
        return False
    avg = float(rho_g_col[fluid_col].mean().item())
    return avg > threshold


def _save_saturation_snapshot(
    run_dir: Path,
    step: int,
    f_water: torch.Tensor,
    f_gas: torch.Tensor,
    solid_mask: torch.Tensor,
    model: str,
) -> None:
    rho_w = f_water.sum(dim=0).detach().cpu().numpy()
    rho_g = f_gas.sum(dim=0).detach().cpu().numpy()
    total = rho_w + rho_g + 1e-12
    phi_gas = rho_g / total

    solid_np = solid_mask.cpu().numpy()
    phi_gas[solid_np] = float("nan")

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(phi_gas, origin="lower", cmap="RdBu_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label="Gas phase fraction")
    ax.set_title(f"Porous drainage ({model.upper()}) – step {step:d}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    out = run_dir / f"snapshot_{step:06d}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)


def run_porous_drainage(config: PorousDrainageConfig) -> dict[str, object]:
    """Run the 2-D porous medium primary drainage benchmark.

    Gas is continuously injected at the left boundary and water exits from
    the right boundary (open outlet).  The simulation tracks:

    * Water saturation S_w(t) vs. pore volumes injected.
    * Gas breakthrough time.
    * Phase distribution snapshots.

    Args:
        config: Benchmark configuration.

    Returns:
        Dictionary with ``saturation_series``, ``breakthrough_step``, and
        ``porosity`` of the generated medium.
    """
    config.validate()
    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root, "porous_drainage", config.resolved_run_name(), config.overwrite
    )

    ny, nx = config.ny, config.nx
    solid = _make_solid_mask(config, device)
    porosity = float((~solid).float().mean().item())
    print(
        f"Porous drainage  geometry={config.geometry}  model={config.model.upper()}  "
        f"NX={nx}  NY={ny}  porosity={porosity:.3f}  "
        f"G={config.G_12}  steps={config.n_steps}"
    )
    print(f"Run directory: {run_dir}")

    # Initial condition: domain filled with water
    frac = 0.05
    rho_w0 = torch.full((ny, nx), config.rho_water, device=device)
    rho_g0 = torch.full((ny, nx), config.rho_gas * frac, device=device)
    # Gas reservoir: first 3 fluid columns
    gas_cols = 3
    rho_g0[:, :gas_cols] = config.rho_gas
    rho_w0[:, :gas_cols] = config.rho_water * frac

    zero = torch.zeros((ny, nx), device=device)
    f_water = equilibrium(rho_w0, zero, zero)
    f_gas = equilibrium(rho_g0, zero, zero)

    diagnostics: list[dict[str, object]] = []
    saturation_series: list[tuple[int, float, float]] = []
    breakthrough_step: int | None = None

    for step in range(1, config.n_steps + 1):
        # --- wettability adsorption (SC model only) ---
        if config.model == "sc":
            rho_w, _, _ = macroscopic(f_water)
            rho_g, _, _ = macroscopic(f_gas)
            rho_w, rho_g = apply_wall_wettability_sc(
                rho_w, rho_g, solid,
                G_ads1=config.G_ads_water,
                G_ads2=config.G_ads_gas,
            )
            solid_4d = solid.unsqueeze(0)
            f_water = torch.where(solid_4d, equilibrium(rho_w, zero, zero), f_water)
            f_gas = torch.where(solid_4d, equilibrium(rho_g, zero, zero), f_gas)

        # --- collision ---
        if config.model == "sc":
            f_water, f_gas = collide_sc_two_component(
                f_water, f_gas,
                G_12=config.G_12,
                tau1=config.tau_water,
                tau2=config.tau_gas,
                solid_mask=solid,
            )
        else:  # cg
            A_surf = config.G_12 * 0.04
            f_water, f_gas = color_gradient_step(
                f_water, f_gas,
                tau=config.tau_water,
                A=A_surf,
                solid_mask=solid,
            )

        # --- streaming + bounce-back ---
        f_water = stream(f_water)
        f_gas = stream(f_gas)
        f_water = bounce_back_cells(f_water, solid)
        f_gas = bounce_back_cells(f_gas, solid)

        # --- gas injection at left boundary ---
        rho_inlet_w = config.rho_water * frac
        rho_inlet_g = config.rho_gas
        rho_w_col = torch.full((ny, 1), rho_inlet_w, device=device)
        rho_g_col = torch.full((ny, 1), rho_inlet_g, device=device)
        zero_2d = torch.zeros((ny, 1), device=device)
        f_water[:, :, 0:1] = equilibrium(rho_w_col, zero_2d, zero_2d)
        f_gas[:, :, 0:1] = equilibrium(rho_g_col, zero_2d, zero_2d)

        # --- water outlet at right boundary (zero-gradient / open) ---
        f_water[:, :, -1] = f_water[:, :, -2]
        f_gas[:, :, -1] = f_gas[:, :, -2]

        # --- check breakthrough ---
        if breakthrough_step is None and _gas_has_broken_through(f_gas, solid):
            breakthrough_step = step
            print(f"  *** Gas breakthrough at step {step} ***")

        # --- diagnostics ---
        if step % config.output_interval == 0 or step == config.n_steps:
            S_w, S_g = _measure_saturation(f_water, f_gas, solid)
            saturation_series.append((step, S_w, S_g))
            diag: dict[str, object] = {
                "step": step,
                "S_water": round(S_w, 6),
                "S_gas": round(S_g, 6),
            }
            diagnostics.append(diag)
            print(f"step={step:5d}  S_w={S_w:.4f}  S_g={S_g:.4f}")
            _save_saturation_snapshot(run_dir, step, f_water, f_gas, solid, config.model)

    # Save saturation CSV
    sat_csv = run_dir / "saturation.csv"
    with sat_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "S_water", "S_gas"])
        writer.writerows(saturation_series)

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "porosity": round(porosity, 6),
        "breakthrough_step": breakthrough_step,
        "saturation_series": diagnostics,
        "note": (
            "Primary drainage: gas (non-wetting) displaces water (wetting). "
            "S_w + S_g = 1 at each step."
        ),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Saved → {run_dir / 'run_metadata.json'}  "
        f"(breakthrough={'step ' + str(breakthrough_step) if breakthrough_step else 'not reached'})"
    )
    return metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Geometry
    "make_random_cylinder_medium",
    "make_tube_array_medium",
    # Wettability
    "apply_wall_wettability_sc",
    # Laplace pressure test
    "LaplaceTestConfig",
    "run_laplace_test",
    # Capillary invasion
    "CapillaryInvasionConfig",
    "run_capillary_invasion",
    # Two-phase Poiseuille
    "TwoPhasePoiseuilleConfig",
    "run_two_phase_poiseuille",
    # Porous drainage
    "PorousDrainageConfig",
    "run_porous_drainage",
]
