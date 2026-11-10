"""Pore-scale flow solver for TensorLBM — Darcy permeability verification suite.

Implements a cross-scale validation ladder for pore-scale flow in TensorLBM:

1. **Ideal geometries** — parallel plates, cylindrical tube, periodic array
   of cylinders/spheres.  Analytical Darcy permeability is known:
   - Parallel plates (2D): k = h²/12  (h = gap width)
   - Cylindrical tube (3D): k = R²/8  (R = tube radius)

2. **Random porous media** — Gaussian random field thresholding and sphere
   pack generation.  Permeability–porosity relation validated against the
   Kozeny–Carman model:
       k = (φ³ / (K·S²·(1-φ)²))
   where φ = porosity, S = specific surface area, K ≈ 5 (Kozeny constant).

3. **CT-scan-like synthetic geometry** — layered porous medium with
   heterogeneous porosity distribution mimicking real rock CT data.

Darcy permeability is computed from the steady-state flow rate under a
body force:
    k = μ·U / F
where U = average velocity, μ = dynamic viscosity, F = body force per unit mass.

Mass conservation is verified by monitoring total mass drift:
    |M(t) - M(0)| / M(0) ≤ 1e-6

Reynolds number for pore-scale flow:
    Re = ρ·U·d_p / μ
where d_p = characteristic pore diameter.  Darcy's law holds for Re < 1.

References
----------
Darcy (1856) Les Fontaines Publiques de la Ville de Dijon
Kozeny (1927) Sitzungsber. Akad. Wiss. Wien 136, 271
Carman (1937) Trans. Inst. Chem. Eng. 15, 150
Sukop & Thorne (2006) Lattice Boltzmann Modeling, §6.3
Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .d3q19 import C as C3D, OPPOSITE as OPPOSITE3D, equilibrium3d, macroscopic3d
from .d2q9 import C as C2D, OPPOSITE, W, equilibrium, macroscopic
from .boundaries3d import bounce_back_cells_3d
from .boundaries import bounce_back_cells
from .solver import stream
from .solver3d import stream3d
from .utils import resolve_device

__all__ = [
    "make_parallel_plates_2d",
    "make_cylindrical_tube_3d",
    "make_periodic_cylinder_array_2d",
    "make_sphere_pack_3d",
    "make_gaussian_random_field_3d",
    "make_ct_like_rock_3d",
    "PoreFlowConfig2D",
    "PoreFlowConfig3D",
    "run_pore_flow_2d",
    "run_pore_flow_3d",
    "compute_permeability",
    "kozeny_carman_permeability",
    "compute_porosity",
    "compute_specific_surface",
    "compute_pore_reynolds",
    "analytical_parallel_plate_permeability",
    "analytical_tube_permeability",
]

_CS2 = 1.0 / 3.0
_RHO_REF = 1.0


# ===========================================================================
# Geometry generators
# ===========================================================================


def make_parallel_plates_2d(
    ny: int,
    nx: int,
    gap: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """2D parallel-plate channel. Walls at y=0 and y=ny-1 are solid."""
    if device is None:
        device = torch.device("cpu")
    solid = torch.ones((ny, nx), dtype=torch.bool, device=device)
    y_start = (ny - gap) // 2
    y_end = y_start + gap
    solid[y_start:y_end, :] = False
    return solid


def make_cylindrical_tube_3d(
    nz: int,
    ny: int,
    nx: int,
    radius: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """3D cylindrical tube. Tube axis along z."""
    if device is None:
        device = torch.device("cpu")
    solid = torch.ones((nz, ny, nx), dtype=torch.bool, device=device)
    cy, cx = ny / 2.0, nx / 2.0
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
    solid[:, :, :] &= ~inside.unsqueeze(0)
    return solid


def make_periodic_cylinder_array_2d(
    ny: int,
    nx: int,
    n_cyl_y: int,
    n_cyl_x: int,
    cyl_radius: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """2D periodic array of circular cylinders."""
    if device is None:
        device = torch.device("cpu")
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    pitch_y = ny / (n_cyl_y + 1)
    pitch_x = nx / (n_cyl_x + 1)
    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for iy in range(1, n_cyl_y + 1):
        for ix in range(1, n_cyl_x + 1):
            cy = iy * pitch_y
            cx = ix * pitch_x
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= cyl_radius ** 2
            solid |= inside
    return solid


def make_sphere_pack_3d(
    nz: int,
    ny: int,
    nx: int,
    n_spheres: int,
    r_min: float,
    r_max: float,
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """3D random sphere pack (non-overlapping)."""
    if device is None:
        device = torch.device("cpu")
    rng = torch.Generator()
    rng.manual_seed(seed)
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)

    centers: list[tuple[float, float, float]] = []
    radii: list[float] = []

    for _ in range(n_spheres):
        placed = False
        for _attempt in range(500):
            r = float(r_min + (r_max - r_min) * torch.rand(1, generator=rng).item())
            margin = r + 1.5
            cz = float(margin + (nz - 2 * margin) * torch.rand(1, generator=rng).item())
            cy = float(margin + (ny - 2 * margin) * torch.rand(1, generator=rng).item())
            cx = float(margin + (nx - 2 * margin) * torch.rand(1, generator=rng).item())

            overlap = False
            for (ex, ey, ez, er) in zip(
                [c[0] for c in centers], [c[1] for c in centers],
                [c[2] for c in centers], radii,
            ):
                dist = math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2 + (cz - ez) ** 2)
                if dist < r + er + 1.0:
                    overlap = True
                    break
            if overlap:
                continue

            zs = torch.arange(nz, dtype=torch.float32, device=device)
            ys = torch.arange(ny, dtype=torch.float32, device=device)
            xs = torch.arange(nx, dtype=torch.float32, device=device)
            zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
            inside = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= r ** 2
            solid |= inside
            centers.append((cx, cy, cz))
            radii.append(r)
            placed = True
            break
        if not placed:
            break
    return solid


def _make_gaussian_filter_3d(nz, ny, nx, sigma, device):
    """3D Gaussian filter kernel in Fourier space."""
    kz_vals = torch.fft.fftfreq(nz, d=1.0) * nz
    ky_vals = torch.fft.fftfreq(ny, d=1.0) * ny
    kx_vals = torch.fft.fftfreq(nx, d=1.0) * nx
    kz, ky, kx = torch.meshgrid(
        kz_vals.to(device), ky_vals.to(device), kx_vals.to(device), indexing="ij",
    )
    k_sq = kx ** 2 + ky ** 2 + kz ** 2
    return torch.exp(-0.5 * sigma ** 2 * k_sq)


def make_gaussian_random_field_3d(
    nz: int, ny: int, nx: int,
    correlation_length: float, threshold: float,
    seed: int = 42, device: torch.device | None = None,
) -> torch.Tensor:
    """Porous medium from thresholded Gaussian random field."""
    if device is None:
        device = torch.device("cpu")
    rng = torch.Generator()
    rng.manual_seed(seed)
    noise = torch.randn((nz, ny, nx), dtype=torch.float32, device=device, generator=rng)
    filter_kernel = _make_gaussian_filter_3d(nz, ny, nx, correlation_length, device)
    noise_fft = torch.fft.fftn(noise)
    filtered = torch.fft.ifftn(noise_fft * filter_kernel).real
    filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-12)
    solid = filtered > threshold
    solid[0, :, :] = False
    solid[-1, :, :] = False
    return solid


def make_ct_like_rock_3d(
    nz: int, ny: int, nx: int,
    base_porosity: float = 0.25, heterogeneity: float = 0.1,
    seed: int = 42, device: torch.device | None = None,
) -> torch.Tensor:
    """CT-scan-like synthetic rock geometry."""
    if device is None:
        device = torch.device("cpu")
    rng = torch.Generator()
    rng.manual_seed(seed)
    noise = torch.randn((nz, ny, nx), dtype=torch.float32, device=device, generator=rng)
    sigma = 4.0
    filter_kernel = _make_gaussian_filter_3d(nz, ny, nx, sigma, device)
    noise_fft = torch.fft.fftn(noise)
    filtered = torch.fft.ifftn(noise_fft * filter_kernel).real
    filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-12)
    threshold_val = torch.quantile(filtered.flatten(), 1.0 - base_porosity)
    solid = filtered > threshold_val
    micro_noise = torch.randn((nz, ny, nx), dtype=torch.float32, device=device, generator=rng)
    micro_noise = (micro_noise - micro_noise.mean()) / (micro_noise.std() + 1e-12)
    flip_mask = micro_noise.abs() < heterogeneity * 2.0
    solid = torch.where(flip_mask, ~solid, solid)
    solid[0, :, :] = False
    solid[-1, :, :] = False
    return solid


# ===========================================================================
# Analytical solutions
# ===========================================================================


def analytical_parallel_plate_permeability(gap: float) -> float:
    """k = h² / 12"""
    return gap ** 2 / 12.0


def analytical_tube_permeability(radius: float) -> float:
    """k = R² / 8"""
    return radius ** 2 / 8.0


def kozeny_carman_permeability(
    porosity: float, specific_surface: float, kozeny_constant: float = 5.0,
) -> float:
    """k = φ³ / (K · S² · (1-φ)²)"""
    if porosity <= 0 or porosity >= 1:
        return 0.0
    return (porosity ** 3) / (kozeny_constant * specific_surface ** 2 * (1 - porosity) ** 2)


# ===========================================================================
# Analysis utilities
# ===========================================================================


def compute_porosity(solid: torch.Tensor) -> float:
    """Porosity (void fraction)."""
    return (~solid).sum().item() / solid.numel()


def compute_specific_surface(solid: torch.Tensor) -> float:
    """Specific surface area via nearest-neighbor counting."""
    if solid.ndim == 2:
        ny, nx = solid.shape
        fluid = ~solid
        fp = torch.zeros((ny + 2, nx + 2), dtype=torch.bool, device=solid.device)
        fp[1:-1, 1:-1] = fluid
        count = 0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            count += (solid & fp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]).sum().item()
        return count / solid.numel()
    else:
        nz, ny, nx = solid.shape
        fluid = ~solid
        fp = torch.zeros((nz + 2, ny + 2, nx + 2), dtype=torch.bool, device=solid.device)
        fp[1:-1, 1:-1, 1:-1] = fluid
        count = 0
        for dz, dy, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            count += (solid & fp[1+dz:1+dz+nz, 1+dy:1+dy+ny, 1+dx:1+dx+nx]).sum().item()
        return count / solid.numel()


def compute_permeability(U_avg: float, mu: float, F: float) -> float:
    """k = μ·U / F  (from Darcy's law with body force driving)."""
    if abs(F) < 1e-15:
        return 0.0
    return mu * U_avg / F


def compute_pore_reynolds(U: float, d_p: float, nu: float) -> float:
    """Re = U·d_p / ν"""
    if nu < 1e-15:
        return float("inf")
    return U * d_p / nu


# ===========================================================================
# Guo forcing scheme (2D)
# ===========================================================================


def _guo_source_2d(ux: torch.Tensor, uy: torch.Tensor, accel: float, tau: float,
                   device: torch.device) -> torch.Tensor:
    """Guo (2002) discrete force source for x-only body force, D2Q9."""
    w = W.to(device=device, dtype=ux.dtype)
    c = C2D.to(device=device)
    rho = _RHO_REF
    factor = (1.0 - 1.0 / (2.0 * tau))
    S = torch.zeros((9,) + ux.shape, dtype=ux.dtype, device=device)
    for q in range(9):
        cix = c[q, 0].item()
        ci_dot_u = cix * ux  # only x-force
        S[q] = factor * w[q] * (3.0 * cix + 9.0 * ci_dot_u * cix) * accel * rho
    return S


# ===========================================================================
# Guo forcing scheme (3D)
# ===========================================================================


def _guo_source_3d(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
                   accel: float, tau: float, device: torch.device) -> torch.Tensor:
    """Guo (2002) discrete force source for z-only body force, D3Q19."""
    w = torch.tensor(
        (1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
         1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
         1/36, 1/36, 1/36, 1/36),
        dtype=ux.dtype, device=device,
    )
    c = C3D.to(device=device)
    rho = _RHO_REF
    factor = (1.0 - 1.0 / (2.0 * tau))
    S = torch.zeros((19,) + ux.shape, dtype=ux.dtype, device=device)
    for q in range(19):
        ciz = c[q, 2].item()  # z-direction force
        ci_dot_u = ciz * uz
        S[q] = factor * w[q] * (3.0 * ciz + 9.0 * ci_dot_u * ciz) * accel * rho
    return S


# ===========================================================================
# Pore-scale flow solver (2D)
# ===========================================================================


@dataclass(frozen=True)
class PoreFlowConfig2D:
    """2D pore-scale flow configuration (body-force driven, periodic in x)."""
    ny: int = 32
    nx: int = 64
    geometry: str = "parallel_plates"
    gap: int = 10
    n_cyl_y: int = 3
    n_cyl_x: int = 8
    cyl_radius: float = 3.0
    tau: float = 1.0
    accel: float = 1e-5  # body force in x-direction
    n_steps: int = 5000
    warmup_steps: int = 1000
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5")


def run_pore_flow_2d(config: PoreFlowConfig2D) -> dict:
    """Run 2D pore-scale flow with D2Q9 LBM + Guo forcing + periodic BC."""
    device = resolve_device(config.device)
    ny, nx = config.ny, config.nx

    if config.geometry == "parallel_plates":
        solid = make_parallel_plates_2d(ny, nx, config.gap, device=device)
    elif config.geometry == "cylinder_array":
        solid = make_periodic_cylinder_array_2d(
            ny, nx, config.n_cyl_y, config.n_cyl_x, config.cyl_radius, device=device,
        )
    else:
        raise ValueError(f"Unknown geometry: {config.geometry}")

    # Initialize
    rho0 = torch.ones((ny, nx), dtype=torch.float64, device=device)
    ux0 = torch.zeros((ny, nx), dtype=torch.float64, device=device)
    uy0 = torch.zeros((ny, nx), dtype=torch.float64, device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)

    tau = config.tau
    omega = 1.0 / tau
    cs2 = _CS2
    nu = cs2 * (tau - 0.5)
    mu = _RHO_REF * nu
    accel = config.accel

    mass_initial = f.sum().item()

    for step in range(config.n_steps):
        # Macroscopic variables
        rho, ux, uy = macroscopic(f, device=device)

        # Half-force velocity for equilibrium
        ux_half = ux + 0.5 * accel / rho.clamp(min=1e-12)
        feq = equilibrium(rho, ux_half, uy, device=device)

        # Collision
        f = f - omega * (f - feq)

        # Guo force source
        S = _guo_source_2d(ux, uy, accel, tau, device)
        f = f + S

        # Bounce-back
        f = bounce_back_cells(f, solid)

        # Stream (periodic in x by default in solver.stream)
        f = stream(f)

    # Final measurement
    rho, ux, uy = macroscopic(f, device=device)
    fluid = ~solid

    # Average velocity in fluid region
    U_avg = (ux * fluid.float()).sum().item() / fluid.float().sum().item()

    # Mass conservation
    mass_final = f.sum().item()
    mass_drift = abs(mass_final - mass_initial) / max(mass_initial, 1e-15)

    # Permeability: k = μ·U / F
    k = compute_permeability(U_avg, mu, accel)

    # Analytical
    if config.geometry == "parallel_plates":
        k_analytical = analytical_parallel_plate_permeability(float(config.gap))
    else:
        k_analytical = None

    porosity = compute_porosity(solid)

    return {
        "permeability": k,
        "analytical_permeability": k_analytical,
        "relative_error": abs(k - k_analytical) / k_analytical if k_analytical else None,
        "mass_drift": mass_drift,
        "porosity": porosity,
        "avg_velocity": U_avg,
        "viscosity": mu,
        "config": config,
        "solid": solid,
        "rho": rho,
        "ux": ux,
        "uy": uy,
    }


# ===========================================================================
# Pore-scale flow solver (3D)
# ===========================================================================


@dataclass(frozen=True)
class PoreFlowConfig3D:
    """3D pore-scale flow configuration (body-force driven, periodic in z)."""
    nz: int = 32
    ny: int = 32
    nx: int = 32
    geometry: str = "tube"
    radius: float = 8.0
    n_spheres: int = 15
    r_min: float = 2.0
    r_max: float = 4.0
    correlation_length: float = 3.0
    threshold: float = 0.0
    base_porosity: float = 0.25
    heterogeneity: float = 0.05
    tau: float = 1.0
    accel: float = 1e-5  # body force in z-direction
    n_steps: int = 5000
    warmup_steps: int = 1000
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5")


def run_pore_flow_3d(config: PoreFlowConfig3D) -> dict:
    """Run 3D pore-scale flow with D3Q19 LBM + Guo forcing + periodic BC."""
    device = resolve_device(config.device)
    nz, ny, nx = config.nz, config.ny, config.nx

    if config.geometry == "tube":
        solid = make_cylindrical_tube_3d(nz, ny, nx, config.radius, device=device)
    elif config.geometry == "sphere_pack":
        solid = make_sphere_pack_3d(
            nz, ny, nx, config.n_spheres, config.r_min, config.r_max,
            seed=config.seed, device=device,
        )
    elif config.geometry == "grf":
        solid = make_gaussian_random_field_3d(
            nz, ny, nx, config.correlation_length, config.threshold,
            seed=config.seed, device=device,
        )
    elif config.geometry == "ct_like":
        solid = make_ct_like_rock_3d(
            nz, ny, nx, config.base_porosity, config.heterogeneity,
            seed=config.seed, device=device,
        )
    else:
        raise ValueError(f"Unknown geometry: {config.geometry}")

    # Initialize
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float64, device=device)
    ux0 = torch.zeros((nz, ny, nx), dtype=torch.float64, device=device)
    uy0 = torch.zeros((nz, ny, nx), dtype=torch.float64, device=device)
    uz0 = torch.zeros((nz, ny, nx), dtype=torch.float64, device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    tau = config.tau
    omega = 1.0 / tau
    cs2 = _CS2
    nu = cs2 * (tau - 0.5)
    mu = _RHO_REF * nu
    accel = config.accel

    mass_initial = f.sum().item()

    for step in range(config.n_steps):
        rho, ux, uy, uz = macroscopic3d(f, device=device)

        # Half-force velocity for equilibrium
        uz_half = uz + 0.5 * accel / rho.clamp(min=1e-12)
        feq = equilibrium3d(rho, ux, uy, uz_half, device=device)

        # Collision
        f = f - omega * (f - feq)

        # Guo force source
        S = _guo_source_3d(ux, uy, uz, accel, tau, device)
        f = f + S

        # Bounce-back
        f = bounce_back_cells_3d(f, solid)

        # Stream (periodic in z by default)
        f = stream3d(f)

    # Final measurement
    rho, ux, uy, uz = macroscopic3d(f, device=device)
    fluid = ~solid

    U_avg = (uz * fluid.float()).sum().item() / fluid.float().sum().item()

    mass_final = f.sum().item()
    mass_drift = abs(mass_final - mass_initial) / max(mass_initial, 1e-15)

    k = compute_permeability(U_avg, mu, accel)

    if config.geometry == "tube":
        k_analytical = analytical_tube_permeability(config.radius)
    else:
        porosity = compute_porosity(solid)
        S = compute_specific_surface(solid)
        k_analytical = kozeny_carman_permeability(porosity, S)

    porosity = compute_porosity(solid)
    d_p = 2.0 * math.sqrt(porosity / (math.pi * (1 - porosity) + 1e-15)) if porosity < 1 else 1.0
    Re_pore = compute_pore_reynolds(U_avg, d_p, nu)

    return {
        "permeability": k,
        "analytical_permeability": k_analytical,
        "relative_error": abs(k - k_analytical) / k_analytical if k_analytical > 0 else None,
        "mass_drift": mass_drift,
        "porosity": porosity,
        "specific_surface": compute_specific_surface(solid),
        "avg_velocity": U_avg,
        "viscosity": mu,
        "pore_reynolds": Re_pore,
        "config": config,
        "solid": solid,
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "uz": uz,
    }
