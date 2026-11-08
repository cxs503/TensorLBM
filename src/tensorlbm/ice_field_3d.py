"""3D Ice Field Generation and DEM-based碎冰场 (broken ice field) simulation.

Generates a realistic broken ice field upstream of a ship hull for
ice-resistance prediction. The ice field is characterized by:

- Ice concentration C_i (area/volume fraction of ice in the domain)
- Floe size distribution (D_floe, typically log-normal or uniform)
- Ice thickness h_ice
- Ice velocity V_ice (drift speed)

The DEM solver handles ice-ice contact and ice-water drag. The resulting
ice field is mapped to a porous medium representation for the LBM/RANS
flow solver.

References
----------
- Riska, K. (1989). "A Model for Icebreaking Resistance."
  Proceedings of the 10th International Conference on Port and Ocean
  Engineering under Arctic Conditions (POAC), Luleå, Sweden.
- Lindqvist, G. (1989). "A Simplified Method for Calculation of Ship
  Resistance in Ice." POAC 89, Vol. 3, pp. 845-859.
- Cundall & Strack (1979). DEM for granular assemblies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from .dem import DEMSolver, DEMConfig, IceParticle, ParticleSystem
from .dem.contact import detect_particle_particle_contact


@dataclass
class IceFieldConfig:
    """Configuration for 3D ice field generation.

    Parameters
    ----------
    domain_size : tuple of float
        Domain size (Lx, Ly, Lz) in meters.
    ice_concentration : float
        Ice area/volume concentration (0-1). Typical: 0.3-0.9.
    ice_thickness : float
        Ice thickness in meters. Typical: 0.1-1.0 m.
    floe_size_mean : float
        Mean floe diameter in meters. Typical: 0.5-5.0 m.
    floe_size_std : float
        Standard deviation of floe size (for log-normal distribution).
    floe_size_min : float
        Minimum floe diameter in meters.
    floe_size_max : float
        Maximum floe diameter in meters.
    ice_density : float
        Ice density in kg/m³. Default: 917 kg/m³.
    ice_velocity : tuple of float
        Ice drift velocity (Vx, Vy, Vz) in m/s.
    n_particles : int, optional
        Number of ice particles. If None, computed from concentration.
    seed : int
        Random seed for reproducibility.
    """
    domain_size: Tuple[float, float, float] = (100.0, 30.0, 5.0)
    ice_concentration: float = 0.6
    ice_thickness: float = 0.15
    floe_size_mean: float = 1.0
    floe_size_std: float = 0.3
    floe_size_min: float = 0.2
    floe_size_max: float = 3.0
    ice_density: float = 917.0
    ice_velocity: Tuple[float, float, float] = (0.5, 0.0, 0.0)
    n_particles: Optional[int] = None
    seed: int = 42

    def __post_init__(self):
        if not (0.0 < self.ice_concentration <= 1.0):
            raise ValueError(f"ice_concentration must be in (0, 1], got {self.ice_concentration}")
        if self.ice_thickness <= 0:
            raise ValueError(f"ice_thickness must be positive, got {self.ice_thickness}")
        if self.floe_size_mean <= 0:
            raise ValueError(f"floe_size_mean must be positive, got {self.floe_size_mean}")
        if self.floe_size_min <= 0:
            raise ValueError(f"floe_size_min must be positive, got {self.floe_size_min}")
        if self.floe_size_max <= self.floe_size_min:
            raise ValueError("floe_size_max must be > floe_size_min")


@dataclass
class IceFloe:
    """Represents a single ice floe (approximated as a disk/cylinder).

    Parameters
    ----------
    position : torch.Tensor
        Center position (x, y, z) in meters.
    velocity : torch.Tensor
        Velocity (vx, vy, vz) in m/s.
    diameter : float
        Floe diameter in meters.
    thickness : float
        Floe thickness in meters.
    mass : float
        Floe mass in kg.
    """
    position: torch.Tensor
    velocity: torch.Tensor
    diameter: float
    thickness: float
    mass: float

    @property
    def radius(self) -> float:
        """Equivalent spherical radius for DEM."""
        return self.diameter / 2.0

    @property
    def volume(self) -> float:
        """Volume as a cylinder."""
        return math.pi * self.radius**2 * self.thickness

    @property
    def cross_section_area(self) -> float:
        """Cross-sectional area normal to flow."""
        return self.diameter * self.thickness


def generate_floe_sizes(
    n: int,
    mean: float,
    std: float,
    min_size: float,
    max_size: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate floe sizes from a truncated log-normal distribution.

    Parameters
    ----------
    n : int
        Number of floes.
    mean : float
        Mean floe diameter.
    std : float
        Standard deviation.
    min_size : float
        Minimum diameter.
    max_size : float
        Maximum diameter.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Array of floe diameters.
    """
    rng = np.random.RandomState(seed)
    # Log-normal distribution
    mu = math.log(mean**2 / math.sqrt(std**2 + mean**2))
    sigma = math.sqrt(math.log(1 + std**2 / mean**2))

    sizes = rng.lognormal(mu, sigma, size=n * 2)  # Generate extra to account for truncation
    sizes = sizes[(sizes >= min_size) & (sizes <= max_size)]

    if len(sizes) < n:
        # Fill remaining with uniform distribution
        remaining = n - len(sizes)
        extra = rng.uniform(min_size, max_size, size=remaining * 2)
        extra = extra[(extra >= min_size) & (extra <= max_size)]
        sizes = np.concatenate([sizes, extra])

    return sizes[:n]


def estimate_n_particles(config: IceFieldConfig) -> int:
    """Estimate number of ice particles from concentration.

    Uses the relation: C_i = N * A_floe / A_domain
    where A_floe is the average floe cross-sectional area.

    Parameters
    ----------
    config : IceFieldConfig
        Ice field configuration.

    Returns
    -------
    int
        Estimated number of particles.
    """
    Lx, Ly, Lz = config.domain_size
    # Surface area of the water plane
    A_domain = Lx * Ly

    # Average floe area (circular approximation)
    A_floe = math.pi * (config.floe_size_mean / 2)**2

    # Number of floes
    n = int(config.ice_concentration * A_domain / A_floe)

    return max(1, n)


def generate_ice_field(config: IceFieldConfig) -> List[IceParticle]:
    """Generate a 3D ice field as DEM particles.

    Creates ice floes distributed in the domain with the specified
    concentration and size distribution. Floes are placed randomly
    without overlap (using a simple rejection sampling).

    Parameters
    ----------
    config : IceFieldConfig
        Ice field configuration.

    Returns
    -------
    list of IceParticle
        List of ice particles representing the ice field.
    """
    rng = np.random.RandomState(config.seed)
    torch.manual_seed(config.seed)

    Lx, Ly, Lz = config.domain_size
    h_ice = config.ice_thickness

    # Estimate number of particles
    n_particles = config.n_particles or estimate_n_particles(config)

    # Generate floe sizes
    diameters = generate_floe_sizes(
        n_particles,
        config.floe_size_mean,
        config.floe_size_std,
        config.floe_size_min,
        config.floe_size_max,
        config.seed,
    )

    # Place particles without overlap (rejection sampling)
    particles = []
    max_attempts = 1000
    placed_positions = []

    for i, d in enumerate(diameters):
        r = d / 2.0
        placed = False

        for _ in range(max_attempts):
            # Random position (with margin for radius)
            x = rng.uniform(r + 0.1, Lx - r - 0.1)
            y = rng.uniform(r + 0.1, Ly - r - 0.1)
            z = h_ice / 2.0  # Centered at half ice thickness

            pos = torch.tensor([x, y, z], dtype=torch.float64)

            # Check for overlap with existing particles
            overlap = False
            for existing_pos, existing_r in placed_positions:
                dist = torch.norm(pos - existing_pos).item()
                if dist < (r + existing_r) * 0.95:  # 5% tolerance
                    overlap = True
                    break

            if not overlap:
                vel = torch.tensor(config.ice_velocity, dtype=torch.float64)
                omega = torch.zeros(3, dtype=torch.float64)

                particle = IceParticle(
                    position=pos,
                    velocity=vel,
                    angular_velocity=omega,
                    radius=r,
                    density=config.ice_density,
                )
                particles.append(particle)
                placed_positions.append((pos.clone(), r))
                placed = True
                break

        if not placed:
            # Skip this particle if we can't place it
            continue

    return particles


def compute_ice_concentration(
    particles: List[IceParticle],
    domain_size: Tuple[float, float, float],
) -> dict:
    """Compute ice concentration statistics from particle list.

    Parameters
    ----------
    particles : list of IceParticle
        Ice particles.
    domain_size : tuple of float
        Domain size (Lx, Ly, Lz).

    Returns
    -------
    dict
        Concentration statistics including:
        - C_i: area concentration
        - C_v: volume concentration
        - D50: median floe diameter
        - D_mean: mean floe diameter
        - n_floes: number of floes
        - total_ice_area: total ice area
        - total_ice_volume: total ice volume
    """
    Lx, Ly, Lz = domain_size
    A_domain = Lx * Ly
    V_domain = Lx * Ly * Lz

    diameters = [2.0 * p.radius for p in particles]
    areas = [math.pi * p.radius**2 for p in particles]
    volumes = [p.volume for p in particles]

    total_area = sum(areas)
    total_volume = sum(volumes)

    C_i = total_area / A_domain if A_domain > 0 else 0.0
    C_v = total_volume / V_domain if V_domain > 0 else 0.0

    # D50 (median diameter)
    sorted_d = sorted(diameters)
    n = len(sorted_d)
    if n > 0:
        D50 = sorted_d[n // 2]
        D_mean = sum(diameters) / n
    else:
        D50 = 0.0
        D_mean = 0.0

    return {
        "C_i": C_i,
        "C_v": C_v,
        "D50": D50,
        "D_mean": D_mean,
        "D_min": min(diameters) if diameters else 0.0,
        "D_max": max(diameters) if diameters else 0.0,
        "n_floes": len(particles),
        "total_ice_area": total_area,
        "total_ice_volume": total_volume,
        "total_ice_mass": sum(p.mass for p in particles),
    }


def create_ice_field_dem_solver(
    config: IceFieldConfig,
    particles: Optional[List[IceParticle]] = None,
) -> DEMSolver:
    """Create a DEM solver for the ice field.

    Parameters
    ----------
    config : IceFieldConfig
        Ice field configuration.
    particles : list of IceParticle, optional
        Pre-generated particles. If None, generates from config.

    Returns
    -------
    DEMSolver
        Configured DEM solver for the ice field.
    """
    if particles is None:
        particles = generate_ice_field(config)

    Lx, Ly, Lz = config.domain_size
    min_corner = torch.zeros(3, dtype=torch.float64)
    max_corner = torch.tensor([Lx, Ly, Lz], dtype=torch.float64)

    dem_config = DEMConfig(
        time_step=1e-3,
        gravity=torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
        domain_bounds=(min_corner, max_corner),
        enable_wall_contacts=True,
        drag_model="schiller_naumann",
        fluid_density=1025.0,  # Seawater
        fluid_viscosity=1.8e-3,  # Seawater dynamic viscosity at 0°C
    )

    solver = DEMSolver(dem_config)
    solver.initialize(particles)

    return solver


def ice_field_to_porous_medium(
    particles: List[IceParticle],
    grid_resolution: Tuple[int, int, int],
    domain_size: Tuple[float, float, float],
) -> torch.Tensor:
    """Map ice field to porous medium volume fraction field.

    Creates a 3D grid where each cell contains the ice volume fraction
    (porosity = 1 - alpha_i).

    Parameters
    ----------
    particles : list of IceParticle
        Ice particles.
    grid_resolution : tuple of int
        Grid resolution (nx, ny, nz).
    domain_size : tuple of float
        Domain size (Lx, Ly, Lz).

    Returns
    -------
    torch.Tensor
        Ice volume fraction field, shape (nz, ny, nx).
    """
    nx, ny, nz = grid_resolution
    Lx, Ly, Lz = domain_size

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    alpha_i = torch.zeros((nz, ny, nx), dtype=torch.float64)

    for particle in particles:
        # Determine grid cells covered by this particle
        x_min = max(0, int((particle.position[0].item() - particle.radius) / dx))
        x_max = min(nx - 1, int((particle.position[0].item() + particle.radius) / dx))
        y_min = max(0, int((particle.position[1].item() - particle.radius) / dy))
        y_max = min(ny - 1, int((particle.position[1].item() + particle.radius) / dy))
        z_min = max(0, int((particle.position[2].item() - particle.radius) / dz))
        z_max = min(nz - 1, int((particle.position[2].item() + particle.radius) / dz))

        # Add volume fraction to covered cells
        cell_volume = dx * dy * dz
        particle_vol = particle.volume

        for iz in range(z_min, z_max + 1):
            for iy in range(y_min, y_max + 1):
                for ix in range(x_min, x_max + 1):
                    # Check if cell center is within particle
                    cx = (ix + 0.5) * dx
                    cy = (iy + 0.5) * dy
                    cz = (iz + 0.5) * dz

                    dist_sq = (
                        (cx - particle.position[0].item())**2
                        + (cy - particle.position[1].item())**2
                        + (cz - particle.position[2].item())**2
                    )

                    if dist_sq < particle.radius**2:
                        alpha_i[iz, iy, ix] += particle_vol / cell_volume

    # Clamp to [0, 1]
    alpha_i = torch.clamp(alpha_i, 0.0, 1.0)

    return alpha_i


def validate_ice_field(
    particles: List[IceParticle],
    config: IceFieldConfig,
    tolerance_concentration: float = 0.05,
    tolerance_d50: float = 0.10,
) -> dict:
    """Validate ice field against target specifications.

    Parameters
    ----------
    particles : list of IceParticle
        Generated ice particles.
    config : IceFieldConfig
        Target configuration.
    tolerance_concentration : float
        Acceptable relative error in concentration (default 5%).
    tolerance_d50 : float
        Acceptable relative error in D50 (default 10%).

    Returns
    -------
    dict
        Validation results with pass/fail for each criterion.
    """
    stats = compute_ice_concentration(particles, config.domain_size)

    # Concentration error
    C_i_target = config.ice_concentration
    C_i_error = abs(stats["C_i"] - C_i_target) / C_i_target if C_i_target > 0 else float("inf")

    # D50 error (target is mean floe size as approximation)
    D50_target = config.floe_size_mean
    D50_error = abs(stats["D50"] - D50_target) / D50_target if D50_target > 0 else float("inf")

    results = {
        "C_i_target": C_i_target,
        "C_i_actual": stats["C_i"],
        "C_i_error": C_i_error,
        "C_i_pass": C_i_error < tolerance_concentration,
        "D50_target": D50_target,
        "D50_actual": stats["D50"],
        "D50_error": D50_error,
        "D50_pass": D50_error < tolerance_d50,
        "n_floes": stats["n_floes"],
        "total_ice_mass": stats["total_ice_mass"],
    }

    return results
