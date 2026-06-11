"""Parametric propeller CAD module for TensorLBM.

Generates voxelised propeller geometry suitable for LBM simulations.
Supports hub + N blades with configurable chord, pitch, skew, rake,
and thickness distributions.

The geometry is built by voxelising the propeller directly on the
target grid using analytical distance-to-surface computations,
avoiding expensive surface point generation.

Reference data
--------------
Fujisawa, J. et al. (2000), "Measurements of the Local Flow Field around
a Ship Propeller", J. Soc. Naval Architects of Japan, Vol. 188.
SIMMAN 2008/2014 Workshop — KCS hull + KP505 propeller open-water data.

Public API
----------
- :class:`PropellerGeometryConfig` – parametric geometry configuration.
- :func:`build_propeller_mask` – 3-D boolean mask (hub + blades).
- :func:`propeller_statistics` – geometry statistics dictionary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


__all__ = [
    "PropellerGeometryConfig",
    "build_propeller_mask",
    "propeller_statistics",
    "KP505_PRESET",
    "GENERIC_PRESET",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PropellerGeometryConfig:
    """Parametric propeller geometry configuration.

    All length parameters are in lattice units unless stated otherwise.
    The propeller rotates about the x-axis (flow direction).

    Parameters
    ----------
    n_blades : int
        Number of blades (3–7).
    diameter : float
        Propeller diameter [lu].
    hub_diameter_ratio : float
        Hub diameter / propeller diameter. Typically 0.16–0.20.
    hub_length_ratio : float
        Hub axial length / diameter. Typically 0.3–0.5.
    pitch_ratio_07 : float
        Face pitch / diameter at r/R = 0.7. Typically 0.7–1.4.
    blade_area_ratio : float
        Expanded area ratio A_E/A_0. Typically 0.3–0.8.
    skew_deg : float
        Maximum blade skew angle [degrees]. Typically 0–30.
    rake_ratio : float
        Rake / diameter. Typically 0.0–0.05.
    max_thickness_ratio : float
        Maximum blade thickness / chord at r/R = 0.7.
    """

    n_blades: int = 5
    diameter: float = 48.0
    hub_diameter_ratio: float = 0.18
    hub_length_ratio: float = 0.5
    pitch_ratio_07: float = 0.95
    blade_area_ratio: float = 0.65
    skew_deg: float = 0.0
    rake_ratio: float = 0.0
    max_thickness_ratio: float = 0.06

    def __post_init__(self) -> None:
        if self.n_blades < 2:
            raise ValueError("n_blades must be >= 2")
        if self.diameter <= 0:
            raise ValueError("diameter must be > 0")
        if not (0.05 <= self.hub_diameter_ratio <= 0.40):
            raise ValueError("hub_diameter_ratio must be in [0.05, 0.40]")
        if self.hub_length_ratio <= 0:
            raise ValueError("hub_length_ratio must be > 0")

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def hub_radius(self) -> float:
        return self.radius * self.hub_diameter_ratio

    @property
    def hub_length(self) -> float:
        return self.diameter * self.hub_length_ratio

    @property
    def mean_chord(self) -> float:
        """Mean chord length from expanded area ratio."""
        return (math.pi * self.diameter / 2.0 * self.blade_area_ratio) / self.n_blades


# Preset configurations
KP505_PRESET = PropellerGeometryConfig(
    n_blades=5,
    diameter=48.0,
    hub_diameter_ratio=0.18,
    hub_length_ratio=0.45,
    pitch_ratio_07=0.95,
    blade_area_ratio=0.65,
    skew_deg=0.0,
    rake_ratio=0.0,
    max_thickness_ratio=0.06,
)

GENERIC_PRESET = PropellerGeometryConfig(
    n_blades=4,
    diameter=40.0,
    hub_diameter_ratio=0.18,
    hub_length_ratio=0.4,
    pitch_ratio_07=1.0,
    blade_area_ratio=0.55,
    skew_deg=0.0,
    rake_ratio=0.0,
    max_thickness_ratio=0.05,
)


# ---------------------------------------------------------------------------
# Radial distribution helpers (normalised to [0, 1])
# ---------------------------------------------------------------------------

def _chord_frac(r_frac: torch.Tensor, hub_frac: float) -> torch.Tensor:
    """Elliptic chord distribution c(r)/c_max vs r/R."""
    x_norm = (r_frac - 0.7) / 0.8
    val = 1.0 - x_norm**2
    val = torch.clamp(val, min=0.0)
    return torch.sqrt(val)


def _thickness_frac(r_frac: torch.Tensor) -> torch.Tensor:
    """Relative thickness t(r)/t_max, parabolic distribution."""
    return 4.0 * r_frac * (1.0 - r_frac)


def _skew_angle(r_frac: torch.Tensor, skew_deg: float) -> torch.Tensor:
    """Skew angle in radians as function of radius."""
    return torch.deg2rad(torch.tensor(skew_deg, dtype=r_frac.dtype)) * r_frac


def _naca4_half_thickness(xc_norm: torch.Tensor) -> torch.Tensor:
    r"""NACA 4-digit half-thickness profile y_t/c (normalised).

    .. math::
        y_t/c = (t/c) · 5 · (0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1015x⁴)
    """
    x = torch.clamp(xc_norm, 0.0, 1.0)
    shape = (
        0.2969 * torch.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    return 5.0 * shape


# ---------------------------------------------------------------------------
# Voxelisation
# ---------------------------------------------------------------------------

def build_propeller_mask(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    angle_deg: float = 0.0,
    config: PropellerGeometryConfig | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a 3-D boolean mask for propeller hub and blades.

    Uses analytic distance computations — no polygon loops.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions.  nx is the axial (flow) direction.
    cx, cy, cz : float
        Propeller centre in lattice coordinates.
    angle_deg : float
        Initial rotation angle of first blade [degrees].
    config : PropellerGeometryConfig, optional
        Uses ``KP505_PRESET`` by default.
    device : str or torch.device
        Target device.

    Returns
    -------
    mask : torch.Tensor of bool, shape (nz, ny, nx)
    """
    if config is None:
        config = KP505_PRESET

    R = config.radius
    R_hub = config.hub_radius
    L_hub = config.hub_length

    # Build meshgrid in (nz, ny, nx) order (LBM convention)
    yy, zz, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Centred coordinates
    dx = xx - cx
    dy = yy - cy
    dz = zz - cz

    # Radial and azimuthal coordinates
    r = torch.sqrt(dy**2 + dz**2)  # (ny, nz, nx)
    r_frac = r / R
    theta = torch.atan2(dz, dy)  # azimuth angle, (ny, nz, nx)

    # --- Hub mask ---
    hub_mask = (r <= R_hub) & (torch.abs(dx) <= L_hub / 2.0)
    blade_mask = torch.zeros_like(hub_mask)

    # --- Blade mask ---
    # Only consider cells in blade annulus
    blade_annulus = (r_frac >= config.hub_diameter_ratio * 0.95) & (r_frac <= 0.995)
    if blade_annulus.any():
        r_ann = r[blade_annulus]
        r_frac_ann = r_frac[blade_annulus]
        dx_ann = dx[blade_annulus]
        theta_ann = theta[blade_annulus]

        # Chord and thickness at each radius
        chord_frac_mid = _chord_frac(r_frac_ann, config.hub_diameter_ratio)
        chord = chord_frac_mid * config.mean_chord  # chord length at each radius
        t_frac = _thickness_frac(r_frac_ann)
        t_max_local = config.max_thickness_ratio * chord  # max half-thickness at each radius

        # Pitch: advance per radian
        pitch_per_rad = config.pitch_ratio_07 * config.diameter / (2.0 * math.pi)

        # Skew
        skew = _skew_angle(r_frac_ann, config.skew_deg)

        # Rake
        rake = config.rake_ratio * config.diameter * r_frac_ann

        # Blade centreline azimuth
        first_blade_theta = math.radians(angle_deg)

        # half-chord: widen by 1 lu for voxel visibility
        half_chord = chord / 2.0 + 1.0

        for k in range(config.n_blades):
            blade_theta = first_blade_theta + 2.0 * math.pi * k / config.n_blades

            # Azimuthal offset from blade centreline
            dtheta = theta_ann - (blade_theta + skew)
            dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))

            # Arc length at this radius
            arc_dist = r_ann * dtheta.abs()

            # Axial position of blade pitch surface: x = rake + θ * P/(2π)
            x0_blade = rake + dtheta * pitch_per_rad

            # Chordwise coordinate (0 = centre, 1 = LE/TE)
            chordwise_pos = arc_dist / half_chord.clamp(min=1e-6)

            # NACA thickness shape, clamped to [0,1]
            thickness_frac_local = _naca4_half_thickness(
                chordwise_pos.clamp(0.0, 1.0)
            )
            # Scale by max thickness AND ensure minimum 2 lu for voxel capture
            local_half_thickness = torch.maximum(
                t_max_local * thickness_frac_local,
                torch.tensor(2.0, device=r_ann.device, dtype=r_ann.dtype),
            )

            cell_inside = (
                (arc_dist < half_chord)
                & (torch.abs(dx_ann - x0_blade) < local_half_thickness)
            )

            # Also include cells within 1 lu of the blade (for thin sections)
            # Use a simpler check: combined distance
            cell_mask_i = torch.zeros_like(blade_mask, dtype=torch.bool)
            cell_mask_i[blade_annulus] = cell_inside
            blade_mask = blade_mask | cell_mask_i

    # --- Combined mask ---
    mask = hub_mask | blade_mask

    # Permute from (ny, nz, nx) to (nz, ny, nx) for LBM convention
    mask = mask.permute(1, 0, 2).contiguous()

    return mask


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def propeller_statistics(
    config: PropellerGeometryConfig,
    mask: torch.Tensor,
) -> dict[str, object]:
    """Compute geometric statistics for a propeller mask."""
    nz, ny, nx = mask.shape
    total_cells = nx * ny * nz
    solid_cells = int(mask.sum().item())
    solid_fraction = solid_cells / total_cells

    disk_area = math.pi * (config.radius**2)
    blade_planform = config.blade_area_ratio * disk_area
    hub_area = 2.0 * math.pi * config.hub_radius * config.hub_length
    blade_wetted = 2.0 * blade_planform * config.n_blades
    estimated_wetted = hub_area + blade_wetted

    return {
        "n_cells": total_cells,
        "solid_cells": solid_cells,
        "solid_fraction": solid_fraction,
        "disk_area_cells": disk_area,
        "projected_area_cells": blade_planform,
        "estimated_wetted_area": estimated_wetted,
    }
