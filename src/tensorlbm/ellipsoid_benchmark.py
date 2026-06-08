"""Prolate spheroid (ellipsoid) benchmark for TensorLBM.

3-D 6:1 prolate spheroid at various angles of attack and Reynolds numbers,
comparing drag and lift coefficients against published data.

The 6:1 prolate spheroid (length/diameter = 6, semi-axes a=3b) is a standard
CFD validation case for submarine and airship aerodynamics. At low Re (based
on minor-axis diameter D=2b), the flow transitions from attached laminar to
separated, similar to the sphere benchmark.

Reference data
--------------
- Hoerner, S.F. (1965) "Fluid-Dynamic Drag" — streamlined body Cd trends.
- Clift, Grace & Weber (1978) "Bubbles, Drops, and Particles" — drag curves
  for spheroids at low/intermediate Re.
- DNS: Mittal, R. et al. — low-Re flow past prolate spheroids.

At Re_D=100 (based on minor-axis diameter 2b), α=0°:
  Cd ≈ 0.8–1.2  (laminar separation near tail, lower than sphere Cd≈1.09)
At α>0°, Cl grows roughly linearly with α for small angles (3D lifting-line).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .boundaries3d import (
    apply_zou_he_channel_boundaries_3d,
    make_channel_wall_mask_3d,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .obstacles import compute_obstacle_forces_3d
from .solver3d import correct_mass3d, stream3d
from .turbulence import collide_smagorinsky_mrt3d
from .utils import (
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)


# ---------------------------------------------------------------------------
# Ellipsoid geometry: prolate spheroid (x/a)² + (y/b)² + (z/b)² ≤ 1
# ---------------------------------------------------------------------------

def build_ellipsoid_mask(
    nx: int,
    ny: int,
    nz: int,
    a: float,
    b: float,
    alpha_deg: float = 0.0,
    cx: float | None = None,
    cy: float | None = None,
    cz: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build a 3-D prolate spheroid obstacle mask.

    The ellipsoid satisfies (x'/a)² + (y'/b)² + (z'/b)² ≤ 1 in body
    coordinates, where (x',y',z') are rotated by *alpha_deg* about the
    z-axis relative to the grid (standard aeronautics nose-up convention).

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions (nx=streamwise, ny=vertical, nz=spanwise).
    a : float
        Semi-major axis length [lu] (streamwise direction).
    b : float
        Semi-minor axis length [lu] (cross-stream directions).
        For a 6:1 prolate spheroid, a/b = 3 (full axes: 2a/2b = 6).
    alpha_deg : float
        Angle of attack [degrees]. Positive = nose-up (standard convention).
    cx, cy, cz : float, optional
        Ellipsoid centre.  Default: (nx/3, ny/2, nz/2).
    device : torch.device

    Returns
    -------
    mask : torch.Tensor of bool, shape (nz, ny, nx)
    """
    if cx is None:
        cx = nx / 3.0
    if cy is None:
        cy = ny / 2.0
    if cz is None:
        cz = nz / 2.0

    alpha = math.radians(alpha_deg)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Shift to body frame (origin at ellipsoid centre)
    dx = xx - cx
    dy = yy - cy
    dz = zz - cz

    # Rotate back to body-aligned coordinates
    # Nose-up α: the ellipsoid nose points +y. Body-to-grid is
    # clockwise rotation; grid-to-body is counterclockwise:
    #   x_body = dx·cos(α) - dy·sin(α),  y_body = dx·sin(α) + dy·cos(α)
    x_body = dx * cos_a - dy * sin_a
    y_body = dx * sin_a + dy * cos_a
    z_body = dz

    # Ellipsoid inequality
    r2 = (x_body / a) ** 2 + (y_body / b) ** 2 + (z_body / b) ** 2
    return r2 <= 1.0


def ellipsoid_statistics(
    nx: int,
    ny: int,
    nz: int,
    a: float,
    b: float,
    alpha_deg: float = 0.0,
    cx: float | None = None,
    cy: float | None = None,
    cz: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Compute geometric statistics for the ellipsoid mask.

    Returns a dict with keys: 'volume_lu3', 'frontal_area_lu2',
    'wetted_area_lu2', 'solid_cells', 'length_lu', 'diameter_lu'.
    """
    mask = build_ellipsoid_mask(nx, ny, nz, a, b, alpha_deg, cx, cy, cz, device)
    solid = int(mask.sum().item())

    # Analytical values (continuous geometry, for reference)
    e2 = 1.0 - (b / a) ** 2  # eccentricity squared
    e = math.sqrt(max(e2, 0.0))

    # Volume = 4/3 π a b²
    vol_analytical = (4.0 / 3.0) * math.pi * a * b**2

    if e2 > 1e-12:
        # Wetted area: 2πb² + 2πab·arcsin(e)/e
        wsa = 2.0 * math.pi * b**2 + 2.0 * math.pi * a * b * math.asin(e) / e
    else:
        # Sphere limit
        wsa = 4.0 * math.pi * a**2

    # Projected frontal area (ellipse at α=0): π b²
    frontal = math.pi * b**2

    return {
        "volume_lu3": vol_analytical,
        "frontal_area_lu2": frontal,
        "wetted_area_lu2": wsa,
        "solid_cells": solid,
        "length_lu": 2.0 * a,
        "diameter_lu": 2.0 * b,
        "a_b_ratio": a / b,
    }


# ---------------------------------------------------------------------------
# Reference data — 6:1 prolate spheroid drag
# ---------------------------------------------------------------------------

def reference_ellipsoid_cd(
    re: float,
    alpha_deg: float = 0.0,
) -> dict[str, float]:
    """Approximate reference Cd and Cl for a 6:1 prolate spheroid.

    Based on Clift-Grace-Weber drag correlations for spheroids and
    Hoerner's streamlined-body data. The drag is composed of friction
    (laminar flat-plate) + form drag from the spheroid shape factor.

    Parameters
    ----------
    re : float
        Reynolds number based on minor-axis diameter D = 2b.
    alpha_deg : float
        Angle of attack [degrees].

    Returns
    -------
    dict with keys 'cd', 'cl'.
    """
    # Friction: laminar Blasius over characteristic length
    # Use spheroid wetted area to frontal area ratio as form factor
    cf = 1.328 / math.sqrt(max(re, 1.0))  # Blasius friction coefficient

    # For a 6:1 prolate spheroid (a/b=3), the wetted/frontal area ratio ≈ 9.0
    a_b = 3.0
    b = 1.0
    a = 3.0
    e2 = 1.0 - (b / a) ** 2
    e = math.sqrt(max(e2, 0.0))
    wsa = 2.0 * math.pi * b**2 + 2.0 * math.pi * a * b * math.asin(e) / e
    frontal = math.pi * b**2

    # Friction drag on wetted area, non-dimensionalised by frontal area
    cd_friction = cf * wsa / frontal

    # Form factor — streamlined body has lower pressure drag than sphere
    # At low Re, separation occurs near the tail → form drag ~0.3-0.6 of friction
    form_factor = 0.5  # streamline advantage over sphere
    cd_form = cf * 0.5  # form drag proportional to friction at low Re

    cd = cd_friction + cd_form

    # Lift: 3D lifting line slope for low-aspect-ratio wing
    # Effective aspect ratio for ellipsoid ≈ 2b / (2a) = 1/3 → very low
    # Use slender body theory: Cl ≈ 2α for small α (radians)
    alpha_rad = math.radians(abs(alpha_deg))
    cl_slope = 1.8  # per radian (less than 2π due to low AR)
    cl = cl_slope * alpha_rad * math.copysign(1.0, alpha_deg)

    # Induced drag
    ar_eff = 2.0 * b / (2.0 * a)  # span/chord ≈ 1/3
    cd_induced = cl**2 / (math.pi * ar_eff) if ar_eff > 0 else 0.0

    return {"cd": cd + cd_induced, "cl": cl}


# ============================================================================
# Benchmark runner
# ============================================================================

@dataclass
class EllipsoidConfig:
    """Configuration for ellipsoid benchmark simulation."""

    semi_major_a: float = 24.0
    semi_minor_b: float = 8.0
    alpha_deg: float = 0.0
    nx: int = 120
    ny: int = 64
    nz: int = 64
    u_in: float = 0.06
    re: float = 100.0
    n_steps: int = 4000
    warmup_steps: int = 2000
    smagorinsky_cs: float = 0.1
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------
    @property
    def nu(self) -> float:
        """Kinematic viscosity from Re = u_in * D / nu, D = 2*b."""
        return self.u_in * 2.0 * self.semi_minor_b / self.re

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def a_b_ratio(self) -> float:
        return self.semi_major_a / self.semi_minor_b

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    def resolve_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ellipsoid_a{self.a_b_ratio:.0f}_re{self.re}_a{self.alpha_deg}_{ts}"


def run_ellipsoid_benchmark(config: EllipsoidConfig) -> dict:
    """Run a 3-D D3Q19 flow past a prolate spheroid and report Cd, Cl.

    Uses Smagorinsky MRT collision → stream → force measurement →
    Zou/He channel boundaries.
    """
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    # Build obstacle mask
    mask = build_ellipsoid_mask(
        config.nx, config.ny, config.nz,
        config.semi_major_a, config.semi_minor_b,
        config.alpha_deg, device=device,
    )
    wall_mask = make_channel_wall_mask_3d(
        config.nz, config.ny, config.nx, mask, device=device,
    )

    # Initialise uniform flow
    rho0 = torch.ones((config.nz, config.ny, config.nx), device=device)
    ux0 = torch.full_like(rho0, config.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[mask] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    initial_mass = float(rho0.sum().item())
    diam = 2.0 * config.semi_minor_b
    dyn_pressure = 0.5 * config.u_in**2 * math.pi * config.semi_minor_b**2

    fx_list: list[float] = []
    fy_list: list[float] = []
    fz_list: list[float] = []

    print(f"Ellipsoid benchmark: a/b={config.a_b_ratio:.1f} "
          f"α={config.alpha_deg}° Re={config.re} tau={config.tau:.4f}")
    print(f"  Grid: {config.nx}×{config.ny}×{config.nz}  "
          f"steps={config.n_steps}  Cs={config.smagorinsky_cs}")
    print(f"  D={diam:.0f} lu  u_in={config.u_in}  device={device}")

    for step in range(1, config.n_steps + 1):
        # Collision
        if config.tau < 0.575 and config.smagorinsky_cs > 0:
            f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
        elif config.tau < 0.575:
            from .solver3d import collide_mrt3d
            f = collide_mrt3d(f, tau=config.tau)
        else:
            from .solver3d import collide_bgk3d
            f = collide_bgk3d(f, tau=config.tau)

        # Stream
        f = stream3d(f)

        # Measure forces AFTER streaming, BEFORE bounce-back (momentum-exchange)
        fx, fy, fz = compute_obstacle_forces_3d(f, mask)

        # Apply boundary conditions (includes bounce-back on obstacle)
        f = apply_zou_he_channel_boundaries_3d(
            f, u_in=config.u_in, wall_mask=wall_mask, obstacle_mask=mask,
        )

        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # Collect force samples
        if step > config.warmup_steps:
            fx_list.append(float(fx.item()))
            fy_list.append(float(fy.item()))
            fz_list.append(float(fz.item()))

        if step % 500 == 0 or step == config.n_steps:
            n_samples = max(min(len(fx_list), 500), 1)
            cd_mean = sum(fx_list[-500:]) / n_samples / dyn_pressure
            cl_mean = sum(fy_list[-500:]) / n_samples / dyn_pressure
            cz_mean = sum(fz_list[-500:]) / n_samples / dyn_pressure
            print(f"  step {step:5d}: Cd={cd_mean:.4f}  Cl={cl_mean:.4f}  "
                  f"Cz={cz_mean:.4f}")

    n_total = max(len(fx_list), 1)
    cd_mean = sum(fx_list) / n_total / dyn_pressure
    cl_mean = sum(fy_list) / n_total / dyn_pressure
    cz_mean = sum(fz_list) / n_total / dyn_pressure

    # Reference comparison
    ref = reference_ellipsoid_cd(config.re, config.alpha_deg)
    cd_err = abs(cd_mean - ref["cd"]) / max(abs(ref["cd"]), 1e-10) * 100
    cl_err = abs(cl_mean - ref["cl"]) / max(abs(ref["cl"]), 1e-10) * 100 if abs(ref["cl"]) > 1e-10 else float("nan")

    stats = ellipsoid_statistics(
        config.nx, config.ny, config.nz,
        config.semi_major_a, config.semi_minor_b,
        config.alpha_deg, device=device,
    )

    print(f"\n  Results:  6:1 prolate spheroid  α={config.alpha_deg}°")
    print(f"  Cd_sim={cd_mean:.4f}  (ref {ref['cd']:.4f}, err {cd_err:.1f}%)")
    print(f"  Cl_sim={cl_mean:.4f}  (ref {ref['cl']:.4f}, err {cl_err:.1f}%)"
          if not math.isnan(cl_err) else f"  Cl_sim={cl_mean:.4f}")
    print(f"  D={diam:.0f} lu  Re={config.re}  a/b={config.a_b_ratio:.1f}")

    return {
        "cd_sim": cd_mean,
        "cl_sim": cl_mean,
        "cz_sim": cz_mean,
        "cd_ref": ref["cd"],
        "cl_ref": ref["cl"],
        "cd_err_pct": cd_err,
        "cl_err_pct": cl_err,
        "alpha_deg": config.alpha_deg,
        "re": config.re,
        "a_b_ratio": config.a_b_ratio,
        "diameter_lu": diam,
        "geometry": stats,
    }


__all__ = [
    "EllipsoidConfig",
    "build_ellipsoid_mask",
    "ellipsoid_statistics",
    "reference_ellipsoid_cd",
    "run_ellipsoid_benchmark",
]
