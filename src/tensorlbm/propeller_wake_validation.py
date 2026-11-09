"""Propeller wake field validation: body-force vs sliding-mesh vs PIV.

Implements two LBM-based methods for predicting propeller wake fields:

1. **Body-force (actuator disk)**: Thrust and swirl forces distributed over
   the propeller disk region, derived from KT/KQ curves.
2. **Sliding-mesh**: Rotating inner domain with resolved blade geometry,
   coupled to a static outer domain via a sliding interface.

Both methods are validated against synthetic PIV data at downstream
stations x/D = 0.5, 1.0, 2.0.

Acceptance criteria
-------------------
A1. Axial velocity U/U0 at x/D=1.0 vs PIV RMS ≤ 10%
A2. Tangential velocity V/U0 vs PIV RMS ≤ 15%
A3. Body-force vs sliding-mesh far-field (x/D>2) deviation ≤ 5%
A4. Thrust coefficient K_T vs B-series ≤ 8%
A5. No-propeller negative control: velocity uniformity ≤ 1%
A6. pytest ≥ 12 cases all green

References
----------
- ITTC (2014) "Recommended Procedures: Open Water Test", 7.5-02-03-02.1
- Fujisawa et al. (2000), J. Soc. Naval Architects of Japan, Vol. 188.
- Goldstein (1978), "On the Vortex Theory of Screw Propellers".
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .actuator_disk import (
    ActuatorDiskConfig,
    apply_actuator_disk,
    _interp_kt,
    _interp_kq,
)
from .d3q19 import C, W, equilibrium3d, macroscopic3d
from .propeller_cad import KP505_PRESET, PropellerGeometryConfig, build_propeller_mask
from .solver3d import collide_bgk3d, stream3d
from .utils import get_reproducibility_metadata, prepare_run_dir, resolve_device

__all__ = [
    "WakeValidationConfig",
    "WakeStation",
    "WakeProfile",
    "PIVReference",
    "compute_wake_profile",
    "generate_piv_reference",
    "run_body_force_wake",
    "run_sliding_mesh_wake",
    "run_wake_validation",
    "validate_wake_results",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WakeStation:
    """Downstream measurement station for wake profiling."""

    x_over_D: float  # axial distance / propeller diameter
    n_radial: int = 20  # radial sampling points
    n_azimuthal: int = 36  # azimuthal sampling points


@dataclass
class WakeProfile:
    """Velocity profile at a wake station."""

    x_over_D: float
    r_over_R: torch.Tensor  # normalised radius [0, 1]
    theta: torch.Tensor  # azimuth angle [rad]
    u_axial: torch.Tensor  # axial velocity U/U0
    u_tangential: torch.Tensor  # tangential velocity V/U0
    u_radial: torch.Tensor  # radial velocity W/U0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_over_D": self.x_over_D,
            "r_over_R": self.r_over_R.tolist(),
            "theta": self.theta.tolist(),
            "u_axial": self.u_axial.tolist(),
            "u_tangential": self.u_tangential.tolist(),
            "u_radial": self.u_radial.tolist(),
        }


@dataclass(frozen=True)
class WakeValidationConfig:
    """Configuration for propeller wake validation."""

    # Domain
    nx: int = 160
    ny: int = 80
    nz: int = 80
    tau: float = 0.65
    smagorinsky_cs: float = 0.08

    # Propeller
    diameter: float = 32.0
    hub_diameter_ratio: float = 0.18
    rpm_lu: float = 0.0015
    inflow_velocity: float = 0.04

    # Simulation
    n_steps: int = 3000
    warmup_steps: int = 500
    sampling_interval: int = 50

    # Wake stations
    stations: tuple[WakeStation, ...] = (
        WakeStation(x_over_D=0.5),
        WakeStation(x_over_D=1.0),
        WakeStation(x_over_D=2.0),
    )

    # Output
    output_root: Path = field(default_factory=lambda: Path("outputs"))
    run_name: str | None = None
    device: str = "cpu"
    seed: int = 42
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def propeller_cx(self) -> float:
        return self.nx * 0.25

    @property
    def propeller_cy(self) -> float:
        return self.ny / 2.0

    @property
    def propeller_cz(self) -> float:
        return self.nz / 2.0

    @property
    def advance_ratio(self) -> float:
        return self.inflow_velocity / (self.rpm_lu * self.diameter)

    @property
    def kt_target(self) -> float:
        return _interp_kt(max(0.1, min(1.1, self.advance_ratio)))

    @property
    def kq_target(self) -> float:
        return _interp_kq(max(0.1, min(1.1, self.advance_ratio)))


# ---------------------------------------------------------------------------
# PIV reference data (synthetic, based on actuator disk theory + empirical)
# ---------------------------------------------------------------------------


class PIVReference:
    """Synthetic PIV reference data for propeller wake validation.

    Based on momentum theory for the axial velocity deficit and
    free-vortex model for the tangential (swirl) component.
    Calibrated to KP505 open-water characteristics.
    """

    def __init__(
        self,
        diameter: float,
        kt: float,
        kq: float,
        inflow_velocity: float,
        rpm: float,
        n_radial: int = 20,
        n_azimuthal: int = 36,
    ):
        self.D = diameter
        self.R = diameter / 2.0
        self.KT = kt
        self.KQ = kq
        self.U0 = inflow_velocity
        self.n = rpm
        self.n_r = n_radial
        self.n_theta = n_azimuthal

    def axial_velocity_deficit(
        self, x_over_D: float, r_over_R: torch.Tensor
    ) -> torch.Tensor:
        """Axial velocity deficit from momentum theory with empirical decay.

        U/U0 = 1 - delta_u(r) * exp(-x / (k * D))
        where delta_u follows a Gaussian radial distribution.
        """
        # Peak deficit at disk (x/D=0) from momentum theory
        # C_T = KT / J^2, delta_u_max ~ sqrt(C_T/2)
        J = self.U0 / (self.n * self.D) if self.n > 0 else 0.5
        CT = self.KT / (J**2 + 1e-10)
        delta_u_max = math.sqrt(CT / 2.0) * 0.8  # empirical reduction factor

        # Radial distribution: Gaussian centred at r/R=0.7 (blade loading peak)
        r_peak = 0.7
        sigma_r = 0.15
        radial_profile = torch.exp(-((r_over_R - r_peak) ** 2) / (2 * sigma_r**2))

        # Hub correction: no deficit in hub region
        hub_ratio = 0.18
        hub_mask = r_over_R < hub_ratio
        radial_profile = torch.where(hub_mask, torch.zeros_like(radial_profile), radial_profile)

        # Tip correction: rapid decay near tip
        tip_mask = r_over_R > 0.95
        tip_decay = torch.exp(-((r_over_R - 0.95) ** 2) / (2 * 0.02**2))
        radial_profile = torch.where(tip_mask, radial_profile * tip_decay, radial_profile)

        # Axial decay
        axial_decay = math.exp(-x_over_D / 1.5)

        return 1.0 - delta_u_max * radial_profile * axial_decay

    def tangential_velocity(
        self, x_over_D: float, r_over_R: torch.Tensor
    ) -> torch.Tensor:
        """Tangential (swirl) velocity from free-vortex model.

        V_theta ~ Gamma / (2*pi*r) * exp(-x / (k*D))
        with circulation Gamma derived from KQ.
        """
        # Circulation from torque: Q = rho * n * Gamma * A_disk
        # Simplified: V_theta ~ KQ * (D/r) * decay
        J = self.U0 / (self.n * self.D) if self.n > 0 else 0.5
        V_theta_peak = self.KQ * 0.5 / (J + 0.1)  # empirical scaling

        # Radial distribution: ~1/r in outer region, linear in hub
        r_safe = r_over_R.clamp(min=0.05)
        vortex_profile = torch.where(
            r_over_R < 0.2,
            r_over_R / 0.2,  # linear rise in hub region
            0.2 / r_safe,  # 1/r decay outside
        )

        # Tip correction
        tip_mask = r_over_R > 0.9
        tip_decay = torch.where(tip_mask, torch.exp(-((r_over_R - 0.9) ** 2) / 0.005), torch.ones_like(r_over_R))
        vortex_profile = vortex_profile * tip_decay

        # Axial decay
        axial_decay = math.exp(-x_over_D / 2.0)

        return V_theta_peak * vortex_profile * axial_decay

    def radial_velocity(
        self, x_over_D: float, r_over_R: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Radial velocity from wake expansion (small, ~1% of U0)."""
        # Wake expansion causes small radial outflow
        expansion_rate = 0.01 * math.exp(-x_over_D / 1.0)
        # Azimuthal variation due to blade passage
        azimuthal_var = 1.0 + 0.1 * torch.cos(5 * theta)  # 5 blades
        return expansion_rate * azimuthal_var * (1.0 - r_over_R)

    def get_profile(self, x_over_D: float) -> WakeProfile:
        """Generate PIV reference profile at a given station."""
        r_over_R = torch.linspace(0.0, 1.0, self.n_r)
        theta = torch.linspace(0, 2 * math.pi, self.n_theta)

        # Create meshgrid for full profile
        rr, tt = torch.meshgrid(r_over_R, theta, indexing="ij")

        u_axial = self.axial_velocity_deficit(x_over_D, rr)
        u_tangential = self.tangential_velocity(x_over_D, rr)
        u_radial = self.radial_velocity(x_over_D, rr, tt)

        return WakeProfile(
            x_over_D=x_over_D,
            r_over_R=rr,
            theta=tt,
            u_axial=u_axial,
            u_tangential=u_tangential,
            u_radial=u_radial,
        )


# ---------------------------------------------------------------------------
# Body-force wake simulation
# ---------------------------------------------------------------------------


def run_body_force_wake(
    config: WakeValidationConfig,
) -> dict[str, Any]:
    """Run body-force (actuator disk) wake simulation.

    Returns velocity fields and wake profiles at configured stations.
    """
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    nx, ny, nz = config.nx, config.ny, config.nz
    cx = config.propeller_cx
    cy = config.propeller_cy
    cz = config.propeller_cz
    u_in = config.inflow_velocity

    print(f"Body-force wake: D={config.diameter}, J={config.advance_ratio:.3f}, "
          f"KT={config.kt_target:.4f}, KQ={config.kq_target:.4f}")

    # Initialize with uniform inflow
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, uy0, uz0)

    # Wall mask (channel walls)
    from .boundaries3d import make_channel_wall_mask_3d
    hub_mask = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, hub_mask, device=device)

    # Run simulation
    wake_profiles: list[WakeProfile] = []
    step_count = 0

    for step in range(1, config.n_steps + 1):
        # Collision
        f = collide_bgk3d(f, config.tau)

        # Streaming
        f = stream3d(f)

        # Apply wall bounce-back
        from .boundaries3d import bounce_back_cells_3d
        f = bounce_back_cells_3d(f, wall_mask)

        # Apply body force (actuator disk)
        rho, ux, uy, uz = macroscopic3d(f)
        fx, fy, fz = apply_actuator_disk(
            ux, uy, uz,
            cx, cy, cz,
            config.diameter,
            config.hub_diameter_ratio,
            config.rpm_lu,
        )
        # Add force to velocity
        ux = ux + fx
        uy = uy + fy
        uz = uz + fz
        # Re-equilibrate
        f = equilibrium3d(rho, ux, uy, uz)

        # Sample wake profiles
        if step > config.warmup_steps and (step - config.warmup_steps) % config.sampling_interval == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            profiles = _sample_wake_stations(
                ux, uy, uz, u_in, config, cx, cy, cz, device
            )
            wake_profiles.extend(profiles)
            step_count += 1

    # Average profiles
    if wake_profiles:
        avg_profiles = _average_profiles(wake_profiles, device)
    else:
        avg_profiles = []

    # Compute thrust coefficient from body force
    thrust = float((-fx).sum().item())
    D = config.diameter
    n = config.rpm_lu
    kt_computed = thrust / (n**2 * D**4 + 1e-30)

    return {
        "method": "body_force",
        "wake_profiles": [p.to_dict() for p in avg_profiles],
        "kt_computed": kt_computed,
        "kt_target": config.kt_target,
        "kt_error_pct": abs(kt_computed - config.kt_target) / max(config.kt_target, 1e-10) * 100,
        "n_samples": step_count,
        "advance_ratio": config.advance_ratio,
    }


# ---------------------------------------------------------------------------
# Sliding-mesh wake simulation
# ---------------------------------------------------------------------------


def run_sliding_mesh_wake(
    config: WakeValidationConfig,
) -> dict[str, Any]:
    """Run sliding-mesh wake simulation with resolved blade geometry.

    Uses rotating inner domain with propeller mask, coupled to static
    outer domain via sliding interface.
    """
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    nx, ny, nz = config.nx, config.ny, config.nz
    cx = config.propeller_cx
    cy = config.propeller_cy
    cz = config.propeller_cz
    u_in = config.inflow_velocity
    omega = 2.0 * math.pi * config.rpm_lu

    print(f"Sliding-mesh wake: D={config.diameter}, J={config.advance_ratio:.3f}, "
          f"omega={omega:.6f}")

    # Initialize with uniform inflow
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = torch.full_like(rho0, u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, uy0, uz0)

    # Build propeller mask (at initial angle)
    prop_mask = build_propeller_mask(
        nx, ny, nz, cx, cy, cz, angle_deg=0.0,
        config=KP505_PRESET, device=device,
    )

    # Wall mask
    from .boundaries3d import make_channel_wall_mask_3d, bounce_back_cells_3d
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, prop_mask, device=device)

    # Sliding interface: annulus around propeller disk
    R = config.diameter / 2.0
    interface_width = 2.0
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r_sq = (yy - cy) ** 2 + (zz - cz) ** 2
    r = torch.sqrt(r_sq)
    interface_mask = (
        (r >= R - interface_width) & (r <= R + interface_width) &
        (torch.abs(xx - cx) <= 3.0)
    )

    # Run simulation
    wake_profiles: list[WakeProfile] = []
    theta = 0.0
    step_count = 0

    for step in range(1, config.n_steps + 1):
        # Update rotation angle
        theta += omega

        # Collision
        f = collide_bgk3d(f, config.tau)

        # Streaming
        f = stream3d(f)

        # Apply wall + propeller bounce-back
        f = bounce_back_cells_3d(f, wall_mask)

        # Apply sliding mesh BC at interface
        f = _apply_sliding_mesh_3d(f, interface_mask, theta, omega, config.tau, device)

        # Sample wake profiles
        if step > config.warmup_steps and (step - config.warmup_steps) % config.sampling_interval == 0:
            rho, ux, uy, uz = macroscopic3d(f)
            profiles = _sample_wake_stations(
                ux, uy, uz, u_in, config, cx, cy, cz, device
            )
            wake_profiles.extend(profiles)
            step_count += 1

    # Average profiles
    if wake_profiles:
        avg_profiles = _average_profiles(wake_profiles, device)
    else:
        avg_profiles = []

    # Compute thrust from momentum deficit
    if avg_profiles:
        kt_computed = _compute_kt_from_wake(avg_profiles, config, u_in)
    else:
        kt_computed = 0.0

    return {
        "method": "sliding_mesh",
        "wake_profiles": [p.to_dict() for p in avg_profiles],
        "kt_computed": kt_computed,
        "kt_target": config.kt_target,
        "kt_error_pct": abs(kt_computed - config.kt_target) / max(config.kt_target, 1e-10) * 100,
        "n_samples": step_count,
        "advance_ratio": config.advance_ratio,
    }


def _apply_sliding_mesh_3d(
    f: torch.Tensor,
    interface_mask: torch.Tensor,
    theta: float,
    omega: float,
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    """Apply 3-D sliding mesh boundary condition at interface.

    Rotates the velocity field in the inner zone and blends at interface.
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho, ux, uy, uz = macroscopic3d(f)

    # Compute rotor velocity at interface
    cx_abs = nx / 2.0
    cy_abs = ny / 2.0
    cz_abs = nz / 2.0

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Rotation about x-axis
    r_y = yy - cy_abs
    r_z = zz - cz_abs
    u_wall_x = torch.zeros_like(ux)
    u_wall_y = -omega * r_z
    u_wall_z = omega * r_y

    # Apply moving-wall equilibrium at interface
    ux_wall = torch.where(interface_mask, u_wall_y, ux)
    uy_wall = torch.where(interface_mask, u_wall_y, uy)
    uz_wall = torch.where(interface_mask, u_wall_z, uz)

    f_eq = equilibrium3d(rho, ux_wall, uy_wall, uz_wall)
    omega_relax = 1.0 / tau

    mask_exp = interface_mask.unsqueeze(0).expand(19, -1, -1, -1)
    f_new = f - omega_relax * (f - f_eq)
    return torch.where(mask_exp, f_new, f)


def _compute_kt_from_wake(
    profiles: list[WakeProfile],
    config: WakeValidationConfig,
    u_in: float,
) -> float:
    """Estimate K_T from wake momentum deficit."""
    if not profiles:
        return 0.0
    # Use profile at x/D=1.0
    for p in profiles:
        if abs(p.x_over_D - 1.0) < 0.1:
            # Integrate momentum deficit
            u_deficit = 1.0 - p.u_axial
            # Simplified: average deficit over disk area
            avg_deficit = float(u_deficit.mean().item())
            CT = 2.0 * avg_deficit  # momentum theory: CT = 2 * delta_u
            return CT * (u_in**2) / (config.rpm_lu**2 * config.diameter**4 + 1e-30)
    return 0.0


# ---------------------------------------------------------------------------
# Wake profile sampling
# ---------------------------------------------------------------------------


def _sample_wake_stations(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    u_in: float,
    config: WakeValidationConfig,
    cx: float,
    cy: float,
    cz: float,
    device: torch.device,
) -> list[WakeProfile]:
    """Sample velocity profiles at configured wake stations."""
    profiles = []
    R = config.diameter / 2.0

    for station in config.stations:
        x_station = cx + station.x_over_D * config.diameter
        x_idx = int(round(x_station))
        if x_idx < 0 or x_idx >= config.nx:
            continue

        # Extract plane at x_station
        ux_plane = ux[:, :, x_idx]  # (nz, ny)
        uy_plane = uy[:, :, x_idx]
        uz_plane = uz[:, :, x_idx]

        # Convert to cylindrical coordinates
        nz, ny = ux_plane.shape
        zz, yy = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(ny, device=device, dtype=torch.float32),
            indexing="ij",
        )
        dy = yy - cy
        dz = zz - cz
        r = torch.sqrt(dy**2 + dz**2)
        theta = torch.atan2(dz, dy)

        # Sample at station grid
        r_over_R = torch.linspace(0.0, 1.0, station.n_radial)
        theta_vals = torch.linspace(0, 2 * math.pi, station.n_azimuthal)
        rr, tt = torch.meshgrid(r_over_R, theta_vals, indexing="ij")

        # Interpolate (simplified: nearest neighbor for speed)
        r_abs = rr * R
        y_sample = cy + r_abs * torch.cos(tt)
        z_sample = cz + r_abs * torch.sin(tt)

        y_idx = torch.clamp(torch.round(y_sample).long(), 0, ny - 1)
        z_idx = torch.clamp(torch.round(z_sample).long(), 0, nz - 1)

        u_axial = ux_plane[z_idx, y_idx] / max(u_in, 1e-10)
        # Tangential: projection onto theta direction
        u_tangential = (-uy_plane[z_idx, y_idx] * torch.sin(tt) +
                        uz_plane[z_idx, y_idx] * torch.cos(tt)) / max(u_in, 1e-10)
        # Radial: projection onto r direction
        u_radial = (uy_plane[z_idx, y_idx] * torch.cos(tt) +
                    uz_plane[z_idx, y_idx] * torch.sin(tt)) / max(u_in, 1e-10)

        profiles.append(WakeProfile(
            x_over_D=station.x_over_D,
            r_over_R=rr,
            theta=tt,
            u_axial=u_axial,
            u_tangential=u_tangential,
            u_radial=u_radial,
        ))

    return profiles


def _average_profiles(
    profiles: list[WakeProfile],
    device: torch.device,
) -> list[WakeProfile]:
    """Average multiple wake profile samples."""
    if not profiles:
        return []

    # Group by station
    by_station: dict[float, list[WakeProfile]] = {}
    for p in profiles:
        by_station.setdefault(p.x_over_D, []).append(p)

    avg_profiles = []
    for x_d, station_profiles in sorted(by_station.items()):
        if not station_profiles:
            continue

        # Average tensors
        u_axial_avg = torch.stack([p.u_axial for p in station_profiles]).mean(dim=0)
        u_tangential_avg = torch.stack([p.u_tangential for p in station_profiles]).mean(dim=0)
        u_radial_avg = torch.stack([p.u_radial for p in station_profiles]).mean(dim=0)

        avg_profiles.append(WakeProfile(
            x_over_D=x_d,
            r_over_R=station_profiles[0].r_over_R,
            theta=station_profiles[0].theta,
            u_axial=u_axial_avg,
            u_tangential=u_tangential_avg,
            u_radial=u_radial_avg,
        ))

    return avg_profiles


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_wake_results(
    body_force_results: dict[str, Any],
    sliding_mesh_results: dict[str, Any],
    config: WakeValidationConfig,
) -> dict[str, Any]:
    """Validate wake results against acceptance criteria."""

    # Generate PIV reference with matching dimensions
    # Use the first station's sampling parameters
    n_r = config.stations[0].n_radial if config.stations else 20
    n_theta = config.stations[0].n_azimuthal if config.stations else 36
    piv = PIVReference(
        diameter=config.diameter,
        kt=config.kt_target,
        kq=config.kq_target,
        inflow_velocity=config.inflow_velocity,
        rpm=config.rpm_lu,
        n_radial=n_r,
        n_azimuthal=n_theta,
    )

    validation = {
        "A1_axial_rms_pct": None,
        "A2_tangential_rms_pct": None,
        "A3_farfield_deviation_pct": None,
        "A4_kt_error_pct": body_force_results.get("kt_error_pct", 999),
        "A5_uniformity_pct": None,
        "A6_pytest_count": None,
        "passed": {},
        "details": {},
    }

    # A1: Axial velocity at x/D=1.0 vs PIV
    bf_profiles = body_force_results.get("wake_profiles", [])
    sm_profiles = sliding_mesh_results.get("wake_profiles", [])

    for prof in bf_profiles:
        if abs(prof["x_over_D"] - 1.0) < 0.1:
            u_axial_sim = torch.tensor(prof["u_axial"])
            piv_prof = piv.get_profile(1.0)
            u_axial_piv = piv_prof.u_axial
            rms_error = float(torch.sqrt(((u_axial_sim - u_axial_piv) ** 2).mean()).item())
            validation["A1_axial_rms_pct"] = rms_error * 100
            validation["passed"]["A1"] = rms_error <= 0.10
            break

    # A2: Tangential velocity at x/D=1.0 vs PIV
    for prof in bf_profiles:
        if abs(prof["x_over_D"] - 1.0) < 0.1:
            u_tan_sim = torch.tensor(prof["u_tangential"])
            piv_prof = piv.get_profile(1.0)
            u_tan_piv = piv_prof.u_tangential
            rms_error = float(torch.sqrt(((u_tan_sim - u_tan_piv) ** 2).mean()).item())
            validation["A2_tangential_rms_pct"] = rms_error * 100
            validation["passed"]["A2"] = rms_error <= 0.15
            break

    # A3: Far-field comparison (x/D=2.0)
    bf_far = None
    sm_far = None
    for prof in bf_profiles:
        if abs(prof["x_over_D"] - 2.0) < 0.1:
            bf_far = torch.tensor(prof["u_axial"])
            break
    for prof in sm_profiles:
        if abs(prof["x_over_D"] - 2.0) < 0.1:
            sm_far = torch.tensor(prof["u_axial"])
            break

    if bf_far is not None and sm_far is not None:
        farfield_dev = float(((bf_far - sm_far).abs().mean() / (bf_far.abs().mean() + 1e-10)).item())
        validation["A3_farfield_deviation_pct"] = farfield_dev * 100
        validation["passed"]["A3"] = farfield_dev <= 0.05

    # A4: K_T error (already computed)
    validation["passed"]["A4"] = validation["A4_kt_error_pct"] <= 8.0

    # A5: No-propeller uniformity (run separately)
    # This is tested in the test suite

    return validation


# ---------------------------------------------------------------------------
# Full validation runner
# ---------------------------------------------------------------------------


def run_wake_validation(
    config: WakeValidationConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run full wake validation: body-force + sliding-mesh + PIV comparison."""
    if config is None:
        config = WakeValidationConfig(**{k: v for k, v in kwargs.items() if hasattr(WakeValidationConfig, k)})

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "propeller_wake_validation",
        config.run_name or f"pwv_D{int(config.diameter)}_J{config.advance_ratio:.2f}",
        config.overwrite,
    )

    print(f"\n{'=' * 60}")
    print(f"Propeller Wake Validation")
    print(f"{'=' * 60}")
    print(f"  Diameter: {config.diameter} lu")
    print(f"  Inflow:   {config.inflow_velocity} lu/step")
    print(f"  RPM:      {config.rpm_lu}")
    print(f"  J:        {config.advance_ratio:.3f}")
    print(f"  KT target:{config.kt_target:.4f}")
    print(f"  Domain:   {config.nx}x{config.ny}x{config.nz}")
    print(f"  Device:   {device}")
    print()

    # Run body-force method
    print("--- Body Force Method ---")
    bf_results = run_body_force_wake(config)
    print(f"  KT computed: {bf_results['kt_computed']:.4f} (target: {bf_results['kt_target']:.4f})")
    print(f"  KT error:    {bf_results['kt_error_pct']:.1f}%")
    print(f"  Samples:     {bf_results['n_samples']}")

    # Run sliding-mesh method
    print("\n--- Sliding Mesh Method ---")
    sm_results = run_sliding_mesh_wake(config)
    print(f"  KT computed: {sm_results['kt_computed']:.4f} (target: {sm_results['kt_target']:.4f})")
    print(f"  KT error:    {sm_results['kt_error_pct']:.1f}%")
    print(f"  Samples:     {sm_results['n_samples']}")

    # Validate
    print("\n--- Validation ---")
    validation = validate_wake_results(bf_results, sm_results, config)

    # Save results
    results = {
        "body_force": bf_results,
        "sliding_mesh": sm_results,
        "validation": validation,
        "config": asdict(config),
        "reproducibility": get_reproducibility_metadata(),
    }

    results_path = run_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {results_path}")

    return results
