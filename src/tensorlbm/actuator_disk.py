"""Actuator disk model for propeller simulation.

Applies thrust and swirl body forces in a disk region to model
a propeller without resolving blade geometry.  This is the standard
approach in ship CFD (RANS + body-force propeller).

Model
-----
A disk of diameter D at position (cx, cy, cz) normal to the x-axis.
For each grid cell inside the disk, applies:

  f_x = T_vol  (thrust density, negative = forward thrust)
  f_θ = Q_vol  (swirl density, tangential)

where T_vol and Q_vol are computed from KT and KQ curves:

  T  = KT(J) * rho * n^2 * D^4
  Q  = KQ(J) * rho * n^2 * D^5

  T_vol = T / V_disk  [force per unit volume]
  Q_vol = Q / (V_disk * r)  [torque density, distributed radially]

The advance ratio J = u_in / (n * D) is computed from the local
inflow velocity at the disk face.

Reference
---------
- ITTC (2014) "Recommended Procedures: Open Water Test", 7.5-02-03-02.1
- KP505 open-water data: Fujisawa et al. (2000), SIMMAN 2008.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .solver3d import stream3d
from .turbulence import collide_smagorinsky_mrt3d
from .boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from .propeller_cad import PropellerGeometryConfig, KP505_PRESET
from .utils import get_reproducibility_metadata, prepare_run_dir, resolve_device


# ---------------------------------------------------------------------------
# Reference KT/KQ curves (KP505, Fujisawa et al. 2000)
# ---------------------------------------------------------------------------

_KP505_KT_J = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
_KP505_KT_V = [0.45, 0.44, 0.42, 0.40, 0.37, 0.33, 0.29, 0.24, 0.17, 0.10, 0.04]
_KP505_KQ_J = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
_KP505_KQ_V = [0.065, 0.063, 0.061, 0.059, 0.055, 0.051, 0.047, 0.041, 0.033, 0.024, 0.015]


def _interp_kt(j_val: float) -> float:
    """Linear interpolation of KT from KP505 data."""
    j_t = torch.tensor(_KP505_KT_J)
    kt_t = torch.tensor(_KP505_KT_V)
    idx = int(torch.searchsorted(j_t, torch.tensor(j_val)).clamp(1, len(j_t) - 1).item())
    return float(kt_t[idx])


def _interp_kq(j_val: float) -> float:
    """Linear interpolation of KQ from KP505 data."""
    j_t = torch.tensor(_KP505_KQ_J)
    kq_t = torch.tensor(_KP505_KQ_V)
    idx = int(torch.searchsorted(j_t, torch.tensor(j_val)).clamp(1, len(j_t) - 1).item())
    return float(kq_t[idx])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ActuatorDiskConfig:
    """Configuration for actuator disk propeller simulation.

    Parameters
    ----------
    diameter : float
        Propeller disk diameter [lu].
    hub_diameter_ratio : float
        Hub diameter / propeller diameter.
    rpm_lu : float
        Revolutions per LATTICE STEP.  For J=0.7 with u_in=0.04 and D=48:
        rpm_lu = 0.04/(0.7*48) ≈ 0.00119.  Use this (not physical rps).
    inflow_velocities : tuple[float, ...]
        Inflow velocities [lu/step] for the J-sweep.
    nx, ny, nz : int
        Domain size.
    tau : float
        Relaxation time.
    smagorinsky_cs : float
        Smagorinsky constant.
    n_steps : int
        Total simulation steps per speed.
    warmup_steps : int
        Warmup steps before sampling.
    output_root : Path
        Output directory.
    device : str
        Computation device.
    run_name : str or None
        Custom run name.
    seed : int
        Random seed.
    overwrite : bool
        Overwrite existing output.
    """

    diameter: float = 48.0
    hub_diameter_ratio: float = 0.18
    rpm_lu: float = 0.0012
    inflow_velocities: tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10)
    nx: int = 200
    ny: int = 100
    nz: int = 100
    tau: float = 0.58
    smagorinsky_cs: float = 0.1
    n_steps: int = 5000
    warmup_steps: int = 1000
    output_root: Path = field(default_factory=lambda: Path("outputs"))
    device: str = "cpu"
    run_name: str | None = None
    seed: int = 0
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.device = self.device.lower()

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def hub_radius(self) -> float:
        return self.radius * self.hub_diameter_ratio

    @property
    def disk_volume(self) -> float:
        disk_area = math.pi * (self.radius**2 - self.hub_radius**2)
        return disk_area * 2.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        d = asdict(self)
        d["output_root"] = str(d["output_root"])
        path.write_text(f"{json.dumps(d, indent=2, sort_keys=True)}\\n", encoding="utf-8")
        return path


# ============================================================================
# Actuator disk force application
# ============================================================================

def apply_actuator_disk(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
    diameter: float,
    hub_diameter_ratio: float,
    rpm: float,
    rho: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute body forces for an actuator disk propeller.

    Parameters
    ----------
    ux, uy, uz : torch.Tensor, shape (nz, ny, nx)
        Velocity fields.
    cx, cy, cz : float
        Disk center coordinates.
    diameter : float
        Disk diameter [lu].
    hub_diameter_ratio : float
        Hub diameter ratio.
    rpm : float
        Revolutions per second [rps].
    rho : float
        Reference density.

    Returns
    -------
    fx, fy, fz : torch.Tensor, shape (nz, ny, nx)
        Body force fields.
    """
    device = ux.device
    nz, ny, nx = ux.shape
    R = diameter / 2.0
    R_hub = R * hub_diameter_ratio

    # Disk mask
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r_sq = (yy - cy)**2 + (zz - cz)**2
    disk_mask = (r_sq >= R_hub**2) & (r_sq <= R**2) & (torch.abs(xx - cx) <= 1.0)
    r = torch.sqrt(r_sq.clamp(min=1e-10))

    # Local inflow at disk (averaged over disk for J computation)
    u_in_local = ux[disk_mask].mean().clamp(min=1e-6).item() if disk_mask.any() else 0.0
    J = u_in_local / (rpm * diameter) if rpm > 0 else 0.0

    # Interpolate KT, KQ from reference data
    kt = _interp_kt(max(0.1, min(1.1, J)))
    kq = _interp_kq(max(0.1, min(1.1, J)))

    # Thrust and torque
    D = diameter
    n2d4 = rpm**2 * D**4
    n2d5 = rpm**2 * D**5
    thrust_total = kt * rho * n2d4
    torque_total = kq * rho * n2d5

    # Volume-specific forces
    disk_area = math.pi * (R**2 - R_hub**2)
    thickness = 2.0
    vol = disk_area * thickness
    t_vol = thrust_total / max(vol, 1e-10)  # force per unit volume
    q_vol = torque_total / max(vol, 1e-10)  # torque density

    # Apply: thrust is NEGATIVE x-direction (pushes fluid backward)
    fx = torch.zeros_like(ux)
    fy = torch.zeros_like(uy)
    fz = torch.zeros_like(uz)

    # Thrust: uniform over disk
    fx[disk_mask] = -t_vol  # negative = pushes fluid in -x direction

    # Swirl: tangential force f_theta = Q_vol / r
    theta = torch.atan2(zz - cz, yy - cy)
    swirl_magnitude = q_vol / r.clamp(min=1e-10)
    fz[disk_mask] += swirl_magnitude[disk_mask] * torch.cos(theta[disk_mask])
    fy[disk_mask] += -swirl_magnitude[disk_mask] * torch.sin(theta[disk_mask])

    return fx, fy, fz


# ============================================================================
# Simulation runner
# ============================================================================

def _compute_thrust_torque(
    fx_disk: torch.Tensor,
    fy_disk: torch.Tensor,
    fz_disk: torch.Tensor,
    cy: float,
    cz: float,
) -> tuple[float, float]:
    """Compute thrust and torque from actuator disk body forces.

    Thrust = -sum(fx) over disk (negative fx = forward thrust)
    Torque = sum(r × f)_x = sum(-(z-cz)*fy + (y-cy)*fz)
    """
    thrust = float(-fx_disk.sum().item())
    device = fx_disk.device
    nz, ny, nx = fx_disk.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    torque = float((-((zz - cz) * fy_disk - (yy - cy) * fz_disk)).sum().item())
    return thrust, torque


def run_actuator_disk_benchmark(
    config: ActuatorDiskConfig,
) -> dict[str, object]:
    """Run actuator disk propeller benchmark."""
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    nx, ny, nz = config.nx, config.ny, config.nz
    cx = int(nx * 0.3)
    cy = ny // 2
    cz = nz // 2

    print(f"Actuator Disk Propeller Benchmark")
    print(f"  Device:     {device}")
    print(f"  Diameter:   {config.diameter} lu   "
          f"Hub ratio: {config.hub_diameter_ratio}")
    print(f"  RPM:        {config.rpm_lu} rps   "
          f"Domain: {nx}x{ny}x{nz}")
    print(f"  tau:        {config.tau:.3f}  Cs: {config.smagorinsky_cs:.2f}")
    print(f"  Steps:      {config.n_steps}  warmup: {config.warmup_steps}")
    print()

    run_dir = prepare_run_dir(
        config.output_root, "actuator_disk",
        f"ad_D{int(config.diameter)}_rpm{int(config.rpm_lu)}_nx{nx}",
        config.overwrite,
    )
    config.save(run_dir / "config.json")

    results: list[dict[str, object]] = []

    for u_in in config.inflow_velocities:
        J = u_in / (config.rpm_lu * config.diameter)
        kt_target = _interp_kt(max(0.1, min(1.1, J)))
        kq_target = _interp_kq(max(0.1, min(1.1, J)))
        print(f"{'='*60}")
        print(f"  u_in={u_in:.3f}  J={J:.3f}  "
              f"KT_target={kt_target:.4f}  KQ_target={kq_target:.4f}")
        print(f"{'='*60}")

        # Initialize
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
        ux0 = torch.full_like(rho0, u_in)
        f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)

        # Optional: small cylinder as hub model
        yy, zz, xx = torch.meshgrid(
            torch.arange(ny, device=device, dtype=torch.float32),
            torch.arange(nz, device=device, dtype=torch.float32),
            torch.arange(nx, device=device, dtype=torch.float32),
            indexing="ij",
        )
        r_sq = (xx.permute(1, 0, 2) - cx)**2
        hub_r_sq = (yy.permute(1, 0, 2) - cy)**2 + (zz.permute(1, 0, 2) - cz)**2
        hub_mask = hub_r_sq <= (config.hub_radius * 1.5)**2
        wall_mask = make_channel_wall_mask_3d(nz, ny, nx, hub_mask, device=device)

        thrust_list: list[float] = []
        torque_list: list[float] = []
        t_start = time.perf_counter()

        for step in range(1, config.n_steps + 1):
            # Collision + streaming
            f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
            f = stream3d(f)

            # Macroscopic fields
            rho, ux, uy, uz = macroscopic3d(f)

            # Apply actuator disk body forces
            fx_ad, fy_ad, fz_ad = apply_actuator_disk(
                ux, uy, uz, cx, cy, cz, config.diameter,
                config.hub_diameter_ratio, config.rpm_lu,
            )
            # Add forces to velocity (direct forcing)
            ux = ux + fx_ad
            uy = uy + fy_ad
            uz = uz + fz_ad

            # Re-equilibrate after force
            f = equilibrium3d(rho, ux, uy, uz, device=device)

            # Boundary conditions
            f = apply_zou_he_channel_boundaries_3d(
                f, u_in=u_in, wall_mask=wall_mask,
                obstacle_mask=hub_mask,
            )

            if step > config.warmup_steps and step % 10 == 0:
                thr, tor = _compute_thrust_torque(fx_ad, fy_ad, fz_ad, cy, cz)
                thrust_list.append(thr)
                torque_list.append(tor)

            if step % 500 == 0 or step == config.n_steps:
                elapsed = time.perf_counter() - t_start
                pct = 100 * step / config.n_steps
                n_samples = len(thrust_list)
                T_avg = sum(thrust_list[-100:]) / max(min(n_samples, 100), 1) if thrust_list else 0
                Q_avg = sum(torque_list[-100:]) / max(min(n_samples, 100), 1) if torque_list else 0
                print(f"  step {step}/{config.n_steps} ({pct:.0f}%)  "
                      f"T={T_avg:.2e}  Q={Q_avg:.2e}  elapsed={elapsed:.1f}s")

        # Compute KT, KQ from measured thrust/torque
        n2d4 = config.rpm_lu**2 * config.diameter**4
        n2d5 = config.rpm_lu**2 * config.diameter**5
        T_mean = sum(thrust_list) / max(len(thrust_list), 1)
        Q_mean = sum(torque_list) / max(len(torque_list), 1)
        kt_measured = T_mean / n2d4 if n2d4 > 0 else 0
        kq_measured = Q_mean / n2d5 if n2d5 > 0 else 0
        eta = (J / (2.0 * math.pi)) * (kt_measured / kq_measured) if kq_measured > 0 else 0

        print(f"  -> KT_prev={kt_target:.4f}  KT_meas={kt_measured:.6f}  "
              f"KQ_meas={kq_measured:.6f}  eta={eta:.4f}\n")

        results.append({
            "j": J, "u_in": u_in,
            "kt_target": kt_target, "kq_target": kq_target,
            "kt_measured": kt_measured, "kq_measured": kq_measured,
            "eta_o": eta,
            "thrust_mean": T_mean, "torque_mean": Q_mean,
            "runtime_s": time.perf_counter() - t_start,
        })

    # Write CSV
    csv_path = run_dir / "actuator_disk.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["J", "KT_target", "KT_meas", "KQ_target", "KQ_meas", "eta"])
        for r in results:
            writer.writerow([
                f"{r['j']:.4f}", f"{r['kt_target']:.4f}",
                f"{r['kt_measured']:.6f}", f"{r['kq_target']:.6f}",
                f"{r['kq_measured']:.6f}", f"{r['eta_o']:.4f}",
            ])

    # Summary table
    print(f"\n{'='*65}")
    print("  Actuator Disk Results")
    print(f"{'='*65}")
    print(f"  {'J':>6s}  {'KT_target':>10s}  {'KT_meas':>10s}  "
          f"{'KQ_target':>10s}  {'KQ_meas':>10s}  {'eta':>8s}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['j']:6.3f}  {r['kt_target']:10.4f}  "
              f"{r['kt_measured']:10.6f}  {r['kq_target']:10.4f}  "
              f"{r['kq_measured']:10.6f}  {r['eta_o']:8.4f}")

    metadata = {
        "name": "actuator_disk",
        "config": asdict(config),
        "results": results,
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }
    json_path = run_dir / "run_metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\n  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    return metadata


__all__ = [
    "ActuatorDiskConfig",
    "apply_actuator_disk",
    "run_actuator_disk_benchmark",
]
