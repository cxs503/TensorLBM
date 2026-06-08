"""IBM (Immersed Boundary Method) propeller benchmark.

Uses Lagrangian marker points on the propeller surface, rotating them
each step and applying IBM direct forcing to impose the no-slip
condition on the moving blades.

The IBM approach differs from Ladd bounce-back in two ways:
1. Marker points TRACK the rotating geometry (position updates each step)
2. Forces are SPREAD to nearby Eulerian cells via a regularized delta
   kernel, which is more stable than Ladd's local correction

Reference
---------
- Uhlmann (2005) "An immersed boundary method with direct forcing"
- Breugem (2012) "A second-order accurate immersed boundary method"
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .ibm import (
    ibm_direct_forcing_3d,
    ibm_apply_body_force_3d,
    ibm_force_spread_3d,
    ibm_velocity_interpolate_3d,
)
from .d3q19 import equilibrium3d, macroscopic3d
from .solver3d import stream3d
from .turbulence import collide_smagorinsky_mrt3d
from .boundaries3d import apply_zou_he_channel_boundaries_3d, make_channel_wall_mask_3d
from .propeller_cad import PropellerGeometryConfig, KP505_PRESET
from .utils import get_reproducibility_metadata, prepare_run_dir, resolve_device


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IBMPropellerConfig:
    """Configuration for IBM propeller simulation."""

    geometry: PropellerGeometryConfig = field(default_factory=lambda: KP505_PRESET)
    inflow_velocities: tuple[float, ...] = (0.005, 0.010, 0.015)
    rpm: float = 0.00001
    nx: int = 120
    ny: int = 60
    nz: int = 60
    tau: float = 0.58
    smagorinsky_cs: float = 0.1
    n_revolutions: int = 1
    warmup_steps: int = 200
    marker_spacing: float = 1.5
    ibm_dt_substeps: int = 1
    device: str = "cpu"
    output_root: Path = field(default_factory=lambda: Path("outputs"))
    run_name: str | None = None
    seed: int = 0
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.device = self.device.lower()

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.rpm


# ============================================================================
# Marker generation on propeller surface
# ============================================================================

def _generate_propeller_markers(
    config: PropellerGeometryConfig,
    cx: float,
    cy: float,
    cz: float,
    angle_deg: float = 0.0,
    spacing: float = 1.5,
) -> torch.Tensor:
    """Generate Lagrangian marker points on propeller surface.

    Uses voxel mask to extract surface cells, then places markers
    at cell centres.  Rotated by angle_deg about the x-axis.

    Returns:
        markers: torch.Tensor of shape (N, 3) in (x, y, z) coordinates.
    """
    from .propeller_cad import build_propeller_mask

    # Build mask at double resolution for better surface detection
    nx = int(config.diameter * 6)
    ny = int(config.diameter * 3)
    nz = ny
    cx2 = nx // 2
    cy2 = ny // 2
    cz2 = nz // 2

    mask = build_propeller_mask(
        nx=nx, ny=ny, nz=nz, cx=cx2, cy=cy2, cz=cz2,
        angle_deg=angle_deg, config=config, device="cpu",
    )

    # Extract surface cells (solid cells with at least one fluid neighbor)
    nz2, ny2, nx2 = mask.shape
    surface = torch.zeros_like(mask)
    # Pad for boundary
    m_pad = torch.nn.functional.pad(mask.float(), (1, 1, 1, 1, 1, 1), value=0.0) > 0.5
    # Surface = solid AND (NOT all neighbors solid)
    for di, dj, dk in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        ni = m_pad[1 + di:1 + di + nz2, 1 + dj:1 + dj + ny2, 1 + dk:1 + dk + nx2]
        surface = surface | (mask & ~ni)

    # Sample markers at regular spacing
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz2, dtype=torch.float32),
        torch.arange(ny2, dtype=torch.float32),
        torch.arange(nx2, dtype=torch.float32),
        indexing="ij",
    )
    surf_indices = surface.nonzero(as_tuple=False)
    step = max(1, int(spacing))
    selected = surf_indices[::step]
    if selected.numel() == 0:
        return torch.zeros(0, 3)

    markers_x = selected[:, 2].float() / 2.0 + cx - cx2 / 2.0
    markers_y = selected[:, 1].float() / 2.0 + cy - cy2 / 2.0
    markers_z = selected[:, 0].float() / 2.0 + cz - cz2 / 2.0
    markers = torch.stack([markers_x, markers_y, markers_z], dim=1)

    # Apply rotation about x-axis
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = torch.tensor([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a],
    ], dtype=torch.float32)
    markers_rel = markers - torch.tensor([cx, cy, cz], dtype=torch.float32)
    markers = markers_rel @ rot.T + torch.tensor([cx, cy, cz], dtype=torch.float32)

    return markers


# ============================================================================
# Simulation runner
# ============================================================================

def _compute_propeller_forces(
    markers: torch.Tensor,
    markers_fx: torch.Tensor,
    markers_fy: torch.Tensor,
    markers_fz: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
) -> tuple[float, float]:
    """Compute thrust (Fx) and torque (Mx) from IBM marker forces."""
    fx = float(markers_fx.sum().item())
    dy = markers[:, 1] - cy
    dz = markers[:, 2] - cz
    mx = float((dy * markers_fz - dz * markers_fy).sum().item())
    return fx, mx


def run_ibm_propeller_benchmark(
    config: IBMPropellerConfig,
) -> dict[str, object]:
    """Run IBM propeller benchmark."""
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    nx, ny, nz = config.nx, config.ny, config.nz
    cx = int(nx * 0.35)
    cy = ny // 2
    cz = nz // 2
    D = config.geometry.diameter

    print(f"IBM Propeller Benchmark")
    print(f"  Device:     {device}")
    print(f"  Blades:     {config.geometry.n_blades}")
    print(f"  Diameter:   {D} lu")
    print(f"  Domain:     {nx}x{ny}x{nz}")
    print(f"  tau:        {config.tau:.3f}  Cs: {config.smagorinsky_cs:.2f}")
    print(f"  RPM:        {config.rpm:.2e}")
    print(f"  omega:      {config.omega:.2e} rad/step")
    tip_ma = config.omega * D / 2.0 / 0.577
    print(f"  tip Ma:     {tip_ma:.4f}")
    print()

    # Generate markers once
    t0 = time.perf_counter()
    markers_base = _generate_propeller_markers(
        config.geometry, cx, cy, cz, angle_deg=0.0,
        spacing=config.marker_spacing,
    )
    n_markers = markers_base.shape[0]
    print(f"Generated {n_markers} markers in {time.perf_counter() - t0:.1f}s")
    if n_markers == 0:
        raise RuntimeError("No markers generated — check propeller CAD")

    # Centre markers
    markers_centred = markers_base - torch.tensor([cx, cy, cz], dtype=torch.float32)

    # Wall mask
    wall_mask = make_channel_wall_mask_3d(
        nz, ny, nx, torch.zeros(nz, ny, nx, dtype=torch.bool, device=device), device=device,
    )

    run_dir = prepare_run_dir(
        config.output_root, "ibm_propeller",
        f"ibm_n{D}_nx{nx}_rpm{config.rpm:.1e}_{n_markers}markers",
        config.overwrite,
    )

    results: list[dict[str, object]] = []

    for u_in in config.inflow_velocities:
        J = u_in / (config.rpm * D)
        print(f"{'='*60}")
        print(f"  u_in={u_in:.3f}  J={J:.1f}  markers={n_markers}")
        print(f"{'='*60}")

        steps_per_rev = max(1, int(1.0 / max(config.rpm, 1e-10)))
        n_sampling = config.n_revolutions * steps_per_rev
        n_total = config.warmup_steps + n_sampling

        # Initialise fluid
        rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
        ux0 = torch.full_like(rho0, u_in)
        f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)

        fx_list: list[float] = []
        mx_list: list[float] = []
        t_start = time.perf_counter()

        for step in range(1, n_total + 1):
            # Collision + streaming
            f = collide_smagorinsky_mrt3d(f, tau=config.tau, C_s=config.smagorinsky_cs)
            f = stream3d(f)

            # Current rotation angle
            angle_rad = (step * config.omega) % (2.0 * math.pi)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Rotate markers: y' = y*cos - z*sin, z' = y*sin + z*cos
            mx = markers_centred[:, 0] + cx
            my = markers_centred[:, 1] * cos_a - markers_centred[:, 2] * sin_a + cy
            mz = markers_centred[:, 1] * sin_a + markers_centred[:, 2] * cos_a + cz

            # Target velocity at markers (rigid-body rotation)
            u_tgt_x = torch.zeros(n_markers, device=device, dtype=torch.float32)
            u_tgt_y = -config.omega * (mz - cz)
            u_tgt_z = config.omega * (my - cy)
            u_tgt_x_dev = u_tgt_x.to(device)
            u_tgt_y_dev = u_tgt_y.to(device)
            u_tgt_z_dev = u_tgt_z.to(device)

            # IBM direct forcing (interpolation + spread in one call)
            rho, ux, uy, uz = macroscopic3d(f)
            fx_body, fy_body, fz_body = ibm_direct_forcing_3d(
                ux, uy, uz,
                mx.to(device), my.to(device), mz.to(device),
                u_tgt_x_dev, u_tgt_y_dev, u_tgt_z_dev,
                kernel="hat",
            )
            f = ibm_apply_body_force_3d(f, fx_body, fy_body, fz_body)

            # Boundary conditions
            f = apply_zou_he_channel_boundaries_3d(
                f, u_in=u_in, wall_mask=wall_mask,
                obstacle_mask=torch.zeros_like(wall_mask),
            )

            # Compute thrust + torque from the IBM body forces (spread to grid)
            if step > config.warmup_steps:
                thrust = float(-fx_body.sum().item())  # thrust = -Fx
                torque = float(-(my.to(device) - cy) * fx_body[0, 0, :].sum().item())  # simplified
                fx_list.append(thrust)
                mx_list.append(torque)

            # Progress
            if step % 1000 == 0 or step == n_total:
                elapsed = time.perf_counter() - t_start
                pct = 100 * step / n_total
                T_mean = sum(fx_list[-500:]) / max(min(len(fx_list), 500), 1) if fx_list else 0
                Q_mean = sum(mx_list[-500:]) / max(min(len(mx_list), 500), 1) if mx_list else 0
                print(f"  step {step}/{n_total} ({pct:.0f}%)  "
                      f"T={T_mean:.2e}  Q={Q_mean:.2e}  "
                      f"markers={n_markers}  elapsed={elapsed:.1f}s")

        # Compute KT/KQ
        fx_mean = sum(fx_list) / max(len(fx_list), 1)
        mx_mean = sum(mx_list) / max(len(mx_list), 1)
        n2d4 = config.rpm**2 * D**4
        n2d5 = config.rpm**2 * D**5
        kt = fx_mean / n2d4 if n2d4 > 0 else 0
        kq = mx_mean / n2d5 if n2d5 > 0 else 0
        eta = (J / (2.0 * math.pi)) * (kt / kq) if kq > 0 else 0

        print(f"  -> J={J:.1f}  KT={kt:.2f}  KQ={kq:.2f}  "
              f"10KQ={10*kq:.2f}  eta={eta:.4f}\n")

        results.append({
            "j": J, "u_in": u_in,
            "kt": kt, "kq": kq, "eta_o": eta,
            "n_markers": n_markers,
            "runtime_s": time.perf_counter() - t_start,
        })

    # Write CSV
    csv_path = run_dir / "ibm_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["J", "KT", "10KQ", "eta", "n_markers"])
        for r in results:
            writer.writerow([
                f"{r['j']:.4f}", f"{r['kt']:.2f}",
                f"{10*r['kq']:.2f}", f"{r['eta_o']:.4f}",
                f"{r['n_markers']}",
            ])

    # Summary table
    print(f"\n{'='*55}")
    print("  IBM Propeller Results")
    print(f"{'='*55}")
    print(f"  {'J':>6s}  {'KT':>10s}  {'10KQ':>10s}  {'eta':>8s}  {'markers':>8s}")
    print(f"  {'-'*48}")
    for r in results:
        print(f"  {r['j']:6.2f}  {r['kt']:10.0f}  {10*r['kq']:10.0f}  "
              f"{r['eta_o']:8.4f}  {r['n_markers']:8d}")

    metadata = {
        "name": "ibm_propeller",
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
    "IBMPropellerConfig",
    "_generate_propeller_markers",
    "run_ibm_propeller_benchmark",
]
