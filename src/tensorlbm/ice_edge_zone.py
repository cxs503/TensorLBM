"""Ice-edge zone flow simulation using D3Q19 LBM.

Simulates the transition from ice-covered water to open water channel,
capturing the flow structure and drag mechanism at the ice edge.

Geometry (3D rectangular domain):
    - x: streamwise (flow direction), inlet → outlet
    - y: wall-normal (bottom = sea bed, top = ice/water surface)
    - z: spanwise (periodic)

Boundary conditions:
    - Bottom (y=0): no-slip (sea bed)
    - Top, x < x_edge: partial-slip (ice cover bottom, slip length Ls_ice)
    - Top, x >= x_edge: free-slip (open water surface)
    - Inlet (x=0): uniform velocity U_0
    - Outlet (x=nx-1): zero-gradient (convective)
    - Spanwise (z): periodic

Physical parameters:
    - Re = U_0 * H / nu  (H = channel height)
    - Slip length ratio: Ls_ice / H
    - Ice coverage ratio: L_ice / Lx

References:
    - Succi, S. (2001). The Lattice Boltzmann Equation.
    - Navier slip BC: Zhang et al. (2012).
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tensorlbm.d3q19 import C, W, OPPOSITE, W_EXACT64
from tensorlbm.boundary.navier_slip_bc import slip_length_to_beta

__all__ = [
    "IceEdgeZoneConfig",
    "IceEdgeZoneResult",
    "run_ice_edge_zone",
]


# ---------------------------------------------------------------------------
# D3Q19 mirror tables for free-slip (specular reflection)
# ---------------------------------------------------------------------------
def _build_mirror_y() -> list[int]:
    """Mirror table for y-axis specular reflection."""
    mirror = list(range(19))
    for i in range(19):
        cx, cy, cz = C[i].tolist()
        for j in range(19):
            if C[j].tolist() == [cx, -cy, cz]:
                mirror[i] = j
                break
    return mirror

def _build_mirror_z() -> list[int]:
    """Mirror table for z-axis specular reflection."""
    mirror = list(range(19))
    for i in range(19):
        cx, cy, cz = C[i].tolist()
        for j in range(19):
            if C[j].tolist() == [cx, cy, -cz]:
                mirror[i] = j
                break
    return mirror

MIRROR_Y = _build_mirror_y()
MIRROR_Z = _build_mirror_z()


@dataclass(frozen=True)
class IceEdgeZoneConfig:
    """Configuration for ice-edge zone LBM simulation."""
    # Grid dimensions
    nx: int = 200       # streamwise
    ny: int = 40        # wall-normal (channel height)
    nz: int = 20        # spanwise (periodic)

    # Physical parameters
    u0: float = 0.05        # inlet velocity (lattice units, Ma < 0.1)
    re: float = 200.0       # Reynolds number Re = U0 * H / nu
    ice_edge_frac: float = 0.5  # ice edge position as fraction of Lx
    ls_ice: float = 2.0     # slip length under ice (lattice units)

    # Simulation control
    n_steps: int = 30000
    output_interval: int = 5000
    stats_start: int = 15000    # start time-averaging after this step

    # Output
    output_root: Path = Path("runs/ice_edge_zone")
    run_name: str | None = None
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def H(self) -> int:
        """Channel height in lattice units (fluid region)."""
        return self.ny - 2  # exclude wall cells

    @property
    def nu(self) -> float:
        """Kinematic viscosity from Re = U0 * H / nu."""
        return self.u0 * self.H / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return 3.0 * self.nu + 0.5

    @property
    def omega(self) -> float:
        """Collision frequency."""
        return 1.0 / self.tau

    @property
    def x_edge(self) -> int:
        """Ice edge x-position in lattice units."""
        return max(1, int(self.nx * self.ice_edge_frac))

    @property
    def beta_ice(self) -> float:
        """Slip interpolation parameter for ice cover."""
        return slip_length_to_beta(self.ls_ice)

    def validate(self) -> None:
        assert self.nx >= 20, f"nx={self.nx} too small"
        assert self.ny >= 10, f"ny={self.ny} too small"
        assert self.nz >= 4, f"nz={self.nz} too small"
        assert 0.0 < self.u0 < 0.2, f"u0={self.u0} out of range (Ma too high)"
        assert self.re > 0, f"re={self.re} must be positive"
        assert 0.0 < self.ice_edge_frac < 1.0
        assert self.ls_ice >= 0.0
        assert self.tau > 0.5, f"tau={self.tau} must be > 0.5"

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return (f"ice_edge_nx{self.nx}_ny{self.ny}_nz{self.nz}"
                f"_re{self.re:.0f}_ls{self.ls_ice:.1f}_ef{self.ice_edge_frac:.2f}")


@dataclass
class IceEdgeZoneResult:
    """Results from ice-edge zone simulation."""
    config: dict
    # Time-averaged fields (3D)
    u_mean: np.ndarray      # (nz, ny, nx)
    v_mean: np.ndarray
    w_mean: np.ndarray
    p_mean: np.ndarray
    # Instantaneous final fields
    u_final: np.ndarray
    v_final: np.ndarray
    w_final: np.ndarray
    # Derived quantities
    vorticity_y_mean: np.ndarray    # spanwise vorticity (omega_z actually)
    shear_stress_bottom: np.ndarray  # (nz, nx) bottom wall
    shear_stress_ice: np.ndarray     # (nz, nx) ice cover region
    drag_coefficient: float
    drag_ice_component: float
    drag_open_component: float
    # Metadata
    n_steps_run: int
    sha256: str
    output_dir: str


def _equilibrium_d3q19(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> torch.Tensor:
    """Compute D3Q19 equilibrium distribution.
    
    f has shape (19, nz, ny, nx).
    rho, ux, uy, uz have shape (nz, ny, nx).
    """
    device = rho.device
    c = C.to(device).float()  # (19, 3)
    w = W.to(device).float()  # (19,)

    # u·u
    u_sq = ux**2 + uy**2 + uz**2  # (nz, ny, nx)

    # c·u for each direction: (19, nz, ny, nx)
    cu = (c[:, 0].view(19,1,1,1) * ux.unsqueeze(0) +
          c[:, 1].view(19,1,1,1) * uy.unsqueeze(0) +
          c[:, 2].view(19,1,1,1) * uz.unsqueeze(0))

    feq = w.view(19,1,1,1) * rho.unsqueeze(0) * (
        1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq.unsqueeze(0)
    )
    return feq


def _stream_d3q19(f: torch.Tensor) -> torch.Tensor:
    """Streaming step for D3Q19.
    
    f shape: (19, nz, ny, nx)
    Dimensions: 0=pop, 1=z(spanwise,periodic), 2=y(wall-normal,bounded), 3=x(streamwise,bounded)
    """
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_new = torch.empty_like(f)

    for q in range(19):
        cx = int(C[q, 0].item())
        cy = int(C[q, 1].item())
        cz = int(C[q, 2].item())

        if cx == 0 and cy == 0 and cz == 0:
            f_new[q] = f[q]
            continue

        src = f[q]  # (nz, ny, nx)

        # Shift in z (dim=0 of src, periodic via roll)
        if cz != 0:
            src = torch.roll(src, shifts=cz, dims=0)

        # Shift in y (dim=1 of src, bounded)
        if cy > 0:
            tmp = torch.zeros_like(src)
            tmp[:, cy:, :] = src[:, :ny-cy, :]
            src = tmp
        elif cy < 0:
            tmp = torch.zeros_like(src)
            tmp[:, :ny+cy, :] = src[:, -cy:, :]
            src = tmp

        # Shift in x (dim=2 of src, bounded)
        if cx > 0:
            tmp = torch.zeros_like(src)
            tmp[:, :, cx:] = src[:, :, :nx-cx]
            src = tmp
        elif cx < 0:
            tmp = torch.zeros_like(src)
            tmp[:, :, :nx+cx] = src[:, :, -cx:]
            src = tmp

        f_new[q] = src

    return f_new


def _apply_inlet_bc(f: torch.Tensor, u0: float, device: torch.device) -> torch.Tensor:
    """Zou-He velocity inlet at x=0."""
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    
    rho_in = 1.0
    ux_in = u0
    uy_in = 0.0
    uz_in = 0.0

    feq_in = _equilibrium_d3q19(
        torch.full((nz, ny), rho_in, device=device),
        torch.full((nz, ny), ux_in, device=device),
        torch.full((nz, ny), uy_in, device=device),
        torch.full((nz, ny), uz_in, device=device),
    )

    # Inlet directions: cx > 0 (directions 1, 7, 9, 11, 13)
    inlet_dirs = [1, 7, 9, 11, 13]
    for q in inlet_dirs:
        oq = int(OPPOSITE[q].item())
        f[q, :, :, 0] = feq_in[q] + f[oq, :, :, 0] - feq_in[oq]

    return f


def _apply_outlet_bc(f: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Zero-gradient outlet at x=nx-1 (copy from interior)."""
    f[:, :, :, -1] = f[:, :, :, -2]
    return f


def _apply_bottom_wall(f: torch.Tensor, device: torch.device) -> torch.Tensor:
    """No-slip bottom wall (y=0) via bounce-back."""
    for q in range(19):
        oq = int(OPPOSITE[q].item())
        if q != oq:
            f[q, :, 0, :] = f[oq, :, 0, :]
    return f


def _apply_top_wall_ice(f: torch.Tensor, x_edge: int, beta_ice: float, device: torch.device) -> torch.Tensor:
    """Top wall (y=ny-1): partial-slip under ice (x < x_edge), free-slip for open water."""
    ny = f.shape[2]

    # Ice region: partial slip (x < x_edge)
    if beta_ice > 0 and x_edge > 0:
        for q in range(19):
            oq = int(OPPOSITE[q].item())
            mq = MIRROR_Y[q]  # specular reflection direction
            # Partial slip: f_q = (1-beta)*f_opp + beta*f_mirror
            f[q, :, ny-1, :x_edge] = (
                (1.0 - beta_ice) * f[oq, :, ny-1, :x_edge] +
                beta_ice * f[mq, :, ny-1, :x_edge]
            )

    # Open water region: free-slip (x >= x_edge)
    for q in range(19):
        mq = MIRROR_Y[q]
        if q != mq:
            f[q, :, ny-1, x_edge:] = f[mq, :, ny-1, x_edge:]

    return f


def _apply_side_walls(f: torch.Tensor, device: torch.device) -> torch.Tensor:
    """No-slip side walls (z=0, z=nz-1) via bounce-back."""
    nz = f.shape[1]
    for q in range(19):
        oq = int(OPPOSITE[q].item())
        cz = int(C[q, 2].item())
        if cz != 0:  # only directions with z-component
            # z=0 wall
            f[q, 0, :, :] = f[oq, 0, :, :]
            # z=nz-1 wall
            f[q, nz-1, :, :] = f[oq, nz-1, :, :]
    return f


def _compute_macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute macroscopic variables from distribution functions."""
    device = f.device
    c = C.to(device).float()

    rho = f.sum(dim=0)  # (nz, ny, nx)

    # Momentum
    jx = (f * c[:, 0].view(19,1,1,1)).sum(dim=0)
    jy = (f * c[:, 1].view(19,1,1,1)).sum(dim=0)
    jz = (f * c[:, 2].view(19,1,1,1)).sum(dim=0)

    ux = jx / rho
    uy = jy / rho
    uz = jz / rho

    return rho, ux, uy, uz


def _compute_pressure(rho: torch.Tensor, cs2: float = 1.0/3.0) -> torch.Tensor:
    """Compute pressure from density: p = cs2 * rho (lattice units, p0 = cs2)."""
    return cs2 * rho


def _compute_vorticity(ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute vorticity using central differences.
    
    ux, uy, uz shape: (nz, ny, nx)
    """
    # omega_x = duz/dy - duy/dz
    # omega_y = dux/dz - duz/dx
    # omega_z = duy/dx - dux/dy

    # dim 0 = z, dim 1 = y, dim 2 = x
    dux_dy = (torch.roll(ux, -1, dims=1) - torch.roll(ux, 1, dims=1)) / 2.0
    dux_dz = (torch.roll(ux, -1, dims=0) - torch.roll(ux, 1, dims=0)) / 2.0
    duy_dx = (torch.roll(uy, -1, dims=2) - torch.roll(uy, 1, dims=2)) / 2.0
    duy_dz = (torch.roll(uy, -1, dims=0) - torch.roll(uy, 1, dims=0)) / 2.0
    duz_dx = (torch.roll(uz, -1, dims=2) - torch.roll(uz, 1, dims=2)) / 2.0
    duz_dy = (torch.roll(uz, -1, dims=1) - torch.roll(uz, 1, dims=1)) / 2.0

    omega_x = duz_dy - duy_dz
    omega_y = dux_dz - duz_dx
    omega_z = duy_dx - dux_dy

    return omega_x, omega_y, omega_z


def _compute_wall_shear(ux: torch.Tensor, uy: torch.Tensor, nu: float,
                        wall: str = "bottom") -> torch.Tensor:
    """Compute wall shear stress tau_w = mu * du/dy at wall.

    Returns (nz, nx) array of streamwise shear stress.
    ux shape: (nz, ny, nx)
    """
    if wall == "bottom":
        # du/dy at y=0 using forward difference
        du_dy = (ux[:, 1, :] - ux[:, 0, :])
    else:  # top
        du_dy = (ux[:, -1, :] - ux[:, -2, :])

    return nu * du_dy  # mu = nu * rho, rho ≈ 1


def run_ice_edge_zone(config: IceEdgeZoneConfig) -> IceEdgeZoneResult:
    """Run the ice-edge zone LBM simulation."""
    config.validate()
    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    nx, ny, nz = config.nx, config.ny, config.nz
    omega = config.omega
    nu = config.nu
    u0 = config.u0
    x_edge = config.x_edge
    beta_ice = config.beta_ice

    # Prepare output directory
    run_name = config.resolved_run_name()
    run_dir = config.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}
    with open(run_dir / "config.json", "w") as fp:
        json.dump(config_dict, fp, indent=2)

    print(f"[IceEdge] Run: {run_name}")
    print(f"[IceEdge] Grid: {nx}×{ny}×{nz}, Re={config.re}, tau={config.tau:.4f}")
    print(f"[IceEdge] Ice edge at x={x_edge}, beta_ice={beta_ice:.4f}")
    print(f"[IceEdge] nu={nu:.6f}, u0={u0}")

    # Initialize distribution functions to equilibrium with uniform flow
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u0, device=device)
    uy0 = torch.zeros(nz, ny, nx, device=device)
    uz0 = torch.zeros(nz, ny, nx, device=device)

    f = _equilibrium_d3q19(rho0, ux0, uy0, uz0)

    # Time-averaging accumulators
    u_sum = torch.zeros(nz, ny, nx, device=device)
    v_sum = torch.zeros(nz, ny, nx, device=device)
    w_sum = torch.zeros(nz, ny, nx, device=device)
    p_sum = torch.zeros(nz, ny, nx, device=device)
    u_sq_sum = torch.zeros(nz, ny, nx, device=device)
    n_avg = 0

    # Main time loop
    for step in range(1, config.n_steps + 1):
        # Collision (BGK)
        rho, ux, uy, uz = _compute_macroscopic(f)
        feq = _equilibrium_d3q19(rho, ux, uy, uz)
        f = f - omega * (f - feq)

        # Streaming
        f = _stream_d3q19(f)

        # Boundary conditions
        f = _apply_inlet_bc(f, u0, device)
        f = _apply_outlet_bc(f, device)
        f = _apply_bottom_wall(f, device)
        f = _apply_top_wall_ice(f, x_edge, beta_ice, device)
        f = _apply_side_walls(f, device)

        # Time averaging
        if step >= config.stats_start and step % 2 == 0:
            rho, ux, uy, uz = _compute_macroscopic(f)
            u_sum += ux
            v_sum += uy
            w_sum += uz
            p_sum += _compute_pressure(rho)
            u_sq_sum += ux**2
            n_avg += 1

        # Diagnostics
        if step % config.output_interval == 0 or step == 1:
            rho, ux, uy, uz = _compute_macroscopic(f)
            u_max = ux.max().item()
            u_mean_val = ux.mean().item()
            rho_mean = rho.mean().item()
            print(f"  Step {step:6d}: u_max={u_max:.5f}, u_mean={u_mean_val:.5f}, rho_mean={rho_mean:.6f}")

    # Final fields
    rho_final, ux_final, uy_final, uz_final = _compute_macroscopic(f)
    p_final = _compute_pressure(rho_final)

    # Time-averaged fields
    if n_avg > 0:
        u_mean = u_sum / n_avg
        v_mean = v_sum / n_avg
        w_mean = w_sum / n_avg
        p_mean = p_sum / n_avg
        u_rms = torch.sqrt(torch.clamp(u_sq_sum / n_avg - u_mean**2, min=0))
    else:
        u_mean = ux_final
        v_mean = uy_final
        w_mean = uz_final
        p_mean = p_final
        u_rms = torch.zeros_like(ux_final)

    # Vorticity (from mean field)
    _, _, omega_z_mean = _compute_vorticity(u_mean, v_mean, w_mean)

    # Wall shear stress
    tau_bottom = _compute_wall_shear(u_mean, v_mean, nu, wall="bottom")
    tau_top_ice = _compute_wall_shear(u_mean, v_mean, nu, wall="top")

    # Drag coefficient calculation
    # C_D = F_D / (0.5 * rho * U0^2 * A_ref)
    # F_D = integral of tau_w over wetted surface
    # For the ice edge problem, A_ref = H * nz (projected area of ice edge)
    A_ref = config.H * nz

    # Total drag from bottom wall
    F_bottom = tau_bottom.abs().sum().item()

    # Drag from ice cover (top wall, ice region)
    F_ice = tau_top_ice[:, :x_edge].abs().sum().item()

    # Drag from open water surface (should be ~0 for free-slip)
    F_open = tau_top_ice[:, x_edge:].abs().sum().item()

    F_total = F_bottom + F_ice + F_open
    C_D = F_total / (0.5 * 1.0 * u0**2 * A_ref)
    C_D_ice = F_ice / (0.5 * 1.0 * u0**2 * A_ref)
    C_D_open = F_open / (0.5 * 1.0 * u0**2 * A_ref)

    print(f"\n[Results] C_D = {C_D:.6f}")
    print(f"[Results] C_D_ice = {C_D_ice:.6f}, C_D_open = {C_D_open:.6f}")
    print(f"[Results] n_avg = {n_avg}")

    # Convert to numpy
    result = IceEdgeZoneResult(
        config=config_dict,
        u_mean=u_mean.detach().cpu().numpy(),
        v_mean=v_mean.detach().cpu().numpy(),
        w_mean=w_mean.detach().cpu().numpy(),
        p_mean=p_mean.detach().cpu().numpy(),
        u_final=ux_final.detach().cpu().numpy(),
        v_final=uy_final.detach().cpu().numpy(),
        w_final=uz_final.detach().cpu().numpy(),
        vorticity_y_mean=omega_z_mean.detach().cpu().numpy(),
        shear_stress_bottom=tau_bottom.detach().cpu().numpy(),
        shear_stress_ice=tau_top_ice.detach().cpu().numpy(),
        drag_coefficient=C_D,
        drag_ice_component=C_D_ice,
        drag_open_component=C_D_open,
        n_steps_run=config.n_steps,
        sha256="",
        output_dir=str(run_dir),
    )

    # Compute sha256 of results for provenance
    h = hashlib.sha256()
    h.update(result.u_mean.tobytes())
    h.update(result.p_mean.tobytes())
    h.update(result.vorticity_y_mean.tobytes())
    result.sha256 = h.hexdigest()

    # Save results
    _save_results(result, run_dir)

    return result


def _save_results(result: IceEdgeZoneResult, run_dir: Path) -> None:
    """Save simulation results to disk."""
    # Save key fields as numpy arrays
    np.savez_compressed(
        run_dir / "fields.npz",
        u_mean=result.u_mean,
        v_mean=result.v_mean,
        w_mean=result.w_mean,
        p_mean=result.p_mean,
        vorticity_z=result.vorticity_y_mean,
        tau_bottom=result.shear_stress_bottom,
        tau_top=result.shear_stress_ice,
    )

    # Save summary JSON
    summary = {
        "drag_coefficient": result.drag_coefficient,
        "drag_ice_component": result.drag_ice_component,
        "drag_open_component": result.drag_open_component,
        "n_steps": result.n_steps_run,
        "sha256": result.sha256,
        "config": result.config,
    }
    with open(run_dir / "results.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[IceEdge] Results saved to {run_dir}")
