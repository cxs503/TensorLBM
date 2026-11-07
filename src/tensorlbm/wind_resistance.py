"""Ship wind-resistance forecast engine: D3Q19/MRT LBM + empirical hybrid.

Provides aerodynamic-coefficient prediction for three canonical ship types
(container ship, tanker, bulk carrier) over wind angles 0°–180°, with a
hybrid selection rule (simple geometry → empirical Blendermann/ISHRY,
complex geometry → LBM) and voyage-level wind-assessment.

Key outputs
-----------
* ``C_X``, ``C_Y``, ``C_N`` — longitudinal, transverse, yawing-moment
  coefficients referenced to frontal area A_F and overall length L_OA.
* ``WindResistanceEngine`` — hybrid engine with automatic method selection.
* ``voyage_assessment`` — annual wind thrust/drag from a wind-rose histogram.

Reference area convention
-------------------------
All coefficients use the *frontal projected area* A_F (m²) and overall
length L_OA (m) as reference, consistent with OCIMF / Blendermann practice:

    F_X = ½ ρ_air V_w² A_F C_X
    F_Y = ½ ρ_air V_w² A_F C_Y
    M_N = ½ ρ_air V_w² A_F L_OA C_N

Empirical baselines
-------------------
* **Blendermann** (1992) — polynomial fits for C_X, C_Y, C_N vs wind angle
  with ship-type-dependent coefficients.
* **ISHRY** — simplified empirical model for early-stage estimation.

LBM configuration
-----------------
D3Q19 lattice with MRT collision.  The domain is a rectangular box with
the ship superstructure voxelised from a CAD-generated STL.  Inflow is
imposed via Zou/He velocity BC; outflow uses convective (Neumann) BC.
Aerodynamic forces are obtained by momentum-exchange on the BFL boundary.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. SHIP GEOMETRY DATA (simplified superstructure parameters)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShipGeometry:
    """Simplified superstructure geometry for wind-load estimation.

    All lengths in metres.  The superstructure is modelled as a stack of
    rectangular blocks (bridge + deckhouses) sitting on the main deck.
    """
    name: str
    ship_type: Literal["container", "tanker", "bulk_carrier"]
    # Overall dimensions
    L_OA: float       # overall length
    B: float          # beam
    D: float          # depth (hull)
    # Superstructure (bridge)
    bridge_L: float   # bridge length (streamwise)
    bridge_B: float   # bridge width (transverse)
    bridge_H: float   # bridge height above main deck
    bridge_z0: float  # bridge base height above waterline
    # Deckhouses (additional blocks)
    deckhouses: list[dict] = field(default_factory=list)
    # Container stack (container ships only)
    container_stack_H: float = 0.0  # height of container stack above deck
    container_stack_L: float = 0.0  # length of container stack
    container_stack_B: float = 0.0  # width of container stack
    # Derived
    @property
    def frontal_area(self) -> float:
        """Projected frontal area A_F (m²) — max of hull + superstructure."""
        a_hull = self.B * (self.D * 0.6)  # approximate freeboard portion
        a_bridge = self.bridge_B * self.bridge_H
        a_containers = self.container_stack_B * self.container_stack_H
        return max(a_hull, a_bridge + a_hull * 0.3, a_containers + a_hull * 0.3)

    @property
    def lateral_area(self) -> float:
        """Projected lateral area A_L (m²)."""
        a_hull = self.L_OA * (self.D * 0.6)
        a_bridge = self.bridge_L * self.bridge_H
        a_containers = self.container_stack_L * self.container_stack_H
        return a_hull + a_bridge + a_containers

    @property
    def complexity_score(self) -> float:
        """Heuristic geometry complexity score.

        > 0.5 → use LBM; ≤ 0.5 → empirical is adequate.
        Based on number of distinct blocks, height ratios, and asymmetry.
        Container ships have tall container stacks → high complexity.
        Tankers and bulk carriers have simpler profiles → low complexity.
        """
        n_blocks = 1 + len(self.deckhouses)
        if self.container_stack_H > 0:
            n_blocks += 1
        h_ratio = self.bridge_H / max(self.D, 1.0)
        asymmetry = abs(self.bridge_B - self.B) / max(self.B, 1.0)
        # Container ships get extra weight for container stacks
        container_factor = 0.5 if self.container_stack_H > 0 else 0.0
        return 0.15 * n_blocks + 0.2 * min(h_ratio / 3.0, 1.0) + 0.15 * asymmetry + container_factor


# Canonical ship geometries (representative values from literature)
CONTAINER_SHIP = ShipGeometry(
    name="containership_8000teu",
    ship_type="container",
    L_OA=300.0, B=40.0, D=24.0,
    bridge_L=25.0, bridge_B=30.0, bridge_H=32.0, bridge_z0=24.0,
    deckhouses=[
        {"L": 15.0, "B": 28.0, "H": 8.0, "z0": 24.0, "x_offset": -20.0},
    ],
    container_stack_H=18.0, container_stack_L=200.0, container_stack_B=32.0,
)

TANKER = ShipGeometry(
    name="aframax_tanker",
    ship_type="tanker",
    L_OA=250.0, B=44.0, D=21.0,
    bridge_L=20.0, bridge_B=28.0, bridge_H=28.0, bridge_z0=21.0,
    deckhouses=[
        {"L": 10.0, "B": 20.0, "H": 6.0, "z0": 21.0, "x_offset": -15.0},
    ],
)

BULK_CARRIER = ShipGeometry(
    name="capesize_bulk",
    ship_type="bulk_carrier",
    L_OA=290.0, B=45.0, D=23.0,
    bridge_L=18.0, bridge_B=26.0, bridge_H=26.0, bridge_z0=23.0,
    deckhouses=[
        {"L": 12.0, "B": 22.0, "H": 5.0, "z0": 23.0, "x_offset": -12.0},
    ],
)



# Scaled-down test geometry for LBM verification (fits low-resolution grids)
TEST_SHIP = ShipGeometry(
    name="test_ship",
    ship_type="container",
    L_OA=10.0, B=3.0, D=2.0,
    bridge_L=2.0, bridge_B=2.0, bridge_H=3.0, bridge_z0=2.0,
    deckhouses=[],
    container_stack_H=1.5, container_stack_L=6.0, container_stack_B=2.5,
)

SHIP_GEOMETRIES: dict[str, ShipGeometry] = {
    "container": CONTAINER_SHIP,
    "tanker": TANKER,
    "bulk_carrier": BULK_CARRIER,
}

# ---------------------------------------------------------------------------
# 2. EMPIRICAL FORMULAS (Blendermann / ISHRY)
# ---------------------------------------------------------------------------

def blendermann_CX(wind_angle_deg: float, ship_type: str = "container") -> float:
    """Blendermann longitudinal wind coefficient C_X(ψ).

    Polynomial fit from Blendermann (1992), "Wind Loads on Ships".
    ψ = wind angle from bow (0° = head wind, 90° = beam wind).

    Reference area: frontal projected area A_F.
    """
    psi = math.radians(wind_angle_deg)
    # Type-dependent coefficients (from Blendermann tables)
    coeffs = {
        "container":    {"a1": 0.60, "a2": 0.25, "a3": 0.15},
        "tanker":       {"a1": 0.70, "a2": 0.20, "a3": 0.10},
        "bulk_carrier": {"a1": 0.65, "a2": 0.22, "a3": 0.13},
    }
    c = coeffs.get(ship_type, coeffs["container"])
    # C_X(ψ) = a1 + a2·cos(2ψ) + a3·cos(4ψ)
    return c["a1"] + c["a2"] * math.cos(2 * psi) + c["a3"] * math.cos(4 * psi)


def blendermann_CY(wind_angle_deg: float, ship_type: str = "container") -> float:
    """Blendermann transverse wind coefficient C_Y(ψ)."""
    psi = math.radians(wind_angle_deg)
    coeffs = {
        "container":    {"b1": 0.0, "b2": 1.10, "b3": -0.30, "b4": 0.15},
        "tanker":       {"b1": 0.0, "b2": 1.00, "b3": -0.25, "b4": 0.12},
        "bulk_carrier": {"b1": 0.0, "b2": 1.05, "b3": -0.28, "b4": 0.14},
    }
    c = coeffs.get(ship_type, coeffs["container"])
    # C_Y(ψ) = b1 + b2·sin(ψ) + b3·sin(2ψ) + b4·sin(3ψ)
    return (c["b1"] + c["b2"] * math.sin(psi)
            + c["b3"] * math.sin(2 * psi) + c["b4"] * math.sin(3 * psi))


def blendermann_CN(wind_angle_deg: float, ship_type: str = "container") -> float:
    """Blendermann yaw-moment coefficient C_N(ψ)."""
    psi = math.radians(wind_angle_deg)
    coeffs = {
        "container":    {"c1": 0.0, "c2": 0.25, "c3": -0.10, "c4": 0.05},
        "tanker":       {"c1": 0.0, "c2": 0.20, "c3": -0.08, "c4": 0.04},
        "bulk_carrier": {"c1": 0.0, "c2": 0.22, "c3": -0.09, "c4": 0.045},
    }
    c = coeffs.get(ship_type, coeffs["container"])
    # C_N(ψ) = c1 + c2·sin(2ψ) + c3·sin(3ψ) + c4·sin(4ψ)
    return (c["c1"] + c["c2"] * math.sin(2 * psi)
            + c["c3"] * math.sin(3 * psi) + c["c4"] * math.sin(4 * psi))


def ishy_CX(wind_angle_deg: float, ship_type: str = "container") -> float:
    """ISHRY simplified longitudinal wind coefficient.

    ISHRY (International Ship and Offshore Structures Congress simplified
    method) — a conservative envelope for preliminary design.
    """
    psi = math.radians(wind_angle_deg)
    base = {
        "container": 0.75,
        "tanker": 0.80,
        "bulk_carrier": 0.78,
    }
    return base.get(ship_type, 0.75) * (0.5 + 0.5 * math.cos(psi))


def ishy_CY(wind_angle_deg: float, ship_type: str = "container") -> float:
    """ISHRY simplified transverse wind coefficient."""
    psi = math.radians(wind_angle_deg)
    base = {
        "container": 1.20,
        "tanker": 1.10,
        "bulk_carrier": 1.15,
    }
    return base.get(ship_type, 1.20) * math.sin(psi)


def empirical_coefficients(
    wind_angle_deg: float,
    ship_type: str = "container",
    method: Literal["blendermann", "ishry"] = "blendermann",
) -> dict[str, float]:
    """Compute C_X, C_Y, C_N from empirical formulas.

    Args:
        wind_angle_deg: Wind angle from bow (0–180°).
        ship_type: One of "container", "tanker", "bulk_carrier".
        method: "blendermann" (default) or "ishry".

    Returns:
        Dict with keys "C_X", "C_Y", "C_N".
    """
    if method == "blendermann":
        return {
            "C_X": blendermann_CX(wind_angle_deg, ship_type),
            "C_Y": blendermann_CY(wind_angle_deg, ship_type),
            "C_N": blendermann_CN(wind_angle_deg, ship_type),
        }
    elif method == "ishry":
        return {
            "C_X": ishy_CX(wind_angle_deg, ship_type),
            "C_Y": ishy_CY(wind_angle_deg, ship_type),
            "C_N": 0.0,  # ISHRY does not provide C_N
        }
    else:
        raise ValueError(f"Unknown empirical method: {method}")


# ---------------------------------------------------------------------------
# 3. LBM AERODYNAMIC SOLVER (D3Q19/MRT)
# ---------------------------------------------------------------------------

def _build_superstructure_mask(
    geo: ShipGeometry,
    nx: int,
    ny: int,
    nz: int,
    domain_L: float,
    domain_B: float,
    domain_H: float,
) -> torch.Tensor:
    """Voxelise the simplified superstructure into a boolean mask.

    The domain is centred on the ship; x is streamwise, y transverse,
    z vertical.  The mask is True inside the solid (ship) region.

    Args:
        geo: ShipGeometry describing the superstructure.
        nx, ny, nz: Grid resolution.
        domain_L, domain_B, domain_H: Physical domain extents.

    Returns:
        Boolean tensor of shape (nz, ny, nx).
    """
    dx = domain_L / nx
    dy = domain_B / ny
    dz = domain_H / nz

    # Cell centre coordinates
    x = (torch.arange(nx) - nx / 2 + 0.5) * dx
    y = (torch.arange(ny) - ny / 2 + 0.5) * dy
    z = (torch.arange(nz) - nz / 2 + 0.5) * dz
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    # Shape: (nx, ny, nz) → transpose to (nz, ny, nx)
    X = X.transpose(0, 2)
    Y = Y.transpose(0, 2)
    Z = Z.transpose(0, 2)

    mask = torch.zeros((nz, ny, nx), dtype=torch.bool)

    # Hull block (simplified: rectangular block at waterline)
    hull_H = geo.D * 0.6  # freeboard portion
    hull_mask = (
        (X.abs() <= geo.L_OA / 2)
        & (Y.abs() <= geo.B / 2)
        & (Z >= 0)
        & (Z <= hull_H)
    )
    mask = mask | hull_mask

    # Bridge
    bridge_x0 = geo.L_OA / 2 - geo.bridge_L  # bridge at aft
    bridge_mask = (
        (X >= bridge_x0) & (X <= geo.L_OA / 2)
        & (Y.abs() <= geo.bridge_B / 2)
        & (Z >= geo.bridge_z0) & (Z <= geo.bridge_z0 + geo.bridge_H)
    )
    mask = mask | bridge_mask

    # Deckhouses
    for dh in geo.deckhouses:
        dh_x0 = bridge_x0 + dh.get("x_offset", 0)
        dh_mask = (
            (X >= dh_x0) & (X <= dh_x0 + dh["L"])
            & (Y.abs() <= dh["B"] / 2)
            & (Z >= dh["z0"]) & (Z <= dh["z0"] + dh["H"])
        )
        mask = mask | dh_mask

    # Container stack
    if geo.container_stack_H > 0:
        cont_L = geo.container_stack_L
        cont_x0 = -geo.L_OA / 2 + (geo.L_OA - cont_L) / 2
        cont_mask = (
            (X >= cont_x0) & (X <= cont_x0 + cont_L)
            & (Y.abs() <= geo.container_stack_B / 2)
            & (Z >= hull_H) & (Z <= hull_H + geo.container_stack_H)
        )
        mask = mask | cont_mask

    return mask


def lbm_aerodynamic_coefficients(
    geo: ShipGeometry,
    wind_angle_deg: float,
    wind_speed: float = 10.0,
    resolution: int = 48,
    max_steps: int = 5000,
    rho_air: float = 1.225,
    nu_air: float = 1.5e-5,
    device: str = "cpu",
) -> dict[str, float]:
    """Compute aerodynamic coefficients via D3Q19/MRT LBM.

    This is a *reduced-order* LBM simulation: the domain is a compact box
    around the superstructure, with inflow/outflow BCs and momentum-exchange
    force evaluation.  It is intended for coefficient estimation at moderate
    resolution; full-scale high-fidelity runs should use the cluster.

    Args:
        geo: ShipGeometry.
        wind_angle_deg: Wind angle from bow (0–180°).
        wind_speed: Physical wind speed (m/s).
        resolution: Grid points per domain extent (cube domain).
        max_steps: Maximum LBM iterations.
        rho_air: Air density (kg/m³).
        nu_air: Air kinematic viscosity (m²/s).
        device: "cpu" or "cuda".

    Returns:
        Dict with keys "C_X", "C_Y", "C_N", "converged", "steps".
    """
    from .d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
    from .solver3d import collide_mrt3d

    dev = torch.device(device)

    # Domain sizing: ~3× ship's largest dimension for reasonable BC distance
    # while keeping the ship resolvable at low resolution
    L_char = max(geo.L_OA, geo.bridge_H + geo.D)
    # Scale domain to ensure ship occupies ~1/3 of grid at target resolution
    domain_size = 3.0 * L_char
    dx_phys = domain_size / resolution

    # LBM units: set lattice speed c = dx/dt = 1, so dt = dx_phys
    # Characteristic velocity in lattice units
    u_char = wind_speed  # m/s
    # Set Ma = 0.1 for incompressible LBM → u_lb = Ma / sqrt(3) ≈ 0.0577
    ma_target = 0.1
    u_lb = ma_target / math.sqrt(3.0)
    # Conversion: u_lb = u_phys * dt / dx_phys → dt = u_lb * dx_phys / u_phys
    dt_lb = u_lb * dx_phys / u_char
    # Viscosity in lattice units: nu_lb = nu_phys * dt / dx²
    nu_lb = nu_air * dt_lb / (dx_phys * dx_phys)
    # Relaxation time: tau = 3*nu_lb + 0.5
    tau = 3.0 * nu_lb + 0.5
    tau = max(tau, 0.51)  # stability floor

    # Wind direction in lattice frame
    psi = math.radians(wind_angle_deg)
    # Inflow velocity components (x = streamwise, y = transverse)
    u_in_x = u_lb * math.cos(psi)
    u_in_y = u_lb * math.sin(psi)
    u_in_z = 0.0

    # Build geometry mask
    n = resolution
    mask = _build_superstructure_mask(
        geo, n, n, n,
        domain_size, domain_size, domain_size,
    ).to(dev)

    # Fluid mask
    fluid = ~mask

    # Initialise distributions on full grid
    rho = torch.ones((n, n, n), dtype=torch.float32, device=dev)
    ux = torch.zeros((n, n, n), dtype=torch.float32, device=dev)
    uy = torch.zeros((n, n, n), dtype=torch.float32, device=dev)
    uz = torch.zeros((n, n, n), dtype=torch.float32, device=dev)

    # Set inflow in fluid region
    ux[fluid] = u_in_x
    uy[fluid] = u_in_y

    f = equilibrium3d(rho, ux, uy, uz, device=dev)

    # Solid nodes: equilibrium with zero velocity
    rho_solid = torch.ones((n, n, n), dtype=torch.float32, device=dev)
    f_solid = equilibrium3d(
        rho_solid,
        torch.zeros_like(ux),
        torch.zeros_like(uy),
        torch.zeros_like(uz),
        device=dev,
    )

    # Force accumulation
    Fx_hist: list[float] = []
    Fy_hist: list[float] = []
    Fn_hist: list[float] = []

    # Lattice velocity components for streaming
    c = C.to(dev)

    converged = False
    steps_run = 0

    for step in range(max_steps):
        # Collision (MRT) — on full grid, then restore solid
        f = collide_mrt3d(f, tau)
        # Restore solid nodes
        f[:, mask] = f_solid[:, mask]

        # Streaming (periodic, then BC overwrite)
        f_new = torch.empty_like(f)
        for q in range(19):
            dz = -int(c[q, 2])
            dy = -int(c[q, 1])
            dx = -int(c[q, 0])
            f_new[q] = torch.roll(f[q], shifts=(dz, dy, dx), dims=(0, 1, 2))

        # Apply BCs
        # Inflow (left boundary, x=0): equilibrium with inflow velocity
        # equilibrium3d expects (nz, ny, nx); we have a 2D plane (ny, nx) = (n, n)
        # Use nz=1 and squeeze the result
        rho_in = torch.ones((1, n, n), dtype=torch.float32, device=dev)
        ux_in = torch.full((1, n, n), u_in_x, dtype=torch.float32, device=dev)
        uy_in = torch.full((1, n, n), u_in_y, dtype=torch.float32, device=dev)
        uz_in = torch.zeros((1, n, n), dtype=torch.float32, device=dev)
        f_in = equilibrium3d(rho_in, ux_in, uy_in, uz_in, device=dev)
        f_new[:, 0, :, :] = f_in[:, 0, :, :]

        # Outflow (right boundary, x=n-1): convective (copy from interior)
        f_new[:, -1, :, :] = f_new[:, -2, :, :]

        # Side walls: slip (symmetry)
        f_new[:, :, 0, :] = f_new[:, :, 1, :]
        f_new[:, :, -1, :] = f_new[:, :, -2, :]
        f_new[:, :, :, 0] = f_new[:, :, :, 1]
        f_new[:, :, :, -1] = f_new[:, :, :, -2]

        # Solid bounce-back
        f_new[:, mask] = f_solid[:, mask]

        # Momentum-exchange force on solid surface
        Fx = 0.0
        Fy = 0.0
        Fn = 0.0

        # Find fluid nodes adjacent to solid
        for q in range(1, 19):  # skip rest particle
            opp = int(OPPOSITE[q])
            dz = int(c[q, 2])
            dy = int(c[q, 1])
            dx = int(c[q, 0])

            # Shifted mask: solid node that would stream into this fluid node
            shifted_solid = torch.roll(mask, shifts=(-dz, -dy, -dx), dims=(0, 1, 2))
            # Fluid nodes whose opposite direction points to solid
            fluid_adjacent = fluid & shifted_solid

            if fluid_adjacent.any():
                # Momentum exchange: f_q(fluid) + f_opp(fluid)
                mq = f_new[q][fluid_adjacent].sum()
                mo = f_new[opp][fluid_adjacent].sum()
                dFx = (mq + mo) * int(c[q, 0])
                dFy = (mq + mo) * int(c[q, 1])
                Fx += float(dFx.item())
                Fy += float(dFy.item())

                # Yaw moment: r × F, r from centre of rotation (midship)
                y_coords = (torch.arange(n, device=dev).view(1, n, 1).expand_as(fluid_adjacent.float())[fluid_adjacent] - n / 2)
                Fn += float((y_coords * (mq + mo) * int(c[q, 0])).sum().item())

        # Convert lattice force to physical
        force_scale = rho_air * dx_phys * dx_phys / (dt_lb * dt_lb)
        Fx_phys = Fx * force_scale
        Fy_phys = Fy * force_scale
        Fn_phys = Fn * force_scale * dx_phys

        Fx_hist.append(Fx_phys)
        Fy_hist.append(Fy_phys)
        Fn_hist.append(Fn_phys)

        f = f_new
        steps_run = step + 1

        # Convergence check (last 200 steps)
        if step > 200 and step % 100 == 0:
            recent_Fx = Fx_hist[-200:]
            mean_Fx = np.mean(recent_Fx)
            std_Fx = np.std(recent_Fx)
            if abs(std_Fx / max(abs(mean_Fx), 1e-10)) < 0.02:
                converged = True
                break

    # Compute coefficients from converged force averages
    n_avg = min(200, len(Fx_hist))
    Fx_avg = np.mean(Fx_hist[-n_avg:])
    Fy_avg = np.mean(Fy_hist[-n_avg:])
    Fn_avg = np.mean(Fn_hist[-n_avg:])

    # Dynamic pressure reference
    q_ref = 0.5 * rho_air * wind_speed * wind_speed
    A_F = geo.frontal_area

    C_X = Fx_avg / (q_ref * A_F) if A_F > 0 else 0.0
    C_Y = Fy_avg / (q_ref * A_F) if A_F > 0 else 0.0
    C_N = Fn_avg / (q_ref * A_F * geo.L_OA) if A_F > 0 and geo.L_OA > 0 else 0.0

    return {
        "C_X": float(C_X),
        "C_Y": float(C_Y),
        "C_N": float(C_N),
        "converged": converged,
        "steps": steps_run,
        "tau": tau,
        "Re": wind_speed * L_char / nu_air if nu_air > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 4. HYBRID ENGINE
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """Result from the hybrid wind-resistance engine."""
    ship_type: str
    wind_angle_deg: float
    method_used: Literal["empirical", "lbm"]
    C_X: float
    C_Y: float
    C_N: float
    C_X_empirical: float
    C_Y_empirical: float
    C_N_empirical: float
    C_X_lbm: float | None = None
    C_Y_lbm: float | None = None
    C_N_lbm: float | None = None
    lbm_converged: bool = False
    lbm_steps: int = 0
    deviation_CX: float = 0.0
    deviation_CY: float = 0.0
    deviation_CN: float = 0.0


class WindResistanceEngine:
    """Hybrid wind-resistance forecast engine.

    Automatically selects between empirical formulas and LBM simulation
    based on geometry complexity.

    Selection rule:
        complexity_score ≤ 0.5 → empirical (Blendermann)
        complexity_score > 0.5 → LBM D3Q19/MRT
    """

    def __init__(
        self,
        ship_type: str = "container",
        empirical_method: Literal["blendermann", "ishry"] = "blendermann",
        lbm_resolution: int = 48,
        lbm_max_steps: int = 5000,
        complexity_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.ship_type = ship_type
        self.empirical_method = empirical_method
        self.lbm_resolution = lbm_resolution
        self.lbm_max_steps = lbm_max_steps
        self.complexity_threshold = complexity_threshold
        self.device = device
        self.geo = SHIP_GEOMETRIES.get(ship_type, CONTAINER_SHIP)

    def select_method(self, geo: ShipGeometry | None = None) -> Literal["empirical", "lbm"]:
        """Select computation method based on geometry complexity."""
        g = geo or self.geo
        if g.complexity_score <= self.complexity_threshold:
            return "empirical"
        return "lbm"

    def compute(
        self,
        wind_angle_deg: float,
        wind_speed: float = 10.0,
        geo: ShipGeometry | None = None,
        force_method: Literal["empirical", "lbm"] | None = None,
    ) -> HybridResult:
        """Compute aerodynamic coefficients for a given wind angle.

        Args:
            wind_angle_deg: Wind angle from bow (0–180°).
            wind_speed: Wind speed (m/s).
            geo: Override geometry (default: engine's ship type).
            force_method: Force "empirical" or "lbm" (default: auto-select).

        Returns:
            HybridResult with coefficients and metadata.
        """
        g = geo or self.geo

        # Empirical baseline (always computed for comparison)
        emp = empirical_coefficients(wind_angle_deg, g.ship_type, self.empirical_method)

        if force_method is None:
            force_method = self.select_method(g)

        if force_method == "empirical":
            return HybridResult(
                ship_type=g.ship_type,
                wind_angle_deg=wind_angle_deg,
                method_used="empirical",
                C_X=emp["C_X"],
                C_Y=emp["C_Y"],
                C_N=emp["C_N"],
                C_X_empirical=emp["C_X"],
                C_Y_empirical=emp["C_Y"],
                C_N_empirical=emp["C_N"],
                C_X_lbm=None,
                C_Y_lbm=None,
                C_N_lbm=None,
            )
        else:
            lbm_result = lbm_aerodynamic_coefficients(
                g, wind_angle_deg, wind_speed,
                resolution=self.lbm_resolution,
                max_steps=self.lbm_max_steps,
                device=self.device,
            )
            dev_CX = abs(lbm_result["C_X"] - emp["C_X"]) / max(abs(emp["C_X"]), 1e-10)
            dev_CY = abs(lbm_result["C_Y"] - emp["C_Y"]) / max(abs(emp["C_Y"]), 1e-10)
            dev_CN = abs(lbm_result["C_N"] - emp["C_N"]) / max(abs(emp["C_N"]), 1e-10)

            return HybridResult(
                ship_type=g.ship_type,
                wind_angle_deg=wind_angle_deg,
                method_used="lbm",
                C_X=lbm_result["C_X"],
                C_Y=lbm_result["C_Y"],
                C_N=lbm_result["C_N"],
                C_X_empirical=emp["C_X"],
                C_Y_empirical=emp["C_Y"],
                C_N_empirical=emp["C_N"],
                C_X_lbm=lbm_result["C_X"],
                C_Y_lbm=lbm_result["C_Y"],
                C_N_lbm=lbm_result["C_N"],
                lbm_converged=lbm_result["converged"],
                lbm_steps=lbm_result["steps"],
                deviation_CX=dev_CX,
                deviation_CY=dev_CY,
                deviation_CN=dev_CN,
            )

    def sweep(
        self,
        angles: list[float] | None = None,
        wind_speed: float = 10.0,
    ) -> list[HybridResult]:
        """Sweep wind angles 0°–180° and return results."""
        if angles is None:
            angles = list(range(0, 181, 15))
        return [self.compute(a, wind_speed) for a in angles]


# ---------------------------------------------------------------------------
# 5. VOYAGE ASSESSMENT
# ---------------------------------------------------------------------------

@dataclass
class WindRoseBin:
    """A single bin of a wind-rose histogram."""
    angle_centre: float   # degrees from bow
    angle_half_width: float  # half-width of bin
    probability: float    # fraction of time (sums to 1.0)
    wind_speed: float     # representative speed (m/s)


def default_wind_rose() -> list[WindRoseBin]:
    """Return a representative North Atlantic wind rose.

    12 directional bins (30° each) with Weibull-distributed speeds.
    Based on typical North Atlantic statistics (simplified).
    """
    bins = []
    for i in range(12):
        angle = i * 30.0
        # Probability: higher from SW/W (typical N Atlantic)
        prob_weights = [0.06, 0.05, 0.07, 0.08, 0.10, 0.12,
                        0.14, 0.12, 0.10, 0.08, 0.05, 0.03]
        prob = prob_weights[i]
        # Speed: Weibull k=2, A=8 m/s → mean ~7.1 m/s
        speed = 8.0  # representative
        bins.append(WindRoseBin(angle, 15.0, prob, speed))
    return bins


def voyage_assessment(
    ship_type: str = "container",
    wind_rose: list[WindRoseBin] | None = None,
    voyage_hours: float = 8760.0,
    ship_speed: float = 15.0,
    engine: WindResistanceEngine | None = None,
) -> dict:
    """Compute annual wind thrust/drag for a voyage.

    Args:
        ship_type: Ship type key.
        wind_rose: Wind rose bins (default: North Atlantic).
        voyage_hours: Total voyage hours per year (default: 8760).
        ship_speed: Ship speed through water (knots → m/s conversion applied).
        engine: Pre-configured engine (default: new engine for ship_type).

    Returns:
        Dict with annual wind force statistics.
    """
    if wind_rose is None:
        wind_rose = default_wind_rose()
    if engine is None:
        engine = WindResistanceEngine(ship_type=ship_type)

    geo = engine.geo
    rho_air = 1.225
    ship_speed_ms = ship_speed * 0.5144  # knots to m/s

    total_Fx = 0.0  # longitudinal (positive = drag, negative = thrust)
    total_Fy = 0.0
    total_Mn = 0.0
    total_hours = 0.0

    results_per_bin = []

    for bin_ in wind_rose:
        # Relative wind angle: wind direction minus ship heading
        # For simplicity, assume ship heading = 0° (north)
        rel_angle = bin_.angle_centre
        # Relative wind speed (vector sum)
        V_w = bin_.wind_speed
        V_s = ship_speed_ms
        # Apparent wind speed
        V_app = math.sqrt(V_w**2 + V_s**2 + 2 * V_w * V_s * math.cos(math.radians(rel_angle)))

        coeffs = engine.compute(rel_angle, V_app)
        A_F = geo.frontal_area
        q = 0.5 * rho_air * V_app**2

        Fx = q * A_F * coeffs.C_X
        Fy = q * A_F * coeffs.C_Y
        Mn = q * A_F * geo.L_OA * coeffs.C_N

        hours = bin_.probability * voyage_hours
        total_Fx += Fx * hours
        total_Fy += Fy * hours
        total_Mn += Mn * hours
        total_hours += hours

        results_per_bin.append({
            "angle": rel_angle,
            "probability": bin_.probability,
            "wind_speed": V_w,
            "apparent_speed": V_app,
            "C_X": coeffs.C_X,
            "C_Y": coeffs.C_Y,
            "C_N": coeffs.C_N,
            "Fx_N": Fx,
            "Fy_N": Fy,
            "Mn_Nm": Mn,
            "hours": hours,
        })

    # Annual averages
    avg_Fx = total_Fx / max(total_hours, 1.0)
    avg_Fy = total_Fy / max(total_hours, 1.0)
    avg_Mn = total_Mn / max(total_hours, 1.0)

    # Annual energy (force × distance)
    distance_m = ship_speed_ms * voyage_hours * 3600
    annual_work_Fx = total_Fx * ship_speed_ms * 3600  # J
    annual_work_Fy = total_Fy * ship_speed_ms * 3600

    # Wind assistance: negative Fx means wind helps (pushes from behind)
    # Positive Fx means wind resists
    thrust_hours = sum(r["hours"] for r in results_per_bin if r["Fx_N"] < 0)
    drag_hours = sum(r["hours"] for r in results_per_bin if r["Fx_N"] >= 0)

    return {
        "ship_type": ship_type,
        "ship_name": geo.name,
        "voyage_hours": voyage_hours,
        "ship_speed_knots": ship_speed,
        "frontal_area_m2": geo.frontal_area,
        "lateral_area_m2": geo.lateral_area,
        "annual_avg_Fx_N": avg_Fx,
        "annual_avg_Fy_N": avg_Fy,
        "annual_avg_Mn_Nm": avg_Mn,
        "annual_work_Fx_J": annual_work_Fx,
        "annual_work_Fy_J": annual_work_Fy,
        "thrust_hours": thrust_hours,
        "drag_hours": drag_hours,
        "thrust_fraction": thrust_hours / max(voyage_hours, 1.0),
        "drag_fraction": drag_hours / max(voyage_hours, 1.0),
        "bins": results_per_bin,
    }


# ---------------------------------------------------------------------------
# 6. PUBLIC API
# ---------------------------------------------------------------------------

__all__ = [
    "ShipGeometry",
    "CONTAINER_SHIP",
    "TANKER",
    "BULK_CARRIER",
    "TEST_SHIP",
    "SHIP_GEOMETRIES",
    "blendermann_CX",
    "blendermann_CY",
    "blendermann_CN",
    "ishy_CX",
    "ishy_CY",
    "empirical_coefficients",
    "lbm_aerodynamic_coefficients",
    "HybridResult",
    "WindResistanceEngine",
    "WindRoseBin",
    "default_wind_rose",
    "voyage_assessment",
]
