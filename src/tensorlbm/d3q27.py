"""D3Q27 lattice constants and equilibrium distribution.

The D3Q27 lattice has 27 velocity directions covering all combinations of
(cx, cy, cz) ∈ {−1, 0, 1}³. Compared to D3Q19 it includes the 8 corner
directions (|c| = √3) and therefore achieves 4th-order isotropy, which can
reduce numerical artefacts in flows with strong corner-region gradients
(e.g. flows past bluff bodies or in confined geometries).

Lattice weights (Qian, 1992):

- Rest (0,0,0):           w = 8/27
- Face-centre (|c|=1):    w = 2/27  (×6)
- Edge-centre (|c|=√2):   w = 1/54  (×12)
- Corner     (|c|=√3):    w = 1/216 (×8)
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import torch

# Cache for streaming index tensors keyed by (nz, ny, nx, device_type, device_index)
_stream27_cache: dict[
    tuple[Any, ...],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}

_C_DATA = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [1, 0, 1],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, 1, 1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [1, 1, -1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
]

C = torch.tensor(_C_DATA, dtype=torch.int64)

_w_rest = 8.0 / 27.0
_w_face = 2.0 / 27.0
_w_edge = 1.0 / 54.0
_w_corner = 1.0 / 216.0

_W_DATA = [_w_rest] + [_w_face] * 6 + [_w_edge] * 12 + [_w_corner] * 8
W = torch.tensor(_W_DATA, dtype=torch.float32)


def _build_opposite() -> torch.Tensor:
    c_list = [tuple(row) for row in _C_DATA]
    opp = []
    for cx, cy, cz in c_list:
        target = (-cx, -cy, -cz)
        opp.append(c_list.index(target))
    return torch.tensor(opp, dtype=torch.int64)


OPPOSITE = _build_opposite()


@dataclass(frozen=True)
class PropellerLoadReport:
    """Nondimensional moving-wall propeller loads in lattice units.

    ``force_on_fluid`` and ``torque_on_fluid`` are linkwise momentum-exchange
    totals.  ``force_on_wall`` and ``torque_on_wall`` are their Newton-third-law
    reactions. ``thrust`` is positive along ``axis``; ``shaft_torque`` is the
    positive torque supplied by the shaft to maintain positive rotation.
    This is a consistency report, not a blade-resolved open-water result.
    """

    force_on_fluid: torch.Tensor
    torque_on_fluid: torch.Tensor
    force_on_wall: torch.Tensor
    torque_on_wall: torch.Tensor
    thrust: float
    shaft_torque: float
    advance_ratio: float
    kt: float
    kq: float
    eta_o: float
    max_mach: float


@dataclass(frozen=True)
class ControlVolumeMomentumReport27:
    """Distribution-momentum balance for a D3Q27 control-volume diagnostic.

    ``distribution_momentum_change`` is the population momentum after minus
    before the sampled update. ``force_on_fluid`` is the accumulated linkwise
    moving-wall ME load for the same update, while ``residual`` is their
    difference. In a closed or periodic volume containing every reflected
    population and no body force, the residual is exact to roundoff. For an
    open volume it also contains unreported face fluxes; for a volume with
    collision/forcing it contains their momentum source. It therefore bounds
    the unaccounted contribution rather than validating a boundary treatment.
    """

    distribution_momentum_change: torch.Tensor
    force_on_fluid: torch.Tensor
    force_on_wall: torch.Tensor
    residual: torch.Tensor
    residual_norm: float
    max_mach: float

    def within_tolerance(self, *, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
        """Whether the unresolved control-volume contribution is bounded."""
        if atol < 0.0 or rtol < 0.0:
            raise ValueError("atol and rtol must be nonnegative")
        reference = float(torch.linalg.vector_norm(self.force_on_fluid).item())
        return self.residual_norm <= atol + rtol * reference


def control_volume_momentum_balance27(
    populations_before: torch.Tensor,
    populations_after: torch.Tensor,
    force_on_fluid: torch.Tensor,
    *,
    max_lattice_speed: float,
    low_mach_limit: float = 0.1,
) -> ControlVolumeMomentumReport27:
    """Compare D3Q27 distribution momentum change with linkwise wall ME.

    This is diagnostic-only and never evolves populations. Inputs may have
    shape ``(27, ...)``; all spatial/control-volume dimensions are summed.
    Exact equality is meaningful only when the selected volume is closed or
    periodic and the before/after states bracket only link reflection. Open
    boundaries, collision, or forcing supply additional momentum and must be
    accounted for separately; their net effect is exposed as ``residual``.
    The low-Mach gate prevents presenting a compressible update as an
    incompressible propeller consistency result.
    """
    if populations_before.shape != populations_after.shape or populations_before.ndim < 1:
        raise ValueError("populations_before and populations_after must have identical shape (27, ...)")
    if populations_before.shape[0] != 27:
        raise ValueError("population direction dimension must have length 27")
    if force_on_fluid.shape != (3,):
        raise ValueError("force_on_fluid must have shape (3,)")
    if not math.isfinite(max_lattice_speed) or max_lattice_speed < 0.0:
        raise ValueError("max_lattice_speed must be finite and nonnegative")
    if not math.isfinite(low_mach_limit) or low_mach_limit <= 0.0:
        raise ValueError("low_mach_limit must be finite and positive")
    if not (torch.isfinite(populations_before).all() and torch.isfinite(populations_after).all()
            and torch.isfinite(force_on_fluid).all()):
        raise ValueError("populations and force_on_fluid must be finite")

    max_mach = max_lattice_speed * math.sqrt(3.0)
    if max_mach >= low_mach_limit:
        raise ValueError(
            f"invalid low-Mach control-volume diagnostic: max Mach {max_mach:.6g} >= "
            f"limit {low_mach_limit:.6g}"
        )
    directions = C.to(device=populations_before.device, dtype=populations_before.dtype)
    delta_populations = populations_after - populations_before
    distribution_momentum_change = (
        directions * delta_populations.reshape(27, -1).sum(dim=1, keepdim=True)
    ).sum(dim=0)
    force = force_on_fluid.to(device=populations_before.device, dtype=populations_before.dtype)
    residual = distribution_momentum_change - force
    return ControlVolumeMomentumReport27(
        distribution_momentum_change=distribution_momentum_change,
        force_on_fluid=force,
        force_on_wall=-force,
        residual=residual,
        residual_norm=float(torch.linalg.vector_norm(residual).item()),
        max_mach=max_mach,
    )


def report_propeller_linkwise_loads(
    force_on_fluid: torch.Tensor,
    torque_on_fluid: torch.Tensor,
    *,
    advance_speed: float,
    rotation_rate: float,
    diameter: float,
    density: float = 1.0,
    axis: tuple[float, float, float] | torch.Tensor = (1.0, 0.0, 0.0),
    max_lattice_speed: float | None = None,
    low_mach_limit: float = 0.1,
) -> PropellerLoadReport:
    """Report signed ``J``, ``K_T``, ``K_Q``, and ideal propulsive efficiency.

    ``rotation_rate`` is revolutions per lattice time step and its sign defines
    the positive shaft-rotation direction.  The supplied linkwise totals must
    be **loads on the fluid**, as returned by
    :func:`moving_wall_linkwise_me_force_torque`.  The helper first forms the
    wall reaction, then uses ``T = F_wall . axis`` and
    ``Q = -sign(n) * M_wall . axis``.  Thus a propeller transferring positive
    axial momentum and angular momentum to the fluid has positive ``KT`` and
    ``KQ``.

    The incompressible LBM interpretation is rejected unless the supplied (or
    conservative ``max(|U_A|, pi*D*|n|)``) lattice speed is strictly below
    ``low_mach_limit * c_s``, where ``c_s=1/sqrt(3)``.  This diagnostic does
    not establish blade-resolved or validated open-water performance.
    """
    if force_on_fluid.shape != (3,) or torque_on_fluid.shape != (3,):
        raise ValueError("force_on_fluid and torque_on_fluid must each have shape (3,)")
    values = (advance_speed, rotation_rate, diameter, density, low_mach_limit)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("reference quantities must be finite")
    if rotation_rate == 0.0 or diameter <= 0.0 or density <= 0.0:
        raise ValueError("rotation_rate must be nonzero and diameter/density must be positive")
    if low_mach_limit <= 0.0:
        raise ValueError("low_mach_limit must be positive")
    if not torch.isfinite(force_on_fluid).all() or not torch.isfinite(torque_on_fluid).all():
        raise ValueError("linkwise force and torque must be finite")

    axis_tensor = torch.as_tensor(axis, device=force_on_fluid.device, dtype=force_on_fluid.dtype)
    if axis_tensor.shape != (3,) or not torch.isfinite(axis_tensor).all():
        raise ValueError("axis must be a finite vector with shape (3,)")
    axis_norm = float(torch.linalg.vector_norm(axis_tensor).item())
    if axis_norm == 0.0:
        raise ValueError("axis must be nonzero")
    axis_unit = axis_tensor / axis_norm

    conservative_speed = max(abs(advance_speed), math.pi * diameter * abs(rotation_rate))
    speed = conservative_speed if max_lattice_speed is None else max_lattice_speed
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("max_lattice_speed must be finite and nonnegative")
    max_mach = speed * math.sqrt(3.0)
    if max_mach >= low_mach_limit:
        raise ValueError(
            f"invalid low-Mach report: max Mach {max_mach:.6g} >= limit {low_mach_limit:.6g}"
        )

    force_on_wall = -force_on_fluid
    torque_on_wall = -torque_on_fluid
    thrust = float(torch.dot(force_on_wall, axis_unit).item())
    rotation_sign = math.copysign(1.0, rotation_rate)
    shaft_torque = float((-rotation_sign * torch.dot(torque_on_wall, axis_unit)).item())
    denominator_t = density * rotation_rate * rotation_rate * diameter**4
    denominator_q = density * rotation_rate * rotation_rate * diameter**5
    kt = thrust / denominator_t
    kq = shaft_torque / denominator_q
    advance_ratio = advance_speed / (abs(rotation_rate) * diameter)
    eta_o = advance_ratio * kt / (2.0 * math.pi * kq) if kq != 0.0 else float("nan")
    return PropellerLoadReport(
        force_on_fluid=force_on_fluid,
        torque_on_fluid=torque_on_fluid,
        force_on_wall=force_on_wall,
        torque_on_wall=torque_on_wall,
        thrust=thrust,
        shaft_torque=shaft_torque,
        advance_ratio=advance_ratio,
        kt=kt,
        kq=kq,
        eta_o=eta_o,
        max_mach=max_mach,
    )


def moving_wall_linkwise_me_force_torque(
    outgoing: torch.Tensor,
    directions: torch.Tensor,
    weights: torch.Tensor,
    wall_velocity: torch.Tensor,
    positions: torch.Tensor,
    origin: tuple[float, float, float] | torch.Tensor = (0.0, 0.0, 0.0),
    density: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return D3Q27 moving-wall link momentum-exchange force and torque.

    Each row describes one fluid--solid link, with ``directions`` pointing
    from the fluid cell into the solid.  ``link_force``, ``force``, and
    ``torque`` are the momentum and angular-momentum transfer **to the
    fluid**.  The equal-and-opposite load on the wall is ``-force`` and
    ``-torque`` about the same origin.  This is a diagnostic-only primitive:
    it does not mutate any population or participate in collision/streaming.

    The reflected population follows the same moving-wall correction used by
    link bounce-back, ``f_r = f_o - 2 rho w (c . u_w) / cs^2``, with
    ``cs^2 = 1/3``.  The link force is ``-(f_o + f_r) c``, i.e. the change in
    fluid momentum as an incident population along ``c`` is reflected along
    ``-c``.  Consequently a stationary wall recovers conventional stationary
    momentum exchange.
    """
    if outgoing.ndim != 1:
        raise ValueError("outgoing must have shape (n_links,)")
    n_links = outgoing.shape[0]
    for name, value in {
        "directions": directions,
        "wall_velocity": wall_velocity,
        "positions": positions,
    }.items():
        if value.shape != (n_links, 3):
            raise ValueError(f"{name} must have shape (n_links, 3)")
    if weights.shape != (n_links,):
        raise ValueError("weights must have shape (n_links,)")

    directions = directions.to(device=outgoing.device, dtype=outgoing.dtype)
    wall_velocity = wall_velocity.to(device=outgoing.device, dtype=outgoing.dtype)
    positions = positions.to(device=outgoing.device, dtype=outgoing.dtype)
    weights = weights.to(device=outgoing.device, dtype=outgoing.dtype)
    rho = torch.as_tensor(density, device=outgoing.device, dtype=outgoing.dtype)
    rho = torch.broadcast_to(rho, (n_links,))
    correction = 6.0 * rho * weights * (directions * wall_velocity).sum(dim=1)
    reflected = outgoing - correction
    link_force = -(outgoing + reflected).unsqueeze(1) * directions
    force = link_force.sum(dim=0)
    origin_tensor = torch.as_tensor(origin, device=outgoing.device, dtype=outgoing.dtype)
    if origin_tensor.shape != (3,):
        raise ValueError("origin must have shape (3,)")
    torque = torch.cross(positions - origin_tensor, link_force, dim=1).sum(dim=0)
    return reflected, link_force, force, torque


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_on(device: torch.device) -> torch.Tensor:
    return W.to(device)


def equilibrium27(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    device: torch.device | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute D3Q27 Maxwell-Boltzmann equilibrium distribution.

    Args:
        rho: Density field, shape ``(nz, ny, nx)``.
        ux: x-velocity field, shape ``(nz, ny, nx)``.
        uy: y-velocity field, shape ``(nz, ny, nx)``.
        uz: z-velocity field, shape ``(nz, ny, nx)``.
        device: Target device (inferred from *rho* if *None*).
        out: optional pre-allocated output tensor of shape ``(27, nz, ny, nx)``.
            If provided, the result is written into this tensor in-place,
            avoiding a new allocation.

    Returns:
        Equilibrium distribution of shape ``(27, nz, ny, nx)``.
    """
    if device is None:
        device = rho.device
    c = _c_on(device).float()
    w = _w_on(device).view(27, 1, 1, 1)

    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    u_sq = ux * ux + uy * uy + uz * uz
    cu = cx * ux + cy * uy + cz * uz
    result = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
    if out is not None:
        out.copy_(result)
        return out
    return result


def macroscopic27(
    f: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover (rho, ux, uy, uz) from D3Q27 distributions.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        device: Target device (inferred from *f* if *None*).

    Returns:
        Tuple ``(rho, ux, uy, uz)`` of shape ``(nz, ny, nx)`` each.
    """
    if device is None:
        device = f.device
    c = _c_on(device).float()
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * cx).sum(dim=0) / rho_safe
    uy = (f * cy).sum(dim=0) / rho_safe
    uz = (f * cz).sum(dim=0) / rho_safe
    return rho, ux, uy, uz


def collide_bgk27(f: torch.Tensor, tau: float) -> torch.Tensor:
    """D3Q27 single-relaxation-time BGK collision.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Relaxation time τ > 0.5.

    Returns:
        Post-collision distribution of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    return f - (f - feq) / tau


def collide_trt27(
    f: torch.Tensor,
    tau_plus: float,
    lambda_trt: float = 3.0 / 16.0,
) -> torch.Tensor:
    """Two-relaxation-time (TRT) collision step for D3Q27.

    Uses two independent relaxation rates: *τ₊* controls the symmetric part
    (sets viscosity ν = (τ₊ − ½) / 3) and *τ₋* controls the anti-symmetric
    part (derived from the magic parameter Λ). Setting Λ = 3/16 eliminates
    wall-placement errors in Poiseuille flow (Ginzburg 2008).

    The symmetric/anti-symmetric decomposition uses the D3Q27
    :data:`OPPOSITE` direction map, which pairs every direction with its
    negation (including the 8 corner directions absent from D3Q19).

    Reference
    ---------
    Ginzburg, I. (2008). Two-relaxation-time lattice Boltzmann scheme.
    *Commun. Comput. Phys.* 3(2), 427–478.

    Args:
        f:           Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau_plus:    Symmetric relaxation time (τ₊ > 0.5).
        lambda_trt:  Magic parameter Λ (default 3/16).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)

    tau_minus = 0.5 + lambda_trt / (tau_plus - 0.5)

    opp = OPPOSITE.to(f.device)
    f_plus = 0.5 * (f + f[opp])
    f_minus = 0.5 * (f - f[opp])
    feq_plus = 0.5 * (feq + feq[opp])
    feq_minus = 0.5 * (feq - feq[opp])

    return f - (f_plus - feq_plus) / tau_plus - (f_minus - feq_minus) / tau_minus


def collide_rlbm27(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Regularized BGK (RLBM) collision step for D3Q27.

    Projects the non-equilibrium distribution onto the second-order Hermite
    polynomial subspace before BGK relaxation, filtering out higher-order
    ghost modes for improved stability at low viscosity (τ → 0.5).
    See Latt & Chopard, *Math. Comput. Simul.* (2006).

    The D3Q27 lattice has 4th-order isotropy (it includes the 8 corner
    directions), so the second-order Hermite projection is exact for the
    hydrodynamic stress tensor — the same projection formula as D3Q19,
    applied with the D3Q27 weights and velocity set.

    Args:
        f:   Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Relaxation time (τ > 0.5). Kinematic viscosity ν = (τ − ½)/3.

    Returns:
        Updated distribution tensor of the same shape.
    """
    device = f.device
    c = _c_on(device).to(f.dtype)
    w = _w_on(device).to(f.dtype)

    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    fneq = f - feq

    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    # Second-order non-equilibrium moments Π_αβ
    pi_xx = (cx * cx * fneq).sum(dim=0)
    pi_yy = (cy * cy * fneq).sum(dim=0)
    pi_zz = (cz * cz * fneq).sum(dim=0)
    pi_xy = (cx * cy * fneq).sum(dim=0)
    pi_xz = (cx * cz * fneq).sum(dim=0)
    pi_yz = (cy * cz * fneq).sum(dim=0)

    cs2 = 1.0 / 3.0
    h_xx = cx * cx - cs2
    h_yy = cy * cy - cs2
    h_zz = cz * cz - cs2
    h_xy = cx * cy
    h_xz = cx * cz
    h_yz = cy * cz
    w_view = w.view(27, 1, 1, 1)
    fneq_reg = (9.0 / 2.0) * w_view * (
        h_xx * pi_xx
        + h_yy * pi_yy
        + h_zz * pi_zz
        + 2.0 * h_xy * pi_xy
        + 2.0 * h_xz * pi_xz
        + 2.0 * h_yz * pi_yz
    )

    return feq + (1.0 - 1.0 / tau) * fneq_reg


def _build_d3q27_mrt_matrices() -> tuple[list[list[float]], list[list[float]]]:
    """Compute and return (M, M_inv) for the D3Q27 MRT transformation.

    Constructs the 27×27 transformation matrix using the Gram–Schmidt
    orthogonalised polynomial basis over the D3Q27 velocity set
    ``{cx, cy, cz} ∈ {−1, 0, 1}³``.  The basis polynomials follow the
    Qian/d'Humières moment hierarchy:

    * Row 0:  1                               (mass)
    * Row 1:  cx                              (x-momentum)
    * Row 2:  cy                              (y-momentum)
    * Row 3:  cz                              (z-momentum)
    * Row 4:  cx² + cy² + cz²                (energy, e)
    * Row 5:  cx²                             (normal stress xx; raw)
    * Row 6:  cy² − cz²                       (normal stress yy–zz; raw)
    * Row 7:  cx·cy                           (shear stress xy)
    * Row 8:  cx·cz                           (shear stress xz)
    * Row 9:  cy·cz                           (shear stress yz)
    * Rows 10–26: higher-order moments via Gram–Schmidt orthogonalisation.

    The resulting matrix is verified to be full rank (rank 27).
    """
    import numpy as np

    c_np = np.array(_C_DATA, dtype=np.float64)  # (27, 3)
    cx, cy, cz = c_np[:, 0], c_np[:, 1], c_np[:, 2]
    e2 = cx**2 + cy**2 + cz**2

    # Define raw moment vectors (length 27 each) in physical significance order
    raw_rows: list[np.ndarray] = [
        np.ones(27),           # 0: mass
        cx,                    # 1: jx
        cy,                    # 2: jy
        cz,                    # 3: jz
        e2,                    # 4: energy e = |c|^2
        3.0 * cx**2 - e2,      # 5: Nxx  (normal stress xx)
        cy**2 - cz**2,         # 6: Nyy  (normal stress yy-zz)
        cx * cy,               # 7: Pxy  (shear stress xy)
        cx * cz,               # 8: Pxz  (shear stress xz)
        cy * cz,               # 9: Pyz  (shear stress yz)
        # 3rd-order raw moments
        cx * e2,               # 10: qx
        cy * e2,               # 11: qy
        cz * e2,               # 12: qz
        cx**2 * cy,            # 13
        cx**2 * cz,            # 14
        cy**2 * cx,            # 15
        cy**2 * cz,            # 16
        cz**2 * cx,            # 17
        cz**2 * cy,            # 18
        # 4th-order raw moments
        e2**2,                 # 19
        cx**2 * e2,            # 20
        cy**2 * e2,            # 21
        cz**2 * e2,            # 22
        cx**2 * cy**2,         # 23
        cx**2 * cz**2,         # 24
        cy**2 * cz**2,         # 25
        cx * cy * cz,          # 26
    ]

    # Gram–Schmidt orthogonalisation to ensure full rank
    orth_rows: list[np.ndarray] = []
    for row in raw_rows:
        v = row.copy()
        for prev in orth_rows:
            v = v - (np.dot(v, prev) / np.dot(prev, prev)) * prev
        norm = np.sqrt(np.dot(v, v))
        if norm < 1e-14:
            # Row is linearly dependent — replace with an orthogonal complement
            # by searching for a standard-basis vector not yet represented
            for i in range(27):
                e_i = np.zeros(27)
                e_i[i] = 1.0
                u = e_i.copy()
                for prev in orth_rows:
                    u = u - (np.dot(u, prev) / np.dot(prev, prev)) * prev
                if np.sqrt(np.dot(u, u)) > 1e-10:
                    v = u
                    break
        orth_rows.append(v)

    matrix = np.array(orth_rows, dtype=np.float64)
    assert np.linalg.matrix_rank(matrix) == 27, "D3Q27 MRT matrix is rank-deficient"
    matrix_inv = np.linalg.inv(matrix)
    return matrix.tolist(), matrix_inv.tolist()


_M_D3Q27_DATA, _M_D3Q27_INV_DATA = _build_d3q27_mrt_matrices()


@functools.cache
def _get_d3q27_mrt_matrices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.tensor(_M_D3Q27_DATA, dtype=torch.float32, device=device)
    matrix_inv = torch.tensor(_M_D3Q27_INV_DATA, dtype=torch.float32, device=device)
    return matrix, matrix_inv


def collide_mrt27(
    f: torch.Tensor,
    tau: float,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q27 multi-relaxation-time (MRT) collision step.

    Shear viscosity is controlled by *tau*: ν = (τ − ½)/3.  Independent
    relaxation rates for non-hydrodynamic moments improve stability at high
    Reynolds numbers.

    Relaxation rates:
        * Rows 0–3  (mass, momenta):  0 (conserved)
        * Row  4    (energy e):       s_e
        * Rows 5–9  (stress modes):   1/tau
        * Rows 10–18 (3rd-order):     s_q
        * Rows 19–26 (4th-order+):    s_pi (defaults to s_e)

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Relaxation time for shear stress (τ > ½).
        s_e: Relaxation rate for the energy moment.
        s_eps: Relaxation rate for the energy-square moment (row 19).
        s_q: Relaxation rate for 3rd-order heat-flux moments (rows 10–18).
        s_pi: Relaxation rate for 4th-order moments (rows 20–26);
              defaults to *s_e* when *None*.

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    matrix, matrix_inv = _get_d3q27_mrt_matrices(device)

    s_nu = 1.0 / tau
    s_vec = torch.tensor(
        [
            0.0,   # 0  mass
            0.0,   # 1  jx
            0.0,   # 2  jy
            0.0,   # 3  jz
            s_e,   # 4  energy
            s_nu,  # 5  Nxx
            s_nu,  # 6  Nyy
            s_nu,  # 7  Pxy
            s_nu,  # 8  Pxz
            s_nu,  # 9  Pyz
            s_q,   # 10 qx
            s_q,   # 11 qy
            s_q,   # 12 qz
            s_q,   # 13
            s_q,   # 14
            s_q,   # 15
            s_q,   # 16
            s_q,   # 17
            s_q,   # 18
            s_eps, # 19 e²
            s_pi,  # 20
            s_pi,  # 21
            s_pi,  # 22
            s_pi,  # 23
            s_pi,  # 24
            s_pi,  # 25
            s_pi,  # 26
        ],
        dtype=f.dtype,
        device=device,
    )

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    feq_flat = feq.reshape(27, -1)

    moments = matrix @ f_flat
    moments_eq = matrix @ feq_flat
    moments_star = moments - s_vec.unsqueeze(1) * (moments - moments_eq)
    return (matrix_inv @ moments_star).reshape(27, nz, ny, nx)


def correct_mass27(f: torch.Tensor, target_mass: float) -> torch.Tensor:
    """Redistribute mass uniformly to correct global mass drift (D3Q27).

    Rescales the entire distribution tensor so that the sum of all
    populations equals *target_mass*. This corrects slow mass drift
    accumulated by inexact boundary conditions over many time steps.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        target_mass: Desired total mass (sum of all populations).

    Returns:
        Rescaled distribution tensor of the same shape.
    """
    current = f.sum()
    if current.abs() < 1e-30:
        return f
    return f * (target_mass / current)
_STREAM27_SHIFTS = None

def _init_stream27_shifts():
    """Pre-compute D3Q27 streaming shifts as Python tuples (no host sync)."""
    global _STREAM27_SHIFTS
    if _STREAM27_SHIFTS is not None:
        return
    from .d3q27 import C as C27
    shifts = [(0, 0, 0)]
    for q in range(1, 27):
        cx, cy, cz = C27[q].tolist()
        shifts.append((int(cx), int(cy), int(cz)))
    _STREAM27_SHIFTS = shifts

def stream27_roll(f: torch.Tensor) -> torch.Tensor:
    """Memory-optimized D3Q27 streaming using torch.roll.

    Uses pre-computed Python-tuple shifts (no .item() host sync)
    and torch.empty_like (no per-step allocation). Eliminates the
    4×[27,N] int64 index tensors that cause OOM on large grids.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.

    Returns:
        Streamed tensor of the same shape.
    """
    _init_stream27_shifts()
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _STREAM27_SHIFTS[q]
        if sx == 0 and sy == 0 and sz == 0:
            out[q] = f[q]
        else:
            out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out

# Backward-compatible alias
stream27 = stream27_roll

__all__ = [
    "W",
    "OPPOSITE",
    "PropellerLoadReport",
    "ControlVolumeMomentumReport27",
    "moving_wall_linkwise_me_force_torque",
    "control_volume_momentum_balance27",
    "report_propeller_linkwise_loads",
    "equilibrium27",
    "macroscopic27",
    "collide_bgk27",
    "collide_trt27",
    "collide_rlbm27",
    "collide_mrt27",
    "stream27",
    "stream27_roll",
    "correct_mass27",
    "_get_d3q27_mrt_matrices",
]
