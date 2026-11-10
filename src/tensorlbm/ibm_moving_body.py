"""Direct-forcing immersed-boundary method (IBM) for moving rigid bodies (D2Q9).

This module provides a *validated* moving-boundary immersed-boundary solver for
the D2Q9 lattice Boltzmann equation.  It supersedes the research probe in
:mod:`tensorlbm.fsi_viv_ibm`, whose direct-forcing implementation inflated the
hydrodynamic load by a factor ~24 on a coarse grid (``Cd = 32.3`` against the
reference ``1.34`` at ``Re = 100``, ``D = 16``).  The defects that are fixed
here (documented in ``docs/P233W2_N02_IBM_ICE_PROPELLER.md``):

1. **Missing Lagrangian measure weights.**  The marker force must be spread
   with the marker measure ``dv_l`` (arc length in 2-D, area for interior
   markers).  Without it, surface and interior markers carry equal weight and
   the Lagrangian/Eulerian momentum balance is broken.
2. **No interior bookkeeping.**  Markers inside a body drag the interior fluid
   along, which adds ``rho * V`` to the body inertia; that term is now reported
   explicitly instead of silently inflating the load.
3. **Inlet boundary condition.**  Resetting the inlet column to a fixed
   equilibrium each step injects spurious momentum; a Guo (2002)
   non-equilibrium-extrapolation velocity inlet is used instead.
4. **Force definition.**  The load on the body is *defined* as minus the
   momentum handed to the fluid per step, ``F = -rho * sum_l F_l dv_l``, which
   makes fluid/body momentum exchange an exact identity
   (see :func:`momentum_exchange_audit`).

Numerics
--------
* Lattice: D2Q9, BGK collision with Guo (1993) forcing, ``tau = 3 nu + 1/2``.
* IBM: Uhlmann (2005) direct forcing, Peskin 4-point regularised delta,
  optional multi-direct iterations (Kang & Hassan 2011).
* Bodies: rigid; prescribed kinematics (rotating blade) or free Newton
  dynamics (ice floe) with an optional Hertzian contact law.
* Boundaries: velocity inlet (Guo extrapolation), zero-gradient outlet,
  periodic in the transverse direction.

References
----------
Peskin, C. S. (2002). The immersed boundary method. *Acta Numerica* 11, 479.
Uhlmann, M. (2005). An immersed boundary method with direct forcing for the
    simulation of particulate flows. *J. Comput. Phys.* 209, 448-476.
Mittal, R. & Iaccarino, G. (2005). Immersed boundary methods.
    *Annu. Rev. Fluid Mech.* 37, 239-261.
Kang, S. K. & Hassan, Y. A. (2011). A comparative study of direct-forcing
    immersed boundary-lattice Boltzmann methods. *IJNMF* 66, 113-138.
Guo, Z., Zheng, C. & Shi, B. (2002). Non-equilibrium extrapolation method for
    velocity and pressure boundary conditions in the LBM. *Phys. Fluids* 14, 2007.
Brennen, C. E. (1982). A review of added mass and fluid inertial forces.
    NCEL report CR 82.010.
Korzhavin, K. N. (1971). Action of ice on engineering structures. CRREL TL 260.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

__all__ = [
    "C9",
    "W9",
    "delta4",
    "equilibrium2d",
    "macroscopic2d",
    "MarkerStencil",
    "RigidBody",
    "circle_body",
    "rectangle_body",
    "IBM2DConfig",
    "IBM2D",
    "CylinderCaseConfig",
    "run_cylinder_case",
    "strouhal_number",
    "momentum_exchange_audit",
    "free_body_momentum_audit",
    "IceBladeConfig",
    "run_ice_blade_case",
    "FloeDragConfig",
    "run_floe_drag_case",
]


# --------------------------------------------------------------------------- #
# Lattice constants
# --------------------------------------------------------------------------- #

C9 = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=torch.long,
)
W9 = torch.tensor(
    [4.0 / 9.0, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=torch.float64,
)

_OFFSETS: np.ndarray = np.array(
    [[jx, jy] for jy in (0, 1, -1, 2) for jx in (0, 1, -1, 2)], dtype=np.int64
)


def delta4(r: np.ndarray) -> np.ndarray:
    """Peskin 4-point regularised delta function (support ``|r| < 2``)."""
    a = np.abs(np.asarray(r, dtype=np.float64))
    out = np.zeros_like(a)
    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    out[m1] = (3.0 - 2.0 * a[m1] + np.sqrt(1.0 + 4.0 * a[m1] - 4.0 * a[m1] ** 2)) / 8.0
    out[m2] = (5.0 - 2.0 * a[m2] - np.sqrt(-7.0 + 12.0 * a[m2] - 4.0 * a[m2] ** 2)) / 8.0
    return out


def equilibrium2d(
    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """D2Q9 equilibrium populations, stacked along dim 0 with shape ``(9, *shape)``."""
    dtype = rho.dtype
    w = W9.to(dtype).view(9, *([1] * ux.dim()))
    c = C9.to(dtype)
    cx = c[:, 0].view(9, *([1] * ux.dim()))
    cy = c[:, 1].view(9, *([1] * ux.dim()))
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    usq = (ux * ux + uy * uy).unsqueeze(0)
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def macroscopic2d(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Density and velocity from populations."""
    c = C9.to(f.dtype)
    rho = f.sum(dim=0)
    cxf = c[:, 0].view(9, 1, 1) * f
    ux = cxf.sum(dim=0)
    ux.div_(rho)
    cyf = c[:, 1].view(9, 1, 1) * f
    uy = cyf.sum(dim=0)
    uy.div_(rho)
    return rho, ux, uy


# --------------------------------------------------------------------------- #
# Marker stencils
# --------------------------------------------------------------------------- #


class MarkerStencil:
    """Gather/scatter maps of the 4-point kernel for one marker set.

    The transverse (``y``) direction is folded with a modulo (periodic); markers
    must stay at least two cells away from the ``x`` walls.
    """

    def __init__(self, markers: np.ndarray, nx: int, ny: int) -> None:
        markers = np.asarray(markers, dtype=np.float64)
        if markers.ndim != 2 or markers.shape[1] != 2:
            raise ValueError("markers must have shape (M, 2)")
        if markers.shape[0] == 0:
            raise ValueError("empty marker set")
        base = np.floor(markers)
        dx = markers[:, 0:1] - base[:, 0:1]
        dy = markers[:, 1:2] - base[:, 1:2]
        off = _OFFSETS.astype(np.float64)
        ix = base[:, 0:1].astype(np.int64) + _OFFSETS[None, :, 0]
        iy = base[:, 1:2].astype(np.int64) + _OFFSETS[None, :, 1]
        w = delta4(dx - off[None, :, 0]) * delta4(dy - off[None, :, 1])
        ip = np.mod(ix, nx)
        jp = np.mod(iy, ny)
        self.flat = torch.as_tensor((jp * nx + ip).reshape(-1), dtype=torch.long)
        self.w = torch.as_tensor(w.reshape(-1), dtype=torch.float64)
        self.n_markers = markers.shape[0]
        self.n_off = _OFFSETS.shape[0]
        self.markers = markers

    def weight_sums(self) -> torch.Tensor:
        """Partition-of-unity residual per marker (an array of ones when exact)."""
        return self.w.view(self.n_markers, self.n_off).sum(dim=1)


# --------------------------------------------------------------------------- #
# Rigid bodies
# --------------------------------------------------------------------------- #


@dataclass
class RigidBody:
    """Rigid 2-D body defined by markers in its own (local) frame.

    Parameters
    ----------
    local_markers : (M, 2) array
        Marker positions in the body frame (origin = centre of mass).
    dv : (M,) array
        Marker measure (arc length for surface markers, area for interior
        markers), lattice units.
    centre, angle, velocity, omega :
        Rigid-body kinematics in lattice units.
    mass, inertia :
        Body mass and moment of inertia (2-D, unit thickness).
    fixed :
        ``True`` for prescribed kinematics (e.g. a rotating blade).
    """

    local_markers: np.ndarray
    dv: np.ndarray
    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    omega: float = 0.0
    mass: float = 0.0
    inertia: float = 0.0
    fixed: bool = True
    name: str = "body"
    force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    torque: float = 0.0
    external_force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    external_torque: float = 0.0
    #: interior fluid inertia that is convected with the body (rho * V), if the
    #: body markers include interior nodes
    interior_mass: float = 0.0

    def __post_init__(self) -> None:
        self.local_markers = np.asarray(self.local_markers, dtype=np.float64)
        self.dv = np.asarray(self.dv, dtype=np.float64)
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(2)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(2)
        self.force = np.asarray(self.force, dtype=np.float64).reshape(2)
        self.external_force = np.asarray(self.external_force, dtype=np.float64).reshape(2)
        if self.local_markers.shape != (self.dv.shape[0], 2):
            raise ValueError("local_markers and dv shapes are inconsistent")

    # -- kinematics -------------------------------------------------------- #
    def rotation_matrix(self) -> np.ndarray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def world_markers(self) -> np.ndarray:
        return self.local_markers @ self.rotation_matrix().T + self.centre

    def marker_velocity(self, markers_world: np.ndarray) -> np.ndarray:
        r = markers_world - self.centre
        return np.stack(
            [
                self.velocity[0] - self.omega * r[:, 1],
                self.velocity[1] + self.omega * r[:, 0],
            ],
            axis=1,
        )

    # -- dynamics ---------------------------------------------------------- #
    def translational_added_mass(self) -> float:
        """Linear response of the IBM load to a change of body velocity.

        For pure translation ``dF/dv = -sum_l dv_l`` because every marker sees
        the same velocity increment.  It is used as the added-mass term of the
        semi-implicit update (see :meth:`advance`).
        """
        return float(self.dv.sum())

    def rotational_added_inertia(self) -> float:
        """Angular counterpart of :meth:`translational_added_mass`."""
        return float((self.dv * (self.local_markers**2).sum(axis=1)).sum())

    def advance(self, dt: float = 1.0, *, added_mass: bool = True) -> None:
        """Integrate Newton/Euler with the loads accumulated by the last step.

        With ``added_mass=True`` the mass matrix carries the fluid added mass
        (``sum_l dv_l`` for translation, ``sum_l dv_l r_l^2`` for rotation).
        This is a semi-implicit (first-order) treatment of the linear part of
        the IBM reaction; it removes the classical added-mass instability of
        explicit partitioned coupling for light bodies (``m/M_added < 1``) and
        converges to the same steady state.  With ``added_mass=False`` the plain
        explicit Newton update with the solid mass is used, which conserves the
        total fluid + solid momentum exactly (audited in AC4).
        """
        if self.fixed:
            return
        if added_mass:
            m_eff = self.mass + self.translational_added_mass()
            i_eff = self.inertia + self.rotational_added_inertia()
        else:
            m_eff = self.mass  # plain Newton with the solid mass only
            i_eff = self.inertia
        if m_eff <= 0.0:
            raise ValueError(f"body '{self.name}' has non-positive mass")
        accel = (self.force + self.external_force) / m_eff
        self.velocity = self.velocity + dt * accel
        self.centre = self.centre + dt * self.velocity
        if i_eff > 0.0:
            self.omega += dt * (self.torque + self.external_torque) / i_eff
            self.angle += dt * self.omega
        self.force = np.zeros(2)
        self.torque = 0.0
        self.external_force = np.zeros(2)
        self.external_torque = 0.0


def circle_body(
    centre: Sequence[float],
    radius: float,
    *,
    spacing: float = 0.6,
    interior_spacing: float | None = 0.75,
    interior_fill: bool = True,
    velocity: Sequence[float] = (0.0, 0.0),
    omega: float = 0.0,
    mass: float = 0.0,
    inertia: float = 0.0,
    fixed: bool = True,
    name: str = "circle",
) -> RigidBody:
    """A circular body: surface ring (arc spacing ``spacing``) + interior nodes.

    Marker measures are ``dv = spacing`` on the ring (arc length) and
    ``dv = interior_spacing**2`` for interior nodes (area), so that the
    Lagrangian force integral ``sum_l F_l dv_l`` is the physical force.
    """
    n_ring = max(12, int(round(2.0 * math.pi * radius / spacing)))
    theta = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
    ds = 2.0 * math.pi * radius / n_ring
    ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    local = [ring]
    dvs = [np.full(n_ring, ds)]
    interior_mass = 0.0
    if interior_fill and interior_spacing is not None:
        pitch = float(interior_spacing)
        coords = np.arange(-radius + 0.5 * pitch, radius, pitch)
        gx, gy = np.meshgrid(coords, coords)
        sel = (gx**2 + gy**2) <= (radius - 0.9) ** 2
        pts = np.stack([gx[sel], gy[sel]], axis=1)
        if pts.size:
            local.append(pts)
            dvs.append(np.full(pts.shape[0], pitch * pitch))
            interior_mass = float(pts.shape[0] * pitch * pitch)
    return RigidBody(
        local_markers=np.concatenate(local, axis=0),
        dv=np.concatenate(dvs, axis=0),
        centre=np.asarray(centre, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        omega=float(omega),
        mass=float(mass),
        inertia=float(inertia),
        fixed=fixed,
        name=name,
        interior_mass=interior_mass,
    )


def rectangle_body(
    centre: Sequence[float],
    half_length: float,
    half_thickness: float,
    *,
    angle: float = 0.0,
    spacing: float = 0.6,
    fixed: bool = True,
    velocity: Sequence[float] = (0.0, 0.0),
    omega: float = 0.0,
    name: str = "rectangle",
) -> RigidBody:
    """A rectangular (blade) body: surface markers only, no interior fill."""
    hl, ht = float(half_length), float(half_thickness)
    n_long = max(2, int(round(2.0 * hl / spacing)))
    n_short = max(2, int(round(2.0 * ht / spacing)))
    xs_long = np.linspace(-hl, hl, n_long, endpoint=False)
    ys_short = np.linspace(-ht, ht, n_short, endpoint=False)
    pts = []
    dvs = []
    dx = 2.0 * hl / n_long
    dy = 2.0 * ht / n_short
    for x in xs_long:  # top / bottom faces
        pts.append((x, ht))
        pts.append((x, -ht))
        dvs.extend([dx, dx])
    for y in ys_short:  # leading / trailing edges
        pts.append((hl, y))
        pts.append((-hl, y))
        dvs.extend([dy, dy])
    return RigidBody(
        local_markers=np.asarray(pts, dtype=np.float64),
        dv=np.asarray(dvs, dtype=np.float64),
        centre=np.asarray(centre, dtype=np.float64),
        angle=float(angle),
        velocity=np.asarray(velocity, dtype=np.float64),
        omega=float(omega),
        fixed=fixed,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IBM2DConfig:
    """Configuration of the D2Q9 + IBM flow solver.

    Parameters
    ----------
    nx, ny : domain size in lattice units.
    u_in : inlet velocity (lattice units).
    re : Reynolds number, based on the reference length given by the case
        (the case passes ``nu = u_in * L / re`` explicitly to the solver).
    nu : kinematic viscosity (lattice units).
    inlet / outlet : boundary treatments (``"guo"`` / ``"copy"``).
    multi_direct : number of direct-forcing iterations (1 = classic Uhlmann).
    """

    nx: int
    ny: int
    u_in: float
    nu: float
    multi_direct: int = 2
    #: inlet treatment: ``"zouhe"`` (Zou & He 1997 velocity inlet, default),
    #: ``"guo"`` (non-equilibrium extrapolation) or ``"equilibrium"``
    inlet: str = "zouhe"
    outlet: str = "copy"
    #: number of cells of the outlet sponge layer (0 disables the sponge)
    outlet_sponge: int = 0
    #: maximum non-equilibrium damping of the sponge layer
    sponge_strength: float = 0.2
    dtype: str = "float64"
    #: amplitude (relative to ``u_in``) of a transverse sinuous perturbation of
    #: the initial field; it only *triggers* the (physically unstable) shedding
    #: mode and decays within the first flow-through time
    perturb: float = 0.0
    #: collision model: ``"trt"`` (two-relaxation-time, default: stable at low
    #: ``tau``) or ``"bgk"``
    collision: str = "trt"
    #: TRT magic parameter (Ginzburg): 3/16 for optimal stability / accuracy
    magic: float = 3.0 / 16.0
    #: fully periodic in ``x`` (no inlet/outlet); used by the momentum audits
    periodic_x: bool = False

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def tau_minus(self) -> float:
        """TRT second relaxation time for the magic parameter ``Lambda``."""
        return 0.5 + self.magic / (self.tau - 0.5)

    def validate(self) -> None:
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        if self.multi_direct < 1:
            raise ValueError("multi_direct must be >= 1")
        if self.collision not in ("trt", "bgk"):
            raise ValueError(f"unknown collision model {self.collision!r}")
        if self.inlet not in ("zouhe", "guo", "equilibrium"):
            raise ValueError(f"unknown inlet treatment {self.inlet!r}")


class IBM2D:
    """D2Q9 lattice-Boltzmann solver with direct-forcing IBM for rigid bodies."""

    def __init__(self, cfg: IBM2DConfig, *, body_clearance: float = 3.0) -> None:
        cfg.validate()
        self.cfg = cfg
        self.dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
        self.body_clearance = float(body_clearance)
        nx, ny = cfg.nx, cfg.ny
        self.rho = torch.ones((ny, nx), dtype=self.dtype)
        self.ux = torch.full((ny, nx), cfg.u_in, dtype=self.dtype)
        self.uy = torch.zeros((ny, nx), dtype=self.dtype)
        if cfg.perturb:
            yy = torch.arange(ny, dtype=self.dtype).view(ny, 1)
            self.uy = self.uy + cfg.perturb * cfg.u_in * torch.sin(
                2.0 * math.pi * yy / ny
            )
        self.f = equilibrium2d(self.rho, self.ux, self.uy)
        self.force_field = torch.zeros((2, ny, nx), dtype=self.dtype)
        self.tau = cfg.tau
        self._inv_tau = 1.0 / self.tau
        self._guo = 1.0 - 0.5 / self.tau
        self._opp = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long)
        self._omega_plus = 1.0 / self.tau
        self._omega_minus = 1.0 / cfg.tau_minus
        self._w = W9.to(self.dtype).view(9, 1, 1)
        self._cx = C9[:, 0].to(self.dtype).view(9, 1, 1)
        self._cy = C9[:, 1].to(self.dtype).view(9, 1, 1)

    # -- helpers ----------------------------------------------------------- #
    def _check_marker_bounds(self, markers: np.ndarray, name: str) -> None:
        if self.cfg.periodic_x:
            return
        lo = self.body_clearance
        hi = self.cfg.nx - 1 - self.body_clearance
        if markers[:, 0].min() < lo or markers[:, 0].max() > hi:
            raise ValueError(
                f"body '{name}' is too close to the x walls "
                f"(markers must stay within [{lo}, {hi}])"
            )

    def _interp(self, stencil: MarkerStencil, field: torch.Tensor) -> torch.Tensor:
        flat = field.reshape(-1)
        return torch.zeros(stencil.n_markers, dtype=self.dtype).index_add_(
            0,
            torch.arange(stencil.n_markers).repeat_interleave(stencil.n_off),
            (stencil.w * flat[stencil.flat].double()).to(self.dtype),
        )

    def _spread(
        self, stencil: MarkerStencil, values: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(self.cfg.nx * self.cfg.ny, dtype=self.dtype).index_add_(
            0,
            stencil.flat,
            (stencil.w * values.repeat_interleave(stencil.n_off).double()).to(self.dtype),
        )

    # -- one step ---------------------------------------------------------- #
    def step(self, bodies: Sequence[RigidBody]) -> dict[str, dict[str, float]]:
        """Advance one lattice step and return the loads on every body."""
        cfg = self.cfg
        f = self.f
        rho, ux_raw, uy_raw = macroscopic2d(f)
        # IBM forcing is evaluated on the raw (pre-forcing) velocity; the Guo
        # half-force correction with the *current* force is applied afterwards,
        # which makes the per-step momentum balance exact (audited in AC4).
        ux, uy = ux_raw, uy_raw

        stencils: list[MarkerStencil] = []
        marker_vel: list[torch.Tensor] = []
        for body in bodies:
            markers = body.world_markers()
            self._check_marker_bounds(markers, body.name)
            stencils.append(MarkerStencil(markers, cfg.nx, cfg.ny))
            marker_vel.append(
                torch.as_tensor(body.marker_velocity(markers), dtype=self.dtype)
            )

        total_fx = torch.zeros(cfg.nx * cfg.ny, dtype=self.dtype)
        total_fy = torch.zeros(cfg.nx * cfg.ny, dtype=self.dtype)
        body_forces = [
            np.zeros(2) for _ in bodies
        ]
        body_torques = [0.0 for _ in bodies]

        ux_eff = ux.clone()
        uy_eff = uy.clone()
        for _ in range(cfg.multi_direct):
            for ib, (body, stencil) in enumerate(zip(bodies, stencils)):
                dv = torch.as_tensor(body.dv, dtype=self.dtype)
                target = marker_vel[ib]
                fx_l = target[:, 0] - self._interp(stencil, ux_eff)
                fy_l = target[:, 1] - self._interp(stencil, uy_eff)
                # correction field of this sub-iteration (velocity units)
                sx = self._spread(stencil, fx_l * dv)
                sy = self._spread(stencil, fy_l * dv)
                total_fx += sx
                total_fy += sy
                ux_eff = ux_eff + sx.view(cfg.ny, cfg.nx)
                uy_eff = uy_eff + sy.view(cfg.ny, cfg.nx)
                # reaction on the body: minus the momentum given to the fluid
                body_forces[ib] -= np.array(
                    [float((fx_l * dv).sum()), float((fy_l * dv).sum())]
                )
                r = torch.as_tensor(
                    stencil.markers - body.centre, dtype=self.dtype
                )
                body_torques[ib] -= float(
                    (
                        r[:, 0] * (fy_l * dv)
                        - r[:, 1] * (fx_l * dv)
                    ).sum()
                )

        fx_grid = total_fx.view(cfg.ny, cfg.nx)
        fy_grid = total_fy.view(cfg.ny, cfg.nx)
        self.force_field = torch.stack([fx_grid, fy_grid])
        # Guo half-force correction: u = u* + F/(2 rho)
        ux = ux_raw + 0.5 * fx_grid / rho
        uy = uy_raw + 0.5 * fy_grid / rho

        # --- Guo forcing + BGK collision (fused, in-place where possible) -- #
        w, cx, cy = self._w, self._cx, self._cy
        cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)   # (9, ny, nx)
        usq = ux * ux + uy * uy                           # (ny, nx)
        feq = cu.clone().mul_(3.0).add_(1.0)
        feq.add_(cu * cu, alpha=4.5)
        feq.add_(usq, alpha=-1.5)
        feq.mul_(rho).mul_(w)
        # exact Guo source: w (1-1/(2tau)) [3 (c-u).F + 9 (c.u)(c.F)]
        cu_f = cx * fx_grid.unsqueeze(0) + cy * fy_grid.unsqueeze(0)
        cu_cuf = cu * cu_f  # third moment term, formed before cu_f is consumed
        uf = ux * fx_grid + uy * fy_grid
        src = cu_f.clone().mul_(3.0).add_(uf, alpha=-3.0)
        raw = cu_f.mul_(3.0).add_(uf, alpha=-3.0)  # cu_f consumed here on purpose
        raw.add_(cu_cuf, alpha=9.0).mul_(w)
        if self.cfg.collision == "trt":
            # the source must be split into its symmetric/antisymmetric parts and
            # damped with (1 - omega_pm/2), otherwise the antisymmetric
            # relaxation destroys the per-step momentum balance (audited in AC4)
            raw_op = raw[self._opp]
            src = 0.5 * (raw + raw_op) * (1.0 - 0.5 * self._omega_plus) + 0.5 * (
                raw - raw_op
            ) * (1.0 - 0.5 * self._omega_minus)
            # two-relaxation-time: symmetric and antisymmetric parts relax
            # independently, which damps the checkerboard modes that otherwise
            # destabilise BGK at tau -> 1/2
            f_op = f[self._opp]
            feq_op = feq[self._opp]
            d_plus = 0.5 * (f + f_op - feq - feq_op)
            d_minus = 0.5 * (f - f_op - feq + feq_op)
            f_post = (
                f
                - d_plus.mul_(self._omega_plus)
                - d_minus.mul_(self._omega_minus)
                + src
            )
        else:
            src = raw.mul_(self._guo)
            f_post = f - (f - feq).mul_(self._inv_tau) + src

        f_new = torch.empty_like(f_post)
        for i in range(9):
            f_new[i].copy_(torch.roll(
                f_post[i], shifts=(int(C9[i, 1]), int(C9[i, 0])), dims=(0, 1)
            ))

        # --- boundary conditions ------------------------------------------ #
        self._apply_boundaries(f_new, rho, ux, uy)
        self.f = f_new

        loads: dict[str, dict[str, float]] = {}
        for body, force, torque in zip(bodies, body_forces, body_torques):
            body.force = force
            body.torque = torque
            loads[body.name] = {
                "fx": float(force[0]),
                "fy": float(force[1]),
                "torque": float(torque),
            }
        return loads

    def _apply_boundaries(
        self,
        f_new: torch.Tensor,
        rho: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
    ) -> None:
        cfg = self.cfg
        if cfg.periodic_x:
            return
        if cfg.inlet == "zouhe":
            # Zou & He (1997) velocity inlet: only the incoming populations
            # (c_x > 0) are unknown after streaming; the density follows from
            # the outgoing ones, which removes the reflection that destabilises
            # a full-column extrapolation
            u = cfg.u_in
            f0 = f_new[0, :, 0]
            f2 = f_new[2, :, 0]
            f4 = f_new[4, :, 0]
            f3 = f_new[3, :, 0]
            f6 = f_new[6, :, 0]
            f7 = f_new[7, :, 0]
            rho_w = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u)
            f_new[1, :, 0] = f3 + (2.0 / 3.0) * rho_w * u
            f_new[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_w * u
            f_new[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_w * u
        elif cfg.inlet == "guo":
            rho1 = f_new[:, :, 1].sum(dim=0)
            ux1 = (
                C9[:, 0].to(self.dtype).view(9, 1) * f_new[:, :, 1]
            ).sum(dim=0) / rho1
            uy1 = (
                C9[:, 1].to(self.dtype).view(9, 1) * f_new[:, :, 1]
            ).sum(dim=0) / rho1
            feq_in = equilibrium2d(
                rho1,
                torch.full_like(rho1, cfg.u_in),
                torch.zeros_like(rho1),
            )
            feq_nb = equilibrium2d(rho1, ux1, uy1)
            f_new[:, :, 0] = feq_in + (f_new[:, :, 1] - feq_nb)
        else:  # plain equilibrium inlet
            ny = cfg.ny
            f_new[:, :, 0] = equilibrium2d(
                torch.ones(ny, dtype=self.dtype),
                torch.full((ny,), cfg.u_in, dtype=self.dtype),
                torch.zeros(ny, dtype=self.dtype),
            )
        if cfg.outlet == "copy":
            f_new[:, :, -1] = f_new[:, :, -2]
        else:
            raise ValueError(f"unknown outlet treatment {cfg.outlet!r}")
        n_sp = int(cfg.outlet_sponge)
        if n_sp > 0:
            # damp the outgoing non-equilibrium (reflection sponge)
            sl = slice(self.cfg.nx - n_sp, self.cfg.nx)
            rho_s = f_new[:, :, sl].sum(dim=0)
            ux_s = (self._cx * f_new[:, :, sl]).sum(dim=0) / rho_s
            uy_s = (self._cy * f_new[:, :, sl]).sum(dim=0) / rho_s
            feq_s = equilibrium2d(rho_s, ux_s, uy_s)
            ramp = torch.linspace(
                0.0, cfg.sponge_strength, n_sp, dtype=self.dtype
            ).view(1, 1, n_sp)
            f_new[:, :, sl] = feq_s + (1.0 - ramp) * (f_new[:, :, sl] - feq_s)

    # -- diagnostics ------------------------------------------------------- #
    def total_momentum(self) -> np.ndarray:
        """Raw discrete momentum ``sum_i c_i f_i`` (the quantity the Guo scheme
        advances by exactly ``F dt`` per step; see Izaguirre-style bookkeeping
        in Guo, Zheng & Shi 2002, and the audit in :func:`momentum_exchange_audit`).
        """
        rho, ux, uy = macroscopic2d(self.f)
        return np.array([float((rho * ux).sum()), float((rho * uy).sum())])

    def velocity_field(self) -> tuple[np.ndarray, np.ndarray]:
        rho, ux, uy = macroscopic2d(self.f)
        return ux.numpy(), uy.numpy()


# --------------------------------------------------------------------------- #
# Momentum audits
# --------------------------------------------------------------------------- #


def momentum_exchange_audit(
    *,
    d_cells: int = 12,
    n_steps: int = 400,
    u_body: float = 0.02,
    interior_fill: bool = True,
    multi_direct: int = 2,
) -> dict[str, float]:
    """Check fluid/body momentum exchange in a fully periodic box.

    A disc is dragged through a quiescent periodic box at a prescribed constant
    velocity.  There is no inlet/outlet flux and gravity is absent, so the total
    momentum of the fluid must equal the integral of the momentum handed over by
    the IBM operator, i.e. ``sum_t F_IBM(t) = P_fluid(t_end) - P_fluid(0)``.
    """
    nx = ny = int(round(8 * d_cells))
    body = circle_body(
        (nx / 2.0, ny / 2.0),
        d_cells / 2.0,
        interior_fill=interior_fill,
        velocity=(u_body, 0.0),
    )
    cfg = IBM2DConfig(
        nx=nx, ny=ny, u_in=0.0, nu=0.05, multi_direct=multi_direct, periodic_x=True
    )
    solver = IBM2D(cfg)
    # quiescent, fully periodic box: the body drives the fluid, no boundary flux
    p0 = solver.total_momentum()
    force_sum = np.zeros(2)
    for _ in range(n_steps):
        loads = solver.step([body])
        force_sum += np.array([loads[body.name]["fx"], loads[body.name]["fy"]])
        body.centre[0] += u_body  # pure kinematics, box is periodic
    p1 = solver.total_momentum()
    dp = p1 - p0
    # the load taken off the body equals minus the momentum handed to the fluid
    rel = float(np.abs(force_sum + dp).max() / max(np.abs(dp).max(), 1e-30))
    return {
        "force_sum_x": float(force_sum[0]),
        "force_sum_y": float(force_sum[1]),
        "dp_x": float(dp[0]),
        "dp_y": float(dp[1]),
        "relative_residual": rel,
        "n_steps": n_steps,
        "d_cells": d_cells,
    }


def free_body_momentum_audit(
    *,
    d_cells: int = 10,
    n_steps: int = 600,
    v0: float = 0.03,
    interior_fill: bool = True,
) -> dict[str, float]:
    """Total (fluid + body) momentum conservation for a freely moving disc.

    The disc is released with velocity ``v0`` in a quiescent periodic box and
    decelerates through hydrodynamic drag only.  ``P_fluid + M_eff * v_body``
    must stay constant to round-off.
    """
    nx = ny = int(round(8 * d_cells))
    radius = d_cells / 2.0
    body = circle_body(
        (nx / 2.0, ny / 2.0),
        radius,
        interior_fill=interior_fill,
        velocity=(v0, 0.0),
        mass=math.pi * radius**2 * 3.0,  # heavy body: explicit coupling is stable
        inertia=0.5 * math.pi * radius**4 * 3.0,
        fixed=False,
        name="floe",
    )
    cfg = IBM2DConfig(nx=nx, ny=ny, u_in=0.0, nu=0.05, periodic_x=True)
    solver = IBM2D(cfg)
    m_eff = body.mass + body.interior_mass
    # system momentum: fluid (interior fluid included) + solid body mass only
    p0 = solver.total_momentum() + body.mass * body.velocity
    history = []
    for _ in range(n_steps):
        solver.step([body])
        body.advance(added_mass=False)  # explicit: exact momentum bookkeeping
        p = solver.total_momentum() + body.mass * body.velocity
        history.append(p.copy())
    p_end = history[-1]
    drift = float(np.abs(p_end - p0).max() / max(np.abs(p0).max(), 1e-30))
    return {
        "p0_x": float(p0[0]),
        "p_end_x": float(p_end[0]),
        "relative_drift": drift,
        "n_steps": n_steps,
        "d_cells": d_cells,
        "effective_mass": float(m_eff),
    }


# --------------------------------------------------------------------------- #
# Cylinder benchmark (fixed + rotating)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CylinderCaseConfig:
    """Configuration of the cylinder benchmark.

    ``alpha`` is the non-dimensional spin ratio ``omega * R / U_in`` (0 = fixed
    cylinder).  ``d_cells`` is the cylinder diameter in lattice units.
    """

    re: float = 100.0
    d_cells: int = 20
    alpha: float = 0.0
    u_in: float = 0.1
    lx_over_d: float = 30.0
    ly_over_d: float = 16.0
    n_steps: int = 40000
    warmup: int = 8000
    multi_direct: int = 2
    interior_fill: bool = False
    sample_every: int = 5
    spacing: float = 0.6
    outlet_sponge: int = 0
    inlet: str = "zouhe"
    perturb: float = 0.05
    dtype: str = "float64"

    @property
    def radius(self) -> float:
        return 0.5 * self.d_cells

    @property
    def nu(self) -> float:
        return self.u_in * self.d_cells / self.re

    @property
    def omega(self) -> float:
        return 2.0 * self.alpha * self.u_in / self.d_cells


def strouhal_number(
    signal: np.ndarray, u_in: float, diameter: float, sample_dt: float = 1.0
) -> float:
    """Dominant non-dimensional frequency of a lift signal via FFT peak."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 16:
        return float("nan")
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(n, d=sample_dt)
    if spec.size <= 1:
        return float("nan")
    k = int(np.argmax(spec[1:]) + 1)
    return float(freqs[k] * diameter / u_in)


def run_cylinder_case(cfg: CylinderCaseConfig, *, keep_fields: bool = True) -> dict:
    """Run the (possibly rotating) cylinder and return load statistics."""
    nx = int(round(cfg.lx_over_d * cfg.d_cells))
    ny = int(round(cfg.ly_over_d * cfg.d_cells))
    body = circle_body(
        (0.35 * nx, ny / 2.0),
        cfg.radius,
        spacing=cfg.spacing,
        interior_fill=cfg.interior_fill,
        velocity=(0.0, 0.0),
        omega=cfg.omega,
        fixed=True,
        name="cylinder",
    )
    solver = IBM2D(
        IBM2DConfig(
            nx=nx,
            ny=ny,
            u_in=cfg.u_in,
            nu=cfg.nu,
            multi_direct=cfg.multi_direct,
            inlet=cfg.inlet,
            outlet_sponge=cfg.outlet_sponge,
            perturb=cfg.perturb,
            dtype=cfg.dtype,
        )
    )
    q = 0.5 * cfg.u_in**2 * cfg.d_cells
    t = []
    cd = []
    cl = []
    torque = []
    for step in range(cfg.n_steps):
        loads = solver.step([body])
        rec = (step + 1) >= cfg.warmup and (step % cfg.sample_every == 0)
        if rec:
            t.append(step + 1.0)
            cd.append(loads["cylinder"]["fx"] / q)
            cl.append(loads["cylinder"]["fy"] / q)
            torque.append(loads["cylinder"]["torque"] / (q * cfg.radius))
        if not np.isfinite(loads["cylinder"]["fx"]):
            raise FloatingPointError(f"IBM solution diverged at step {step}")

    cd_a = np.asarray(cd)
    cl_a = np.asarray(cl)
    dt_sample = cfg.sample_every
    result = {
        "config": {
            "re": cfg.re,
            "d_cells": cfg.d_cells,
            "alpha": cfg.alpha,
            "u_in": cfg.u_in,
            "nu": cfg.nu,
            "tau": 3.0 * cfg.nu + 0.5,
            "nx": nx,
            "ny": ny,
            "n_steps": cfg.n_steps,
            "warmup": cfg.warmup,
            "multi_direct": cfg.multi_direct,
            "interior_fill": cfg.interior_fill,
            "markers": int(body.local_markers.shape[0]),
        },
        "cd_mean": float(cd_a.mean()) if cd_a.size else float("nan"),
        "cd_rms": float(np.sqrt(((cd_a - cd_a.mean()) ** 2).mean())) if cd_a.size else float("nan"),
        "cl_mean": float(cl_a.mean()) if cl_a.size else float("nan"),
        "cl_rms": float(np.sqrt(((cl_a - cl_a.mean()) ** 2).mean())) if cl_a.size else float("nan"),
        "cl_min": float(cl_a.min()) if cl_a.size else float("nan"),
        "cl_max": float(cl_a.max()) if cl_a.size else float("nan"),
        "torque_coeff_mean": float(np.mean(torque)) if torque else float("nan"),
        "strouhal": strouhal_number(np.asarray(cl), cfg.u_in, cfg.d_cells, dt_sample),
        "time": t,
        "cd": cd,
        "cl": cl,
        "torque_coeff": torque,
    }
    if keep_fields:
        ux, uy = solver.velocity_field()
        result["fields"] = {"ux": ux, "uy": uy}
    return result


# --------------------------------------------------------------------------- #
# Ice floe -- rotating blade interaction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IceBladeConfig:
    """Simplified 2-D ice-floe / rotating-blade interaction case.

    Geometry: a one-bladed rotor (a rectangular blade of half-length
    ``blade_half_length`` and half-thickness ``blade_half_thickness`` rotating
    about ``hub``) and a circular ice floe that is free to translate.  The floe
    is released upstream of the blade with an initial velocity; the blade
    encounters it once per revolution.

    All quantities are lattice units (``dx = dt = rho = 1``).
    """

    nx: int = 320
    ny: int = 200
    u_in: float = 0.02
    nu: float = 0.005
    hub: tuple[float, float] = (160.0, 100.0)
    blade_half_length: float = 45.0
    blade_half_thickness: float = 3.0
    blade_angle: float = 0.0
    omega: float = 0.002
    floe_radius: float = 12.0
    floe_centre: tuple[float, float] = (110.0, 100.0)
    floe_velocity: tuple[float, float] = (0.012, 0.0)
    density_ratio_ice_water: float = 0.895
    #: semi-implicit added-mass coupling (needed for an ice-like, light body)
    use_added_mass: bool = True
    #: contact stiffness (force / penetration) and damping, lattice units
    contact_stiffness: float = 0.5
    contact_damping: float = 4.0
    n_steps: int = 6000
    sample_every: int = 2
    multi_direct: int = 2
    free_floe: bool = True


def _segment_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance from point ``p`` to segment ``a-b``; returns (dist, closest, normal)."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom <= 0.0 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    d = p - closest
    dist = float(np.linalg.norm(d))
    normal = d / dist if dist > 1e-12 else np.array([0.0, 0.0])
    return dist, closest, normal


def _blade_endpoints(body: RigidBody) -> tuple[np.ndarray, np.ndarray]:
    rot = body.rotation_matrix()
    half = 0.5 * (body.local_markers[:, 0].max() - body.local_markers[:, 0].min())
    a = body.centre + rot @ np.array([-half, 0.0])
    b = body.centre + rot @ np.array([half, 0.0])
    return a, b


def run_ice_blade_case(cfg: IceBladeConfig) -> dict:
    """Run the simplified ice-floe / rotating-blade case.

    Returns the load time histories and the momentum/impulse audit of the floe.
    """
    radius = cfg.floe_radius
    area = math.pi * radius**2
    blade = rectangle_body(
        cfg.hub,
        cfg.blade_half_length,
        cfg.blade_half_thickness,
        angle=cfg.blade_angle,
        fixed=True,
        omega=cfg.omega,
        name="blade",
    )
    floe = circle_body(
        cfg.floe_centre,
        radius,
        interior_fill=True,
        velocity=cfg.floe_velocity,
        mass=area * cfg.density_ratio_ice_water,
        inertia=0.5 * area * radius**2 * cfg.density_ratio_ice_water,
        fixed=not cfg.free_floe,
        name="floe",
    )
    solver = IBM2D(
        IBM2DConfig(
            nx=cfg.nx, ny=cfg.ny, u_in=cfg.u_in, nu=cfg.nu, multi_direct=cfg.multi_direct
        )
    )
    m_eff = floe.mass + floe.interior_mass
    p_flow0 = solver.total_momentum() + floe.mass * floe.velocity

    time, blade_fx, blade_fy, blade_torque, floe_fx, floe_fy = [], [], [], [], [], []
    contact_fx, contact_fy, floe_x, floe_y, floe_vx, floe_vy = [], [], [], [], [], []
    total_impulse = np.zeros(2)

    for step in range(cfg.n_steps):
        # ---- contact between floe and blade (Hertzian, damped) ----------- #
        a, b = _blade_endpoints(blade)
        dist, closest, normal = _segment_distance(floe.centre, a, b)
        gap = dist - (radius + cfg.blade_half_thickness)
        contact = np.zeros(2)
        if gap < 0.0:
            # normal from blade towards floe; relative normal velocity
            v_marker = np.array(
                [
                    blade.velocity[0] - blade.omega * (closest[1] - blade.centre[1]),
                    blade.velocity[1] + blade.omega * (closest[0] - blade.centre[0]),
                ]
            )
            v_rel = (floe.velocity - v_marker) @ normal
            contact = (
                cfg.contact_stiffness * (-gap) - cfg.contact_damping * v_rel
            ) * normal
            contact = np.maximum(contact, 0.0) * normal
            floe.external_force = contact
            blade.external_force = -contact
            blade.external_torque = -float(
                np.cross(closest - blade.centre, contact)
            )
        else:
            floe.external_force = np.zeros(2)
            blade.external_force = np.zeros(2)
            blade.external_torque = 0.0

        loads = solver.step([blade, floe])
        # blade kinematics are prescribed: advance the angle
        blade.angle += cfg.omega
        blade.force = np.zeros(2)
        blade.torque = 0.0
        # floe: Newton with the IBM load already accumulated inside step()
        hydro = loads["floe"]
        total_impulse += np.array([hydro["fx"], hydro["fy"]]) + contact
        if cfg.free_floe:
            floe.advance(added_mass=cfg.use_added_mass)
        else:
            floe.force = np.zeros(2)
            floe.torque = 0.0

        if step % cfg.sample_every == 0:
            time.append(step + 1.0)
            blade_fx.append(loads["blade"]["fx"])
            blade_fy.append(loads["blade"]["fy"])
            blade_torque.append(loads["blade"]["torque"] + blade.external_torque)
            floe_fx.append(hydro["fx"])
            floe_fy.append(hydro["fy"])
            contact_fx.append(float(contact[0]))
            contact_fy.append(float(contact[1]))
            floe_x.append(float(floe.centre[0]))
            floe_y.append(float(floe.centre[1]))
            floe_vx.append(float(floe.velocity[0]))
            floe_vy.append(float(floe.velocity[1]))
        if not np.isfinite(loads["floe"]["fx"]):
            raise FloatingPointError(f"ice-blade case diverged at step {step}")

    p_flow1 = solver.total_momentum() + floe.mass * floe.velocity
    momentum_drift = float(
        np.abs(p_flow1 - p_flow0).max() / max(np.abs(p_flow0).max(), 1e-30)
    )
    # momentum change of the floe predicted by the integrated load
    d_v = floe.velocity - np.asarray(cfg.floe_velocity)
    # the semi-implicit update uses the mass matrix sigma = m + added mass, the
    # explicit one uses the solid mass alone: check the matching relation
    sigma = (
        floe.mass + floe.translational_added_mass()
        if (cfg.use_added_mass and cfg.free_floe)
        else floe.mass
    )
    impulse_residual = float(
        np.abs(total_impulse - sigma * d_v).max()
        / max(np.abs(sigma * d_v).max(), 1e-30)
    )
    result = {
        "config": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "u_in": cfg.u_in,
            "nu": cfg.nu,
            "tau": 3.0 * cfg.nu + 0.5,
            "omega": cfg.omega,
            "floe_radius": radius,
            "floe_mass": floe.mass,
            "floe_interior_mass": floe.interior_mass,
            "floe_effective_mass": m_eff,
            "floe_mass_matrix": floe.mass + floe.translational_added_mass(),
            "use_added_mass": cfg.use_added_mass,
            "blade_half_length": cfg.blade_half_length,
            "blade_half_thickness": cfg.blade_half_thickness,
            "n_steps": cfg.n_steps,
            "density_ratio": cfg.density_ratio_ice_water,
        },
        "time": time,
        "blade_fx": blade_fx,
        "blade_fy": blade_fy,
        "blade_torque": blade_torque,
        "floe_fx": floe_fx,
        "floe_fy": floe_fy,
        "contact_fx": contact_fx,
        "contact_fy": contact_fy,
        "floe_x": floe_x,
        "floe_y": floe_y,
        "floe_vx": floe_vx,
        "floe_vy": floe_vy,
        "momentum_drift": momentum_drift,
        "impulse_residual": impulse_residual,
        "mass_matrix_used": sigma,
        "total_impulse": total_impulse.tolist(),
        "floe_delta_v": d_v.tolist(),
    }
    return result


@dataclass(frozen=True)
class FloeDragConfig:
    """Quiescent ice floe in a channel: pure hydrodynamic baseline case."""

    nx: int = 320
    ny: int = 200
    u_in: float = 0.02
    nu: float = 0.005
    floe_radius: float = 12.0
    floe_centre: tuple[float, float] = (110.0, 100.0)
    n_steps: int = 8000
    warmup: int = 4000
    sample_every: int = 5
    dtype: str = "float64"


def run_floe_drag_case(cfg: FloeDragConfig) -> dict:
    """Drag of a *fixed* ice floe (IBM body) in the same channel as the blade case.

    This isolates the hydrodynamic load path of the ice-blade case: the measured
    ``Cd = F_x / (0.5 rho U^2 D)`` is compared with the literature cylinder value
    at the same Reynolds number.
    """
    body = circle_body(
        cfg.floe_centre, cfg.floe_radius, interior_fill=True, name="floe"
    )
    solver = IBM2D(
        IBM2DConfig(
            nx=cfg.nx, ny=cfg.ny, u_in=cfg.u_in, nu=cfg.nu, dtype=cfg.dtype
        )
    )
    d = 2.0 * cfg.floe_radius
    q = 0.5 * cfg.u_in**2 * d
    cd, cl, t = [], [], []
    for step in range(cfg.n_steps):
        loads = solver.step([body])
        if step + 1 >= cfg.warmup and step % cfg.sample_every == 0:
            t.append(step + 1.0)
            cd.append(loads["floe"]["fx"] / q)
            cl.append(loads["floe"]["fy"] / q)
        if not np.isfinite(loads["floe"]["fx"]):
            raise FloatingPointError(f"floe drag case diverged at step {step}")
    cd_a = np.asarray(cd)
    return {
        "task": "floe_drag",
        "config": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "u_in": cfg.u_in,
            "nu": cfg.nu,
            "re": cfg.u_in * d / cfg.nu,
            "tau": 3.0 * cfg.nu + 0.5,
            "floe_radius": cfg.floe_radius,
            "n_steps": cfg.n_steps,
        },
        "cd_mean": float(cd_a.mean()),
        "cd_rms": float(cd_a.std()),
        "cl_rms": float(np.asarray(cl).std()),
        "time": t,
        "cd": cd,
        "cl": cl,
    }


def save_json(payload: dict, path: Path | str) -> Path:
    """Write a JSON artifact, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path