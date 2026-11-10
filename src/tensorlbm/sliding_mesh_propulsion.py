"""Sliding-mesh (rotating-zone) LBM solver for a propeller in a static zone.

Motivation
----------
The repository already ships a *sliding-mesh interface primitive*
(:mod:`tensorlbm.sliding_mesh_common`, ``apply_sliding_mesh_bc_3d``) but its
capability contract withholds the coupled-physics claim
(``WITHHELD_NO_COUPLED_SLIDING_MESH_PHYSICS_CONTRACT``): no solver ever drove a
rotating zone with a real geometry across the interface and measured the
resulting loads.  This module closes that gap with a two-zone solver:

* **inner (rotor) zone** -- solved in the frame rotating with angular velocity
  ``Omega`` about the shaft axis ``x``.  The propeller blades are rigidly
  attached to this frame, hence *stationary* in it; no per-step mask rotation
  and no moving-wall bounce-back are required.
* **outer (static) zone** -- ordinary inertial-frame LBM (hull / duct / tank).
* **sliding interface** -- a cylindrical shell at ``r = R_z``.  Each step the
  macroscopic state is exchanged across the shell with

  1. a *frame transformation* ``u_lab = u_rel + Omega x r``,
  2. *azimuthal interpolation* by the accumulated slip angle ``Theta``
     (the zone has physically advanced by ``Theta`` relative to the static
     grid -- this is the ``sliding`` part), and
  3. reconstruction of the *incoming* populations only, so the outgoing
     characteristics of each zone are untouched.

Kinematics in the rotating frame
--------------------------------
The incompressible rotating-frame momentum equation is

    du/dt + (u.grad)u = -grad(p)/rho + nu lap(u) - 2 Omega x u
                        - Omega x (Omega x r)

so the momentum source injected by the Guo scheme is

    F_x = 0,   F_y = rho (2 Omega u_z + Omega^2 dy),   F_z = rho (-2 Omega u_y + Omega^2 dz)

with ``(dy, dz)`` measured from the shaft.  For rotation about ``x`` the axial
force and the shaft torque are frame invariant (``Omega x r`` has no ``x``
component and the fictitious forces carry no axial force and no moment about
``x``), so the momentum-exchange loads measured on the blades *are* the
physical thrust and torque.

Evidence grading
----------------
REAL (verified by ``tests/test_sliding_mesh_propulsion.py``):
    interface transparency for a uniform stream, interface mass/momentum
    budget, rotating-frame solid-body rotation, Taylor--Couette profile across
    the interface (analytic), blade momentum-exchange loads, open-water sweep.
MODEL (documented):
    macroscopically reconstructed interface exchange (``f_eq`` for the incoming
    characteristics) rather than raw population copying; polar-shell radial
    binning over one lattice cell; the lattice Reynolds number is orders of
    magnitude below model-scale propeller Re (see the case module).

References
----------
* Latt, J. et al. (2021) *Palabos*, Comput. Math. Appl. 81, 334-350.
* Krause, M. et al. (2017) *Fluid flow simulation and optimisation with LBM*.
* Guo, Z., Zheng, C., Shi, B. (2002) *Phys. Fluids* 14, 385.
* Ladd, A. J. C. (1994) *J. Fluid Mech.* 271, 285 (momentum-exchange loads).
* Hou, S. et al. (1996) *J. Comput. Phys.* 128, 1 (Smagorinsky in LBM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .d3q19 import OPPOSITE, C, W, equilibrium3d, macroscopic3d
from .propeller_cad import PropellerGeometryConfig, build_propeller_mask
from .solver3d import stream3d

__all__ = [
    "RotatingZoneConfig",
    "PolarShell",
    "build_polar_shell",
    "SlidingMeshPropeller",
    "taylor_couette_velocity",
    "run_taylor_couette",
    "open_water_curve",
]

_C = C.to(torch.float32)
_OPP = OPPOSITE.to(torch.int64)
_W = W.to(torch.float32)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RotatingZoneConfig:
    """Configuration of the two-zone sliding-mesh propeller solver.

    All lengths are lattice units, all velocities lattice units / step.

    Parameters
    ----------
    nx, ny, nz:
        Grid size; ``x`` is the shaft / streamwise axis.
    inflow:
        Free-stream velocity fed at the ``x = 0`` plane.
    omega:
        Angular velocity of the rotor zone about ``+x`` in rad/step.
    tau:
        BGK relaxation time (molecular).
    smagorinsky:
        Smagorinsky constant; ``0`` disables the eddy viscosity.
    zone_radius:
        Radius of the sliding interface in lattice units (must exceed the
        propeller radius).
    n_theta:
        Number of azimuthal bins of the sliding interface.
    interface_width:
        Radial thickness (lattice units) of the shell taken on *each* side.
    sides:
        ``"free_slip"`` (open-water tank) or ``"solid"`` (no-slip box walls).
    streamwise:
        ``"inlet_outlet"`` (uniform inflow, zero-gradient outflow) or
        ``"closed"`` (no-slip end walls, e.g. Taylor--Couette).
    """

    nx: int = 96
    ny: int = 96
    nz: int = 96
    inflow: float = 0.05
    omega: float = 0.0
    tau: float = 0.51
    smagorinsky: float = 0.0
    zone_radius: float = 24.0
    n_theta: int = 72
    interface_width: float = 1.5
    sides: str = "free_slip"
    streamwise: str = "inlet_outlet"
    slip_planes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8 or self.nz < 8:
            raise ValueError("grid must be at least 8 cells in each direction")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        # ``zone_radius < 4`` yields an empty sliding shell, which is the supported
        # way to run a *static-zone only* case (e.g. the hull runs of the KVLCC2
        # case); only a non-positive radius is a configuration error.
        if self.zone_radius < 1.0:
            raise ValueError("zone_radius must be at least 1 lattice unit")
        if self.n_theta < 8:
            raise ValueError("n_theta must be at least 8")
        if self.sides not in ("free_slip", "solid"):
            raise ValueError("sides must be 'free_slip' or 'solid'")
        if self.streamwise not in ("inlet_outlet", "closed"):
            raise ValueError("streamwise must be 'inlet_outlet' or 'closed'")
        allowed = {"y_min", "y_max", "z_min", "z_max"}
        if not set(self.slip_planes) <= allowed:
            raise ValueError(f"slip_planes must be a subset of {sorted(allowed)}")

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0


# --------------------------------------------------------------------------- #
# Sliding interface: polar shell bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class PolarShell:
    """Pre-computed bookkeeping of the cylindrical sliding interface.

    The shell is split into two one-cell-thick layers, the inner layer
    (belonging to the rotor zone) and the outer layer (belonging to the static
    zone).  Cells are binned on a ``(nx, n_theta)`` polar grid so that the
    azimuthal slip can be applied by a fractional ``roll`` + linear
    interpolation.

    Attributes
    ----------
    cell_in / cell_out:
        Flat indices of shell cells, shape ``(N_in,)`` / ``(N_out,)``.
    slot_in / slot_out:
        Polar bin of every shell cell, ``i_x * n_theta + i_theta``.
    rep_in / rep_out:
        Representative cell per polar bin (``-1`` where the bin is empty).
    dirs_in_in / dirs_in_out:
        Boolean ``(19, N)`` masks selecting *incoming* lattice directions for
        each shell cell (inward radial for the rotor layer, outward radial for
        the static layer).
    theta_per_step:
        Azimuthal increment of one polar bin (rad).
    """

    n_theta: int
    n_slots: int
    cell_in: torch.Tensor
    cell_out: torch.Tensor
    slot_in: torch.Tensor
    slot_out: torch.Tensor
    rep_in: torch.Tensor
    rep_out: torch.Tensor
    dirs_in_in: torch.Tensor
    dirs_in_out: torch.Tensor
    theta_per_step: float
    radius: float
    dy: torch.Tensor = field(repr=False)
    dz: torch.Tensor = field(repr=False)

    def count(self) -> dict:
        return {
            "n_cells_inner_layer": int(self.cell_in.numel()),
            "n_cells_outer_layer": int(self.cell_out.numel()),
            "n_polar_slots": int(self.n_slots),
            "n_theta": int(self.n_theta),
            "interface_radius": float(self.radius),
        }


def build_polar_shell(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    *,
    n_theta: int = 72,
    interface_width: float = 1.5,
    solid: torch.Tensor | None = None,
    zone: torch.Tensor | None = None,
) -> PolarShell:
    """Bin the cylindrical shell ``|r - radius| <= interface_width``.

    ``shape`` is ``(nz, ny, nx)`` and the rotation axis is ``x`` through
    ``(cy, cz)`` taken from ``center``.
    """
    nz, ny, nx = shape
    _, cy, cz = center
    zz, yy = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        indexing="ij",
    )
    dy = (yy - cy).contiguous()
    dz = (zz - cz).contiguous()
    r2d = torch.sqrt(dy * dy + dz * dz)
    theta2d = torch.remainder(torch.atan2(dz, dy), 2.0 * math.pi)
    # (nz, ny, 1) keeps the shell invariant along the shaft (x) axis
    r = r2d.unsqueeze(-1)
    theta = theta2d.unsqueeze(-1)

    if solid is None:
        solid = torch.zeros(shape, dtype=torch.bool)
    if zone is None:
        zone = torch.ones(shape, dtype=torch.bool)

    layer_in = (~solid) & zone & (r <= radius) & (r > radius - interface_width)
    layer_out = (~solid) & (~zone) & (r > radius) & (r <= radius + interface_width)

    theta_bin = torch.floor(theta / (2.0 * math.pi / n_theta)).long().clamp(0, n_theta - 1)
    x_index = torch.arange(nx, dtype=torch.long).view(1, 1, nx).expand(nz, ny, nx)
    slot = (x_index * n_theta + theta_bin).contiguous()
    n_slots = nx * n_theta

    def _pack(layer: torch.Tensor):
        cell = torch.nonzero(layer.reshape(-1), as_tuple=False).squeeze(1)
        if cell.numel() == 0:
            return cell, cell, torch.full((n_slots,), -1, dtype=torch.long)
        sl = slot.reshape(-1)[cell]
        rep = torch.full((n_slots,), -1, dtype=torch.long)
        rep.scatter_(0, sl, cell)  # last occurrence wins; empty bins stay -1
        return cell, sl, rep

    cell_in, slot_in, rep_in = _pack(layer_in)
    cell_out, slot_out, rep_out = _pack(layer_out)

    def _incoming(cell: torch.Tensor) -> torch.Tensor:
        if cell.numel() == 0:
            return torch.zeros((19, 0), dtype=torch.bool)
        fz = cell // (ny * nx)
        fy = (cell % (ny * nx)) // nx
        ry = dy.reshape(-1)[fy]
        rz = dz.reshape(-1)[fz]
        rn = torch.sqrt(ry * ry + rz * rz).clamp(min=1e-8)
        n_y = ry / rn
        n_z = rz / rn
        radial = _C[:, 1].view(19, 1) * n_y.view(1, -1) + _C[:, 2].view(19, 1) * n_z.view(1, -1)
        return radial

    radial_in = _incoming(cell_in)  # rotor layer: inward radial component
    radial_out = _incoming(cell_out)  # static layer
    dirs_in_in = radial_in < -1e-6
    dirs_in_out = radial_out > 1e-6

    return PolarShell(
        n_theta=n_theta,
        n_slots=n_slots,
        cell_in=cell_in,
        cell_out=cell_out,
        slot_in=slot_in,
        slot_out=slot_out,
        rep_in=rep_in,
        rep_out=rep_out,
        dirs_in_in=dirs_in_in,
        dirs_in_out=dirs_in_out,
        theta_per_step=2.0 * math.pi / n_theta,
        radius=float(radius),
        dy=dy,
        dz=dz,
    )


def _polar_average(
    field: torch.Tensor,
    cells: torch.Tensor,
    slots: torch.Tensor,
    n_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average a cell-centred field on the polar bins; empty bins -> 0."""
    flat = field.reshape(-1).to(torch.float32)
    total = torch.zeros(n_slots, dtype=torch.float32, device=flat.device)
    count = torch.zeros(n_slots, dtype=torch.float32, device=flat.device)
    if cells.numel() == 0:
        return total, count
    total.index_add_(0, slots, flat[cells])
    count.index_add_(0, slots, torch.ones_like(slots, dtype=torch.float32))
    mean = total / count.clamp(min=1.0)
    return mean, count


def _shift_azimuth(values: torch.Tensor, n_theta: int, shift: float) -> torch.Tensor:
    """Fractional periodic shift along the azimuth (last axis of length n_theta)."""
    head = values.shape[:-1]
    v = values.reshape(-1, n_theta)
    i0 = math.floor(shift)
    frac = shift - i0
    i0 = i0 % n_theta
    v0 = torch.roll(v, shifts=i0, dims=1)
    v1 = torch.roll(v, shifts=(i0 + 1) % n_theta, dims=1)
    out = (1.0 - frac) * v0 + frac * v1
    return out.reshape(*head, n_theta)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


class SlidingMeshPropeller:
    """Two-zone (static + sliding rotating zone) D3Q19 solver.

    Parameters
    ----------
    cfg:
        :class:`RotatingZoneConfig`.
    geometry:
        Propeller geometry (lattice units).  ``None`` disables the rotor blades
        (used by the verification cases).
    center:
        ``(cx, cy, cz)`` of the propeller / shaft in lattice coordinates.
    outer_solid:
        Optional additional static solid mask ``(nz, ny, nx)`` in the outer
        zone (e.g. the hull or an outer cylinder).
    device:
        Torch device string.
    """

    def __init__(
        self,
        cfg: RotatingZoneConfig,
        geometry: PropellerGeometryConfig | None = None,
        *,
        center: Sequence[float] | None = None,
        outer_solid: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        shape = (cfg.nz, cfg.ny, cfg.nx)
        cx = center[0] if center is not None else cfg.nx / 2.0
        cy = center[1] if center is not None else cfg.ny / 2.0
        cz = center[2] if center is not None else cfg.nz / 2.0
        self.center = (float(cx), float(cy), float(cz))

        zz, yy, xx = torch.meshgrid(
            torch.arange(cfg.nz, dtype=torch.float32),
            torch.arange(cfg.ny, dtype=torch.float32),
            torch.arange(cfg.nx, dtype=torch.float32),
            indexing="ij",
        )
        self.dy = (yy - cy).contiguous()
        self.dz = (zz - cz).contiguous()
        self.r = torch.sqrt(self.dy * self.dy + self.dz * self.dz)

        self.zone = self.r <= cfg.zone_radius

        if outer_solid is None:
            outer_solid = torch.zeros(shape, dtype=torch.bool)
        else:
            outer_solid = outer_solid.to(torch.bool)
        static_solid = outer_solid & (~self.zone)

        if geometry is not None:
            blades = build_propeller_mask(
                cfg.nx, cfg.ny, cfg.nz, cx, cy, cz, 0.0, geometry, device=self.device
            )
            blades = blades & self.zone & (~outer_solid)
        else:
            blades = torch.zeros(shape, dtype=torch.bool)

        self.blade = blades
        self.solid = static_solid | blades

        # Sides / end walls
        self.wall_solid = torch.zeros(shape, dtype=torch.bool)
        if cfg.sides == "solid":
            self.wall_solid[0, :, :] = True
            self.wall_solid[-1, :, :] = True
            self.wall_solid[:, 0, :] = True
            self.wall_solid[:, -1, :] = True
        if cfg.streamwise == "closed":
            self.wall_solid[:, :, 0] = True
            self.wall_solid[:, :, -1] = True
        self.solid = self.solid | self.wall_solid

        self.shell = build_polar_shell(
            shape,
            self.center,
            cfg.zone_radius,
            n_theta=cfg.n_theta,
            interface_width=cfg.interface_width,
            solid=self.solid,
            zone=self.zone,
        )

        # Pre-computed fluid -> solid link lists (geometry is static in its zone)
        self._links: list[tuple[int, torch.Tensor, float, float, float]] = []
        solid_flat = self.solid
        for q in range(1, 19):
            cq = _C[q]
            ix, iy, iz = int(cq[0].item()), int(cq[1].item()), int(cq[2].item())
            shifted = torch.roll(solid_flat, (-iz, -iy, -ix), dims=(0, 1, 2))
            cross = (~solid_flat) & shifted
            idx = torch.nonzero(cross.reshape(-1), as_tuple=False).squeeze(1)
            if idx.numel():
                dyc = self.dy.reshape(-1)[idx]
                dzc = self.dz.reshape(-1)[idx]
                tcoef = dyc * iz - dzc * iy
                self._links.append((q, idx, float(ix), float(iy), float(iz), tcoef))

        self._slip = self._build_slip_tables(cfg.slip_planes)

        # Inlet / outlet planes (all directions point into the domain)
        self._io = cfg.streamwise == "inlet_outlet"
        # Initialise in the *lab* frame (uniform stream); inside the rotor zone the
        # stored velocity is the frame-relative one, u_rel = u_lab - Omega x r.
        ones = torch.ones(shape, device=self.device)
        u_init_x = torch.full(shape, cfg.inflow, device=self.device)
        u_init_y = torch.where(self.zone, cfg.omega * self.dz, torch.zeros(shape))
        u_init_z = torch.where(self.zone, -cfg.omega * self.dy, torch.zeros(shape))
        self.f = equilibrium3d(ones, u_init_x, u_init_y, u_init_z)
        self.body_force: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.inflow_profile: torch.Tensor | None = None
        self.time = 0
        self.theta = 0.0
        self._mass0 = float(self.f.sum().item())
        self.last_force = (0.0, 0.0, 0.0)
        self.last_torque = 0.0
        self.history: list[dict] = []

    # -- free-slip planes -------------------------------------------------- #

    @staticmethod
    def _build_slip_tables(planes: tuple[str, ...]):
        """Specular-reflection tables for free-slip planes (no tangential stress)."""
        tables = []
        for name in planes:
            axis = 1 if name.startswith("y") else 2
            hi = name.endswith("max")
            mirrored = {}
            for q in range(1, 19):
                cq = [int(_C[q][0]), int(_C[q][1]), int(_C[q][2])]
                if cq[axis] == 0:
                    continue
                # population that would leave the domain across this plane
                outward = cq[axis] > 0 if hi else cq[axis] < 0
                if not outward:
                    continue
                target = list(cq)
                target[axis] = -cq[axis]
                for qq in range(19):
                    c2 = [int(_C[qq][0]), int(_C[qq][1]), int(_C[qq][2])]
                    if c2 == target:
                        mirrored[q] = qq
                        break
            tables.append((axis, hi, list(mirrored.items())))
        return tables

    def _apply_slip(self, f: torch.Tensor) -> torch.Tensor:
        for axis, hi, pairs in self._slip:
            idx = -1 if hi else 0
            sl = [slice(None)] * 4
            sl[axis + 1] = idx
            sl = tuple(sl)
            plane = f[sl].clone()
            for q_out, q_in in pairs:
                # f whose velocity points *into* the domain is the mirror image of
                # the outgoing population: specular (slip) reflection, no shear.
                f[(slice(q_in, q_in + 1),) + sl[1:]] = plane[q_out : q_out + 1]
        return f

    # -- diagnostics ------------------------------------------------------ #

    def macroscopic(self):
        rho, ux, uy, uz = macroscopic3d(self.f)
        return rho, ux, uy, uz

    def mass(self) -> float:
        return float(self.f.sum().item())

    # -- physics kernels -------------------------------------------------- #

    def _eddy_viscosity(self, rho: torch.Tensor, ux, uy, uz):
        """Smagorinsky eddy viscosity from velocity gradients (Hou et al. 1996).

        ``nu_t = (C_s Delta)^2 |S|`` with ``|S| = sqrt(2 S_ab S_ab)`` built from
        second-order central differences of the velocity field.  The
        velocity-gradient form is used instead of the non-equilibrium-moment
        form because it costs O(9) roll/difference operations per step instead of
        O(19^2) tensor products.
        """
        cfg = self.cfg
        if cfg.smagorinsky <= 0.0:
            return None

        def d(field, axis):
            return 0.5 * (torch.roll(field, -1, dims=axis) - torch.roll(field, 1, dims=axis))

        comps = (ux, uy, uz)
        du = [[d(c, axis) for axis in (0, 1, 2)] for c in comps]  # du[i][j] = d u_i / d x_j
        s2 = 0.0
        for i in range(3):
            for j in range(3):
                s_ij = 0.5 * (du[i][j] + du[j][i])
                s2 = s2 + s_ij * s_ij
        s_mag = torch.sqrt(torch.clamp(2.0 * s2, min=0.0))
        nu_t = (cfg.smagorinsky**2) * s_mag
        return 0.5 + 3.0 * (cfg.nu + nu_t)

    def set_body_force(self, mask: torch.Tensor, fx, fy, fz) -> None:
        """Inject an arbitrary momentum source (e.g. an actuator-disk loading).

        ``fx``/``fy``/``fz`` are force *densities* (mass x acceleration) already
        multiplied by the density, restricted to ``mask``.
        """
        zeros = torch.zeros((self.cfg.nz, self.cfg.ny, self.cfg.nx), dtype=torch.float32)
        if mask is not None:
            fx = torch.where(mask, fx, zeros)
            fy = torch.where(mask, fy, zeros)
            fz = torch.where(mask, fz, zeros)
        self.body_force = (fx, fy, fz)

    def clear_body_force(self) -> None:
        self.body_force = None

    def set_inflow_profile(self, profile: torch.Tensor | None) -> None:
        """Non-uniform axial inflow profile ``(nz, ny)`` (lattice units)."""
        self.inflow_profile = None if profile is None else profile.to(torch.float32)

    def _apply_rotating_sources(self, rho, ux, uy, uz):
        cfg = self.cfg
        if cfg.omega == 0.0:
            return None
        om = cfg.omega
        fy = rho * (2.0 * om * uz + om * om * self.dy)
        fz = rho * (-2.0 * om * uy + om * om * self.dz)
        mask = self.zone & (~self.solid)
        fy = torch.where(mask, fy, torch.zeros_like(fy))
        fz = torch.where(mask, fz, torch.zeros_like(fz))
        return fy, fz

    def _guo_source(self, rho, ux, uy, uz, fx, fy, fz, tau):
        """Guo (2002) forcing: S_i = (1 - 1/(2 tau)) w_i [3 (c_i - u) + 9 (c_i.u) c_i] . F."""
        if fy is None and fz is None and fx is None:
            return None
        cu = (
            _C[:, 0].view(19, 1, 1, 1) * ux
            + _C[:, 1].view(19, 1, 1, 1) * uy
            + _C[:, 2].view(19, 1, 1, 1) * uz
        )
        w = _W.view(19, 1, 1, 1)
        pref = 1.0 - 1.0 / (2.0 * tau)
        s = torch.zeros((19,) + tuple(rho.shape), dtype=rho.dtype, device=rho.device)
        for a, fa in enumerate((fx, fy, fz)):
            if fa is None:
                continue
            ca = _C[:, a].view(19, 1, 1, 1)
            ua = (ux, uy, uz)[a].unsqueeze(0)
            s = s + (3.0 * (ca - ua) + 9.0 * cu * ca) * fa.unsqueeze(0)
        return pref * w * s

    def _bounce_back(self, f: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Half-way bounce-back on all solids, accumulating the linked loads.

        Momentum-exchange (Ladd 1994) force on the wall per link:
        ``dF = 2 f_i c_i`` for the population ``f_i`` travelling from the fluid
        cell towards the solid neighbour.  The shaft torque follows as
        ``Q_x = -(r x F)_x`` and the thrust as ``T = -F_x`` (the propulsor pushes
        the fluid downstream, so the blade reaction is upstream).
        """
        ff = f.view(19, -1)
        out = f.clone()
        of = out.view(19, -1)
        tx = ty = tz = tq = 0.0
        for q, idx, cqx, cqy, cqz, tcoef in self._links:
            fi = ff[q][idx]
            of[int(_OPP[q])][idx] = fi
            contrib = 2.0 * fi
            if cqx:
                tx += float(contrib.sum().item()) * cqx
            if cqy:
                ty += float(contrib.sum().item()) * cqy
            if cqz:
                tz += float(contrib.sum().item()) * cqz
            if tcoef is not None:
                tq += float((contrib * tcoef).sum().item())
        return out, {"fx": -tx, "fy": -ty, "fz": -tz, "torque": -tq}

    def _bounce_back_field(self, f: torch.Tensor) -> torch.Tensor:
        """Per-cell momentum-exchange force field (for field diagnostics)."""
        nz, ny, nx = self.cfg.nz, self.cfg.ny, self.cfg.nx
        force = torch.zeros((3, nz, ny, nx), dtype=f.dtype, device=f.device)
        ff = f.view(19, -1)
        for q, idx, cqx, cqy, cqz, _ in self._links:
            contrib = 2.0 * ff[q][idx]
            if cqx:
                force[0].view(-1).index_add_(0, idx, contrib * cqx)
            if cqy:
                force[1].view(-1).index_add_(0, idx, contrib * cqy)
            if cqz:
                force[2].view(-1).index_add_(0, idx, contrib * cqz)
        return force

    def _apply_io(self, f: torch.Tensor) -> torch.Tensor:
        """Uniform lab-frame inflow at ``x = 0``; zero-gradient outflow at ``x = nx-1``.

        The rotor zone spans the whole cylinder, so the inlet plane also carries
        zone cells; there the imposed *lab-frame* velocity ``(U, 0, 0)`` is
        converted to the rotor frame, ``u_rel = u_lab - Omega x r``.
        """
        if not self._io:
            return f
        cfg = self.cfg
        om = cfg.omega
        z2 = torch.zeros((cfg.nz, cfg.ny, 1), dtype=f.dtype, device=f.device)
        if self.inflow_profile is None:
            ux = torch.full((cfg.nz, cfg.ny, 1), cfg.inflow, dtype=f.dtype, device=f.device)
        else:
            ux = self.inflow_profile.reshape(cfg.nz, cfg.ny, 1).to(f.dtype).to(f.device)
        dy0 = self.dy[:, :, 0:1]
        dz0 = self.dz[:, :, 0:1]
        zone0 = self.zone[:, :, 0:1]
        uy = torch.where(zone0, om * dz0, z2)
        uz = torch.where(zone0, -om * dy0, z2)
        one = torch.ones_like(ux)
        f[:, :, :, 0:1] = equilibrium3d(one, ux, uy, uz)
        f[:, :, :, -1:] = f[:, :, :, -2:-1]
        return f

    def _exchange(self, f: torch.Tensor) -> dict:
        """Sliding-interface exchange (frame transform + azimuthal slip)."""
        cfg = self.cfg
        shell = self.shell
        if shell.cell_in.numel() == 0 or shell.cell_out.numel() == 0:
            return f, {}
        rho, ux, uy, uz = macroscopic3d(f)
        om = cfg.omega
        # lab-frame velocity field (zone cells already hold u_rel)
        uy_lab = uy + torch.where(self.zone, -om * self.dz, torch.zeros_like(self.dz))
        uz_lab = uz + torch.where(self.zone, om * self.dy, torch.zeros_like(self.dy))

        n_theta = shell.n_theta

        # polar-bin the two layers
        def _bin(field):
            mean, cnt = _polar_average(field, shell.cell_in, shell.slot_in, shell.n_slots)
            mean_out, cnt_out = _polar_average(field, shell.cell_out, shell.slot_out, shell.n_slots)
            return (
                mean.reshape(cfg.nx, n_theta),
                cnt.reshape(cfg.nx, n_theta),
                mean_out.reshape(cfg.nx, n_theta),
                cnt_out.reshape(cfg.nx, n_theta),
            )

        rho_i, c_i, rho_o, c_o = _bin(rho)
        ux_i, _, ux_o, _ = _bin(ux)
        uy_i, _, uy_o, _ = _bin(uy_lab)
        uz_i, _, uz_o, _ = _bin(uz_lab)

        # fill empty bins with the layer mean so the shift never sees zeros
        def _fill(v, c):
            if c.sum() == 0:
                return torch.zeros_like(v)
            m = v.sum() / c.sum()
            return torch.where(c > 0, v, m)

        rho_i, rho_o = _fill(rho_i, c_i), _fill(rho_o, c_o)
        ux_i, ux_o = _fill(ux_i, c_i), _fill(ux_o, c_o)
        uy_i, uy_o = _fill(uy_i, c_i), _fill(uy_o, c_o)
        uz_i, uz_o = _fill(uz_i, c_i), _fill(uz_o, c_o)

        # sliding slip: zone has advanced by self.theta relative to the grid
        slip = self.theta / shell.theta_per_step
        src_from_zone = _shift_azimuth(torch.stack([rho_i, ux_i, uy_i, uz_i]), n_theta, -slip)
        src_from_static = _shift_azimuth(torch.stack([rho_o, ux_o, uy_o, uz_o]), n_theta, slip)

        # gather the polar source state back onto the receiving shell cells
        def _gather(state, cells, slots):
            out = torch.empty((state.shape[0], cells.numel()), dtype=state.dtype)
            s = state.reshape(state.shape[0], -1)
            out = s[:, slots]
            return out

        # static layer receives from the rotor zone (lab frame already)
        st_o = _gather(src_from_zone, shell.cell_out, shell.slot_out)
        # rotor layer receives from the static zone (lab) -> convert to zone frame
        st_i = _gather(src_from_static, shell.cell_in, shell.slot_in)

        def _set_incoming(f_buf, cells, dirs_mask, rho_s, ux_s, uy_s, uz_s):
            if cells.numel() == 0:
                return
            shp = (1, 1, -1)
            eq = equilibrium3d(
                rho_s.reshape(shp), ux_s.reshape(shp), uy_s.reshape(shp), uz_s.reshape(shp)
            ).reshape(19, -1)
            flat = f_buf.view(19, -1)
            # Stress continuity: the neighbour supplies the *velocity* (and hence the
            # advected momentum), while the non-equilibrium part of the incoming
            # characteristics is taken from the receiving side so that the tangential
            # shear stress is not destroyed by the exchange.  Rebuilding the incoming
            # set from a pure equilibrium (the naive velocity-Dirichlet interface)
            # pins the tangential velocity without carrying any shear, which at a
            # curved interface acts as a strong artificial damper: Taylor-Couette
            # then lost ~80% of the azimuthal velocity inside the rotor zone.
            f_local = flat[:, cells]
            rho_c = f_local.sum(0).clamp(min=1e-12)
            mom = _C.T.to(f_local.dtype) @ f_local  # (3, N) momentum density
            eq_local = equilibrium3d(
                rho_c.reshape(shp),
                (mom[0] / rho_c).reshape(shp),
                (mom[1] / rho_c).reshape(shp),
                (mom[2] / rho_c).reshape(shp),
            ).reshape(19, -1)
            neq = f_local - eq_local
            # Mass-neutral reconstruction.  The incoming characteristics carry the
            # neighbour's velocity but are rebuilt from the *local* density, so the
            # mass they inject need not equal the mass they displace.  Rescaling the
            # reconstructed equilibrium by the per-cell mass ratio
            # ``sum(old)/sum(new)`` over the incoming set makes the exchange an exact
            # zero-net-mass operation (momentum direction and tangential data are
            # untouched), which is what removes the O(1e-4/step) density drift of the
            # naive velocity-Dirichlet interface in a closed rotating box.
            m_old = torch.zeros(cells.numel(), dtype=flat.dtype, device=flat.device)
            m_eq = torch.zeros_like(m_old)
            m_neq = torch.zeros_like(m_old)
            for q in range(19):
                sel = dirs_mask[q]
                if not bool(sel.any()):
                    continue
                m_old[sel] += f_local[q][sel]
                m_eq[sel] += eq[q][sel]
                m_neq[sel] += neq[q][sel]
            # keep ``scale*eq + neq`` mass-neutral over the incoming set
            scale = torch.where(
                m_eq.abs() > 1e-12,
                ((m_old - m_neq) / m_eq).clamp(0.2, 5.0),
                torch.ones_like(m_eq),
            )
            for q in range(19):
                sel = dirs_mask[q]
                if not bool(sel.any()):
                    continue
                flat[q][cells[sel]] = eq[q][sel] * scale[sel] + neq[q][sel]

        # Velocity-Dirichlet interface: the *velocity* crosses the interface, the
        # density (hence the pressure, which must carry the centrifugal head
        # rho Omega^2 r^2 / 2 inside the rotor zone) stays local.
        rho_local = rho.reshape(-1)
        _set_incoming(
            f,
            shell.cell_out,
            shell.dirs_in_out,
            rho_local[shell.cell_out],
            st_o[1],
            st_o[2],
            st_o[3],
        )
        # frame conversion for the state entering the rotor zone
        ci = shell.cell_in
        dyc = self.dy.reshape(-1)[ci]
        dzc = self.dz.reshape(-1)[ci]
        uy_i_rel = st_i[2] + om * dzc
        uz_i_rel = st_i[3] - om * dyc
        _set_incoming(
            f,
            shell.cell_in,
            shell.dirs_in_in,
            rho_local[shell.cell_in],
            st_i[1],
            uy_i_rel,
            uz_i_rel,
        )
        return f, {
            "rho_zone_layer_mean": float(st_o[0].mean().item()),
            "rho_static_layer_mean": float(st_i[0].mean().item()),
            "u_lab_zone_layer_mean": float(st_o[1].mean().item()),
        }

    # -- time stepping ---------------------------------------------------- #

    def step(self, *, measure: bool = True) -> dict:
        cfg = self.cfg
        f = self.f
        rho, ux, uy, uz = macroscopic3d(f)
        tau_eff = self._eddy_viscosity(rho, ux, uy, uz)
        tau_use = cfg.tau if tau_eff is None else tau_eff
        f_eq = equilibrium3d(rho, ux, uy, uz)
        # f <- f (1 - 1/tau) + f_eq / tau, done in place to avoid 100 MB temporaries
        if isinstance(tau_use, torch.Tensor):
            inv = 1.0 / tau_use
            f.mul_(1.0 - inv).add_(f_eq * inv)
        else:
            inv = 1.0 / tau_use
            f.mul_(1.0 - inv).add_(f_eq, alpha=inv)

        src = self._apply_rotating_sources(rho, ux, uy, uz)
        fbx = fby = fbz = None
        if self.body_force is not None:
            fbx, fby, fbz = self.body_force
        if src is not None:
            fby = src[0] if fby is None else fby + src[0]
            fbz = src[1] if fbz is None else fbz + src[1]
        if fbx is not None or fby is not None or fbz is not None:
            s = self._guo_source(rho, ux, uy, uz, fbx, fby, fbz, tau_use)
            f = f + s

        f = stream3d(f)
        f, force = self._bounce_back(f)
        f = self._apply_io(f)
        if self._slip:
            f = self._apply_slip(f)
        f, iface_diag = self._exchange(f)

        self.f = f
        self.time += 1
        self.theta += cfg.omega

        diag = {}
        if measure:
            self.last_force = (force["fx"], force["fy"], force["fz"])
            self.last_torque = force["torque"]
            diag = {
                "thrust": force["fx"],
                "force_y": force["fy"],
                "force_z": force["fz"],
                "torque": force["torque"],
                **iface_diag,
            }
        return diag

    def run(self, steps: int, *, warmup: int = 0, measure_every: int = 1) -> dict:
        """Advance ``steps`` and return the window-averaged blade loads."""
        cfg = self.cfg
        t_sum = q_sum = 0.0
        samples = 0
        for it in range(steps):
            diag = self.step(measure=(it >= warmup and (it % measure_every == 0)))
            if diag:
                t_sum += diag["thrust"]
                q_sum += diag["torque"]
                samples += 1
        return {
            "steps": steps,
            "warmup": warmup,
            "samples": samples,
            "thrust_lattice": t_sum / max(samples, 1),
            "torque_lattice": q_sum / max(samples, 1),
            "omega": cfg.omega,
            "time": self.time,
            "theta": self.theta,
        }

    # -- post-processing --------------------------------------------------- #

    def disk_plane_profile(self, x_index: int | None = None) -> dict:
        """Azimuthally averaged axial-velocity profile at the propeller plane."""
        rho, ux, uy, uz = self.macroscopic()
        x_index = self.cfg.nx // 2 if x_index is None else x_index
        r = self.r
        ux_plane = ux[:, :, x_index]
        r_plane = r[:, :, x_index]
        # rotate to the lab frame (rotor zone values are relative)
        # u_x is invariant under the frame transformation u_lab = u_rel + Omega x r
        nbins = 24
        rmax = float(r_plane.max().item())
        edges = torch.linspace(0.0, rmax, nbins + 1)
        prof = []
        for b in range(nbins):
            m = (r_plane >= edges[b]) & (r_plane < edges[b + 1])
            if bool(m.any()):
                prof.append(
                    {
                        "r": float((0.5 * (edges[b] + edges[b + 1])).item()),
                        "u_axial": float(ux_plane[m].mean().item()),
                        "samples": int(m.sum().item()),
                    }
                )
        return {"x_index": int(x_index), "profile": prof}


# --------------------------------------------------------------------------- #
# Analytic verification: Taylor--Couette
# --------------------------------------------------------------------------- #


def taylor_couette_velocity(r, r_inner: float, r_outer: float, omega_inner: float):
    """Exact azimuthal velocity of laminar Taylor--Couette flow.

    ``v_theta(r) = A r + B / r`` with ``A = -omega_i R_i^2 / (R_o^2 - R_i^2)``
    and ``B = omega_i R_i^2 R_o^2 / (R_o^2 - R_i^2)`` (outer cylinder at rest).
    """
    ri, ro = float(r_inner), float(r_outer)
    if not (ri < ro):
        raise ValueError("r_inner must be smaller than r_outer")
    a = -omega_inner * ri * ri / (ro * ro - ri * ri)
    b = omega_inner * ri * ri * ro * ro / (ro * ro - ri * ri)
    return a * r + b / r


def run_taylor_couette(
    *,
    n: int = 48,
    r_inner: float = 8.0,
    r_outer: float = 20.0,
    zone_radius: float = 14.0,
    omega: float = 2.0e-4,
    tau: float = 0.8,
    steps: int = 12000,
    n_theta: int = 96,
    device: str = "cpu",
) -> dict:
    """Verify the sliding interface against the exact Taylor--Couette profile.

    The inner cylinder (radius ``r_inner``) sits inside the rotor zone and is at
    rest *in the rotor frame*, i.e. it rotates at ``omega`` in the lab frame.
    The outer cylinder (radius ``r_outer``) is a static no-slip wall in the
    static zone.  The sliding interface lies in the middle of the gap, so the
    computed profile tests the frame transformation and the azimuthal slip
    together.
    """
    cfg = RotatingZoneConfig(
        nx=n,
        ny=n,
        nz=n,
        inflow=0.0,
        omega=omega,
        tau=tau,
        zone_radius=zone_radius,
        n_theta=n_theta,
        streamwise="closed",
        sides="solid",
    )
    cy = cz = n / 2.0
    zz, yy = torch.meshgrid(
        torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij"
    )
    rr = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    outer_solid = torch.zeros((n, n, n), dtype=torch.bool)
    outer_solid |= (rr > r_outer).unsqueeze(-1).expand(n, n, n)
    inner_solid = (rr < r_inner).unsqueeze(-1).expand(n, n, n)

    solver = SlidingMeshPropeller(
        cfg, None, center=(n / 2.0, cy, cz), outer_solid=outer_solid | inner_solid, device=device
    )
    solver.run(steps, warmup=steps - 1)

    rho, ux, uy, uz = solver.macroscopic()
    # lab-frame azimuthal velocity on the mid-axial slice
    k = n // 2
    zone_k = solver.zone[:, :, k]
    dy_k = solver.dy[:, :, k]
    dz_k = solver.dz[:, :, k]
    uy_lab = uy[:, :, k] + torch.where(zone_k, -omega * dz_k, torch.zeros_like(dz_k))
    uz_lab = uz[:, :, k] + torch.where(zone_k, omega * dy_k, torch.zeros_like(dy_k))
    vtheta = (dy_k * uz_lab - dz_k * uy_lab) / solver.r[:, :, k].clamp(min=1e-6)

    samples = []
    # probe radii span the fluid gap (the hard-coded 9.5..18.5 list assumed one
    # particular box size and silently sampled the solid outer cylinder otherwise)
    probe = [r_inner + (r_outer - r_inner) * (0.10 + 0.80 * k / 6.0) for k in range(7)]
    for rad in probe:
        m = (rr - rad).abs() < 0.6
        if bool(m.any()):
            num = float(vtheta[m].mean().item())
            ref = float(taylor_couette_velocity(rad, r_inner, r_outer, omega))
            samples.append(
                {
                    "r": rad,
                    "v_theta_lbm": num,
                    "v_theta_exact": ref,
                    "rel_error": (num - ref) / ref if ref else float("nan"),
                    "samples": int(m.sum().item()),
                }
            )
    # the exact profile passes through zero near r = sqrt(B/|A|); a relative error
    # there is meaningless, so only samples carrying a finite fraction of the peak
    # azimuthal velocity enter the error statistics
    peak = max((abs(s["v_theta_exact"]) for s in samples), default=0.0)
    useful = [
        s
        for s in samples
        if math.isfinite(s["rel_error"]) and abs(s["v_theta_exact"]) > 1e-2 * peak
    ]
    errs = [abs(s["rel_error"]) for s in useful]
    band = [
        abs(s["rel_error"])
        for s in useful
        if abs(s["r"] - zone_radius) <= 1.0 and math.isfinite(s["rel_error"])
    ]
    wall_band = [
        abs(s["rel_error"])
        for s in useful
        if abs(s["r"] - zone_radius) > 1.0 and math.isfinite(s["rel_error"])
    ]
    return {
        "samples": samples,
        # error split: the sliding interface band versus the no-slip wall bands.  The
        # wall bands are limited by the half-way bounce-back wall position and by the
        # cells-per-gap resolution, not by the interface mechanism.
        "interface_band_max_rel_error": max(band) if band else float("nan"),
        "interface_band_mean_rel_error": sum(band) / len(band) if band else float("nan"),
        "wall_band_max_rel_error": max(wall_band) if wall_band else float("nan"),
        "n_error_samples": len(useful),
        "max_rel_error": max(errs) if errs else float("nan"),
        "mean_rel_error": sum(errs) / len(errs) if errs else float("nan"),
        "config": {
            "n": n,
            "r_inner": r_inner,
            "r_outer": r_outer,
            "zone_radius": zone_radius,
            "omega": omega,
            "tau": tau,
            "steps": steps,
            "re_inner": omega * r_inner * r_inner / ((tau - 0.5) / 3.0),
        },
        "solver_time": solver.time,
        "mass": solver.mass(),
        "mass0": solver._mass0,
    }


# --------------------------------------------------------------------------- #
# Open-water sweep
# --------------------------------------------------------------------------- #


def open_water_curve(
    geometry: PropellerGeometryConfig,
    *,
    j_values: Sequence[float],
    base: RotatingZoneConfig | None = None,
    steps: int = 2400,
    warmup: int = 1200,
    device: str = "cpu",
    omega_sign: float = -1.0,
) -> list[dict]:
    """Run the sliding-mesh solver at several advance ratios ``J = U / (n D)``.

    ``n`` (rev/step) and ``omega = 2 pi n`` follow from ``J`` and ``U``; the
    thrust and torque coefficients are ``K_T = T / (rho n^2 D^4)`` and
    ``K_Q = Q / (rho n^2 D^5)``.

    ``omega_sign`` selects the rotation sense.  With the lattice convention of
    :func:`tensorlbm.propeller_cad.build_propeller_mask` and a positive
    ``pitch_ratio_07``, ``omega > 0`` drives the fluid *upstream* (it behaves as
    a turbine); the propulsive sense is ``omega < 0``.
    """
    d = geometry.diameter
    out = []
    for j in j_values:
        cfg = base or RotatingZoneConfig(
            nx=96,
            ny=96,
            nz=96,
            inflow=0.05,
            tau=0.503,
            zone_radius=d * 0.62,
            n_theta=72,
            smagorinsky=0.10,
        )
        u = cfg.inflow
        n_rps = u / (j * d)
        omega = omega_sign * 2.0 * math.pi * n_rps
        cfg = RotatingZoneConfig(
            nx=cfg.nx,
            ny=cfg.ny,
            nz=cfg.nz,
            inflow=u,
            omega=omega,
            tau=cfg.tau,
            smagorinsky=cfg.smagorinsky,
            zone_radius=max(cfg.zone_radius, d * 0.62),
            n_theta=cfg.n_theta,
            interface_width=cfg.interface_width,
            sides=cfg.sides,
            streamwise=cfg.streamwise,
        )
        solver = SlidingMeshPropeller(cfg, geometry, device=device)
        res = solver.run(steps, warmup=warmup)
        thrust = res["thrust_lattice"]
        torque = res["torque_lattice"]
        kt = thrust / (1.0 * n_rps * n_rps * d**4) if n_rps > 0 else float("nan")
        kq = torque / (1.0 * n_rps * n_rps * d**5) if n_rps > 0 else float("nan")
        eta0 = j * kt / (2.0 * math.pi * kq) if kq > 0 else float("nan")
        out.append(
            {
                "j": j,
                "n_rps": n_rps,
                "omega": omega,
                "u_inflow": u,
                "thrust": thrust,
                "torque": torque,
                "kt": kt,
                "kq": kq,
                "eta0": eta0,
                "steps": steps,
                "warmup": warmup,
                "samples": res["samples"],
                "tip_mach": omega * d / 2.0,
                "re_d": u * d / cfg.nu,
            }
        )
    return out
