"""Wind-load prediction for LNG-carrier (Moss-type) superstructures with LBM.

The module provides a self-contained, wall-modelled large-eddy LBM solver for
the *above-deck* aerodynamics of a LNG carrier: a stepped (voxel) description
of the deck house / funnel / masts / pipe racks is immersed in a uniform wind,
and the aerodynamic force and moment coefficients

    C_X(theta)  longitudinal (drag) force    / (q A_F)
    C_Y(theta)  transverse (side) force      / (q A_L)
    C_K(theta)  heeling moment about x       / (q A_L H)
    C_N(theta)  yaw moment about z           / (q A_L B)

are integrated from the wall momentum exchange (pressure dominated) plus a
log-law wall-function shear contribution (unresolved boundary layer).

Model stack
-----------
* D3Q19, BGK collision in lattice units, single precision, PyTorch backend.
* Smagorinsky sub-grid model, ``nu_t = (Cs dx)^2 |S|`` with the strain rate
  recovered from the non-equilibrium stress ``S_ab ~ -3 Pi_ab^neq /(2 rho tau)``.
* Half-way bounce-back on the stepped surface (Ladd momentum exchange).
* Log-law wall function (``u+ = 1/kappa ln(y+) + B``) supplying the unresolved
  turbulent wall shear, so the near-wall resolution may be much coarser than
  ``y+ ~ 1`` (wall-modelled LES regime).
* Velocity inlet, zero-gradient outlet (optionally flux-normalised),
  free-slip top / lateral / sea-surface planes.

Reference-area and moment-arm conventions follow the wind-load literature
(Blendermann 1994; OCIMF 1994): ``q = 0.5 rho_air U^2``, ``A_F`` = projected
frontal area, ``A_L`` = projected lateral area, ``B``/``H`` = superstructure
breadth/height above deck.

The module has no import-time side effects and does not modify the shared
``tensorlbm.__init__`` namespace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# D3Q19 lattice
# ---------------------------------------------------------------------------

_D3Q19_C: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
)
_D3Q19_W: tuple[float, ...] = (
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)
C_LATTICE = _D3Q19_C
W_LATTICE = _D3Q19_W
OPP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)
CS2 = 1.0 / 3.0


def _perm_for_axis(axis: int) -> tuple[int, ...]:
    """Direction permutation reflecting the ``axis`` component of ``c_i``."""
    out = []
    for c in _D3Q19_C:
        cc = list(c)
        cc[axis] = -cc[axis]
        out.append(_D3Q19_C.index(tuple(cc)))
    return tuple(out)


PERM_X = _perm_for_axis(0)
PERM_Y = _perm_for_axis(1)
PERM_Z = _perm_for_axis(2)

RHO_AIR = 1.225          # kg/m^3 (15 degC, 1013 hPa)
NU_AIR = 1.5e-5          # m^2/s, kinematic viscosity of air
KAPPA = 0.41             # von Karman constant
B_LOG = 5.2              # log-law intercept


def solve_friction_velocity(ut, y, nu, *, iters: int = 4):
    """Invert the log law for the friction velocity (Newton iteration).

    Solves ``ut = u_tau (ln(y u_tau / nu)/kappa + B)`` — the standard Kader
    wall law written with ``B = 5.2`` — for ``u_tau``; ``ut``, ``y`` and ``nu``
    may be scalars or tensors.  Returns ``u_tau`` (clipped to be positive).
    """
    import torch as _torch

    if isinstance(ut, _torch.Tensor):
        clamp = lambda v: _torch.clamp(v, min=1e-6)      # noqa: E731
        logf = lambda v: _torch.log(_torch.clamp(v, min=1.05))   # noqa: E731
    else:
        arr = np.asarray(ut, dtype=float)
        clamp = lambda v: np.maximum(v, 1e-6)                     # noqa: E731
        logf = lambda v: np.log(np.maximum(v, 1.05))              # noqa: E731
        ut = arr
        u_tau = clamp(0.04 * arr)
        for _ in range(iters):
            ln = logf(y * u_tau / nu)
            g = u_tau * (ln / KAPPA + B_LOG) - ut
            dg = ln / KAPPA + B_LOG + 1.0 / KAPPA
            u_tau = clamp(u_tau - g / dg)
        return u_tau if arr.ndim else float(u_tau)

    u_tau = clamp(0.04 * ut)
    for _ in range(iters):
        ln = logf(y * u_tau / nu)
        g = u_tau * (ln / KAPPA + B_LOG) - ut
        dg = ln / KAPPA + B_LOG + 1.0 / KAPPA
        u_tau = clamp(u_tau - g / dg)
    return u_tau


# ---------------------------------------------------------------------------
# Geometry: stepped LNG Moss-type superstructure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """Axis-aligned solid block in metres, superstructure frame.

    ``x`` runs fore-aft (0 = aft end of the superstructure block, positive
    towards the bow), ``y`` is transverse (starboard positive, 0 = centreline),
    ``z`` is height above the main deck / sea-surface plane.
    """

    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    label: str = "block"

    @property
    def volume(self) -> float:
        return (max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)
                * max(0.0, self.z1 - self.z0))


def lng_moss_superstructure() -> list[Box]:
    """Simplified Moss-type LNG carrier superstructure, 50 m x 30 m x 25 m.

    Layout of a typical 4-sphere Moss carrier with aft machinery space: aft
    Moss-sphere dome trunks, deck pipe racks, main deck house, bridge deck,
    bridge wings, funnel casing, two masts and a foredeck windlass house.
    """
    return [
        Box(-14.0, -2.0, -13.0, -3.0, 0.0, 8.0, "moss_dome_ps"),
        Box(-14.0, -2.0, 3.0, 13.0, 0.0, 8.0, "moss_dome_sb"),
        Box(-2.0, 6.0, -8.0, -4.0, 0.0, 2.5, "pipe_rack_ps"),
        Box(-2.0, 6.0, 4.0, 8.0, 0.0, 2.5, "pipe_rack_sb"),
        Box(6.0, 30.0, -14.0, 14.0, 0.0, 18.0, "deck_house"),
        Box(8.0, 18.0, -11.0, 11.0, 18.0, 21.0, "bridge_deck"),
        Box(8.0, 18.0, -15.0, -11.0, 18.0, 19.5, "bridge_wing_ps"),
        Box(8.0, 18.0, 11.0, 15.0, 18.0, 19.5, "bridge_wing_sb"),
        Box(20.0, 26.0, -5.0, 5.0, 18.0, 25.0, "funnel"),
        Box(32.0, 33.5, -14.0, -12.5, 18.0, 25.0, "mast_ps"),
        Box(32.0, 33.5, 12.5, 14.0, 18.0, 25.0, "mast_sb"),
        Box(40.0, 50.0, -6.0, 6.0, 0.0, 4.0, "forecastle_house"),
    ]


@dataclass
class VoxelGeometry:
    """Voxelised superstructure for one wind angle.

    The wind always blows along ``+x`` in the simulation; a wind angle of
    ``theta`` degrees is realised by rotating the *body* about the vertical
    axis through its planform centroid.  ``theta = 0`` is head wind,
    ``theta = 90`` a beam wind from starboard.
    """

    solid: np.ndarray                       # bool (nz, ny, nx), True = solid
    dx: float                               # metre per cell
    origin: tuple[float, float, float]      # world coordinate of cell (0,0,0) corner
    angle_deg: float
    nz: int = field(init=False)
    ny: int = field(init=False)
    nx: int = field(init=False)
    solid_volume: float = field(init=False)
    area_frontal: float = field(init=False)
    area_lateral: float = field(init=False)
    area_plan: float = field(init=False)

    def __post_init__(self) -> None:
        self.nz, self.ny, self.nx = self.solid.shape
        self.solid_volume = float(self.solid.sum()) * self.dx ** 3
        # projections: x -> (z,y) frontal, y -> (z,x) lateral, z -> (y,x) plan
        self.area_frontal = float(self.solid.any(axis=2).sum()) * self.dx ** 2
        self.area_lateral = float(self.solid.any(axis=1).sum()) * self.dx ** 2
        self.area_plan = float(self.solid.any(axis=0).sum()) * self.dx ** 2

    @property
    def height(self) -> float:
        zs = np.where(self.solid.any(axis=(1, 2)))[0]
        return float(zs.max() + 1) * self.dx if zs.size else 0.0

    @property
    def breadth(self) -> float:
        ys = np.where(self.solid.any(axis=(0, 2)))[0]
        return float(ys.max() - ys.min() + 1) * self.dx if ys.size else 0.0

    @property
    def length(self) -> float:
        xs = np.where(self.solid.any(axis=(0, 1)))[0]
        return float(xs.max() - xs.min() + 1) * self.dx if xs.size else 0.0

    def to_dict(self) -> dict:
        return {
            "angle_deg": self.angle_deg,
            "dx_m": self.dx,
            "grid_xyz": [self.nx, self.ny, self.nz],
            "solid_cells": int(self.solid.sum()),
            "solid_volume_m3": self.solid_volume,
            "area_frontal_m2": self.area_frontal,
            "area_lateral_m2": self.area_lateral,
            "area_plan_m2": self.area_plan,
            "extent_m": [self.length, self.breadth, self.height],
            "area_schema": "v2",
        }


def voxelise(
    boxes: Sequence[Box],
    dx: float,
    angle_deg: float = 0.0,
    *,
    margin_upstream: float = 40.0,
    margin_downstream: float = 40.0,
    margin_side: float = 15.0,
    margin_top: float = 25.0,
) -> VoxelGeometry:
    """Stepped (staircase) voxelisation of a box-composed superstructure.

    Cell centres are tested for membership in the union of the (rotated)
    boxes: exact staircase for axis-aligned wind angles, a stepped
    approximation in between.  The domain is sized from the body extent so the
    three grid levels of a convergence study share the same physical domain.
    """
    if dx <= 0:
        raise ValueError("dx must be positive")
    if not boxes:
        raise ValueError("at least one box is required")

    xmin = min(b.x0 for b in boxes)
    xmax = max(b.x1 for b in boxes)
    ymin = min(b.y0 for b in boxes)
    ymax = max(b.y1 for b in boxes)
    zmax = max(b.z1 for b in boxes)
    xc, yc = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    # rotated bounding half-extents so every wind angle fits the same domain
    hx, hy = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
    a = math.radians(angle_deg)
    ca, sa = abs(math.cos(a)), abs(math.sin(a))
    hx_rot, hy_rot = ca * hx + sa * hy, sa * hx + ca * hy

    x_lo = xc - hx_rot - margin_upstream
    x_hi = xc + hx_rot + margin_downstream
    y_lo = yc - hy_rot - margin_side
    y_hi = yc + hy_rot + margin_side
    z_lo = 0.0
    z_hi = zmax + margin_top

    nx = int(round((x_hi - x_lo) / dx))
    ny = int(round((y_hi - y_lo) / dx))
    nz = int(round((z_hi - z_lo) / dx))

    xs = x_lo + (np.arange(nx) + 0.5) * dx
    ys = y_lo + (np.arange(ny) + 0.5) * dx
    zs = z_lo + (np.arange(nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")   # (ny, nx, nz)
    Xc, Yc = X - xc, Y - yc
    if abs(angle_deg) > 1e-12:
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        Xb, Yb = ca * Xc + sa * Yc, -sa * Xc + ca * Yc
    else:
        Xb, Yb = Xc, Yc
    Xw, Yw = Xb + xc, Yb + yc

    solid_xyz = np.zeros((ny, nx, nz), dtype=bool)
    for b in boxes:
        solid_xyz |= (
            (Xw >= b.x0) & (Xw < b.x1) & (Yw >= b.y0) & (Yw < b.y1)
            & (Z >= b.z0) & (Z < b.z1)
        )
    solid = np.transpose(solid_xyz, (2, 0, 1)).copy()   # (nz, ny, nx)
    return VoxelGeometry(solid=solid, dx=dx, origin=(x_lo, y_lo, z_lo),
                         angle_deg=angle_deg)


def cube_case(side: float = 1.0, cells_per_side: int = 10,
              domain: tuple[float, float, float] = (14.0, 6.0, 6.0),
              *, centre_height: float = 2.5) -> VoxelGeometry:
    """Single cube in uniform flow — canonical bluff-body validation geometry.

    ``centre_height`` (in cube sides) places the cube in the free stream so the
    drag coefficient can be compared with free-stream references; set it to 0
    for a wall-mounted cube.
    """
    dx = side / cells_per_side
    x0, y0, z0 = 4.0 * side, -0.5 * side, centre_height * side
    box = Box(x0, x0 + side, y0, y0 + side, z0, z0 + side, "cube")
    return voxelise([box], dx, 0.0, margin_upstream=x0,
                    margin_downstream=domain[0] - x0 - side,
                    margin_side=0.5 * domain[1] - 0.5 * side,
                    margin_top=domain[2] - z0 - side)


# ---------------------------------------------------------------------------
# Physical <-> lattice units
# ---------------------------------------------------------------------------

@dataclass
class LatticeUnits:
    """Similarity mapping between physical and lattice units.

    ``dx`` (m) and ``dt`` (s) are chosen so the inlet velocity maps onto a
    stable lattice velocity ``u_lat`` (default 0.08, Mach ~ 0.14); the
    kinematic viscosity then follows from ``nu_lat = nu_phys dt / dx^2``.
    """

    dx: float
    dt: float
    u_lat: float
    u_phys: float
    nu_phys: float = NU_AIR
    rho_phys: float = RHO_AIR

    @classmethod
    def from_velocity(cls, dx: float, u_phys: float, u_lat: float = 0.08,
                      nu_phys: float = NU_AIR,
                      rho_phys: float = RHO_AIR) -> "LatticeUnits":
        if not 0.0 < u_lat < 0.2:
            raise ValueError("u_lat must lie in (0, 0.2) for LBM stability")
        if u_phys <= 0:
            raise ValueError("u_phys must be positive")
        return cls(dx=dx, dt=dx * u_lat / u_phys, u_lat=u_lat, u_phys=u_phys,
                   nu_phys=nu_phys, rho_phys=rho_phys)

    @classmethod
    def from_reynolds(cls, dx: float, re_target: float, length_m: float,
                      u_lat: float = 0.08, nu_phys: float = NU_AIR,
                      rho_phys: float = RHO_AIR) -> "LatticeUnits":
        """Units that reproduce a target Reynolds number at fixed ``u_lat``."""
        if re_target <= 0:
            raise ValueError("re_target must be positive")
        length_cells = length_m / dx
        nu_lat = u_lat * length_cells / re_target
        u_phys = nu_phys * u_lat / (nu_lat * dx)
        return cls(dx=dx, dt=dx * u_lat / u_phys, u_lat=u_lat, u_phys=u_phys,
                   nu_phys=nu_phys, rho_phys=rho_phys)

    @property
    def nu_lat(self) -> float:
        return self.nu_phys * self.dt / self.dx ** 2

    @property
    def tau0(self) -> float:
        return 0.5 + 3.0 * self.nu_lat

    def re_lattice(self, length_cells: float) -> float:
        return self.u_lat * length_cells / self.nu_lat

    def re_phys(self, length_m: float) -> float:
        return self.u_phys * length_m / self.nu_phys

    def force_to_physical(self, f_lat):
        """Lattice force -> newtons (f_lat has units rho_lat l^4/t^2)."""
        return np.asarray(f_lat) * (self.rho_phys / 1.0) * (self.dx ** 4 / self.dt ** 2)

    def to_dict(self) -> dict:
        return {"dx_m": self.dx, "dt_s": self.dt, "u_lat": self.u_lat,
                "u_phys_ms": self.u_phys, "nu_lat": self.nu_lat,
                "tau0": self.tau0, "nu_phys_m2s": self.nu_phys,
                "rho_phys_kgm3": self.rho_phys}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class ForceSample:
    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float
    fx_pressure: float = 0.0
    fy_pressure: float = 0.0
    fx_wall: float = 0.0
    fy_wall: float = 0.0


class WindLoadSolver:
    """Wall-modelled Smagorinsky-LBM solver for external bluff-body wind flow.

    Force integration is gather based: the momentum-exchange links (fluid cell
    -> solid neighbour) and the wall-adjacent cells are indexed once at setup,
    so the per-step cost is dominated by the lattice update itself.
    """

    def __init__(
        self,
        geometry: VoxelGeometry,
        units: LatticeUnits,
        *,
        cs_smag: float = 0.16,
        wall_model: bool = True,
        wall_feedback: bool = True,
        tau_min: float = 0.51,
        tau_max: float = 1.9,
        u_clip: float = 0.25,
        normalise_outflow: bool = True,
        ref_areas: tuple[float, float] | None = None,
        ref_breadth_height: tuple[float, float] | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.geom = geometry
        # reference areas of the *unrotated* body (ship convention: A_F and A_L
        # are the 0 deg frontal / lateral projected areas, independent of the
        # wind angle used for the run)
        self.ref_AF, self.ref_AL = (ref_areas if ref_areas
                                   else (geometry.area_frontal, geometry.area_lateral))
        self.ref_B, self.ref_H = (ref_breadth_height
                                  if ref_breadth_height
                                  else (geometry.breadth, geometry.height))
        self.units = units
        self.cs_smag = float(cs_smag)
        self.use_wall_model = bool(wall_model)
        self.wall_feedback = bool(wall_feedback)
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.u_clip = float(u_clip)
        self.normalise_outflow = bool(normalise_outflow)
        self.device = torch.device(device)
        self.dtype = dtype

        nz, ny, nx = geometry.solid.shape
        self.shape = (nz, ny, nx)
        self.ncell = nz * ny * nx
        self.solid = torch.as_tensor(geometry.solid, device=self.device)
        self.fluid = ~self.solid

        self.c = torch.tensor(_D3Q19_C, dtype=dtype, device=self.device)
        self.w = torch.tensor(_D3Q19_W, dtype=dtype, device=self.device)

        # --- momentum-exchange links (fluid x with solid at x + c_i) --------
        self.link_idx: list[torch.Tensor | None] = [None] * 19
        self.link_x: list[torch.Tensor | None] = [None] * 19
        self.link_y: list[torch.Tensor | None] = [None] * 19
        self.link_z: list[torch.Tensor | None] = [None] * 19
        self.bounce_mask: list[torch.Tensor] = [None] * 19   # type: ignore
        ox, oy, oz = geometry.origin
        zz = torch.arange(nz, dtype=dtype, device=self.device) + 0.5
        yy = torch.arange(ny, dtype=dtype, device=self.device) + 0.5
        xx = torch.arange(nx, dtype=dtype, device=self.device) + 0.5
        self.X = (xx * geometry.dx + ox)[None, None, :]
        self.Y = (yy * geometry.dx + oy)[None, :, None]
        self.Z = (zz * geometry.dx + oz)[:, None, None]
        Xf = self.X.expand(self.shape).reshape(-1)
        Yf = self.Y.expand(self.shape).reshape(-1)
        Zf = self.Z.expand(self.shape).reshape(-1)
        for i, ci in enumerate(_D3Q19_C):
            ahead = self._shift_mask(self.solid, ci)          # x - c_i is solid
            self.bounce_mask[i] = ahead
            enter = self.fluid & self._shift_mask(self.solid, (-ci[0], -ci[1], -ci[2]))
            idx = enter.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            if idx.numel():
                self.link_idx[i] = idx
                self.link_x[i] = Xf[idx]
                self.link_y[i] = Yf[idx]
                self.link_z[i] = Zf[idx]

        # --- wall-adjacent cells (log-law wall model) -----------------------
        self.wall_idx: list[torch.Tensor] = []
        self.wall_n: list[tuple[int, int, int]] = []
        for axis in range(3):
            for sgn in (1, -1):
                n = [0, 0, 0]
                n[axis] = sgn
                adj = self.fluid & self._shift_mask(self.solid, tuple(n))
                idx = adj.reshape(-1).nonzero(as_tuple=False).squeeze(1)
                if idx.numel():
                    self.wall_idx.append(idx)
                    self.wall_n.append(tuple(n))
        self.n_wall_cells = int(sum(len(i) for i in self.wall_idx))
        self._wall_force = torch.zeros((3,) + self.shape, dtype=dtype, device=self.device)

        self.u_in = units.u_lat
        self.nu0 = units.nu_lat
        self.tau0 = units.tau0
        self.step_count = 0
        self.time = 0.0
        self.force_history: list[ForceSample] = []
        self.mass_history: list[tuple[float, float]] = []

        rho = torch.ones(self.shape, dtype=dtype, device=self.device)
        u = torch.zeros((3,) + self.shape, dtype=dtype, device=self.device)
        u[0] = self.u_in
        self.f = self.equilibrium(rho, u)
        self.apply_boundaries(self.f)
        self.f[:, self.solid] = 0.0

    # -- helpers ----------------------------------------------------------
    def _shift_mask(self, m: torch.Tensor, shift: tuple[int, int, int]) -> torch.Tensor:
        """Return ``m`` shifted so entry ``x`` holds ``m[x - shift]``."""
        dims, sh = [], []
        for axis, s in enumerate(shift):        # axis 0 -> x, 1 -> y, 2 -> z
            if s:
                dims.append(2 - axis)           # tensor dims are (z, y, x)
                sh.append(s)
        if not dims:
            return m.clone()
        return torch.roll(m, shifts=tuple(sh), dims=tuple(dims))

    def equilibrium(self, rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cv = self.c.view(19, 3, 1, 1, 1)
        cu = cv[:, 0] * u[0] + cv[:, 1] * u[1] + cv[:, 2] * u[2]
        usq = (u ** 2).sum(0)
        shape = (19,) + (1,) * len(self.shape)
        return self.w.view(shape) * rho.unsqueeze(0) * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0)
        )

    def macroscopic(self, f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rho = f.sum(0)
        cv = self.c.view(19, 3, 1, 1, 1)
        u = torch.stack([(cv[:, k] * f).sum(0) for k in range(3)])
        return rho, u / rho.clamp_min(1e-8)

    # -- physics ----------------------------------------------------------
    def smagorinsky_viscosity(self, f: torch.Tensor, feq: torch.Tensor,
                              rho: torch.Tensor) -> torch.Tensor:
        """Smagorinsky eddy viscosity from the non-equilibrium stress."""
        fne = f - feq
        cx, cy, cz = self.c[:, 0], self.c[:, 1], self.c[:, 2]
        v = (19, 1, 1, 1)
        P = {}
        for key, fct in (("xx", cx * cx), ("yy", cy * cy), ("zz", cz * cz),
                         ("xy", cx * cy), ("xz", cx * cz), ("yz", cy * cz)):
            P[key] = (fct.view(*v) * fne).sum(0)
        pref = -1.5 / (rho.clamp_min(1e-6) * self.tau0)
        Sxx, Syy, Szz = pref * P["xx"], pref * P["yy"], pref * P["zz"]
        Sxy, Sxz, Syz = pref * P["xy"], pref * P["xz"], pref * P["yz"]
        s2 = (Sxx * Sxx + Syy * Syy + Szz * Szz
              + 2.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz))
        Smag = torch.sqrt(torch.clamp(2.0 * s2, min=0.0))
        del P, Sxx, Syy, Szz, Sxy, Sxz, Syz
        return ((self.cs_smag ** 2) * Smag).clamp(min=0.0, max=0.5)

    def wall_model(self, u: torch.Tensor, rho: torch.Tensor,
                   nu: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Log-law wall shear on wall-adjacent cells (lattice units).

        ``u_tau`` solves ``u_t = u_tau (ln(y u_tau / nu)/kappa + B)`` at the
        first cell centre ``y = 0.5 dx``.  The reaction on the fluid is
        scattered into ``self._wall_force`` (per-cell force) so that the
        unresolved wall shear is *fed back* into the momentum balance, as
        required for a wall-modelled LES (not merely a post-processing term).
        """
        F = np.zeros(3)
        M = np.zeros(3)
        self._wall_force.zero_()
        if not self.use_wall_model or not self.wall_idx:
            return F, M
        y = 0.5
        ux, uy, uz = u[0].reshape(-1), u[1].reshape(-1), u[2].reshape(-1)
        rr = rho.reshape(-1)
        nn = nu.reshape(-1)
        fx_buf, fy_buf, fz_buf = self._wall_force[0].reshape(-1), \
            self._wall_force[1].reshape(-1), self._wall_force[2].reshape(-1)
        for idx, n in zip(self.wall_idx, self.wall_n):
            un = n[0] * ux[idx] + n[1] * uy[idx] + n[2] * uz[idx]
            tx = ux[idx] - un * n[0]
            ty = uy[idx] - un * n[1]
            tz = uz[idx] - un * n[2]
            ut = torch.sqrt(tx * tx + ty * ty + tz * tz + 1e-30)
            nu_l = nn[idx]
            u_tau = solve_friction_velocity(ut, y, nu_l)
            scale = rr[idx] * u_tau * u_tau / ut        # |tau_w| / |u_t|
            wfx, wfy, wfz = -scale * tx, -scale * ty, -scale * tz
            F -= np.array([wfx.sum().item(), wfy.sum().item(), wfz.sum().item()])
            xw = self.X.expand(self.shape).reshape(-1)[idx]
            yw = self.Y.expand(self.shape).reshape(-1)[idx]
            zw = self.Z.expand(self.shape).reshape(-1)[idx]
            Mx = (yw * wfz - zw * wfy).sum().item()
            My = (zw * wfx - xw * wfz).sum().item()
            Mz = (xw * wfy - yw * wfx).sum().item()
            M += np.array([Mx, My, Mz])
            if self.wall_feedback:
                coords = (idx,)
                fx_buf.index_put_(coords, -wfx, accumulate=True)
                fy_buf.index_put_(coords, -wfy, accumulate=True)
                fz_buf.index_put_(coords, -wfz, accumulate=True)
        return F, M

    def momentum_exchange(self, f_col: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Ladd momentum-exchange force and moment on the stepped surface."""
        F = np.zeros(3)
        M = np.zeros(3)
        for i in range(1, 19):
            idx = self.link_idx[i]
            if idx is None:
                continue
            p = f_col[i].reshape(-1)[idx]
            s = p.sum().item()
            if s == 0.0:
                continue
            ci = _D3Q19_C[i]
            F += 2.0 * s * np.array(ci, dtype=float)
            sx = (p * self.link_x[i]).sum().item()
            sy = (p * self.link_y[i]).sum().item()
            sz = (p * self.link_z[i]).sum().item()
            M += 2.0 * np.cross([sx, sy, sz], ci)
        return F, M

    # -- time stepping ----------------------------------------------------
    def step(self) -> ForceSample:
        f = self.f
        rho, u = self.macroscopic(f)
        usq = (u ** 2).sum(0)
        too_fast = usq > self.u_clip ** 2
        if bool(too_fast.any()):
            fac = torch.where(too_fast,
                              self.u_clip / torch.sqrt(torch.clamp(usq, min=1e-12)),
                              torch.ones_like(usq))
            u = u * fac.unsqueeze(0)
        # global mass budget on the inlet/outlet planes (cheap reductions)
        jx = rho * u[0] * self.fluid
        self.mass_history.append((float(jx[:, :, 0].sum().item()),
                                  float(jx[:, :, -1].sum().item())))
        nu_t = self.smagorinsky_viscosity(
            f, self.equilibrium(rho, u), rho)
        nu_eff = self.nu0 + nu_t
        tau = (0.5 + 3.0 * nu_eff).clamp(self.tau_min, self.tau_max)

        # wall-model shear -> force on fluid (fed back) and on solid (reported)
        F_w, M_w = self.wall_model(u, rho, nu_eff)
        if self.wall_feedback and self.n_wall_cells:
            u = u + 0.5 * self._wall_force / rho.clamp_min(1e-6).unsqueeze(0)

        feq = self.equilibrium(rho, u)
        f_col = f - (f - feq) / tau.unsqueeze(0)

        F_p, M_p = self.momentum_exchange(f_col)

        # streaming with half-way bounce-back
        out = torch.empty_like(f_col)
        out[0] = f_col[0]
        for i in range(1, 19):
            ci = _D3Q19_C[i]
            dims, sh = [], []
            if ci[2]:
                dims.append(0); sh.append(ci[2])
            if ci[1]:
                dims.append(1); sh.append(ci[1])
            if ci[0]:
                dims.append(2); sh.append(ci[0])
            src = torch.roll(f_col[i], shifts=tuple(sh), dims=tuple(dims)) if dims else f_col[i]
            out[i] = torch.where(self.bounce_mask[i], f_col[OPP[i]], src)
        out[:, self.solid] = 0.0
        self.apply_boundaries(out)
        out[:, self.solid] = 0.0        # deck-plane BCs overlap the solid body
        self.f = out
        self.step_count += 1
        self.time += self.units.dt

        sample = ForceSample(
            fx=F_p[0] + F_w[0], fy=F_p[1] + F_w[1], fz=F_p[2] + F_w[2],
            mx=M_p[0] + M_w[0], my=M_p[1] + M_w[1], mz=M_p[2] + M_w[2],
            fx_pressure=F_p[0], fy_pressure=F_p[1],
            fx_wall=F_w[0], fy_wall=F_w[1],
        )
        self.force_history.append(sample)
        return sample

    def apply_boundaries(self, f: torch.Tensor) -> None:
        """Velocity inlet, zero-gradient outlet, free-slip top/lateral/deck."""
        rho = torch.ones(self.shape, dtype=self.dtype, device=self.device)
        u = torch.zeros((3,) + self.shape, dtype=self.dtype, device=self.device)
        u[0] = self.u_in
        f[:, :, :, 0] = self.equilibrium(rho, u)[:, :, :, 0]
        f[:, :, :, -1] = f[:, :, :, -2]
        f[:, -1, :, :] = f[list(PERM_Z), -1, :, :]
        f[:, 0, :, :] = f[list(PERM_Z), 0, :, :]
        f[:, :, -1, :] = f[list(PERM_Y), :, -1, :]
        f[:, :, 0, :] = f[list(PERM_Y), :, 0, :]
        if self.normalise_outflow:
            self._normalise_outflow(f)

    def _normalise_outflow(self, f: torch.Tensor) -> None:
        """Pin the outlet plane density to 1 (density-normalised outflow).

        A pure zero-gradient outlet lets compressibility waves accumulate mass
        in the domain; rescaling every outlet cell to ``rho = 1`` (velocity
        unchanged) removes that drift without the aggressive global feedback of
        a flux controller, which is numerically unstable for a bluff-body wake.
        """
        rho, _ = self.macroscopic(f)
        rho_out = rho[:, :, -1]
        fac = torch.where(rho_out > 1e-8, 1.0 / rho_out.clamp_min(1e-8),
                          torch.ones_like(rho_out))
        f[:, :, :, -1] *= fac.unsqueeze(0)

    # -- diagnostics ------------------------------------------------------
    def mass_flux_error(self) -> float:
        """|Q_out - Q_in| / Q_in for the current state (lattice units)."""
        rho, u = self.macroscopic(self.f)
        fluid = self.fluid
        q_in = float((rho * u[0] * fluid)[:, :, 0].sum().item())
        q_out = float((rho * u[0] * fluid)[:, :, -1].sum().item())
        if abs(q_in) < 1e-12:
            return float("inf")
        return abs(q_out - q_in) / abs(q_in)

    def mass_budget(self, from_index: int = 0) -> dict:
        """Steady-state mass balance: time-averaged inlet vs outlet flux.

        Global mass is conserved exactly by the streaming step; any residual is
        created/destroyed by the inlet/outlet boundary conditions, so the
        physically meaningful measure is the *time-averaged* imbalance over the
        averaging window (the instantaneous value equals dM/dt of the domain).
        """
        hist = self.mass_history[from_index:]
        if not hist:
            raise RuntimeError("empty mass window")
        q_in = np.array([h[0] for h in hist])
        q_out = np.array([h[1] for h in hist])
        m_in, m_out = float(q_in.mean()), float(q_out.mean())
        denom = abs(m_in) if abs(m_in) > 1e-12 else 1.0
        return {
            "q_in_mean": m_in, "q_out_mean": m_out,
            "mean_rel_error": abs(m_out - m_in) / denom,
            "inst_rel_error_max": float(np.max(np.abs(q_out - q_in)) / denom),
            "n_samples": len(hist),
        }

    def probe(self, x_m: float, y_m: float, z_m: float) -> np.ndarray:
        """Velocity at a physical probe location (metres, geometry frame)."""
        g = self.geom
        i = int(round((x_m - g.origin[0]) / g.dx - 0.5))
        j = int(round((y_m - g.origin[1]) / g.dx - 0.5))
        k = int(round((z_m - g.origin[2]) / g.dx - 0.5))
        i = min(max(i, 0), self.shape[2] - 1)
        j = min(max(j, 0), self.shape[1] - 1)
        k = min(max(k, 0), self.shape[0] - 1)
        _, u = self.macroscopic(self.f)
        return np.array([u[0, k, j, i].item(), u[1, k, j, i].item(),
                         u[2, k, j, i].item()])

    def y_plus_first_cell(self, u_tau_phys: float) -> float:
        return u_tau_phys * 0.5 * self.geom.dx / self.units.nu_phys

    # -- post-processing --------------------------------------------------
    def coefficient_summary(self, from_index: int = 0) -> dict:
        hist = self.force_history[from_index:]
        if not hist:
            raise RuntimeError("empty averaging window")
        keys = ("fx", "fy", "fz", "mx", "my", "mz", "fx_pressure",
                "fy_pressure", "fx_wall", "fy_wall")
        arr = {k: np.array([getattr(s, k) for s in hist]) for k in keys}
        mean = {k: float(v.mean()) for k, v in arr.items()}
        std = {k: float(v.std()) for k, v in arr.items()}
        q = 0.5 * self.units.u_lat ** 2                     # rho_lat = 1
        A_F = self.ref_AF / self.geom.dx ** 2
        A_L = self.ref_AL / self.geom.dx ** 2
        B_l = self.ref_B / self.geom.dx
        H_l = self.ref_H / self.geom.dx
        # the solver works in the *flow* frame (wind along +x, body rotated by
        # theta); the ship coefficients need the body frame, i.e. the force and
        # moment vectors rotated by -theta about z
        th = math.radians(self.geom.angle_deg)
        c, s_ = math.cos(th), math.sin(th)

        def to_ship(fx, fy, mx, my, mz):
            return (c * fx + s_ * fy, -s_ * fx + c * fy,
                    c * mx + s_ * my, mz)

        fx_s, fy_s, mx_s, mz_s = to_ship(mean["fx"], mean["fy"], mean["mx"],
                                         mean["my"], mean["mz"])
        fxp_s, fyp_s, _, _ = to_ship(mean["fx_pressure"], mean["fy_pressure"],
                                     0.0, 0.0, 0.0)
        fxw_s, fyw_s, _, _ = to_ship(mean["fx_wall"], mean["fy_wall"],
                                     0.0, 0.0, 0.0)
        sd = np.cov(np.vstack([arr["fx"], arr["fy"]])) if len(hist) > 1 else np.zeros((2, 2))
        var_x = c * c * sd[0, 0] + 2 * c * s_ * sd[0, 1] + s_ * s_ * sd[1, 1]
        var_y = s_ * s_ * sd[0, 0] - 2 * c * s_ * sd[0, 1] + c * c * sd[1, 1]
        return {
            "C_X": fx_s / (q * A_F),
            "C_Y": fy_s / (q * A_L),
            "C_K": mx_s / (q * A_L * H_l),
            "C_N": mz_s / (q * A_L * B_l),
            "C_X_std": math.sqrt(max(var_x, 0.0)) / (q * A_F),
            "C_Y_std": math.sqrt(max(var_y, 0.0)) / (q * A_L),
            "C_X_pressure": fxp_s / (q * A_F),
            "C_X_wall": fxw_s / (q * A_F),
            "C_Y_pressure": fyp_s / (q * A_L),
            "C_Y_wall": fyw_s / (q * A_L),
            "world_frame_mean": {"fx": mean["fx"], "fy": mean["fy"],
                                 "mx": mean["mx"], "my": mean["my"],
                                 "mz": mean["mz"]},
            "n_samples": len(hist),
            "q_lattice": q,
            "area_frontal_cells": A_F,
            "area_lateral_cells": A_L,
        }

    def run(self, n_steps: int, *, average_from: int | None = None) -> dict:
        avg_from = self.step_count if average_from is None else average_from
        for _ in range(n_steps):
            self.step()
        return self.coefficient_summary(from_index=avg_from)

    # -- state I/O (restart across cluster jobs) --------------------------
    def save_state(self, path: str) -> None:
        np.savez_compressed(
            path, f=self.f.cpu().numpy(), step_count=self.step_count,
            force=np.array([[s.fx, s.fy, s.fz, s.mx, s.my, s.mz,
                             s.fx_pressure, s.fy_pressure, s.fx_wall, s.fy_wall]
                            for s in self.force_history], dtype=np.float64),
        )

    def load_state(self, path: str) -> None:
        d = np.load(path)
        self.f = torch.as_tensor(d["f"], device=self.device, dtype=self.dtype)
        self.step_count = int(d["step_count"])
        arr = d["force"]
        self.force_history = [ForceSample(*row) for row in arr]

def prolongate_state(coarse_f: np.ndarray, rho_u: tuple[np.ndarray, np.ndarray],
                     factor: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Upsample a coarse macroscopic state by ``factor`` (nearest neighbour).

    Used to initialise a finer grid from a converged coarser solution, which
    removes 1-2 flow-through times of transient from a refinement study.
    """
    rho, u = rho_u
    def up(a: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(a, factor, axis=-3), factor, axis=-2)[
            ..., : a.shape[-1] * factor
        ]
    # z, y, x axes are the three spatial axes of (nz, ny, nx)
    rho_f = np.repeat(np.repeat(rho, factor, 0), factor, 1)
    rho_f = np.repeat(rho_f, factor, 2)
    u_f = np.repeat(np.repeat(np.repeat(u, factor, 1), factor, 2), factor, 3)
    return rho_f, u_f


# ---------------------------------------------------------------------------
# Wall-time / statistics helpers
# ---------------------------------------------------------------------------

def strouhal_from_series(signal: Sequence[float] | np.ndarray, dt: float,
                         length_m: float, u_phys: float,
                         *, detrend: bool = True) -> dict:
    """Dominant Strouhal number of a force / velocity time series.

    ``St = f_peak L / U`` with ``L`` the body height (or breadth) in metres.
    Returns the peak frequency, the Welch-style spectral peak ratio and the
    band-limited energy fraction so the result can be audited.
    """
    x = np.asarray(signal, dtype=float)
    if x.size < 16:
        raise ValueError("series too short for a spectral estimate")
    if detrend:
        x = x - x.mean()
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    f_peak = float(freqs[k])
    total = float(spec.sum())
    band = max(1, int(0.1 * spec.size))
    lo, hi = max(0, k - band), min(spec.size, k + band + 1)
    return {
        "f_peak_hz": f_peak,
        "St": f_peak * length_m / u_phys,
        "peak_energy_fraction": float(spec[lo:hi].sum() / total) if total > 0 else 0.0,
        "n_samples": int(x.size),
        "dt_s": dt,
        "freqs_hz": freqs.tolist(),
        "spectrum": (spec / total).tolist() if total > 0 else spec.tolist(),
    }


def gci_study(f_coarse: float, f_medium: float, f_fine: float,
              r: float = 2.0, safety: float = 1.25) -> dict:
    """Roache grid-convergence index for a three-level refinement study.

    ``f`` are the same integral quantity on the three grids (coarse -> fine).
    Returns the observed order of accuracy ``p``, the extrapolated value and
    the fine-grid GCI (``U_GCI``), which is the reported uncertainty.
    """
    if r <= 1.0:
        raise ValueError("refinement ratio must exceed 1")
    d21 = f_medium - f_coarse
    d32 = f_fine - f_medium
    if abs(d32) < 1e-14 or abs(d21) < 1e-14:
        p = float("nan")
    else:
        ratio = abs(d21 / d32)
        p = math.log(ratio) / math.log(r) if ratio > 0 else float("nan")
    if not math.isfinite(p) or p <= 0:
        p_eff = 1.0
        p_status = "oscillatory_or_nonmonotone"
    else:
        p_eff = min(p, 6.0)
        p_status = "ok"
    e_fine = abs(d32) / abs(f_fine) if f_fine != 0 else float("inf")
    gci_fine = safety * e_fine / max(abs(r ** p_eff - 1.0), 1e-6)
    return {
        "p_observed": p, "p_used": p_eff, "p_status": p_status,
        "r": r, "safety_factor": safety,
        "relative_change_medium_fine": e_fine,
        "U_GCI": gci_fine,
        "f_extrapolated": f_fine + (f_fine - f_medium) / max(r ** p_eff - 1.0, 1e-6),
    }