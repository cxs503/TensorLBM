"""P48-N01 - curved-wall boundary-format force-error calibration (TensorLBM).

Calibration bench for curved-wall boundary treatments: canonical flows whose wall
force (or wall torque) is known analytically, against which three wall formats are
measured through two independent force routes.

Wall formats (the ``order`` factor)
    * ``halfway``            -- classical half-way bounce-back (q := 1/2)
    * ``bouzidi_linear``     -- 2-point interpolated bounce-back (BFL 2001)
    * ``bouzidi_quadratic``  -- 3-point interpolated bounce-back (2nd order)

All three are implemented through a *single* geometric statement, which removes
the usual branch duplication (``q < 1/2`` interpolation / ``q >= 1/2`` closure):

    the population arriving at fluid node ``x_f`` from the wall equals the
    post-collision population at the specular *mirror of the solid node*,
    ``x_m = x_f + (2q - 1) c_i``, reconstructed by 2- or 3-point Lagrange
    interpolation over the nearest available nodes of the link.

Stencil choice (``xi = 2q - 1`` is the mirror offset in link units):

    format      xi < 0 (q < 1/2)     xi >= 0 (q >= 1/2)
    ----------  -------------------  ---------------------
    linear      offsets {0, -1}      offsets {0, +1}
    quadratic   offsets {-2,-1, 0}   offsets {-1, 0, +1}

The ``+1`` node is the solid-side neighbour: its value is the population that the
wall returned on the previous step (Bouzidi's two-time-level closure), kept in
``Solver.buf``.  Both stencils *interpolate* the mirror point, so all weights are
bounded by 1 and the scheme stays stable, while at ``xi = 0`` (``q = 1/2``) every
format collapses exactly onto half-way bounce-back -- the C6 negative control.
Extrapolating the mirror point from the fluid side for ``q > 1/2`` was tried
first and is unstable (weights up to 3 on the outgoing population); the buffered
interpolation replaces it.

A wall moving with velocity ``u_w`` enters through the Ladd momentum correction
``-2 w_i rho_w (c_i . u_w) / c_s^2`` with ``c_i`` pointing into the wall, added
*after* the interpolation so that the momentum-exchange budget
``F = sum_links (f_i^+ + f_opp) c_i`` stays an exact Newton statement and the
buffer keeps the interpolation-only value.

Force routes (the ``force_method`` factor)
    * ``momentum_exchange`` -- link-wise momentum exchange (ME), one Newton
      budget per cut link, torque arm at the analytic wall crossing point.
    * ``stress_integral``   -- control-surface stress integration (CV),
      ``F_j = -closed_integral [rho u_j u_k + p d_jk - mu(d_k u_j + d_j u_k)] n_k dS``.

Reference cases
    ``stokes_second``    oscillating flat plate: ``u = U0 exp(-y/d) cos(wt - y/d)``
    ``plane_couette``    flat-wall degenerate limit: ``tau_w = mu U / H`` (negative control)
    ``annulus_couette``  Taylor-Couette annulus: exact NS solution ``u_t = A r + B/r``
                         and exact torque ``T = 4 pi mu Omega R1^2 R2^2 /(R2^2-R1^2)``
    ``sphere_drag``      uniform flow past a sphere: Schiller-Naumann / experimental anchor

All state is plain torch tensors (float32); the drivers return JSON-ready dicts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .d3q19 import C as _C
from .d3q19 import OPPOSITE as _OPP
from .d3q19 import W as _W

# ---------------------------------------------------------------------------
# lattice constants
# ---------------------------------------------------------------------------

CS2 = 1.0 / 3.0
BOUNDARY_FORMATS = ("halfway", "bouzidi_linear", "bouzidi_quadratic")
FORCE_METHODS = ("momentum_exchange", "stress_integral")
COLLISION_MODELS = ("bgk", "regularized")


def schiller_naumann(reynolds: float) -> float:
    """Schiller-Naumann drag correlation ``24/Re (1 + 0.15 Re^0.687)``."""
    return 24.0 / reynolds * (1.0 + 0.15 * reynolds ** 0.687)


#: C2 anchors: Re=100 -> 1.09 (Schiller-Naumann), Re=1000 -> 0.47 (experimental
#: value quoted by the campaign metadata; the correlation value 0.428 is carried
#: along as a transparency cross-reference in the report).
CD_ANCHORS = {100.0: 1.09, 1000.0: 0.47}


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


_C_RAW = [tuple(int(v) for v in row) for row in _C.tolist()]
_OPP_RAW = [int(v) for v in _OPP.tolist()]
_W_RAW = [float(v) for v in _W.tolist()]
_N_DIR = len(_C_RAW)

GEOMETRY_KINDS = ("sphere", "cylinder", "annulus", "plane")

#: Wall-frame -> lab-frame shift of the solid-side closure population (see
#: ``Solver.apply_cut_links``).  Kept as a module flag so the A/B evidence of
#: artifacts/annulus_couette_verification_20261109 can be reproduced in one process.
MOVING_WALL_CLOSURE_SHIFT = True


# ---------------------------------------------------------------------------
# geometry: solid mask + flattened cut-link table with analytic fractions
# ---------------------------------------------------------------------------


@dataclass
class Geometry:
    """Solid mask + flat cut-link table of one body configuration."""

    mask: torch.Tensor                  # True = solid, shape (nz, ny, nx)
    kind: str
    spec: dict[str, Any] = field(default_factory=dict)
    dir: torch.Tensor = None            # (L,) lattice direction of the cut link
    opp: torch.Tensor = None            # (L,) opposite direction
    idx: torch.Tensor = None            # (L,) flat index of the fluid node
    q: torch.Tensor = None              # (L,) analytic link fraction in (0, 1]
    idx_m1: torch.Tensor = None         # (L,) flat index x_f - c_i
    ok_m1: torch.Tensor = None
    idx_m2: torch.Tensor = None         # (L,) flat index x_f - 2 c_i
    ok_m2: torch.Tensor = None
    xw: torch.Tensor = None             # (L,) wall crossing point
    yw: torch.Tensor = None
    zw: torch.Tensor = None
    uw: torch.Tensor = None             # (L, 3) wall-velocity pattern
    group_id: torch.Tensor = None       # (L,) wall partition index
    groups: tuple[str, ...] = ()
    torque_center: tuple[float, float] = (0.0, 0.0)
    n_nodes: int = 0

    def n_links(self) -> int:
        return 0 if self.idx is None else int(self.idx.numel())

    def summary(self) -> dict[str, Any]:
        q = self.q if self.q is not None else torch.zeros(1)
        counts = {g: int((self.group_id == i).sum().item()) for i, g in enumerate(self.groups)}
        return {
            "kind": self.kind,
            "spec": self.spec,
            "n_links": self.n_links(),
            "n_solid": int(self.mask.sum().item()),
            "links_per_group": counts,
            "q_min": float(q.min().item()),
            "q_max": float(q.max().item()),
            "q_mean": float(q.mean().item()),
            "q_frac_below_half": float((q < 0.5).to(torch.float64).mean().item()),
        }


def _analytic_q(
    spec: dict[str, Any],
    ix: torch.Tensor,
    iy: torch.Tensor,
    iz: torch.Tensor,
    c: tuple[int, int, int],
) -> torch.Tensor:
    """Analytic fraction ``t in (0, 1]`` of the link at which the wall is crossed."""
    kind = spec["kind"]
    cx, cy, cz = (float(v) for v in c)

    if kind in ("sphere", "cylinder"):
        cz0, cy0, cx0 = spec["center"]
        px = ix.to(torch.float64) - float(cx0)
        py = iy.to(torch.float64) - float(cy0)
        if kind == "sphere":
            pz = iz.to(torch.float64) - float(cz0)
            a = cx * cx + cy * cy + cz * cz
            b = 2.0 * (cx * px + cy * py + cz * pz)
            c0 = px * px + py * py + pz * pz - float(spec["radius"]) ** 2
        else:  # z-invariant body: the crossing lives in the x-y projection
            a = cx * cx + cy * cy
            b = 2.0 * (cx * px + cy * py)
            c0 = px * px + py * py - float(spec["radius"]) ** 2
        if a <= 1e-12:
            return torch.full_like(c0, 0.5)
        disc = (b * b - 4.0 * a * c0).clamp_min(0.0)
        root = torch.sqrt(disc)
        t1 = (-b - root) / (2.0 * a)
        t2 = (-b + root) / (2.0 * a)
        eps = 1e-9
        cand = torch.where(t1 > eps, t1, torch.where(t2 > eps, t2, torch.full_like(t1, 0.5)))
        return cand.clamp(1e-6, 1.0)

    if kind == "plane":
        y_w = spec["y_bottom"] if cy < 0.0 else spec["y_top"]
        if cy == 0.0:
            return torch.full_like(ix.to(torch.float64), 0.5)
        q = (float(y_w) - iy.to(torch.float64)) / float(cy)
        return q.clamp(1e-6, 1.0)

    raise ValueError(f"unknown geometry kind {kind!r}")


def build_geometry(
    kind: str,
    nz: int,
    ny: int,
    nx: int,
    spec: dict[str, Any],
    format: str = "bouzidi_quadratic",
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Geometry:
    """Build the solid mask and the flat cut-link table of one configuration."""
    if kind not in GEOMETRY_KINDS:
        raise ValueError(f"unknown geometry kind {kind!r}")
    spec = dict(spec)
    spec["kind"] = kind
    dev = torch.device(device)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    if kind in ("sphere", "cylinder", "annulus"):
        cz0, cy0, cx0 = spec["center"]
        px, py, pz = xx - float(cx0), yy - float(cy0), zz - float(cz0)
        if kind == "sphere":
            mask = px * px + py * py + pz * pz <= float(spec["radius"]) ** 2
        elif kind == "cylinder":
            mask = px * px + py * py <= float(spec["radius"]) ** 2
        else:  # annulus: solid inside R1 and outside R2
            rr2 = px * px + py * py
            mask = (rr2 <= float(spec["radius"]) ** 2) | (
                rr2 >= float(spec["radius_outer"]) ** 2
            )
    else:  # plane channel
        mask = (yy <= float(spec["y_bottom"])) | (yy >= float(spec["y_top"]))

    mask = mask.to(dev)
    fluid_flat = (~mask).reshape(-1)
    nyx = ny * nx

    cols: dict[str, list[torch.Tensor]] = {k: [] for k in (
        "dir", "opp", "idx", "q", "idx_m1", "ok_m1", "idx_m2", "ok_m2",
        "xw", "yw", "zw", "uw", "group")}
    group_names: list[str] = []

    for d in range(1, _N_DIR):
        cx, cy, cz = _C_RAW[d]
        # neighbour in direction +c: roll by -c maps out[x] = mask[x + c]
        nb = torch.roll(mask, shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
        cut = (~mask) & nb
        n_cut = int(cut.sum().item())
        if n_cut == 0:
            continue
        iz, iy, ix = torch.nonzero(cut, as_tuple=True)
        idx = iz.to(torch.int64) * nyx + iy.to(torch.int64) * nx + ix.to(torch.int64)

        if kind == "annulus":
            cz0, cy0, cx0 = spec["center"]
            r_nb2 = (ix.to(torch.float64) + cx - float(cx0)) ** 2 + (
                iy.to(torch.float64) + cy - float(cy0)
            ) ** 2
            inner = r_nb2 <= float(spec["radius"]) ** 2
            spec_in = dict(spec, kind="cylinder", radius=float(spec["radius"]))
            spec_out = dict(spec, kind="cylinder", radius=float(spec["radius_outer"]))
            q = torch.where(
                inner,
                _analytic_q(spec_in, ix, iy, iz, (cx, cy, cz)),
                _analytic_q(spec_out, ix, iy, iz, (cx, cy, cz)),
            ).to(dtype)
            group = torch.where(
                inner,
                torch.zeros(n_cut, dtype=torch.int64),
                torch.ones(n_cut, dtype=torch.int64),
            )
            if not group_names:
                group_names = ["inner", "outer"]
        else:
            q = _analytic_q(spec, ix, iy, iz, (cx, cy, cz)).to(dtype)
            if kind == "plane":
                gname = "top" if cy > 0 else "bottom"
            else:
                gname = "body"
            if gname not in group_names:
                group_names.append(gname)
            group = torch.full((n_cut,), group_names.index(gname), dtype=torch.int64)

        def _offset(steps: int) -> tuple[torch.Tensor, torch.Tensor]:
            sz = iz - steps * cz
            sy = iy - steps * cy
            sx = ix - steps * cx
            ok = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny) & (sz >= 0) & (sz < nz)
            flat = (
                sz.clamp(0, nz - 1) * nyx
                + sy.clamp(0, ny - 1) * nx
                + sx.clamp(0, nx - 1)
            )
            return flat.to(torch.int64), ok & fluid_flat[flat]

        idx_m1, ok_m1 = _offset(1)
        idx_m2, ok_m2 = _offset(2)

        qq = q.to(torch.float64)
        xw = (ix.to(torch.float64) + qq * cx).to(dtype)
        yw = (iy.to(torch.float64) + qq * cy).to(dtype)
        zw = (iz.to(torch.float64) + qq * cz).to(dtype)

        uw = torch.zeros((n_cut, 3), dtype=dtype)
        wall_speed = float(spec.get("wall_speed", 0.0))
        moving_side = str(spec.get("moving_side", "top"))
        if wall_speed != 0.0 and kind == "plane":
            if (cy > 0 and moving_side == "top") or (cy < 0 and moving_side == "bottom"):
                uw[:, 0] = wall_speed
        if float(spec.get("omega", 0.0)) != 0.0 and kind in ("cylinder", "annulus"):
            omega = float(spec["omega"])
            vx = (-omega * (yw.to(torch.float64) - float(cy0))).to(dtype)
            vy = (omega * (xw.to(torch.float64) - float(cx0))).to(dtype)
            if kind == "annulus":
                uw[:, 0] = torch.where(inner, vx, torch.zeros_like(vx))
                uw[:, 1] = torch.where(inner, vy, torch.zeros_like(vy))
            else:
                uw[:, 0] = vx
                uw[:, 1] = vy

        cols["dir"].append(torch.full((n_cut,), d, dtype=torch.int64))
        cols["opp"].append(torch.full((n_cut,), _OPP_RAW[d], dtype=torch.int64))
        cols["idx"].append(idx)
        cols["q"].append(q)
        cols["idx_m1"].append(idx_m1)
        cols["ok_m1"].append(ok_m1)
        cols["idx_m2"].append(idx_m2)
        cols["ok_m2"].append(ok_m2)
        cols["xw"].append(xw)
        cols["yw"].append(yw)
        cols["zw"].append(zw)
        cols["uw"].append(uw)
        cols["group"].append(group)

    geo = Geometry(mask=mask, kind=kind, n_nodes=nz * ny * nx)
    if cols["idx"]:
        rename = {"group": "group_id"}
        for key in cols:
            geo.__dict__[rename.get(key, key)] = torch.cat(cols[key]).to(dev)
    else:
        empty = torch.zeros(0, device=dev)
        for key in ("dir", "opp", "idx", "idx_m1", "idx_m2"):
            geo.__dict__[key] = empty.to(torch.int64)
        geo.group_id = empty.to(torch.int64)
        for key in ("q", "xw", "yw", "zw"):
            geo.__dict__[key] = empty
        geo.uw = torch.zeros((0, 3), device=dev)
        geo.ok_m1 = torch.zeros(0, dtype=torch.bool, device=dev)
        geo.ok_m2 = torch.zeros(0, dtype=torch.bool, device=dev)
    geo.groups = tuple(group_names)
    if format == "halfway" and geo.q is not None and geo.q.numel():
        geo.q = torch.full_like(geo.q, 0.5)

    centre = spec.get("center", (0.0, 0.0, 0.0))
    geo.torque_center = (float(centre[2]), float(centre[1]))
    geo.spec = {
        k: (list(v) if isinstance(v, (tuple, list)) else v)
        for k, v in spec.items()
        if k in ("center", "radius", "radius_outer", "y_bottom", "y_top",
                 "wall_speed", "omega", "moving_side")
    }
    return geo


# ---------------------------------------------------------------------------
# D3Q19 solver
# ---------------------------------------------------------------------------


@dataclass
class FlowConfig:
    """Numerical configuration of one solve."""

    nz: int
    ny: int
    nx: int
    tau: float
    format: str = "bouzidi_quadratic"
    collision: str = "regularized"
    dtype: str = "float32"
    device: str = "cpu"
    free_slip: tuple[str, ...] = ()      # faces among ("y0","y1","z0","z1")
    inlet_outlet: bool = False
    u_inflow: float = 0.0        # prescribed inlet velocity (streamwise)
    sponge_width: int = 0
    sponge_strength: float = 0.2
    force_theta: float = 0.0      # azimuthal body force per unit mass (annulus)
    force_x: float = 0.0          # streamwise body force per unit mass

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    def validate(self) -> None:
        if self.format not in BOUNDARY_FORMATS:
            raise ValueError(f"unknown boundary format {self.format!r}")
        if self.collision not in COLLISION_MODELS:
            raise ValueError(f"unknown collision model {self.collision!r}")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 1/2")


class Solver:
    """Vectorized D3Q19 solver (pull streaming by slab copies, torch)."""

    def __init__(self, cfg: FlowConfig, geometry: Geometry):
        cfg.validate()
        self.cfg = cfg
        self.geo = geometry
        self.dev = torch.device(cfg.device)
        self.dt = _dtype(cfg.dtype)
        self.nz, self.ny, self.nx = cfg.nz, cfg.ny, cfg.nx
        self.n_nodes = self.nz * self.ny * self.nx

        self.c = torch.tensor(_C_RAW, dtype=torch.int64, device=self.dev)
        self.w = torch.tensor(_W_RAW, dtype=self.dt, device=self.dev)
        self.cx = self.c[:, 0].reshape(_N_DIR, 1, 1, 1).to(self.dt)
        self.cy = self.c[:, 1].reshape(_N_DIR, 1, 1, 1).to(self.dt)
        self.cz = self.c[:, 2].reshape(_N_DIR, 1, 1, 1).to(self.dt)
        self.w4 = self.w.reshape(_N_DIR, 1, 1, 1)

        rho0 = torch.ones((self.nz, self.ny, self.nx), dtype=self.dt, device=self.dev)
        self.f = self.equilibrium(rho0, None)
        self.last_force = torch.zeros(3, dtype=self.dt, device=self.dev)
        self.last_torque = 0.0
        self.last_group_force: dict[str, torch.Tensor] = {}
        self.last_group_torque: dict[str, torch.Tensor] = {}

        # per-link buffer holding the wall-returned population of the last step
        # (the solid-side node value used by the q >= 1/2 stencils)
        self.buf = torch.zeros(geometry.n_links(), dtype=self.dt, device=self.dev)
        self._link_dir = geometry.dir * self.n_nodes if geometry.dir is not None else None
        self._link_opp = geometry.opp * self.n_nodes if geometry.opp is not None else None
        self._link_c = torch.tensor(
            [_C_RAW[int(d)] for d in geometry.dir.tolist()], dtype=self.dt, device=self.dev
        ) if geometry.n_links() else torch.zeros((0, 3), dtype=self.dt, device=self.dev)
        self._link_w = torch.tensor(
            [_W_RAW[int(d)] for d in geometry.dir.tolist()], dtype=self.dt, device=self.dev
        ) if geometry.n_links() else torch.zeros(0, dtype=self.dt, device=self.dev)

        self._spec_dir = {}
        for axis in (0, 1, 2):
            for d in range(1, _N_DIR):
                cc = list(_C_RAW[d])
                cc[axis] = -cc[axis]
                self._spec_dir[(axis, d)] = _C_RAW.index(tuple(cc))
        self._inlet_dirs = [d for d in range(1, _N_DIR) if _C_RAW[d][0] > 0]
        self._outlet_dirs = [d for d in range(1, _N_DIR) if _C_RAW[d][0] < 0]
        self._ystart_dirs = [d for d in range(1, _N_DIR) if _C_RAW[d][1] < 0]
        self._yend_dirs = [d for d in range(1, _N_DIR) if _C_RAW[d][1] > 0]
        self._zstart_dirs = [d for d in range(1, _N_DIR) if _C_RAW[d][2] < 0]
        self._zend_dirs = [d for d in range(1, _N_DIR) if _C_RAW[d][2] > 0]

    # -- moments / collision -------------------------------------------------

    def equilibrium(self, rho: torch.Tensor, u: Any) -> torch.Tensor:
        if u is None:
            ux = uy = uz = torch.zeros_like(rho)
        else:
            ux, uy, uz = u
        cu = 3.0 * (self.cx * ux + self.cy * uy + self.cz * uz)
        usq = ux * ux + uy * uy + uz * uz
        return self.w4 * rho * (1.0 + cu * (1.0 + 0.5 * cu) - 1.5 * usq)

    def moments(self, f: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        rho = f.sum(0)
        ux = (f * self.cx).sum(0) / rho
        uy = (f * self.cy).sum(0) / rho
        uz = (f * self.cz).sum(0) / rho
        return rho, (ux, uy, uz)

    def _feq_dir(self, d, rho, ux, uy, uz):
        cx, cy, cz = _C_RAW[d]
        cu = 3.0 * (cx * ux + cy * uy + cz * uz)
        usq = ux * ux + uy * uy + uz * uz
        return float(_W_RAW[d]) * rho * (1.0 + cu * (1.0 + 0.5 * cu) - 1.5 * usq)

    def body_force(self, rho: torch.Tensor, u: tuple) -> tuple | None:
        """Return the (fx, fy, fz) force-per-unit-volume field, or None."""
        fx = self.cfg.force_x
        fth = self.cfg.force_theta
        if fx == 0.0 and fth == 0.0:
            return None
        if fth != 0.0:
            zz, yy, xx = torch.meshgrid(
                torch.arange(self.nz, dtype=self.dt, device=self.dev),
                torch.arange(self.ny, dtype=self.dt, device=self.dev),
                torch.arange(self.nx, dtype=self.dt, device=self.dev),
                indexing="ij",
            )
            cy0 = 0.5 * (self.ny - 1)
            cx0 = 0.5 * (self.nx - 1)
            px, py = xx - cx0, yy - cy0
            rr = torch.sqrt(px * px + py * py).clamp_min(1e-6)
            # f = fth * e_theta with e_theta = (-py, px, 0)/r
            return (fth * rho * (-py / rr), fth * rho * (px / rr), torch.zeros_like(px))
        return (fx * rho, torch.zeros_like(rho), torch.zeros_like(rho))

    def collide(self, f: torch.Tensor, rho: torch.Tensor, u: tuple) -> torch.Tensor:
        feq = self.equilibrium(rho, u)
        if self.cfg.collision == "bgk":
            return f + self.cfg.omega * (feq - f)
        # regularized BGK: keep only the second-order Hermite component of the
        # non-equilibrium part (Latt & Chopard regularization)
        fneq = f - feq
        pxx = (fneq * self.cx * self.cx).sum(0)
        pyy = (fneq * self.cy * self.cy).sum(0)
        pzz = (fneq * self.cz * self.cz).sum(0)
        pxy = (fneq * self.cx * self.cy).sum(0)
        pxz = (fneq * self.cx * self.cz).sum(0)
        pyz = (fneq * self.cy * self.cz).sum(0)
        qq = (
            (self.cx * self.cx - CS2) * pxx
            + (self.cy * self.cy - CS2) * pyy
            + (self.cz * self.cz - CS2) * pzz
            + 2.0 * self.cx * self.cy * pxy
            + 2.0 * self.cx * self.cz * pxz
            + 2.0 * self.cy * self.cz * pyz
        )
        return feq + (1.0 - self.cfg.omega) * 4.5 * self.w4 * qq

    # -- streaming -----------------------------------------------------------

    def stream(self, fpost: torch.Tensor) -> torch.Tensor:
        """out[d, x] = fpost[d, x - c_d] (wrap-around everywhere).

        The wrap-around slabs are exactly the boundary planes: inlet / outlet /
        free-slip overwrite theirs explicitly, the cut-link kernels overwrite
        every population entering a fluid node from a solid neighbour, and the
        remaining wrapped planes are solid nodes whose values are never read.
        """
        out = torch.empty_like(fpost)
        for d in range(_N_DIR):
            cx, cy, cz = _C_RAW[d]
            if cx == 0 and cy == 0 and cz == 0:
                out[d] = fpost[d]
            else:
                out[d] = torch.roll(fpost[d], shifts=(cz, cy, cx), dims=(0, 1, 2))
        return out

    # -- boundary conditions -------------------------------------------------

    def apply_cut_links(
        self,
        fpost: torch.Tensor,
        fnew: torch.Tensor,
        rho: torch.Tensor,
        u: tuple,
        wall_scale: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        """Interpolated bounce-back on every cut link; returns (F, T_z).

        Mirror-point (``x_m = x_f + xi c_i``, ``xi = 2q - 1``) reconstruction:

        ``q < 1/2`` (mirror point inside the fluid half of the link)
            2-point (linear) / 3-point (quadratic) Lagrange interpolation of the
            outgoing post-collision populations at offsets ``0, -1 (, -2)`` plus
            the Ladd wall-velocity correction.  This *is* the classical Bouzidi
            branch (the weights reduce to ``2q, 1-2q``).
        ``q >= 1/2`` (mirror point beyond the fluid node)
            the required solid-side value is closed with the same-node opposite
            non-equilibrium part, ``v(+1) = w rho + n_opp``, which keeps every
            interpolation weight bounded (extrapolating the raw populations was
            measured to be unconditionally unstable here: weights up to 3 on the
            outgoing population).  Continuous with the ``q < 1/2`` branch at
            ``q = 1/2``, but the closure is only first-order accurate in the wall
            position for *moving* walls with ``q > 1/2``; the size of that defect
            is measured and reported (see docs/curved_boundary_calibration_p48_n01.md).
        """
        geo = self.geo
        if geo.n_links() == 0:
            self.last_group_force = {}
            self.last_group_torque = {}
            return torch.zeros(3, dtype=self.dt, device=self.dev), 0.0
        fp = fpost.reshape(-1)
        idx = geo.idx
        c = self._link_c
        w_l = self._link_w

        rho_l = rho.reshape(-1)[idx]
        base = self._link_dir + idx
        v0 = fp[base]
        v1 = torch.where(geo.ok_m1, fp[self._link_dir + geo.idx_m1], v0)
        v_opp = fp[self._link_opp + idx]

        # same-node opposite non-equilibrium (closure for the solid-side node)
        ux0 = u[0].reshape(-1)[idx]
        uy0 = u[1].reshape(-1)[idx]
        uz0 = u[2].reshape(-1)[idx]
        usq0 = ux0 * ux0 + uy0 * uy0 + uz0 * uz0
        cu0 = 3.0 * (c[:, 0] * ux0 + c[:, 1] * uy0 + c[:, 2] * uz0)
        feq_opp0 = w_l * rho_l * (1.0 - cu0 * (1.0 - 0.5 * cu0) - 1.5 * usq0)
        v_p1 = w_l * rho_l + (v_opp - feq_opp0)

        xi = 2.0 * geo.q - 1.0
        low = xi < 0.0
        fmt = self.cfg.format
        uwx = geo.uw[:, 0] * wall_scale
        uwy = geo.uw[:, 1] * wall_scale
        uwz = geo.uw[:, 2] * wall_scale
        cuw = c[:, 0] * uwx + c[:, 1] * uwy + c[:, 2] * uwz
        wall_corr = -2.0 * w_l * rho_l * cuw / CS2
        if MOVING_WALL_CLOSURE_SHIFT:
            # The solid-side closure is built in the *wall frame* (stationary wall),
            # so its equilibrium part is feq(rho, 0); restoring it to the lab frame
            # needs the same shift that the fluid-side populations carry, i.e.
            # 3 w rho (c_i . u_w).  Without it the q >= 1/2 stencils inject
            # 6 w rho (c_i . u_w) [1 + xi(1+xi)/4] instead of 6 w rho (c_i . u_w):
            # a +50% momentum over-injection at xi = 1 (q = 1) that is invisible on
            # a stationary wall (u_w = 0 leaves this term exactly zero).
            v_p1 = v_p1 + 3.0 * w_l * rho_l * cuw
        if fmt == "halfway":
            val = v0 + wall_corr
        elif fmt == "bouzidi_linear":
            val = torch.where(low, v0 + xi * (v0 - v1), v0 + xi * (v_p1 - v0)) + wall_corr
        else:
            v2 = torch.where(geo.ok_m2, fp[self._link_dir + geo.idx_m2], v1)
            val_low = (
                v0 * (xi + 1.0) * (xi + 2.0) * 0.5
                - v1 * xi * (xi + 2.0)
                + v2 * xi * (xi + 1.0) * 0.5
            )
            val_high = (
                v1 * xi * (xi - 1.0) * 0.5
                + v0 * (1.0 - xi * xi)
                + v_p1 * xi * (xi + 1.0) * 0.5
            )
            val = torch.where(low, val_low, val_high) + wall_corr

        fnew.reshape(-1)[self._link_opp + idx] = val

        flux = v0 + val
        f_link = c * flux.reshape(-1, 1)
        torque_link = (geo.xw - geo.torque_center[0]) * f_link[:, 1] - (
            geo.yw - geo.torque_center[1]
        ) * f_link[:, 0]
        force = f_link.sum(0)
        torque = float(torque_link.sum().item())
        ng = len(geo.groups)
        gf = torch.zeros((ng, 3), dtype=self.dt, device=self.dev)
        gt = torch.zeros(ng, dtype=self.dt, device=self.dev)
        gf.index_add_(0, geo.group_id, f_link)
        gt.index_add_(0, geo.group_id, torque_link)
        self.last_group_force = {geo.groups[k]: gf[k] for k in range(ng)}
        self.last_group_torque = {geo.groups[k]: float(gt[k].item()) for k in range(ng)}
        return force, torque

    def apply_domain_bcs(self, fpost, fnew, rho, u, u_in: float = 0.0) -> None:
        nz, ny, nx = self.nz, self.ny, self.nx
        if self.cfg.inlet_outlet:
            rho1 = rho[:, :, 1]
            ux1, uy1, uz1 = u[0][:, :, 1], u[1][:, :, 1], u[2][:, :, 1]
            zeros = torch.zeros_like(rho1)
            ones = torch.ones_like(rho1)
            for d in self._inlet_dirs:
                feq_in = self._feq_dir(d, ones, u_in * ones, zeros, zeros)
                feq_1 = self._feq_dir(d, rho1, ux1, uy1, uz1)
                fnew[d][:, :, 0] = feq_in + (fpost[d][:, :, 1] - feq_1)
            for d in self._outlet_dirs:
                fnew[d][:, :, nx - 1] = fpost[d][:, :, nx - 2]
        fs = self.cfg.free_slip
        if "y0" in fs:   # particles enter the domain from below: cy > 0
            for d in self._yend_dirs:
                fnew[d][:, 0, :] = fpost[self._spec_dir[(1, d)]][:, 0, :]
        if "y1" in fs:
            for d in self._ystart_dirs:
                fnew[d][:, ny - 1, :] = fpost[self._spec_dir[(1, d)]][:, ny - 1, :]
        if "z0" in fs:
            for d in self._zend_dirs:
                fnew[d][0, :, :] = fpost[self._spec_dir[(2, d)]][0, :, :]
        if "z1" in fs:
            for d in self._zstart_dirs:
                fnew[d][nz - 1, :, :] = fpost[self._spec_dir[(2, d)]][nz - 1, :, :]

    def apply_sponge(self, fpost, rho, u_mean: float) -> None:
        w = self.cfg.sponge_width
        if w <= 0:
            return
        nx = self.nx
        ramp = torch.linspace(0.0, 1.0, w, dtype=self.dt, device=self.dev)
        sigma = self.cfg.sponge_strength * ramp * ramp
        sub_rho = rho[:, :, nx - w: nx]
        ones = torch.ones_like(sub_rho)
        zeros = torch.zeros_like(sub_rho)
        feq = self.equilibrium(sub_rho, (u_mean * ones, zeros, zeros))
        fpost[:, :, :, nx - w: nx] += sigma.reshape(1, 1, 1, w) * (
            feq - fpost[:, :, :, nx - w: nx]
        )

    def _forcing_source(self, rho: torch.Tensor, u: tuple, force: tuple) -> torch.Tensor:
        """Guo (1993) source term ``(1 - 1/(2 tau)) w_i [3(c_i-u).F + 9(c_i.u)(c_i.F)]``."""
        fx, fy, fz = force
        return 3.0 * self.w4 * (self.cx * fx + self.cy * fy + self.cz * fz)

    def step(self, wall_scale: float = 1.0) -> tuple[torch.Tensor, float]:
        rho, u = self.moments(self.f)
        body = self.body_force(rho, u)
        fpost = self.collide(self.f, rho, u)
        if body is not None:
            fpost = fpost + self._forcing_source(rho, u, body)
        if self.cfg.sponge_width > 0:
            self.apply_sponge(fpost, rho, float(u[0].mean().item()))
        fnew = self.stream(fpost)
        force, torque = self.apply_cut_links(fpost, fnew, rho, u, wall_scale)
        if self.cfg.inlet_outlet or self.cfg.free_slip:
            self.apply_domain_bcs(fpost, fnew, rho, u, self.cfg.u_inflow)
        self.f = fnew
        self.last_force = force
        self.last_torque = torque
        return force, torque

    def velocity(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rho, (ux, uy, uz) = self.moments(self.f)
        body = self.body_force(rho, (ux, uy, uz))
        if body is not None:
            ux = ux + 0.5 * body[0] / rho
            uy = uy + 0.5 * body[1] / rho
            uz = uz + 0.5 * body[2] / rho
        return ux, uy, uz


# ---------------------------------------------------------------------------
# control-surface (stress-integral) force route
# ---------------------------------------------------------------------------


def _grad_on_plane(u: torch.Tensor, plane_axis: int, index: int, grad_axis: int) -> torch.Tensor:
    """Derivative of ``u`` along ``grad_axis`` evaluated on the node plane.

    ``plane_axis``/``index`` select the 2-D plane; the normal derivative
    (``grad_axis == plane_axis``) is a central difference across the plane and
    an in-plane derivative uses :func:`torch.gradient`.
    """
    if grad_axis == plane_axis:
        sl_m = [slice(None)] * 3
        sl_p = [slice(None)] * 3
        sl_m[plane_axis] = max(index - 1, 0)
        sl_p[plane_axis] = min(index + 1, u.shape[plane_axis] - 1)
        return 0.5 * (u[tuple(sl_p)] - u[tuple(sl_m)])
    sl = [slice(None)] * 3
    sl[plane_axis] = index
    base = u[tuple(sl)]
    dims = [a for a in (0, 1, 2) if a != plane_axis]
    d = dims.index(grad_axis)
    if base.shape[d] < 2:
        return torch.zeros_like(base)
    return torch.gradient(base, dim=d)[0]


def stress_integral_drag(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    rho: torch.Tensor,
    nu: float,
    box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> float:
    """x-drag through a closed axis-aligned node-plane control surface.

    ``box`` gives inclusive ``(z, y, x)`` bounds.  Surface quadrature points are
    de-duplicated at box edges (1/2) and corners (1/4), so every point carries
    unit area exactly once.
    """
    (z0, z1), (y0, y1), (x0, x1) = box
    bounds = {0: (z0, z1), 1: (y0, y1), 2: (x0, x1)}
    p = rho * CS2

    def face_weight(axis: int) -> torch.Tensor:
        w = torch.ones_like(ux)
        for ax in (0, 1, 2):
            if ax == axis:
                continue
            lo, hi = bounds[ax]
            idx = torch.arange(ux.shape[ax], device=ux.device)
            wa = torch.where(
                (idx == lo) | (idx == hi),
                torch.tensor(0.5, dtype=ux.dtype, device=ux.device),
                torch.tensor(1.0, dtype=ux.dtype, device=ux.device),
            )
            shape = [1, 1, 1]
            shape[ax] = ux.shape[ax]
            w = w * wa.reshape(shape)
        return w

    total = 0.0
    for axis, lo, hi in ((2, x0, x1), (1, y0, y1), (0, z0, z1)):
        w = face_weight(axis)
        for index, sign in ((lo, -1.0), (hi, 1.0)):
            sl = tuple([index if a == axis else slice(None) for a in range(3)])
            wsl = w[sl]
            d_norm = _grad_on_plane(ux, axis, index, axis)
            if axis == 2:
                flux = rho[sl] * ux[sl] * ux[sl] + p[sl] - 2.0 * nu * d_norm
            elif axis == 1:
                flux = rho[sl] * ux[sl] * uy[sl] - nu * (
                    d_norm + _grad_on_plane(uy, axis, index, 2)
                )
            else:
                flux = rho[sl] * ux[sl] * uz[sl] - nu * (
                    d_norm + _grad_on_plane(uz, axis, index, 2)
                )
            total -= float(sign) * float((flux * wsl).sum().item())
    return total


def plane_momentum_flux(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    rho: torch.Tensor,
    nu: float,
    axis: int,
    index: int,
) -> float:
    """x-momentum flux ``Integral (rho u_x u_k + p d_xk - mu d_k u_x) dS``."""
    sl = tuple([index if a == axis else slice(None) for a in range(3)])
    d_norm = _grad_on_plane(ux, axis, index, axis)
    if axis == 2:
        flux = rho[sl] * ux[sl] ** 2 + CS2 * rho[sl] - 2.0 * nu * d_norm
    elif axis == 1:
        flux = rho[sl] * ux[sl] * uy[sl] - nu * (
            d_norm + _grad_on_plane(uy, axis, index, 2)
        )
    else:
        flux = rho[sl] * ux[sl] * uz[sl] - nu * (
            d_norm + _grad_on_plane(uz, axis, index, 2)
        )
    return float(flux.sum().item())


# ---------------------------------------------------------------------------
# case 1: Stokes' second problem (oscillating flat plate)
# ---------------------------------------------------------------------------


def run_stokes_second(
    u0: float = 0.05,
    delta: float = 4.0,
    tau: float = 0.65,
    format: str = "bouzidi_quadratic",
    wall_offset: float = 0.5,
    periods: int = 3,
    samples_per_period: int = 48,
    warmup_periods: int = 2,
    nx: int = 2,
    nz: int = 2,
    collision: str = "regularized",
) -> dict[str, Any]:
    """Oscillating flat plate against the analytic Stokes layer.

    The plate sits ``q = wall_offset`` below the first fluid row (q = 1/2 is
    half-way bounce-back) and moves with ``u_x = u0 cos(omega t)``.  The domain
    is ``9 delta`` tall, so the stationary top wall perturbs the sampled band
    ``y' <= 3 delta`` by ``exp(-6) ~ 0.25 %``.
    """
    nu = (tau - 0.5) / 3.0
    omega = 2.0 * nu / (delta * delta)
    period_steps = int(round(2.0 * math.pi / omega))
    omega = 2.0 * math.pi / period_steps
    delta = math.sqrt(2.0 * nu / omega)
    ny = int(round(9.0 * delta)) + 6
    y_bottom = 1.0 - wall_offset
    y_top = float(ny - 2) + 0.5

    spec = {"y_bottom": y_bottom, "y_top": y_top, "wall_speed": u0,
            "moving_side": "bottom"}
    geo = build_geometry("plane", nz, ny, nx, spec, format=format)
    cfg = FlowConfig(nz=nz, ny=ny, nx=nx, tau=tau, format=format, collision=collision)
    solver = Solver(cfg, geo)

    yy = torch.arange(ny, dtype=solver.dt).reshape(1, ny, 1)
    yprime = (yy - y_bottom).clamp_min(0.0)
    u_init = u0 * torch.exp(-yprime / delta) * torch.cos(-yprime / delta)
    ux0 = u_init.expand(nz, ny, nx).contiguous()
    zeros = torch.zeros_like(ux0)
    rho_init = torch.ones((nz, ny, nx), dtype=solver.dt)
    solver.f = solver.equilibrium(rho_init, (ux0, zeros, zeros))

    n_warm = int(round(warmup_periods * period_steps))
    for k in range(n_warm):
        solver.step(math.cos(k * omega))

    dt = period_steps / samples_per_period
    n_samples = periods * samples_per_period
    rows = []
    for f in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        r = int(round(y_bottom + f * delta))
        if 1 <= r <= ny - 2:
            rows.append(r)
    rows = sorted(set(rows))
    records = {r: [] for r in rows}
    times: list[float] = []
    step = 0
    for k in range(n_samples):
        target = int(round(k * dt))
        while step < target:
            solver.step(math.cos((warmup_periods * period_steps + step) * omega))
            step += 1
        _, (ux, _, _) = solver.moments(solver.f)
        times.append((warmup_periods * period_steps + step) * omega)
        for r in rows:
            records[r].append(float(ux[:, r, :].mean().item()))

    tarr = torch.tensor(times, dtype=torch.float64)
    kern = torch.exp(-1j * tarr)
    rows_out = []
    worst_complex = worst_amp = worst_phase = 0.0
    for r in rows:
        sig = torch.tensor(records[r], dtype=torch.float64)
        zc = 2.0 / sig.numel() * (sig.to(torch.complex128) * kern).sum()
        yp = float(r) - y_bottom
        z_ana = complex(
            u0 * math.exp(-yp / delta) * math.cos(-yp / delta),
            u0 * math.exp(-yp / delta) * math.sin(-yp / delta),
        )
        amp_num, amp_ana = abs(zc), abs(z_ana)
        ph_num = math.atan2(zc.imag, zc.real)
        ph_ana = math.atan2(z_ana.imag, z_ana.real)
        dphi = (ph_num - ph_ana + math.pi) % (2 * math.pi) - math.pi
        rel = abs(zc - z_ana) / amp_ana
        worst_complex = max(worst_complex, rel)
        worst_amp = max(worst_amp, abs(amp_num - amp_ana) / amp_ana)
        worst_phase = max(worst_phase, abs(dphi))
        rows_out.append({
            "row": r,
            "y_over_delta": yp / delta,
            "amp_num": amp_num,
            "amp_ana": amp_ana,
            "amp_rel_err": abs(amp_num - amp_ana) / amp_ana,
            "phase_num_rad": ph_num,
            "phase_ana_rad": ph_ana,
            "phase_err_rad": dphi,
            "complex_rel_err": rel,
        })

    return {
        "case": "stokes_second",
        "format": format,
        "collision": collision,
        "tau": tau,
        "nu": nu,
        "u0": u0,
        "delta": delta,
        "wall_offset_q": wall_offset,
        "period_steps": period_steps,
        "periods_sampled": periods,
        "warmup_periods": warmup_periods,
        "nz": nz, "ny": ny, "nx": nx,
        "rows": rows_out,
        "max_amp_rel_err": worst_amp,
        "max_phase_err_rad": worst_phase,
        "max_complex_rel_err": worst_complex,
        "reference": "u = u0 exp(-y/delta) cos(omega t - y/delta), delta = sqrt(2 nu / omega)",
    }


# ---------------------------------------------------------------------------
# case 2: plane Couette (flat-wall degenerate limit / negative control)
# ---------------------------------------------------------------------------


def _initialise_analytic_channel(
    solver: Solver, profile: Callable[[torch.Tensor], torch.Tensor], ny: int
) -> None:
    yy = torch.arange(ny, dtype=solver.dt).reshape(1, ny, 1)
    ux = profile(yy).expand(solver.cfg.nz, ny, solver.cfg.nx).contiguous()
    zeros = torch.zeros_like(ux)
    rho = torch.ones((solver.cfg.nz, ny, solver.cfg.nx), dtype=solver.dt)
    solver.f = solver.equilibrium(rho, (ux, zeros, zeros))


def run_plane_couette(
    wall_speed: float = 0.05,
    tau: float = 0.65,
    format: str = "bouzidi_quadratic",
    q_bottom: float = 0.5,
    q_top: float = 0.5,
    ny: int = 18,
    nx: int = 2,
    nz: int = 2,
    steps: int = 4000,
    avg_fraction: float = 0.2,
    collision: str = "regularized",
) -> dict[str, Any]:
    """Plane Couette flow: analytic wall shear ``tau_w = mu U / H``.

    ``q_bottom = q_top = 1/2`` is the degenerate straight-wall configuration;
    other values shift the analytic wall inside the last cell, so the
    interpolation weights are exercised on a straight wall whose exact answer is
    still known.
    """
    nu = (tau - 0.5) / 3.0
    y_bottom = 1.0 - q_bottom
    y_top = float(ny - 2) + q_top
    spec = {"y_bottom": y_bottom, "y_top": y_top, "wall_speed": wall_speed}
    geo = build_geometry("plane", nz, ny, nx, spec, format=format)
    cfg = FlowConfig(nz=nz, ny=ny, nx=nx, tau=tau, format=format, collision=collision)
    solver = Solver(cfg, geo)
    height = y_top - y_bottom

    def profile(yy: torch.Tensor) -> torch.Tensor:
        return wall_speed * (yy - y_bottom) / height

    _initialise_analytic_channel(solver, profile, ny)

    f_top_hist: list[float] = []
    f_bot_hist: list[float] = []
    for k in range(steps):
        solver.step(1.0)
        if k >= steps - int(steps * avg_fraction):
            f_top_hist.append(float(solver.last_group_force["top"][0].item()))
            f_bot_hist.append(float(solver.last_group_force["bottom"][0].item()))
    force_top = sum(f_top_hist) / len(f_top_hist)
    force_bottom = sum(f_bot_hist) / len(f_bot_hist)

    ux, uy, uz = solver.velocity()
    rho, _ = solver.moments(solver.f)
    cv_flux = plane_momentum_flux(ux, uy, uz, rho, nu, 1, ny // 2)
    area = nx * nz
    tau_ana = nu * wall_speed / height
    force_ana = tau_ana * area
    prof, ymax = [], 0.0
    for iy in range(1, ny - 1):
        u_num = float(ux[:, iy, :].mean().item())
        u_ana = float(profile(torch.tensor(iy, dtype=solver.dt)).item())
        prof.append({"y": iy, "u_num": u_num, "u_ana": u_ana})
        ymax = max(ymax, abs(u_num - u_ana) / wall_speed)

    return {
        "case": "plane_couette",
        "format": format,
        "collision": collision,
        "tau": tau,
        "nu": nu,
        "wall_speed": wall_speed,
        "q_bottom": q_bottom,
        "q_top": q_top,
        "height": height,
        "steps": steps,
        "force_top_x": force_top,
        "force_bottom_x": force_bottom,
        "force_analytic": force_ana,
        "force_rel_err": abs(abs(force_top) - force_ana) / force_ana,
        "force_bottom_rel_err": abs(abs(force_bottom) - force_ana) / force_ana,
        "cv_flux_midplane": cv_flux,
        "cv_rel_err": abs(abs(cv_flux) - force_ana) / force_ana,
        "me_vs_cv_rel_diff": abs(abs(force_top) - abs(cv_flux)) / max(abs(force_top), 1e-30),
        "profile_max_rel_err": ymax,
        "profile": prof,
        "reference": "Couette: u = U (y - y_b)/(y_t - y_b), tau_w = mu U / H",
    }


# ---------------------------------------------------------------------------
# case 3: Taylor-Couette annulus (exact NS solution, exact torque)
# ---------------------------------------------------------------------------


def annulus_exact(radius_inner: float, radius_outer: float, omega: float, nu: float) -> dict[str, float]:
    """Exact azimuthal Couette solution ``u_t = A r + B/r`` and its torque.

    The profile satisfies the steady Navier-Stokes equations exactly (the
    convective term ``-u_t^2/r`` is balanced by the radial pressure gradient), so
    the reference is valid at any Reynolds number as long as the flow stays
    axisymmetric and laminar.
    """
    r1, r2 = float(radius_inner), float(radius_outer)
    a = -omega * r1 * r1 / (r2 * r2 - r1 * r1)
    b = omega * r1 * r1 * r2 * r2 / (r2 * r2 - r1 * r1)
    torque = 4.0 * math.pi * nu * abs(b)            # per unit span, rho = 1
    return {"A": a, "B": b, "torque_per_span": torque, "u_inner": omega * r1}


def run_annulus_couette(
    radius_inner: int = 16,
    radius_outer: int = 32,
    reynolds_theta: float = 2.0,
    tau: float = 0.65,
    format: str = "bouzidi_quadratic",
    nz: int = 2,
    steps: int = 6000,
    avg_fraction: float = 0.25,
    collision: str = "regularized",
) -> dict[str, Any]:
    """Rotating inner cylinder inside a fixed outer cylinder: exact torque."""
    nu = (tau - 0.5) / 3.0
    gap = float(radius_outer - radius_inner)
    omega = reynolds_theta * nu / (float(radius_inner) * gap)
    nx = ny = 2 * int(radius_outer) + 3
    center = (0.0, (ny - 1) / 2.0, (nx - 1) / 2.0)
    spec = {"center": center, "radius": float(radius_inner),
            "radius_outer": float(radius_outer), "omega": omega}
    geo = build_geometry("annulus", nz, ny, nx, spec, format=format)
    cfg = FlowConfig(nz=nz, ny=ny, nx=nx, tau=tau, format=format, collision=collision)
    solver = Solver(cfg, geo)

    exact = annulus_exact(radius_inner, radius_outer, omega, nu)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=solver.dt),
        torch.arange(ny, dtype=solver.dt),
        torch.arange(nx, dtype=solver.dt),
        indexing="ij",
    )
    px = xx - float(center[2])
    py = yy - float(center[1])
    rr = torch.sqrt(px * px + py * py).clamp_min(1e-6)
    ut = exact["A"] * rr + exact["B"] / rr
    ux0 = -ut * py / rr
    uy0 = ut * px / rr
    rho0 = torch.ones((nz, ny, nx), dtype=solver.dt)
    solver.f = solver.equilibrium(rho0, (ux0, uy0, torch.zeros_like(ux0)))

    t_inner: list[float] = []
    t_outer: list[float] = []
    t0 = time.perf_counter()
    umax = 0.0
    for k in range(steps):
        solver.step(1.0)
        if k % 200 == 0:
            ux, uy, _ = solver.velocity()
            umax = max(umax, float((ux * ux + uy * uy).max().sqrt().item()))
        if k >= steps - int(steps * avg_fraction):
            t_inner.append(solver.last_group_torque["inner"])
            t_outer.append(solver.last_group_torque["outer"])
    wall = time.perf_counter() - t0

    ux, uy, uz = solver.velocity()
    _, _ = solver.moments(solver.f)
    rr_np = torch.sqrt(px * px + py * py)
    theta = torch.atan2(py, px)
    ut_num = -ux * torch.sin(theta) + uy * torch.cos(theta)
    ur_num = ux * torch.cos(theta) + uy * torch.sin(theta)

    prof = []
    for r in range(radius_inner + 1, radius_outer):
        band = (rr_np - r).abs() < 0.5
        if not bool(band.any()):
            continue
        prof.append({
            "r": r,
            "ut_num": float(ut_num[band].mean().item()),
            "ut_ana": float(exact["A"] * r + exact["B"] / r),
            "ur_rms": float(ur_num[band].abs().mean().item()),
        })
    u_scale = abs(exact["u_inner"])
    prof_err = max((abs(row["ut_num"] - row["ut_ana"]) / u_scale for row in prof), default=float("nan"))

    rc = 0.5 * (radius_inner + radius_outer)
    band = (rr_np - rc).abs() < 2.0
    rr_sel = rr_np[band].to(torch.float64)
    ut_sel = ut_num[band].to(torch.float64)
    mat = torch.stack((rr_sel, 1.0 / rr_sel), dim=1)
    coef = torch.linalg.lstsq(mat, ut_sel).solution
    a_fit = float(coef[0].item())
    b_fit = float(coef[1].item())
    dut_dr = a_fit - b_fit / (rc * rc)
    ut_rc = a_fit * rc + b_fit / rc
    t_cv = 2.0 * math.pi * rc * rc * nu * (dut_dr - ut_rc / rc) * nz

    torque_inner = sum(t_inner) / len(t_inner)
    torque_outer = sum(t_outer) / len(t_outer)
    torque_ana = exact["torque_per_span"] * nz
    return {
        "case": "annulus_couette",
        "format": format,
        "collision": collision,
        "tau": tau,
        "nu": nu,
        "radius_inner": radius_inner,
        "radius_outer": radius_outer,
        "gap_ratio": radius_outer / radius_inner,
        "reynolds_theta": reynolds_theta,
        "omega": omega,
        "nz": nz, "ny": ny, "nx": nx,
        "steps": steps,
        "wall_seconds": wall,
        "u_max_observed": umax,
        "torque_me": torque_outer,
        "torque_me_inner": torque_inner,
        "torque_analytic": torque_ana,
        "torque_rel_err": abs(abs(torque_outer) - torque_ana) / torque_ana,
        "torque_inner_rel_err": abs(abs(torque_inner) - torque_ana) / torque_ana,
        "torque_cv": t_cv,
        "torque_cv_rel_err": abs(abs(t_cv) - torque_ana) / torque_ana,
        "me_vs_cv_rel_diff": abs(abs(torque_outer) - abs(t_cv)) / max(abs(torque_outer), 1e-30),
        "inner_vs_outer_rel_diff": abs(abs(torque_inner) - abs(torque_outer)) / max(abs(torque_outer), 1e-30),
        "profile_max_rel_err": prof_err,
        "profile": prof,
        "geometry": geo.summary(),
        "reference": "u_t = A r + B/r; T = 4 pi mu Omega R1^2 R2^2 / (R2^2 - R1^2)",
    }


# ---------------------------------------------------------------------------
# case 4: uniform flow past a sphere (and its spanwise-cylinder companion)
# ---------------------------------------------------------------------------


def run_body_drag(
    kind: str = "sphere",
    diameter: int = 16,
    reynolds: float = 100.0,
    u_in: float = 0.1,
    format: str = "bouzidi_quadratic",
    collision: str = "regularized",
    tau: float | None = None,
    upstream: float = 2.0,
    downstream: float = 4.0,
    lateral: float = 3.0,
    steps_per_diameter: float = 300.0,
    avg_fraction: float = 0.4,
    cv_margin: int = 4,
    nz: int | None = None,
    sponge_width: int = 6,
    cv_every: int = 50,
) -> dict[str, Any]:
    """Uniform flow past a sphere (or a spanwise cylinder) at fixed Reynolds."""
    if kind not in ("sphere", "cylinder"):
        raise ValueError("kind must be 'sphere' or 'cylinder'")
    nu = u_in * diameter / reynolds
    tau_used = 0.5 + 3.0 * nu if tau is None else float(tau)
    radius = 0.5 * diameter

    nx = int(round((upstream + downstream) * diameter)) + 1
    ny = int(round(2.0 * lateral * diameter)) + 1
    if kind == "cylinder":
        nz = 2 if nz is None else int(nz)
        free_slip = ("y0", "y1")
    else:
        nz = ny if nz is None else int(nz)
        free_slip = ("y0", "y1", "z0", "z1")
    cx0 = float(int(round(upstream * diameter)))
    cy0 = float((ny - 1) // 2)
    cz0 = float((nz - 1) // 2)
    spec = {"center": (cz0, cy0, cx0), "radius": radius}
    geo = build_geometry(kind, nz, ny, nx, spec, format=format)
    cfg = FlowConfig(
        nz=nz, ny=ny, nx=nx, tau=tau_used, format=format, collision=collision,
        free_slip=free_slip, inlet_outlet=True, u_inflow=u_in,
        sponge_width=sponge_width, sponge_strength=0.2,
    )
    solver = Solver(cfg, geo)

    steps = int(round(steps_per_diameter * diameter))
    window = max(1, int(steps * avg_fraction))
    cd_hist: list[float] = []
    cd_cv_hist: list[float] = []
    box = None
    if kind == "sphere":
        m = radius + cv_margin
        box = (
            (int(max(cz0 - m, 1)), int(min(cz0 + m, nz - 2))),
            (int(max(cy0 - m, 1)), int(min(cy0 + m, ny - 2))),
            (int(max(cx0 - m, 1)), int(min(cx0 + m, nx - 2))),
        )
    area = math.pi * radius * radius if kind == "sphere" else diameter * nz
    t0 = time.perf_counter()
    rho_mean = 1.0
    for k in range(steps):
        solver.step(1.0)
        if k >= steps - window:
            rho, _ = solver.moments(solver.f)
            rho_mean = float(rho.mean().item())
            cd_hist.append(float(solver.last_force[0].item()) /
                           (0.5 * rho_mean * u_in * u_in * area))
            if box is not None and (k - (steps - window)) % cv_every == 0:
                ux, uy, uz = solver.moments(solver.f)[1]
                cd_cv_hist.append(
                    stress_integral_drag(ux, uy, uz, rho, nu, box)
                    / (0.5 * rho_mean * u_in * u_in * area)
                )
    wall = time.perf_counter() - t0

    ux, uy, uz = solver.velocity()
    rho, _ = solver.moments(solver.f)
    umag = torch.sqrt(ux * ux + uy * uy + uz * uz)
    cd = sum(cd_hist) / len(cd_hist)
    cd_std = float(torch.tensor(cd_hist).std().item()) if len(cd_hist) > 1 else 0.0
    cd_ref = CD_ANCHORS.get(float(reynolds), schiller_naumann(reynolds))
    ic = int(cz0)
    return {
        "case": f"{kind}_drag",
        "kind": kind,
        "format": format,
        "collision": collision,
        "diameter": diameter,
        "reynolds": reynolds,
        "u_in": u_in,
        "nu": nu,
        "tau": tau_used,
        "nx": nx, "ny": ny, "nz": nz,
        "cells": int(nx * ny * nz),
        "steps": steps,
        "wall_seconds": wall,
        "cd": cd,
        "cd_std": cd_std,
        "cd_reference": cd_ref,
        "cd_reference_name": ("campaign anchor" if float(reynolds) in CD_ANCHORS
                              else "Schiller-Naumann"),
        "cd_schiller_naumann": schiller_naumann(reynolds),
        "cd_rel_err": abs(cd - cd_ref) / cd_ref,
        "cd_cv_mean": (sum(cd_cv_hist) / len(cd_cv_hist)) if cd_cv_hist else None,
        "cd_cv_std": (float(torch.tensor(cd_cv_hist).std().item())
                      if len(cd_cv_hist) > 1 else None),
        "me_vs_cv_rel_diff": (abs(cd - sum(cd_cv_hist) / len(cd_cv_hist)) / abs(cd)
                              if cd_cv_hist else None),
        "cv_box": box,
        "mass_drift": float(rho.mean().item()) - 1.0,
        "max_mach": float(umag.max().item()) * math.sqrt(3.0),
        "geometry": geo.summary(),
        "fields": {
            "ux_mid": (ux[:, :, :] if kind == "cylinder" else ux[ic, :, :]).clone(),
            "uy_mid": (uy[:, :, :] if kind == "cylinder" else uy[ic, :, :]).clone(),
            "umag_mid": (umag[:, :, :] if kind == "cylinder" else umag[ic, :, :]).clone(),
            "center": (int(cz0), int(cy0), int(cx0)),
            "radius": radius,
        },
    }


# ---------------------------------------------------------------------------
# convergence-order bookkeeping
# ---------------------------------------------------------------------------


def convergence_order(sizes: Sequence[float], errors: Sequence[float]) -> dict[str, Any]:
    """Fit the observed order ``p`` from ``error ~ h^p`` (``h ~ 1/size``)."""
    pts = sorted(zip(sizes, errors))
    if len(pts) < 2:
        return {"order": float("nan"), "order_finest_pair": float("nan"), "pairs": []}
    pairs = []
    for (s0, e0), (s1, e1) in zip(pts[:-1], pts[1:]):
        if e0 <= 0 or e1 <= 0:
            continue
        pairs.append({"size_coarse": s0, "size_fine": s1,
                      "p": math.log(e0 / e1) / math.log(s1 / s0)})
    order_finest = pairs[-1]["p"] if pairs else float("nan")
    # global least-squares slope over all points
    xs = [math.log(s) for s, e in pts if e > 0]
    ys = [math.log(e) for s, e in pts if e > 0]
    order = float("nan")
    if len(xs) >= 2:
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        order = num / den if den > 0 else float("nan")
    return {"order": order, "order_finest_pair": order_finest, "pairs": pairs}


def annulus_poiseuille_exact(radius_inner: float, radius_outer: float, g_theta: float,
                             nu: float) -> dict[str, float]:
    """Exact steady swirl between two fixed cylinders driven by ``f = G e_theta``.

    Solving ``nu [u'' + u'/r - u/r^2] = -G`` with ``u(R1) = u(R2) = 0`` gives
    ``u(r) = a r^2 + b r + c/r``; the wall torque per unit span follows from
    ``tau_rtheta = rho nu (du/dr - u/r)``.
    """
    r1, r2 = float(radius_inner), float(radius_outer)
    a = -g_theta / (3.0 * nu)
    b = -a * (r1 ** 3 - r2 ** 3) / (r1 ** 2 - r2 ** 2)
    c = -a * r1 ** 3 - b * r1 ** 2
    torque = lambda r: 2.0 * math.pi * nu * (a * r ** 3 - c)  # rho = 1, per span
    return {"a": a, "b": b, "c": c,
            "torque_inner": torque(r1), "torque_outer": torque(r2),
            "u_max": max(abs(a * r * r + b * r + c / r)
                         for r in [r1 + 0.01 * (r2 - r1) * i for i in range(101)])}


def run_annulus_poiseuille(
    radius_inner: int = 16,
    radius_outer: int = 32,
    u_target: float = 0.03,
    tau: float = 0.65,
    format: str = "bouzidi_quadratic",
    nz: int = 2,
    steps: int = 6000,
    avg_fraction: int | float = 0.25,
    collision: str = "regularized",
) -> dict[str, Any]:
    """Body-force-driven swirl between two *stationary* cylinders (exact torque).

    Both walls are at rest, so this case isolates the geometric accuracy of the
    wall format from any moving-wall momentum injection.
    """
    nu = (tau - 0.5) / 3.0
    gap = float(radius_outer - radius_inner)
    # first guess for the force scale from the narrow-gap estimate u ~ G gap^2/(12 nu)
    g = 12.0 * nu * u_target / (gap * gap)
    exact = annulus_poiseuille_exact(radius_inner, radius_outer, g, nu)
    g *= u_target / max(exact["u_max"], 1e-12)
    exact = annulus_poiseuille_exact(radius_inner, radius_outer, g, nu)

    nx = ny = 2 * int(radius_outer) + 3
    center = (0.0, (ny - 1) / 2.0, (nx - 1) / 2.0)
    spec = {"center": center, "radius": float(radius_inner),
            "radius_outer": float(radius_outer)}
    geo = build_geometry("annulus", nz, ny, nx, spec, format=format)
    cfg = FlowConfig(nz=nz, ny=ny, nx=nx, tau=tau, format=format, collision=collision,
                     force_theta=g)
    solver = Solver(cfg, geo)

    # start from the exact field
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=solver.dt),
        torch.arange(ny, dtype=solver.dt),
        torch.arange(nx, dtype=solver.dt),
        indexing="ij",
    )
    px = xx - float(center[2])
    py = yy - float(center[1])
    rr = torch.sqrt(px * px + py * py).clamp_min(1e-6)
    ut = exact["a"] * rr * rr + exact["b"] * rr + exact["c"] / rr
    inside = (rr > radius_inner) & (rr < radius_outer)
    ut = torch.where(inside, ut, torch.zeros_like(ut))
    ux0 = -ut * py / rr
    uy0 = ut * px / rr
    rho0 = torch.ones((nz, ny, nx), dtype=solver.dt)
    solver.f = solver.equilibrium(rho0, (ux0, uy0, torch.zeros_like(ux0)))

    t_in: list[float] = []
    t_out: list[float] = []
    t0 = time.perf_counter()
    n_avg = max(1, int(steps * avg_fraction))
    for k in range(steps):
        solver.step(1.0)
        if k >= steps - n_avg:
            t_in.append(solver.last_group_torque["inner"])
            t_out.append(solver.last_group_torque["outer"])
    wall = time.perf_counter() - t0

    ux, uy, uz = solver.velocity()
    _, _ = solver.moments(solver.f)
    rr_np = torch.sqrt(px * px + py * py)
    theta = torch.atan2(py, px)
    ut_num = -ux * torch.sin(theta) + uy * torch.cos(theta)
    ur_num = ux * torch.cos(theta) + uy * torch.sin(theta)

    prof = []
    for r in range(radius_inner + 1, radius_outer):
        band = (rr_np - r).abs() < 0.5
        if not bool(band.any()):
            continue
        prof.append({
            "r": r,
            "ut_num": float(ut_num[band].mean().item()),
            "ut_ana": float(exact["a"] * r * r + exact["b"] * r + exact["c"] / r),
            "ur_rms": float(ur_num[band].abs().mean().item()),
        })
    scale = max(abs(exact["u_max"]), 1e-12)
    prof_err = max((abs(row["ut_num"] - row["ut_ana"]) / scale for row in prof), default=float("nan"))

    # control-surface torque at the mid radius from a least-squares fit of u_t(r)
    rc = 0.5 * (radius_inner + radius_outer)
    band = (rr_np - rc).abs() < 2.0
    rr_sel = rr_np[band].to(torch.float64)
    ut_sel = ut_num[band].to(torch.float64)
    mat = torch.stack((rr_sel * rr_sel, rr_sel, 1.0 / rr_sel), dim=1)
    coef = torch.linalg.lstsq(mat, ut_sel).solution
    a_fit, b_fit, c_fit = (float(v.item()) for v in coef)
    dut_dr = 2.0 * a_fit * rc + b_fit
    ut_rc = a_fit * rc * rc + b_fit * rc + c_fit / rc
    t_cv = 2.0 * math.pi * rc * rc * nu * (dut_dr - ut_rc / rc) * nz

    torque_in = sum(t_in) / len(t_in)
    torque_out = sum(t_out) / len(t_out)
    ana_in = abs(exact["torque_inner"]) * nz
    ana_out = abs(exact["torque_outer"]) * nz
    ana_rc = abs(2.0 * math.pi * nu * (exact["a"] * rc ** 3 - exact["c"])) * nz
    body_torque = 2.0 * math.pi * g * (radius_outer ** 3 - radius_inner ** 3) / 3.0 * nz
    return {
        "case": "annulus_poiseuille",
        "format": format,
        "collision": collision,
        "tau": tau,
        "nu": nu,
        "radius_inner": radius_inner,
        "radius_outer": radius_outer,
        "gap_ratio": radius_outer / radius_inner,
        "g_theta": g,
        "u_max_exact": exact["u_max"],
        "reynolds_theta": exact["u_max"] * gap / nu,
        "nz": nz, "ny": ny, "nx": nx,
        "steps": steps,
        "wall_seconds": wall,
        "torque_inner_me": torque_in,
        "torque_outer_me": torque_out,
        "torque_inner_analytic": ana_in,
        "torque_outer_analytic": ana_out,
        "torque_inner_rel_err": abs(abs(torque_in) - ana_in) / ana_in,
        "torque_outer_rel_err": abs(abs(torque_out) - ana_out) / ana_out,
        "torque_cv": t_cv,
        "torque_cv_analytic_rc": ana_rc,
        "torque_cv_rel_err": abs(abs(t_cv) - ana_rc) / ana_rc,
        "body_torque": body_torque,
        "me_balance_rel_err": abs(abs(torque_out) - abs(torque_in) - body_torque) / body_torque,
        "me_vs_cv_rel_diff": abs(abs(t_cv) - ana_rc) / ana_rc,
        "inner_vs_outer_rel_diff": abs(abs(torque_in) - abs(torque_out))
        / max(abs(torque_in), 1e-30),
        "profile_max_rel_err": prof_err,
        "profile": prof,
        "geometry": geo.summary(),
        "reference": "u = a r^2 + b r + c/r (fixed walls), T(r) = 2 pi rho nu (a r^3 - c)",
    }
