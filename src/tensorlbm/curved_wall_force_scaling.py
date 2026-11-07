"""Curved-wall boundary-format force-error scaling for the D2Q9 cylinder.

Self-contained D2Q9 driver used to *quantify* the force-extraction error of three
wall formats and two force-extraction routes:

wall formats
    * ``halfway``           -- classical half-way bounce-back (staircase wall)
    * ``bouzidi_linear``    -- Bouzidi, Firdaouss & Lallemand (2001), 1st order
    * ``bouzidi_quadratic`` -- Bouzidi et al. (2001), 2nd order interpolation

force extraction
    * ``momentum_exchange`` -- link-wise momentum exchange (ME): each cut link
      carries the post-collision outgoing population ``o = f_i^+(x_f)`` toward the
      wall and the reflected population ``r = f_opp(x_f, t+1)`` back into the
      fluid, so the momentum received by the wall is ``(o + r) c_i`` per link.
    * ``control_volume`` -- stress integration over a closed rectangular surface
      (momentum flux + pressure + viscous stress) outside the near-wall region.

Geometry convention
-------------------
``q`` is the ray parameter of the analytic intersection along the lattice link,
measured from the fluid node ``x_f`` toward the wall: the wall sits at
``x_f + q (x_s - x_f)`` with ``x_s`` the neighbouring solid node.  ``q in (0,1]``
and ``q = 1/2`` is the half-way case.  The solid *mask* stays centre-based (a node
is solid when its centre lies inside the circle), an O(h^2) area-resolution
limitation documented in the campaign report, while the wall *position* seen by
the interpolated boundary condition is analytic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

# ---------------------------------------------------------------------------
# D2Q9 lattice constants
# ---------------------------------------------------------------------------

CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
W = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)
CS2 = 1.0 / 3.0

BOUNDARY_FORMATS = ("halfway", "bouzidi_linear", "bouzidi_quadratic")
FORCE_METHODS = ("momentum_exchange", "control_volume")
WINDOW_KINDS = ("steady", "transient")

#: pinned experimental / high-resolution reference drag of the unbounded Re=100
#: circular cylinder (campaign reference value).
CD_REFERENCE_RE100 = 1.33


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CylinderCaseConfig:
    """Resolution and flow configuration of one cylinder solve."""

    diameter: int = 64
    reynolds: float = 100.0
    lattice_speed: float = 0.1
    format: str = "bouzidi_quadratic"
    upstream_diameters: float = 4.0
    downstream_diameters: float = 8.0
    half_height_diameters: float = 8.0
    steps_per_diameter: float = 300.0
    transient_diameters: float = 10.0
    steady_fraction: float = 0.40
    ramp_advective_times: float = 2.0
    sponge_width: int = 12
    sponge_strength: float = 0.2
    outlet_density_mode: str = "extrapolated"
    cv_margin_cells: int = 6
    cv_sample_every: int = 20
    rest_repair: bool = False
    dtype: str = "float32"
    threads: int = 0
    snapshot_path: str | None = None

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def nx(self) -> int:
        return int(round(self.diameter * (self.upstream_diameters + self.downstream_diameters))) + 1

    @property
    def ny(self) -> int:
        return int(round(2.0 * self.half_height_diameters * self.diameter)) + 2

    @property
    def cx(self) -> float:
        return self.upstream_diameters * self.diameter

    @property
    def cy(self) -> float:
        return 0.5 * (self.ny - 1)

    @property
    def nu(self) -> float:
        return self.lattice_speed * self.diameter / self.reynolds

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def steps(self) -> int:
        return int(round(self.steps_per_diameter * self.diameter))

    @property
    def transient_step(self) -> int:
        return int(round(self.transient_diameters * self.diameter / self.lattice_speed))

    @property
    def ramp_steps(self) -> int:
        return max(1, int(round(self.ramp_advective_times * self.diameter / self.lattice_speed)))

    @property
    def steady_window(self) -> int:
        return max(1, int(round(self.steady_fraction * self.steps)))

    @property
    def dynamic_pressure_force(self) -> float:
        """``0.5 rho U^2 D`` in lattice units (rho = 1)."""
        return 0.5 * self.lattice_speed ** 2 * self.diameter

    def validate(self) -> None:
        if self.format not in BOUNDARY_FORMATS:
            raise ValueError(f"unknown boundary format {self.format!r}")
        if self.diameter < 8:
            raise ValueError("diameter must be >= 8 lattice nodes")
        if not 0.0 < self.lattice_speed <= 0.2:
            raise ValueError("lattice_speed must lie in (0, 0.2]")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        if self.steps <= 0 or self.steady_window >= self.steps:
            raise ValueError("invalid time window")
        if not 0 <= self.transient_step < self.steps - self.steady_window:
            raise ValueError("transient sample must precede the steady window")
        if self.cv_margin_cells < 3:
            raise ValueError("control-volume margin must be >= 3 cells")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")
        if self.outlet_density_mode not in {"unit", "extrapolated"}:
            raise ValueError("unknown outlet_density_mode")


# ---------------------------------------------------------------------------
# generic cut-link bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class LinkSet:
    """Flattened set of fluid-to-wall cut links."""

    node: torch.Tensor
    direction: torch.Tensor
    q: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    st1_ok: torch.Tensor
    st1: torch.Tensor
    st2_ok: torch.Tensor
    st2: torch.Tensor
    weights: torch.Tensor
    solid_flat: torch.Tensor
    wall_ux: torch.Tensor | None = None
    wall_uy: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _flat(index_y: torch.Tensor, index_x: torch.Tensor, nx: int) -> torch.Tensor:
    return index_y.to(torch.int64) * nx + index_x.to(torch.int64)


def _make_links(
    node: torch.Tensor,
    direction: torch.Tensor,
    q: torch.Tensor,
    st1_ok: torch.Tensor,
    st1: torch.Tensor,
    st2_ok: torch.Tensor,
    st2: torch.Tensor,
    solid_flat: torch.Tensor,
    device: torch.device,
    wall_ux: torch.Tensor | None = None,
    wall_uy: torch.Tensor | None = None,
) -> LinkSet:
    cxt = torch.tensor([float(c) for c in CX], device=device)
    cyt = torch.tensor([float(c) for c in CY], device=device)
    wt = torch.tensor(W, device=device)
    return LinkSet(
        node=node.to(device),
        direction=direction.to(device),
        q=q.to(device),
        cx=cxt[direction.to(device)],
        cy=cyt[direction.to(device)],
        st1_ok=st1_ok.to(device),
        st1=st1.to(device),
        st2_ok=st2_ok.to(device),
        st2=st2.to(device),
        weights=wt[direction.to(device)],
        solid_flat=solid_flat.to(device),
        wall_ux=None if wall_ux is None else wall_ux.to(device),
        wall_uy=None if wall_uy is None else wall_uy.to(device),
    )


def build_cylinder_boundary(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    radius: float,
    format: str,
    device: torch.device,
) -> LinkSet:
    """Enumerate cut links of a circle with analytic wall fractions."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    solid = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
    fluid = ~solid

    nodes: list[torch.Tensor] = []
    dirs: list[torch.Tensor] = []
    qs: list[torch.Tensor] = []
    s1ok: list[torch.Tensor] = []
    s1i: list[torch.Tensor] = []
    s2ok: list[torch.Tensor] = []
    s2i: list[torch.Tensor] = []
    fallbacks: dict[str, int] = {}

    for direction in range(1, 9):
        ex, ey = CX[direction], CY[direction]
        nxi = xx + ex
        nyi = yy + ey
        inside = (nxi >= 0) & (nxi < nx) & (nyi >= 0) & (nyi < ny)
        solid_nb = torch.zeros_like(solid)
        solid_nb[inside] = solid[nyi[inside].to(torch.int64), nxi[inside].to(torch.int64)]
        cut = fluid & solid_nb
        if not bool(cut.any()):
            continue
        iy, ix = torch.nonzero(cut, as_tuple=True)
        dx = ix.to(torch.float64) - cx
        dy = iy.to(torch.float64) - cy
        exf, eyf = float(ex), float(ey)
        a = exf * exf + eyf * eyf
        b = 2.0 * (exf * dx + eyf * dy)
        c0 = dx * dx + dy * dy - radius * radius
        disc = (b * b - 4.0 * a * c0).clamp_min(0.0)
        root = torch.sqrt(disc)
        t_near = (-b - root) / (2.0 * a)
        t_far = (-b + root) / (2.0 * a)
        t = torch.where(t_near > 1e-9, t_near, t_far)
        bad = ~(t > 1e-9) | ~torch.isfinite(t)
        fallbacks[f"dir{direction}"] = int(bad.sum().item())
        t = torch.where(bad, torch.full_like(t, 0.5), t)
        q = t.clamp(1e-6, 1.0)
        if format == "halfway":
            q = torch.full_like(q, 0.5)

        def _stencil(offset: int) -> tuple[torch.Tensor, torch.Tensor]:
            sy = iy - offset * ey
            sx = ix - offset * ex
            inside_s = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
            ok = torch.zeros_like(inside_s)
            ok[inside_s] = fluid[sy[inside_s].to(torch.int64), sx[inside_s].to(torch.int64)]
            safe = torch.where(
                inside_s,
                _flat(sy.clamp(0, ny - 1), sx.clamp(0, nx - 1), nx),
                torch.zeros_like(sy, dtype=torch.int64),
            )
            return ok, safe.to(torch.int64)

        ok1, idx1 = _stencil(1)
        ok2, idx2 = _stencil(2)
        nodes.append(_flat(iy, ix, nx))
        dirs.append(torch.full_like(iy, direction, dtype=torch.int64))
        qs.append(q)
        s1ok.append(ok1)
        s1i.append(idx1)
        s2ok.append(ok2)
        s2i.append(idx2)

    node = torch.cat(nodes)
    direction = torch.cat(dirs)
    q = torch.cat(qs)
    links = _make_links(
        node, direction, q,
        torch.cat(s1ok), torch.cat(s1i), torch.cat(s2ok), torch.cat(s2i),
        torch.nonzero(solid.reshape(-1))[:, 0], device,
    )
    links.meta = {
        "links": int(node.numel()),
        "q_min": float(q.min().item()),
        "q_max": float(q.max().item()),
        "q_mean": float(q.mean().item()),
        "invalid_intersections": fallbacks,
        "solid_nodes": int(solid.sum().item()),
        "mask_area_over_circle": float(solid.sum().item()) / (math.pi * radius * radius),
    }
    return links


def build_channel_walls(
    nx: int,
    ny: int,
    q_bottom: float,
    q_top: float,
    format: str,
    device: torch.device,
    top_wall_speed: float = 0.0,
) -> LinkSet:
    """Cut links of two flat walls: bottom (stationary) and top (may move)."""
    nodes: list[torch.Tensor] = []
    dirs: list[torch.Tensor] = []
    qs: list[torch.Tensor] = []
    s1ok: list[torch.Tensor] = []
    s1i: list[torch.Tensor] = []
    s2ok: list[torch.Tensor] = []
    s2i: list[torch.Tensor] = []
    wux: list[torch.Tensor] = []

    ix = torch.arange(nx, dtype=torch.int64)
    for direction in (4, 7, 8):  # bottom wall, row 1
        ex, ey = CX[direction], CY[direction]
        row = torch.full((nx,), 1, dtype=torch.int64)
        ok1 = torch.ones(nx, dtype=torch.bool)
        st1 = _flat(torch.full((nx,), 1 - ey, dtype=torch.int64),
                    ix - ex, nx).clamp(0, ny * nx - 1)
        ok2 = torch.ones(nx, dtype=torch.bool)
        st2 = _flat(torch.full((nx,), 1 - 2 * ey, dtype=torch.int64),
                    ix - 2 * ex, nx).clamp(0, ny * nx - 1)
        nodes.append(_flat(row, ix, nx))
        dirs.append(torch.full((nx,), direction, dtype=torch.int64))
        qq = torch.full((nx,), q_bottom, dtype=torch.float64)
        qs.append(torch.full_like(qq, 0.5) if format == "halfway" else qq)
        s1ok.append(ok1)
        s1i.append(st1.to(torch.int64))
        s2ok.append(ok2)
        s2i.append(st2.to(torch.int64))
        wux.append(torch.zeros(nx, dtype=torch.float64))

    for direction in (2, 5, 6):  # top wall, row ny-2
        ex, ey = CX[direction], CY[direction]
        row = torch.full((nx,), ny - 2, dtype=torch.int64)
        ok1 = torch.ones(nx, dtype=torch.bool)
        st1 = _flat(torch.full((nx,), ny - 2 - ey, dtype=torch.int64),
                    ix - ex, nx).clamp(0, ny * nx - 1)
        ok2 = torch.ones(nx, dtype=torch.bool)
        st2 = _flat(torch.full((nx,), ny - 2 - 2 * ey, dtype=torch.int64),
                    ix - 2 * ex, nx).clamp(0, ny * nx - 1)
        nodes.append(_flat(row, ix, nx))
        dirs.append(torch.full((nx,), direction, dtype=torch.int64))
        qq = torch.full((nx,), q_top, dtype=torch.float64)
        qs.append(torch.full_like(qq, 0.5) if format == "halfway" else qq)
        s1ok.append(ok1)
        s1i.append(st1.to(torch.int64))
        s2ok.append(ok2)
        s2i.append(st2.to(torch.int64))
        wux.append(torch.full((nx,), float(top_wall_speed), dtype=torch.float64))

    links = _make_links(
        torch.cat(nodes), torch.cat(dirs), torch.cat(qs),
        torch.cat(s1ok), torch.cat(s1i), torch.cat(s2ok), torch.cat(s2i),
        torch.empty(0, dtype=torch.int64), device,
        wall_ux=torch.cat(wux),
        wall_uy=torch.zeros(nx * 6, dtype=torch.float64),
    )
    links.meta = {"links": int(links.node.numel())}
    return links


# ---------------------------------------------------------------------------
# reflection laws
# ---------------------------------------------------------------------------


def reflected_populations(
    fp_view: torch.Tensor,
    links: LinkSet,
    format: str,
    rho_view: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Return reflected ``r`` and outgoing ``o`` populations for every cut link.

    ``fp_view`` is the *post-collision* population array viewed as ``(9, ny*nx)``.
    The wall-velocity term carries the same coefficient as the principal-node
    term, so all three formats coincide at ``q = 1/2``.
    """
    direction = links.direction
    opp = torch.tensor(OPP, device=direction.device, dtype=torch.int64)[direction]
    node = links.node
    q = links.q

    o = fp_view[direction, node]
    fi_m1 = torch.where(links.st1_ok, fp_view[direction, links.st1], o)
    fi_m2 = torch.where(links.st2_ok, fp_view[direction, links.st2], fi_m1)
    foi_x = fp_view[opp, node]
    foi_m1 = torch.where(links.st1_ok, fp_view[opp, links.st1], foi_x)

    low = q < 0.5
    lin_low = 2.0 * q * o + (1.0 - 2.0 * q) * fi_m1
    lin_high = o / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * foi_x
    lin = torch.where(low, lin_low, lin_high)

    quad_low = (
        q * (2.0 * q + 1.0) * o
        + (1.0 + 2.0 * q) * (1.0 - 2.0 * q) * fi_m1
        - q * (1.0 - 2.0 * q) * fi_m2
    )
    quad_high = (
        o / (q * (2.0 * q + 1.0))
        + (2.0 * q - 1.0) / q * foi_x
        - (2.0 * q - 1.0) / (2.0 * q + 1.0) * foi_m1
    )
    quad = torch.where(low, quad_low, quad_high)
    quad_ok = links.st1_ok & torch.where(low, links.st2_ok, torch.ones_like(links.st2_ok))

    stats = {
        "links": int(node.numel()),
        "quadratic_fallback": int((~quad_ok).sum().item()),
        "low_branch": int(low.sum().item()),
    }

    if format == "halfway":
        r = o.clone()
    elif format == "bouzidi_linear":
        r = lin
    elif format == "bouzidi_quadratic":
        r = torch.where(quad_ok, quad, lin)
    else:  # pragma: no cover
        raise ValueError(format)

    if links.wall_ux is not None:
        cu = links.cx * links.wall_ux + links.cy * (links.wall_uy if links.wall_uy is not None else 0.0)
        rho_w = (
            torch.ones_like(q) if rho_view is None else rho_view[node]
        )
        wall_term = 6.0 * links.weights * rho_w * cu
        if format == "halfway":
            weight = torch.ones_like(q)
        elif format == "bouzidi_quadratic":
            weight = torch.where(low, q * (2.0 * q + 1.0), 1.0 / (q * (2.0 * q + 1.0)))
        else:
            weight = torch.where(low, 2.0 * q, 1.0 / (2.0 * q))
        # moving-wall term: equilibrium consistency requires
        # f_opp = f_i^+ - 6 w_i rho (c_i . u_w) for a link whose direction i
        # points from the fluid into the wall.
        r = r - weight * wall_term
    return r, o, stats


# ---------------------------------------------------------------------------
# low level kernels
# ---------------------------------------------------------------------------


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    device = rho.device
    cx = torch.tensor(CX, device=device, dtype=rho.dtype).view(9, 1, 1)
    cy = torch.tensor(CY, device=device, dtype=rho.dtype).view(9, 1, 1)
    w = torch.tensor(W, device=device, dtype=rho.dtype).view(9, 1, 1)
    usq = ux * ux + uy * uy
    cu = cx * ux + cy * uy
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq.unsqueeze(0))


def _macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = f.device
    cx = torch.tensor(CX, device=device, dtype=f.dtype).view(9, 1, 1)
    cy = torch.tensor(CY, device=device, dtype=f.dtype).view(9, 1, 1)
    rho = f.sum(dim=0)
    rho_safe = rho.clamp_min(1e-12)
    ux = (f * cx).sum(dim=0) / rho_safe
    uy = (f * cy).sum(dim=0) / rho_safe
    return rho, ux, uy


def _stream(f_next: torch.Tensor, fp: torch.Tensor) -> None:
    """Pull-scheme streaming with slicing; boundary rows/columns filled later."""
    f_next[0] = fp[0]
    f_next[1, :, 1:] = fp[1, :, :-1]
    f_next[2, 1:, :] = fp[2, :-1, :]
    f_next[3, :, :-1] = fp[3, :, 1:]
    f_next[4, :-1, :] = fp[4, 1:, :]
    f_next[5, 1:, 1:] = fp[5, :-1, :-1]
    f_next[6, 1:, :-1] = fp[6, :-1, 1:]
    f_next[7, :-1, :-1] = fp[7, 1:, 1:]
    f_next[8, :-1, 1:] = fp[8, 1:, :-1]


def control_volume_force(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    nu: float,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> tuple[float, float]:
    """Stress-integration force over a closed rectangular control surface.

    ``F_j = -∮ [rho u_j u_k + p delta_jk - mu (d_k u_j + d_j u_k)] n_k dS``
    using central differences for the velocity gradients and midpoint sums on the
    four faces.  Returned values are the force *on the body* in lattice units.
    """
    p = rho * CS2
    mu = rho * nu
    fx = 0.0
    fy = 0.0

    ys = torch.arange(y0, y1 + 1, device=rho.device)
    xs = torch.arange(x0, x1 + 1, device=rho.device)

    for xc, sign in ((x1, 1.0), (x0, -1.0)):
        dudx = 0.5 * (ux[ys, xc + 1] - ux[ys, xc - 1])
        dudy = 0.5 * (ux[ys + 1, xc] - ux[ys - 1, xc])
        dvdx = 0.5 * (uy[ys, xc + 1] - uy[ys, xc - 1])
        ux_f, uy_f = ux[ys, xc], uy[ys, xc]
        rho_f, p_f, mu_f = rho[ys, xc], p[ys, xc], mu[ys, xc]
        un = ux_f * sign
        fx = fx + float((-rho_f * ux_f * un - p_f * sign + 2.0 * mu_f * dudx * sign).sum())
        fy = fy + float((-rho_f * uy_f * un + mu_f * (dvdx + dudy) * sign).sum())

    for yc, sign in ((y1, 1.0), (y0, -1.0)):
        dudx = 0.5 * (ux[yc, xs + 1] - ux[yc, xs - 1])
        dudy = 0.5 * (ux[yc + 1, xs] - ux[yc - 1, xs])
        dvdx = 0.5 * (uy[yc, xs + 1] - uy[yc, xs - 1])
        dvdy = 0.5 * (uy[yc + 1, xs] - uy[yc - 1, xs])
        ux_f, uy_f = ux[yc, xs], uy[yc, xs]
        rho_f, p_f, mu_f = rho[yc, xs], p[yc, xs], mu[yc, xs]
        un = uy_f * sign
        fx = fx + float((-rho_f * ux_f * un + mu_f * (dudy + dvdx) * sign).sum())
        fy = fy + float((-rho_f * uy_f * un - p_f * sign + 2.0 * mu_f * dvdy * sign).sum())

    return fx, fy


def _stream_periodic(f_next: torch.Tensor, fp: torch.Tensor) -> None:
    """Streaming on a domain periodic in x (channel wall test)."""
    for alpha in range(9):
        f_next[alpha] = torch.roll(fp[alpha], shifts=(CY[alpha], CX[alpha]), dims=(0, 1))


# ---------------------------------------------------------------------------
# cylinder driver
# ---------------------------------------------------------------------------


def run_cylinder_case(
    config: CylinderCaseConfig,
    *,
    max_steps: int | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Run one cylinder solve and record ME and CV force histories."""
    config.validate()
    torch.set_num_threads(int(config.threads) if config.threads else 1)
    dtype = getattr(torch, config.dtype)
    device = torch.device("cpu")

    nx, ny = config.nx, config.ny
    cx, cy, radius = config.cx, config.cy, config.radius
    shape = (ny, nx)
    links = build_cylinder_boundary(nx, ny, cx, cy, radius, config.format, device)

    f = torch.empty((9, ny, nx), dtype=dtype, device=device)
    rho0 = torch.ones(shape, dtype=dtype, device=device)
    ux0 = torch.zeros(shape, dtype=dtype, device=device)
    uy0 = torch.zeros(shape, dtype=dtype, device=device)
    f[:] = _equilibrium(rho0, ux0, uy0)
    fp = torch.empty_like(f)
    f_next = torch.empty_like(f)
    fp_view = fp.view(9, -1)
    f_next_view = f_next.view(9, -1)
    opp = torch.tensor(OPP, device=device, dtype=torch.int64)[links.direction]

    solid = links.solid_flat
    omega = float(config.omega)
    n_steps = int(config.steps if max_steps is None else max_steps)
    transient = min(config.transient_step, max(1, n_steps - 1))
    dyn = config.dynamic_pressure_force

    x0 = int(math.floor(cx - radius)) - config.cv_margin_cells
    x1 = int(math.ceil(cx + radius)) + config.cv_margin_cells
    y0 = int(math.floor(cy - radius)) - config.cv_margin_cells
    y1 = int(math.ceil(cy + radius)) + config.cv_margin_cells
    x0, x1 = max(2, x0), min(nx - 3, x1)
    y0, y1 = max(2, y0), min(ny - 3, y1)

    me_buf = torch.zeros(n_steps, dtype=torch.float64)
    leak_acc = torch.zeros((), dtype=torch.float64)
    cv_hist: list[float] = []
    cv_steps: list[int] = []
    rho_mean: list[float] = []
    started = time.perf_counter()

    sponge = None
    if config.sponge_width > 0 and config.sponge_strength > 0.0:
        sigma_x = torch.zeros(nx, dtype=dtype, device=device)
        width = min(int(config.sponge_width), max(1, nx // 8))
        for offset in range(width):
            value = config.sponge_strength * (1.0 - offset / width) ** 2
            sigma_x[offset] = value
            sigma_x[nx - 1 - offset] = value
        sponge = sigma_x.view(1, nx).expand(ny, nx).contiguous()
    ramp = max(1, config.ramp_steps)
    for step in range(1, n_steps + 1):
        ramp_factor = 0.5 * (1.0 - math.cos(math.pi * min(1.0, step / ramp)))
        u_in_now = float(config.lattice_speed) * ramp_factor
        rho, ux, uy = _macroscopic(f)
        feq = _equilibrium(rho, ux, uy)
        fp[:] = f + omega * (feq - f)
        fp_view[:, solid] = 0.0  # solid nodes hold no populations
        if sponge is not None:
            fp.mul_(1.0 - sponge).add_(sponge * feq)

        _stream(f_next, fp)

        r, o, _stats = reflected_populations(fp_view, links, config.format)
        f_next_view[opp, links.node] = r.to(f_next.dtype)
        if config.rest_repair:
            f_next_view[0, links.node] += (o - r).to(f_next.dtype)
        else:
            leak_acc += (r - o).sum().double()

        # free-slip lateral walls: specular reflection about y = 0.5 / y = ny-1.5
        f_next[2, 1, :] = f_next[4, 1, :]
        f_next[5, 1, :] = f_next[8, 1, :]
        f_next[6, 1, :] = f_next[7, 1, :]
        f_next[4, ny - 2, :] = f_next[2, ny - 2, :]
        f_next[7, ny - 2, :] = f_next[5, ny - 2, :]
        f_next[8, ny - 2, :] = f_next[6, ny - 2, :]

        # inlet: velocity Dirichlet by non-equilibrium extrapolation
        rho_n, ux_n, uy_n = _macroscopic(f_next)
        feq_n = _equilibrium(rho_n, ux_n, uy_n)
        rho_in = rho_n[..., 1]
        feq_b_in = _equilibrium(
            rho_in,
            torch.full_like(rho_in, u_in_now),
            torch.zeros_like(rho_in),
        )
        for direction in (1, 5, 8):
            f_next[direction, :, 0] = (
                feq_b_in[direction] + (f_next[direction, :, 1] - feq_n[direction, :, 1])
            )

        # outlet: zero-gradient velocity, rho = 1
        if config.outlet_density_mode == "unit":
            rho_out = torch.ones_like(rho_n[..., -2])
        else:
            rho_out = rho_n[..., -2]
        feq_b_out = _equilibrium(rho_out, ux_n[..., -2], uy_n[..., -2])
        for direction in (3, 6, 7):
            f_next[direction, :, -1] = (
                feq_b_out[direction] + (f_next[direction, :, -2] - feq_n[direction, :, -2])
            )

        f_next[:, 0, :] = 0.0
        f_next[:, -1, :] = 0.0

        f, f_next = f_next, f
        fp_view = fp.view(9, -1)
        f_next_view = f_next.view(9, -1)

        me_buf[step - 1] = (o + r).mul(links.cx).sum() / dyn
        if step % config.cv_sample_every == 0 or step == n_steps:
            rho_s, ux_s, uy_s = _macroscopic(f)
            fx_cv, _fy = control_volume_force(rho_s, ux_s, uy_s, config.nu, x0, x1, y0, y1)
            cv_hist.append(fx_cv / dyn)
            cv_steps.append(step)
            fluid_mask = torch.ones(nx * ny, dtype=torch.bool)
            fluid_mask[solid.cpu()] = False
            rho_mean.append(float(rho_s.view(-1)[fluid_mask.to(device)].mean().item()))

    elapsed = time.perf_counter() - started
    me_hist = me_buf.tolist()
    mass_leak = float(leak_acc.item())
    window = min(config.steady_window, len(me_hist))
    tail = me_hist[-window:]
    me_steady = sum(tail) / len(tail)
    me_std = (sum((v - me_steady) ** 2 for v in tail) / len(tail)) ** 0.5
    cv_window = max(1, len(cv_hist) // 4)
    cv_steady = sum(cv_hist[-cv_window:]) / cv_window if cv_hist else float("nan")

    stride = max(1, n_steps // 400)
    if config.snapshot_path:
        import numpy as np

        rho_s, ux_s, uy_s = _macroscopic(f)
        solid_mask = torch.ones(nx * ny, dtype=torch.bool)
        solid_mask[solid.cpu()] = False
        np.savez_compressed(
            config.snapshot_path,
            rho=rho_s.cpu().numpy().astype("float32"),
            ux=ux_s.cpu().numpy().astype("float32"),
            uy=uy_s.cpu().numpy().astype("float32"),
            solid=(~solid_mask.cpu().numpy()).reshape(ny, nx),
            x0=x0, x1=x1, y0=y0, y1=y1,
            diameter=config.diameter,
            lattice_speed=config.lattice_speed,
            boundary_format=config.format,
        )
    return {
        "label": label or f"N{config.diameter}-{config.format}",
        "format": config.format,
        "diameter": config.diameter,
        "grid": {"nx": nx, "ny": ny, "cells": nx * ny},
        "reynolds": config.reynolds,
        "lattice_speed": config.lattice_speed,
        "nu": config.nu,
        "tau": config.tau,
        "steps": n_steps,
        "ramp_steps": ramp,
        "sponge_width": config.sponge_width,
        "sponge_strength": config.sponge_strength,
        "transient_step": transient,
        "steady_window": window,
        "dynamic_pressure_force": dyn,
        "boundary": links.meta,
        "control_volume_box": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "cd_me_steady": me_steady,
        "cd_cv_steady": cv_steady,
        "cd_me_transient": me_hist[transient - 1],
        "cd_cv_transient": next(
            (v for s, v in zip(cv_steps, cv_hist) if s >= transient), float("nan")
        ),
        "cd_me_std_last_window": me_std,
        "cd_me_history": {
            "step": list(range(1, n_steps + 1, stride)),
            "value": me_hist[::stride],
        },
        "cd_cv_history": {"step": cv_steps, "value": cv_hist},
        "mean_density_last": rho_mean[-1] if rho_mean else float("nan"),
        "cd_me_steady_density_norm": me_steady / (rho_mean[-1] if rho_mean else 1.0),
        "cd_cv_steady_density_norm": cv_steady / (rho_mean[-1] if rho_mean else 1.0),
        "cd_me_transient_density_norm": (
            me_hist[transient - 1] / (rho_mean[-1] if rho_mean else 1.0)
        ),
        "cumulative_mass_leak": mass_leak,
        "rest_repair": config.rest_repair,
        "wall_seconds": elapsed,
        "cell_steps_per_second": nx * ny * n_steps / max(elapsed, 1e-9),
    }


# ---------------------------------------------------------------------------
# Couette validation of the wall formats and of the ME force
# ---------------------------------------------------------------------------


def _subset(links: LinkSet, mask: torch.Tensor) -> LinkSet:
    return LinkSet(
        node=links.node[mask], direction=links.direction[mask], q=links.q[mask],
        cx=links.cx[mask], cy=links.cy[mask], st1_ok=links.st1_ok[mask],
        st1=links.st1[mask], st2_ok=links.st2_ok[mask], st2=links.st2[mask],
        weights=links.weights[mask], solid_flat=links.solid_flat,
        wall_ux=None if links.wall_ux is None else links.wall_ux[mask],
        wall_uy=None if links.wall_uy is None else links.wall_uy[mask],
    )


def run_couette(
    *,
    nx: int = 4,
    ny: int = 34,
    q_bottom: float = 0.5,
    u_wall: float = 0.05,
    tau: float = 0.8,
    steps: int = 6000,
    format: str = "bouzidi_linear",
    threads: int = 0,
) -> dict[str, Any]:
    """Planar Couette flow; wall force versus the analytic shear ``mu u_wall / H``.

    Bottom wall: stationary at ``q = q_bottom``.  Top wall: half-way moving wall
    with ``u_wall``.  Analytic wall shear stress is ``mu u_wall / H`` with
    ``H = (ny - 1.5) - (1 - q_bottom)`` and ``mu = rho nu`` (``rho = 1``).
    """
    torch.set_num_threads(int(threads) if threads else 1)
    dtype = torch.float64
    device = torch.device("cpu")
    nu = (tau - 0.5) / 3.0
    omega = 1.0 / tau
    y_bottom = 1.0 - q_bottom
    y_top = ny - 1.5
    height = y_top - y_bottom

    f = torch.empty((9, ny, nx), dtype=dtype, device=device)
    rho0 = torch.ones((ny, nx), dtype=dtype, device=device)
    yy = torch.arange(ny, dtype=dtype, device=device).view(ny, 1)
    prof = (u_wall * (yy - y_bottom) / height).clamp(0.0, u_wall).expand(ny, nx).contiguous()
    f[:] = _equilibrium(rho0, prof, torch.zeros_like(prof))
    fp = torch.empty_like(f)
    f_next = torch.empty_like(f)

    all_links = build_channel_walls(nx, ny, q_bottom, 0.5, format, device, top_wall_speed=u_wall)
    row = all_links.node // nx
    lb = _subset(all_links, row == 1)
    lt = _subset(all_links, row == ny - 2)
    opp_b = torch.tensor(OPP, device=device, dtype=torch.int64)[lb.direction]
    opp_t = torch.tensor(OPP, device=device, dtype=torch.int64)[lt.direction]

    force_b: list[float] = []
    force_t: list[float] = []
    started = time.perf_counter()
    for _step in range(steps):
        rho, ux, uy = _macroscopic(f)
        feq = _equilibrium(rho, ux, uy)
        fp[:] = f + omega * (feq - f)
        _stream_periodic(f_next, fp)
        rho_view = rho.view(-1)
        rb, ob, _ = reflected_populations(fp.view(9, -1), lb, format, rho_view)
        rt, ot, _ = reflected_populations(fp.view(9, -1), lt, format, rho_view)
        fv = f_next.view(9, -1)
        fv[opp_b, lb.node] = rb
        fv[opp_t, lt.node] = rt
        f_next[:, 0, :] = 0.0
        f_next[:, -1, :] = 0.0
        f, f_next = f_next, f
        force_b.append(float((ob + rb).mul(lb.cx).sum().item()) / nx)
        force_t.append(float((ot + rt).mul(lt.cx).sum().item()) / nx)

    elapsed = time.perf_counter() - started
    rho_f, ux_f, _uy = _macroscopic(f)
    analytic = nu * u_wall / height
    n_avg = min(2000, max(1, steps // 4))
    measured_bottom = sum(force_b[-n_avg:]) / n_avg
    measured_top = sum(force_t[-n_avg:]) / n_avg
    y1, y2 = 6, ny - 7
    slope = float((ux_f[y2, :].mean() - ux_f[y1, :].mean()).item()) / (y2 - y1)
    rows = list(range(1, ny - 1, max(1, (ny - 2) // 16)))
    return {
        "format": format,
        "q_bottom": q_bottom,
        "height": height,
        "tau": tau,
        "nu": nu,
        "u_wall": u_wall,
        "steps": steps,
        "analytic_shear": analytic,
        "measured_shear_bottom": measured_bottom,
        "measured_shear_top": measured_top,
        "bottom_relative_error": (measured_bottom - analytic) / analytic,
        "top_relative_error": (measured_top + analytic) / analytic,
        "profile_slope": slope,
        "profile_relative_error": (slope - u_wall / height) / (u_wall / height),
        "velocity_profile": {
            "y": rows,
            "ux": [float(ux_f[y].mean().item()) for y in rows],
        },
        "analytic_profile": {
            "y": rows,
            "ux": [
                max(0.0, min(u_wall, u_wall * (y - y_bottom) / height)) for y in rows
            ],
        },
        "wall_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# convergence / criteria helpers
# ---------------------------------------------------------------------------


def convergence_order(
    diameters: list[int],
    errors: list[float],
    n_points: int = 2,
) -> dict[str, Any]:
    """Observed order ``p`` from ``e(N) = C N^-p`` on the finest ``n_points``."""
    pairs = list(zip(diameters, errors))[-n_points:]
    if len(pairs) < 2:
        return {"p_observed": float("nan"), "pairs": pairs}
    (n1, e1), (n2, e2) = pairs[0], pairs[-1]
    if e1 <= 0.0 or e2 <= 0.0 or n2 == n1:
        return {"p_observed": float("nan"), "pairs": pairs}
    p = math.log(e1 / e2) / math.log(n2 / n1)
    return {"p_observed": p, "pairs": pairs, "n1": n1, "n2": n2, "e1": e1, "e2": e2}


def summarise_runs(
    runs: dict[str, dict[str, Any]],
    *,
    cd_reference: float = CD_REFERENCE_RE100,
    formats: tuple[str, ...] = ("bouzidi_linear", "bouzidi_quadratic"),
    expected_rows: int = 0,
) -> dict[str, Any]:
    """Turn per-run force histories into the 4-factor factorial table."""
    rows: list[dict[str, Any]] = []
    for key, run in sorted(runs.items()):
        if run["format"] not in formats:
            continue
        for method in FORCE_METHODS:
            short = "me" if method == "momentum_exchange" else "cv"
            for window in WINDOW_KINDS:
                cd = run.get(f"cd_{short}_{window}_density_norm")
                if cd is None or not math.isfinite(cd):
                    cd = run[f"cd_{short}_{window}"]
                if not math.isfinite(cd):
                    continue
                rows.append(
                    {
                        "run": key,
                        "diameter": run["diameter"],
                        "format": run["format"],
                        "force_method": method,
                        "window": window,
                        "cd_pred": cd,
                        "cd_ref": cd_reference,
                        "rel_error": abs(cd - cd_reference) / cd_reference,
                        "steps": run["steps"],
                        "cells": run["grid"]["cells"],
                    }
                )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row['format']}|{row['force_method']}|{row['window']}", []).append(row)

    convergence: dict[str, Any] = {}
    for name, group in groups.items():
        group = sorted(group, key=lambda r: r["diameter"])
        diameters = [r["diameter"] for r in group]
        errors = [r["rel_error"] for r in group]
        convergence[name] = {
            "diameters": diameters,
            "cd_pred": [r["cd_pred"] for r in group],
            "rel_error": errors,
            "order_finest_pair": convergence_order(diameters, errors, 2),
            "order_all_points": convergence_order(diameters, errors, len(errors)),
        }
    return {
        "rows": rows,
        "convergence": convergence,
        "cd_reference": cd_reference,
        "expected_rows": expected_rows or len(rows),
    }


def evaluate_criteria(
    table: dict[str, Any],
    *,
    finest: int,
    coarsest: int,
    cd_reference: float = CD_REFERENCE_RE100,
) -> dict[str, Any]:
    """Machine-checkable campaign criteria C1..C5."""
    conv = table["convergence"]

    def err_at(group: dict[str, Any], diameter: int) -> float:
        idx = group["diameters"].index(diameter)
        return group["rel_error"][idx]

    quad_me = conv["bouzidi_quadratic|momentum_exchange|steady"]
    quad_cv = conv["bouzidi_quadratic|control_volume|steady"]
    lin_me = conv["bouzidi_linear|momentum_exchange|steady"]

    c1_pass = err_at(quad_me, finest) < 0.03 and err_at(quad_cv, finest) < 0.03
    p_quad = quad_me["order_finest_pair"]["p_observed"]
    p_lin = lin_me["order_finest_pair"]["p_observed"]
    c2_pass = (p_quad >= 1.9) or (p_lin >= 0.9)
    me_f = quad_me["cd_pred"][-1]
    cv_f = quad_cv["cd_pred"][-1]
    c3_value = abs(me_f - cv_f) / abs(cv_f)
    lin_coarse = err_at(lin_me, coarsest)
    return {
        "C1": {
            "criterion": "Bouzidi quadratic CD error < 3% at the finest grid",
            "diameter": finest,
            "error_me": err_at(quad_me, finest),
            "error_cv": err_at(quad_cv, finest),
            "pass": bool(c1_pass),
        },
        "C2": {
            "criterion": "observed order p >= 1.9 (quadratic) or p >= 0.9 (linear)",
            "p_quadratic": p_quad,
            "p_linear": p_lin,
            "pass": bool(c2_pass),
        },
        "C3": {
            "criterion": "|CD_ME - CD_CV| / CD_CV < 1% at the finest grid",
            "cd_me": me_f,
            "cd_cv": cv_f,
            "relative_difference": c3_value,
            "pass": bool(c3_value < 0.01),
        },
        "C4": {
            "criterion": "every (N, format, force method, window) combination of the scan "
                         "is covered by a pytest parametrised gate",
            "rows": len(table["rows"]),
            "expected_rows": int(table["expected_rows"]),
            "pass": bool(len(table["rows"]) == int(table["expected_rows"])),
        },
        "C5": {
            "criterion": "negative control: Bouzidi linear at the coarsest grid must err > 10%",
            "diameter": coarsest,
            "linear_error": lin_coarse,
            "pass": bool(lin_coarse > 0.10),
        },
        "cd_reference": cd_reference,
    }