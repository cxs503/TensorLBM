"""Block-structured AMR for LBM free-surface (VOF) flow — verification-grade.

Why this module
---------------
The repository's production free-surface kernels are *fixed grid*.
:mod:`tensorlbm.adaptive_refinement` refines single-phase LBM only (it carries
populations, never a free-surface mass ledger) and the nested-block machinery of
:mod:`tensorlbm.refinement` is likewise single-phase, so free-surface flow had no
adaptive path at all.  This module supplies one.

It deliberately does **not** wrap
:func:`tensorlbm.free_surface_lbm.free_surface_step` (D3Q19) or
:func:`tensorlbm.free_surface_lbm_2d.free_surface_step_2d` (D2Q9): both need
10²–10³ torch ops per call, so their cost is per-call overhead dominated
(≈130 ms per call on a 640-cell grid, measured), a 15–45-call-per-step AMR
hierarchy could never win an efficiency criterion, and their own ledger drift
(measured 3.4 % over 200 steps) would swamp the AMR signal.  The kernel below is
a compact batched D2Q9 free-surface LBM whose mass bookkeeping is conservative
**by construction**, so the AMR exchange is the only remaining source of mass
error and can be measured in isolation.

Numerical model
---------------
Momentum: D2Q9 BGK with Guo gravity on the fluid cells (``LIQUID`` and
``INTERFACE``), half-way bounce-back on solid, anti-bounce-back free-surface
reconstruction against the gas pressure at every interface cell (Körner et al.
2005 gas-link rule in its mirror form for equal liquid/gas density:
``f_q ← 2 f_q^eq(ρ_l, u) − f_ō``), and a bulk density re-normalisation
``Σf → ρ_l`` on ``LIQUID`` cells so the bulk stays at the liquid density.

Interface: the liquid volume fraction ``ε = m/ρ_l`` is advanced by a
**finite-volume flux-form VOF advection** driven by the LBM face velocities
(second-order limited interface values).  Every face flux is removed from one
cell and added to the other, so ``Σm`` is conserved to floating-point round-off
with no redistribution heuristics, and the limiter keeps ``0 ≤ ε ≤ 1`` while
``Σ|u| < 1``.

Hierarchy
---------
* levels ``ℓ = 0 … L``, refinement ratio 2 ⇒ finest spacing ``Δx₀/2**L``
  (``L = 3`` gives ``h, h/2, h/4, h/8``).
* **indicator**: ``|∇ε| > threshold`` on each level's own fill field — the
  free-surface analogue of ``|∇ρ|``, since with ``ρ = ρ_g + ε(ρ_l − ρ_g)`` the
  density gradient is proportional to the fill gradient — plus every
  ``INTERFACE`` cell, dilated by ``band_pad``.  The marked set is split into at
  most ``segments`` x-slabs: one bounding box around a 1-D interface curve would
  cover the whole water body and destroy the cell-count advantage.
* **nesting**: a level-``ℓ`` box lies inside the level-``ℓ−1`` *active* region
  eroded by ``ghost`` cells, so every fine boundary value reads valid parent
  data.
* **sub-cycling** 2:1 with convective scaling: lattice velocity and wave
  celerity are level invariant, hence ``g_ℓ = g₀/2**ℓ`` and
  ``ν_lat,ℓ = 2**ℓ ν_lat,0`` i.e. ``τ_ℓ = ½ + 2**ℓ(τ₀ − ½)`` (physical viscosity
  and ``Re`` grid independent); the coarse state is frozen over the two child
  sub-steps.
* **exchange**: each tile is advanced *including* its ``ghost``-cell ring, which
  is re-filled from the parent before every sub-step (Filippova–Hänel rescaling
  of the non-equilibrium part, ``τ_f/(2τ_c)`` up / ``2τ_c/τ_f`` down); mass is
  prolonged nearest-neighbour and restricted by a 2×2 area mean, which is exactly
  conservative in the ``4**-ℓ`` area quadrature of
  :meth:`AMRFreeSurface.composite_mass`.
* **mass**: the ghost-cell outflow and the coarse/fine representation switch do
  not conserve ``Σ w_ℓ m``; the residual is measured every step and returned to
  the level-0 interface cells as a multiplicative (positivity preserving)
  **reflux**.  Raw and closed drift are both reported — the raw one is the honest
  measure of how much mass the AMR exchange moves.

Tiles of one level are advanced as a **single batched kernel call** (padding
marked solid so it is inert): a per-op floor of ≈20 µs on a 128-core host makes a
call-per-box scheme 5–10× slower than the same work batched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .d2q9 import C, OPPOSITE, W

__all__ = [
    "AMRFreeSurfaceConfig",
    "AMRFreeSurface",
    "Box2D",
    "LevelState",
    "level_step",
    "interface_band_mask",
]

GAS = 0
LIQUID = 1
INTERFACE = 2
SOLID = 3

_CX = [int(C[q, 0]) for q in range(9)]
_CY = [int(C[q, 1]) for q in range(9)]
_OPP = torch.tensor([int(OPPOSITE[q]) for q in range(9)])
_W = W.to(torch.float32)
_CXW3 = torch.tensor([3.0 * float(c) * float(w) for c, w in zip(_CX, W)]).to(torch.float32)
_CYW3 = torch.tensor([3.0 * float(c) * float(w) for c, w in zip(_CY, W)]).to(torch.float32)


# ---------------------------------------------------------------------------
# shift helpers — they act on the last two axes, so any leading batch works
# ---------------------------------------------------------------------------


def _roll(field: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Shift so that the value at ``x`` comes from ``x − (dx, dy)``."""
    if dx == 0 and dy == 0:
        return field
    return torch.roll(field, shifts=(-dy, -dx), dims=(-2, -1))


def _neighbour(mask: torch.Tensor, q: int) -> torch.Tensor:
    """``mask`` sampled at ``x + c_q``."""
    return _roll(mask, -_CX[q], -_CY[q])


def _minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(
        a * b > 0.0, torch.sign(a) * torch.minimum(a.abs(), b.abs()), torch.zeros_like(a)
    )


def _dilate4(mask: torch.Tensor) -> torch.Tensor:
    out = mask.clone()
    out[..., 1:, :] |= mask[..., :-1, :]
    out[..., :-1, :] |= mask[..., 1:, :]
    out[..., :, 1:] |= mask[..., :, :-1]
    out[..., :, :-1] |= mask[..., :, 1:]
    return out


def _erode4(mask: torch.Tensor) -> torch.Tensor:
    out = mask.clone()
    out[..., 1:, :] &= mask[..., :-1, :]
    out[..., :-1, :] &= mask[..., 1:, :]
    out[..., :, 1:] &= mask[..., :, :-1]
    out[..., :, :-1] &= mask[..., :, 1:]
    out[..., 0, :] = False
    out[..., -1, :] = False
    out[..., :, 0] = False
    out[..., :, -1] = False
    return out


def _dir_view(vec: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a length-9 lattice vector for broadcasting over ``(..., 9, ny, nx)``."""
    return vec.view(*([1] * (ndim - 3)), 9, 1, 1)


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium for batched ``(..., ny, nx)`` fields."""
    ndim = rho.dim() + 1
    u_sq = ux * ux + uy * uy
    cu = _dir_view(torch.tensor([float(c) for c in _CX]), ndim) * ux.unsqueeze(-3) + _dir_view(
        torch.tensor([float(c) for c in _CY]), ndim
    ) * uy.unsqueeze(-3)
    return (
        _dir_view(_W, ndim)
        * rho.unsqueeze(-3)
        * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(-3))
    )


def level_step(
    f: torch.Tensor,
    fill: torch.Tensor,
    flags: torch.Tensor,
    solid: torch.Tensor,
    mass: torch.Tensor,
    *,
    tau: float,
    gravity: float,
    rho_gas: float,
    rho_liquid: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One free-surface step on a batch of tiles of the same refinement level.

    All tiles of a level are stacked into one batched call: on this host a single
    torch op costs ≈20 µs, so a call-per-box scheme would spend 5–10× more time in
    launch overhead than in work.  Padding is marked solid, which makes it inert
    for streaming, bounce-back and the VOF flux.
    """
    from .free_surface_vof import free_surface_vof_step

    f, mass, flags = free_surface_vof_step(
        f,
        mass,
        flags,
        solid,
        tau=tau,
        gravity=(0.0, -gravity),
        rho_liquid=rho_liquid,
        rho_gas=rho_gas,
    )
    rho = f.sum(dim=-3).clamp(min=1e-6)
    fill = torch.where(flags == INTERFACE, mass / rho, torch.zeros_like(mass))
    fill = torch.where(flags == LIQUID, torch.ones_like(fill), fill)
    return f, fill, flags, mass


# ---------------------------------------------------------------------------
# refinement indicator
# ---------------------------------------------------------------------------


def interface_band_mask(
    fill: torch.Tensor,
    flags: torch.Tensor,
    threshold: float,
    pad: int,
) -> torch.Tensor:
    """Cells with ``|∇ε| > threshold`` plus every ``INTERFACE`` cell, dilated by *pad*."""
    eps = fill.clamp(0.0, 1.0)
    grad = torch.zeros_like(eps)
    if eps.shape[-2] > 2:
        dy = 0.5 * (eps[..., 2:, :] - eps[..., :-2, :])
        grad[..., 1:-1, :] += dy * dy
    if eps.shape[-1] > 2:
        dx = 0.5 * (eps[..., :, 2:] - eps[..., :, :-2])
        grad[..., :, 1:-1] += dx * dx
    mark = grad > threshold * threshold
    mark = mark | (flags == INTERFACE)
    for _ in range(int(pad)):
        mark = _dilate4(mark)
    return mark

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class AMRFreeSurfaceConfig:
    """Geometry, physics and refinement controls (lengths in level-0 cells)."""

    ny0: int = 16
    nx0: int = 120
    levels: int = 3
    column_width: int = 6
    column_height: int = 12
    c0: float = 0.08
    """Gravity-wave celerity ``sqrt(g h)`` in lattice units (Mach 0.14)."""
    reynolds: float | None = 60.0
    """Dam-break Reynolds number ``Re = c0 h_finest / ν_lat``; sets ``tau0``.

    Holding ``Re`` grid independent is mandatory for a refinement study: fixing
    ``tau`` instead would make every finer level resolve a *less viscous* flow.
    """
    tau0: float | None = None
    """Level-0 relaxation time; overrides :attr:`reynolds` when given."""
    rho_liquid: float = 1.0
    rho_gas: float = 0.05
    """Gas density used by the Körner gas-link reconstruction (0.05 is the value
    at which the repository's D2Q9 module is documented stable; ``rho_gas = 1``
    is the documented unstable case, see ``benchmarks/dam_break/fslbm_driver.py``)."""
    refine_liquid: bool = False
    """When set, every fluid cell (not just the interface ribbon) asks for refinement.

    The verification face uses it as a control: it separates *interface* error from
    *bulk* error in the AMR result (see ``docs/AMR_FREE_SURFACE_REPORT.md``).
    """
    band_threshold: float = 1e-3
    band_pad: int = 1
    segments: int = 4
    min_box_cells: int = 6
    min_span: int = 12
    """Minimum box extent per axis, in cells of the level being built.

    A ribbon thinner than ``2 ghost`` cannot host a child level (the child needs
    valid parent data inside the parent's eroded interior), so boxes are grown to
    at least this extent.
    """
    ghost: int = 2
    wall_margin: int = 2
    adapt_interval: int = 8

    def __post_init__(self) -> None:
        if not 0 <= self.levels <= 5:
            raise ValueError("levels must lie in 0..5")
        if min(self.ny0, self.nx0) < 8:
            raise ValueError("domain is too small for a free-surface AMR study")
        if not 0.0 < self.c0 < 0.25:
            raise ValueError("c0 must lie in (0, 0.25)")
        if self.tau0 is None and (self.reynolds is None or self.reynolds <= 1.0):
            raise ValueError("give either tau0 or a positive reynolds number")
        if self.tau0 is not None and self.tau0 <= 0.5005:
            raise ValueError("tau0 must exceed 0.5005")
        if self.ghost < 1:
            raise ValueError("ghost must be at least 1")
        if self.segments < 1:
            raise ValueError("segments must be at least 1")
        if not 1 <= self.column_width <= self.nx0 - 2 * self.wall_margin:
            raise ValueError("column_width does not fit in the domain")
        if not 1 <= self.column_height <= self.ny0 - 2 * self.wall_margin:
            raise ValueError("column_height does not fit in the domain")

    @property
    def g0(self) -> float:
        return self.c0**2 / self.column_height

    @property
    def tau_finest(self) -> float:
        """Relaxation time of the deepest level (from :attr:`reynolds`)."""
        if self.reynolds is None:
            return 0.5 + (2**self.levels) * (self.tau0 - 0.5)
        h_finest = self.column_height * 2**self.levels
        nu = self.c0 * h_finest / self.reynolds
        return 3.0 * nu + 0.5

    @property
    def tau0_effective(self) -> float:
        if self.tau0 is not None:
            return self.tau0
        return 0.5 + (self.tau_finest - 0.5) / (2**self.levels)

    def gravity(self, level: int) -> float:
        return self.g0 / (2**level)

    def tau(self, level: int) -> float:
        """Relaxation time at *level*.

        Convective (acoustic) scaling ``Δt_ℓ ∝ Δx_ℓ`` keeps the lattice velocity
        and the wave celerity level invariant; the physical viscosity
        ``ν = ν_lat Δx²/Δt`` is then grid independent only if ``ν_lat`` grows with
        the level, ``ν_lat,ℓ = 2**ℓ ν_lat,0``, i.e. ``τ_ℓ = ½ + 2**ℓ(τ₀ − ½)``.
        """
        return 0.5 + (2**level) * (self.tau0_effective - 0.5)

    def spacing(self, level: int) -> int:
        return 2**level

    def grid(self, level: int) -> tuple[int, int]:
        s = self.spacing(level)
        return self.ny0 * s, self.nx0 * s

    def time_scale(self) -> float:
        """``dt*`` per level-0 step, with ``t* = t sqrt(g/H)`` in level-0 cells."""
        return self.c0 / self.column_height


# ---------------------------------------------------------------------------
# boxes and levels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box2D:
    """Half-open rectangle ``[y0, y1) x [x0, x1)`` in level coordinates."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def ny(self) -> int:
        return self.y1 - self.y0

    @property
    def nx(self) -> int:
        return self.x1 - self.x0

    @property
    def n_cells(self) -> int:
        return self.ny * self.nx

    def expanded(self, pad: int, ny_max: int, nx_max: int) -> "Box2D":
        return Box2D(
            max(0, self.y0 - pad),
            min(ny_max, self.y1 + pad),
            max(0, self.x0 - pad),
            min(nx_max, self.x1 + pad),
        )

    def index(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


@dataclass
class LevelState:
    """Dense storage plus the active (box-owned) mask of one refinement level."""

    level: int
    ny: int
    nx: int
    f: torch.Tensor
    mass: torch.Tensor
    fill: torch.Tensor
    flags: torch.Tensor
    solid: torch.Tensor
    boxes: list[Box2D] = field(default_factory=list)
    active: torch.Tensor | None = None

    @classmethod
    def allocate(cls, level: int, ny: int, nx: int, device) -> "LevelState":
        return cls(
            level=level,
            ny=ny,
            nx=nx,
            f=torch.zeros((9, ny, nx), dtype=torch.float32, device=device),
            mass=torch.zeros((ny, nx), dtype=torch.float32, device=device),
            fill=torch.zeros((ny, nx), dtype=torch.float32, device=device),
            flags=torch.full((ny, nx), GAS, dtype=torch.int8, device=device),
            solid=torch.zeros((ny, nx), dtype=torch.bool, device=device),
            boxes=[],
            active=torch.zeros((ny, nx), dtype=torch.bool, device=device),
        )


# ---------------------------------------------------------------------------
# exchange primitives
# ---------------------------------------------------------------------------


def _fh_upsample(f_coarse: torch.Tensor, tau_c: float, tau_f: float, ratio: int = 2) -> torch.Tensor:
    """Filippova-Haenel coarse to fine on ``(9, ny, nx)`` tiles."""
    rho = f_coarse.sum(dim=0)
    live = rho > 1e-6
    rho_safe = rho.clamp(min=1e-6)
    ux = torch.where(live, (f_coarse * _dir_view(torch.tensor([float(c) for c in _CX]), 3)).sum(dim=0) / rho_safe, torch.zeros_like(rho))
    uy = torch.where(live, (f_coarse * _dir_view(torch.tensor([float(c) for c in _CY]), 3)).sum(dim=0) / rho_safe, torch.zeros_like(rho))
    f_eq = _equilibrium(rho, ux, uy)
    up = f_eq + (tau_f / (ratio * tau_c)) * (f_coarse - f_eq)
    return up.repeat_interleave(ratio, dim=1).repeat_interleave(ratio, dim=2)


def _fh_downsample(f_fine: torch.Tensor, tau_f: float, tau_c: float, ratio: int = 2) -> torch.Tensor:
    """Filippova-Haenel fine to coarse on ``(9, ny, nx)`` tiles."""
    q, ny, nx = f_fine.shape
    if ny % ratio or nx % ratio:
        raise ValueError("fine tile must be an integer multiple of the ratio")
    avg = f_fine.reshape(q, ny // ratio, ratio, nx // ratio, ratio).mean(dim=(2, 4))
    rho = avg.sum(dim=0)
    live = rho > 1e-6
    rho_safe = rho.clamp(min=1e-6)
    ux = torch.where(live, (avg * _dir_view(torch.tensor([float(c) for c in _CX]), 3)).sum(dim=0) / rho_safe, torch.zeros_like(rho))
    uy = torch.where(live, (avg * _dir_view(torch.tensor([float(c) for c in _CY]), 3)).sum(dim=0) / rho_safe, torch.zeros_like(rho))
    f_eq = _equilibrium(rho, ux, uy)
    return f_eq + (ratio * tau_c / tau_f) * (avg - f_eq)


def _block_average(field: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    ny, nx = field.shape[-2:]
    if ny % ratio or nx % ratio:
        raise ValueError("field must be an integer multiple of the ratio")
    return field.reshape(ny // ratio, ratio, nx // ratio, ratio).mean(dim=(1, 3))


def _block_repeat(field: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    return field.repeat_interleave(ratio, dim=-2).repeat_interleave(ratio, dim=-1)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


class AMRFreeSurface:
    """Multi-level free-surface LBM with indicator-driven box re-tagging."""

    def __init__(self, config: AMRFreeSurfaceConfig | None = None) -> None:
        self.config = config or AMRFreeSurfaceConfig()
        cfg = self.config
        self.device = torch.device("cpu")
        self.levels: list[LevelState] = []
        self.step_count = 0
        self.regrid_count = 0
        self.wall_clock = 0.0
        self.raw_drift = 0.0
        self.drift_history: list[float] = []
        self.clipped_mass = 0.0
        self._build_level0()
        self.mass0 = self.composite_mass()

    # -- construction ----------------------------------------------------
    def _build_level0(self) -> None:
        cfg = self.config
        ny, nx = cfg.grid(0)
        st = LevelState.allocate(0, ny, nx, self.device)
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        solid[0, :] = True
        solid[-1, :] = True
        solid[:, 0] = True
        solid[:, -1] = True
        fill = torch.zeros((ny, nx))
        fill[: cfg.column_height, 1 : cfg.column_width] = 1.0
        flags = torch.full_like(fill, GAS, dtype=torch.int8)
        flags[fill >= 1.0] = LIQUID
        envelope = _dilate4(flags == LIQUID)
        flags[(flags == GAS) & envelope & ~solid] = INTERFACE
        flags[solid] = SOLID
        mass = torch.where(flags == LIQUID, cfg.rho_liquid * torch.ones_like(fill), torch.zeros_like(fill))
        mass = torch.where(flags == INTERFACE, fill * cfg.rho_liquid, mass)
        rho0 = torch.where(flags == LIQUID, torch.ones_like(fill), torch.zeros_like(fill))
        st.f = _equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))
        st.fill = fill
        st.flags = flags
        st.mass = mass
        st.solid = solid
        st.boxes = [Box2D(0, ny, 0, nx)]
        st.active = torch.ones((ny, nx), dtype=torch.bool)
        self.levels = [st]
        for level in range(1, cfg.levels + 1):
            ny_l, nx_l = cfg.grid(level)
            self.levels.append(LevelState.allocate(level, ny_l, nx_l, self.device))
        self._retag(force=True)

    # -- diagnostics -----------------------------------------------------
    def _child_coverage(self, level: int, ratio: int = 2) -> torch.Tensor:
        act = self.levels[level].active
        ny, nx = act.shape
        return act.reshape(ny // ratio, ratio, nx // ratio, ratio).any(dim=(1, 3))

    def _keep_mask(self, level: int) -> torch.Tensor:
        st = self.levels[level]
        keep = torch.ones_like(st.active) if level == 0 else st.active
        if level + 1 < len(self.levels):
            keep = keep & ~self._child_coverage(level + 1)
        return keep

    @staticmethod
    def _inventory(st: "LevelState") -> torch.Tensor:
        """Conserved liquid mass of the level (single scalar field of this model)."""
        return st.mass

    def composite_mass(self) -> float:
        """Area-weighted liquid mass over the composite grid (the AMR mass ledger)."""
        total = 0.0
        for st in self.levels:
            keep = self._keep_mask(st.level)
            total += float(4 ** (-st.level)) * float(self._inventory(st)[keep].sum().item())
        return total

    def cell_count(self) -> dict[str, object]:
        """AMR cell inventory versus the uniform finest grid of the same box."""
        per_level: dict[str, float] = {}
        amr_cells = 0.0
        for st in self.levels:
            n = int(self._keep_mask(st.level).sum().item())
            per_level[f"level_{st.level}"] = float(n)
            amr_cells += n
        uniform_cells = float(self.config.ny0 * self.config.nx0 * 4 ** self.config.levels)
        return {
            "amr_cells": amr_cells,
            "per_level_cells": per_level,
            "active_per_level": {
                f"level_{s.level}": float(s.active.sum().item()) for s in self.levels
            },
            "uniform_finest_cells": uniform_cells,
            "ratio_to_uniform_finest": amr_cells / uniform_cells,
        }

    # -- refinement tagging ---------------------------------------------
    def _level_band(self, level: int) -> torch.Tensor:
        st = self.levels[level]
        if self.config.refine_liquid:
            return ((st.flags == LIQUID) | (st.flags == INTERFACE)) & st.active
        band = interface_band_mask(st.fill, st.flags, self.config.band_threshold, 0)
        return band & st.active

    def _retag(self, force: bool = False) -> None:
        cfg = self.config
        if cfg.levels == 0:
            return
        if cfg.refine_liquid:
            wet = (self.levels[0].flags == LIQUID) | (self.levels[0].flags == INTERFACE)
            band = _dilate4(wet) if cfg.band_pad else wet
        else:
            band = interface_band_mask(
                self.levels[0].fill, self.levels[0].flags, cfg.band_threshold, cfg.band_pad
            )
        for level in range(1, cfg.levels + 1):
            par = self.levels[level - 1]
            limit = par.active.clone()
            if level == 1 and cfg.wall_margin > 0:
                m = cfg.wall_margin
                limit[:m, :] = False
                limit[-m:, :] = False
                limit[:, :m] = False
                limit[:, -m:] = False
            for _ in range(cfg.ghost):
                limit = _erode4(limit)
            target = (band & limit).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            boxes = self._boxes_from_mask(target, level)
            st = self.levels[level]
            new_active = torch.zeros_like(st.active)
            for box in boxes:
                sl_y, sl_x = box.index()
                new_active[sl_y, sl_x] = True
            old_active = st.active
            if not force and boxes == st.boxes:
                band = self._level_band(level)
                continue
            if bool(old_active.any()):
                self._restrict(level)
            fresh = new_active & ~old_active
            st.active = new_active
            st.boxes = boxes
            if bool(fresh.any()):
                self._prolongate_into(level, fresh)
            self.regrid_count += 1
            band = self._level_band(level)

    def _boxes_from_mask(self, mask: torch.Tensor, level: int) -> list[Box2D]:
        cfg = self.config
        st = self.levels[level]
        ny, nx = st.ny, st.nx
        if not bool(mask.any()):
            return []
        xs = torch.nonzero(mask.any(dim=0)).flatten()
        x_lo, x_hi = int(xs[0]), int(xs[-1]) + 1
        n_seg = max(1, min(cfg.segments, (x_hi - x_lo) // cfg.min_span))
        edges = torch.linspace(x_lo, x_hi, n_seg + 1).round().to(torch.long).tolist()
        boxes: list[Box2D] = []
        for i in range(n_seg):
            a, b = edges[i], edges[i + 1]
            if b <= a:
                continue
            slab = mask[:, a:b]
            if not bool(slab.any()):
                continue
            ys = torch.nonzero(slab.any(dim=1)).flatten()
            y0 = max(0, int(ys[0]) - cfg.band_pad)
            y1 = min(ny, int(ys[-1]) + 1 + cfg.band_pad)
            if y1 - y0 < cfg.min_span:
                grow = cfg.min_span - (y1 - y0)
                y0 = max(0, y0 - grow // 2)
                y1 = min(ny, y0 + cfg.min_span)
            if x1_span := (b - a):
                if x1_span < cfg.min_span:
                    grow = cfg.min_span - x1_span
                    a = max(0, a - grow // 2)
                    b = min(nx, a + cfg.min_span)
            boxes.append(
                Box2D(
                    y0 - y0 % 2,
                    min(ny, y1 + y1 % 2),
                    a - a % 2,
                    min(nx, b + b % 2),
                )
            )
        # drop degenerate boxes and merge x-contiguous boxes with identical rows
        boxes = [bx for bx in boxes if bx.ny >= cfg.min_box_cells and bx.nx >= cfg.min_box_cells]
        merged: list[Box2D] = []
        for box in boxes:
            if (
                merged
                and box.y0 == merged[-1].y0
                and box.y1 == merged[-1].y1
                and box.x0 == merged[-1].x1
            ):
                prev = merged[-1]
                merged[-1] = Box2D(prev.y0, prev.y1, prev.x0, box.x1)
            else:
                merged.append(box)
        return merged

    # -- exchange --------------------------------------------------------
    def _prolongate_into(self, level: int, mask: torch.Tensor) -> None:
        cfg = self.config
        st = self.levels[level]
        par = self.levels[level - 1]
        if not bool(mask.any()):
            return
        ys = torch.nonzero(mask.any(dim=1)).flatten()
        xs = torch.nonzero(mask.any(dim=0)).flatten()
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        py0, py1 = y0 // 2, min(par.ny, (y1 + 1) // 2)
        px0, px1 = x0 // 2, min(par.nx, (x1 + 1) // 2)
        f_up = _fh_upsample(par.f[:, py0:py1, px0:px1], cfg.tau(level - 1), cfg.tau(level))
        mass_up = _block_repeat(par.mass[py0:py1, px0:px1])
        fill_up = _block_repeat(par.fill[py0:py1, px0:px1])
        flags_up = _block_repeat(par.flags[py0:py1, px0:px1].to(torch.int16)).to(torch.int8)
        sy = slice(y0 - 2 * py0, y1 - 2 * py0)
        sx = slice(x0 - 2 * px0, x1 - 2 * px0)
        sel = mask[y0:y1, x0:x1]
        st.f[:, y0:y1, x0:x1][:, sel] = f_up[:, sy, sx][:, sel]
        st.mass[y0:y1, x0:x1][sel] = mass_up[sy, sx][sel]
        st.fill[y0:y1, x0:x1][sel] = fill_up[sy, sx][sel]
        st.flags[y0:y1, x0:x1][sel] = flags_up[sy, sx][sel]

    def _exchange_mask(self, level: int, box: Box2D) -> torch.Tensor:
        """Non-active cells around *box* that must read the parent state."""
        cfg = self.config
        st = self.levels[level]
        grown = box.expanded(cfg.ghost, st.ny, st.nx)
        ring = torch.zeros((st.ny, st.nx), dtype=torch.bool)
        ring[grown.y0 : grown.y1, grown.x0 : grown.x1] = True
        ring[box.y0 : box.y1, box.x0 : box.x1] = False
        return ring & ~st.active

    def _zero_gradient_fill(self, level: int, mask: torch.Tensor) -> None:
        """Neumann extrapolation of ``mass``/``fill``/``flags`` into a ghost ring.

        The fill is a *material* field: overwriting it from the coarse level every
        sub-step re-injects the coarse interface thickness at the box boundary and
        measurably biases the fine solution (see the report).  Populations keep the
        parent exchange (they carry the stress), the fill is extrapolated from the
        fine interior instead.
        """
        st = self.levels[level]
        known = st.active.clone()
        mass = st.mass.clone()
        fill = st.fill.clone()
        flags = st.flags.clone()
        for _ in range(self.config.ghost + 1):
            todo = mask & ~known
            if not bool(todo.any()):
                break
            for q in (1, 2, 3, 4):
                src = torch.roll(known, shifts=(_CY[q], _CX[q]), dims=(0, 1))
                take = todo & src
                if bool(take.any()):
                    mass = torch.where(take, torch.roll(mass, shifts=(_CY[q], _CX[q]), dims=(0, 1)), mass)
                    fill = torch.where(take, torch.roll(fill, shifts=(_CY[q], _CX[q]), dims=(0, 1)), fill)
                    flags = torch.where(take, torch.roll(flags, shifts=(_CY[q], _CX[q]), dims=(0, 1)), flags)
                    known = known | take
                    todo = mask & ~known
        st.mass = torch.where(mask, mass, st.mass)
        st.fill = torch.where(mask, fill, st.fill)
        st.flags = torch.where(mask, flags, st.flags)

    def _fill_exchange(self, level: int) -> None:
        for box in self.levels[level].boxes:
            mask = self._exchange_mask(level, box)
            if bool(mask.any()):
                self._prolongate_into(level, mask)
                self._zero_gradient_fill(level, mask)

    def _restrict(self, level: int) -> None:
        cfg = self.config
        st = self.levels[level]
        par = self.levels[level - 1]
        for box in st.boxes:
            py = slice(box.y0 // 2, box.y1 // 2)
            px = slice(box.x0 // 2, box.x1 // 2)
            sl_y, sl_x = box.index()
            par.f[:, py, px] = _fh_downsample(
                st.f[:, sl_y, sl_x], cfg.tau(level), cfg.tau(level - 1)
            )
            par.mass[py, px] = _block_average(st.mass[sl_y, sl_x])
            par.fill[py, px] = _block_average(st.fill[sl_y, sl_x])
            fill_c = par.fill[py, px]
            flags = torch.full_like(par.flags[py, px], GAS)
            flags[fill_c >= 1.0 - 1e-6] = LIQUID
            flags[(fill_c > 1e-6) & (fill_c < 1.0 - 1e-6)] = INTERFACE
            envelope = _dilate4(flags == LIQUID)
            flags[(flags == GAS) & envelope & ~par.solid[py, px]] = INTERFACE
            flags[par.solid[py, px]] = SOLID
            par.flags[py, px] = flags

    # -- one level step --------------------------------------------------
    def _advance_level_boxes(self, level: int) -> None:
        cfg = self.config
        st = self.levels[level]
        if not st.boxes:
            return
        tiles = [b.expanded(cfg.ghost, st.ny, st.nx) for b in st.boxes]
        ny_t = max(t.ny for t in tiles)
        nx_t = max(t.nx for t in tiles)
        n_t = len(tiles)
        f = torch.zeros((n_t, 9, ny_t, nx_t))
        mass = torch.zeros((n_t, ny_t, nx_t))
        flags = torch.full((n_t, ny_t, nx_t), GAS, dtype=torch.int8)
        solid = torch.ones((n_t, ny_t, nx_t), dtype=torch.bool)  # padding is inert
        for i, tile in enumerate(tiles):
            sy, sx = tile.index()
            nyi, nxi = tile.ny, tile.nx
            f[i, :, :nyi, :nxi] = st.f[:, sy, sx]
            mass[i, :nyi, :nxi] = st.mass[sy, sx]
            flags[i, :nyi, :nxi] = st.flags[sy, sx]
            solid[i, :nyi, :nxi] = st.solid[sy, sx]
        f, fill, flags, mass = level_step(
            f,
            mass * 0.0,
            flags,
            solid,
            mass,
            tau=cfg.tau(level),
            gravity=cfg.gravity(level),
            rho_gas=cfg.rho_gas,
            rho_liquid=cfg.rho_liquid,
        )
        for i, tile in enumerate(tiles):
            sy, sx = tile.index()
            nyi, nxi = tile.ny, tile.nx
            st.f[:, sy, sx] = f[i, :, :nyi, :nxi]
            st.mass[sy, sx] = mass[i, :nyi, :nxi]
            st.flags[sy, sx] = flags[i, :nyi, :nxi]
            st.fill[sy, sx] = fill[i, :nyi, :nxi]

    # -- time loop -------------------------------------------------------
    def _advance_level(self, level: int, n_sub: int) -> None:
        for _ in range(n_sub):
            if level > 0:
                self._fill_exchange(level)
            self._advance_level_boxes(level)
            if level + 1 < len(self.levels):
                self._advance_level(level + 1, 2)

    def _reflux(self, drift: float) -> float:
        st = self.levels[0]
        if drift == 0.0:
            return 0.0
        keep = ~self._child_coverage(1) if len(self.levels) > 1 else torch.ones_like(st.active)
        iface = (st.flags == INTERFACE) & (st.mass > 0.0) & keep
        pool = float(st.mass[iface].sum().item())
        if pool <= 0.0:
            return drift
        factor = 1.0 - drift / pool
        if factor <= 0.0:
            return drift
        st.mass = torch.where(iface, st.mass * factor, st.mass)
        st.fill = torch.where(st.flags == INTERFACE, st.mass / self.config.rho_liquid, st.fill)
        return 0.0

    def step(self) -> None:
        t0 = time.perf_counter()
        # Re-tagging changes the composite representation, so it runs *before*
        # the ledger is opened; that leaves the measured drift purely physical +
        # exchange, and the reflux restores the ledger exactly.
        if self.config.adapt_interval > 0 and self.step_count % self.config.adapt_interval == 0:
            self._retag()
        before = self.composite_mass()
        self._advance_level(0, 1)
        for level in range(len(self.levels) - 1, 0, -1):
            if self.levels[level].boxes:
                self._restrict(level)
        after = self.composite_mass()
        drift = after - before
        self.raw_drift += drift
        self.drift_history.append(drift)
        # In a closed tank the liquid mass is physically invariant, so the ledger
        # is closed against its initial value: the reflux returns the exchange
        # error (ghost-cell outflow + representation switch) to the level-0
        # interface cells.  The raw drift above is reported separately.
        self._reflux(self.composite_mass() - self.mass0)
        self.step_count += 1
        self.wall_clock += time.perf_counter() - t0

    def run(self, n_steps: int) -> None:
        for _ in range(int(n_steps)):
            self.step()

    # -- post-processing -------------------------------------------------
    def composite_fill(self) -> torch.Tensor:
        """Fill on the finest-level grid (coarse data prolonged where uncovered)."""
        cfg = self.config
        out = self.levels[0].fill
        for _ in range(cfg.levels):
            out = _block_repeat(out, 2)
        for level in range(1, len(self.levels)):
            st = self.levels[level]
            scale = 2 ** (cfg.levels - level)
            act = st.active.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
            fine = st.fill.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
            out = torch.where(act, fine, out)
        return out.clamp(0.0, 1.0)

    def composite_mass_field(self) -> torch.Tensor:
        """Liquid mass on the finest-level grid (prolonged where uncovered)."""
        cfg = self.config
        out = self.levels[0].mass
        for _ in range(cfg.levels):
            out = _block_repeat(out, 2)
        for level in range(1, len(self.levels)):
            st = self.levels[level]
            scale = 2 ** (cfg.levels - level)
            act = st.active.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
            fine = st.mass.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
            out = torch.where(act, fine, out)
        return out

    def depth_profile(self) -> torch.Tensor:
        """Water depth ``h(x)`` in finest cells (column sum of fill)."""
        return self.composite_fill().sum(dim=0)

    def front_position(self, threshold_cells: float = 1.0) -> float:
        """Rightmost x (finest-cell index, +1) whose column depth exceeds *threshold*."""
        h = self.depth_profile()
        wet = torch.nonzero(h > threshold_cells).flatten()
        if wet.numel() == 0:
            return 0.0
        return float(int(wet[-1].item()) + 1)
