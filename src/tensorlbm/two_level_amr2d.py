"""Two-level (ratio 2) static local refinement for the 2-D D2Q9 LBM.

The module implements a single strictly-interior refined block of refinement
ratio ``r = 2`` coupled to the coarse grid with a Filippova--Haenel /
Lagrava interface operator:

* **coarse -> fine (prolongation).**  Coarse populations are sampled at the
  fine ghost-cell centres with bilinear interpolation (optionally in time
  between ``t`` and ``t + dt``) and split into equilibrium and
  non-equilibrium parts.  Under the convective scaling used here
  (``dx, dt -> dx/r, dt/r``, identical lattice velocity ``u``) the
  Chapman--Enskog amplitude of the non-equilibrium part obeys

      f^neq_fine / f^neq_coarse = tau_fine / (r * tau_coarse),

  because in lattice units ``f^(1) ~ tau * dt * d(u)/dx``.

* **fine -> coarse (restriction).**  Population-wise arithmetic mean over the
  ``r x r`` children (volume weighting ``r**-2``) followed by the reciprocal
  non-equilibrium rescaling ``r * tau_coarse / tau_fine``, so that
  prolongation and restriction are exact inverses on the non-equilibrium
  part.

* **reflux.**  Mass uses *target* bookkeeping (the volume-weighted physical
  mass before and after a composite step is compared and the mismatch is
  redistributed over the coarse cells of the one-cell ring surrounding the
  patch).  Momentum cannot be handled that way in the presence of walls,
  solid obstacles or body forces, so the spurious momentum defect is computed
  from an explicit crossing-link ledger of *both* levels and redistributed via
  a mass-neutral velocity offset.

Scaling conventions follow :mod:`tensorlbm.static_block_amr` (3-D) and
:mod:`tensorlbm.amr_population_transfer`.

References
----------
Filippova O., Haenel D. (1998) Grid refinement for lattice-BGK models,
    J. Comput. Phys. 147, 219-228.
Lagrava D., Malaspinas O., Latt J., Chopard B. (2012) Advances in multi-domain
    lattice Boltzmann grid refinement, J. Comput. Phys. 231, 4808-4822.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .d2q9 import _c_on, _w_for, equilibrium, macroscopic

__all__ = [
    "TwoLevelConfig",
    "obstacle_force",
    "UniformD2Q9",
    "collide_bgk",
    "stream_mask",
    "TwoLevelD2Q9",
    "convective_refined_tau",
    "neq_scale_factor",
    "cylinder_solid_mask",
]


# ---------------------------------------------------------------------------
# Lattice bookkeeping
# ---------------------------------------------------------------------------

_OPP_CACHE: dict[tuple[int, ...], list[int]] = {}


def _opposite_indices(c: torch.Tensor) -> list[int]:
    """Index of the population travelling in the opposite direction."""
    key = tuple(int(v) for v in c.reshape(-1))
    cached = _OPP_CACHE.get(key)
    if cached is not None:
        return cached
    table = []
    for i in range(c.shape[0]):
        match = [
            j
            for j in range(c.shape[0])
            if int(c[j, 0]) == -int(c[i, 0]) and int(c[j, 1]) == -int(c[i, 1])
        ]
        if len(match) != 1:
            raise RuntimeError("lattice has no unique opposite direction")
        table.append(match[0])
    _OPP_CACHE[key] = table
    return table


def convective_refined_tau(tau_coarse: float, ratio: int = 2) -> float:
    """Relaxation time preserving the physical viscosity under ``dt, dx -> /r``.

    With convective scaling the lattice viscosity grows by ``ratio`` on the
    finer level, hence ``tau_f - 1/2 = ratio (tau_c - 1/2)``.
    """
    if not tau_coarse > 0.5:
        raise ValueError("tau_coarse must be greater than 0.5")
    if int(ratio) < 1:
        raise ValueError("ratio must be a positive integer")
    return 0.5 + ratio * (tau_coarse - 0.5)


def neq_scale_factor(tau_source: float, tau_target: float, ratio: float, mode: str) -> float:
    """Non-equilibrium rescaling factor of the transfer operators.

    ``mode="chapman_enskog"`` is the physically consistent operator; the other
    modes are deliberate defects exercised by the negative controls of the
    capability gate.
    """
    if mode == "chapman_enskog":
        return tau_target / (ratio * tau_source)
    if mode == "none":
        return 1.0
    if mode == "tau_only":
        return tau_target / tau_source
    if mode == "inverse":
        return ratio * tau_source / tau_target
    raise ValueError(f"unknown neq scale mode {mode!r}")


# ---------------------------------------------------------------------------
# Array helper
# ---------------------------------------------------------------------------


def _sample(a: torch.Tensor, sy: int, sx: int, fill: float | None) -> torch.Tensor:
    """``out[..., j, i] = a[..., j - sy, i - sx]`` (wrapping or ``fill``)."""
    if sy == 0 and sx == 0:
        return a
    b = torch.roll(a, shifts=(sy, sx), dims=(-2, -1))
    if fill is None:
        return b
    h, w = a.shape[-2], a.shape[-1]
    if sy > 0:
        b[..., :sy, :] = fill
    elif sy < 0:
        b[..., sy:, :] = fill
    if sx > 0:
        b[..., :, :sx] = fill
    elif sx < 0:
        b[..., :, sx:] = fill
    return b


def _sample_xy(
    a: torch.Tensor,
    sy: int,
    sx: int,
    fill_y: float | None,
    fill_x: float | None,
) -> torch.Tensor:
    """Per-axis sampling: ``out[j, i] = a[j - sy, i - sx]``.

    ``fill_y``/``fill_x`` are ``None`` for a periodic axis (wrap) and a scalar
    otherwise (out-of-domain value).  Handling the axes independently is
    essential for mixed periodicity: a diagonal population leaving a
    non-periodic direction must still wrap in a periodic direction.
    """
    if sy == 0 and sx == 0:
        return a
    b = torch.roll(a, shifts=(sy, sx), dims=(-2, -1))
    h, w = a.shape[-2], a.shape[-1]
    if sy != 0 and fill_y is not None:
        if sy > 0:
            b[..., :sy, :] = fill_y
        else:
            b[..., sy:, :] = fill_y
    if sx != 0 and fill_x is not None:
        if sx > 0:
            b[..., :, :sx] = fill_x
        else:
            b[..., :, sx:] = fill_x
    return b


def collide_bgk(
    f: torch.Tensor,
    tau: float,
    force: tuple[float, float],
    c: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """BGK collision with a simple (Ladd) body-force term.

    The force density ``force`` (momentum per cell per step) is injected
    through ``+ 3 w_i (c_i . F)``, which carries exactly the momentum ``F``
    and zero mass per step.  This keeps the momentum balance of the composite
    grid auditable; the residual ``O(F)`` forcing error is negligible for the
    low-Mach regimes used by the capability gate.
    """
    rho, ux, uy = macroscopic(f)
    fx, fy = (float(force[0]), float(force[1]))
    eq = equilibrium(rho, ux, uy)
    out = f - (f - eq) / tau
    if fx == 0.0 and fy == 0.0:
        return out
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    w_b = w.view(9, 1, 1)
    return out + 3.0 * w_b * (cx * fx + cy * fy)


def stream_mask(
    f_post: torch.Tensor,
    solid: torch.Tensor,
    periodic: tuple[bool, bool],
    c: torch.Tensor,
    opp: list[int],
) -> torch.Tensor:
    """Streaming with half-way bounce-back on ``solid`` and per-axis periodicity."""
    f_new = torch.zeros_like(f_post)
    solid_f = solid.to(f_post.dtype)
    for i in range(9):
        dx, dy = int(c[i, 0]), int(c[i, 1])
        fill_y = None if (dy == 0 or periodic[1]) else 1.0
        fill_x = None if (dx == 0 or periodic[0]) else 1.0
        nb_solid = _sample_xy(solid_f, dy, dx, fill_y, fill_x)
        pop_y = None if fill_y is None else 0.0
        pop_x = None if fill_x is None else 0.0
        src = _sample_xy(f_post[i], dy, dx, pop_y, pop_x)
        f_new[i] = torch.where(nb_solid > 0.5, f_post[opp[i]], src)
    f_new[:, solid] = 0.0
    return f_new


@dataclass(frozen=True)
class TwoLevelConfig:
    """Static description of one 2:1 refined block inside a coarse grid."""

    shape: tuple[int, int]  # (ny, nx) coarse grid
    box: tuple[int, int, int, int]  # (x0, x1, y0, y1) in coarse indices
    tau_coarse: float = 0.6
    ratio: int = 2
    force: tuple[float, float] = (0.0, 0.0)
    periodic: tuple[bool, bool] = (True, True)  # (x, y)
    reflux: bool = True
    momentum_reflux: bool = True
    neq_scale: str = "chapman_enskog"
    restrict_spatial_ratio: float | None = None
    ghost_neq_damping: float = 1.0
    ghost_interp: str = "bilinear"
    time_interp: bool = True
    reflux_iterations: int = 2
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        ny, nx = self.shape
        x0, x1, y0, y1 = self.box
        if not (1 <= x0 < x1 <= nx - 1 and 1 <= y0 < y1 <= ny - 1):
            raise ValueError("refined block must be strictly interior with a one-cell margin")
        convective_refined_tau(self.tau_coarse, self.ratio)
        neq_scale_factor(self.tau_coarse, self.tau_coarse, float(self.ratio), self.neq_scale)
        if self.ghost_interp not in ("bilinear", "injection"):
            raise ValueError("ghost_interp must be bilinear or injection")

    @property
    def tau_fine(self) -> float:
        return convective_refined_tau(self.tau_coarse, self.ratio)

    @property
    def interior_coarse_slice(self) -> tuple[slice, slice]:
        x0, x1, y0, y1 = self.box
        return (slice(y0, y1), slice(x0, x1))

    @property
    def fine_interior_shape(self) -> tuple[int, int]:
        x0, x1, y0, y1 = self.box
        r = self.ratio
        return ((y1 - y0) * r, (x1 - x0) * r)


class TwoLevelD2Q9:
    """Coarse D2Q9 grid with one static 2:1 refined block."""

    def __init__(
        self,
        config: TwoLevelConfig,
        solid_coarse: torch.Tensor | None = None,
        solid_fine: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        ny, nx = config.shape
        self.dtype = config.dtype
        c = _c_on(torch.device("cpu")).to(torch.float64)
        self.c = c
        self.w = _w_for(torch.float64, torch.device("cpu"))
        self.opp = _opposite_indices(c)
        self._int_c = [(int(c[i, 1]), int(c[i, 0])) for i in range(c.shape[0])]  # (dy, dx)

        if solid_coarse is None:
            solid_coarse = torch.zeros((ny, nx), dtype=torch.bool)
        self.solid_coarse = solid_coarse.to(torch.bool)
        fh, fw = self.config.fine_interior_shape
        if solid_fine is None:
            x0, x1, y0, y1 = config.box
            sub = self.solid_coarse[y0:y1, x0:x1]
            r = config.ratio
            sub_f = sub.repeat_interleave(r, dim=0).repeat_interleave(r, dim=1)
            solid_fine = torch.zeros((fh + 2, fw + 2), dtype=torch.bool)
            solid_fine[1:-1, 1:-1] = sub_f
        self.solid_fine = solid_fine.to(torch.bool)
        if tuple(self.solid_fine.shape) != (fh + 2, fw + 2):
            raise ValueError(
                f"solid_fine must have shape {(fh + 2, fw + 2)}, got {tuple(self.solid_fine.shape)}"
            )
        if (
            self.solid_fine[0].any()
            or self.solid_fine[-1].any()
            or self.solid_fine[:, 0].any()
            or self.solid_fine[:, -1].any()
        ):
            raise ValueError("the fine ghost ring must be fluid")

        self.coarse = torch.zeros((9, ny, nx), dtype=self.dtype)
        self.fine = torch.zeros((9, fh + 2, fw + 2), dtype=self.dtype)

        self._interior_mask = torch.zeros((ny, nx), dtype=torch.bool)
        ys, xs = config.interior_coarse_slice
        self._interior_mask[ys, xs] = True
        self._band_mask = self._build_band_mask()
        self._fine_ghost_mask = torch.zeros((fh + 2, fw + 2), dtype=torch.bool)
        self._fine_ghost_mask[0, :] = True
        self._fine_ghost_mask[-1, :] = True
        self._fine_ghost_mask[:, 0] = True
        self._fine_ghost_mask[:, -1] = True
        self._weights = self._build_ghost_weights()

        self._coarse_links = self._build_crossing_masks(
            self._interior_mask, ghost_mask=None
        )
        self._fine_links = self._build_crossing_masks(
            ~self._fine_ghost_mask, ghost_mask=self._fine_ghost_mask
        )
        self._c_weights = [
            (self.c[:, 0].view(9, 1, 1), self.c[:, 1].view(9, 1, 1))
        ]
        self.time = 0
        self.ledger: dict[str, float] = {}
        self.crossing_ledger: dict[str, float] = {}

    # -- construction helpers -----------------------------------------------

    def _build_band_mask(self) -> torch.Tensor:
        x0, x1, y0, y1 = self.config.box
        band = torch.zeros_like(self._interior_mask)
        band[max(y0 - 1, 0) : y1 + 1, max(x0 - 1, 0) : x1 + 1] = True
        band &= ~self._interior_mask
        band &= ~self.solid_coarse
        return band

    def _build_ghost_weights(self) -> tuple[torch.Tensor, ...]:
        """Bilinear coarse stencil for every padded fine cell.

        A padded fine cell ``(jj, ii)`` has its centre at coarse-space
        coordinate ``(x0 + (ii - 0.5)/r, y0 + (jj - 0.5)/r)``; coarse cell
        ``k`` has its centre at ``k + 0.5``.
        """
        x0, _, y0, _ = self.config.box
        r = self.config.ratio
        fh, fw = self.config.fine_interior_shape
        ii = torch.arange(fw + 2, dtype=torch.float64)
        jj = torch.arange(fh + 2, dtype=torch.float64)
        kx = x0 + (ii - 0.5) / r - 0.5
        ky = y0 + (jj - 0.5) / r - 0.5
        kx0 = torch.floor(kx).to(torch.long)
        ky0 = torch.floor(ky).to(torch.long)
        wx = (kx - kx0.to(torch.float64)).clamp(0.0, 1.0)
        wy = (ky - ky0.to(torch.float64)).clamp(0.0, 1.0)
        ny, nx = self.config.shape
        kx0 = kx0.clamp(0, nx - 1)
        ky0 = ky0.clamp(0, ny - 1)
        kx1 = (kx0 + 1).clamp(0, nx - 1)
        ky1 = (ky0 + 1).clamp(0, ny - 1)
        return (ky0, ky1, kx0, kx1, wy, wx)

    @staticmethod
    def _bilinear(
        field: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        ky0, ky1, kx0, kx1, wy, wx = weights
        iy0 = ky0[:, None]
        iy1 = ky1[:, None]
        ix0 = kx0[None, :]
        ix1 = kx1[None, :]
        if field.ndim == 3:
            v00 = field[:, iy0, ix0]
            v01 = field[:, iy0, ix1]
            v10 = field[:, iy1, ix0]
            v11 = field[:, iy1, ix1]
        else:
            v00 = field[iy0, ix0]
            v01 = field[iy0, ix1]
            v10 = field[iy1, ix0]
            v11 = field[iy1, ix1]
        wy_b = wy.view(-1, 1)
        wx_b = wx.view(1, -1)
        top = v00 + (v01 - v00) * wx_b
        bot = v10 + (v11 - v10) * wx_b
        return top + (bot - top) * wy_b

    # -- initialisation ------------------------------------------------------

    def fill_uniform(self, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> None:
        """Initialise both levels from coarse-grid equilibrium fields."""
        ny, nx = self.config.shape
        if tuple(rho.shape) != (ny, nx):
            raise ValueError("rho shape does not match the coarse grid")
        rho = rho.to(self.dtype)
        ux = ux.to(self.dtype)
        uy = uy.to(self.dtype)
        self.coarse = equilibrium(rho, ux, uy)
        self.coarse[:, self.solid_coarse] = 0.0
        rho_s = self._bilinear(rho, self._weights)
        ux_s = self._bilinear(ux, self._weights)
        uy_s = self._bilinear(uy, self._weights)
        self.fine = equilibrium(rho_s, ux_s, uy_s)
        self.fine[:, self.solid_fine] = 0.0

    # -- interface operators -------------------------------------------------

    def prolongate(self, f_coarse: torch.Tensor) -> torch.Tensor:
        """Coarse populations -> fine ghost cells (FH reconstruction)."""
        cfg = self.config
        w = self._weights
        if cfg.ghost_interp == "injection":
            ky0, ky1, kx0, kx1, wy, wx = w
            w = (
                torch.where(wy >= 0.5, ky1, ky0),
                torch.where(wy >= 0.5, ky1, ky0),
                torch.where(wx >= 0.5, kx1, kx0),
                torch.where(wx >= 0.5, kx1, kx0),
                torch.zeros_like(wy),
                torch.zeros_like(wx),
            )
        sampled = self._bilinear(f_coarse, w)
        rho, ux, uy = macroscopic(sampled)
        eq = equilibrium(rho, ux, uy)
        scale = neq_scale_factor(
            cfg.tau_coarse, cfg.tau_fine, float(cfg.ratio), cfg.neq_scale
        )
        # optional interface damping of the injected non-equilibrium part
        return eq + cfg.ghost_neq_damping * scale * (sampled - eq)

    def restrict(self, f_fine: torch.Tensor) -> torch.Tensor:
        """Fine populations -> coarse patch cells (mean + reciprocal rescale)."""
        cfg = self.config
        r = cfg.ratio
        fh, fw = self.config.fine_interior_shape
        interior = f_fine[:, 1:-1, 1:-1]
        if tuple(interior.shape) != (9, fh, fw):
            raise ValueError("fine populations have an unexpected shape")
        ny_b, nx_b = fh // r, fw // r
        averaged = interior.reshape(9, ny_b, r, nx_b, r).mean(dim=(2, 4))
        rho, ux, uy = macroscopic(averaged)
        eq = equilibrium(rho, ux, uy)
        # fine -> coarse: the *target* spacing is the coarse one, i.e. the
        # spatial ratio of this transfer is ``1 / r`` (not ``r``)
        ratio = (
            1.0 / cfg.ratio
            if cfg.restrict_spatial_ratio is None
            else float(cfg.restrict_spatial_ratio)
        )
        scale = neq_scale_factor(cfg.tau_fine, cfg.tau_coarse, ratio, cfg.neq_scale)
        return eq + scale * (averaged - eq)

    # -- collision / streaming ----------------------------------------------

    def _collide(self, f: torch.Tensor, tau: float, force: tuple[float, float]) -> torch.Tensor:
        return collide_bgk(f, tau, force, self.c, self.w)

    def _stream_coarse(self, f_post: torch.Tensor) -> torch.Tensor:
        return stream_mask(f_post, self.solid_coarse, self.config.periodic, self.c, self.opp)

    def _stream_fine(self, f_post: torch.Tensor) -> torch.Tensor:
        f_new = torch.zeros_like(f_post)
        solid_f = self.solid_fine.to(f_post.dtype)
        for i in range(9):
            dy, dx = self._int_c[i]
            nb_solid = _sample(solid_f, dy, dx, 1.0)
            src = _sample(f_post[i], dy, dx, 0.0)
            f_new[i] = torch.where(nb_solid > 0.5, f_post[self.opp[i]], src)
        f_new[:, self.solid_fine] = 0.0
        return f_new

    # -- crossing ledgers ----------------------------------------------------

    def _build_crossing_masks(
        self,
        interior: torch.Tensor,
        *,
        ghost_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Static link masks (outgoing / incoming) for the interface ledger."""
        out_mask = torch.zeros((9,) + tuple(interior.shape), dtype=torch.bool)
        in_mask = torch.zeros((9,) + tuple(interior.shape), dtype=torch.bool)
        int_f = interior.to(torch.float64)
        ext_f = (~interior).to(torch.float64)
        for i in range(9):
            dy, dx = self._int_c[i]
            nb_in = _sample(int_f, -dy, -dx, 0.0)
            nb_out = _sample(ext_f, -dy, -dx, 0.0)
            if ghost_mask is None:
                out_mask[i] = (~interior) & (nb_in > 0.5)
                in_mask[i] = interior & (nb_out > 0.5)
            else:
                # fine: a ghost cell feeding the interior is an *incoming* link
                out_mask[i] = (~ghost_mask) & (nb_out > 0.5)
                in_mask[i] = ghost_mask & (nb_in > 0.5)
        return out_mask, in_mask

    def _crossing(self, f_post: torch.Tensor, masks: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        out_mask, in_mask = masks
        cx = self.c[:, 0].view(9, 1, 1)
        cy = self.c[:, 1].view(9, 1, 1)
        fo = f_post * out_mask
        fi = f_post * in_mask
        return {
            "mass_out": float(fo.sum()),
            "mass_in": float(fi.sum()),
            "px_out": float((fo * cx).sum()),
            "px_in": float((fi * cx).sum()),
            "py_out": float((fo * cy).sum()),
            "py_in": float((fi * cy).sum()),
        }

    def _coarse_crossing(self, f_post: torch.Tensor) -> dict[str, float]:
        """Mass/momentum crossing the patch boundary on the coarse level."""
        return self._crossing(f_post, self._coarse_links)

    def _fine_crossing(self, f_post: torch.Tensor) -> dict[str, float]:
        """Mass/momentum crossing the fine patch boundary (ghost ring)."""
        return self._crossing(f_post, self._fine_links)

    # -- moment totals -------------------------------------------------------

    def totals(self) -> tuple[float, float, float]:
        """Volume-weighted physical mass and momentum of the composite grid."""
        ext = ~self._interior_mask & ~self.solid_coarse
        rho_c, ux_c, uy_c = macroscopic(self.coarse)
        mass = float(rho_c[ext].sum())
        mom_x = float((rho_c * ux_c)[ext].sum())
        mom_y = float((rho_c * uy_c)[ext].sum())
        fluid_f = ~self.solid_fine[1:-1, 1:-1]
        rho_f, ux_f, uy_f = macroscopic(self.fine[:, 1:-1, 1:-1])
        w = 1.0 / (self.config.ratio**2)
        mass += w * float(rho_f[fluid_f].sum())
        mom_x += w * float((rho_f * ux_f)[fluid_f].sum())
        mom_y += w * float((rho_f * uy_f)[fluid_f].sum())
        return mass, mom_x, mom_y

    # -- composite step ------------------------------------------------------

    def step(self) -> None:
        cfg = self.config
        rho_before, px_before, py_before = self.totals()

        # --- coarse advance (t -> t + dt) ----------------------------------
        f_c_old = self.coarse
        f_c_post = self._collide(f_c_old, cfg.tau_coarse, cfg.force)
        f_c_new = self._stream_coarse(f_c_post)
        coarse_x = self._coarse_crossing(f_c_post)

        # --- fine advance (ratio sub-steps) --------------------------------
        f_f = self.fine
        ghost_mask = self._fine_ghost_mask.unsqueeze(0)
        fine_x = {"mass_in": 0.0, "mass_out": 0.0, "px_in": 0.0, "px_out": 0.0, "py_in": 0.0, "py_out": 0.0}
        for k in range(cfg.ratio):
            alpha = (k + 0.5) / cfg.ratio if cfg.time_interp else (k + 1.0) / cfg.ratio
            ghost = self.prolongate((1.0 - alpha) * f_c_old + alpha * f_c_new)
            f_f = torch.where(ghost_mask, ghost, f_f)
            f_f_post = self._collide(f_f, cfg.tau_fine, cfg.force)
            f_f_post = torch.where(ghost_mask, f_f, f_f_post)  # ghost cells are prescribed
            step_x = self._fine_crossing(f_f_post)
            for key, value in step_x.items():
                fine_x[key] += value
            f_f = self._stream_fine(f_f_post)

        # --- restriction ----------------------------------------------------
        restricted = self.restrict(f_f)
        ys, xs = cfg.interior_coarse_slice
        f_c_new[:, ys, xs] = restricted

        self.coarse = f_c_new
        self.fine = f_f

        # --- reflux ---------------------------------------------------------
        w = 1.0 / (cfg.ratio**2)
        # Exchange imbalance of the two representations of the SAME boundary.
        # A closed composite domain conserves mass and momentum, so the sum of
        # the coarse-side exchange and the volume-weighted fine-side exchange
        # must vanish; the residual is the interface defect.
        defect_mass = (coarse_x["mass_in"] - coarse_x["mass_out"]) + w * (
            fine_x["mass_in"] - fine_x["mass_out"]
        )
        defect_px = (coarse_x["px_in"] - coarse_x["px_out"]) + w * (
            fine_x["px_in"] - fine_x["px_out"]
        )
        defect_py = (coarse_x["py_in"] - coarse_x["py_out"]) + w * (
            fine_x["py_in"] - fine_x["py_out"]
        )
        self.crossing_ledger = {
            "coarse_mass_out": coarse_x["mass_out"],
            "coarse_mass_in": coarse_x["mass_in"],
            "fine_mass_in": fine_x["mass_in"],
            "fine_mass_out": fine_x["mass_out"],
            "defect_mass": defect_mass,
            "defect_px": defect_px,
            "defect_py": defect_py,
        }
        if cfg.reflux:
            dpx = -defect_px if cfg.momentum_reflux else 0.0
            dpy = -defect_py if cfg.momentum_reflux else 0.0
            self._apply_reflux(dpx, dpy, rho_before)
        self.time += 1
        mass_after, px_after, py_after = self.totals()
        self.ledger = {
            "mass_residual": mass_after - rho_before,
            "momentum_residual_x": px_after - px_before,
            "momentum_residual_y": py_after - py_before,
        }

    def _apply_reflux(self, d_px: float, d_py: float, rho_before: float) -> None:
        """Redistribute mass (target-based) and momentum (ledger) defects."""
        cfg = self.config
        band = self._band_mask
        if not bool(band.any()):
            raise RuntimeError("no reflux band cells available")
        count = float(band.sum())

        for it in range(max(cfg.reflux_iterations, 1)):
            mass_now, _, _ = self.totals()
            d_mass = rho_before - mass_now
            if d_mass != 0.0:
                # pure density addition: exactly mass-carrying, momentum-free
                w_b = self.w.view(9, 1, 1)
                delta_rho = torch.zeros_like(self._band_mask, dtype=self.dtype)
                delta_rho[band] = d_mass / count
                self.coarse = self.coarse + w_b * delta_rho.unsqueeze(0)
            if it == 0 and (d_px != 0.0 or d_py != 0.0):
                rho_c, _, _ = macroscopic(self.coarse)
                rho_b = torch.clamp(rho_c[band], min=1e-12)
                dux = (d_px / count) / rho_b
                duy = (d_py / count) / rho_b
                cx = self.c[:, 0].view(9, 1)
                cy = self.c[:, 1].view(9, 1)
                w_b = self.w.view(9, 1)
                delta = 3.0 * w_b * rho_b.unsqueeze(0) * (
                    cx * dux.unsqueeze(0) + cy * duy.unsqueeze(0)
                )
                self.coarse[:, band] = self.coarse[:, band] + delta
        self.coarse[:, self.solid_coarse] = 0.0

    # -- post-processing -----------------------------------------------------

    def composite_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Density/velocity on the coarse grid, patch cells from fine data."""
        rho, ux, uy = macroscopic(self.coarse)
        rho = rho.clone()
        ux = ux.clone()
        uy = uy.clone()
        rho_f, ux_f, uy_f = macroscopic(self.fine[:, 1:-1, 1:-1])
        r = self.config.ratio
        ny_b, nx_b = rho_f.shape
        rho_p = rho_f.reshape(ny_b // r, r, nx_b // r, r).mean(dim=(1, 3))
        ux_p = ux_f.reshape(ny_b // r, r, nx_b // r, r).mean(dim=(1, 3))
        uy_p = uy_f.reshape(ny_b // r, r, nx_b // r, r).mean(dim=(1, 3))
        ys, xs = self.config.interior_coarse_slice
        rho[ys, xs] = rho_p
        ux[ys, xs] = ux_p
        uy[ys, xs] = uy_p
        return rho, ux, uy

    def fine_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Density/velocity on the fine (unpadded) grid."""
        return macroscopic(self.fine[:, 1:-1, 1:-1])

    def cost(self) -> dict[str, float]:
        fluid_c = float((~self.solid_coarse).sum())
        fluid_f = float((~self.solid_fine).sum())
        equivalent = fluid_c * self.config.ratio**2
        return {
            "coarse_cells": fluid_c,
            "fine_cells": fluid_f,
            "uniform_fine_equivalent_cells": equivalent,
            "cell_saving_fraction": 1.0 - (fluid_c + fluid_f) / equivalent,
        }


class UniformD2Q9:
    """Plain uniform D2Q9 solver (the reference/baseline of the gate suite).

    Supports periodic directions, half-way bounce-back solids and a uniform
    body force; used as the *uniform coarse* and *uniform fine* reference of
    the two-level capability gate.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        tau: float,
        *,
        force: tuple[float, float] = (0.0, 0.0),
        periodic: tuple[bool, bool] = (True, True),
        solid: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        ny, nx = shape
        self.shape = (ny, nx)
        self.tau = float(tau)
        self.force = (float(force[0]), float(force[1]))
        self.periodic = periodic
        self.dtype = dtype
        c = _c_on(torch.device("cpu")).to(torch.float64)
        self.c = c
        self.w = _w_for(torch.float64, torch.device("cpu"))
        self.opp = _opposite_indices(c)
        self.solid = (
            torch.zeros((ny, nx), dtype=torch.bool) if solid is None else solid.to(torch.bool)
        )
        self.f = torch.zeros((9, ny, nx), dtype=dtype)
        self.time = 0

    def fill_uniform(self, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> None:
        self.f = equilibrium(rho.to(self.dtype), ux.to(self.dtype), uy.to(self.dtype))
        self.f[:, self.solid] = 0.0

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.f = stream_mask(
                collide_bgk(self.f, self.tau, self.force, self.c, self.w),
                self.solid,
                self.periodic,
                self.c,
                self.opp,
            )
            self.time += 1

    def fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return macroscopic(self.f)

    def totals(self) -> tuple[float, float, float]:
        rho, ux, uy = macroscopic(self.f)
        fluid = ~self.solid
        return (
            float(rho[fluid].sum()),
            float((rho * ux)[fluid].sum()),
            float((rho * uy)[fluid].sum()),
        )


def obstacle_force(
    f_post: torch.Tensor,
    solid: torch.Tensor,
    c: torch.Tensor,
    *,
    weight: float = 1.0,
) -> tuple[float, float]:
    """Momentum-exchange force on a bounce-back obstacle.

    ``f_post`` must be the post-collision state *before* streaming.  Every link
    that would carry a population into a solid cell contributes
    ``2 c_i f_i`` to the force on the obstacle (half-way bounce-back).
    """
    solid_f = solid.to(f_post.dtype)
    fx = 0.0
    fy = 0.0
    for i in range(9):
        dx, dy = int(c[i, 0]), int(c[i, 1])
        target_solid = torch.roll(solid_f, shifts=(-dy, -dx), dims=(-2, -1))
        # out-of-domain targets are not part of the obstacle
        hit = (~solid) & (target_solid > 0.5)
        contribution = 2.0 * float(f_post[i][hit].sum())
        fx += dx * contribution
        fy += dy * contribution
    return weight * fx, weight * fy


def cylinder_solid_mask(
    shape: tuple[int, int],
    centre: tuple[float, float],
    radius: float,
    *,
    spacing: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Digitised disk mask; ``spacing`` is the cell size in coarse-grid units."""
    ny, nx = shape
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * spacing + origin[1]
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * spacing + origin[0]
    yy = y.view(-1, 1)
    xx = x.view(1, -1)
    return (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius**2