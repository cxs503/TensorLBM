"""Composite-grid 2-level AMR for D2Q9 LBM with conservative reflux correction.

Motivation
----------
The nested-patch scheme in :mod:`tensorlbm.amr_taylor_green2d` restricted the
fine patch back onto the *whole* coarse patch and measured the kinetic energy on
the coarse grid only; it carried no interface flux ledger, so it conserved mass
(by construction of the restriction) but dissipated energy at ~10 % level and
gave no way to *prove* flux conservation across the coarse--fine interface.

This module implements a proper Berger--Colella composite grid for D2Q9 BGK LBM:

* the coarse grid is advanced on the *active* cells only (the cells covered by a
  fine patch are excluded from the coarse update, "hole" in the coarse grid);
* every fine patch is a rectangle of coarse cells refined by ``ratio = 2`` with
  one layer of ghost fine cells around it;
* ghost fine cells are reconstructed from the coarse state with a
  Filippova--Hanel style equilibrium/non-equilibrium split (bilinear in space,
  linear in time between ``t^n`` and ``t^{n+1}``);
* covered coarse cells are filled from the fine solution with the reverse
  Filippova--Hanel rescaling of the two-substep average;
* the mass and momentum crossing the interface are logged on *both* sides
  (coarse -> patch, patch -> coarse, ghost -> fine, fine -> ghost) and the
  mismatch is fed back as a reflux correction (flux averaging, Berger & Colella
  1989) so that the composite conserved quantities are conserved to round-off.

Conservation identity implemented here
--------------------------------------
Let ``A`` = mass the coarse update pushed into the patch region, ``B`` = mass the
coarse update received from the patch region, ``G`` = mass the fine patch
received from its ghost ring and ``H`` = mass the fine patch sent into the ghost
ring (all per coarse step, in coarse mass units: a fine cell carries ``rho/4``).
The uncorrected composite mass change is ``(B - A) + (G - H)``.  With the
averaged interface fluxes ``F_in = (A + H)/2`` and ``F_out = (B + G)/2`` the
corrections are ``(F_out - B) - (F_in - A)`` on the coarse side and
``(F_in - G) - (F_out - H)`` on the fine side, whose sum is identically zero:

    dM_composite = 0  (exactly, up to floating-point round-off)

The same construction is applied to the two momentum components, hence mass and
momentum are both conserved across the interface (mass only, for wall-bounded
cases where the walls exchange momentum by construction).

Units
-----
Everything is in lattice units with ``dx_c = dt_c = 1`` on the coarse grid and
``dx_f = dt_f = 1/2`` on the fine grid (convective / "same viscosity" scaling):
``tau_f = 2 tau_c - 1/2`` so that the kinematic viscosity in lattice units is
``nu_f = 2 nu_c`` (identical physical viscosity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# D2Q9 lattice
# ---------------------------------------------------------------------------

C = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
CX = C[:, 0]
CY = C[:, 1]
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
W = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
              1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
W3 = W[:, None, None]


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium populations, shape ``(9, ...)``."""
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return W3 * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Density and velocity from populations."""
    rho = f.sum(axis=0)
    rho_safe = np.where(np.abs(rho) < 1e-300, 1e-300, rho)
    ux = np.tensordot(CX, f, axes=(0, 0)) / rho_safe
    uy = np.tensordot(CY, f, axes=(0, 0)) / rho_safe
    return rho, ux, uy


def collide(f: np.ndarray, tau: float) -> np.ndarray:
    """BGK collision."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def stream_periodic(f: np.ndarray) -> np.ndarray:
    """Periodic streaming (wrap-around)."""
    out = np.empty_like(f)
    for k in range(9):
        out[k] = np.roll(np.roll(f[k], CY[k], axis=0), CX[k], axis=1)
    return out


def _shift_mask(mask: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """``out[y, x] = mask[y - cy, x - cx]`` with ``False`` outside the grid."""
    out = np.zeros_like(mask)
    ny, nx = mask.shape
    ys_dst = slice(max(0, cy), ny + min(0, cy))
    ys_src = slice(max(0, -cy), ny + min(0, -cy))
    xs_dst = slice(max(0, cx), nx + min(0, cx))
    xs_src = slice(max(0, -cx), nx + min(0, -cx))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def stream_walls(
    f_post: np.ndarray,
    fluid: np.ndarray,
    wall_ux: np.ndarray | None = None,
    wall_uy: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Streaming with half-way bounce-back on solid neighbours.

    ``fluid`` is ``True`` on fluid cells.  For every fluid cell and direction ``k``
    whose upstream neighbour is solid (or outside the domain) the incoming
    population is replaced by the reflected self population of the opposite
    direction (plus the moving-wall correction ``6 w_k rho (c_k . u_wall)``).

    Returns the streamed populations and a dictionary with the per-link momentum
    exchange (``force_x``, ``force_y`` = momentum per unit time transferred
    *from the fluid to the solid*, i.e. drag on the solid needs the negative).
    """
    out = stream_periodic(f_post)  # wrap-around; overridden where the source is solid
    rho = f_post.sum(axis=0)
    fx = 0.0
    fy = 0.0
    for k in range(9):
        src_fluid = _shift_mask(fluid, int(CX[k]), int(CY[k]))
        bad = fluid & ~src_fluid
        if not bad.any():
            continue
        if wall_ux is None:
            bounce = f_post[OPP[k]]
        else:
            # moving wall: the wall velocity is read on the *solid* neighbour cell;
            # the reflected (incoming) population carries
            #   f_k = f_opp(k)^* + 6 w_k rho (c_opp(k) . u_wall)
            # (c_opp(k) is the direction of the population that hits the wall).
            wux_src = _shift_mask(wall_ux, int(CX[k]), int(CY[k]))
            wuy_src = _shift_mask(wall_uy, int(CX[k]), int(CY[k]))
            uwc = CX[k] * wux_src + CY[k] * wuy_src
            bounce = f_post[OPP[k]] + 6.0 * W[k] * rho * uwc
        out[k] = np.where(bad, bounce, out[k])
        # momentum exchange (Ladd): 2 c_k f_k^post
        m = float(np.sum(f_post[k][bad]))
        fx += 2.0 * CX[k] * m
        fy += 2.0 * CY[k] * m
    return out, {"momentum_to_solid_x": fx, "momentum_to_solid_y": fy}


def solid_force(f_post: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """Momentum-exchange force (per unit depth) on the solids.

    ``F = sum over boundary links of 2 c_k f_k^post`` (half-way bounce-back,
    Ladd 1994).  The force acts on the solid, so it is the momentum *leaving*
    the fluid.
    """
    force = np.zeros(2)
    for k in range(9):
        cx, cy = int(CX[k]), int(CY[k])
        if cx == 0 and cy == 0:
            continue
        src_solid = _shift_mask(~fluid, -cx, -cy)
        links = fluid & src_solid
        if not links.any():
            continue
        m = float(np.sum(f_post[k][links]))
        force[0] += 2.0 * cx * m
        force[1] += 2.0 * cy * m
    return force


# ---------------------------------------------------------------------------
# Fine patch
# ---------------------------------------------------------------------------


@dataclass
class PatchSpec:
    """Fine patch specification in coarse-cell indices (0-based, half-open)."""

    j0: int  # first covered coarse row
    i0: int  # first covered coarse column
    nj: int  # number of covered coarse rows
    ni: int  # number of covered coarse columns

    @property
    def j1(self) -> int:
        return self.j0 + self.nj

    @property
    def i1(self) -> int:
        return self.i0 + self.ni


@dataclass
class Patch:
    """Runtime state of a fine patch (fine array + precomputed index tables)."""

    spec: PatchSpec
    f: np.ndarray
    fluid: np.ndarray
    radius: int = 2  # refinement ratio
    ghost_rows: np.ndarray | None = None
    ghost_cols: np.ndarray | None = None
    ghost_w: np.ndarray | None = None
    interior_mask: np.ndarray | None = None
    ledger: dict = field(default_factory=dict)

    @property
    def ratio(self) -> int:
        return self.radius


def _bilinear_tables(spec: PatchSpec, ratio: int, nfy: int, nfx: int):
    """Precompute bilinear interpolation tables for the fine ghost ring.

    Fine cell ``(a, b)`` (``a, b`` in ``0 .. nfy-1`` / ``0 .. nfx-1``, ghost ring
    included) has its centre at coarse coordinate

        x = i0 + (b - 0.5) / ratio,  y = j0 + (a - 0.5) / ratio

    Using the lower-left coarse cell index ``floor(x), floor(y)`` with weights.
    """
    a, b = np.meshgrid(np.arange(nfy), np.arange(nfx), indexing="ij")
    x = spec.i0 + (b - 0.5) / ratio
    y = spec.j0 + (a - 0.5) / ratio
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    wx = x - x0
    wy = y - y0
    iz = slice(0, 2)
    return (y0, x0, wy, wx)


class CompositeAMR2D:
    """Two-level composite grid (coarse grid with holes + fine patches)."""

    def __init__(
        self,
        f_coarse: np.ndarray,
        tau_coarse: float,
        patches: list[PatchSpec],
        ratio: int = 2,
        solid_coarse: np.ndarray | None = None,
        wall_ux_coarse: np.ndarray | None = None,
        wall_uy_coarse: np.ndarray | None = None,
        periodic: bool = True,
        reflux: bool = True,
        reflux_momentum: bool = True,
        fine_solid=None,
        diag_budget: bool = False,
        reflux_spread: str = "interface",
    ) -> None:
        self.fc = np.array(f_coarse, dtype=np.float64, copy=True)
        self.nyc, self.nxc = self.fc.shape[1], self.fc.shape[2]
        self.tau_c = float(tau_coarse)
        self.tau_f = 2.0 * self.tau_c - 0.5
        self.nu_c = (self.tau_c - 0.5) / 3.0
        self.nu_f = 2.0 * self.nu_c
        self.ratio = int(ratio)
        self.periodic = bool(periodic)
        self.reflux = bool(reflux)
        self.diag_budget = bool(diag_budget)
        # "interface": correct only the cells touching the interface
        # "global":   spread the same (exactly conservative) correction over the
        #             whole active coarse region / fine patch interior
        self.reflux_spread = str(reflux_spread)
        self.reflux_momentum = bool(reflux_momentum)
        # optional analytic fine-scale geometry: fine_solid(ypos, xpos) -> bool
        self.fine_solid = fine_solid

        if solid_coarse is None:
            solid_coarse = np.zeros((self.nyc, self.nxc), dtype=bool)
        self.solid_c = np.array(solid_coarse, dtype=bool)
        self.fluid_c = ~self.solid_c
        if wall_ux_coarse is None:
            wall_ux_coarse = np.zeros((self.nyc, self.nxc))
            wall_uy_coarse = np.zeros((self.nyc, self.nxc))
        self.wall_ux_c = np.asarray(wall_ux_coarse, dtype=np.float64)
        self.wall_uy_c = np.asarray(wall_uy_coarse, dtype=np.float64)

        self.covered = np.zeros((self.nyc, self.nxc), dtype=bool)
        self.patches: list[Patch] = []
        for spec in patches:
            self.covered[spec.j0:spec.j1, spec.i0:spec.i1] = True

        for spec in patches:
            self.patches.append(self._make_patch(spec))
        self.active_c = self.fluid_c & ~self.covered

        # covered coarse cells (filled from the fine solution each step)
        self.fc_covered = np.zeros((9, self.nyc, self.nxc))
        self._init_covered_from_coarse()

        self.scale_c2f = self.tau_f / (self.ratio * self.tau_c)
        self.scale_f2c = self.ratio * self.tau_c / self.tau_f

        self.step_count = 0
        self.diagnostics: list[dict] = []
        self.momentum_to_solid = np.zeros(2)
        self.force_wall = np.zeros(2)
        self._force_window: list[float] = []

    # -- construction -------------------------------------------------------
    def _make_patch(self, spec: PatchSpec) -> Patch:
        r = self.ratio
        nfy, nfx = r * spec.nj + 2, r * spec.ni + 2
        # fine cell (a, b) sits at coarse coordinate (j0 + (a-.5)/r, i0 + (b-.5)/r)
        aa = np.arange(nfy)
        bb = np.arange(nfx)
        crow = np.clip(np.floor(spec.j0 + (aa - 0.5) / r), 0, self.nyc - 1).astype(int)
        ccol = np.clip(np.floor(spec.i0 + (bb - 0.5) / r), 0, self.nxc - 1).astype(int)
        f_fine = self.fc[:, crow][:, :, ccol]

        # fine fluid mask: refine the coarse solid mask, but only the interior;
        # ghost cells are fluid (they are fed by interpolation) unless the
        # coarse cell they sit on is solid (i.e. the patch abuts a wall).
        if self.fine_solid is None:
            solid_f_int = self.solid_c[crow[1:-1]][:, ccol[1:-1]]
        else:
            a_i = np.arange(1, nfy - 1)[:, None] * np.ones((1, nfx - 2))
            b_i = np.ones((nfy - 2, 1)) * np.arange(1, nfx - 1)[None, :]
            solid_f_int = np.asarray(self.fine_solid(
                spec.j0 + (a_i - 0.5) / r, spec.i0 + (b_i - 0.5) / r), dtype=bool)
        assert solid_f_int.shape == (nfy - 2, nfx - 2)
        fluid = np.ones((nfy, nfx), dtype=bool)
        fluid[1:-1, 1:-1] = ~solid_f_int
        # ghost ring: solid where the interpolated position falls on a solid cell
        # *or* outside the domain (the patch abuts a wall → bounce-back there).
        y0, x0, wy, wx = _bilinear_tables(spec, r, nfy, nfx)
        ghost = np.zeros_like(fluid)
        ghost[0, :] = ghost[-1, :] = True
        ghost[:, 0] = ghost[:, -1] = True
        a_idx = np.arange(nfy)[:, None] * np.ones((1, nfx))
        b_idx = np.ones((nfy, 1)) * np.arange(nfx)[None, :]
        ypos = spec.j0 + (a_idx - 0.5) / r
        xpos = spec.i0 + (b_idx - 0.5) / r
        outside = (ypos < 0) | (ypos >= self.nyc) | (xpos < 0) | (xpos >= self.nxc)
        solid_at = self.solid_c[np.clip(y0, 0, self.nyc - 1), np.clip(x0, 0, self.nxc - 1)]
        fluid[ghost & (solid_at | outside)] = False
        interior = np.zeros_like(fluid)
        interior[1:-1, 1:-1] = True
        interior &= fluid
        return Patch(
            spec=spec,
            f=f_fine,
            fluid=fluid,
            radius=r,
            ghost_rows=y0,
            ghost_cols=x0,
            ghost_w=(wy, wx),
            interior_mask=interior,
        )

    def _init_covered_from_coarse(self) -> None:
        for p in self.patches:
            spec = p.spec
            self.fc_covered[:, spec.j0:spec.j1, spec.i0:spec.i1] = \
                self.fc[:, spec.j0:spec.j1, spec.i0:spec.i1]

    # -- interface helpers --------------------------------------------------
    def _interp_coarse(self, field: np.ndarray, p: Patch) -> np.ndarray:
        """Bilinear interpolation of a coarse field onto the fine ghost ring."""
        wy, wx = p.ghost_w
        y0 = np.clip(p.ghost_rows, 0, self.nyc - 1)
        x0 = np.clip(p.ghost_cols, 0, self.nxc - 1)
        y1 = np.clip(y0 + 1, 0, self.nyc - 1)
        x1 = np.clip(x0 + 1, 0, self.nxc - 1)
        return (field[y0, x0] * (1 - wy) * (1 - wx)
                + field[y0, x1] * (1 - wy) * wx
                + field[y1, x0] * wy * (1 - wx)
                + field[y1, x1] * wy * wx)

    def _fill_ghosts(self, p: Patch, state: np.ndarray) -> np.ndarray:
        """Reconstruct the fine ghost ring from an interpolated coarse state."""
        rho_c, ux_c, uy_c = macroscopic(state)
        feq_c = equilibrium(rho_c, ux_c, uy_c)
        neq_c = state - feq_c
        rho = self._interp_coarse(rho_c, p)
        ux = self._interp_coarse(ux_c, p)
        uy = self._interp_coarse(uy_c, p)
        feq = equilibrium(rho, ux, uy)
        neq = np.stack([self._interp_coarse(neq_c[k], p) for k in range(9)])
        fghost = feq + self.scale_c2f * neq
        return fghost

    @staticmethod
    def _crossing_fluxes(f_post: np.ndarray, src: np.ndarray, dst: np.ndarray):
        """Mass/momentum crossing from ``src`` cells into ``dst`` cells.

        Returns ``(mass, px, py)`` where momentum uses the lattice direction of
        each crossing population.
        """
        mass = 0.0
        px = 0.0
        py = 0.0
        for k in range(9):
            cx, cy = int(CX[k]), int(CY[k])
            if cx == 0 and cy == 0:
                continue
            m = src & _shift_mask(dst, -cx, -cy)
            if not m.any():
                continue
            s = float(f_post[k][m].sum())
            mass += s
            px += cx * s
            py += cy * s
        return mass, px, py

    def _restrict_patch(self, p: Patch, f_fine_avg: np.ndarray) -> np.ndarray:
        """Filippova-Hanel restriction of the substep-averaged fine state."""
        spec = p.spec
        r = self.ratio
        inner = f_fine_avg[:, 1:-1, 1:-1]
        block = inner.reshape(9, spec.nj, r, spec.ni, r).mean(axis=(2, 4))
        rho, ux, uy = macroscopic(block)
        feq = equilibrium(rho, ux, uy)
        neq = block - feq
        return feq + self.scale_f2c * neq

    # -- conserved quantities ----------------------------------------------
    def composite_mass(self) -> float:
        rho_c, _, _ = macroscopic(self.fc)
        m = float(np.sum(rho_c[self.active_c]))
        for p in self.patches:
            rho_f, _, _ = macroscopic(p.f)
            m += float(np.sum(rho_f[p.interior_mask])) / (self.ratio ** 2)
        return m

    def composite_momentum(self) -> np.ndarray:
        rho_c, ux_c, uy_c = macroscopic(self.fc)
        px = float(np.sum((rho_c * ux_c)[self.active_c]))
        py = float(np.sum((rho_c * uy_c)[self.active_c]))
        for p in self.patches:
            rho_f, ux_f, uy_f = macroscopic(p.f)
            px += float(np.sum((rho_f * ux_f)[p.interior_mask])) / self.ratio ** 2
            py += float(np.sum((rho_f * uy_f)[p.interior_mask])) / self.ratio ** 2
        return np.array([px, py])

    def composite_energy(self) -> float:
        """Domain-averaged kinetic energy on the composite grid."""
        rho_c, ux_c, uy_c = macroscopic(self.fc)
        e_c = 0.5 * float(np.sum((ux_c ** 2 + uy_c ** 2)[self.active_c]))
        for p in self.patches:
            rho_f, ux_f, uy_f = macroscopic(p.f)
            e_f = 0.5 * float(np.sum((ux_f ** 2 + uy_f ** 2)[p.interior_mask]))
            e_c += e_f / self.ratio ** 2
        n_equiv = float(self.active_c.sum()) + sum(
            float(p.interior_mask.sum()) / self.ratio ** 2 for p in self.patches)
        return e_c / n_equiv

    # -- correction injection ---------------------------------------------
    @staticmethod
    def _inject(f: np.ndarray, mask: np.ndarray, dm: float, dpx: float, dpy: float) -> None:
        """Add mass ``dm`` and momentum ``(dpx, dpy)`` per cell over ``mask``.

        The correction is injected as a *local equilibrium perturbation*: the
        cell is given the density/velocity pair ``(rho + dm, (rho u + dp)/(rho + dm))``
        and the population increment is ``feq(new) - feq(old)``.  This keeps the
        summed mass and momentum exact (round-off) while injecting no spurious
        stress -- injecting into ``f_0`` alone (a mass-only kick) was found to
        seed an acoustic instability at the interface.
        """
        if mask.sum() == 0:
            return
        rho = f.sum(axis=0)
        rho_safe = np.where(np.abs(rho) < 1e-300, 1e-300, rho)
        ux = np.tensordot(CX, f, axes=(0, 0)) / rho_safe
        uy = np.tensordot(CY, f, axes=(0, 0)) / rho_safe
        rho_n = rho + dm
        rho_n_safe = np.where(np.abs(rho_n) < 1e-300, 1e-300, rho_n)
        ux_n = (rho * ux + dpx) / rho_n_safe
        uy_n = (rho * uy + dpy) / rho_n_safe
        feq_old = equilibrium(rho, ux, uy)
        feq_new = equilibrium(rho_n, ux_n, uy_n)
        f[:, mask] += (feq_new - feq_old)[:, mask]

    def composite_field(self) -> dict:
        """Assemble the composite solution on a uniform fine (ratio 2) grid.

        Coarse active cells are replicated to the fine resolution (nearest
        neighbour) and the refined patches are then stamped on top, so the
        returned field has the fine resolution everywhere and can be compared
        directly with a uniform run at the fine resolution.
        """
        r = self.ratio
        ny_f, nx_f = self.nyc * r, self.nxc * r
        rho_c, ux_c, uy_c = macroscopic(self.fc)
        out = {}
        for name, arr in (("rho", rho_c), ("ux", ux_c), ("uy", uy_c)):
            out[name] = np.repeat(np.repeat(arr, r, axis=0), r, axis=1)
        for p in self.patches:
            spec = p.spec
            rho_f, ux_f, uy_f = macroscopic(p.f)
            for name, arr in (("rho", rho_f), ("ux", ux_f), ("uy", uy_f)):
                out[name][r * spec.j0:r * spec.j1, r * spec.i0:r * spec.i1] = \
                    arr[1:-1, 1:-1]
        return out

    def patch_force(self, idx: int = 0) -> np.ndarray:
        """Momentum-exchange force (per unit depth) on the solids inside a patch.

        Ladd momentum exchange over the links crossing the solid boundary:
        ``F = sum_links 2 c_k f_k^post`` with ``k`` the direction leaving the
        fluid cell towards a solid neighbour.
        """
        p = self.patches[idx]
        rho, ux, uy = macroscopic(p.f)  # noqa: F841 (kept for debugging)
        acc = solid_force(p.f, p.fluid)
        return acc

    # -- one coarse step ----------------------------------------------------
    def step(self) -> dict:
        r = self.ratio
        m_start = self.composite_mass() if self.diag_budget else 0.0
        fc_n = self.fc.copy()  # state at t^n (for time interpolation)

        # ---- coarse advance ------------------------------------------------
        if self.periodic:
            fc_post = collide(self.fc, self.tau_c)
            # covered cells stream with the restricted fine values (injected)
            for p in self.patches:
                spec = p.spec
                fc_post[:, spec.j0:spec.j1, spec.i0:spec.i1] = \
                    self.fc_covered[:, spec.j0:spec.j1, spec.i0:spec.i1]
            fc_next = stream_periodic(fc_post)
        else:
            fc_post = collide(self.fc, self.tau_c)
            for p in self.patches:
                spec = p.spec
                fc_post[:, spec.j0:spec.j1, spec.i0:spec.i1] = \
                    self.fc_covered[:, spec.j0:spec.j1, spec.i0:spec.i1]
            fc_next, mom = stream_walls(fc_post, self.fluid_c,
                                        self.wall_ux_c, self.wall_uy_c)
            self.momentum_to_solid = np.array(
                [mom["momentum_to_solid_x"], mom["momentum_to_solid_y"]])
            self.force_wall = self.momentum_to_solid.copy()

        diag: dict = {"step": self.step_count + 1}
        ledgers = []
        for p in self.patches:
            spec = p.spec
            cover = np.zeros_like(self.covered)
            cover[spec.j0:spec.j1, spec.i0:spec.i1] = True
            # coarse -> patch (using the post-collision coarse state)
            A = self._crossing_fluxes(fc_post, self.fluid_c & ~cover, cover)
            # patch -> coarse (using the restricted fine data the coarse used)
            B = self._crossing_fluxes(self.fc_covered, cover, self.fluid_c & ~cover)
            ledgers.append({"A": np.array(A), "B": np.array(B),
                            "G": np.zeros(3), "H": np.zeros(3)})

        # ---- fine sub-cycling ---------------------------------------------
        for sub in range(r):
            alpha = (sub + 1) / r  # 0.5, 1.0
            state = (1.0 - alpha) * fc_n + alpha * fc_next
            for idx, p in enumerate(self.patches):
                fghost = self._fill_ghosts(p, state)
                # put ghost populations into the fine array (streaming sources)
                ghost_mask = ~p.interior_mask
                p.f[:, ghost_mask] = fghost[:, ghost_mask]
                f_post_f = collide(p.f, self.tau_f)
                # freeze ghost cells (they are boundary data, not evolved)
                f_post_f[:, ghost_mask] = fghost[:, ghost_mask]
                if self.periodic_patch(p):
                    f_next = stream_periodic(f_post_f)
                else:
                    wux_f, wuy_f = self._fine_wall_velocity(p)
                    f_next, _ = stream_walls(f_post_f, p.fluid, wux_f, wuy_f)
                # fluxes across the patch boundary
                inside = p.interior_mask
                outside = ~p.interior_mask
                # both fluxes are read from the *pre-streaming* state: the ghost
                # cells hold the boundary data that will be streamed in (G) and
                # the interior populations that will be streamed out (H).
                G = self._crossing_fluxes(f_post_f, outside, inside)  # ghost -> fine
                H = self._crossing_fluxes(f_post_f, inside, outside)  # fine -> ghost
                ledgers[idx]["G"] = ledgers[idx]["G"] + np.array(G)
                ledgers[idx]["H"] = ledgers[idx]["H"] + np.array(H)
                p.f = f_next
                if sub == 0:
                    p._f_first = f_next.copy()
                else:
                    p._f_avg = 0.5 * (p._f_first + f_next)

        # ---- reflux + restriction ----------------------------------------
        for idx, p in enumerate(self.patches):
            spec = p.spec
            A = ledgers[idx]["A"]
            B = ledgers[idx]["B"]
            G = ledgers[idx]["G"]
            H = ledgers[idx]["H"]
            # coarse fluxes (A, B) are already in coarse mass units; a fine face
            # crossing carries f_k / r^2 in coarse mass units (face 1/r x dt 1/r).
            sc = 1.0 / (r * r)
            A = np.array(A, dtype=float)
            B = np.array(B, dtype=float)
            G = np.array(G, dtype=float) * sc
            H = np.array(H, dtype=float) * sc
            f_in = 0.5 * (A + H)
            f_out = 0.5 * (B + G)
            # coarse side corrections (per coarse step)
            dm_c = (f_out[0] - B[0]) - (f_in[0] - A[0])
            dpx_c = (f_out[1] - B[1]) - (f_in[1] - A[1])
            dpy_c = (f_out[2] - B[2]) - (f_in[2] - A[2])
            cover = np.zeros_like(self.covered)
            cover[spec.j0:spec.j1, spec.i0:spec.i1] = True
            if self.reflux_spread == "global":
                ring_c = self.active_c.copy()
            else:
                ring_c = self.fluid_c & ~cover & _dilate_mask(cover)
            ncells_c = float(ring_c.sum())
            if not self.reflux:
                dm_c = 0.0
            if not (self.reflux and self.reflux_momentum):
                dpx_c = dpy_c = 0.0
            if ncells_c > 0:
                self._inject(fc_next, ring_c, dm_c / ncells_c,
                             dpx_c / ncells_c, dpy_c / ncells_c)
            # fine side corrections (fine cell mass unit = 1 / r^2 coarse)
            dm_f = r * r * ((f_in[0] - G[0]) - (f_out[0] - H[0]))
            dpx_f = r * r * ((f_in[1] - G[1]) - (f_out[1] - H[1]))
            dpy_f = r * r * ((f_in[2] - G[2]) - (f_out[2] - H[2]))
            if self.reflux_spread == "global":
                boundary_f = p.interior_mask
            else:
                boundary_f = p.interior_mask & _dilate_mask(~p.interior_mask)
            ncells_f = float(boundary_f.sum())
            if not self.reflux:
                dm_f = 0.0
            if not (self.reflux and self.reflux_momentum):
                dpx_f = dpy_f = 0.0
            if ncells_f > 0:
                self._inject(p.f, boundary_f, dm_f / ncells_f,
                             dpx_f / ncells_f, dpy_f / ncells_f)
            p.ledger = {
                "A_coarse_to_patch": float(A[0]),
                "B_patch_to_coarse": float(B[0]),
                "G_ghost_to_fine": float(G[0]),
                "H_fine_to_ghost": float(H[0]),
                "reflux_mass_coarse": float(dm_c),
                "reflux_mass_fine": float(dm_f),
                "interface_flux_scale": float(abs(A[0]) + abs(B[0]) + 1e-300),
            }
            diag.update({f"patch{idx}_{k}": v for k, v in p.ledger.items()})
            diag[f"patch{idx}_f_in"] = float(f_in[0])
            diag[f"patch{idx}_f_out"] = float(f_out[0])
            # covered coarse cells for the next step
            restricted = self._restrict_patch(p, p._f_avg)
            self.fc_covered[:, spec.j0:spec.j1, spec.i0:spec.i1] = restricted
            fc_next[:, spec.j0:spec.j1, spec.i0:spec.i1] = restricted

        self.fc = fc_next
        self.step_count += 1
        if self.diag_budget:
            m_end = self.composite_mass()
            f_in = sum(diag[f"patch{i}_f_in"] for i in range(len(self.patches)))
            f_out = sum(diag[f"patch{i}_f_out"] for i in range(len(self.patches)))
            diag["flux_in_total"] = f_in
            diag["flux_out_total"] = f_out
            diag["mass_change"] = m_end - m_start
            # observed interface-driven mass change must equal F_in - F_out
            diag["reflux_budget_mismatch"] = (m_end - m_start) - (f_in - f_out)
        self.diagnostics.append(diag)
        return diag

    def periodic_patch(self, p: Patch) -> bool:
        """A patch is periodic only if the whole domain is periodic."""
        if not self.periodic:
            return False
        return True

    def _fine_wall_velocity(self, p: Patch):
        """Wall velocity sampled (nearest neighbour, clipped) at fine cell centres.

        Ghost cells that lie outside the coarse domain sample the boundary coarse
        cell, so a moving lid keeps driving the fine patch across the patch
        boundary.
        """
        r = self.ratio
        nfy, nfx = p.f.shape[1], p.f.shape[2]
        a_idx = np.arange(nfy)[:, None] * np.ones((1, nfx))
        b_idx = np.ones((nfy, 1)) * np.arange(nfx)[None, :]
        ypos = p.spec.j0 + (a_idx - 0.5) / r
        xpos = p.spec.i0 + (b_idx - 0.5) / r
        jj = np.clip(np.floor(ypos), 0, self.nyc - 1).astype(int)
        ii = np.clip(np.floor(xpos), 0, self.nxc - 1).astype(int)
        return self.wall_ux_c[jj, ii], self.wall_uy_c[jj, ii]


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """One-cell dilation of a boolean mask (4-neighbourhood + self)."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


# ---------------------------------------------------------------------------
# Uniform-grid reference solver (same kernels, no refinement)
# ---------------------------------------------------------------------------


def run_uniform(
    f0: np.ndarray,
    tau: float,
    steps: int,
    periodic: bool = True,
    solid: np.ndarray | None = None,
    wall_ux: np.ndarray | None = None,
    wall_uy: np.ndarray | None = None,
    probe=None,
) -> dict:
    """Advance a uniform D2Q9 BGK simulation and report diagnostics.

    ``probe`` is an optional callable ``probe(step, f, solver_state) -> dict``.
    """
    f = np.array(f0, dtype=np.float64, copy=True)
    ny, nx = f.shape[1], f.shape[2]
    fluid = np.ones((ny, nx), dtype=bool) if solid is None else ~np.asarray(solid, bool)
    out: dict = {"energy": [], "mass": [], "steps": [], "force_x": [], "force_y": []}
    force = np.zeros(2)
    for step in range(steps + 1):
        if probe is not None:
            info = probe(step, f, {"force": force})
            for k, v in info.items():
                out.setdefault(k, []).append(v)
        if step == steps:
            break
        f_post = collide(f, tau)
        if periodic:
            f = stream_periodic(f_post)
        else:
            f, mom = stream_walls(f_post, fluid, wall_ux, wall_uy)
            force = np.array([-mom["momentum_to_solid_x"], -mom["momentum_to_solid_y"]])
    out["force"] = force
    return out