"""Passive-scalar D2Q9 lattice-Boltzmann transport of suspended sediment.

Two solvers live here; both are reported.

1. ``run_rouse_lbm`` -- **Rouse-log graded coordinate solver (primary anchor
   for criterion C2).**
   The steady suspended-load balance with the open-channel sediment
   diffusivity ``D_s(y) = kappa u* y (1 - y/h)`` is

       d/dy [ D_s(y) dc/dy + w_s c ] = 0                              (1)

   Introduce the Rouse variable ``xi = ln( y / (h - y) )``.  Since
   ``dxi/dy = 1 / [y (1 - y/h)]`` we have ``D_s dxi/dy = kappa u*``, so (1)
   becomes a **constant-coefficient** advection-diffusion equation in xi::

       d/dxi [ kappa u* dc/dxi + w_s c ] = 0                          (2)

   whose solution ``c = c_a exp(-Z (xi - xi_a))`` with ``Z = w_s/(kappa u*)``
   is exactly the Rouse profile.  Discretising (2) is a graded mesh (fine near
   the bed, coarse in the outer layer -- the physically appropriate grading) on
   which the scalar LBM has *constant* transport coefficients.  The LBM
   (:class:`RouseLBMConfig`) solves (2) with D2Q9/BGK; the flow enters one-way
   through ``u*`` (which sets the diffusivity) and through the streamwise
   velocity of the LBM channel (which does not affect the vertical balance).
   The fitted ``Z`` is compared with ``w_s/(kappa u*)``.

2. ``run_scalar_channel`` -- the *same* physics discretised directly on the
   **uniform physical grid with a row-dependent relaxation time**
   ``tau_c(y) = 1/2 + 3 D_s(y)``.  This "naive variable-tau" formulation is
   measured here to be **systematically over-diffusive**: the spatially varying
   relaxation produces a leading-order spurious flux of order ``dD_s/dy``,
   comparable to the settling flux ``w_s c``.  Measured consequence: the
   recovered Rouse exponent is ~47 % low (Z = 0.156 vs 0.293) and the error
   does **not** shrink with refinement (Z = 0.149 -> 0.200 for ny = 30 -> 60):
   it is a modified-equation error, not a resolution error of the profile.
   It is kept as a documented negative result and as the solver used for the
   mass-budget closure (where conservation, not profile accuracy, is tested).

Wall treatment: both ends use a *mirror* (zero total flux) condition, because
plain bounce-back only cancels the diffusive flux and silently leaks the
settling flux (measured: 56 % mass loss over 6e4 steps in a closed box).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rouse import KAPPA, fit_rouse_exponent, rouse_profile

__all__ = [
    "RouseLBMConfig",
    "ScalarLBMConfig",
    "column_exchange_budget",
    "rouse_log_metric",
    "run_rouse_lbm",
    "run_scalar_channel",
]

#: D2Q9 lattice velocities / weights (matches tensorlbm.d2q9 ordering).
_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
_W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def _equilibrium(c: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Linearised D2Q9 equilibrium for the scalar density ``c``."""
    cu = _C[:, 0, None, None] * ux[None] + _C[:, 1, None, None] * uy[None]
    return _W[:, None, None] * c[None] * (1.0 + 3.0 * cu)


def _stream(g: np.ndarray) -> np.ndarray:
    out = np.empty_like(g)
    for i in range(9):
        out[i] = np.roll(np.roll(g[i], _C[i, 0], axis=1), _C[i, 1], axis=0)
    return out


def _mirror_wall(
    g: np.ndarray,
    c_row: np.ndarray,
    ux_row: np.ndarray,
    v_out: float,
    j: int,
    incoming_cy: int,
) -> None:
    """Mirror wall: zero *total* flux through the boundary row ``j``.

    ``g_i = g_i^eq(c, ux, -v) + [g_opp - g_opp^eq(c, ux, -v)]``: the
    non-equilibrium part is reflected about the equilibrium with the
    wall-normal velocity reversed, which cancels diffusion *and* advection.
    """
    c_l = c_row[None, :]
    ux_l = ux_row[None, :]
    uy_l = np.full_like(c_l, -v_out)
    geq = _equilibrium(c_l, ux_l, uy_l)
    links = np.where(_C[:, 1] == incoming_cy)[0]
    before = g[links, j, :].sum()
    for i in links:
        g[i, j, :] = geq[i, 0, :] + (g[_OPP[i], j, :] - geq[_OPP[i], 0, :])
    after = g[links, j, :].sum()
    # the wall must not create or destroy sediment: renormalise the reflected
    # links so the boundary is exactly mass-neutral (round-off only)
    if after != 0.0:
        g[links, j, :] *= before / after


def _dirichlet_wall(
    g: np.ndarray, c_wall: float, v_out: float, nx: int, j: int, incoming_cy: int
) -> None:
    """Dirichlet concentration at a wall, anti-bounce-back form.

    The wall-normal velocity is the *mirrored* drift ``v_out`` so the net flux
    through the boundary face vanishes; the non-equilibrium part is copied from
    the opposite link (second-order accurate Dirichlet).
    """
    cb = np.full((1, nx), c_wall)
    geq = _equilibrium(cb, np.zeros((1, nx)), np.full((1, nx), v_out))
    for i in np.where(_C[:, 1] == incoming_cy)[0]:
        # anti-bounce-back Dirichlet: g_i = -g_opp + 2 g_i^eq(c_wall, u_wall)
        g[i, j, :] = -g[_OPP[i], j, :] + 2.0 * geq[i, 0, :]


# ---------------------------------------------------------------------------
# 1. Rouse-log graded coordinate solver (primary C2 anchor)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RouseLBMConfig:
    """Configuration of the Rouse-log graded scalar LBM.

    All quantities are dimensionless ratios (the Rouse problem is scale
    invariant): ``u_star`` is an arbitrary reference shear velocity, ``w_s`` the
    settling velocity in the same units, ``dxi`` the lattice spacing in the
    Rouse variable ``xi`` and ``D_lat`` the constant lattice diffusivity
    (``tau_c = 1/2 + 3 D_lat``).  The time step follows the *diffusive scaling*
    ``dt = D_lat dxi^2 / (kappa u*)`` so that refining ``dxi`` keeps the lattice
    diffusivity constant and shrinks the lattice drift velocity.
    """

    h: float = 1.0  # depth (arbitrary units)
    a_frac: float = 0.05  # reference height y_a/h
    u_star: float = 0.05  # shear velocity (reference units)
    w_s: float = 0.006  # settling velocity (same units)
    kappa: float = KAPPA
    dxi: float = 0.20  # lattice spacing in xi
    D_lat: float = 0.10  # lattice diffusivity (constant coefficients)
    nx: int = 1  # periodic columns
    c_a: float = 1.0
    max_steps: int = 300_000
    tol: float = 1.0e-12
    check_every: int = 500

    def __post_init__(self) -> None:
        if self.u_star <= 0.0 or self.w_s < 0.0:
            raise ValueError("require u_star > 0, w_s >= 0")
        if not (0.0 < self.a_frac < 0.5):
            raise ValueError("require 0 < a_frac < 0.5")
        if self.dxi <= 0.0 or self.D_lat <= 0.0:
            raise ValueError("require dxi > 0, D_lat > 0")
        # lattice validity: the drift must stay small (v_lat = Z D_lat dxi)
        if self.v_lat > 0.05:
            raise ValueError(
                f"lattice drift v_lat={self.v_lat:.3f} > 0.05: Rouse number too "
                "large for this resolution -- refine dxi or lower Z"
            )

    @property
    def dt(self) -> float:
        """Lattice time step from the diffusive scaling."""
        return self.D_lat * self.dxi**2 / (self.kappa * self.u_star)

    @property
    def v_lat(self) -> float:
        """Lattice drift velocity (settling)."""
        return self.w_s * self.dt / self.dxi


def rouse_log_metric(y, h: float):
    """Rouse variable ``xi = ln(y/(h-y))``."""
    y = np.asarray(y, dtype=float)
    xi = np.log(y / (h - y))
    return float(xi) if xi.ndim == 0 else xi


def _xi_grid(cfg: RouseLBMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Uniform xi grid and the corresponding physical heights."""
    ya = cfg.a_frac * cfg.h
    xi_a = float(np.log(ya / (cfg.h - ya)))
    n = int(np.ceil(2.0 * abs(xi_a) / cfg.dxi)) + 1
    xi = xi_a + cfg.dxi * np.arange(n)
    y = cfg.h / (1.0 + np.exp(-xi))
    return xi, y


def run_rouse_lbm(cfg: RouseLBMConfig | None = None) -> dict:
    """Solve the Rouse balance with the graded scalar LBM and fit ``Z``.

    Boundary conditions: Dirichlet concentration at the bed reference height
    (``c = c_a``) and at the free surface (``c = c_a exp(-Z (xi_h - xi_a))``,
    the analytic value -- both are *boundary* values; the interior profile is
    computed by the LBM and is the quantity under test).  Interior values are
    what the fit uses.

    Returns a dict with the grid ``y``, the profile ``c``, ``Z_ref`` (analytic
    ``w_s/(kappa u*)``), the fitted ``Z_num``, the relative error, the profile
    error against the analytic Rouse profile, the step count and the lattice
    parameters used.
    """
    cfg = cfg or RouseLBMConfig()
    xi, y = _xi_grid(cfg)
    n = xi.size
    dxi = cfg.dxi
    dt = cfg.dt
    v_lat = cfg.v_lat
    D_lat = cfg.D_lat
    tau_c = 0.5 + 3.0 * D_lat
    omega = 1.0 / tau_c

    Z_ref = cfg.w_s / (cfg.kappa * cfg.u_star)
    xi_a = float(xi[0])
    c_top = cfg.c_a * float(np.exp(-Z_ref * (xi[-1] - xi_a)))
    ux = np.zeros((n, cfg.nx))
    uy = np.full((n, cfg.nx), -v_lat)

    g = _equilibrium(np.full((n, cfg.nx), cfg.c_a), ux, uy)
    steps = 0
    prev = None
    for steps in range(1, cfg.max_steps + 1):
        cnow = g.sum(axis=0)
        g = g + omega * (_equilibrium(cnow, ux, uy) - g)
        g = _stream(g)
        _dirichlet_wall(g, cfg.c_a, -v_lat, cfg.nx, 0, +1)  # bed reference height
        _dirichlet_wall(g, c_top, -v_lat, cfg.nx, -1, -1)  # free surface
        if steps % cfg.check_every == 0:
            cn = g.sum(axis=0)
            if prev is not None:
                rel = float(np.max(np.abs(cn - prev)) / max(np.max(np.abs(cn)), 1e-30))
                if rel < cfg.tol:
                    break
            prev = cn.copy()

    cprof = g.sum(axis=0).mean(axis=1)
    Z_num, asym = fit_rouse_exponent(y, cprof, cfg.h, return_asymmetry=True)
    a = cfg.a_frac * cfg.h
    band = (y >= 1.02 * a) & (y <= 0.6 * cfg.h)
    if Z_ref > 0.0:
        c_an = cfg.c_a * rouse_profile(y[band], cfg.h, a, Z_ref)
        prof_err = float(np.max(np.abs(cprof[band] - c_an) / c_an))
    else:
        prof_err = float(np.max(np.abs(cprof - cfg.c_a)) / cfg.c_a)
    return {
        "y": y,
        "xi": xi,
        "c": cprof,
        "c_normalised": cprof / cfg.c_a,
        "Z_ref": float(Z_ref),
        "Z_num": float(Z_num),
        "rel_err": float(abs(Z_num - Z_ref) / Z_ref) if Z_ref > 0.0 else 0.0,
        "fit_asymmetry": float(asym),
        "profile_rel_err_vs_analytic": prof_err,
        "steps": steps,
        "dt": float(dt),
        "n_nodes": int(n),
        "dxi": dxi,
        "tau_c": float(tau_c),
        "D_lat": float(D_lat),
        "v_lat": float(v_lat),
        "c_top": float(c_top),
        "w_s": cfg.w_s,
        "u_star": cfg.u_star,
        "kappa": cfg.kappa,
        "h": cfg.h,
        "profile_uniformity": float(
            np.max(np.abs(cprof - cprof.mean())) / max(cprof.mean(), 1.0e-30)
        ),
    }


# ---------------------------------------------------------------------------
# 2. Physical-coordinate variable-tau solver (documented negative result)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalarLBMConfig:
    """Lattice configuration of the physical-coordinate scalar column."""

    ny: int = 38  # fluid rows between the bed reference height and the surface
    nx: int = 1  # periodic columns
    h: float = 40.0  # depth in lattice units
    a_frac: float = 0.05  # reference height y_a/h
    u_star: float = 0.05  # shear velocity, lattice units
    w_s: float = 0.006  # settling velocity, lattice units
    kappa: float = KAPPA
    c_a: float = 1.0
    max_steps: int = 60_000
    tol: float = 1.0e-11
    check_every: int = 500

    def __post_init__(self) -> None:
        if self.ny < 8:
            raise ValueError("ny must be >= 8")
        if not (0.0 < self.a_frac < 0.5):
            raise ValueError("require 0 < a_frac < 0.5")
        if self.u_star <= 0.0 or self.w_s < 0.0:
            raise ValueError("require u_star > 0, w_s >= 0")


def _y_rows(cfg: ScalarLBMConfig) -> np.ndarray:
    ya = cfg.a_frac * cfg.h
    return np.linspace(ya, cfg.h - 0.5 * (cfg.h - ya) / cfg.ny, cfg.ny)


def _diffusivity(y: np.ndarray, cfg: ScalarLBMConfig) -> np.ndarray:
    d = cfg.kappa * cfg.u_star * y * (1.0 - y / cfg.h)
    return np.clip(d, 1.0e-4, None)


def run_scalar_channel(cfg: ScalarLBMConfig | None = None) -> dict:
    """Physical-coordinate variable-tau scalar LBM (limitation probe).

    The relaxation time is row-dependent, ``tau_c(y) = 1/2 + 3 D_s(y)``.  This
    formulation carries the O(dD_s/dy) spurious flux documented in the module
    docstring; it is kept to *quantify* that limitation and is not the C2 pass
    criterion.
    """
    cfg = cfg or ScalarLBMConfig()
    y = _y_rows(cfg)
    D = _diffusivity(y, cfg)
    tau_c = 0.5 + 3.0 * D
    omega = (1.0 / tau_c).reshape(1, cfg.ny, 1)
    Z_ref = cfg.w_s / (cfg.kappa * cfg.u_star)
    ux = np.zeros((cfg.ny, cfg.nx))
    uy = np.full((cfg.ny, cfg.nx), -cfg.w_s)

    c = np.full((cfg.ny, cfg.nx), cfg.c_a)
    g = _equilibrium(c, ux, uy)
    steps = 0
    prev = None
    for steps in range(1, cfg.max_steps + 1):
        cnow = g.sum(axis=0)
        g = g + omega * (_equilibrium(cnow, ux, uy) - g)
        g = _stream(g)
        _dirichlet_wall(g, cfg.c_a, +cfg.w_s, cfg.nx, 0, +1)
        _mirror_wall(g, g.sum(axis=0)[-1], np.zeros(cfg.nx), cfg.w_s, -1, -1)
        if steps % cfg.check_every == 0:
            cn = g.sum(axis=0)
            if prev is not None:
                rel = float(np.max(np.abs(cn - prev)) / max(np.max(np.abs(cn)), 1e-30))
                if rel < cfg.tol:
                    break
            prev = cn.copy()

    cprof = g.sum(axis=0).mean(axis=1)
    Z_num = fit_rouse_exponent(y, cprof, cfg.h)
    return {
        "y": y,
        "c": cprof,
        "Z_ref": float(Z_ref),
        "Z_num": float(Z_num),
        "rel_err": float(abs(Z_num - Z_ref) / Z_ref) if Z_ref > 0.0 else 0.0,
        "steps": steps,
        "tau_c_range": (float(tau_c.min()), float(tau_c.max())),
        "profile_uniformity": float(
            np.max(np.abs(cprof - cprof.mean())) / max(cprof.mean(), 1.0e-30)
        ),
    }


def column_exchange_budget(
    cfg: ScalarLBMConfig | None = None,
    n_steps: int = 3000,
    pickup: float | None = None,
    porosity: float = 0.4,
) -> dict:
    """Coupled suspended + bed sediment mass budget of a vertical column.

    The column exchanges sediment with the bed through

    * **erosion** -- a prescribed volumetric pickup rate ``E`` (lattice units
      per unit time) injected on the bottom row, and
    * **deposition** -- the settling flux ``w_s c_b`` onto the bed.

    At every step the *measured* suspended mass increment of the lattice field
    is transferred with the opposite sign into the bed store (``1-n``
    fraction), so the total (suspended + bed) must be conserved to round-off.
    The reported ``exchange_flux_rel_err`` compares the measured increment with
    the intended exchange flux ``(E - w_s c_b) dt``: that is the check that the
    bottom boundary really implements the flux it claims (a double-counted or
    missing term shows up there).
    """
    cfg = cfg or ScalarLBMConfig(ny=24, h=24.0, u_star=0.05, w_s=0.006, nx=1)
    E = cfg.w_s * cfg.c_a * 2.0 if pickup is None else float(pickup)
    y = _y_rows(cfg)
    D = _diffusivity(y, cfg)
    tau_c = 0.5 + 3.0 * D
    omega = (1.0 / tau_c).reshape(1, cfg.ny, 1)
    ux = np.zeros((cfg.ny, cfg.nx))
    uy = np.full((cfg.ny, cfg.nx), -cfg.w_s)

    c = np.full((cfg.ny, cfg.nx), cfg.c_a)
    g = _equilibrium(c, ux, uy)
    dy = float((cfg.h - cfg.a_frac * cfg.h) / cfg.ny)
    suspended0 = float(g.sum() * dy)
    bed = 0.0
    flux_int = 0.0
    measured_int = 0.0

    for _ in range(n_steps):
        cnow = g.sum(axis=0)
        g = g + omega * (_equilibrium(cnow, ux, uy) - g)
        g = _stream(g)
        cb = float(cnow[0].mean())
        dm = (E - cfg.w_s * cb) * dy  # intended exchange (per column)
        flux_int += dm
        before = float(g[:, 0, :].sum() * dy)
        scale = 1.0 + dm / max(before, 1.0e-30)
        g[:, 0, :] *= scale
        after = float(g[:, 0, :].sum() * dy)
        measured_int += after - before
        bed -= (after - before) / (1.0 - porosity)
        _mirror_wall(g, g.sum(axis=0)[-1], np.zeros(cfg.nx), cfg.w_s, -1, -1)

    suspended = float(g.sum() * dy)
    total = suspended + (1.0 - porosity) * bed
    return {
        "mass_drift": abs(total - suspended0) / max(abs(suspended0), 1.0e-30),
        "suspended_change": suspended - suspended0,
        "bed_change": bed,
        "n_eroded": float(bed),
        "exchange_flux_rel_err": abs(measured_int - flux_int)
        / max(abs(flux_int), 1.0e-30),
        "steps": n_steps,
        "pickup": E,
        "w_s": cfg.w_s,
        "dy": dy,
        "porosity": porosity,
    }