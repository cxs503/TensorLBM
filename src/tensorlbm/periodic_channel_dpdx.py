"""D2Q9 plane channel driven by a **constant pressure gradient** with **x-periodic**
boundaries — the single-phase driving mechanism behind the OSS ``single-phase-
periodic-x-plus-constant-dpdx-driving`` tag (``joconnor22/LIFE``).

Why this module exists
----------------------
The repository has body-force channel kernels (``channel_kernel_numpy`` is the
D3Q19 turbulent-channel kernel; ``bitflag_d2q9`` is the D2Q9 boundary-treatment
XVAL solver) but neither offers the *combination* that a pressure-driven
periodic channel needs, and neither exposes the two closures that decide the
answer at finite resolution:

* **wall closure** — halfway bounce-back at the link midpoint (repo convention,
  ``bitflag_d2q9`` docstring: ``f_i <- f_opp(i)`` at the fluid node, wall at
  ``y = 1/2``) versus the *node-based regularised closure* of Latt et al. (2008)
  used by the OSS reference (its wall lives at the wall-node centre);
* **force injection** — the split Guo source used by the production wall kernel
  (``channel_kernel_numpy`` docstring: force injected post-collision, full
  weight, no half-step factor, ``u = sum(c f)/rho``) versus the textbook Guo
  form with the ``(1 - omega/2)`` weight and the ``+F/2`` velocity correction.

Both closures are implemented here so that the difference can be *measured*
instead of argued about (see ``benchmarks/cross_validation/periodic_channel_dpdx`` and the round-2
report in ``docs/reports/XVAL_LBM_PERIODIC_X_CONSTANT_DPDX.md``).

Physics / discretisation
------------------------
D2Q9, second-order equilibrium, BGK with ``omega = 1/tau``, ``c_s^2 = 1/3``.
Solid walls in ``y``; ``x`` is periodic.  A constant body force density ``F``
(= ``dp/dx``) is applied to every lattice node, decomposing as
``nu u''(y) = -G`` with ``G = F / rho_0``.

The steady state is the exact plane-Poiseuille parabola

.. math:: u(y) = G (y - y_w) (H - (y - y_w)) / (2 nu),  H = y_{w,top} - y_{w,bot}

sampled on the wall-normalised coordinate ``s = (y - y_w)/H``.  **Both** the wall
origin and the height depend on the closure for a *fixed* ``ny``: halfway
bounce-back puts the walls on the link midpoints (``0.5`` and ``ny - 1.5``,
``H = ny - 2``) while the node-based regularised closure puts them on the
wall-node centres (``0`` and ``ny - 1``, ``H = ny - 1``).  That is a genuine
``O(dx)`` geometry offset, not a discretisation error, and it is why the xval
runs a matched-height pair (``ny`` and ``ny + 1``) as well as the matched-grid
pair.

Round-2 defect history (2026-11-15)
-----------------------------------
Round 1 of the xval left the two ported arms on an honest FAIL: both blew up
after a few thousand steps.  Two transcription defects were the cause and are
fixed here; ``tests/test_xval_life_periodic_dpdx.py`` now pins both.

1. ``_shift`` streamed with the *opposite* sign (``out[x] = a[x + c]`` instead
   of ``out[x] = a[x - c]``), i.e. every population travelled backwards and the
   wall closures were fed from the wrong neighbour row.
2. ``equilibrium`` used ``1 + 3 cu + 1.5 cu^2 - 1.5 u^2`` instead of the
   second-order Hermite form ``1 + 3 cu + 4.5 cu^2 - 1.5 u^2``.  The wrong
   quadratic coefficient breaks the equilibrium's own moments
   (``sum f = 0.9975 rho`` and ``sum c^2 f = 0.6658 rho`` for ``u = 0.05``), so
   the effective sound speed and the mass are both off; the arm then diverges
   exponentially (density growth ~1 %/100 steps measured in round 1).

Unit conversion
---------------
``dpdx_to_lattice_force`` converts a physical pressure gradient [Pa/m] to the
lattice force density.  Two conventions are offered because the OSS source and
the standard LBM scaling disagree by ``1/Dx`` (see the xval report): the
``"standard"`` convention is ``F = dpdx * Dt^2 / (rho_scale * Dx)`` and
``"life_compat"`` reproduces ``LIFE/src/Grid.cpp:1042`` verbatim
(``F = dpdx * Dx * Dt^2 / Dm`` with ``Dm = rho_scale * Dx^3``).

Self-contained (NumPy only) so it runs on the loong64 nodes without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "D2Q9_C",
    "D2Q9_W",
    "D2Q9_OPP",
    "CS2",
    "WALL_HALFWAY_BB",
    "WALL_REGULARISED",
    "WALL_NONE",
    "FORCE_SPLIT_GUO",
    "FORCE_GUO_HALF_FROZEN",
    "DpdxChannel",
    "dpdx_to_lattice_force",
    "life_force_xy",
    "poiseuille_analytic",
    "poiseuille_transient",
    "observed_order",
    "fit_parabola",
]

# ---------------------------------------------------------------------------
# D2Q9 lattice — same table/order as ``tensorlbm.d2q9`` (verified by
# ``tests/test_xval_life_periodic_dpdx.py``); duplicated as plain tuples so the
# module stays NumPy-only (``tensorlbm.d2q9`` imports torch at module level).
# ---------------------------------------------------------------------------
D2Q9_C: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)
D2Q9_W: tuple[float, ...] = (
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)
D2Q9_OPP: tuple[int, ...] = (0, 3, 4, 1, 2, 7, 8, 5, 6)
CS2 = 1.0 / 3.0

WALL_HALFWAY_BB = "halfway_bb"
WALL_REGULARISED = "regularised"
WALL_NONE = "none"

FORCE_SPLIT_GUO = "split_guo"
FORCE_GUO_HALF_FROZEN = "guo_half_frozen"

_CX = np.array([c[0] for c in D2Q9_C], dtype=np.float64)
_CY = np.array([c[1] for c in D2Q9_C], dtype=np.float64)
_CXI = np.array([c[0] for c in D2Q9_C], dtype=np.int64)
_CYI = np.array([c[1] for c in D2Q9_C], dtype=np.int64)
_W = np.array(D2Q9_W, dtype=np.float64)
_OPP = np.array(D2Q9_OPP, dtype=np.int64)


def _shift(a: np.ndarray, d0: int, d1: int) -> np.ndarray:
    """Pull streaming for lattice direction ``d = (d0, d1)``.

    ``out[x, y] = a[x - d0, y - d1]`` — a population that belongs to direction
    ``d`` at ``(x, y)`` after the step comes from ``(x, y) - d`` *before* it
    (``p[recv] = p[recv - d]``, the pull form of ``Grid.cpp:229``).  The wrap is
    periodic in both axes; the ``y`` wrap only ever touches the two wall rows,
    whose entries are rewritten by the wall closure right afterwards
    (``WALL_HALFWAY_BB``) or rebuilt from the moments (``WALL_REGULARISED``),
    so no information crosses the walls.
    """
    return np.roll(a, shift=(d0, d1), axis=(0, 1))


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order D2Q9 equilibrium, shape ``(9, nx, ny)`` (vectorised).

    ``w_i rho [1 + 3 (c_i.u) + 4.5 (c_i.u)^2 - 1.5 u^2]`` — the Hermite form
    that reproduces ``sum_i f^eq = rho``, ``sum_i c_i f^eq = rho u`` and
    ``sum_i c_i^2 f^eq = 2 rho c_s^2`` exactly (asserted in the test file).
    """
    cu = _CX[:, None, None] * ux[None] + _CY[:, None, None] * uy[None]
    usqr = 1.5 * (ux * ux + uy * uy)
    return _W[:, None, None] * rho[None] * (1.0 + cu * (3.0 + 4.5 * cu) - usqr[None])


def guo_source(u: np.ndarray, force: np.ndarray, weight: float) -> np.ndarray:
    """Guo (2002) source ``S_i = weight * 3 w_i [(c-u) + 3 c (c.u)].F`` (vectorised)."""
    ux, uy = u[0], u[1]
    fx, fy = force[0], force[1]
    cu = _CX[:, None, None] * ux[None] + _CY[:, None, None] * uy[None]
    s = (_CX[:, None, None] - ux[None] + 3.0 * _CXI[:, None, None] * cu) * fx[None] + (
        _CY[:, None, None] - uy[None] + 3.0 * _CYI[:, None, None] * cu) * fy[None]
    return (weight * 3.0) * _W[:, None, None] * s


def poiseuille_analytic(
    ny: int,
    nu: float,
    g: float,
    *,
    wall: str = WALL_HALFWAY_BB,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact steady plane-Poiseuille profile on the wall-normalised coordinate.

    Returns ``(s, u, y_w, H)`` with ``s = (y - y_w)/H`` on the fluid rows
    ``y = 1 .. ny-2`` and ``u = 4 u_max s (1 - s)``,
    ``u_max = g H^2 / (8 nu)``.

    ``wall`` selects the wall origin convention: ``1/2`` for halfway
    bounce-back (link midpoint) and ``0`` for the node-based regularised
    closure (wall-node centre).
    """
    y = np.arange(1, ny - 1, dtype=np.float64)
    # halfway bounce-back: the two solid rows cost one half cell at each end,
    # so the wall planes are y = 1/2 and y = ny - 3/2 and H = ny - 2.
    # the node-based regularised closure puts the walls on the wall-node
    # centres y = 0 and y = ny - 1 and H = ny - 1.
    h = float(ny - 2) if wall == WALL_HALFWAY_BB else float(ny - 1)
    y_w = 0.5 if wall == WALL_HALFWAY_BB else 0.0
    s = (y - y_w) / h
    u_max = g * h * h / (8.0 * nu)
    return s, 4.0 * u_max * s * (1.0 - s), y_w, h


def poiseuille_transient(
    ny: int,
    nu: float,
    g: float,
    t: float,
    *,
    wall: str = WALL_HALFWAY_BB,
    n_modes: int | None = None,
    tol: float = 1e-14,
    max_modes: int = 200_001,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Exact start-up solution of the forced plane channel (from rest).

    The channel is at rest and a constant body-force density ``g`` (= the
    lattice acceleration, ``nu u'' = -g``) is switched on at ``t = 0+``.  The
    classical Fourier-series solution is

    .. math::

        u(s,t) = u_\infty(s)
                 - \frac{4 g H^2}{\nu \pi^3}
                   \sum_{k=0}^{\infty} \frac{\sin((2k+1)\pi s)}{(2k+1)^3}
                   e^{-(2k+1)^2 \pi^2 \nu t / H^2},

    with the steady parabola ``u_\infty = g H^2 s (1-s) / (2 nu)`` on
    ``s = (y - y_w)/H``.  It is the transient counterpart of
    :func:`poiseuille_analytic` and the only benchmark that exercises the
    *time* accuracy of the force injection (the steady benchmark cannot see an
    ``O(\Delta t)`` offset in the reported velocity -- see the round-4 report
    ``docs/reports/XVAL_LBM_PERIODIC_X_CONSTANT_DPDX_R4.md``).

    Returns ``(s, u, y_w, H, n_modes)``: the wall-normalised coordinate and
    profile on the fluid rows ``y = 1 .. ny-2``, the closure's wall origin and
    height (same conventions as :func:`poiseuille_analytic`), and the number of
    odd Fourier modes actually summed.

    Parameters
    ----------
    n_modes:
        Fixed number of odd modes; by default the series is summed until the
        largest remaining term is below ``tol`` times the steady peak.
        ``t`` must be positive -- at ``t = 0`` the exact value is ``u = 0``
        everywhere (the series is only conditionally convergent there), so the
        analytic value is returned early rather than summed.
    """
    if not t > 0.0:
        raise ValueError(
            f"transient solution needs t > 0 (the exact value at t = 0 is u = 0); got t={t!r}")
    if not nu > 0.0:
        raise ValueError(f"nu must be positive, got {nu!r}")
    h = float(ny - 2) if wall == WALL_HALFWAY_BB else float(ny - 1)
    y_w = 0.5 if wall == WALL_HALFWAY_BB else 0.0
    y = np.arange(1, ny - 1, dtype=np.float64)
    s = (y - y_w) / h
    u_ss = g * h * h / (2.0 * nu) * s * (1.0 - s)
    u = u_ss.copy()
    amp = 4.0 * g * h * h / (nu * np.pi ** 3)
    if amp == 0.0:
        return s, u, y_w, h, 0
    tau_d = nu * t / (h * h)          # non-dimensional diffusive time
    scale = float(np.max(np.abs(u_ss))) / abs(amp)   # tolerance in series units
    limit = int(n_modes) if n_modes is not None else int(max_modes)
    used = 0
    for k in range(limit):
        n = 2 * k + 1
        term = np.sin(n * np.pi * s) / n ** 3 * np.exp(-(n * n * np.pi ** 2) * tau_d)
        u -= amp * term
        used = k + 1
        if n_modes is None and float(np.max(np.abs(term))) < tol * scale:
            break
    else:  # loop exhausted
        if n_modes is None:
            raise ValueError(
                "transient series did not reach tol before max_modes: "
                f"t={t!r} is too small for nu={nu!r}, H={h}; the series needs "
                f"O((nu t / H^2)^-1/2) modes")
    return s, u, y_w, h, used


def dpdx_to_lattice_force(
    dpdx: float,
    *,
    dx: float,
    dt: float,
    rho_scale: float = 1.0,
    convention: str = "standard",
) -> float:
    """Convert a physical pressure gradient [Pa/m] to a lattice force density.

    ``"standard"``
        ``F = dpdx * dt^2 / (rho_scale * dx)`` — momentum scaling
        ``F_p / (rho_scale * dx/dt^2)`` with the 2-D cell (per unit thickness)
        force ``dpdx * dx^2``; this is the value that reproduces
        ``u_max = (dpdx/rho_p) H_p^2 / (8 nu_p)``.

    ``"life_compat"``
        ``F = dpdx * dx * dt^2 / Dm`` with ``Dm = rho_scale * dx^3``, i.e. the
        OSS expression ``LIFE/src/Grid.cpp:1042`` verbatim.  It equals the
        standard value divided by ``dx`` (derived in the xval report).
    """
    if convention == "standard":
        return dpdx * dt * dt / (rho_scale * dx)
    if convention == "life_compat":
        return life_force_xy(dpdx, dx=dx, dt=dt, rho_scale=rho_scale)
    raise ValueError(f"unknown convention {convention!r}")


def life_force_xy(
    dpdx: float,
    *,
    dx: float,
    dt: float,
    rho_scale: float = 1.0,
    gravity: float = 0.0,
    rho: float = 1.0,
) -> float:
    """Verbatim transcription of ``LIFE/src/Grid.cpp:1042`` (zero-Womersley branch).

    ``force_xy = (rho * Drho * gravityX + dpdx) * Dx * Dt^2 / Dm`` with
    ``Dm = (rho_p/rho0) * Dx^3`` and ``Drho = rho_p/rho0`` (``Grid.cpp:1257-1260``).
    """
    dm = rho_scale * dx ** 3
    return (rho * rho_scale * gravity + dpdx) * dx * dt * dt / dm


@dataclass
class DpdxChannel:
    """Periodic-x D2Q9 plane channel driven by a constant body force density.

    Parameters
    ----------
    nx, ny:
        Lattice size.  ``x`` is periodic (the state is x-invariant, so ``nx``
        only costs memory); rows ``y = 0`` and ``y = ny - 1`` are the walls.
    tau:
        BGK relaxation time, ``nu = (tau - 1/2)/3``.
    force:
        Lattice body-force density (``rho_lb * lattice acceleration``).
    wall:
        ``"halfway_bb"`` (link midpoint, default — repo convention),
        ``"regularised"`` (Latt et al. 2008 node closure, OSS convention) or
        ``"none"`` (fully periodic in y; defect-injection control lane).
    force_scheme:
        ``"split_guo"`` (post-collision, full weight, ``u = sum(c f)/rho``;
        repo production convention) or ``"guo_half_frozen"``
        (``(1 - omega/2)`` weight, ``+F/2`` velocity correction, force frozen
        at ``rho = rho0``; OSS convention).
    """

    nx: int
    ny: int
    tau: float
    force: float = 0.0
    wall: str = WALL_HALFWAY_BB
    force_scheme: str = FORCE_SPLIT_GUO
    rho0: float = 1.0
    u_init: np.ndarray | None = None

    f: np.ndarray = field(init=False)
    rho: np.ndarray = field(init=False)
    ux: np.ndarray = field(init=False)
    uy: np.ndarray = field(init=False)
    steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.wall not in (WALL_HALFWAY_BB, WALL_REGULARISED, WALL_NONE):
            raise ValueError(f"unknown wall closure {self.wall!r}")
        if self.force_scheme not in (FORCE_SPLIT_GUO, FORCE_GUO_HALF_FROZEN):
            raise ValueError(f"unknown force scheme {self.force_scheme!r}")
        if self.ny < 4:
            raise ValueError("ny >= 4 required")
        self._omega = 1.0 / float(self.tau)
        self.rho = np.full((self.nx, self.ny), float(self.rho0))
        if self.u_init is None:
            self.ux = np.zeros((self.nx, self.ny))
            self.uy = np.zeros((self.nx, self.ny))
        else:
            self.ux = np.asarray(self.u_init[0], dtype=np.float64).copy()
            self.uy = np.asarray(self.u_init[1], dtype=np.float64).copy()
        self.f = equilibrium(self.rho, self.ux, self.uy)

    # -- properties --------------------------------------------------------
    @property
    def nu(self) -> float:
        return (1.0 / self._omega - 0.5) / 3.0

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def h(self) -> float:
        """Channel height in lattice units for this closure (see module docs)."""
        return float(self.ny - 2) if self.wall == WALL_HALFWAY_BB else float(self.ny - 1)

    @property
    def y_wall(self) -> float:
        return 0.5 if self.wall == WALL_HALFWAY_BB else 0.0

    @property
    def fluid_rows(self) -> np.ndarray:
        return np.arange(1, self.ny - 1)

    @property
    def is_fluid(self) -> np.ndarray:
        mask = np.zeros((self.nx, self.ny), dtype=bool)
        mask[:, 1:-1] = True
        return mask

    @property
    def force_vector(self) -> np.ndarray:
        """Body-force density field, shape ``(2, nx, ny)``.

        ``split_guo`` recomputes it from the instantaneous density every step
        (the source is a *density*, so ``F = rho0 * G`` stays constant — the
        point is that it is not re-scaled by ``rho``); ``guo_half_frozen``
        freezes it at initialisation from ``rho = rho0``, which is what the OSS
        code does (``Grid.cpp:1042-1043`` is outside the time loop).
        """
        out = np.zeros((2, self.nx, self.ny))
        out[0] = self.force
        return out

    # -- operators ---------------------------------------------------------
    def macros(self) -> None:
        rho = self.f.sum(axis=0)
        mx = np.tensordot(_CX, self.f, axes=(0, 0))
        my = np.tensordot(_CY, self.f, axes=(0, 0))
        if self.force_scheme == FORCE_GUO_HALF_FROZEN:
            mx = mx + 0.5 * self.force
        self.rho = rho
        self.ux = mx / rho
        self.uy = my / rho

    def _collide(self) -> np.ndarray:
        u = np.stack([self.ux, self.uy])
        feq = equilibrium(self.rho, self.ux, self.uy)
        f_post = self.f + self._omega * (feq - self.f)
        weight = 1.0 if self.force_scheme == FORCE_SPLIT_GUO else (1.0 - 0.5 * self._omega)
        f_post = f_post + guo_source(u, self.force_vector, weight)
        if self.wall == WALL_HALFWAY_BB:
            m = self.is_fluid[None]
            f_post = np.where(m, f_post, self.f)
        return f_post

    def _stream(self, f_post: np.ndarray) -> np.ndarray:
        out = np.empty_like(f_post)
        for i in range(9):
            out[i] = _shift(f_post[i], _CXI[i], _CYI[i])
        if self.wall == WALL_HALFWAY_BB:
            # halfway bounce-back at the link midpoint of the two solid rows
            for i in range(9):
                # pull form: node y=1 pulls direction i from y=1-cy_i; the link
                # crosses the wall for cy_i = +1 (bottom) and cy_i = -1 (top).
                if _CYI[i] == 1:   # bottom link closed -> halfway bounce-back
                    out[i][:, 1] = f_post[D2Q9_OPP[i]][:, 1]
                elif _CYI[i] == -1:  # top link closed
                    out[i][:, -2] = f_post[D2Q9_OPP[i]][:, -2]
        elif self.wall == WALL_REGULARISED:
            out = self._regularised_wall(out)
        return out

    def _regularised_wall(self, f_streamed: np.ndarray) -> np.ndarray:
        """Latt et al. (2008) regularised closure on the two wall-node rows.

        Transcribed from ``LIFE/src/Grid.cpp`` ``regularisedBC`` (the axis
        aligned, non-corner branch): ``rho = 2 f+ + f0`` from the incoming and
        tangential populations, unknown directions closed by the
        non-equilibrium bounce-back ``f_i = feq_i + (f_opp - feq_opp)``, then
        *all* directions rebuilt from ``feq + (w_i / 2 c_s^4) [(cx^2-cs^2) Sxx +
        (cy^2-cs^2) Syy + 2 cx cy Sxy]``.
        """
        f = f_streamed.copy()
        for row, ny_dir in ((0, +1), (self.ny - 1, -1)):
            sel_in = _CYI == -ny_dir          # c . n < 0 : incoming
            sel_zero = _CYI == 0              # tangential
            sel_unk = _CYI == ny_dir          # c . n > 0 : unknown (from the solid)
            fplus = f[sel_in, :, row].sum(axis=0)
            fzero = f[sel_zero, :, row].sum(axis=0)
            rho_w = 2.0 * fplus + fzero       # u_wall . n = 0
            feq = _W[:, None] * rho_w[None, :]        # equilibrium at u = 0
            # non-equilibrium bounce-back for the unknown directions
            f[sel_unk, :, row] = feq[sel_unk] + (
                f[_OPP[sel_unk], :, row] - feq[_OPP[sel_unk]]
            )
            fneq = f[:, :, row] - feq
            sxx = (_CX * _CX) @ fneq
            syy = (_CY * _CY) @ fneq
            sxy = (_CX * _CY) @ fneq
            coef = (
                (_CX * _CX - CS2)[:, None] * sxx[None, :]
                + (_CY * _CY - CS2)[:, None] * syy[None, :]
                + 2.0 * (_CX * _CY)[:, None] * sxy[None, :]
            )
            f[:, :, row] = feq + (_W[:, None] / (2.0 * CS2 * CS2)) * coef
        return f

    def step(self) -> None:
        f_post = self._collide()
        self.f = self._stream(f_post)
        self.macros()
        self.steps += 1

    # -- diagnostics -------------------------------------------------------
    def profile(self) -> np.ndarray:
        """``u_x`` averaged over ``x`` on the fluid rows (shape ``(ny-2,)``)."""
        return self.ux.mean(axis=0)[1:-1]

    def y_centres(self) -> np.ndarray:
        return np.arange(1, self.ny - 1, dtype=np.float64)

    def analytic(self) -> tuple[np.ndarray, np.ndarray]:
        s, u, _, _ = poiseuille_analytic(self.ny, self.nu, self.force, wall=self.wall)
        return s, u

    def rel_l2_vs_analytic(self) -> float:
        _, u_ana = self.analytic()
        u = self.profile()
        return float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana))

    def u_max(self) -> float:
        return float(np.max(np.abs(self.profile())))

    def mass(self) -> float:
        """Total lattice mass over the fluid rows (per unit thickness)."""
        return float(self.rho[:, 1:-1].sum())

    def run_until_steady(
        self,
        *,
        tol: float = 1e-11,
        max_steps: int = 400_000,
        check_every: int = 200,
    ) -> dict:
        """Advance until the wall-normal profile stops changing.

        The stopping rule is predeclared in ``xval/.../predeclared.json``:
        ``max|u(y,t) - u(y,t-200)| / u_max < tol``.  Diffusive and not
        oscillatory for a body-force-start channel, so no averaging window is
        needed.
        """
        prev = self.profile()
        n = 0
        delta = float("nan")
        while n < max_steps:
            for _ in range(check_every):
                self.step()
            n += check_every
            cur = self.profile()
            scale = max(float(np.max(np.abs(cur))), 1e-30)
            delta = float(np.max(np.abs(cur - prev))) / scale
            prev = cur
            if not np.isfinite(delta):
                return {"converged": False, "delta": float("nan"), "steps": self.steps}
            if delta < tol:
                return {"converged": True, "delta": delta, "steps": self.steps}
        return {"converged": False, "delta": delta, "steps": self.steps}

# ---------------------------------------------------------------------------
# verification utilities (round 3)
# ---------------------------------------------------------------------------
def observed_order(
    err_coarse: float,
    err_fine: float,
    h_coarse: float,
    h_fine: float,
) -> float:
    """Observed order of accuracy ``p = log(e_coarse/e_fine) / log(h_fine/h_coarse)``.

    ``h`` is a *resolution length*: bigger means better resolved.  For a plane
    channel that length is the wall distance (``H = ny - 2`` for halfway
    bounce-back, ``H = ny - 1`` for the node-based regularised closure), **not**
    ``ny`` and not an assumed number of refinement steps.  The round-2 xval
    estimated this order as ``log2(e_17 / e_25)``, i.e. it assumed the
    ``17 -> 25`` grid pair doubles the resolution while ``H`` actually grows by
    ``24/16 = 1.5``; that under-reports a second-order scheme as
    ``p = 1.15`` (see ``docs/reports/XVAL_LBM_PERIODIC_X_CONSTANT_DPDX_R3.md``).

    Raises ``ValueError`` when the errors are not positive or the pair does not
    refine, so a meaningless order cannot silently reach a report.
    """
    e1, e2 = float(err_coarse), float(err_fine)
    h1, h2 = float(h_coarse), float(h_fine)
    if not (e1 > 0.0 and e2 > 0.0):
        raise ValueError(f"errors must be positive, got {e1!r}, {e2!r}")
    if not (h1 > 0.0 and h2 > h1):
        raise ValueError(
            f"h must be a resolution length that grows under refinement: "
            f"h_coarse={h1!r}, h_fine={h2!r}")
    return float(np.log(e1 / e2) / np.log(h2 / h1))


def fit_parabola(u: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares parabola fit to a wall-normal profile.

    Returns the *geometric* quantities a wall closure is judged by, so the wall
    origin is **measured** instead of assumed:
    ``y_wall`` (lower root), ``y_wall_top`` (upper root), ``h`` (root spread),
    ``u_max`` (vertex value), ``u_max_target``-independent, and
    ``rms_residual``.  A closure whose convention is right fits with
    ``y_wall`` and ``h`` equal to its nominal values to the discretisation
    error, and the residual then measures the *non-parabolic* part of the
    discrete solution.
    """
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if u.shape != y.shape or u.size < 4:
        raise ValueError("need matching arrays with >= 4 samples")
    a, b, c = np.polyfit(y, u, 2)
    if not a < 0.0:
        raise ValueError(f"profile is not concave (a={a!r}); no channel solution")
    # curvature significance: a genuine channel profile has
    # |a| span^2 ~ 4 u_max, a flat/noisy one has |a| span^2 / max|u| ~ 1e-16.
    # Without this a nearly flat array still yields "roots" ~1e8 cells away.
    span = float(y.max() - y.min())
    scale = float(np.max(np.abs(u)))
    if scale <= 0.0 or -a * span * span < 1e-6 * scale:
        raise ValueError(
            "profile curvature is not significant "
            f"(|a|*span^2/|u|={-a * span * span / max(scale, 1e-300):.2e})")
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        raise ValueError(f"no real wall roots (disc={disc!r})")
    roots = np.sort((-b + np.array([-1.0, 1.0]) * np.sqrt(disc)) / (2.0 * a))
    y_bot, y_top = float(roots[0]), float(roots[1])
    y_vertex = -b / (2.0 * a)
    u_max = float(a * y_vertex ** 2 + b * y_vertex + c)
    resid = u - (a * y * y + b * y + c)
    return {
        "y_wall": y_bot,
        "y_wall_top": y_top,
        "h": y_top - y_bot,
        "y_vertex": float(y_vertex),
        "u_max": u_max,
        "rms_residual": float(np.sqrt(np.mean(resid ** 2))),
        "rel_l2_residual": float(np.linalg.norm(resid) / np.linalg.norm(u)),
    }
