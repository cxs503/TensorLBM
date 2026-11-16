"""Brinkman (Darcy-drag) penalisation for LBM, verified against the exact
Brinkman--Poiseuille channel solution.

Physics
-------
A channel of half-width ``h`` (walls at ``y = +-h``) filled with a homogeneous
porous medium of permeability ``K`` and driven by a constant body acceleration
``g`` along ``x``.  Volume averaging the Stokes equations over the porous
microstructure and closing the drag by Darcy's law gives the Brinkman equation

    nu u''(y) - (nu/K) u(y) + g = 0 ,      u(+-h) = 0                    (1)

(lattice units, ``rho = 1``, ``mu_eff = mu``).  With ``beta = h/sqrt(K)``

    u(y)  = (g K / nu) [ 1 - cosh(y/sqrt(K)) / cosh(beta) ]              (2)
    u_bar = (g K / nu) [ 1 - tanh(beta) / beta ]   = u_D * R(beta)       (2a)
    u_c   = (g K / nu) [ 1 - sech(beta) ]                                (2b)
    tau_w = mu u'(h) = g sqrt(K) tanh(beta)                             (2c)

``u_D = g K / nu`` is the Darcy superficial velocity and ``R(beta) = 1 -
tanh(beta)/beta`` the *shape function* (``R -> 1`` deep Darcy, ``R -> beta^2/3``
in the Stokes limit ``beta -> 0`` where (2) collapses onto the Poiseuille
parabola ``g (h^2 - y^2)/(2 nu)``).

The drag closure used by the solver is the per-unit-mass force

    a(y) = g - (nu/K) u(y)                                              (3)

so the volumetric force is ``F = rho a``; with Guo forcing (Guo, Zheng & Shi
2002, PRE 65, 046308) the lattice-BGK momentum equation reduces to (1) at
steady state.

Discretisation contract (what the reference is compared against)
----------------------------------------------------------------
* streamwise (x) periodicity, walls at ``y = 0.5`` and ``y = ny - 1.5`` from
  half-way bounce-back applied **on the boundary fluid nodes** (post-streaming
  form: the unknown incoming populations of the wall-adjacent fluid node are
  the reflected post-collision populations of that same node);
* fluid rows are ``1 .. ny-2``, so ``H = ny - 2`` and ``h = H/2``;
* the body force (driving + Darcy drag) acts on fluid nodes only;
* the drag is evaluated explicitly on the current velocity.

Explicit-drag stiffness (honest limit)
--------------------------------------
Because ``a`` is explicit, the drag relaxation rate ``lam = nu/K`` must obey
``lam * dt << 1``.  This caps how small ``K`` -- i.e. how thin the Brinkman
layer -- can be resolved: ``K`` has to stay comfortably above ``nu`` in lattice
units.  Deep-Darcy configurations whose screening length is sub-grid need an
implicit/semi-implicit penalisation and are *out of scope* here; they are
reported as such and never claimed.

References
----------
* H. C. Brinkman (1947) Appl. Sci. Res. A1 27.
* K. Vafai & S. Kim (1990) ASME J. Heat Transfer 112 642 (Brinkman channel).
* Z. Guo, C. Zheng, B. Shi (2002) PRE 65 046308 (forcing scheme).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CS2 = 1.0 / 3.0
# D2Q9 ordering: 0 rest, 1 (1,0), 2 (0,1), 3 (-1,0), 4 (0,-1),
#                5 (1,1), 6 (-1,1), 7 (-1,-1), 8 (1,-1)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float64)
# populations leaving the domain downwards (c_y = -1) and their incoming partners
_BOTTOM_OUT = [4, 7, 8]
_BOTTOM_IN = [2, 5, 6]
_TOP_OUT = [2, 5, 6]
_TOP_IN = [4, 7, 8]


# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------


def exact_brinkman_profile(
    accel: float,
    nu: float,
    permeability: float,
    half_width: float,
    n: int = 401,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the exact Brinkman--Poiseuille profile (2), ``y`` from the centre."""
    if permeability <= 0.0:
        raise ValueError("permeability must be > 0")
    if half_width <= 0.0:
        raise ValueError("half_width must be > 0")
    delta = math.sqrt(permeability)
    beta = half_width / delta
    y = np.linspace(-half_width, half_width, n)
    amp = accel * permeability / nu
    u = amp * (1.0 - np.cosh(y / delta) / math.cosh(beta))
    return y, u


def exact_shape_function(permeability: float, half_width: float) -> float:
    """``R(beta) = u_bar / u_Darcy = 1 - tanh(beta)/beta``."""
    beta = half_width / math.sqrt(permeability)
    return 1.0 - math.tanh(beta) / beta


def exact_mean_velocity(accel: float, nu: float, permeability: float,
                        half_width: float) -> float:
    return accel * permeability / nu * exact_shape_function(permeability, half_width)


def exact_centreline_velocity(accel: float, nu: float, permeability: float,
                              half_width: float) -> float:
    beta = half_width / math.sqrt(permeability)
    return accel * permeability / nu * (1.0 - 1.0 / math.cosh(beta))


def exact_wall_shear(accel: float, permeability: float, half_width: float) -> float:
    """``tau_w = mu u'(h)`` with ``mu = rho nu``, ``rho = 1`` (see (2c))."""
    beta = half_width / math.sqrt(permeability)
    return accel * math.sqrt(permeability) * math.tanh(beta)


def sample_exact(accel: float, nu: float, permeability: float,
                 half_width: float, y: float) -> float:
    """Evaluate (2) at a single abscissa ``y`` (measured from the centreline)."""
    delta = math.sqrt(permeability)
    beta = half_width / delta
    return accel * permeability / nu * (1.0 - math.cosh(y / delta) / math.cosh(beta))


def verify_exact_solution(
    accel: float = 1.0e-5,
    nu: float = 0.1,
    permeability: float = 40.96,
    half_width: float = 32.0,
) -> dict[str, float]:
    """Re-derive the reference inside the repo: ODE residual, BCs and limits.

    Every residual is evaluated on the closed form (2) itself, so the reference
    cannot silently drift from the code that consumes it.
    """
    y, u = exact_brinkman_profile(accel, nu, permeability, half_width, n=20001)
    dy = float(y[1] - y[0])
    u_yy = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dy**2
    resid = nu * u_yy - (nu / permeability) * u[1:-1] + accel
    ode_resid = float(np.abs(resid).max()) / accel
    u_scale = max(abs(float(u.max())), 1e-300)
    u_bar_quad = float(np.trapezoid(u, y)) / (2.0 * half_width)
    u_bar_closed = exact_mean_velocity(accel, nu, permeability, half_width)
    h_st = half_width
    u_bar_st = exact_mean_velocity(accel, nu, 1.0e8 * h_st**2, h_st)
    u_bar_poiseuille = accel * h_st**2 / (3.0 * nu)
    k_da = 1.0e-8 * h_st**2
    u_bar_da = exact_mean_velocity(accel, nu, k_da, h_st)
    u_darcy = accel * k_da / nu
    return {
        "ode_residual_rel": ode_resid,
        "bc_left_rel": float(abs(u[0])) / u_scale,
        "bc_right_rel": float(abs(u[-1])) / u_scale,
        "mean_closed_vs_quad_rel": abs(u_bar_closed - u_bar_quad) / abs(u_bar_closed),
        "stokes_limit_rel": abs(u_bar_st - u_bar_poiseuille) / u_bar_poiseuille,
        "darcy_limit_rel": abs(u_bar_da - u_darcy) / u_darcy,
    }


# ---------------------------------------------------------------------------
# lattice-BGK with Guo forcing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrinkmanChannelConfig:
    """Streamwise-periodic Brinkman channel (walls: half-way bounce-back).

    ``ny`` counts the two solid wall rows, so the fluid channel height is
    ``H = ny - 2`` and the walls are the half-way planes ``y = 0.5`` and
    ``y = ny - 1.5`` (``h = H/2``).  ``nx`` is a small periodic width; the
    problem is x-independent at steady state, the extra columns only serve as a
    check that the x direction really stays uniform.
    """

    ny: int = 66
    nx: int = 4
    tau: float = 0.8
    accel: float = 1.0e-5
    permeability: float = 40.96
    n_steps: int = 200000
    steady_tol: float = 2.0e-11
    check_every: int = 500

    def __post_init__(self) -> None:
        if self.tau <= 0.5:
            raise ValueError("tau must be > 0.5")
        if self.ny < 8 or self.nx < 2:
            raise ValueError("need ny >= 8 (>= 6 fluid rows) and nx >= 2")
        if self.permeability <= 0.0:
            raise ValueError("permeability must be > 0")

    @property
    def nu(self) -> float:
        return CS2 * (self.tau - 0.5)

    @property
    def height(self) -> float:
        return float(self.ny - 2)

    @property
    def half_width(self) -> float:
        return 0.5 * self.height

    @property
    def drag_rate(self) -> float:
        """``lam = nu/K`` (per-step drag relaxation rate; needs ``lam << 1``)."""
        return self.nu / self.permeability


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium stacked on axis 0 (``(9, ny, nx)``)."""
    usq = ux * ux + uy * uy
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    return W[:, None, None] * rho[None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None]
    )


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    inv = 1.0 / rho
    ux = (CX[:, None, None] * f).sum(axis=0) * inv
    uy = (CY[:, None, None] * f).sum(axis=0) * inv
    return rho, ux, uy


def guo_source_2d(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Zero-sum Guo (2002) force source for a general 2-D force ``rho*(ax, ay)``.

    ``S_i = (1 - omega/2) w_i rho [ 3 (c_i - u) + 9 (c_i.u) c_i ] . a``

    The ``9 (c_i.u) c_ix`` partner is *not* optional: dropping it leaves
    ``sum_i S_i = -3 rho a_x != 0``, i.e. a per-step mass sink.  A
    negative-control test pins that defect down.
    """
    cu = CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None]
    a_dot_c = CX[:, None, None] * ax[None] + CY[:, None, None] * ay[None]
    u_dot_a = ux * ax + uy * ay
    factor = (1.0 - 0.5 / tau) * W[:, None, None] * rho[None]
    return factor * (3.0 * (a_dot_c - u_dot_a[None]) + 9.0 * cu * a_dot_c)


_STREAM_IDX: dict[tuple[int, int], np.ndarray] = {}


def _stream_index(ny: int, nx: int) -> np.ndarray:
    """Cached flattened source index for one fused periodic-streaming gather."""
    key = (ny, nx)
    idx = _STREAM_IDX.get(key)
    if idx is None:
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        idx = np.empty((9, ny * nx), dtype=np.int64)
        for q in range(9):
            idx[q] = (((yy - CY[q]) % ny) * nx + ((xx - CX[q]) % nx)).ravel()
        _STREAM_IDX[key] = idx
    return idx


def _stream(f: np.ndarray) -> np.ndarray:
    """Periodic streaming as one fused gather (identical to 9 nested rolls)."""
    ny, nx = f.shape[1], f.shape[2]
    out = np.take_along_axis(f.reshape(9, -1), _stream_index(ny, nx), axis=1)
    return out.reshape(9, ny, nx)


def _step(f: np.ndarray, cfg: "BrinkmanChannelConfig", lam: float,
          drag: bool) -> np.ndarray:
    omega = 1.0 / cfg.tau
    ny = cfg.ny
    rho, ux, uy = macroscopic(f)
    if drag:
        ax = cfg.accel - lam * ux
        ay = -lam * uy
    else:  # K -> infinity limit: pure body-force Poiseuille
        ax = np.full_like(ux, cfg.accel)
        ay = np.zeros_like(uy)
    # the body force acts on fluid nodes only (y stays bounded, x stays periodic)
    ax[0, :] = 0.0
    ax[-1, :] = 0.0
    ay[0, :] = 0.0
    ay[-1, :] = 0.0
    ux_h = ux + 0.5 * ax
    uy_h = uy + 0.5 * ay
    feq = equilibrium(rho, ux_h, uy_h)
    fc = f - omega * (f - feq) + guo_source_2d(rho, ux_h, uy_h, ax, ay, cfg.tau)
    f = _stream(fc)
    # half-way bounce-back on the wall-adjacent *fluid* nodes: the populations
    # that just streamed in from the (unused) wall row are replaced by the
    # reflected post-collision populations of the same fluid node.  This places
    # the no-slip plane half a lattice unit outside that node (y = 0.5 and
    # y = ny - 1.5), which is the wall position used by the reference.
    f[_BOTTOM_IN, 1] = fc[_BOTTOM_OUT, 1]
    f[_TOP_IN, ny - 2] = fc[_TOP_OUT, ny - 2]
    return f


def run_brinkman_channel(cfg: BrinkmanChannelConfig, drag: bool = True) -> dict[str, Any]:
    """Integrate the drag-closed channel flow to steady state and return metrics.

    ``drag=False`` switches the Darcy term off (pure body-force Poiseuille, the
    ``K -> inf`` limit of (1)); it exercises the same code path and is the
    degenerate control for the drag closure.
    """
    ny, nx = cfg.ny, cfg.nx
    nu = cfg.nu
    lam = cfg.drag_rate if drag else 0.0

    rho = np.ones((ny, nx))
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    f = equilibrium(rho, ux, uy)

    fluid_rows = slice(1, ny - 1)
    mass0 = float(f.sum())
    u_bar_prev = 0.0
    steady_rel = float("inf")
    steps_run = cfg.n_steps
    history: list[tuple[int, float]] = []

    for step in range(1, cfg.n_steps + 1):
        f = _step(f, cfg, lam, drag)
        if step % cfg.check_every == 0:
            _, ux_c, _ = macroscopic(f)
            u_bar = float(ux_c[fluid_rows].mean())
            history.append((step, u_bar))
            steady_rel = abs(u_bar - u_bar_prev) / max(abs(u_bar), 1e-300)
            u_bar_prev = u_bar
            if steady_rel < cfg.steady_tol:
                steps_run = step
                break

    rho, ux, uy = macroscopic(f)
    profile = ux[fluid_rows].mean(axis=1)
    y_c = 0.5 * (ny - 1)
    y_off = np.arange(1, ny - 1) - y_c
    if drag:
        u_ex = np.array([
            sample_exact(cfg.accel, nu, cfg.permeability, cfg.half_width, float(y))
            for y in y_off
        ])
        beta = cfg.half_width / math.sqrt(cfg.permeability)
        u_bar_ex = exact_mean_velocity(cfg.accel, nu, cfg.permeability, cfg.half_width)
        shape_ex = exact_shape_function(cfg.permeability, cfg.half_width)
        u_c_ex = exact_centreline_velocity(cfg.accel, nu, cfg.permeability,
                                           cfg.half_width)
        u_D = cfg.accel * cfg.permeability / nu
    else:
        h = cfg.half_width
        u_ex = cfg.accel * (h * h - y_off**2) / (2.0 * nu)
        beta = float("inf")
        u_bar_ex = cfg.accel * h * h / (3.0 * nu)
        u_D = cfg.accel * cfg.permeability / nu
        shape_ex = u_bar_ex / u_D
        u_c_ex = cfg.accel * h * h / (2.0 * nu)

    u_bar_num = float(profile.mean())
    return {
        "config": cfg,
        "drag": drag,
        "nu": nu,
        "drag_rate": lam,
        "beta": beta,
        "eps_inf": float(np.abs(profile - u_ex).max()) / float(np.abs(u_ex).max()),
        "eps_2": float(np.linalg.norm(profile - u_ex)) / float(np.linalg.norm(u_ex)),
        "amp_err": u_bar_num / u_bar_ex - 1.0,
        "u_bar_num": u_bar_num,
        "u_bar_ex": u_bar_ex,
        "u_darcy": u_D,
        "shape_num": u_bar_num / u_D,
        "shape_ex": shape_ex,
        "u_centre_num": float(profile[len(profile) // 2]),
        "u_centre_ex": u_c_ex,
        "max_mach": float(np.abs(ux).max()) / math.sqrt(CS2),
        "x_nonuniformity": float(np.abs(profile[:, None] - ux[fluid_rows]).max())
        / max(float(np.abs(profile).max()), 1e-300),
        "mass_drift": abs(float(f.sum()) - mass0) / mass0,
        "steady_rel_change": steady_rel,
        "steps_run": steps_run,
        "reached_steady": steady_rel < cfg.steady_tol,
        "y": [float(v) for v in y_off],
        "u_num": [float(v) for v in profile],
        "u_exact_profile": [float(v) for v in u_ex],
        "history_tail": history[-5:],
    }