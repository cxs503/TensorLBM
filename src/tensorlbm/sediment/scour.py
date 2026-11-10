"""Single-pile local scour: 2-D LBM flow past a cylinder + movable-bed relaxation.

Engineering problem
-------------------
A vertical circular pile of diameter ``D`` in a steady current on a sandy bed
develops a local scour hole.  For steady, live-bed conditions the equilibrium
depth normalised by the pile diameter is the classical Melville & Chiew (1999)
band ``S/D = 1.0 .. 2.4`` (design value ~1.3 at ``h/D ~ 3``, decreasing with
``h/D``).

What is computed here
---------------------
1. **Flow (real LBM, D2Q9/BGK, numpy).**  Flow past a circular cylinder with
   half-way bounce-back on the solid nodes, periodic streamwise and free-slip
   lateral boundaries, driven by a uniform body force.  The measured quantity
   is the *near-pile velocity amplification*

       A = max_{annulus 1.05D/2 < r < 2.5D/2} |U| / (< |U| >_upstream)

   taken from the time-averaged LBM field, so ``A`` is a simulation output, not
   an input.  The solver is verified against the analytic Poiseuille profile in
   the pile-free case (:func:`verify_flow_solver`).

2. **Bed (2-D morphodynamic relaxation).**  Depth-averaged Shields number
   ``theta(x) = theta_0 (|U(x)|/U_far)^2`` with the locally-uniform-flow
   feedback of the deepening hole (local depth ``h + S``)

       theta(x, S) = theta_0 (|U(x)|/U_far)^2 ( h / (h + S) )^p,   p = 2,

   and a pickup-type erosion law ``dS/dt = omega_0 [theta/theta_cr - 1]_+``.
   The equilibrium condition ``theta = theta_cr`` at the fastest point gives
   the closed form ``S_eq = h [ sqrt(theta_0 A^2/theta_cr) - 1 ]``, which is
   used only as a consistency check: the reported ``S/D`` is the *relaxed
   field's* maximum.

Modelling limitations (declared)
--------------------------------
* Depth-averaged 2-D flow: the horseshoe vortex / downflow that drives real
  pile scour is **not** resolved; scour is driven only by the horizontal-flow
  amplification inside the hole.
* The flow field is frozen during the bed relaxation (one-way coupling); the
  only feedback of the hole on the flow is the depth scaling.
* Hence the *magnitude* of ``S/D`` is closure-dependent and is judged against
  the empirical band only (C5 is an interval criterion); the trend with ``D``
  is judged separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PileConfig",
    "ScourConfig",
    "relax_scour",
    "run_pile_flow",
    "run_single_pile_scour",
    "verify_flow_solver",
]

#: D2Q9 lattice velocities / weights / opposite links.
_C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
_W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4)
_OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
#: specular (free-slip) link map: (cx, cy) -> (cx, -cy)
_SPEC = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5])


def _macro(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    ux = (f * _C[:, 0, None, None]).sum(axis=0) / rho
    uy = (f * _C[:, 1, None, None]).sum(axis=0) / rho
    return rho, ux, uy


def _lbm_step(
    f: np.ndarray,
    solid: np.ndarray,
    tau: float,
    body_force: float,
    lateral: str = "free_slip",
) -> np.ndarray:
    """One D2Q9/BGK step: collide, force, stream, bounce-back, free-slip."""
    rho, ux, uy = _macro(f)
    cu = 3.0 * (_C[:, 0, None, None] * ux[None] + _C[:, 1, None, None] * uy[None])
    usq = 1.5 * (ux * ux + uy * uy)
    feq = _W[:, None, None] * rho[None] * (1.0 + cu + 0.5 * cu * cu - usq[None])
    f = f - (f - feq) / tau
    # source term: momentum increment per step = c_s^2 F rho, so multiply by 3
    f = f + 3.0 * (_W[:, None, None] * _C[:, 0, None, None] * body_force * rho[None])
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(np.roll(f[i], _C[i, 0], axis=1), _C[i, 1], axis=0)
    # half-way bounce-back on the solid nodes
    out[:, solid] = out[_OPP][:, solid]
    if lateral == "free_slip":
        # specular reflection: (cx, cy) -> (cx, -cy) at the lateral boundaries
        for i in (5, 6, 2):
            out[_SPEC[i], -1, :] = out[i, -1, :]
        for i in (7, 8, 4):
            out[_SPEC[i], 0, :] = out[i, 0, :]
    return out


def _apply_inlet_outlet(f: np.ndarray, u_in: float) -> None:
    """Velocity inlet (left) + zero-gradient outflow (right), in place.

    Inlet: the incoming links (cx = +1) of column 0 are set to the equilibrium
    at ``(u_in, 0)``.  Outlet: the incoming links (cx = -1) of the last column
    copy the neighbouring column (zero-gradient / convective outflow).
    """
    rho_l = f[:, :, 0].sum(axis=0).mean()
    for i in (1, 5, 8):
        feq = _W[i] * rho_l * (1.0 + 3.0 * u_in + 4.5 * (u_in * _C[i, 0]) ** 2
                               - 1.5 * u_in**2)
        f[i, :, 0] = feq
    for i in (3, 6, 7):
        f[i, :, -1] = f[i, :, -2]


@dataclass(frozen=True)
class PileConfig:
    """Lattice setup of the flow past a circular pile."""

    nx: int = 96
    ny: int = 96
    diameter: float = 10.0  # pile diameter in lattice units
    nu: float = 0.005  # lattice viscosity -> Re = U_in D / nu
    inlet_velocity: float = 0.06  # uniform approach velocity (lattice units)
    n_steps: int = 8000
    n_avg: int = 1500  # averaging window at the end (steps)

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5


def _pile_mask(nx: int, ny: int, diameter: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cx, cy = 0.30 * nx, 0.5 * ny
    r = 0.5 * diameter
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def run_pile_flow(cfg: PileConfig | None = None) -> dict:
    """Run the LBM flow past the pile; return the near-field amplification.

    Returns ``A`` (near-field amplification), ``U_far`` (upstream mean speed,
    lattice units), ``Re = U_far D / nu``, the time-averaged speed field and
    the run parameters.
    """
    cfg = cfg or PileConfig()
    nx, ny = cfg.nx, cfg.ny
    solid = _pile_mask(nx, ny, cfg.diameter)
    rho0 = np.ones((ny, nx))
    ux0 = np.full((ny, nx), cfg.inlet_velocity)
    feq = _W[:, None, None] * rho0[None] * (
        1.0
        + 3.0 * _C[:, 0, None, None] * ux0[None]
        + 4.5 * (_C[:, 0, None, None] * ux0[None]) ** 2
        - 1.5 * ux0[None] ** 2
    )
    f = feq.copy()
    acc = np.zeros((ny, nx))
    n_acc = 0
    for step in range(cfg.n_steps):
        f = _lbm_step(f, solid, cfg.tau, 0.0)
        _apply_inlet_outlet(f, cfg.inlet_velocity)
        if step >= cfg.n_steps - cfg.n_avg:
            _, ux, uy = _macro(f)
            sp = np.sqrt(ux * ux + uy * uy)
            sp[solid] = 0.0
            acc += sp
            n_acc += 1
    speed = acc / max(n_acc, 1)

    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cx, cy = 0.30 * nx, 0.5 * ny
    r = 0.5 * cfg.diameter
    rad2 = (xx - cx) ** 2 + (yy - cy) ** 2
    annulus = (rad2 > (1.05 * r) ** 2) & (rad2 < (2.5 * r) ** 2)
    upstream = (xx < cx - 2.0 * r) & (np.abs(yy - cy) < 1.5 * r)
    u_near = float(speed[annulus].max())
    u_far = float(speed[upstream].mean())
    return {
        "A": u_near / max(u_far, 1.0e-30),
        "u_near": u_near,
        "U_far": u_far,
        "Re": u_far * cfg.diameter / cfg.nu,
        "speed_field": speed,
        "x_over_D": (xx - cx) / cfg.diameter,
        "y_over_D": (yy - cy) / cfg.diameter,
        "nx": nx,
        "ny": ny,
        "diameter": cfg.diameter,
        "n_steps": cfg.n_steps,
        "n_avg": cfg.n_avg,
        "nu": cfg.nu,
        "inlet_velocity": cfg.inlet_velocity,
    }


def verify_flow_solver(
    nx: int = 32, ny: int = 33, nu: float = 0.05, body_force: float = 1.0e-5,
    n_steps: int = 6000,
) -> dict:
    """Verify the LBM flow solver against the analytic Poiseuille profile.

    No pile: the wall-bounded channel driven by a constant body force has the
    exact steady state ``u(y) = F y (h-y) / (2 nu)``, ``h = ny-1``.  Returns the
    max relative error of the simulated profile and the measured wall-shear
    velocity versus the analytic ``u* = sqrt(F h / 2)``.
    """
    solid = np.zeros((ny, nx), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    # half-way bounce-back: walls at y = 0.5 and y = ny - 1.5
    h = ny - 2.0
    f = _W[:, None, None] * np.ones((9, ny, nx))
    for _ in range(n_steps):
        f = _lbm_step(f, solid, 3.0 * nu + 0.5, body_force)
    _, ux, _uy = _macro(f)
    prof = ux[1:-1].mean(axis=1)
    y = np.arange(1, ny - 1) - 0.5
    u_an = body_force / (2.0 * nu) * y * (h - y)
    rel = float(np.max(np.abs(prof - u_an)) / u_an.max())
    # wall-shear velocity from a parabolic fit u = a y (h - y): u* = sqrt(nu a h)
    basis = y * (h - y)
    a_fit = float(np.dot(prof, basis) / np.dot(basis, basis))
    u_star = math.sqrt(nu * a_fit * h)
    return {
        "profile_rel_err": rel,
        "u_star_lbm": u_star,
        "u_star_analytic": math.sqrt(body_force * h / 2.0),
        "u_star_rel_err": abs(u_star - math.sqrt(body_force * h / 2.0))
        / math.sqrt(body_force * h / 2.0),
        "u_max_lbm": float(prof.max()),
        "u_max_analytic": float(u_an.max()),
    }


@dataclass(frozen=True)
class ScourConfig:
    """Physical conditions of the scour case (SI units)."""

    D: float = 0.05  # pile diameter (m)
    h: float = 0.15  # water depth (m)  -> h/D = 3 (Melville & Chiew flume)
    d: float = 3.85e-4  # sand grain diameter (m) (their medium sand)
    s: float = 1.65
    g: float = 9.81
    nu: float = 1.0e-6
    theta_cr: float = 0.047  # incipient-motion Shields stress of the sand
    u_over_uc: float = 0.95  # approach-flow intensity (live bed, near peak scour)
    depth_exponent: float = 2.0  # p in the locally-uniform-flow feedback
    omega0: float = 0.02  # erosion rate scale (1/s)
    relax_steps: int = 30000
    relax_dt: float = 0.2  # s (dS per step must stay << h)
    tol: float = 1.0e-9
    max_step_frac_h: float = 0.1  # limiter: max bed change per step / h

    def __post_init__(self) -> None:
        if self.D <= 0.0 or self.h <= 0.0:
            raise ValueError("require D > 0, h > 0")
        if self.depth_exponent <= 0.0:
            raise ValueError("require depth_exponent > 0")


def relax_scour(
    speed_field: np.ndarray,
    cfg: ScourConfig,
    u_far: float,
    amplification: float,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
) -> dict:
    """Relax the 2-D scour field with the pickup law; return the equilibrium.

    ``theta(x) = theta0 (|U(x)|/U_far)^2 (h/(h+S))^p``; ``amplification`` is used
    only for the closed-form cross-check, the relaxation uses the raw field.
    """
    q = (speed_field / max(u_far, 1.0e-30)) ** 2  # (|U|/U_far)^2, max -> A^2
    theta0 = cfg.u_over_uc**2 * cfg.theta_cr
    p = cfg.depth_exponent
    S = np.zeros_like(speed_field, dtype=float)
    dt, omega0 = cfg.relax_dt, cfg.omega0
    hist: list[tuple[float, float]] = []
    steps = 0
    for steps in range(1, cfg.relax_steps + 1):
        theta = theta0 * q * (cfg.h / (cfg.h + S)) ** p
        drive = np.clip(theta / cfg.theta_cr - 1.0, 0.0, None)
        dS = np.minimum(omega0 * drive * dt, cfg.max_step_frac_h * cfg.h)
        dSmax = float(dS.max())
        S = S + dS
        if steps % 1000 == 0:
            hist.append((steps * dt, float(S.max())))
        if dSmax < cfg.tol * cfg.D:
            break
    S_max = float(S.max())
    closed = cfg.h * (math.sqrt(theta0 * amplification**2 / cfg.theta_cr) - 1.0)
    return {
        "S_max": S_max,
        "S_over_D": S_max / cfg.D,
        "closed_form_S": max(closed, 0.0),
        "closed_form_S_over_D": max(closed, 0.0) / cfg.D,
        "theta0": theta0,
        "u_over_uc": cfg.u_over_uc,
        "amplification": amplification,
        "relax_steps": steps,
        "history": hist,
        "scour_field": S,
        "x_over_D": x_over_d,
        "y_over_D": y_over_d,
    }


def run_single_pile_scour(
    pile: PileConfig | None = None, cfg: ScourConfig | None = None
) -> dict:
    """Full chain: LBM flow -> amplification -> movable-bed relaxation -> S/D."""
    pile = pile or PileConfig()
    cfg = cfg or ScourConfig()
    flow = run_pile_flow(pile)
    bed = relax_scour(
        flow["speed_field"], cfg, flow["U_far"], flow["A"],
        flow["x_over_D"], flow["y_over_D"],
    )
    return {
        "S_over_D": bed["S_over_D"],
        "S_max": bed["S_max"],
        "A": flow["A"],
        "Re": flow["Re"],
        "U_far": flow["U_far"],
        "theta0": bed["theta0"],
        "u_over_uc": bed["u_over_uc"],
        "h_over_D": cfg.h / cfg.D,
        "closed_form_S_over_D": bed["closed_form_S_over_D"],
        "melville_chiew_band": (1.0, 2.4),
        "in_band": bool(1.0 <= bed["S_over_D"] <= 2.4),
        "relax_steps": bed["relax_steps"],
        "history": bed["history"],
        "scour_field": bed["scour_field"],
        "speed_field": flow["speed_field"],
        "x_over_D": flow["x_over_D"],
        "y_over_D": flow["y_over_D"],
        "flow": {k: v for k, v in flow.items() if k != "speed_field"},
    }