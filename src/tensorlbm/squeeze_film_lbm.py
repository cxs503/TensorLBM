"""D2Q9 lattice-Boltzmann solver for the quasi-steady Reynolds squeeze film.

Why a mass source instead of a moving wall
------------------------------------------
The physical problem is two plates closing at the relative rate ``V = -dh/dt``.
Written in the plate-fixed frame with the plate held at a *fixed* grid position
(the standard quasi-steady reduction of the squeeze-film problem), the effect of
the plate approach is a uniform volumetric source of strength ``V`` per unit area
at the approaching plate: the film loses ``V A`` of volume per unit time, so a
fixed-geometry analog has to inject it there.  The injected fluid is at rest in
the plate frame (no-slip), i.e. the source carries **zero momentum**.

This is not a cosmetic choice.  A halfway-bounce-back wall moving *normal* to
itself has an O(V) momentum-exchange artefact (it injects ``rho V`` of
wall-normal momentum per unit area per step whereas the physical flux is
``rho V^2``); with the film pressure scaling as ``mu V a^2 / h^3`` that artefact
is ``h^3 / (nu a^2)`` of the load and would sit right on top of the quantity
being measured.  The zero-momentum source keeps the momentum-exchange (ME) force
on both plates a clean, unambiguous measurement.

Discretisation
--------------
* ``ny`` fluid rows, planar halfway walls at ``y = -1/2`` and ``y = ny - 1/2``
  (so the film thickness is ``h = ny`` in lattice units);
* ``nx`` fluid columns, pressure-release (Zou-He, ``rho = rho0``) faces at
  ``x = 0`` and ``x = nx - 1``; the plate half-width is ``a = (nx - 1)/2``;
* BGK collision, ``nu = (tau - 1/2) cs^2``;
* mass source ``f_i += w_i rho V`` on the top wall row (mass in, momentum out).

Force measurement
-----------------
Momentum exchange at the midway wall: for a rest wall the fluid loses
``2 f_i^* c_i`` per link crossing the wall, so

    F_top    = +2 sum_{i in {2,5,6}} f_i^* (x, ny-1)      (c_y = +1)
    F_bottom = -2 sum_{i in {4,7,8}} f_i^* (x, 0)         (c_y = -1)

with ``f^*`` the post-collision populations.  Both are reported; in exact
arithmetic ``F_top = -F_bottom`` up to the (negligible) wall-normal momentum flux
through the pressure-release faces.

``force_top`` / ``force_bottom`` are **absolute** loads: they contain the static
term ``sum_x rho0 cs^2 = nx rho0/3`` (what a source-free run returns).  The
quantities comparable with the Reynolds / Stefan anchor are the gauge loads
``force_gauge_top = force_top - nx rho0 cs^2`` and
``force_gauge_bottom = force_bottom + nx rho0 cs^2``; both are returned
explicitly, together with ``baseline_static_force``.  ``force_pressure_integral``
is sampled on the *source-free* wall row (``y = 0``): the source row ``y = ny-1``
carries a pressure artefact of the injection/corner treatment (kept as
``force_pressure_integral_top_row`` for auditing).

Momentum ledger
---------------
The streaming step is a permutation of the populations and therefore conserves
the domain momentum exactly; all momentum change happens at the boundary links.
The ledger recomputes, from the boundary links alone,

    dP_y/dt  =  (momentum carried in at the pressure faces)
              - (momentum carried out at the pressure faces)
              - (F_top + F_bottom)  -  (source momentum = 0)

and the residual of that identity (an exhaustiveness audit of the ME accumulator:
no missing links, no double counting) is reported as ``ledger_residual``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "D2Q9_W",
    "D2Q9_CX",
    "D2Q9_CY",
    "D2Q9_OPP",
    "simulate_squeeze",
    "lattice_reynolds_force",
]

CS2 = 1.0 / 3.0
W = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
              1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

#: directions whose upstream node is the solid below / above the channel.
_TOP_INCOMING = (4, 7, 8)      # c_y = -1, upstream y = ny
_BOT_INCOMING = (2, 5, 6)      # c_y = +1, upstream y = -1
_LEFT_INCOMING = (1, 5, 8)     # c_x = +1, upstream x = -1
_RIGHT_INCOMING = (3, 6, 7)    # c_x = -1, upstream x = nx

D2Q9_W = W
D2Q9_CX = CX
D2Q9_CY = CY
D2Q9_OPP = OPP


def _equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    f = np.empty((9,) + rho.shape, dtype=rho.dtype)
    u2 = ux * ux + uy * uy
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        f[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
    return f


def _macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = f.sum(axis=0)
    ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
    uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
    return rho, ux, uy


def lattice_reynolds_force(half_width: float, gap: float, V: float, nu: float,
                           density: float = 1.0, length: float = 1.0) -> float:
    """Plane-strain Reynolds load in consistent (lattice) units: ``8 mu V a^3 L/h^3``."""
    return 8.0 * density * nu * V * half_width**3 * length / gap**3


def _stream(fpost: np.ndarray) -> np.ndarray:
    """Pure streaming (periodic wrap); boundary rows/cols are overwritten after."""
    f = np.empty_like(fpost)
    f[0] = fpost[0]
    for i in range(1, 9):
        f[i] = np.roll(np.roll(fpost[i], CX[i], axis=0), CY[i], axis=1)
    return f


def _apply_pressure_faces(f: np.ndarray, uy_prev: np.ndarray, rho0: float,
                          nx: int) -> None:
    """Zou-He pressure-release faces (``rho = rho0``, tangential velocity extrapolated)."""
    # ---- left face (x = 0): incoming 1, 5, 8 -------------------------------
    f0 = f[0, 0, :]
    f2 = f[2, 0, :]
    f3 = f[3, 0, :]
    f4 = f[4, 0, :]
    f6 = f[6, 0, :]
    f7 = f[7, 0, :]
    f1 = f[1, 0, :]
    f5 = f[5, 0, :]
    f8 = f[8, 0, :]
    A = rho0 - (f0 + f2 + f3 + f4 + f6 + f7)
    ux = (A - (f3 + f6 + f7)) / rho0
    f1n = f3 + (2.0 / 3.0) * rho0 * ux
    uy_b = uy_prev[1, :]
    B = rho0 * uy_b - (f2 - f4) - (f6 - f7)
    f[1, 0, :] = f1n
    f[5, 0, :] = 0.5 * (A - f1n + B)
    f[8, 0, :] = 0.5 * (A - f1n - B)

    # ---- right face (x = nx-1): incoming 3, 6, 7 ---------------------------
    f0 = f[0, -1, :]
    f1 = f[1, -1, :]
    f2 = f[2, -1, :]
    f4 = f[4, -1, :]
    f5 = f[5, -1, :]
    f8 = f[8, -1, :]
    f3 = f[3, -1, :]
    f6 = f[6, -1, :]
    f7 = f[7, -1, :]
    A = rho0 - (f0 + f1 + f2 + f4 + f5 + f8)
    ux = ((f1 + f5 + f8) - A) / rho0
    f3n = f1 - (2.0 / 3.0) * rho0 * ux
    uy_b = uy_prev[-2, :]
    B = rho0 * uy_b - (f2 - f4) - (f5 - f8)
    f[3, -1, :] = f3n
    f[6, -1, :] = 0.5 * (A - f3n + B)
    f[7, -1, :] = 0.5 * (A - f3n - B)


def _apply_walls(f: np.ndarray, fpost: np.ndarray) -> None:
    """Stationary halfway bounce-back on the bottom (y=0) and top (y=ny-1) walls."""
    for i in _BOT_INCOMING:
        f[i, :, 0] = fpost[OPP[i], :, 0]
    for i in _TOP_INCOMING:
        f[i, :, -1] = fpost[OPP[i], :, -1]


def simulate_squeeze(
    *,
    half_width: int,
    gap: int,
    nu: float,
    V: float,
    rho0: float = 1.0,
    max_steps: int = 200_000,
    tol: float = 1e-12,
    check_every: int = 50,
    source_on: bool = True,
) -> dict:
    """Run the quasi-steady squeeze film to convergence.

    Args:
        half_width: plate half-width ``a`` in lattice units (``nx = 2a + 1``).
        gap: film thickness ``h`` in lattice units (``ny = h`` fluid rows).
        nu: kinematic viscosity in lattice units.
        V: approach speed ``-dh/dt`` in lattice units (Mach number, ``V << cs``).
        rho0: reference density of the pressure-release faces (usually 1).
        max_steps: iteration cap.
        tol: convergence tolerance on the relative change of the total ME load.
        check_every: convergence is tested every this many steps.
        source_on: ``False`` runs the same geometry without the squeeze source
            (negative control: the load must collapse to zero).

    Returns:
        Dict with the converged force (``force_top``, ``force_bottom``), the ME
        ledger (``ledger_residual``), the pressure / velocity profiles, and the
        convergence diagnostics (``steps``, ``residual``, ``history``).
    """
    if half_width < 2:
        raise ValueError("half_width must be >= 2")
    if gap < 3:
        raise ValueError("gap must be >= 3 (need >= 3 rows to resolve the film)")
    if not (0.5001 < 1.0 / (3.0 * nu) + 0.5):  # tau = 3 nu + 1/2 > 1/2
        raise ValueError("nu must be > 0")
    if not (nu > 0.0):
        raise ValueError("nu must be > 0")
    if not (0.0 <= V < 0.1):
        raise ValueError("V must satisfy 0 <= V < 0.1 (low-Mach)")

    nx = 2 * half_width + 1
    ny = gap
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau

    f = _equilibrium(np.full((nx, ny), rho0), np.zeros((nx, ny)), np.zeros((nx, ny)))
    _, _, uy_prev = _macroscopic(f)

    hist: list[dict] = []
    force_top = 0.0
    force_prev = 0.0
    residual = math.inf
    ledger = {}
    steps_done = 0

    for step in range(1, max_steps + 1):
        rho, ux, uy = _macroscopic(f)
        feq = _equilibrium(rho, ux, uy)
        fpost = f - omega * (f - feq)
        if source_on:
            # zero-momentum mass source on the approaching (top) plate row
            for i in range(9):
                fpost[i, :, -1] += W[i] * rho[:, -1] * V

        # momentum-exchange loads from the post-collision state
        force_top = float(2.0 * np.sum(fpost[2, :, -1] + fpost[5, :, -1] + fpost[6, :, -1]))
        force_bot = float(-2.0 * np.sum(fpost[4, :, 0] + fpost[7, :, 0] + fpost[8, :, 0]))

        f_new = _stream(fpost)
        uy_prev = uy
        _apply_pressure_faces(f_new, uy_prev, rho0, nx)
        _apply_walls(f_new, fpost)

        # ---- momentum ledger -------------------------------------------
        py_before = float(np.sum(CY[:, None, None] * fpost))
        py_after = float(np.sum(CY[:, None, None] * f_new))
        dP = py_after - py_before
        influx = 0.0
        outflux = 0.0
        for i in _LEFT_INCOMING:
            influx += float(np.sum(CY[i] * f_new[i, 0, :]))
        for i in _RIGHT_INCOMING:
            influx += float(np.sum(CY[i] * f_new[i, -1, :]))
        for i in _RIGHT_INCOMING:
            outflux += float(np.sum(CY[i] * fpost[i, 0, :]))
        for i in _LEFT_INCOMING:
            outflux += float(np.sum(CY[i] * fpost[i, -1, :]))
        budget = influx - outflux - force_top + (-force_bot)
        ledger = {
            "dP": dP,
            "outlet_influx": influx - outflux,
            "force_walls_on_fluid": -force_top - force_bot,
            "residual": dP - budget,
            "residual_rel": abs(dP - budget) / max(abs(force_top), 1e-300),
        }

        f = f_new
        steps_done = step

        if step % check_every == 0:
            residual = abs(force_top - force_prev) / max(abs(force_top), 1e-300)
            hist.append({"step": step, "force_top": force_top,
                         "force_bottom": force_bot, "rel_change": residual})
            if residual < tol and step > 10 * check_every:
                break
            force_prev = force_top

    rho, ux, uy = _macroscopic(f)
    p = CS2 * (rho - rho0)
    xs = np.arange(nx) - (nx - 1) / 2.0
    ys = np.arange(ny) + 0.5

    # wall-normal (normal-stress corrected) load: p - 2 mu dv/dy at the top wall
    dvdy_top = (-3.0 * uy[:, -1] + 4.0 * uy[:, -2] - uy[:, -3]) / 2.0
    force_top_viscous = float(np.sum(-2.0 * rho0 * nu * dvdy_top))
    # The top row is the *source* row: the injected populations carry no momentum
    # and the wall-adjacent corner cells are overwritten by the bounce-back after
    # the pressure condition is applied, so ``p`` there is a boundary artefact
    # (its edge values even change sign).  The wall load has to be read off the
    # source-free wall row of the opposite wall.
    force_pressure_integral = float(np.sum(p[:, 0]))
    force_pressure_integral_top_row = float(np.sum(p[:, -1]))

    # The momentum-exchange load is an *absolute* load: it contains the static
    # term sum_x rho0 cs^2 = nx rho0 / 3, which is exactly the value the
    # source-free negative control returns.  Only the gauge part is comparable
    # with the Reynolds / Stefan anchor, so it is exposed here instead of leaving
    # every caller to rediscover the offset.
    baseline_static_force = float(nx * CS2 * rho0)
    if not (math.isfinite(force_top) and math.isfinite(force_bot)):
        raise RuntimeError(
            "squeeze-film LBM diverged (non-finite momentum-exchange load) for "
            f"half_width={half_width}, gap={gap}, nu={nu}, V={V}; the "
            "fixed-density pressure faces are unstable once the film is wide "
            "enough (see docs/squeeze_film_lbm_verification_20261108.md)"
        )

    return {
        "force_top": force_top,
        "force_bottom": force_bot,
        "baseline_static_force": baseline_static_force,
        "force_gauge_top": float(force_top - baseline_static_force),
        "force_gauge_bottom": float(force_bot + baseline_static_force),
        "force_pressure_integral": force_pressure_integral,
        "force_pressure_integral_top_row": force_pressure_integral_top_row,
        "force_normal_viscous": force_top_viscous,
        "force_net_balance": float(force_top + force_bot),
        "ledger": ledger,
        "steps": steps_done,
        "converged": bool(residual < tol),
        "final_rel_change": float(residual),
        "history": hist,
        "x": xs,
        "y": ys,
        "p": p,
        "ux": ux,
        "uy": uy,
        "rho": rho,
        "half_width": half_width,
        "gap": gap,
        "nu": nu,
        "V": V,
        "rho0": rho0,
        "tau": tau,
        "source_on": source_on,
    }