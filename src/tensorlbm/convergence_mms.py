"""Second-order convergence evidence for D2Q9: manufactured solutions + Guo forcing
+ wall-position correction for interpolated bounce-back.

Why this module exists
----------------------
TensorLBM's verification ledger carried an open item ("二阶收敛欠账"): individual
cases reported first-order-looking global orders, but nobody could say whether the
cause was (i) the numerical scheme, (ii) the way the body force was injected, or
(iii) the *reference* against which the error was measured.  Every candidate
explanation was plausible with the evidence that existed, so the debt could not be
paid by argument -- only by an experiment that separates the three factors.

This module is that experiment.  It provides four *fully documented, dependency-
light* (NumPy only) reference configurations whose exact solutions are known in
closed form, plus a machine-checkable evidence record:

1. ``mms_force``  -- 2-D manufactured solution (steady Taylor-Green vortex),
   injected through **Guo body force**.  The analytic fields are

   .. math::
       u_x = U_0\\cos(kx)\\sin(ky),\\qquad
       u_y = -U_0\\sin(kx)\\cos(ky),

   which satisfy the steady incompressible Navier-Stokes equations with
   :math:`p/\\rho = -(U_0^2/4)(\\cos 2kx+\\cos 2ky)` and a body force
   :math:`\\mathbf{F} = 2\\rho\\nu k^2\\,\\mathbf{u}` (derivation in
   ``docs/reports/M7059_lbm_second_order_convergence.md`` §2).  The field is
   computed in the incompressible limit by construction, so *no* boundary layer
   exists anywhere (fully periodic box): the only error sources are the lattice
   discretisation and the force-injection scheme.  Observed order must be 2.

2. ``mms_no_force`` -- negative control: identical geometry and initial data with
   the manufactured source switched off.  Periodic viscous decay drives the field
   to rest, so the error against the manufactured field cannot converge.  This is
   the **gate-can-fail** witness for (1).

3. ``poiseuille_bouzidi`` -- plane Poiseuille channel with the wall placed at a
   *fractional* position ``q = 1/4`` cell from the first fluid node and resolved
   by quadratic interpolated bounce-back (Bouzidi et al. 2001).  Second order.

4. ``poiseuille_halfway`` -- same channel, but the wall is modelled as half-way
   bounce-back, i.e. the effective wall is *fixed* at :math:`y_w = 0.5\\Delta`
   although the true wall sits at :math:`0.25\\Delta`.  Measured against the exact
   profile for the true geometry this is first order -- the classic
   "wall position" error, not a scheme error.  Measuring the *same* data against
   the exact profile shifted to :math:`y_w = 0.5\\Delta` recovers second order,
   which is the quantitative statement of the wall-position correction idea
   (``ref = "effective_wall"``).

The four configurations together answer the three candidate explanations:
scheme order is 2, Guo injection preserves it, and first-order *measurements* are
explained quantitatively by a fixed :math:`O(\\Delta)` wall-position offset.

Conventions
-----------
* D2Q9, BGK, lattice units (:math:`\\Delta t=\\Delta x=1`, :math:`c_s^2=1/3`).
* Guo force scheme (Guo, Zheng & Shi 2002): half-force macroscopic velocity
  :math:`\\rho\\mathbf{u} = \\sum_i \\mathbf{c}_i f_i + \\tfrac12\\mathbf{F}`, discrete
  force term :math:`S_i = (1-\\tfrac{1}{2\\tau})w_i[3(\\mathbf{c}_i\\cdot\\mathbf{F})
  + 9(\\mathbf{c}_i\\cdot\\mathbf{u})(\\mathbf{c}_i\\cdot\\mathbf{F})
  - 3(\\mathbf{u}\\cdot\\mathbf{F})]`.
* Array layout ``(9, ny, nx)`` with ``y`` the wall-normal index, float64.

Everything is deterministic: no random numbers, no torch, no threading.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "C",
    "W",
    "OPPOSITE",
    "CS2",
    "equilibrium",
    "macroscopic",
    "guo_source",
    "mms_fields",
    "mms_force_field",
    "study_mms",
    "poiseuille_exact",
    "study_poiseuille",
    "observed_order",
    "observed_order_pairwise",
    "richardson_order_nested",
    "fit_effective_wall",
    "build_evidence",
    "SCHEMA_VERSION",
]

# --------------------------------------------------------------------------------------
# lattice
# --------------------------------------------------------------------------------------
C = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int64,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0
SCHEMA_VERSION = "lbm-convergence-evidence/1.0"


def _w3() -> np.ndarray:
    return W.reshape(9, 1, 1)


def _cx3() -> np.ndarray:
    return C[:, 0].astype(np.float64).reshape(9, 1, 1)


def _cy3() -> np.ndarray:
    return C[:, 1].astype(np.float64).reshape(9, 1, 1)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """D2Q9 equilibrium, second order in Mach number (NumPy, float64).

    Bit-compatible with :func:`tensorlbm.d2q9.equilibrium` for equal inputs
    (cross-checked in ``tests/test_convergence_mms.py``).
    """
    cu = _cx3() * ux + _cy3() * uy
    usq = ux * ux + uy * uy
    return _w3() * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (zero-force) moments: ``rho = sum f``, ``u = sum c f / rho``."""
    rho = f.sum(axis=0)
    rho_safe = np.where(rho > 0.0, rho, 1.0)
    ux = (f * _cx3()).sum(axis=0) / rho_safe
    uy = (f * _cy3()).sum(axis=0) / rho_safe
    return rho, ux, uy


def guo_source(
    f: np.ndarray, fx: np.ndarray, fy: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Guo (2002) discrete force term and the matching half-force velocity.

    Returns ``(S, rho, ux, uy)`` where ``(rho, ux, uy)`` already include the
    half-force correction and are therefore the velocities to be used in the
    equilibrium and in the reported diagnostics.
    """
    rho, ux_raw, uy_raw = macroscopic(f)
    rho_safe = np.where(rho > 0.0, rho, 1.0)
    ux = ux_raw + 0.5 * fx / rho_safe
    uy = uy_raw + 0.5 * fy / rho_safe
    cu = _cx3() * ux + _cy3() * uy
    cf = _cx3() * fx + _cy3() * fy
    uf = ux * fx + uy * fy
    s = (1.0 - 0.5 / tau) * _w3() * (3.0 * cf + 9.0 * cu * cf - 3.0 * uf)
    return s, rho, ux, uy


def _stream(fpost: np.ndarray) -> np.ndarray:
    """Periodic streaming: ``out[i, y, x] = fpost[i, y - cy, x - cx]``."""
    out = np.empty_like(fpost)
    for i in range(9):
        out[i] = np.roll(np.roll(fpost[i], int(C[i, 1]), axis=0), int(C[i, 0]), axis=1)
    return out


# --------------------------------------------------------------------------------------
# manufactured solution
# --------------------------------------------------------------------------------------
def mms_fields(
    n: int, u0: float, waves: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Steady Taylor-Green manufactured solution on an ``n x n`` lattice patch.

    Lattice units throughout (``h = 1``), so one wavelength spans ``n / waves``
    cells and the *lattice* wavenumber is ``k = 2 pi waves / n``.  Refining ``n``
    therefore refines the same physical field -- the standard MMS refinement
    protocol.  (Using a fixed ``k = 2 pi`` in lattice units would mean a
    wavelength of one cell, which is not a resolved field at all.)

    Returns ``(x, y, ux, uy, h, k)``.
    """
    h = 1.0
    x1 = (np.arange(n) + 0.5) * h
    x, y = np.meshgrid(x1, x1)
    k = 2.0 * math.pi * waves / n
    ux = u0 * np.cos(k * x) * np.sin(k * y)
    uy = -u0 * np.sin(k * x) * np.cos(k * y)
    return x, y, ux, uy, h, k


def mms_force_field(
    rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, nu: float, k: float
) -> tuple[np.ndarray, np.ndarray]:
    """Manufactured body force for the steady Taylor-Green field.

    Derivation (steady, divergence-free field
    ``u = U0 (cos kx sin ky, -sin kx cos ky)``)::

        (u.grad)u = -(U0^2 k/2) (sin 2kx, sin 2ky)  =  grad Phi,
        Phi       =  (U0^2/4)(cos 2kx + cos 2ky)

    Choosing ``p/rho = -Phi`` makes the pressure gradient cancel the convective
    term exactly, so the momentum residual is purely viscous and
    ``F = -rho nu lap(u) = 2 rho nu k^2 u`` (because ``lap(u) = -2 k^2 u``).
    """
    return 2.0 * rho * nu * k * k * ux, 2.0 * rho * nu * k * k * uy


def run_mms_2d(
    n: int,
    u0: float = 0.005,
    tau: float = 1.0,
    waves: int = 1,
    force: bool = True,
    init: str = "exact",
    max_steps: int = 200000,
    tol: float = 1e-14,
    check_every: int = 25,
) -> dict[str, Any]:
    """March the periodic box to steady state and return the L2 error.

    ``init="exact"`` starts from the manufactured field (the steady state is the
    unique fixed point of the forced system, so this only removes the acoustic
    transient); ``init="rest"`` starts from ``u = 0`` and is used as an
    independent route to the *same* steady state.  The manufactured force is
    evaluated from the analytic velocity, never from the simulated one, so the
    run measures discretisation + force-scheme error and not a feedback loop.
    """
    if init not in {"exact", "rest"}:
        raise ValueError(f"unknown init {init!r}")
    nu = CS2 * (tau - 0.5)
    x, y, ux_ex, uy_ex, h, k = mms_fields(n, u0, waves)
    if init == "exact":
        f = equilibrium(np.ones((n, n)), ux_ex, uy_ex)
    else:
        f = equilibrium(np.ones((n, n)), np.zeros((n, n)), np.zeros((n, n)))

    ux_prev = None
    uy_prev = None
    residual = math.inf
    steps = 0
    while steps < max_steps:
        rho, _, _ = macroscopic(f)
        if force:
            fx, fy = mms_force_field(rho, ux_ex, uy_ex, nu, k)
            s, rho, ux, uy = guo_source(f, fx, fy, tau)
        else:
            fx = np.zeros_like(rho)
            fy = np.zeros_like(rho)
            s = np.zeros_like(f)
            rho, ux, uy = macroscopic(f)
        feq = equilibrium(rho, ux, uy)
        f = _stream(f - (f - feq) / tau + s)
        steps += 1
        if ux_prev is not None and steps % check_every == 0:
            du = max(float(np.max(np.abs(ux - ux_prev))), float(np.max(np.abs(uy - uy_prev))))
            residual = du
            if du < tol:
                break
        ux_prev = ux.copy()
        uy_prev = uy.copy()

    rho, ux, uy = macroscopic(f)
    if force:
        rho_safe = np.where(rho > 0.0, rho, 1.0)
        ux = ux + 0.5 * fx / rho_safe
        uy = uy + 0.5 * fy / rho_safe
    return {
        "n": n,
        "h": h,
        "k": k,
        "waves": waves,
        "steps": steps,
        "residual": residual,
        "converged": bool(residual < tol),
        "l2_rel": rel_l2(ux, uy, ux_ex, uy_ex),
        "linf_rel": rel_linf(ux, uy, ux_ex, uy_ex),
        "mass_drift": float(np.max(np.abs(rho - 1.0))),
        "div_l2": float(np.sqrt(np.mean(divergence(ux, uy) ** 2))),
        "fields": {"ux": ux, "uy": uy, "ux_exact": ux_ex, "uy_exact": uy_ex},
    }


def divergence(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Spectral-free central-difference divergence on a periodic lattice patch."""
    dx = 0.5 * (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1))
    dy = 0.5 * (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0))
    return dx + dy


# --------------------------------------------------------------------------------------
# plane Poiseuille with wall-position variants
# --------------------------------------------------------------------------------------
def poiseuille_exact(
    y: np.ndarray, g: float, nu: float, y_left: float, y_right: float
) -> np.ndarray:
    """Exact steady profile ``u(y) = G/(2 nu) (y - y_l)(y_r - y)``."""
    return (g / (2.0 * nu)) * (y - y_left) * (y_right - y)


def _wall_terms(fpost: np.ndarray, unknown: int, idx: np.ndarray, q: float) -> np.ndarray:
    """Quadratic interpolated bounce-back (Bouzidi 2001) for ``f[unknown]``.

    ``idx`` are the three inward samples of ``f`` in direction ``opposite(unknown)``
    (rows): index 0 the boundary node, then one and two steps into the fluid.
    Formula for ``q <= 1/2``::

        f_i(x_b) = q(2q+1) f_ib(x_b) + (1-4q^2) f_ib(x_b+c_ib) - q(1-2q) f_ib(x_b+2c_ib)

    At ``q = 1/2`` this collapses to the half-way bounce-back rule
    ``f_i(x_b) = f_ib*(x_b)`` exactly, so the half-way limit needs no special case.
    """
    a0, a1, a2 = idx[0], idx[1], idx[2]
    return q * (2.0 * q + 1.0) * a0 + (1.0 - 4.0 * q * q) * a1 - q * (1.0 - 2.0 * q) * a2


def run_poiseuille_2d(
    n_f: int,
    q: float = 0.25,
    wall_model: str = "bouzidi",
    g: float = 1e-4,
    tau: float = 1.0,
    init: str = "exact",
    max_steps: int = 400000,
    tol: float = 1e-12,
    check_every: int = 50,
    n_down: int = 2,
) -> dict[str, Any]:
    """Plane Poiseuille channel driven by a constant body force (Guo injection).

    Geometry (lattice units, ``h = 1``): ``n_f`` fluid rows at ``y_j = q + j``
    (``j = 0 .. n_f-1``), walls at ``y = 0`` and ``y = n_f - 1 + 2q``; periodic in
    ``x`` over ``n_down`` columns.  There are no solid nodes: the wall condition
    is imposed directly on the boundary row, which keeps the *true* wall position
    under explicit control.

    ``wall_model``:
      * ``"bouzidi"`` -- quadratic interpolated bounce-back at the true ``q``;
      * ``"halfway"`` -- half-way bounce-back; the effective wall is then fixed at
        ``0.5`` cell from the first fluid node irrespective of ``q``.

    ``init="exact"`` starts from the exact parabola of the *model's own* wall
    positions (the steady state is unique, so this only skips the viscous
    transient); ``init="rest"`` starts from rest.
    """
    if wall_model not in {"bouzidi", "halfway"}:
        raise ValueError(f"unknown wall_model {wall_model!r}")
    if init not in {"exact", "rest"}:
        raise ValueError(f"unknown init {init!r}")
    q_true = float(q)
    q_eff = 0.5 if wall_model == "halfway" else q_true
    if not (0.0 < q_eff <= 0.5):
        raise ValueError("quadratic interpolation implemented for 0 < q <= 0.5")

    nu = CS2 * (tau - 0.5)
    nx = int(n_down)
    y = q_true + np.arange(n_f, dtype=np.float64)
    height_true = (n_f - 1) + 2.0 * q_true
    y_left_true = 0.0
    y_right_true = height_true
    # walls the *model* actually represents
    y_left_model = q_true - q_eff
    y_right_model = (q_true + n_f - 1) + q_eff

    if init == "exact":
        u0 = poiseuille_exact(y, g, nu, y_left_model, y_right_model)
        ux_init = np.repeat(u0.reshape(n_f, 1), nx, axis=1)
    else:
        ux_init = np.zeros((n_f, nx))
    f = equilibrium(np.ones((n_f, nx)), ux_init, np.zeros((n_f, nx)))

    # --- wall update operators, precomputed -------------------------------------
    up = (2, 5, 6)  # unknown populations at the bottom row (incoming from below)
    down = (4, 7, 8)  # unknown populations at the top row (incoming from above)
    ib_up = [int(OPPOSITE[i]) for i in up]
    ib_down = [int(OPPOSITE[i]) for i in down]
    xsrc_up = [(np.arange(nx) - int(C[j, 0])) % nx for j in ib_up]
    xsrc_down = [(np.arange(nx) - int(C[j, 0])) % nx for j in ib_down]
    i_up = np.array(up)
    i_down = np.array(down)
    ib_up_a = np.array(ib_up)
    ib_down_a = np.array(ib_down)
    c0 = q_eff * (2.0 * q_eff + 1.0)
    c1 = 1.0 - 4.0 * q_eff * q_eff
    c2 = -q_eff * (1.0 - 2.0 * q_eff)
    fx3 = _cx3()
    last = n_f - 1

    ux_prev = None
    residual = math.inf
    steps = 0
    while steps < max_steps:
        rho = f.sum(axis=0)
        rho_safe = np.where(rho > 0.0, rho, 1.0)
        ux_raw = (f * fx3).sum(axis=0) / rho_safe + 0.5 * g  # F = rho*g -> F/(2 rho) = g/2
        uy = (f * _cy3()).sum(axis=0) / rho_safe
        feq = equilibrium(rho, ux_raw, uy)
        cu = _cx3() * ux_raw + _cy3() * uy
        # Guo source with F = (rho*g, 0), evaluated at the half-force velocity
        cf = _cx3() * (rho * g)
        uf = ux_raw * (rho * g)
        s = (1.0 - 0.5 / tau) * _w3() * (3.0 * cf + 9.0 * cu * cf - 3.0 * uf)
        fpost = f - (f - feq) / tau + s
        f = _stream(fpost)

        # bottom wall (row 0): rebuild f[2], f[5], f[6] from f[4], f[7], f[8]
        a0 = fpost[ib_up_a, 0, :]
        a1 = fpost[ib_up_a, 1, :][np.arange(3)[:, None], np.stack(xsrc_up)]
        a2 = fpost[ib_up_a, 2, :][np.arange(3)[:, None], np.stack(xsrc_up)]
        f[i_up, 0, :] = c0 * a0 + c1 * a1 + c2 * a2
        # top wall (row last): rebuild f[4], f[7], f[8] from f[2], f[5], f[6]
        b0 = fpost[ib_down_a, last, :]
        b1 = fpost[ib_down_a, last - 1, :][np.arange(3)[:, None], np.stack(xsrc_down)]
        b2 = fpost[ib_down_a, last - 2, :][np.arange(3)[:, None], np.stack(xsrc_down)]
        f[i_down, last, :] = c0 * b0 + c1 * b1 + c2 * b2

        steps += 1
        if ux_prev is not None and steps % check_every == 0:
            du = float(np.max(np.abs(ux_raw - ux_prev)))
            residual = du
            if du < tol:
                break
        ux_prev = ux_raw.copy()

    rho = f.sum(axis=0)
    rho_safe = np.where(rho > 0.0, rho, 1.0)
    ux_raw = (f * fx3).sum(axis=0) / rho_safe
    ux = ux_raw + 0.5 * g
    uy = (f * _cy3()).sum(axis=0) / rho_safe
    return {
        "n_f": n_f,
        "q": q_true,
        "wall_model": wall_model,
        "q_eff": q_eff,
        "g": g,
        "nu": nu,
        "tau": tau,
        "init": init,
        "steps": steps,
        "residual": residual,
        "y": y,
        "ux": ux.mean(axis=1),
        "uy_max": float(np.max(np.abs(uy))),
        "height_true": height_true,
        "y_left_true": y_left_true,
        "y_right_true": y_right_true,
        "y_left_model": y_left_model,
        "y_right_model": y_right_model,
        "y_left_effective_halfway": q_true - 0.5,
        "y_right_effective_halfway": (q_true + n_f - 1) + 0.5,
    }


# --------------------------------------------------------------------------------------
# error norms and order estimation
# --------------------------------------------------------------------------------------
def rel_l2(un: np.ndarray, vn: np.ndarray, ue: np.ndarray, ve: np.ndarray) -> float:
    num = float(np.sum((un - ue) ** 2 + (vn - ve) ** 2))
    den = float(np.sum(ue**2 + ve**2))
    return math.sqrt(num / den)


def rel_linf(un: np.ndarray, vn: np.ndarray, ue: np.ndarray, ve: np.ndarray) -> float:
    scale = float(np.max(np.sqrt(ue**2 + ve**2)))
    return float(max(np.max(np.abs(un - ue)), np.max(np.abs(vn - ve)))) / scale


def observed_order(h: Sequence[float], e: Sequence[float]) -> dict[str, Any]:
    """Least-squares slope of ``log e`` versus ``log h`` (the *producer* fit)."""
    hh = np.asarray(h, dtype=np.float64)
    ee = np.asarray(e, dtype=np.float64)
    if hh.size < 2:
        raise ValueError("at least two grid levels are required")
    if np.any(ee <= 0.0):
        raise ValueError("errors must be strictly positive")
    lx = np.log(hh)
    ly = np.log(ee)
    p, c = np.polyfit(lx, ly, 1)
    pred = p * lx + c
    resid = ly - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "method": "loglog_least_squares_numpy_polyfit",
        "p": float(p),
        "intercept": float(c),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
        "rms_log_residual": float(math.sqrt(ss_res / lx.size)),
        "n_levels": int(hh.size),
    }


def observed_order_independent(h: Sequence[float], e: Sequence[float]) -> dict[str, Any]:
    """Second, independent implementation of the same least-squares fit.

    Closed-form normal equations evaluated in Python floats (no NumPy, no
    ``polyfit``), so an agreement test between the two is a genuine
    cross-implementation check rather than a tautology.
    """
    lx = [math.log(float(v)) for v in h]
    ly = [math.log(float(v)) for v in e]
    n = len(lx)
    if n < 2:
        raise ValueError("at least two grid levels are required")
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    sxy = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    p = sxy / sxx
    c = my - p * mx
    ss_res = sum((ly[i] - (p * lx[i] + c)) ** 2 for i in range(n))
    return {
        "method": "loglog_normal_equations_pure_python",
        "p": p,
        "intercept": c,
        "rms_log_residual": math.sqrt(ss_res / n),
        "n_levels": n,
    }


def observed_order_pairwise(h: Sequence[float], e: Sequence[float]) -> list[dict[str, Any]]:
    """Pairwise observed orders ``p_ij = log(e_i/e_j) / log(h_i/h_j)``."""
    out = []
    hh = [float(v) for v in h]
    ee = [float(v) for v in e]
    for i in range(len(hh) - 1):
        for j in range(i + 1, len(hh)):
            out.append(
                {
                    "coarse": i,
                    "fine": j,
                    "ratio_h": hh[i] / hh[j],
                    "ratio_e": ee[i] / ee[j],
                    "p": math.log(ee[i] / ee[j]) / math.log(hh[i] / hh[j]),
                }
            )
    return out


def richardson_order_nested(e: Sequence[float], ratio: float = 2.0) -> float:
    """Three-level Richardson order estimate for geometrically nested grids.

    ``p = log(ratio) / log( (e_0 - e_1) / (e_1 - e_2) )`` -- valid for
    ``e = C h^p``; used only as a sanity cross-check on the LS fit.
    """
    if len(e) < 3:
        raise ValueError("three levels are required")
    num = float(e[0]) - float(e[1])
    den = float(e[1]) - float(e[2])
    if num <= 0.0 or den <= 0.0:
        raise ValueError("errors must decrease monotonically and satisfy (e0-e1)/(e1-e2) > 0")
    return math.log(num / den) / math.log(ratio)


def fit_effective_wall(
    y: np.ndarray, u: np.ndarray, g: float, nu: float, window: float = 0.15
) -> dict[str, Any]:
    """Least-squares recovery of the *effective* wall positions from a profile.

    Fits ``u(y) = A y^2 + B y + C`` to the interior of the profile (excluding
    ``window`` of the span next to each wall to avoid the interpolated-BB stencil
    region), then reports the roots of the fitted parabola.  This is the direct,
    scheme-independent measurement of where the wall "appears" to be.
    """
    yy = np.asarray(y, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64)
    span = yy[-1] - yy[0]
    mask = (yy > yy[0] + window * span) & (yy < yy[-1] - window * span)
    coef = np.polyfit(yy[mask], uu[mask], 2)
    roots = np.sort(np.roots(coef))
    roots = np.real(roots)
    return {
        "quadratic_coeff": float(coef[0]),
        "expected_curvature": float(-g / (2.0 * nu)),
        "wall_left_fit": float(roots[0]),
        "wall_right_fit": float(roots[1]),
        "n_fit_points": int(mask.sum()),
    }


# --------------------------------------------------------------------------------------
# studies
# --------------------------------------------------------------------------------------
@dataclass
class StudySpec:
    """Declared experiment protocol (pre-registered before inspecting results)."""

    name: str
    description: str
    grids: tuple[int, ...]
    thresholds: dict[str, float]
    parameters: dict[str, Any] = field(default_factory=dict)


def transient_lag(
    residual: float, lam: float, check_every: int, u_scale: float
) -> dict[str, float]:
    """Estimate the steady-state lag left behind by a residual-based stopping rule.

    A residual ``du`` measured over ``check_every`` steps means the solution is
    still moving toward the fixed point; if the slowest active mode decays at
    ``lam`` per step the remaining error is ``du / (1 - exp(-lam*check_every))``.
    On fine grids ``lam ~ 2 nu k^2`` can be so small that this lag is *comparable
    to the discretisation error itself* -- which silently corrupts an observed
    order.  Reporting it makes the corruption impossible to miss.
    """
    lam_eff = max(lam, 1e-300) * max(check_every, 1)
    lag = residual / max(1.0 - math.exp(-lam_eff), 1e-300)
    return {
        "decay_rate_per_step": float(lam),
        "lag_abs": float(lag),
        "lag_rel_to_u_scale": float(lag / u_scale) if u_scale > 0 else float("inf"),
    }


def local_orders(hs: Sequence[float], es: Sequence[float]) -> list[float]:
    """Consecutive pairwise orders, indexed by the coarse level."""
    out = []
    for i in range(len(hs) - 1):
        out.append(math.log(es[i] / es[i + 1]) / math.log(hs[i] / hs[i + 1]))
    return out


def study_mms(
    grids: Iterable[int] = (32, 64, 128),
    u0: float = 0.005,
    tau: float = 1.0,
    waves: int = 1,
    force: bool = True,
    init: str = "exact",
    max_steps: int = 200000,
    tol: float = 1e-14,
) -> dict[str, Any]:
    """Grid-refinement study of the manufactured solution (with or without force)."""
    levels = []
    for n in grids:
        r = run_mms_2d(
            n,
            u0=u0,
            tau=tau,
            waves=waves,
            force=force,
            init=init,
            max_steps=max_steps,
            tol=tol,
        )
        levels.append(
            {
                "name": f"N{n}",
                "n": int(r["n"]),
                "h": float(r["h"]),
                "k": float(r["k"]),
                "l2_rel": float(r["l2_rel"]),
                "linf_rel": float(r["linf_rel"]),
                "steps": int(r["steps"]),
                "residual": float(r["residual"]),
                "converged": bool(r["converged"]),
                "mass_drift": float(r["mass_drift"]),
                "div_l2": float(r["div_l2"]),
            }
        )
    hs = [1.0 / lv["n"] for lv in levels]  # h = 1 lattice unit; refinement = n
    es = [lv["l2_rel"] for lv in levels]
    nu_local = CS2 * (tau - 0.5)
    for lv in levels:
        lam = 2.0 * nu_local * lv["k"] ** 2
        lv["transient_lag"] = transient_lag(lv["residual"], lam, 25, u0)
    loc = local_orders(hs, es)
    for i, lv in enumerate(levels[:-1]):
        lv["local_order_to_next"] = loc[i]
    fit = observed_order(hs, es)
    return {
        "case": "mms_taylor_green_guo" if force else "mms_taylor_green_no_force",
        "force": bool(force),
        "parameters": {
            "u0": u0,
            "tau": tau,
            "nu": CS2 * (tau - 0.5),
            "waves": waves,
            "lattice": "D2Q9",
            "collision": "BGK",
            "dtype": "float64",
            "stepper": "guo_half_force",
            "init": init,
        },
        "levels": levels,
        "order_fit": fit,
        "order_fit_independent": observed_order_independent(hs, es),
        "pairwise": observed_order_pairwise(hs, es),
        "diagnostics": {
            "local_orders": loc,
            "loglog_fit_rms_residual": fit["rms_log_residual"],
            "note": (
                "loglog_fit_rms_residual measures how well a single pure power law fits "
                "the data; it is a physics diagnostic (higher-order terms present), not a "
                "recomputation residual."
            ),
        },
    }


def study_poiseuille(
    grids: Iterable[int] = (24, 48, 96),
    q: float = 0.25,
    wall_model: str = "bouzidi",
    reference: str = "true_wall",
    u_max: float = 0.01,
    tau: float = 1.0,
    max_steps: int = 400000,
    tol: float = 1e-12,
    init: str = "exact",
    g: float | None = None,
) -> dict[str, Any]:
    """Grid-refinement study of plane Poiseuille for one wall model.

    ``reference``:
      * ``"true_wall"``       -- exact profile between the *physical* walls;
      * ``"effective_wall"``  -- exact profile between the half-way-bounce-back
        effective walls ``y_w = y_1 - 0.5*h`` and ``y_N + 0.5*h`` (the
        ``y_w = 0.5 delta`` convention).
    """
    if reference not in {"true_wall", "effective_wall"}:
        raise ValueError(f"unknown reference {reference!r}")
    nu = CS2 * (tau - 0.5)
    levels = []
    for n_f in grids:
        height = (n_f - 1) + 2.0 * q
        g_level = g if g is not None else 8.0 * nu * u_max / (height * height)
        r = run_poiseuille_2d(
            n_f,
            q=q,
            wall_model=wall_model,
            g=g_level,
            tau=tau,
            init=init,
            max_steps=max_steps,
            tol=tol,
        )
        y = r["y"]
        if reference == "true_wall":
            ue = poiseuille_exact(y, g_level, nu, r["y_left_true"], r["y_right_true"])
            walls = (r["y_left_true"], r["y_right_true"])
        else:
            ue = poiseuille_exact(
                y, g_level, nu, r["y_left_effective_halfway"], r["y_right_effective_halfway"]
            )
            walls = (r["y_left_effective_halfway"], r["y_right_effective_halfway"])
        err = float(np.sqrt(np.sum((r["ux"] - ue) ** 2) / np.sum(ue**2)))
        wall_fit = fit_effective_wall(y, r["ux"], g_level, nu)
        levels.append(
            {
                "name": f"N{n_f}",
                "n_f": int(n_f),
                "h": 1.0,
                "refinement_ratio": None,
                "l2_rel": err,
                "linf_rel": float(np.max(np.abs(r["ux"] - ue)) / np.max(np.abs(ue))),
                "g": float(g_level),
                "height": float(r["height_true"]),
                "steps": int(r["steps"]),
                "residual": float(r["residual"]),
                "wall_left_fit": wall_fit["wall_left_fit"],
                "wall_right_fit": wall_fit["wall_right_fit"],
                "wall_offset_left": wall_fit["wall_left_fit"] - walls[0],
                "wall_offset_right": wall_fit["wall_right_fit"] - walls[1],
                "u_max_numeric": float(np.max(r["ux"])),
                "u_max_exact": float(np.max(ue)),
            }
        )
    # h decreases like 1/n_f: use 1/n_f as the refinement parameter
    hs = [1.0 / lv["n_f"] for lv in levels]
    es = [lv["l2_rel"] for lv in levels]
    lam_level = nu * (math.pi / max(lv["height"] for lv in levels)) ** 2
    for lv in levels:
        lv["transient_lag"] = transient_lag(lv["residual"], lam_level, 50, u_max)
    loc = local_orders(hs, es)
    for i, lv in enumerate(levels[:-1]):
        lv["local_order_to_next"] = loc[i]
    fit = observed_order(hs, es)
    return {
        "case": f"poiseuille_{wall_model}_ref_{reference}",
        "wall_model": wall_model,
        "reference": reference,
        "q": float(q),
        "parameters": {
            "tau": tau,
            "nu": nu,
            "u_max_target": u_max,
            "lattice": "D2Q9",
            "collision": "BGK",
            "dtype": "float64",
            "wall_q": float(q),
            "reference_walls": reference,
        },
        "levels": levels,
        "order_fit": fit,
        "order_fit_independent": observed_order_independent(hs, es),
        "pairwise": observed_order_pairwise(hs, es),
        "diagnostics": {
            "local_orders": loc,
            "loglog_fit_rms_residual": fit["rms_log_residual"],
            "note": "wall-position and lattice-truncation errors are not pure h^p; see report",
        },
    }


# --------------------------------------------------------------------------------------
# evidence assembly (adapter consumed by the V&V gate)
# --------------------------------------------------------------------------------------
def _sha256(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _criterion(cid: str, description: str, value: float, lo: float, hi: float) -> dict[str, Any]:
    return {
        "id": cid,
        "description": description,
        "value": float(value),
        "bounds": {"min": float(lo), "max": float(hi)},
        "pass": bool(lo <= float(value) <= hi),
    }


def build_evidence(
    mms_grids: Iterable[int] = (32, 64, 128),
    pois_grids: Iterable[int] = (12, 24, 48, 96),
    u0: float = 0.003,
    u_max: float = 0.01,
    tau: float = 1.0,
    q: float = 0.25,
    tol_mms: float = 1e-11,
    tol_pois: float = 1e-10,
    max_steps_mms: int = 400000,
    max_steps_pois: int = 400000,
    consistency_check: bool = True,
) -> dict[str, Any]:
    """Run every configuration and assemble the machine-checkable evidence record.

    Criteria (identifiers follow the campaign node metadata):

    * ``a`` MMS, force on, three grids, L2 order in ``[1.9, 2.1]`` and not at the
      round-off floor (finest error > 1e3 x float64 eps);
    * ``b`` ``|p - 2| <= 0.1``;
    * ``c1`` negative control, force off: observed order <= 1.2;
    * ``c2`` negative control, half-way bounce-back (fixed ``y_w = 0.5 delta``)
      against the true-wall exact profile: observed order <= 1.2;
    * ``d1`` Bouzidi Poiseuille: relative L2 error at the finest grid < 1%;
    * ``d2`` Bouzidi Poiseuille: observed order >= 1.9;
    * ``e`` adapter vs independent order recomputation: ``|dp| < 1e-12``;
    * ``f`` independent implementation residual ``<= 1e-12``;
    * ``floor`` finest MMS error well above float64 round-off.
    """
    t_start = time.time()
    mms = study_mms(mms_grids, u0=u0, tau=tau, force=True, tol=tol_mms, max_steps=max_steps_mms)
    mms_neg = study_mms(
        mms_grids, u0=u0, tau=tau, force=False, tol=tol_mms, max_steps=max_steps_mms
    )
    pois_bfl = study_poiseuille(
        pois_grids,
        q=q,
        wall_model="bouzidi",
        reference="true_wall",
        u_max=u_max,
        tau=tau,
        tol=tol_pois,
        max_steps=max_steps_pois,
    )
    pois_hbb = study_poiseuille(
        pois_grids,
        q=q,
        wall_model="halfway",
        reference="true_wall",
        u_max=u_max,
        tau=tau,
        tol=tol_pois,
        max_steps=max_steps_pois,
    )
    pois_hbb_eff = study_poiseuille(
        pois_grids,
        q=q,
        wall_model="halfway",
        reference="effective_wall",
        u_max=u_max,
        tau=tau,
        tol=tol_pois,
        max_steps=max_steps_pois,
    )

    # --- consistency: the steady state must not depend on how it was reached ---
    consistency: dict[str, Any] = {}
    if consistency_check:
        n_f = int(pois_grids[0]) if isinstance(pois_grids, (list, tuple)) else 24
        height = (n_f - 1) + 2.0 * q
        nu = CS2 * (tau - 0.5)
        g = 8.0 * nu * u_max / (height * height)
        # tight tolerance: the residual stopping rule leaves a lag ~ du/lam, and the
        # comparison below must see the steady state, not the transient
        tol_consistency = 1e-13
        a = run_poiseuille_2d(
            n_f, q=q, wall_model="bouzidi", g=g, tau=tau, init="exact", tol=tol_consistency
        )
        b = run_poiseuille_2d(
            n_f, q=q, wall_model="bouzidi", g=g, tau=tau, init="rest", tol=tol_consistency
        )
        dmax = float(np.max(np.abs(a["ux"] - b["ux"])))
        consistency["poiseuille_steady_state_init_independence"] = {
            "n_f": n_f,
            "init_exact_steps": a["steps"],
            "init_rest_steps": b["steps"],
            "max_profile_difference": dmax,
            "u_max": float(np.max(a["ux"])),
            "relative_difference": dmax / float(np.max(a["ux"])),
            "pass": bool(dmax / float(np.max(a["ux"])) < 1e-8),
        }

    eps = np.finfo(np.float64).eps
    p_mms = mms["order_fit"]["p"]
    p_mms_ind = mms["order_fit_independent"]["p"]
    finest_mms = min(lv["l2_rel"] for lv in mms["levels"])
    p_bfl = pois_bfl["order_fit"]["p"]
    p_bfl_ind = pois_bfl["order_fit_independent"]["p"]
    p_hbb = pois_hbb["order_fit"]["p"]
    p_neg = mms_neg["order_fit"]["p"]

    criteria = [
        _criterion(
            "a", "MMS(Guo force) L2 observed order, three grids, not at floor", p_mms, 1.9, 2.1
        ),
        _criterion("b", "|p_MMS - 2| <= 0.1", abs(p_mms - 2.0), 0.0, 0.1),
        _criterion("c1", "negative control: force off, observed order <= 1.2", p_neg, -5.0, 1.2),
        _criterion(
            "c2",
            "negative control: half-way bounce-back vs true wall, order <= 1.2",
            p_hbb,
            -5.0,
            1.2,
        ),
        _criterion(
            "d1",
            "Bouzidi Poiseuille relative L2 error at finest grid < 1%",
            100.0 * pois_bfl["levels"][-1]["l2_rel"],
            0.0,
            1.0,
        ),
        _criterion("d2", "Bouzidi Poiseuille observed order >= 1.9", p_bfl, 1.9, 6.0),
        _criterion(
            "e",
            "adapter vs independent order recomputation, max |delta p| < 1e-12",
            max(abs(p_mms - p_mms_ind), abs(p_bfl - p_bfl_ind)),
            -1.0,
            1e-12,
        ),
        _criterion(
            "f",
            "order fit recomputed by an independent implementation: max |p_I - p_II| over ALL cases <= 1e-12",
            max(
                abs(c["order_fit"]["p"] - c["order_fit_independent"]["p"])
                for c in (mms, mms_neg, pois_bfl, pois_hbb, pois_hbb_eff)
            ),
            -1.0,
            1e-12,
        ),
        _criterion(
            "floor",
            "finest MMS error / float64 eps (must exceed 1e3 to rule out a round-off floor)",
            finest_mms / eps,
            1e3,
            1e18,
        ),
    ]
    if consistency:
        criteria.append(
            {
                "id": "consistency",
                "description": "Poiseuille steady state independent of the initial state (exact vs rest)",
                "value": consistency["poiseuille_steady_state_init_independence"][
                    "relative_difference"
                ],
                "bounds": {"min": -1.0, "max": 1e-8},
                "pass": consistency["poiseuille_steady_state_init_independence"]["pass"],
            }
        )

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case": "lbm_second_order_convergence_m7059",
        "mission": "m7059",
        "generator": {
            "module": "tensorlbm.convergence_mms",
            "entry_point": "python -m tensorlbm.convergence_mms --out <path>",
            "dtype": "float64",
            "backend": "numpy",
            "deterministic": True,
            "wall_clock_s": None,
        },
        "protocol": {
            "mms_grids": list(mms_grids),
            "poiseuille_grids": list(pois_grids),
            "poiseuille_wall_fraction_q": q,
            "u0": u0,
            "u_max": u_max,
            "tau": tau,
            "nu": CS2 * (tau - 0.5),
            "tol_mms": tol_mms,
            "tol_poiseuille": tol_pois,
            "criteria_predeclared": True,
        },
        "cases": {
            "mms_force": {
                k: mms[k]
                for k in (
                    "case",
                    "parameters",
                    "levels",
                    "order_fit",
                    "order_fit_independent",
                    "pairwise",
                    "diagnostics",
                )
            },
            "mms_no_force": {
                k: mms_neg[k]
                for k in (
                    "case",
                    "parameters",
                    "levels",
                    "order_fit",
                    "order_fit_independent",
                    "pairwise",
                    "diagnostics",
                )
            },
            "poiseuille_bouzidi_true_wall": {
                k: pois_bfl[k]
                for k in (
                    "case",
                    "parameters",
                    "levels",
                    "order_fit",
                    "order_fit_independent",
                    "pairwise",
                    "diagnostics",
                )
            },
            "poiseuille_halfway_true_wall": {
                k: pois_hbb[k]
                for k in (
                    "case",
                    "parameters",
                    "levels",
                    "order_fit",
                    "order_fit_independent",
                    "pairwise",
                    "diagnostics",
                )
            },
            "poiseuille_halfway_effective_wall": {
                k: pois_hbb_eff[k]
                for k in (
                    "case",
                    "parameters",
                    "levels",
                    "order_fit",
                    "order_fit_independent",
                    "pairwise",
                    "diagnostics",
                )
            },
        },
        "consistency": consistency,
        "criteria": criteria,
        "overall_pass": all(c["pass"] for c in criteria),
        "notes": {
            "criterion_e": (
                "The m5740 CLI / its emitted JSON artifact were not present in this worktree, in the "
                "TensorLBM git history, or elsewhere on this machine when this node ran (searched: "
                "git log --all --grep, git grep across refs, filesystem). Criterion (e) is therefore "
                "enforced in two machine-checkable halves instead of by literal schema diff against "
                "m5740: (i) validate_evidence() is fail-closed on the declared field set "
                "(case/levels[].h, levels[].l2_rel, order_fit.p, criteria[].pass, ...), which is the "
                "convergence-evidence convention the platform V&V gate consumes; (ii) every numeric "
                "order field is recomputed by a second, independent implementation (closed-form "
                "normal equations in pure Python, no NumPy) and must agree to < 1e-12. If the m5740 "
                "schema is recovered, only the field-name mapping in this adapter needs updating -- "
                "no physics re-run."
            ),
            "wall_position": (
                "The half-way bounce-back runs report the wall position recovered from the simulated "
                "profile itself (fit_effective_wall). The recovered offset equals -(0.5 - q) cell, i.e. "
                "the scheme is not wrong, its wall is simply fixed at y_w = 0.5 Delta while the "
                "geometry says q."
            ),
        },
    }
    evidence["generator"]["wall_clock_s"] = round(time.time() - t_start, 2)
    evidence["audit"] = {
        "payload_sha256": _sha256(evidence),
        "reproduce": (
            "PYTHONPATH=src python -m tensorlbm.convergence_mms "
            "--out artifacts/lbm_convergence_m7059/convergence_evidence.json"
        ),
    }
    return evidence


# --------------------------------------------------------------------------------------
# adapter validation (criterion e/f: schema completeness + recompute fidelity)
# --------------------------------------------------------------------------------------
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "case",
    "mission",
    "generator",
    "protocol",
    "cases",
    "criteria",
    "overall_pass",
    "audit",
)
OPTIONAL_TOP_LEVEL = ("notes", "consistency", "regression")
REQUIRED_CASE_FIELDS = (
    "case",
    "parameters",
    "levels",
    "order_fit",
    "order_fit_independent",
    "pairwise",
)
REQUIRED_LEVEL_FIELDS = ("name", "h", "l2_rel", "linf_rel", "steps", "residual")


def validate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed structural validation of an evidence record.

    Checks (1) top-level key set exactly equals ``REQUIRED_TOP_LEVEL`` plus
    ``notes``; (2) every case carries ``REQUIRED_CASE_FIELDS``; (3) every level
    carries ``REQUIRED_LEVEL_FIELDS``; (4) the producer / independent order fits
    agree to ``< 1e-12``; (5) the recorded ``payload_sha256`` matches a
    recomputation from the decoded record.
    """
    problems: list[str] = []
    extra = set(evidence) - set(REQUIRED_TOP_LEVEL) - set(OPTIONAL_TOP_LEVEL)
    missing = set(REQUIRED_TOP_LEVEL) - set(evidence)
    if missing:
        problems.append(f"missing top-level keys: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected top-level keys: {sorted(extra)}")

    worst = 0.0
    for name, case in evidence.get("cases", {}).items():
        missing_c = set(REQUIRED_CASE_FIELDS) - set(case)
        if missing_c:
            problems.append(f"{name}: missing case fields {sorted(missing_c)}")
            continue
        for lv in case["levels"]:
            missing_l = set(REQUIRED_LEVEL_FIELDS) - set(lv)
            if missing_l:
                problems.append(
                    f"{name}/{lv.get('name')}: missing level fields {sorted(missing_l)}"
                )
        d = abs(float(case["order_fit"]["p"]) - float(case["order_fit_independent"]["p"]))
        worst = max(worst, d)
        if d > 1e-12:
            problems.append(f"{name}: producer/independent order disagree by {d:.3e}")

    payload = {k: v for k, v in evidence.items() if k != "audit"}
    stored = evidence.get("audit", {}).get("payload_sha256")
    recomputed = _sha256(payload)
    if stored != recomputed:
        problems.append("audit.payload_sha256 mismatch")

    return {
        "ok": not problems,
        "problems": problems,
        "max_order_disagreement": worst,
        "schema_version": evidence.get("schema_version"),
        "payload_sha256": recomputed,
    }


def attach_regression(
    evidence: dict[str, Any],
    baseline: dict[str, int],
    after: dict[str, int],
    commands: dict[str, str],
    note: str,
) -> dict[str, Any]:
    """Attach the criterion-(g) regression ledger and refresh the audit hash.

    ``baseline`` / ``after`` are pytest summaries (``passed``/``failed``/``error``/
    ``skipped``) for the same test selection run before and after this change.
    The criterion is "pass count must not decrease" -- the honest and checkable
    form of "no regression".
    """
    reg = {
        "selection": commands.get("selection", ""),
        "baseline_command": commands.get("baseline_command", ""),
        "after_command": commands.get("after_command", ""),
        "baseline": baseline,
        "after": after,
        "delta_passed": int(after.get("passed", 0) - baseline.get("passed", 0)),
        "delta_failed": int(after.get("failed", 0) - baseline.get("failed", 0)),
        "note": note,
        "pass": bool(
            after.get("passed", 0) >= baseline.get("passed", 0)
            and after.get("failed", 0) <= baseline.get("failed", 0)
        ),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    evidence["regression"] = reg
    evidence["criteria"] = [c for c in evidence["criteria"] if c["id"] != "g"]
    evidence["criteria"].append(
        {
            "id": "g",
            "description": "regression pass count not decreased (same pytest selection before/after)",
            "value": float(reg["delta_passed"]),
            "bounds": {"min": 0.0, "max": 1e9},
            "pass": reg["pass"],
        }
    )
    evidence["overall_pass"] = all(c["pass"] for c in evidence["criteria"])
    payload = {k: v for k, v in evidence.items() if k != "audit"}
    evidence["audit"] = {
        "payload_sha256": _sha256(payload),
        "reproduce": evidence.get("audit", {}).get(
            "reproduce",
            "PYTHONPATH=src python -m tensorlbm.convergence_mms --out <path>",
        ),
    }
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="TensorLBM second-order convergence evidence")
    ap.add_argument("--out", default="artifacts/lbm_convergence_m7059/convergence_evidence.json")
    ap.add_argument("--mms-grids", default="32,64,128")
    ap.add_argument("--pois-grids", default="12,24,48,96")
    ap.add_argument("--u0", type=float, default=0.003)
    ap.add_argument("--u-max", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=0.25)
    args = ap.parse_args(argv)

    mms_grids = tuple(int(v) for v in args.mms_grids.split(","))
    pois_grids = tuple(int(v) for v in args.pois_grids.split(","))
    ev = build_evidence(
        mms_grids=mms_grids,
        pois_grids=pois_grids,
        u0=args.u0,
        u_max=args.u_max,
        tau=args.tau,
        q=args.q,
    )
    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=2))
    check = validate_evidence(json.loads(out.read_text()))
    print(
        json.dumps(
            {"out": str(out), "overall_pass": ev["overall_pass"], "validate": check}, indent=2
        )
    )
    return 0 if ev["overall_pass"] and check["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
