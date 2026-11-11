"""Spurious-current suppression benchmark for two-component Shan-Chen LBM.

Physical problem
----------------
A static droplet at rest in a periodic box is the canonical *null* test of a
multiphase lattice Boltzmann model: with no external forcing the exact solution
is ``u = 0`` everywhere.  Any non-zero velocity left behind is the **spurious
(parasitic) current** ``U_spurious``, produced by the discrete imbalance between
the interaction force and the pressure gradient at a curved interface.
``U_spurious`` sets a floor on the resolved velocity scale and therefore bounds
what a simulation can predict for capillary-driven flows (bubble rise, sloshing,
ice-water interfaces ...).

This module supplies a small, self-contained D2Q9 two-component Shan-Chen
engine in which the knobs the literature uses to suppress spurious currents are
switchable, so that they can be compared on the *same* code path:

``force``
    ``"lattice"``   the standard Shan-Chen lattice sum
                    ``F_a = -G12 psi_a sum_i w_i psi_b(x+e_i) e_i``;
    ``"geometric"`` the interaction force reconstructed from an isotropic
                    4th-order finite-difference gradient of the pseudopotential,
                    ``F_a = -G12 psi_a c_s^2 grad_4(psi_b)``.  The lattice sum is
                    this gradient plus an anisotropic ``O(h^3)`` truncation
                    error (see :func:`lattice_gradient`); the reconstruction
                    removes the anisotropic part, which is the geometric source
                    of the parasitic vortices.

``collision``
    ``"bgk"``   single relaxation time;
    ``"mrt"``   multi-relaxation-time (Lallemand-Luo moments) with the energy
                mode ``s_e`` set from the "magic" parameter
                ``Lambda = (1/s_nu - 1/2)(1/s_e - 1/2) = 3/16``.

Forcing uses the Guo scheme for **both** force variants, so the comparison
isolates the force discretisation rather than the forcing implementation.

Conventions (this library)
--------------------------
* ``G12 > 0`` is repulsive between unlike components.  The library's
  pseudopotential modules use the opposite sign (``G_12 < 0`` repulsive), so
  ``G12 = -G_12`` when comparing to :mod:`tensorlbm.multiphase` /
  :mod:`tensorlbm.mcmp_verification`.
* The pseudopotential is **linear**, ``psi(rho) = rho`` (the density field is
  used directly), matching the MCMP path of this library.
* Two-dimensional droplet = cylinder, so the Young-Laplace law is
  ``dp = sigma / R`` (not ``2 sigma / R``).
* Everything is ``float64`` so that the ``sigma = 0`` control can be asserted at
  the ``1e-10`` level instead of being masked by single-precision round-off.
* The engine is **numpy**, not torch, on purpose: measured on the loong64
  worker, a ``(9,96,96)`` float64 elementwise op costs 0.66 ms under torch and
  0.13 ms under numpy (float64 SIMD is missing from the torch build), which is
  the difference between a 30-minute and a 4-minute parameter sweep.  Torch
  hosts can convert with ``torch.from_numpy``; nothing in the physics depends on
  the array library.

Provenance: node ``p362w2-n06`` (TensorLBM spurious-current suppression
benchmark).  The baseline force/collision formulas follow Shan & Chen (1993),
Sukop & Thorne (2006) and Lallemand & Luo (2000); the isotropic-gradient idea
follows Sbragaglia et al. (2007) (and the EDM family of Kupershtokh).  All
formulas are re-derived here for this library's sign and linear-psi conventions
rather than copied.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

CS2 = 1.0 / 3.0

#: D2Q9 direction order (Lallemand-Luo moment ordering).
EX: tuple[int, ...] = (0, 1, -1, 0, 0, 1, -1, 1, -1)
EY: tuple[int, ...] = (0, 0, 0, 1, -1, 1, -1, -1, 1)
WEIGHTS: tuple[float, ...] = (
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

DTYPE = np.float64

W9 = np.asarray(WEIGHTS, dtype=DTYPE).reshape(9, 1, 1)
EX9 = np.asarray(EX, dtype=DTYPE).reshape(9, 1, 1)
EY9 = np.asarray(EY, dtype=DTYPE).reshape(9, 1, 1)


# ---------------------------------------------------------------------------
# Lattice primitives
# ---------------------------------------------------------------------------


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Second-order Maxwell-Boltzmann equilibrium, shape ``(9, ny, nx)``."""
    eu = EX9 * ux + EY9 * uy
    usq = ux * ux + uy * uy
    return W9 * rho * (1.0 + eu / CS2 + 0.5 * (eu / CS2) ** 2 - 0.5 * usq / CS2)


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rho, ux, uy)`` from raw populations (no force correction)."""
    rho = f.sum(axis=0)
    jx = (f * EX9).sum(axis=0)
    jy = (f * EY9).sum(axis=0)
    safe = np.maximum(rho, 1e-30)
    return rho, jx / safe, jy / safe


def stream(f: np.ndarray) -> np.ndarray:
    """Pull streaming on a periodic torus: ``f_i(x) <- f_i(x - e_i)``."""
    out = np.empty_like(f)
    for i in range(9):
        out[i] = np.roll(f[i], (EY[i], EX[i]), axis=(0, 1))
    return out


def shift(field: np.ndarray, i: int) -> np.ndarray:
    """``field(x + e_i)`` under periodic wrap."""
    return np.roll(field, (-EY[i], -EX[i]), axis=(0, 1))


# ---------------------------------------------------------------------------
# Gradients of the pseudopotential: lattice sum versus isotropic reconstruction
# ---------------------------------------------------------------------------


def lattice_gradient(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Shan-Chen lattice sum ``sum_i w_i psi(x+e_i) e_i``.

    For a smooth field this equals ``c_s^2 grad(psi)`` plus an anisotropic
    ``O(h^3)`` truncation term.  That truncation is exactly what a *geometric*
    reconstruction removes.
    """
    gx = np.zeros_like(psi)
    gy = np.zeros_like(psi)
    for i in range(1, 9):
        shifted = shift(psi, i)
        gx = gx + WEIGHTS[i] * EX[i] * shifted
        gy = gy + WEIGHTS[i] * EY[i] * shifted
    return gx, gy


def gradient_4th(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fourth-order central gradient on the square grid (periodic).

    ``d_x f = [8 (f_{x+1} - f_{x-1}) - (f_{x+2} - f_{x-2})] / 12`` and likewise
    for ``y``.  Exact for cubic polynomials, so the leading anisotropic error of
    :func:`lattice_gradient` is eliminated.
    """
    gx = (
        8.0 * (np.roll(field, -1, 1) - np.roll(field, 1, 1))
        - (np.roll(field, -2, 1) - np.roll(field, 2, 1))
    ) / 12.0
    gy = (
        8.0 * (np.roll(field, -1, 0) - np.roll(field, 1, 0))
        - (np.roll(field, -2, 0) - np.roll(field, 2, 0))
    ) / 12.0
    return gx, gy


def interaction_force(
    psi_self: np.ndarray,
    psi_other: np.ndarray,
    G12: float,
    *,
    scheme: str = "lattice",
) -> tuple[np.ndarray, np.ndarray]:
    """Force density on one component from the other, ``G12 > 0`` repulsive.

    ``F = -G12 psi_self grad(psi_other)`` where ``grad`` is either the lattice
    sum (:func:`lattice_gradient`, ``scheme="lattice"``) or the geometric
    reconstruction ``c_s^2`` times the 4th-order isotropic stencil
    (:func:`gradient_4th`, ``scheme="geometric"``).
    """
    if scheme == "lattice":
        gx, gy = lattice_gradient(psi_other)
    elif scheme in ("geometric", "geo"):
        gx, gy = gradient_4th(psi_other)
        gx, gy = CS2 * gx, CS2 * gy
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"unknown force scheme {scheme!r}")
    return -G12 * psi_self * gx, -G12 * psi_self * gy


# ---------------------------------------------------------------------------
# Collision operators (BGK and MRT) with Guo forcing
# ---------------------------------------------------------------------------

#: Lallemand-Luo moment matrix for the direction order used above.
_M = (
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0),
    (4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0),
    (0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0),
    (0.0, -2.0, 2.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0),
    (0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
    (0.0, 0.0, 0.0, -2.0, 2.0, 1.0, -1.0, -1.0, 1.0),
    (0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0),
)

MOMENT_MATRIX = np.asarray(_M, dtype=DTYPE)
MOMENT_MATRIX_INV = np.linalg.inv(MOMENT_MATRIX)


def _mat_apply(mat: np.ndarray, fields: np.ndarray) -> np.ndarray:
    """``mat @ fields`` over the population axis, without tensordot's reordering.

    ``fields`` has shape ``(9, ny, nx)``; the population axis is contiguous, so a
    plain ``matmul`` on the flattened array is several times faster than
    ``np.tensordot`` (which transposes a ``ny*nx*9`` copy), and it matters: MRT
    applies three of these per component per step.
    """
    flat = fields.reshape(9, -1)
    return (mat @ flat).reshape(fields.shape)


def _to_moments(f: np.ndarray) -> np.ndarray:
    return _mat_apply(MOMENT_MATRIX, f)


def _from_moments(m: np.ndarray) -> np.ndarray:
    return _mat_apply(MOMENT_MATRIX_INV, m)


def _force_source(
    ux: np.ndarray, uy: np.ndarray, Fx: np.ndarray, Fy: np.ndarray
) -> np.ndarray:
    """Guo forcing source ``S_i = w_i [(e_i-u)/c_s^2 + (e_i.u)e_i/c_s^4].F``."""
    eu = EX9 * ux + EY9 * uy
    eF = EX9 * Fx + EY9 * Fy
    return W9 * (((EX9 - ux) * Fx + (EY9 - uy) * Fy) / CS2 + eu * eF / CS2**2)


def magic_s_e(s_nu: float, magic: float = 3.0 / 16.0) -> float:
    """Energy relaxation rate giving the requested magic parameter ``Lambda``.

    ``Lambda = (1/s_nu - 1/2) (1/s_e - 1/2)``; ``Lambda = 3/16`` is the value
    that makes the MRT operator third-order accurate for the D2Q9 stencil and is
    the standard choice used here.
    """
    a = 1.0 / s_nu - 0.5
    if a <= 0.0:
        raise ValueError(f"s_nu must be < 2 (tau > 0.5), got s_nu={s_nu}")
    return 1.0 / (0.5 + magic / a)


def collide(
    f: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
    *,
    tau: float = 1.0,
    collision: str = "bgk",
    magic: float = 3.0 / 16.0,
    s_eps: float = 1.0,
    s_q: float | None = None,
    forcing: str = "guo",
) -> np.ndarray:
    """Post-collision populations with Guo forcing (BGK or MRT).

    ``s_q`` overrides the ghost-mode rate (default ``8(2-s_nu)/(8-s_nu)``); with
    ``s_eps = s_q = 1/tau`` and ``magic = (1/tau - 1/2)^2`` (so ``s_e = 1/tau``)
    the MRT operator must reproduce BGK to round-off, which is asserted in the
    test suite.
    """
    rho = f.sum(axis=0)
    safe = np.maximum(rho, 1e-30)
    jx = (f * EX9).sum(axis=0)
    jy = (f * EY9).sum(axis=0)
    omega = 1.0 / tau
    if forcing == "edm":
        # Kupershtokh's exact difference method: the node velocity is j/rho and
        # the force enters as an exact finite momentum increment F/rho on the
        # equilibrium, with no Guo source term.
        ux = jx / safe
        uy = jy / safe
        feq = equilibrium(rho, ux, uy)
        feq_shift = equilibrium(rho, ux + Fx / safe, uy + Fy / safe)
        if collision == "bgk":
            return f - omega * (f - feq) + (feq_shift - feq)
        raise ValueError("EDM forcing is implemented for the BGK collision")
    if forcing != "guo":
        raise ValueError(f"unknown forcing {forcing!r}")
    # Guo: the momentum carried by the node is j + F/2.
    ux = (jx + 0.5 * Fx) / safe
    uy = (jy + 0.5 * Fy) / safe
    feq = equilibrium(rho, ux, uy)
    src = _force_source(ux, uy, Fx, Fy)
    if collision == "bgk":
        return f - omega * (f - feq) + (1.0 - 0.5 * omega) * src
    if collision == "mrt":
        s_nu = omega
        s_e = magic_s_e(s_nu, magic)
        s_q_eff = (
            float(s_q) if s_q is not None else 8.0 * (2.0 - s_nu) / (8.0 - s_nu)
        )
        s = np.asarray(
            [0.0, s_e, s_eps, 0.0, s_q_eff, 0.0, s_q_eff, s_nu, s_nu], dtype=DTYPE
        )
        m = _to_moments(f)
        meq = _to_moments(feq)
        msrc = _to_moments(src)
        mpost = m - s.reshape(9, 1, 1) * (m - meq) + (1.0 - 0.5 * s.reshape(9, 1, 1)) * msrc
        return _from_moments(mpost)
    raise ValueError(f"unknown collision {collision!r}")


# ---------------------------------------------------------------------------
# Two-component static droplet: initialisation and time stepping
# ---------------------------------------------------------------------------


def grid_radius(n: int) -> np.ndarray:
    """Distance of every cell from the domain centre ``((n-1)/2, (n-1)/2)``."""
    idx = np.arange(n, dtype=DTYPE)
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    centre = (n - 1) / 2.0
    return np.sqrt((xx - centre) ** 2 + (yy - centre) ** 2)


def initialise_droplet(
    n: int,
    radius: float,
    width: float,
    *,
    rho1_in: float = 0.92,
    rho1_out: float = 0.14,
    rho2_in: float = 0.08,
    rho2_out: float = 0.86,
    uniform: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Tanh-profiled circular droplet of component 1 in a component-2 bath at rest.

    ``s = 0.5 (1 - tanh((r - R)/width))`` and
    ``rho = rho_out + s (rho_in - rho_out)``; both components start from the
    local equilibrium at ``u = 0``.  A staircase initialisation is deliberately
    not used: at ``G12 ~ 2.5`` it injects an unbalanced force that diverges
    within a few hundred steps.

    ``uniform=True`` returns the uniform mixture (``rho1 = rho1_in``,
    ``rho2 = rho2_in``) instead.  That is the geometry-free control used to prove
    ``sigma = 0 => U_spurious = 0``: with no interface there is no capillary
    force, so the exact solution ``u = 0`` must be preserved to round-off.
    """
    if uniform:
        rho1 = np.full((n, n), float(rho1_in), dtype=DTYPE)
        rho2 = np.full((n, n), float(rho2_in), dtype=DTYPE)
    else:
        r = grid_radius(n)
        s = 0.5 * (1.0 - np.tanh((r - radius) / max(width, 1e-6)))
        rho1 = rho1_out + s * (rho1_in - rho1_out)
        rho2 = rho2_out + s * (rho2_in - rho2_out)
    zero = np.zeros_like(rho1)
    return equilibrium(rho1, zero, zero), equilibrium(rho2, zero, zero)


def barycentric_velocity(
    f1: np.ndarray,
    f2: np.ndarray,
    F1x: np.ndarray,
    F1y: np.ndarray,
    F2x: np.ndarray,
    F2y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mass-averaged velocity including the Guo half-force of both components."""
    rho1 = f1.sum(axis=0)
    rho2 = f2.sum(axis=0)
    jx = (f1 * EX9).sum(axis=0) + (f2 * EX9).sum(axis=0) + 0.5 * (F1x + F2x)
    jy = (f1 * EY9).sum(axis=0) + (f2 * EY9).sum(axis=0) + 0.5 * (F1y + F2y)
    safe = np.maximum(rho1 + rho2, 1e-30)
    return jx / safe, jy / safe


def run_static_droplet(
    *,
    G12: float = 2.5,
    n: int = 128,
    radius: float | None = None,
    width: float = 2.0,
    steps: int = 12000,
    tau: float = 1.0,
    collision: str = "bgk",
    force: str = "lattice",
    forcing: str = "guo",
    magic: float = 3.0 / 16.0,
    stop_tol: float = 1e-4,
    check_every: int = 200,
    rho1_in: float = 0.92,
    rho1_out: float = 0.14,
    rho2_in: float = 0.08,
    rho2_out: float = 0.86,
    uniform: bool = False,
    keep_fields: bool = True,
) -> dict[str, Any]:
    """Relax a static droplet and report the spurious-current plateau.

    Convergence is declared when ``|u|_max`` changes by less than ``stop_tol``
    (relative) over two consecutive ``check_every`` windows.  A steady parasitic
    vortex is a *plateau*, not a decay, so the plateau is the physical
    measurement; ``|u|_max`` is computed on the barycentric velocity, the field a
    multiphase host code would advect with.
    """
    if radius is None:
        radius = 0.26 * n
    f1, f2 = initialise_droplet(
        n,
        radius,
        width,
        rho1_in=rho1_in,
        rho1_out=rho1_out,
        rho2_in=rho2_in,
        rho2_out=rho2_out,
        uniform=uniform,
    )
    history: list[list[float]] = []
    u_prev = float("inf")
    stable = 0
    converged = False
    steps_run = 0
    u_max = float("nan")
    for step in range(1, steps + 1):
        rho1 = f1.sum(axis=0)
        rho2 = f2.sum(axis=0)
        F1x, F1y = interaction_force(rho1, rho2, G12, scheme=force)
        F2x, F2y = interaction_force(rho2, rho1, G12, scheme=force)
        f1 = stream(
            collide(f1, F1x, F1y, tau=tau, collision=collision, magic=magic,
                    forcing=forcing)
        )
        f2 = stream(
            collide(f2, F2x, F2y, tau=tau, collision=collision, magic=magic,
                    forcing=forcing)
        )
        steps_run = step
        if step % check_every == 0 or step == 1:
            ux, uy = barycentric_velocity(f1, f2, F1x, F1y, F2x, F2y)
            u_max = float(np.sqrt(ux * ux + uy * uy).max())
            if not math.isfinite(u_max):
                history.append([float(step), float("nan")])
                break
            history.append([float(step), u_max])
            if math.isfinite(u_prev) and abs(u_max - u_prev) <= stop_tol * max(
                u_max, 1e-30
            ):
                stable += 1
                if stable >= 2:
                    converged = True
                    break
            else:
                stable = 0
            u_prev = u_max
    rho1 = f1.sum(axis=0)
    rho2 = f2.sum(axis=0)
    F1x, F1y = interaction_force(rho1, rho2, G12, scheme=force)
    F2x, F2y = interaction_force(rho2, rho1, G12, scheme=force)
    ux, uy = barycentric_velocity(f1, f2, F1x, F1y, F2x, F2y)
    out: dict[str, Any] = {
        "G12": G12,
        "n": n,
        "radius_init": radius,
        "width_init": width,
        "tau": tau,
        "collision": collision,
        "force": force,
        "forcing": forcing,
        "magic": magic,
        "steps_requested": steps,
        "steps_run": steps_run,
        "converged": converged,
        "u_max": float(np.sqrt(ux * ux + uy * uy).max()),
        "u_rms": float(np.sqrt(ux * ux + uy * uy).mean()),
        "history": history,
        "mass1": float(rho1.sum()),
        "mass2": float(rho2.sum()),
        "rho1_centre": float(rho1[n // 2, n // 2]),
        "rho1_far": float(rho1[2, 2]),
        "rho_tot_drift": float(np.sum(rho1 + rho2) - (n * n)),
    }
    if keep_fields:
        out["rho1"] = rho1
        out["rho2"] = rho2
        out["ux"] = ux
        out["uy"] = uy
    return out


# ---------------------------------------------------------------------------
# Measurements: pressure (Laplace), interface thickness, Pareto, convergence
# ---------------------------------------------------------------------------


def eos_pressure(rho1: np.ndarray, rho2: np.ndarray, G12: float) -> np.ndarray:
    """Bulk equation of state of the two-component linear-psi Shan-Chen model.

    ``p = c_s^2 (rho_1 + rho_2) - c_s^2 G12 rho_1 rho_2`` with ``G12 > 0``
    repulsive: the cross term *lowers* the pressure where both components
    coexist, which is what generates the surface tension.  In the library's
    opposite sign convention this is ``c_s^2 (rho_1+rho_2) + G_12 c_s^2
    rho_1 rho_2`` with ``G_12 = -G12``, i.e. exactly
    :func:`tensorlbm.mcmp_verification.sc_mcmp_pressure` for ``psi = rho``.
    """
    return CS2 * (rho1 + rho2) - CS2 * G12 * rho1 * rho2


def droplet_pressure(run: dict[str, Any], *, band: float = 3.0) -> dict[str, Any]:
    """Inside/outside pressure of a relaxed droplet and the Laplace difference.

    The interior mean is taken over ``r < R - band*width`` and the exterior mean
    over ``r > R + band*width`` so that no interface cell enters either average.
    """
    rho1, rho2 = run["rho1"], run["rho2"]
    n = rho1.shape[0]
    r = grid_radius(n)
    R = float(run["radius_init"])
    w = float(run["width_init"])
    press = eos_pressure(rho1, rho2, float(run["G12"]))
    inner = r < (R - band * w)
    outer = r > (R + band * w)
    if inner.sum() < 1 or outer.sum() < 1:
        raise ValueError("droplet too large for the banded pressure averages")
    p_in = float(press[inner].mean())
    p_out = float(press[outer].mean())
    return {
        "p_in": p_in,
        "p_out": p_out,
        "dp": p_in - p_out,
        "n_inner": int(inner.sum()),
        "n_outer": int(outer.sum()),
    }


def radial_profile(
    field: np.ndarray, bins: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Shell-averaged radial profile ``(r_centres, mean_value)`` around the centre.

    ``bins`` defaults to four shells per lattice unit, which keeps the bin width
    well below the thinnest interface measured here (W >= 3); a coarse 60-bin
    default biases the 10-90 thickness by several percent.
    """
    n = field.shape[0]
    r = grid_radius(n)
    rmax = (n - 1) / 2.0
    if bins is None:
        bins = int(max(60, round(4.0 * rmax)))
    edges = np.linspace(0.0, rmax, bins + 1)
    idx = np.clip(np.digitize(r.reshape(-1), edges) - 1, 0, bins - 1)
    sums = np.bincount(idx, weights=field.reshape(-1), minlength=bins)
    counts = np.bincount(idx, minlength=bins).astype(DTYPE)
    centres = 0.5 * (edges[1:] + edges[:-1])
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return centres, means


def interface_width_10_90(run: dict[str, Any]) -> dict[str, float]:
    """10-90 interface thickness from the shell-averaged density profile.

    For ``s = 0.5(1 - tanh((r-R)/w))`` the 10-90 thickness is
    ``2 atanh(0.8) w = 2.197 w``; the function reports the measured thickness and
    the equivalent tanh width ``w_eq``.
    """
    centres, prof = radial_profile(run["rho1"])
    valid = ~np.isnan(prof)
    if valid.sum() < 8:
        return {
            "width_10_90": float("nan"),
            "w_eq": float("nan"),
            "radius_fit": float("nan"),
        }
    centres, prof = centres[valid], prof[valid]
    rho_in = float(prof[0])
    rho_out = float(prof[-1])
    span = rho_in - rho_out
    if abs(span) < 1e-6:
        return {
            "width_10_90": float("nan"),
            "w_eq": float("nan"),
            "radius_fit": float("nan"),
        }
    s = (prof - rho_out) / span
    r90 = r10 = float("nan")
    for k in range(len(s) - 1):
        if s[k] >= 0.9 >= s[k + 1]:
            t = (s[k] - 0.9) / (s[k] - s[k + 1])
            r90 = centres[k] + t * (centres[k + 1] - centres[k])
        if s[k] >= 0.1 >= s[k + 1]:
            t = (s[k] - 0.1) / (s[k] - s[k + 1])
            r10 = centres[k] + t * (centres[k + 1] - centres[k])
    width = r10 - r90
    return {
        "width_10_90": float(width),
        "w_eq": float(width / (2.0 * math.atanh(0.8))),
        "radius_fit": float(0.5 * (r10 + r90)),
    }


def laplace_fit(
    radii: Sequence[float], dp: Sequence[float], *, through_origin: bool = False
) -> dict[str, Any]:
    """Least-squares fit of the 2-D Young-Laplace law ``dp = sigma/R + c``.

    Returns the surface tension ``sigma`` (slope), the intercept ``c``, the
    per-radius relative residual against the fit and the coefficient of
    determination.  ``through_origin=True`` forces ``c = 0``.
    """
    if len(radii) != len(dp) or len(radii) < 2:
        raise ValueError("need >= 2 (radius, dp) pairs")
    x = [1.0 / r for r in radii]
    y = [float(v) for v in dp]
    if through_origin:
        sigma = sum(xi * yi for xi, yi in zip(x, y)) / sum(xi * xi for xi in x)
        intercept = 0.0
    else:
        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        sxx = sum((xi - mx) ** 2 for xi in x)
        sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        sigma = sxy / sxx
        intercept = my - sigma * mx
    pred = [sigma * xi + intercept for xi in x]
    resid = [yi - pi for yi, pi in zip(y, pred)]
    rel = [abs(ri) / pi for ri, pi in zip(resid, pred)]
    ss_res = sum(ri * ri for ri in resid)
    ss_tot = sum((yi - sum(y) / len(y)) ** 2 for yi in y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "sigma": float(sigma),
        "intercept": float(intercept),
        "residual": [float(v) for v in resid],
        "rel_error": [float(v) for v in rel],
        "max_rel_error": float(max(rel)),
        "r_squared": float(r2),
        "radii": [float(r) for r in radii],
        "dp": [float(v) for v in y],
        "dp_fit": [float(v) for v in pred],
    }


def pareto_front(points: Sequence[Sequence[float]]) -> list[int]:
    """Indices of the non-dominated rows of ``points`` (all objectives minimised).

    A row ``i`` is dominated when another row is ``<=`` on every objective and
    ``<`` on at least one.  Rows with identical objective vectors are both kept.
    """
    n = len(points)
    keep: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if all(pj <= pi for pj, pi in zip(points[j], points[i])) and any(
                pj < pi for pj, pi in zip(points[j], points[i])
            ):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def convergence_order(
    values: Sequence[float], spacings: Sequence[float]
) -> dict[str, Any]:
    """Observed convergence order from successive refinements.

    ``p = log(|f1 - f2| / |f2 - f3|) / log(h1 / h2)`` with ``h`` the grid
    spacing; a non-monotone sequence is reported as such instead of being fitted.
    """
    if len(values) < 3:
        raise ValueError("need >= 3 grid levels")
    d1 = abs(values[0] - values[1])
    d2 = abs(values[1] - values[2])
    if d1 == 0.0 or d2 == 0.0:
        return {"order": float("inf"), "monotone": True, "d1": float(d1), "d2": float(d2)}
    if d2 > d1:
        return {
            "order": float("nan"),
            "monotone": False,
            "d1": float(d1),
            "d2": float(d2),
            "note": "successive differences grow; refinement path is not asymptotic",
        }
    order = math.log(d1 / d2) / math.log(spacings[0] / spacings[1])
    return {
        "order": float(order),
        "monotone": True,
        "d1": float(d1),
        "d2": float(d2),
        "levels": [float(v) for v in values],
        "spacings": [float(h) for h in spacings],
    }


#: The three suppression strategies compared in this node, plus their combination.
METHODS: dict[str, dict[str, str]] = {
    "baseline_bgk_lattice": {
        "collision": "bgk",
        "force": "lattice",
        "label": "baseline: BGK + lattice-sum force",
    },
    "mrt_lattice": {
        "collision": "mrt",
        "force": "lattice",
        "label": "MRT collision (magic Lambda=3/16) + lattice-sum force",
    },
    "bgk_geometric": {
        "collision": "bgk",
        "force": "geometric",
        "label": "BGK + geometric (isotropic 4th-order) force",
    },
    "mrt_geometric": {
        "collision": "mrt",
        "force": "geometric",
        "label": "MRT + geometric force (combined)",
    },
    "bgk_edm": {
        "collision": "bgk",
        "force": "lattice",
        "forcing": "edm",
        "label": "BGK + EDM (exact difference) forcing",
    },
}
