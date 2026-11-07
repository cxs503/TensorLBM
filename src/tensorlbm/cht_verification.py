"""Conjugate heat transfer (CHT) verification ladder: steady solid conduction
and the fluid-solid interface operator.

Part 1 of the CHT verification ladder of TensorLBM (see
:mod:`tensorlbm.cht_ladder` for the gate runner and
:mod:`tensorlbm.cht_transient` / :mod:`tensorlbm.graetz` for the transient and
Graetz anchors).

Steady solid conduction (1-D / 2-D / 3-D)
-----------------------------------------
* :func:`composite_wall` -- 1-D multi-layer wall on a conservative
  finite-volume network with harmonic-mean face conductances.  Expected to
  reproduce the analytical series resistance ``R = sum(delta_i/k_i)``; the 1-D
  series network *is* the exact discretisation of the analytical resistance
  chain, so the relative error is at round-off level (a strong statement: the
  operator cannot hide a modelling error).
* :func:`cylinder_annulus_2d` -- 2-D steady conduction in an annulus on a polar
  ``(r, theta)`` finite-volume grid, Dirichlet data imposed through exact
  cylindrical boundary conductances.  Anchors: the logarithmic profile
  ``T = T1 + (T2-T1) ln(r/r1)/ln(r2/r1)`` (axisymmetric part) and, with
  ``theta_amplitude = A``, the separable harmonic
  ``A cos(theta) (r - r2^2/r)/(r1 - r2^2/r1)`` which exercises the angular part
  of the polar Laplacian.
* :func:`sphere_shell` -- 3-D steady conduction in a (composite) spherical
  shell with ``4 pi r^2`` conservative radial finite volumes; a genuine
  spherically symmetric 3-D temperature field.  Anchors: ``T ~ 1/r`` and the
  total heat rate ``Q = 4 pi k dT/(1/r1 - 1/r2)``.

Fluid-solid interface
---------------------
* :func:`interface_fluxes` -- the conductivity-weighted interface operator used
  by the CHT coupling: it returns the interface temperature and the heat flux
  seen from each side, so that temperature continuity (``T_f = T_s``) and flux
  continuity (``q_f = q_s``) can be measured directly.  A naive
  arithmetic-mean variant is provided as a *metric* negative control, and
  ``contact_conductance = 0`` gives the adiabatic limit.
* :func:`conservative_two_domain` -- two-domain finite-volume conduction across
  a single shared interface face.

All quantities are float64; lengths are lattice units unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "ConductionResult",
    "InterfaceResult",
    "composite_wall",
    "cylinder_annulus_2d",
    "sphere_shell",
    "interface_fluxes",
    "conservative_two_domain",
    "solve_tridiag",
    "pcg",
    "NU_TUBE_DEVELOPED",
]

F64 = torch.float64

#: Fully developed laminar tube Nusselt number (uniform wall temperature).
NU_TUBE_DEVELOPED = 3.6568


# ===========================================================================
# linear-algebra helpers
# ===========================================================================


def solve_tridiag(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system (float64)."""
    n = diag.size
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def pcg(diag: np.ndarray, matvec, rhs: np.ndarray, tol: float = 1e-13, max_iter: int = 40000):
    """Jacobi-preconditioned conjugate gradient for a symmetric positive
    (semi-)definite finite-volume operator."""
    inv = 1.0 / np.where(np.abs(diag) > 0.0, diag, 1.0)
    x = rhs * inv
    res = rhs - matvec(x)
    z = res * inv
    p = z.copy()
    rz = float(np.dot(res, z))
    bnorm = float(np.linalg.norm(rhs)) + 1e-300
    for _ in range(max_iter):
        if float(np.linalg.norm(res)) / bnorm < tol:
            break
        ap = matvec(p)
        alpha = rz / float(np.dot(p, ap))
        x += alpha * p
        res -= alpha * ap
        z = res * inv
        rz_new = float(np.dot(res, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


# ===========================================================================
# 1. Solid conduction: 1-D composite wall (anchor 1)
# ===========================================================================


@dataclass
class ConductionResult:
    """Container for a steady-conduction verification case."""

    x: np.ndarray
    T: np.ndarray
    T_analytic: np.ndarray
    q: float
    resistance: float
    resistance_exact: float
    rel_err_R: float
    max_err_T: float
    l2_err_T: float
    extra: dict = field(default_factory=dict)


def composite_wall(
    layers: list[tuple[float, float]],
    T_left: float = 1.0,
    T_right: float = 0.0,
    cells: int = 300,
) -> ConductionResult:
    """1-D steady conduction through a multi-layer wall (series resistance).

    Parameters
    ----------
    layers:
        ``[(thickness_i, k_i), ...]`` ordered from the hot (left) to the cold
        (right) side.
    T_left, T_right:
        Dirichlet surface temperatures.
    cells:
        Total number of finite-volume cells, distributed over the layers in
        proportion to their thicknesses.

    Returns
    -------
    :class:`ConductionResult` with the cell-centre temperature field, the wall
    heat flux ``q``, the measured total resistance ``R = dT/q`` and its
    analytical counterpart ``R = sum(delta_i/k_i)``.
    """
    total_len = sum(d for d, _ in layers)
    h_target = total_len / cells
    counts = [max(1, int(round(d / h_target))) for d, _ in layers]

    h_list: list[float] = []
    k_list: list[float] = []
    for (d, kk), c in zip(layers, counts):
        h_list.extend([d / c] * c)
        k_list.extend([float(kk)] * c)
    h = np.asarray(h_list, dtype=np.float64)
    k = np.asarray(k_list, dtype=np.float64)
    n = h.size

    edges = np.concatenate([[0.0], np.cumsum(h)])
    x = 0.5 * (edges[:-1] + edges[1:])

    # face conductances: harmonic mean of the two half-cell resistances
    g_face = 1.0 / (0.5 * h[:-1] / k[:-1] + 0.5 * h[1:] / k[1:])
    g_left = k[0] / (0.5 * h[0])
    g_right = k[-1] / (0.5 * h[-1])

    diag = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)
    diag[0] = g_left + g_face[0]
    upper[0] = -g_face[0]
    rhs[0] = g_left * T_left
    for i in range(1, n - 1):
        diag[i] = g_face[i - 1] + g_face[i]
        lower[i] = -g_face[i - 1]
        upper[i] = -g_face[i]
    diag[-1] = g_face[-1] + g_right
    lower[-1] = -g_face[-1]
    rhs[-1] = g_right * T_right

    T = solve_tridiag(lower, diag, upper, rhs)

    q_all = np.concatenate([g_face * (T[:-1] - T[1:]), [g_left * (T_left - T[0])]])
    q = float(np.mean(q_all))
    q_spread = float(np.max(np.abs(q_all - q)))

    resistance = (T_left - T_right) / q
    resistance_exact = float(sum(d / float(kk) for d, kk in layers))

    T_iface = [T_left]
    for d, kk in layers:
        T_iface.append(T_iface[-1] - (T_left - T_right) * (d / kk) / resistance_exact)
    T_an = np.empty_like(x)
    x_start = 0.0
    for (d, _kk), a, b in zip(layers, T_iface[:-1], T_iface[1:]):
        m = (x >= x_start - 1e-12) & (x <= x_start + d + 1e-12)
        T_an[m] = a + (b - a) * (x[m] - x_start) / d
        x_start += d

    dT = T_left - T_right
    return ConductionResult(
        x=x,
        T=T,
        T_analytic=T_an,
        q=q,
        resistance=resistance,
        resistance_exact=resistance_exact,
        rel_err_R=abs(resistance - resistance_exact) / abs(resistance_exact),
        max_err_T=float(np.max(np.abs(T - T_an)) / dT),
        l2_err_T=float(np.sqrt(np.mean((T - T_an) ** 2)) / dT),
        extra={"q_spread": q_spread, "n_cells": n, "interface_temps": T_iface},
    )


# ===========================================================================
# 2. Solid conduction: 2-D annulus on a polar grid
# ===========================================================================


def cylinder_annulus_2d(
    k: float = 1.0,
    r1: float = 0.2,
    r2: float = 1.0,
    T1: float = 1.0,
    T2: float = 0.0,
    n_r: int = 200,
    n_theta: int = 48,
    theta_amplitude: float = 0.0,
) -> ConductionResult:
    """2-D steady conduction in an annulus on a polar ``(r, theta)`` FV grid.

    Dirichlet data enter through the exact cylindrical boundary conductances
    ``k r_b dtheta / (dr/2)``.  With ``theta_amplitude = A`` the inner surface
    carries ``T1 + A cos(theta)``; the analytical solution then gains the
    separable harmonic ``A cos(theta) (r - r2^2/r)/(r1 - r2^2/r1)``, which is
    exact for Laplace's equation in the annulus.
    """
    dr = (r2 - r1) / n_r
    dth = 2.0 * math.pi / n_theta
    r = r1 + dr * (np.arange(n_r) + 0.5)
    th = dth * (np.arange(n_theta) + 0.5)

    r_face = r1 + dr * (np.arange(n_r) + 1.0)  # r_{i+1/2}; r_face[-1] == r2
    G_r = np.zeros((n_r, n_theta))
    G_r[:-1, :] = k * r_face[:-1][:, None] * dth / dr
    G_th = np.tile((k * dr / (r * dth))[:, None], (1, n_theta))

    G_in = np.zeros((n_r, n_theta))
    G_in[0, :] = k * r1 * dth / (0.5 * dr)
    G_out = np.zeros((n_r, n_theta))
    G_out[-1, :] = k * r2 * dth / (0.5 * dr)
    T1_th = T1 + theta_amplitude * np.cos(th)

    diag = (
        G_r + np.roll(G_r, 1, axis=0) + G_th + np.roll(G_th, 1, axis=1) + G_in + G_out
    ).ravel()
    rhs = (G_in * T1_th[None, :] + G_out * float(T2)).ravel()

    n_theta_idx = np.arange(n_theta)

    def matvec(v: np.ndarray) -> np.ndarray:
        v2 = v.reshape(n_r, n_theta)
        out = (diag * v).reshape(n_r, n_theta)
        out[:-1, :] -= G_r[:-1, :] * v2[1:, :]
        out[1:, :] -= G_r[:-1, :] * v2[:-1, :]
        out[:, :] -= G_th * v2[:, np.roll(n_theta_idx, -1)]
        out[:, :] -= np.roll(G_th, 1, axis=1) * v2[:, np.roll(n_theta_idx, 1)]
        return out.ravel()

    T = pcg(diag, matvec, rhs).reshape(n_r, n_theta)

    RR, TT = np.meshgrid(r, th, indexing="ij")
    T_an = T1 + (T2 - T1) * np.log(RR / r1) / math.log(r2 / r1)
    if theta_amplitude != 0.0:
        T_an = T_an + theta_amplitude * np.cos(TT) * (RR - r2**2 / RR) / (r1 - r2**2 / r1)

    q_in_total = float(np.sum(G_in * (T1_th[None, :] - T[0:1, :])))
    q_exact = 2.0 * math.pi * k * (T1 - T2) / math.log(r2 / r1)
    scale = max(abs(T1 - T2), abs(theta_amplitude), 1e-300)
    resid = float(np.linalg.norm(matvec(T.ravel()) - rhs) / (np.linalg.norm(rhs) + 1e-300))

    return ConductionResult(
        x=r,
        T=T[:, 0],
        T_analytic=T_an[:, 0],
        q=q_in_total,
        resistance=(T1 - T2) / q_in_total if q_in_total != 0.0 else float("inf"),
        resistance_exact=math.log(r2 / r1) / (2.0 * math.pi * k),
        rel_err_R=abs(q_in_total - q_exact) / abs(q_exact),
        max_err_T=float(np.max(np.abs(T - T_an)) / scale),
        l2_err_T=float(np.sqrt(np.mean((T - T_an) ** 2)) / scale),
        extra={
            "n_r": n_r,
            "n_theta": n_theta,
            "q_total": q_in_total,
            "q_total_exact": q_exact,
            "field": T,
            "field_analytic": T_an,
            "r": r,
            "theta": th,
            "linear_residual": resid,
            "n_unknowns": n_r * n_theta,
        },
    )


# ===========================================================================
# 3. Solid conduction: 3-D spherical shell
# ===========================================================================


def sphere_shell(
    layers: list[tuple[float, float]] | None = None,
    r1: float = 0.3,
    r2: float = 1.0,
    T1: float = 1.0,
    T2: float = 0.0,
    n_r: int = 400,
    k: float = 1.0,
) -> ConductionResult:
    """3-D steady conduction in a spherical shell (``4 pi r^2`` radial FV).

    The field is spherically symmetric, i.e. a genuine 3-D solution.  Anchors:
    ``T ~ 1/r`` and the total heat rate
    ``Q = 4 pi k (T1 - T2)/(1/r1 - 1/r2)``.  With ``layers = [(r_outer_i, k_i)]``
    the shell is composite and the interface resistances add in series.
    """
    if layers is None:
        layers = [(r2, k)]
    radii = [r1] + [float(rr) for rr, _ in layers]
    ks = [float(kk) for _, kk in layers]

    edges = np.linspace(r1, r2, n_r + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    kc = np.full(n_r, ks[-1])
    for i, rv in enumerate(rc):
        for (lo, hi), kk in zip(zip(radii[:-1], radii[1:]), ks):
            if lo - 1e-12 <= rv <= hi + 1e-12:
                kc[i] = kk
                break

    dr = edges[1] - edges[0]
    a_face = 4.0 * math.pi * edges[1:-1] ** 2
    g_face = a_face / (0.5 * dr / kc[:-1] + 0.5 * dr / kc[1:])
    g_in = kc[0] * 4.0 * math.pi * r1**2 / (0.5 * dr)
    g_out = kc[-1] * 4.0 * math.pi * r2**2 / (0.5 * dr)

    n = n_r
    diag = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)
    diag[0] = g_in + g_face[0]
    upper[0] = -g_face[0]
    rhs[0] = g_in * T1
    for i in range(1, n - 1):
        diag[i] = g_face[i - 1] + g_face[i]
        lower[i] = -g_face[i - 1]
        upper[i] = -g_face[i]
    diag[-1] = g_face[-1] + g_out
    lower[-1] = -g_face[-1]
    rhs[-1] = g_out * T2

    T = solve_tridiag(lower, diag, upper, rhs)

    Q = float(g_in * (T1 - T[0]))
    Q_spread = float(np.max(np.abs(g_face * (T[:-1] - T[1:]) - Q)))

    R_exact = 0.0
    for (lo, hi), kk in zip(zip(radii[:-1], radii[1:]), ks):
        R_exact += (1.0 / lo - 1.0 / hi) / (4.0 * math.pi * kk)
    Q_exact = (T1 - T2) / R_exact

    T_iface = [T1]
    for (lo, hi), kk in zip(zip(radii[:-1], radii[1:]), ks):
        T_iface.append(
            T_iface[-1] - (T1 - T2) * ((1.0 / lo - 1.0 / hi) / (4.0 * math.pi * kk)) / R_exact
        )
    T_an = np.empty(n_r)
    for i, rv in enumerate(rc):
        T_an[i] = T_iface[-1]
        for (lo, hi), a, b in zip(zip(radii[:-1], radii[1:]), T_iface[:-1], T_iface[1:]):
            if lo - 1e-12 <= rv <= hi + 1e-12:
                T_an[i] = a + (b - a) * (1.0 / lo - 1.0 / rv) / (1.0 / lo - 1.0 / hi)
                break

    dT = T1 - T2
    return ConductionResult(
        x=rc,
        T=T,
        T_analytic=T_an,
        q=Q,
        resistance=(T1 - T2) / Q,
        resistance_exact=R_exact,
        rel_err_R=abs(Q - Q_exact) / abs(Q_exact),
        max_err_T=float(np.max(np.abs(T - T_an)) / dT),
        l2_err_T=float(np.sqrt(np.mean((T - T_an) ** 2)) / dT),
        extra={"Q": Q, "Q_exact": float(Q_exact), "Q_spread": Q_spread, "n_r": n_r},
    )


# ===========================================================================
# 4. Fluid-solid interface operator
# ===========================================================================


@dataclass
class InterfaceResult:
    """Output of :func:`interface_fluxes`."""

    T_interface: np.ndarray
    q_fluid: np.ndarray
    q_solid: np.ndarray
    flux_residual: float
    temperature_residual: float


def interface_fluxes(
    T_fluid,
    T_solid,
    k_fluid: float,
    k_solid: float,
    dx_fluid: float = 0.5,
    dx_solid: float = 0.5,
    contact_conductance: float | None = None,
    interface_weighting: str = "harmonic",
) -> InterfaceResult:
    """Conductivity-weighted fluid-solid interface operator.

    Two adjacent cells (fluid at distance ``dx_fluid`` and solid at distance
    ``dx_solid`` from the interface face) exchange heat through the series
    conductance ``1/G = dx_fluid/k_f + dx_solid/k_s`` (plus ``1/h_int`` when a
    finite contact conductance is supplied).  With ``h_int = inf`` the interface
    temperature is the flux-conserving value
    ``T_i = (G_f T_f + G_s T_s)/(G_f + G_s)``, hence

        ``q_f = G_f (T_f - T_i)``,  ``q_s = G_s (T_i - T_s)``

    are equal to machine precision and ``T_f|_int = T_s|_int = T_i``.

    ``interface_weighting="arithmetic"`` returns the naive arithmetic-mean
    interface temperature; it is kept only as a *negative control for the
    metric* (it violates flux continuity by ``O((k_s-k_f)/(k_s+k_f))``), so a
    passing residual cannot be vacuous.

    ``contact_conductance = 0`` gives the adiabatic interface (``q = 0``).
    """
    T_f = torch.as_tensor(np.asarray(T_fluid), dtype=F64)
    T_s = torch.as_tensor(np.asarray(T_solid), dtype=F64)
    Gf = k_fluid / dx_fluid
    Gs = k_solid / dx_solid

    if interface_weighting == "arithmetic":
        T_i = 0.5 * (T_f + T_s)
        q_f = Gf * (T_f - T_i)
        q_s = Gs * (T_i - T_s)
        T_i_f, T_i_s = T_i, T_i
    elif contact_conductance is None or contact_conductance == float("inf"):
        T_i = (Gf * T_f + Gs * T_s) / (Gf + Gs)
        q_f = Gf * (T_f - T_i)
        q_s = Gs * (T_i - T_s)
        T_i_f, T_i_s = T_i, T_i
    else:
        h = float(contact_conductance)
        if h == 0.0:
            q = torch.zeros_like(T_f)
            T_i = 0.5 * (T_f + T_s)
            T_i_f, T_i_s = T_f, T_s
            q_f, q_s = q, q
        else:
            R_tot = 1.0 / Gf + 1.0 / h + 1.0 / Gs
            q = (T_f - T_s) / R_tot
            T_i_f = T_f - q / Gf
            T_i_s = T_s + q / Gs
            T_i = 0.5 * (T_i_f + T_i_s)
            q_f, q_s = q, q

    q_f_a = np.asarray(q_f)
    q_s_a = np.asarray(q_s)
    scale = 0.5 * (np.abs(q_f_a) + np.abs(q_s_a))
    scale = np.where(scale > 0.0, scale, 1.0)
    flux_res = float(np.max(np.abs(q_f_a - q_s_a) / scale))
    temp_res = float(np.max(np.abs(np.asarray(T_i_f) - np.asarray(T_i_s))))
    return InterfaceResult(
        T_interface=np.asarray(T_i),
        q_fluid=q_f_a,
        q_solid=q_s_a,
        flux_residual=flux_res,
        temperature_residual=temp_res,
    )


def conservative_two_domain(
    k_fluid: float = 1.0,
    k_solid: float = 5.0,
    L_fluid: float = 2.0,
    L_solid: float = 1.0,
    T_left: float = 1.0,
    T_right: float = 0.0,
    n_fluid: int = 200,
    n_solid: int = 100,
) -> dict:
    """Two-domain conduction across a single shared interface face (FV).

    Both sub-domains use node-centred cells and the interface face carries one
    conductance, so the discrete flux is identical on both sides by
    construction and the interface temperature is continuous.  The analytical
    reference is the series chain ``q = dT/(delta_f/k_f + delta_s/k_s)``.
    """
    hf = L_fluid / n_fluid
    hs = L_solid / n_solid
    n = n_fluid + n_solid
    h = np.concatenate([np.full(n_fluid, hf), np.full(n_solid, hs)])
    k = np.concatenate([np.full(n_fluid, float(k_fluid)), np.full(n_solid, float(k_solid))])

    g_faces = 1.0 / (0.5 * h[:-1] / k[:-1] + 0.5 * h[1:] / k[1:])
    g_left = k[0] / (0.5 * h[0])
    g_right = k[-1] / (0.5 * h[-1])

    diag = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)
    diag[0] = g_left + g_faces[0]
    upper[0] = -g_faces[0]
    rhs[0] = g_left * T_left
    for i in range(1, n - 1):
        diag[i] = g_faces[i - 1] + g_faces[i]
        lower[i] = -g_faces[i - 1]
        upper[i] = -g_faces[i]
    diag[-1] = g_faces[-1] + g_right
    lower[-1] = -g_faces[-1]
    rhs[-1] = g_right * T_right
    T = solve_tridiag(lower, diag, upper, rhs)

    i_iface = n_fluid - 1
    g_if = g_faces[i_iface]
    q_fluid = float(g_if * (T[i_iface] - T[i_iface + 1]))
    q_solid = float(g_if * (T[i_iface] - T[i_iface + 1]))  # same face -> identical
    q_exact = (T_left - T_right) / (L_fluid / k_fluid + L_solid / k_solid)

    T_i_fluid = float(T[i_iface] - q_fluid * (0.5 * hf) / k_fluid)
    T_i_solid = float(T[i_iface + 1] + q_fluid * (0.5 * hs) / k_solid)

    return {
        "T": T,
        "x": np.concatenate(
            [hf * (np.arange(n_fluid) + 0.5), L_fluid + hs * (np.arange(n_solid) + 0.5)]
        ),
        "q_fluid": q_fluid,
        "q_solid": q_solid,
        "q_exact": float(q_exact),
        "rel_err_q": abs(q_fluid - q_exact) / abs(q_exact),
        "flux_residual": abs(q_fluid - q_solid) / (abs(q_exact) + 1e-300),
        "T_interface_fluid": T_i_fluid,
        "T_interface_solid": T_i_solid,
        "temperature_residual": abs(T_i_fluid - T_i_solid) / (T_left - T_right),
    }