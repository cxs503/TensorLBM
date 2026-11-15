"""Self-contained 2D shallow-water (SWE) finite-volume solver + Stoker reference.

Purpose
-------
``dam_break_sc`` is the guide-gap case "oblique dam axis / Stoker-type
discontinuity".  The gap exists because the only in-repo dam-break evidence was
a Shan-Chen LBM run judged against digitised Martin & Moyce table points
(``benchmarks/pending/dam_break_sc``, terminal verdict m163) -- i.e. a
multiphase *model* error, not a verifiable reference solution.  This module
builds the missing piece from the other side: an explicit HLL/MUSCL
finite-volume solver for the *shallow-water* equations, which is the equation
set for which an exact closed-form solution (Stoker 1957 wet-bed dam break)
exists, and which therefore admits a machine-checkable error number.

Numerics
--------
* conservative variables ``U = (h, hu, hv)``, MUSCL reconstruction on primitive
  variables ``(h, u, v)`` with minmod limiting, HLL interface flux, SSP-RK2 in
  time, CFL = 0.4, transmissive boundaries.
* no bathymetry, no source term -> the only physics is the flux.
* the initial jump is a straight line whose normal is rotated by ``theta`` from
  the x axis.  The SWE are rotation invariant, so the exact solution is the 1D
  Stoker solution in ``s = n.x``; every deviation is numerical.

Reference
---------
Stoker, J.J. (1957) *Water Waves*, Interscience, Sec. 10.6 (wet-bed dam break).
The four constant states along the similarity variable ``xi = s/t``:

    xi < -c0                : h = hL,               u = 0
    -c0 <= xi <= u2 - c2    : rarefaction, u = 2/3 (xi + c0), c = 1/3 (2c0 - xi)
    u2 - c2 < xi < S        : h = h2,               u = u2
    xi > S                  : h = hR,               u = 0

with ``c0 = sqrt(g hL)``, ``c1 = sqrt(g hR)``, ``c2 = sqrt(g h2)`` and ``h2``
given by the standard root of the rarefaction/shock matching

    2 (c0 - c2) = (h2 - hR) sqrt( g (h2 + hR) / (2 h2 hR) ),

``u2 = 2 (c0 - c2)`` and the shock speed ``S = u2 h2 / (h2 - hR)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# exact reference (Stoker 1957)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StokerSolution:
    hL: float
    hR: float
    g: float
    h2: float
    u2: float
    c2: float
    shock_speed: float

    @property
    def c0(self) -> float:
        return math.sqrt(self.g * self.hL)

    @property
    def toe_speed(self) -> float:
        """Left edge (toe) of the constant plateaux region, ``u2 - c2``."""
        return self.u2 - self.c2

    def profile(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Depth/velocity of the self-similar solution at ``xi = s / t``."""
        xi = np.asarray(xi, dtype=float)
        h = np.full_like(xi, self.hL)
        u = np.zeros_like(xi)
        # rarefaction fan
        fan = (xi > -self.c0) & (xi < self.toe_speed)
        c_fan = (2.0 * self.c0 - xi[fan]) / 3.0
        h[fan] = c_fan**2 / self.g
        u[fan] = 2.0 / 3.0 * (xi[fan] + self.c0)
        # constant plateau behind the shock
        plat = (xi >= self.toe_speed) & (xi < self.shock_speed)
        h[plat] = self.h2
        u[plat] = self.u2
        # undisturbed right state
        right = xi >= self.shock_speed
        h[right] = self.hR
        u[right] = 0.0
        return h, u


def stoker_solution(hL: float, hR: float, g: float = 1.0) -> StokerSolution:
    """Solve the wet-bed dam-break Riemann problem (bisection, 200 iterations)."""
    if not (hR > 0.0 and hL > hR):
        raise ValueError("wet-bed Stoker solution requires hL > hR > 0")
    c0 = math.sqrt(g * hL)
    c1 = math.sqrt(g * hR)

    def residual(h2: float) -> float:
        c2 = math.sqrt(g * h2)
        return 2.0 * (c0 - c2) - (h2 - hR) * math.sqrt(g * (h2 + hR) / (2.0 * h2 * hR))

    lo, hi = hR * (1.0 + 1e-12), hL
    # residual(lo) > 0 (u2 of the rarefaction branch exceeds the shock branch),
    # residual(hi) < 0 -> monotone bracketing
    if residual(lo) < 0 or residual(hi) > 0:
        raise RuntimeError(f"bad bracket for h2 ({residual(lo)}, {residual(hi)})")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    h2 = 0.5 * (lo + hi)
    c2 = math.sqrt(g * h2)
    u2 = 2.0 * (c0 - c2)
    shock = u2 * h2 / (h2 - hR)
    if not (shock > 0.0):
        raise RuntimeError("non-physical shock speed")
    _ = c1  # documented for completeness (right sound speed, unused)
    return StokerSolution(hL=hL, hR=hR, g=g, h2=h2, u2=u2, c2=c2, shock_speed=shock)


# --------------------------------------------------------------------------- #
# finite-volume solver
# --------------------------------------------------------------------------- #
def _primitive(U: np.ndarray, g: float) -> np.ndarray:
    """(h, u, v) from conservative (h, hu, hv); depth floored for positivity."""
    h = np.maximum(U[0], 0.0)
    hs = np.maximum(h, EPS)
    u = np.where(h > EPS, U[1] / hs, 0.0)
    v = np.where(h > EPS, U[2] / hs, 0.0)
    return np.stack([h, u, v])


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _slopes(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unlimited-in-range minmod slopes along x (last axis) and y (middle)."""
    sx = np.zeros_like(W)
    sx[:, :, 1:-1] = _minmod(W[:, :, 1:-1] - W[:, :, :-2], W[:, :, 2:] - W[:, :, 1:-1])
    sy = np.zeros_like(W)
    sy[:, 1:-1, :] = _minmod(W[:, 1:-1, :] - W[:, :-2, :], W[:, 2:, :] - W[:, 1:-1, :])
    return sx, sy


def _flux_x(W: np.ndarray, g: float) -> np.ndarray:
    """Physical flux through an x-interface: (hu, hu^2 + gh^2/2, huv)."""
    h, u, v = W[0], W[1], W[2]
    return np.stack([h * u, h * u * u + 0.5 * g * h * h, h * u * v])


def _flux_y(W: np.ndarray, g: float) -> np.ndarray:
    """Physical flux through a y-interface: (hv, huv, hv^2 + gh^2/2)."""
    h, u, v = W[0], W[1], W[2]
    return np.stack([h * v, h * u * v, h * v * v + 0.5 * g * h * h])


def _hll(WL: np.ndarray, WR: np.ndarray, g: float, fluxfn) -> np.ndarray:
    """HLL flux along the last axis (Davis wave-speed estimate)."""
    hL, uL, vL = WL[0], WL[1], WL[2]
    hR, uR, vR = WR[0], WR[1], WR[2]
    cL = np.sqrt(g * np.maximum(hL, 0.0))
    cR = np.sqrt(g * np.maximum(hR, 0.0))
    # Davis estimates use the interface-normal velocity component
    unL = uL if fluxfn is _flux_x else vL
    unR = uR if fluxfn is _flux_x else vR
    aL = np.minimum(unL - cL, unR - cR)
    aR = np.maximum(unL + cL, unR + cR)
    FL = fluxfn(WL, g)
    FR = fluxfn(WR, g)
    UL = np.stack([hL, hL * uL, hL * vL])
    UR = np.stack([hR, hR * uR, hR * vR])
    den = np.where(np.abs(aR - aL) > EPS, aR - aL, EPS)
    return (aR * FL - aL * FR + aL * aR * (UR - UL)) / den


def _rhs(U: np.ndarray, dx: float, g: float) -> np.ndarray:
    W = _primitive(U, g)
    Wp = np.pad(W, ((0, 0), (1, 1), (1, 1)), mode="edge")
    sx, sy = _slopes(Wp)
    # x interfaces
    WL = Wp[:, :, :-1] + 0.5 * sx[:, :, :-1]
    WR = Wp[:, :, 1:] - 0.5 * sx[:, :, 1:]
    Fx = _hll(WL, WR, g, _flux_x)
    # y interfaces: transpose so that y becomes the last axis
    Wt = np.transpose(Wp, (0, 2, 1))
    sxt, syt = np.transpose(sx, (0, 2, 1)), np.transpose(sy, (0, 2, 1))
    WLt = Wt[:, :, :-1] + 0.5 * syt[:, :, :-1]
    WRt = Wt[:, :, 1:] - 0.5 * syt[:, :, 1:]
    Fyt = _hll(WLt, WRt, g, _flux_y)
    Fy = np.transpose(Fyt, (0, 2, 1))
    _ = sxt
    Fx_i = Fx[:, 1:-1, :]          # (3, ny, nx+1) interfaces
    Fy_i = Fy[:, :, 1:-1]          # (3, ny+1, nx) interfaces
    return -(Fx_i[:, :, 1:] - Fx_i[:, :, :-1]) / dx - (Fy_i[:, 1:, :] - Fy_i[:, :-1, :]) / dx


def swe_march(
    h_init: np.ndarray,
    dx: float,
    g: float,
    t_end: float,
    cfl: float = 0.4,
    max_steps: int = 200_000,
) -> tuple[np.ndarray, dict]:
    """March ``h_init`` (initial depth field, zero velocity) to ``t_end``."""
    ny, nx = h_init.shape
    U = np.stack([h_init.copy(), np.zeros_like(h_init), np.zeros_like(h_init)])
    t = 0.0
    steps = 0
    dt = 0.0
    # discrete mass budget: every change of sum(h) must equal the RK2 net
    # divergence minus the (physical) flux through the open boundaries; the
    # residual below is therefore a round-off / clipping detector.
    budget = 0.0
    while t < t_end and steps < max_steps:
        W = _primitive(U, g)
        amax = float(np.max(np.abs(W[1]) + np.sqrt(g * np.maximum(W[0], 0.0))))
        dt = cfl * dx / max(amax, EPS)
        if t + dt > t_end:
            dt = t_end - t
        m_prev = float(U[0].sum())
        r0 = _rhs(U, dx, g)
        U1 = U + dt * r0
        U1[0] = np.maximum(U1[0], 0.0)
        r1 = _rhs(U1, dx, g)
        U = 0.5 * U + 0.5 * (U1 + dt * r1)
        U[0] = np.maximum(U[0], 0.0)
        m_new = float(U[0].sum())
        predicted = dt * (0.5 * float(r0[0].sum()) + 0.5 * float(r1[0].sum()))
        budget += (m_new - m_prev) - predicted
        t += dt
        steps += 1
    m0 = float(h_init.sum())
    m_end = float(U[0].sum())
    info = {
        "steps": steps,
        "t_end": t,
        "dt_last": dt,
        "mass_drift": abs(m_end - m0) / m0,
        "mass_budget_residual": abs(budget) / m0,
    }
    return U, info


def make_oblique_dam(
    nx: int, ny: int, hL: float, hR: float, theta_deg: float, half: float = 0.5
):
    """Cartesian grid ``[-half,half]^2`` with a dam line rotated by theta."""
    x = -half + (np.arange(nx) + 0.5) * (2.0 * half / nx)
    y = -half + (np.arange(ny) + 0.5) * (2.0 * half / ny)
    X, Y = np.meshgrid(x, y)
    th = math.radians(theta_deg)
    s = X * math.cos(th) + Y * math.sin(th)
    return np.where(s < 0.0, hL, hR), x, y, X, Y


def oblique_plane_average(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    theta_deg: float,
    s_grid: np.ndarray,
    n_rays: int = 401,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a 2D field over lines of constant ``s = n.x`` (ray averaging).

    The exact solution is a function of ``s`` alone, so a plane average is the
    natural 1D reduction; it also removes the staircase noise that a single
    sampled ray would carry for an oblique front.  Points outside the domain are
    masked (count returned) instead of clipped to the boundary value.
    """
    nx, ny = field.shape[1], field.shape[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    th = math.radians(theta_deg)
    n = np.array([math.cos(th), math.sin(th)])
    tau = np.array([-math.sin(th), math.cos(th)])
    x0, y0 = x[0] - 0.5 * dx, y[0] - 0.5 * dy
    x1, y1 = x[-1] + 0.5 * dx, y[-1] + 0.5 * dy
    t_lo, t_hi = -0.5 * math.hypot(x1 - x0, y1 - y0), 0.5 * math.hypot(x1 - x0, y1 - y0)
    t_grid = np.linspace(t_lo, t_hi, n_rays)
    S, T = np.meshgrid(s_grid, t_grid, indexing="ij")
    PX = S * n[0] + T * tau[0]
    PY = S * n[1] + T * tau[1]
    inside = (
        (PX >= x0 + margin)
        & (PX <= x1 - margin)
        & (PY >= y0 + margin)
        & (PY <= y1 - margin)
    )
    # bilinear interpolation on the cell-centred grid
    fi = (PX - x[0]) / dx
    fj = (PY - y[0]) / dy
    i0 = np.clip(np.floor(fi).astype(int), 0, nx - 1)
    j0 = np.clip(np.floor(fj).astype(int), 0, ny - 1)
    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    wx = np.clip(fi - i0, 0.0, 1.0)
    wy = np.clip(fj - j0, 0.0, 1.0)
    vals = (
        field[j0, i0] * (1 - wx) * (1 - wy)
        + field[j0, i1] * wx * (1 - wy)
        + field[j1, i0] * (1 - wx) * wy
        + field[j1, i1] * wx * wy
    )
    vals = np.where(inside, vals, 0.0)
    w = inside.astype(float)
    cnt = w.sum(axis=1)
    prof = np.where(cnt > 0, (vals * w).sum(axis=1) / np.maximum(cnt, 1.0), np.nan)
    return prof, cnt


def front_and_plateau(
    s: np.ndarray,
    hprof: np.ndarray,
    ref: StokerSolution,
    t: float,
) -> dict:
    """Shock position and post-shock plateau depth from a plane-averaged profile.

    Post-processing convention (documented because it is a *convention*, not a
    physics input): the plateau is averaged over the middle half of the
    analytic constant-state interval ``[u2 - c2, S] * t``; the shock position is
    the ``h = (h_plateau + hR)/2`` crossing of the (smeared) front.
    """
    a, b = ref.toe_speed * t, ref.shock_speed * t
    lo = a + 0.25 * (b - a)
    hi = b - 0.25 * (b - a)
    win = (s >= lo) & (s <= hi) & np.isfinite(hprof)
    if win.sum() < 3:
        raise RuntimeError("plateau window is empty")
    h_plateau = float(np.mean(hprof[win]))
    thr = 0.5 * (h_plateau + ref.hR)
    # walk outward from the plateau in +s and find the first sample below thr
    idx = np.where(win)[0]
    i_start = int(idx[-1])
    i_front = None
    for i in range(i_start, len(s)):
        if not np.isfinite(hprof[i]):
            continue
        if hprof[i] <= thr:
            i_front = i
            break
    if i_front is None or i_front == 0:
        raise RuntimeError("shock front not found inside the sampled window")
    # linear interpolation between i_front-1 and i_front
    h_prev, h_cur = hprof[i_front - 1], hprof[i_front]
    frac = (h_prev - thr) / max(h_prev - h_cur, EPS)
    s_front = s[i_front - 1] + frac * (s[i_front] - s[i_front - 1])
    return {
        "h_plateau": h_plateau,
        "shock_position": float(s_front),
        "shock_speed": float(s_front / t),
        "plateau_window": [float(lo), float(hi)],
    }