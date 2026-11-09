"""D3Q19 mixed bounce-back / specular wall: three-dimensional lift of
``partial_slip_wall``, plus two inversion estimators that do not fit curvature.

Why this module exists
----------------------
:mod:`tensorlbm.boundary.partial_slip_wall` implements the mixed (partial-slip)
wall on a **1D column reduction**: the target flows are uniform along the flow
direction, so one lattice column in the wall-normal direction was deemed
sufficient.  That reduction is an *assumption* about the lattice operator (the
D2Q9 diagonals that leave the column, the periodicity along the flow, and the
energy exchange with the transverse direction are all discarded).  This module
removes the assumption: it runs the **full D3Q19 lattice** with the mixed wall
applied link-by-link at the y-walls and periodic boundaries in x and z, then
*measures* the reduction error against the 1D column:

  **C0 (reduction exactness)** -- for an x/z-uniform body-forced channel the
  D3Q19 solution reproduces the 1D-column velocity profile to round-off
  (relative ``<= 1e-12``) and is uniform across the periodic x/z planes to
  round-off.  The 1D reduction is therefore exact for this flow class, not an
  approximation; that is what licenses the companion module's cheap sweeps.

Second claim -- **two inversion estimators that avoid the ``A + B L^2``
cancellation**.  Inverting ``b`` from a quadratic fit forms ``A + B L^2``, a
difference of two same-magnitude quantities, so a curvature error is amplified
by ``L^2``.  Two independent estimators are provided:

* ``bulk`` -- flow-rate based.  The steady momentum balance fixes the *mean*
  velocity exactly::

      plane:  u_mean = g H^2 /(3 nu) + g H b / nu
      pipe :  u_mean = g R^2 /(8 nu) + g R b /(2 nu)

  so ``b_bulk`` needs only the averaged profile (no curvature at all).
* ``model`` -- fixed-curvature least squares.  The curvature
  ``B = -g/(2 nu)`` (plane) / ``-g/(4 nu)`` (pipe) is *given by the analytic
  solution* rather than fitted; only the additive slip term ``g L b/nu`` is
  unknown, so ``b = mean(u - u_noslip) * nu / (g L)`` over a near-wall band.

Both are tested against the quadratic-fit estimator
(:func:`tensorlbm.boundary.partial_slip_wall.invert_slip_length`) on the *same*
fields, so the improvement is a measured delta, not a claim.

The ``b(sigma)`` law, the turbulent drag-reduction law and the net-benefit
ledger are *reused* from the 1D module: the law is a property of the wall
operator, and re-deriving it in 3D is exactly the C0 claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tensorlbm.boundary import partial_slip_wall as ps
from tensorlbm.d3q19 import C as _C_TORCH
from tensorlbm.d3q19 import OPPOSITE as _OPP_TORCH
from tensorlbm.d3q19 import W_EXACT64 as _W_TORCH

__all__ = [
    "C19",
    "W19",
    "OPP19",
    "MIRROR_Y",
    "mirror_table_y",
    "mixed_wall_row_3d",
    "Channel3DResult",
    "channel_poiseuille_3d",
    "bulk_mean_slip_length",
    "model_slip_length",
    "collapse_error",
    "plane_slip_law_3d",
    "freeslip_signature",
]

# ---------------------------------------------------------------------------
# Lattice tables -- taken from the repository's canonical D3Q19 module so the
# mirror table cannot drift away from the lattice constants.
# ---------------------------------------------------------------------------
C19 = np.asarray(_C_TORCH.cpu().numpy(), dtype=np.int64)
W19 = np.asarray(_W_TORCH.cpu().numpy(), dtype=np.float64)
OPP19 = np.asarray(_OPP_TORCH.cpu().numpy(), dtype=np.int64)
_Q = int(C19.shape[0])


def mirror_table_y() -> np.ndarray:
    """Direction with ``c_y`` flipped and ``c_x, c_z`` kept (wall normal ``e_y``).

    Derived from the lattice constants instead of being hand-transcribed, so a
    change of velocity set cannot silently break the specular half of the wall.
    """
    lookup = {tuple(int(v) for v in C19[q]): q for q in range(_Q)}
    return np.array([lookup[(int(c[0]), -int(c[1]), int(c[2]))] for c in C19], dtype=np.int64)


MIRROR_Y = mirror_table_y()

_CX = C19[:, 0].reshape(_Q, 1, 1, 1)
_CY = C19[:, 1].reshape(_Q, 1, 1, 1)
_CZ = C19[:, 2].reshape(_Q, 1, 1, 1)
_W = W19.reshape(_Q, 1, 1, 1)
_INWARD_LOW = (C19[:, 1] > 0).reshape(_Q, 1, 1)     # stream wall -> fluid at j=1
_INWARD_HIGH = (C19[:, 1] < 0).reshape(_Q, 1, 1)    # stream wall -> fluid at j=ny-2


# ---------------------------------------------------------------------------
# Mixed wall row
# ---------------------------------------------------------------------------
def mixed_wall_row_3d(f_post_adjacent: np.ndarray, sigma: float, top: bool):
    """Mixed bounce-back / specular populations for one y-wall plane.

    ``f_post_adjacent`` has shape ``(19, nx, nz)`` and holds the *post-collision*
    populations of the fluid plane next to the wall.  For every direction that
    streams from the wall back into the fluid the returned array carries ::

        f_q = (1 - sigma) f_opp(q) + sigma f_mirror(q)

    (``sigma = 0`` -> bounce-back / no-slip, ``sigma = 1`` -> specular / free
    slip).  Directions that do not point into the fluid keep the adjacent
    values; they are inert because the next streaming step overwrites them and
    never reads them.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    mix = (1.0 - sigma) * f_post_adjacent[OPP19] + sigma * f_post_adjacent[MIRROR_Y]
    inward = _INWARD_HIGH if top else _INWARD_LOW
    return np.where(inward, mix, f_post_adjacent)


# ---------------------------------------------------------------------------
# D3Q19 kernels
# ---------------------------------------------------------------------------
def _equilibrium(rho, ux, uy, uz):
    cu = 3.0 * (_CX * ux + _CY * uy + _CZ * uz)
    usq = ux * ux + uy * uy + uz * uz
    return _W * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)


def _macroscopic(f, fx=0.0, fy=0.0, fz=0.0):
    rho = f.sum(0)
    safe = np.where(rho == 0.0, 1.0, rho)       # wall rows carry no mass
    ux = ((_CX * f).sum(0) + 0.5 * fx) / safe
    uy = ((_CY * f).sum(0) + 0.5 * fy) / safe
    uz = ((_CZ * f).sum(0) + 0.5 * fz) / safe
    return rho, ux, uy, uz


def _collide_bgk_guo(f, tau, fx=0.0, fy=0.0, fz=0.0):
    """BGK collision with Guo forcing (second-order in the force)."""
    rho, ux, uy, uz = _macroscopic(f, fx, fy, fz)
    feq = _equilibrium(rho, ux, uy, uz)
    f_post = f - (f - feq) / tau
    cu = _CX * ux + _CY * uy + _CZ * uz
    src = (
        3.0 * ((_CX - ux) * fx + (_CY - uy) * fy + (_CZ - uz) * fz)
        + 9.0 * cu * (_CX * fx + _CY * fy + _CZ * fz)
    )
    return f_post + (1.0 - 0.5 / tau) * _W * src


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class Channel3DResult:
    """Steady D3Q19 slip channel (plane geometry)."""

    sigma: float
    nx: int
    ny: int
    nz: int
    tau: float
    g: float
    u_profile: np.ndarray        # (ny-2,) plane-averaged streamwise velocity
    u_slice: np.ndarray          # (ny-2, nx) at fixed z -- collapse check data
    plane_spread: float          # max |u - plane mean| / u_max over the x/z plane
    length: float                # half height H of the wall plane
    steps: int
    converged: bool
    residual: float
    momentum_growth: float | None = None
    geometry: str = "plane3d"

    @property
    def u_max(self) -> float:
        return float(self.u_profile.max())


def channel_poiseuille_3d(
    sigma: float,
    ny: int = 33,
    nx: int = 3,
    nz: int = 3,
    tau: float = 0.8,
    g: float = 1e-5,
    steps: int = 60000,
    tol: float = 1e-10,
    track_momentum: bool = False,
) -> Channel3DResult:
    """Steady plane Poiseuille flow between two mixed-slip y-walls in D3Q19.

    Solid rows are ``j = 0`` and ``j = ny-1``; fluid rows are ``1 .. ny-2``.
    A bounce-back wall sits half a lattice unit outside the outermost fluid
    row, so the wall planes are ``y = 0.5`` and ``y = ny - 1.5``, the centre
    plane is ``y = (ny-1)/2`` and the half height is ``H = (ny-2)/2`` -- the
    same convention as :func:`partial_slip_wall.channel_poiseuille`, which is
    what makes C0 a like-for-like comparison.

    ``x`` and ``z`` are periodic.  Nothing in the solver assumes x/z
    uniformity: the mixed wall is applied link-by-link, so the mid-plane
    profile can be compared with the 1D column to *measure* the reduction
    error.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if ny < 8:
        raise ValueError("ny must be >= 8")
    if nx < 1 or nz < 1:
        raise ValueError("nx, nz must be >= 1")
    zero = np.zeros((nx, ny, nz))
    f = _equilibrium(np.ones_like(zero), zero, zero, zero)
    u_prev = np.zeros((ny - 2, nx, nz))
    residual = float("inf")
    converged = False
    mom_hist: list[float] = []
    step = 0
    u = u_prev
    for step in range(1, steps + 1):
        fx = g * f.sum(0)
        f_post = _collide_bgk_guo(f, tau, fx=fx)
        f_post[:, :, 0, :] = mixed_wall_row_3d(f_post[:, :, 1, :], sigma, top=False)
        f_post[:, :, ny - 1, :] = mixed_wall_row_3d(
            f_post[:, :, ny - 2, :], sigma, top=True
        )
        new_f = np.zeros_like(f)
        for q in range(_Q):
            cx, cyq, cz = (int(v) for v in C19[q])
            src = f_post[q]
            if cx:
                src = np.roll(src, cx, axis=0)          # -> src[x] = full[x - cx]
            if cz:
                src = np.roll(src, cz, axis=2)
            new_f[q][:, 1:ny - 1, :] = src[:, 1 - cyq:ny - 1 - cyq, :]
        f = new_f
        _rho, ux, _uy, _uz = _macroscopic(f, g * f.sum(0))
        u = ux[:, 1:ny - 1, :].transpose(1, 0, 2)       # (ny-2, nx, nz)
        residual = float(np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1e-300))
        u_prev = u.copy()
        if track_momentum and step % 1000 == 0:
            mom_hist.append(float(np.abs(u).sum()))
        if residual < tol and step > 200:
            converged = True
            break
    plane_mean = u.mean(axis=(1, 2))
    umax = max(float(np.max(np.abs(u))), 1e-300)
    spread = float(np.max(np.abs(u - plane_mean[:, None, None])) / umax)
    mom_growth = None
    if track_momentum and len(mom_hist) >= 2:
        mom_growth = (mom_hist[-1] - mom_hist[0]) / (len(mom_hist) - 1)
    return Channel3DResult(
        sigma=sigma, nx=nx, ny=ny, nz=nz, tau=tau, g=g,
        u_profile=plane_mean, u_slice=u[:, :, 0], plane_spread=spread,
        length=(ny - 2) / 2.0, steps=step, converged=converged,
        residual=residual, momentum_growth=mom_growth,
    )


def collapse_error(result_3d: Channel3DResult, result_1d=None) -> dict:
    """Error of the 1D column reduction against the full D3Q19 field (claim C0)."""
    u3 = np.asarray(result_3d.u_profile, dtype=float)
    if result_1d is not None:
        u1 = np.asarray(result_1d.u, dtype=float)
        if u1.shape != u3.shape:
            raise ValueError(
                f"profile shapes differ: 1D {u1.shape} vs 3D {u3.shape}; "
                "use the same ny and the same wall rows"
            )
        dev = float(np.max(np.abs(u3 - u1)) / max(np.max(np.abs(u3)), 1e-300))
    else:
        dev = None
    return {
        "profile_points": int(u3.size),
        "rel_profile_deviation": dev,
        "plane_uniformity": result_3d.plane_spread,
        "nx": result_3d.nx,
        "nz": result_3d.nz,
        "ny": result_3d.ny,
    }


# ---------------------------------------------------------------------------
# Inversion estimators (curvature-free)
# ---------------------------------------------------------------------------
def bulk_mean_slip_length(
    u: Sequence[float],
    length: float,
    nu: float,
    g: float,
    geometry: str = "plane",
    coord: Sequence[float] | None = None,
    u_mean: float | None = None,
) -> dict:
    """Slip length from the *mean* velocity (no profile fitting at all).

    Analytic slip-Poiseuille mean velocities::

        plane  u_mean = g H^2/(3 nu) + g H b/nu
        pipe   u_mean = g R^2/(8 nu) + g R b/(2nu)

    written in terms of the *second moment of the sampled coordinates* rather
    than ``H^2/3`` / ``R^2/2``: on a discrete grid the arithmetic (or area
    weighted) average of ``s^2`` differs from its continuum value by
    ``O(h^2)``, and because ``b`` is again a small difference the bias is not
    negligible (measured: ~0.005 lu on a 33-node plane channel).  Passing
    ``coord`` removes it exactly::

        plane  u_mean = g/(2 nu)(H^2 - <s^2>) + g H b/nu
        pipe   u_mean = g/(4 nu)(R^2 - <r^2>) + g R b/(2nu)

    with ``<s^2>`` the arithmetic mean of ``(y - y_c)^2`` (plane) or the
    ``r``-weighted mean of ``r^2`` (pipe).  ``u_mean`` may be supplied directly
    (e.g. the exact area-weighted mean); otherwise the mean of ``u`` is used.
    """
    uu = np.asarray(u, dtype=float)
    um = float(np.mean(uu)) if u_mean is None else float(u_mean)
    if coord is not None:
        c = np.asarray(coord, dtype=float)
        if geometry == "pipe":
            w = np.abs(c)
            m2 = float(np.sum(w * c * c) / np.sum(w))
        else:
            s_ = c - np.mean(c)
            m2 = float(np.mean(s_ * s_))
        source = "bulk_mean_sampled_second_moment"
    else:
        m2 = length ** 2 / 2.0 if geometry == "pipe" else length ** 2 / 3.0
        source = "bulk_mean_continuum_second_moment"
    if geometry == "pipe":
        core = g / (4.0 * nu) * (length ** 2 - m2)
        b = (um - core) * 2.0 * nu / (g * length)
    else:
        core = g / (2.0 * nu) * (length ** 2 - m2)
        b = (um - core) * nu / (g * length)
    return {"b": float(b), "b_source": source, "u_mean": um,
            "u_mean_noslip": float(core), "second_moment": m2,
            "geometry": geometry}


def model_slip_length(
    coord: Sequence[float],
    u: Sequence[float],
    length: float,
    nu: float,
    g: float,
    geometry: str = "plane",
    band: tuple[float, float] = (0.5, 1.0),
) -> dict:
    """Slip length with the analytic curvature fixed (only the offset is fitted).

    ``u_analytic(y) = u_noslip(s) + g L b/nu`` with ``s`` the distance from the
    centre.  Because ``u_noslip`` is known in closed form, the unknown ``b``
    enters *linearly* and the estimate is the mean of
    ``u - u_noslip`` over the band ``band[0]*L <= s <= band[1]*L``.  No
    curvature is fitted, so the ``A + B L^2`` cancellation -- the mechanism by
    which the quadratic fit amplifies curvature error by ``L^2`` -- disappears.
    """
    c = np.asarray(coord, dtype=float)
    uu = np.asarray(u, dtype=float)
    s = np.abs(c - 0.5 * (c.min() + c.max())) if geometry == "plane" else np.abs(c)
    if geometry == "pipe":
        core = g / (4.0 * nu) * (length ** 2 - s ** 2)
        factor = 2.0
    else:
        core = g / (2.0 * nu) * (length ** 2 - s ** 2)
        factor = 1.0
    keep = (s >= band[0] * length) & (s <= band[1] * length)
    if keep.sum() < 3:
        raise ValueError("band contains too few points")
    d = float(np.mean(uu[keep] - core[keep]))
    b = d * factor * nu / (g * length)
    return {
        "b": float(b),
        "b_source": "model_fixed_curvature",
        "band": [float(band[0]), float(band[1])],
        "points": int(keep.sum()),
        "offset_deviation": d,
        "spread_along_band": float(np.std(uu[keep] - core[keep])),
        "geometry": geometry,
    }


# ---------------------------------------------------------------------------
# Free-slip limit signature
# ---------------------------------------------------------------------------
def freeslip_signature(result: Channel3DResult) -> dict:
    """Dynamical fingerprint of ``b -> inf`` (pure specular wall).

    Under a pure specular wall the body force has nothing to balance, so the
    flow is a *plug* that accelerates linearly in time: no steady state, zero
    curvature, momentum growing at a constant rate.  Recording that signature
    turns ``sigma = 1`` into a documented limit rather than a crash.
    """
    inv = ps.slip_length_from_quadratic_fit(
        np.arange(1, result.ny - 1, dtype=float), result.u_profile,
        length=result.length, nu=ps.nu_from_tau(result.tau), g=result.g,
        geometry="plane",
    )
    umax = max(result.u_max, 1e-300)
    return {
        "sigma": result.sigma,
        "converged": result.converged,
        "residual": result.residual,
        "profile_flatness": float((result.u_profile.max() - result.u_profile.min()) / umax),
        "curvature_ratio": inv["curvature_ratio"],
        "momentum_growth_per_1000_steps": result.momentum_growth,
        "steps": result.steps,
        "verdict": "free-slip (b->inf) signature: plug profile + secular energy "
                   "growth, no steady state",
    }


def plane_slip_law_3d(sigmas: Sequence[float], results=None, **solver_kwargs) -> dict:
    """Calibrate ``b(sigma)`` from D3Q19 runs and fit the law ``a s/(1-s) + c``."""
    if results is None:
        results = [channel_poiseuille_3d(s, **solver_kwargs) for s in sigmas]
    bs, model_bs, bulk_bs = [], [], []
    nu = ps.nu_from_tau(results[0].tau)
    for res in results:
        inv = ps.slip_length_from_quadratic_fit(
            np.arange(1, res.ny - 1, dtype=float), res.u_profile,
            length=res.length, nu=nu, g=res.g, geometry="plane",
        )
        bs.append(inv["b"])
        model_bs.append(model_slip_length(
            np.arange(1, res.ny - 1, dtype=float), res.u_profile,
            res.length, nu, res.g, geometry="plane")["b"])
        bulk_bs.append(bulk_mean_slip_length(res.u_profile, res.length, nu, res.g)["b"])
    law = ps.fit_slip_law([r.sigma for r in results], bs)
    return {
        "sigmas": [r.sigma for r in results],
        "b_quadratic": bs,
        "b_model": model_bs,
        "b_bulk": bulk_bs,
        "law": law,
        "n": results[0].ny,
        "H": results[0].length,
    }
