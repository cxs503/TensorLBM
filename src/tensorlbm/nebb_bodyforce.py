"""Force-corrected NEBB wall + body-force driver for D2Q9 (NumPy, float64).

Why this module
---------------
TensorLBM had a body-force path (Guo forcing in :mod:`tensorlbm.gpu_core_path`)
and a crude non-equilibrium bounce-back form (``wave_bc_common``: ``f_k =
feq_k - feq_opp + f_opp``), but no *force-corrected* velocity boundary and no
driver that assembles the three ingredients a pressure-driven / gravity-driven
wall-bounded LBM run needs together:

1. a wall velocity shifted by half a force step
   (:func:`corrected_wall_velocity`, ``u_wall - g/(2 rho)``);
2. a wall kernel that imposes ``(rho_wall, u_wall)`` on the boundary node
   exactly (:func:`nebb_row`, shear-reconstruction NEBB);
3. a post-processing velocity correction (:func:`velocity_correction`,
   ``+ g/(2 rho)``) that restores the second-order-accurate physical velocity.

Cross-validation provenance (XVAL-LBM-FORCE-CORRECTED-NEBB-BODY-FORCE-DRIVER-S)
------------------------------------------------------------------------------
The做法 of items 1-3 and of the two body-force schemes below is taken from the
mature open-source implementation **haimingz/vivsim** (JAX).  Every formula in
this module maps 1:1 onto a named upstream function, so the port can be audited
line by line (no hand-copy: the upstream tree used was ``/tmp/vivsim_src``,
recorded in ``xval/REPORT_XVAL_NEBB_BODYFORCE.md``):

===========================================  ==========================================
this module                                  vivsim source
===========================================  ==========================================
:func:`nebb_row`                             ``lbm/boundary/nebb.py::boundary_nebb``
:func:`corrected_wall_velocity`              ``lbm/boundary/_helpers.py::get_corrected_wall_velocity``
:func:`velocity_correction`                  ``lbm/basic.py::get_velocity_correction``
:func:`guo_forcing_term`                     ``lbm/forcing/guo.py::get_guo_forcing_term``
:func:`forcing_edm`                          ``lbm/forcing/edm.py::forcing_edm``
:func:`forcing_guo_bgk`                      ``lbm/forcing/guo.py::forcing_guo_bgk``
:func:`equilibrium` / :func:`macroscopic`    ``lbm/basic.py::get_equilibrium`` / ``get_macroscopic``
:class:`BodyForceChannel` step order          ``examples/2d/poiseuille_channel.py``
===========================================  ==========================================

Two deliberate deviations from the upstream code, both documented in
``xval/DIFF_OSS_VS_TENSORLBM.md``:

* **Un-shifted populations.**  vivsim stores ``f_i = f_i^full - w_i`` for
  precision; this module stores the full populations.  The NEBB algebra is
  invariant under the shift (all four weights involved in a 2D wall kernel are
  equal: ``w_pure = 1/9``, ``w_diag = 1/36``), so the ported expression is
  term-by-term identical -- asserted by
  ``tests/test_nebb_bodyforce.py::test_nebb_invariant_under_population_shift``.
* **Backend-agnostic wall kernel.**  :func:`nebb_row` only uses slicing,
  element-wise arithmetic and ``stack``/item assignment, so it runs on NumPy
  arrays (this driver, cluster nodes without torch) *and* on ``torch`` tensors
  (used by :mod:`tensorlbm.boundary.antibb_channel_benchmark`).

Array layout: ``f`` is ``(9, nx, ny)`` and row ``j`` of a wall is
``f[:, :, 0]`` / ``f[:, :, -1]`` (the OSS layout; most other TensorLBM 2-D
kernels use ``(9, ny, nx)`` -- the wall kernel takes a ``(9, N)`` slice instead
of a whole field precisely so both layouts can call it).

Entry point::

    python -m tensorlbm.nebb_bodyforce --ny 10 --scheme edm --out runs/nebb

prints (and optionally writes) the profile, the analytic comparison and the
diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "C",
    "W",
    "OPPOSITE",
    "WallSpec",
    "wall_spec",
    "equilibrium",
    "macroscopic",
    "guo_forcing_term",
    "forcing_edm",
    "forcing_guo_bgk",
    "nebb_row",
    "nonequilibrium_nebb_row",
    "wall_density",
    "recovered_wall_density",
    "WALL_MODES",
    "corrected_wall_velocity",
    "velocity_correction",
    "apply_force_corrected_nebb",
    "analytic_poiseuille",
    "effective_wall_offset",
    "profile_errors",
    "ChannelConfig",
    "ChannelResult",
    "BodyForceChannel",
    "main",
]

# --------------------------------------------------------------------------- #
# D2Q9 lattice (numbering identical to ``tensorlbm.d2q9`` and vivsim ``D2Q9``)
# --------------------------------------------------------------------------- #
#        6   2   5
#          \ | /
#        3 - 0 - 1
#          / | \
#        7   4   8
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
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)

assert np.all(C[OPPOSITE] == -C), "OPPOSITE must be the velocity reversal"
assert abs(float(W.sum()) - 1.0) < 1e-15, "weights must sum to one"
# the NEBB port is only shift-invariant when both weights used by a 2-D wall
# kernel are direction-independent (see module docstring)
assert W[1] == W[3] == W[2] == W[4] and W[5] == W[6] == W[7] == W[8]


@dataclass(frozen=True)
class WallSpec:
    """Direction bookkeeping for one straight wall of a 2-D domain.

    Matches ``vivsim.lbm.lattice.D2Q9.boundary_spec[loc]`` field by field but is
    *derived* from the lattice constants above (:func:`wall_spec`) instead of
    being tabulated by hand, so a lattice edit cannot leave stale indices.
    """

    loc: str
    normal_axis: int          # 0 = x, 1 = y
    normal_sign: int          # +1 if the fluid is on the +axis side
    in_dirs: tuple[int, ...]  # populations streaming *into* the domain (unknown)
    out_dirs: tuple[int, ...]  # their opposites (known at the wall node)
    tan_dirs: tuple[int, ...]  # the two pure directions tangent to the wall

    @property
    def tan_axis(self) -> int:
        return 1 - self.normal_axis


#: wall models supported by :class:`BodyForceChannel`
WALL_MODES = (
    "force_corrected_nebb",            # OSS kernel, rho_wall from the field
    "force_corrected_nebb_recovered",  # OSS kernel + OSS density recovery
    "nonequilibrium_nebb",             # classic Zou/He NEBB (feq difference)
    "halfway_bounce_back",             # no force correction at all
)

_WALL_SIGNS = {"bottom": (1, +1), "top": (1, -1), "left": (0, +1), "right": (0, -1)}


def wall_spec(loc: str) -> WallSpec:
    """Derive the :class:`WallSpec` for ``loc`` from the lattice constants."""
    if loc not in _WALL_SIGNS:
        raise ValueError(f"unknown wall {loc!r}; expected one of {sorted(_WALL_SIGNS)}")
    axis, sign = _WALL_SIGNS[loc]
    normal = tuple(i for i in range(9) if C[i, axis] * sign > 0 and C[i, 1 - axis] == 0)
    mixed = [i for i in range(9) if C[i, axis] * sign > 0 and C[i, 1 - axis] != 0]
    # slot 1 carries +tangential, slot 2 -tangential (vivsim's in_dirs order);
    # the ordering rule is derived, not tabulated
    mixed.sort(key=lambda i: -sign * int(C[i, 1 - axis]))
    in_dirs = normal + tuple(mixed)
    if len(in_dirs) != 3:  # pragma: no cover - lattice consistency guard
        raise AssertionError(f"{loc}: expected 3 incoming directions, got {in_dirs}")
    out_dirs = tuple(int(OPPOSITE[i]) for i in in_dirs)
    tan_dirs = tuple(i for i in range(1, 5) if C[i, axis] == 0)
    if len(tan_dirs) != 2:  # pragma: no cover
        raise AssertionError(f"{loc}: expected 2 tangential directions")
    return WallSpec(loc, axis, sign, in_dirs, out_dirs, tan_dirs)


# --------------------------------------------------------------------------- #
# macroscopic / equilibrium (vivsim ``basic.py``)
# --------------------------------------------------------------------------- #
def equilibrium(rho, ux, uy):
    """D2Q9 second-order equilibrium, shape ``(9, *rho.shape)``."""
    rho = np.asarray(rho, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    usq = ux * ux + uy * uy
    cu = C[:, 0, None, None] * ux[None] + C[:, 1, None, None] * uy[None]
    return W[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def macroscopic(f):
    """Return ``(rho, ux, uy)`` -- zeroth and first moments of ``f``."""
    rho = f.sum(axis=0)
    ux = (f * C[:, 0, None, None]).sum(axis=0) / rho
    uy = (f * C[:, 1, None, None]).sum(axis=0) / rho
    return rho, ux, uy


# --------------------------------------------------------------------------- #
# body-force schemes (vivsim ``forcing/``)
# --------------------------------------------------------------------------- #
def guo_forcing_term(gx, gy, ux, uy):
    """Guo's lattice forcing term ``w_i [3 (c_i-u).g + 9 (c_i.u)(c_i.g)]``.

    ``gx, gy`` are *force densities* (per volume), not accelerations: the
    resulting momentum source is ``g`` and the velocity correction is
    ``g/(2 rho)``.  Port of ``vivsim.lbm.forcing.guo.get_guo_forcing_term``.
    """
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    cx = C[:, 0, None, None]
    cy = C[:, 1, None, None]
    cu = cx * ux[None] + cy * uy[None]
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    body = gx[None] * (3.0 * (cx - ux[None]) + 9.0 * cu * cx) + gy[None] * (
        3.0 * (cy - uy[None]) + 9.0 * cu * cy
    )
    return W[:, None, None] * body


def forcing_edm(f, gx, gy, ux, uy):
    """Exact-difference-method forcing: ``f + guo_forcing_term(g, u)``.

    Port of ``vivsim.lbm.forcing.edm.forcing_edm`` (after Kupershtokh; the
    upstream docstring proves it equals ``feq(u+g/2rho) - feq(u-g/2rho)`` and
    hence Guo's scheme without the half-step relaxation factor).
    """
    return f + guo_forcing_term(gx, gy, ux, uy)


def forcing_guo_bgk(f, gx, gy, ux, uy, omega):
    """Guo forcing for BGK: ``f + (1 - omega/2) * guo_forcing_term``.

    Port of ``vivsim.lbm.forcing.guo.forcing_guo_bgk``.
    """
    return f + guo_forcing_term(gx, gy, ux, uy) * (1.0 - 0.5 * omega)


# --------------------------------------------------------------------------- #
# walls
# --------------------------------------------------------------------------- #
def _post(F):
    """Backend namespace for ``stack`` (NumPy module or ``torch``)."""
    if isinstance(F, np.ndarray):
        return np
    import torch  # local import: torch is optional for the NumPy driver

    return torch


def _like(x, ref):
    """Convert ``x`` to the same array backend/dtype/device as ``ref``."""
    if isinstance(ref, np.ndarray):
        return np.asarray(x, dtype=np.float64)
    import torch

    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def nebb_row(wall, spec: WallSpec, rho_wall=1.0, ux_wall=0.0, uy_wall=0.0):
    """Non-equilibrium bounce-back with shear reconstruction on one wall row.

    ``wall`` is the ``(9, N)`` slice of the populations *on* the boundary node
    after streaming.  The three unknown incoming populations are rebuilt from
    the known outgoing ones plus the prescribed ``(rho_wall, ux_wall, uy_wall)``;
    the tangential (shear) part is taken from the actual population difference
    across the wall node instead of from its equilibrium value, which is what
    ``vivsim.lbm.boundary.nebb.boundary_nebb`` does.

    Returns a new array of the same shape/type; works for NumPy and ``torch``.

    Notes
    -----
    With a *known* wall density this is algebraically the classic Zou/He
    NEBB ``f_i = feq_i + (f_opp - feq_opp)`` whenever the wall row's tangential
    populations carry only their equilibrium difference; the two forms are
    both implemented here (:func:`nonequilibrium_nebb_row`) and their
    difference is what the OSS keeps as a second-order correction.
    """
    s = spec.normal_sign
    un = ux_wall if spec.normal_axis == 0 else uy_wall
    ut = uy_wall if spec.normal_axis == 0 else ux_wall
    un = s * _like(un, wall)
    ut = s * _like(ut, wall)
    rho = _like(rho_wall, wall)

    i0, i1, i2 = spec.in_dirs
    o0, o1, o2 = spec.out_dirs
    t0, t1 = spec.tan_dirs

    shear = 0.5 * (wall[t0] - wall[t1]) * s
    normal_term = un * rho / 6.0
    tangential_term = ut * rho / 2.0

    xp = _post(wall)
    new_vals = xp.stack(
        (
            wall[o0] + (2.0 / 3.0) * un * rho,
            wall[o1] - shear + normal_term + tangential_term,
            wall[o2] + shear + normal_term - tangential_term,
        )
    )
    out = wall.clone() if hasattr(wall, "clone") else wall.copy()
    out[list(spec.in_dirs)] = new_vals
    return out


def nonequilibrium_nebb_row(wall, spec: WallSpec, rho_wall=1.0, ux_wall=0.0, uy_wall=0.0):
    """Classic Zou/He NEBB on one wall row, ``f_i = feq_i + (f_opp - feq_opp)``.

    Kept as the reference form the shear-reconstruction kernel is compared
    against (the TensorLBM ``wave_bc_common`` NEBB is the same expression); it
    needs the same ``(9, N)`` wall slice and returns a NumPy array.
    """
    wall = np.asarray(wall, dtype=np.float64)
    N = wall.shape[1]
    rho = np.broadcast_to(np.asarray(rho_wall, dtype=np.float64), (N,))
    ux = np.broadcast_to(np.asarray(ux_wall, dtype=np.float64), (N,))
    uy = np.broadcast_to(np.asarray(uy_wall, dtype=np.float64), (N,))
    feq = equilibrium(rho, ux, uy)
    out = wall.copy()
    for i in spec.in_dirs:
        j = int(OPPOSITE[i])
        out[i] = feq[i] + (wall[j] - feq[j])
    return out


def wall_density(f, loc: str):
    """Density on the wall row recovered from the populations there."""
    spec = wall_spec(loc)
    idx = 0 if loc in ("bottom", "left") else -1
    wall = f[..., idx] if spec.normal_axis == 1 else f[:, idx, :]
    return wall.sum(axis=0)


def recovered_wall_density(f, loc: str, ux_wall=0.0, uy_wall=0.0):
    """Density that makes the NEBB reconstruction of this row self-consistent.

    Port of ``vivsim.lbm.boundary._helpers.get_rho_wall_from_velocity``
    (``numerator / (1 - u_n)``): the NEBB row total is
    ``rho_before + (sum_out - sum_in) + u_n rho_wall``, so this choice of
    ``rho_wall`` makes the reconstructed row carry exactly ``rho_wall`` while the
    prescribed wall velocity is imposed exactly.  The OSS ``wrap_velocity``
    wrapper uses it; the Poiseuille example uses the ``rho_wall = 1`` default of
    ``wrap_force_corrected`` instead (difference measured: 6.4e-14 relative,
    see ``xval/DIFF_OSS_VS_TENSORLBM.md``).
    """
    spec = wall_spec(loc)
    idx = 0 if loc in ("bottom", "left") else -1
    wall = f[:, :, idx] if spec.normal_axis == 1 else f[:, idx, :]
    rho = wall.sum(axis=0)
    sum_in = wall[list(spec.in_dirs)].sum(axis=0)
    sum_out = wall[list(spec.out_dirs)].sum(axis=0)
    un = spec.normal_sign * (ux_wall if spec.normal_axis == 0 else uy_wall)
    return (rho - sum_in + sum_out) / (1.0 - un)


def corrected_wall_velocity(ux_wall=0.0, uy_wall=0.0, gx=0.0, gy=0.0, rho_wall=1.0):
    """``u_wall - g/(2 rho_wall)`` -- half-force-step wall correction.

    Port of ``vivsim.lbm.boundary._helpers.get_corrected_wall_velocity``.
    """
    return (
        ux_wall - velocity_correction(gx, rho_wall),
        uy_wall - velocity_correction(gy, rho_wall),
    )


def velocity_correction(g, rho=1.0):
    """``g/(2 rho)``: add to the moment velocity to get the physical velocity.

    Port of ``vivsim.lbm.basic.get_velocity_correction``.
    """
    return np.asarray(g, dtype=np.float64) * 0.5 / np.asarray(rho, dtype=np.float64)


def apply_force_corrected_nebb(f, loc, gx=0.0, gy=0.0, rho_wall=None, ux_wall=0.0, uy_wall=0.0):
    """Apply the force-corrected NEBB to the ``loc`` wall of a ``(9, nx, ny)`` field.

    ``rho_wall=None`` recovers the wall density from ``f`` (needed because the
    half-step velocity correction is ``g/(2 rho_wall)``); pass a float to
    reproduce the OSS example's ``rho_wall = 1`` default.
    """
    spec = wall_spec(loc)
    idx = 0 if loc in ("bottom", "left") else -1
    if spec.normal_axis == 1:
        wall = f[:, :, idx]
    else:
        wall = f[:, idx, :]
    if rho_wall is None:
        rho_wall = wall.sum(axis=0)
    elif isinstance(rho_wall, str):
        if rho_wall != "recover":
            raise ValueError(f"rho_wall must be a float, None or 'recover', got {rho_wall!r}")
        # one Picard pair: the half-step correction depends on rho_wall
        uxw, uyw = corrected_wall_velocity(ux_wall, uy_wall, gx, gy, 1.0)
        rho_wall = recovered_wall_density(f, loc, uxw, uyw)
    uxw, uyw = corrected_wall_velocity(ux_wall, uy_wall, gx, gy, rho_wall)
    new_wall = nebb_row(wall, spec, rho_wall, uxw, uyw)
    out = f.copy()
    if spec.normal_axis == 1:
        out[:, :, idx] = new_wall
    else:
        out[:, idx, :] = new_wall
    return out


# --------------------------------------------------------------------------- #
# analytic reference + metrics
# --------------------------------------------------------------------------- #
def analytic_poiseuille(ny: int, gx: float, nu: float, wall_offset: float = 0.0):
    """Poiseuille profile ``gx/(2 nu) (y+o)(H-(y+o))``, ``H = ny-1+2o``.

    ``wall_offset`` is the distance of the no-slip surface *outside* the first /
    last node centre: ``0`` places it on the boundary nodes (what the OSS
    example plots), ``0.5`` places it half a cell outside (the convention of
    halfway bounce-back and of the discrete NEBB steady state, see
    :func:`effective_wall_offset`).
    """
    o = float(wall_offset)
    H = (ny - 1) + 2.0 * o
    y = np.arange(ny, dtype=np.float64) + o
    return gx / (2.0 * nu) * y * (H - y)


def effective_wall_offset(profile, gx: float, nu: float) -> float:
    """Wall offset implied by a measured steady profile (least squares).

    Fits ``u = a y^2 + b y + c`` over the interior rows, then uses the known
    curvature ``a = -gx/(2 nu)`` to read off the two zero crossings and hence
    ``o = -y_bottom``.  A steady body-force channel should return a constant
    ``o`` (independent of ``ny``) -- that constancy is what tells the
    lattice-convention offset apart from a numerical error.
    """
    u = np.asarray(profile, dtype=np.float64)
    ny = u.size
    if ny < 4:
        raise ValueError("need at least 4 rows to fit the parabola")
    y = np.arange(1, ny - 1, dtype=np.float64)
    a_fit, b_fit, c_fit = np.polyfit(y, u[1:-1], 2)
    a = -gx / (2.0 * nu)
    disc = b_fit * b_fit - 4.0 * a * c_fit
    if a == 0.0 or disc < 0.0:  # pragma: no cover - degenerate input
        return float("nan")
    r = math.sqrt(disc)
    y_lo, y_hi = (-b_fit - r) / (2.0 * a), (-b_fit + r) / (2.0 * a)
    return float(-min(y_lo, y_hi))


def profile_errors(got, ref) -> dict[str, float]:
    """Relative error metrics of a 1-D profile against a reference."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    scale = max(float(np.abs(ref).max()), 1e-300)
    d = np.abs(got - ref)
    out = {
        "max_abs": float(d.max()),
        "max_rel_global": float(d.max() / scale),
        "l2_rel_global": float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300)),
        "umax_got": float(got.max()),
        "umax_ref": float(ref.max()),
    }
    out["umax_rel_err"] = float(abs(out["umax_got"] - out["umax_ref"]) / scale)
    if got.size > 2:
        interior = slice(1, -1)
        denom = np.maximum(np.abs(ref[interior]), 1e-12)
        out["max_rel_interior"] = float((d[interior] / denom).max())
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChannelConfig:
    """Configuration of a body-force-driven plane channel."""

    nx: int = 8
    ny: int = 10
    nu: float = 0.2
    gx: float = 1e-3
    gy: float = 0.0
    scheme: str = "edm"                  # "edm" | "guo"
    wall: str = "force_corrected_nebb"   # see WALL_MODES
    init: str = "force_shifted"          # "force_shifted" (OSS example) | "rest"
    steps: int = 200_000
    steady_tol: float = 1e-13
    check_every: int = 200

    @property
    def tau(self) -> float:
        return 3.0 * self.nu + 0.5

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    def validate(self) -> None:
        if self.ny < 4:
            raise ValueError("need at least 4 rows (2 wall rows + 2 fluid rows)")
        if self.nx < 1:
            raise ValueError("nx must be positive")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for a positive viscosity")
        if self.scheme not in ("edm", "guo"):
            raise ValueError(f"unknown forcing scheme {self.scheme!r}")
        if self.wall not in WALL_MODES:
            raise ValueError(f"unknown wall model {self.wall!r}")
        if self.init not in ("force_shifted", "rest"):
            raise ValueError(f"unknown initialisation {self.init!r}")


@dataclass
class ChannelResult:
    """Steady fields + diagnostics of one channel run."""

    config: ChannelConfig
    ux: np.ndarray
    uy: np.ndarray
    rho: np.ndarray
    ux_corrected: np.ndarray
    steps_used: int
    steady_residual: float
    mass_initial: float
    mass_final: float
    wall_velocity_raw: float
    wall_velocity_corrected: float
    value_at_walls: float
    history: dict[str, list] = field(default_factory=dict)

    @property
    def profile(self) -> np.ndarray:
        """Streamwise velocity averaged over ``x`` on the *corrected* field."""
        return self.ux_corrected.mean(axis=0)

    @property
    def mass_drift(self) -> float:
        return (self.mass_final - self.mass_initial) / self.mass_initial

    def summary(self) -> dict[str, Any]:
        cfg = asdict(self.config)
        out = {
            "config": cfg,
            "steps_used": self.steps_used,
            "steady_residual": self.steady_residual,
            "mass_drift": self.mass_drift,
            "wall_velocity_raw": self.wall_velocity_raw,
            "wall_velocity_corrected": self.wall_velocity_corrected,
            "umax": float(self.profile.max()),
            "profile": self.profile.tolist(),
            "profile_raw": self.ux.mean(axis=0).tolist(),
        }
        ref_o0 = analytic_poiseuille(self.config.ny, self.config.gx, self.config.nu, 0.0)
        out["analytic_offset0"] = ref_o0.tolist()
        out["errors_offset0"] = profile_errors(self.profile, ref_o0)
        out["effective_wall_offset"] = effective_wall_offset(
            self.profile, self.config.gx, self.config.nu
        )
        ref_of = analytic_poiseuille(
            self.config.ny, self.config.gx, self.config.nu, out["effective_wall_offset"]
        )
        out["errors_offset_fitted"] = profile_errors(self.profile, ref_of)
        return out


class BodyForceChannel:
    """Body-force-driven plane channel: periodic ``x``, two no-slip wall rows.

    Step order is the OSS example's
    (``examples/2d/poiseuille_channel.py``)::

        rho, u = macro(f); feq = feq(rho, u); collide;
        f += forcing(g, u); stream; wall(top); wall(bottom)

    with ``g`` in force-density units.  The physical velocity is
    ``u + g/(2 rho)`` (:func:`velocity_correction`) -- the moment velocity is
    *not* corrected inside the loop, exactly like the OSS implementation.
    """

    def __init__(self, config: ChannelConfig):
        config.validate()
        self.config = config
        self.f: np.ndarray | None = None
        self.history: dict[str, list] = {"step": [], "u_mid": [], "mass": [], "residual": []}
        self.mass_initial = 0.0
        self.reset()

    # -- state ---------------------------------------------------------- #
    def reset(self) -> None:
        cfg = self.config
        rho = np.ones((cfg.nx, cfg.ny))
        if cfg.init == "force_shifted":
            # OSS initialisation: f = feq(rho, u - g/2); the minus sign is the
            # half-step force shift (vivsim example comment "initial distribution")
            ux = np.full((cfg.nx, cfg.ny), -cfg.gx / 2.0)
            uy = np.full((cfg.nx, cfg.ny), -cfg.gy / 2.0)
        else:
            ux = np.zeros((cfg.nx, cfg.ny))
            uy = np.zeros((cfg.nx, cfg.ny))
        self.f = equilibrium(rho, ux, uy)
        self.mass_initial = float(self.f.sum())
        self.history = {"step": [], "u_mid": [], "mass": [], "residual": []}
        self._u_prev: np.ndarray | None = None

    # -- one LBM step ---------------------------------------------------- #
    def step(self) -> None:
        """Advance one collision -> forcing -> streaming -> wall step."""
        cfg = self.config
        omega = cfg.omega
        rho, ux, uy = macroscopic(self.f)

        # OSS recipe (examples/2d/poiseuille_channel.py, update_step_bgk_*):
        # EDM collides at the *uncorrected* moment velocity, Guo-BGK at the
        # force-corrected one.  Mixing this up halves or doubles the injected
        # momentum (measured: 45 % error at ny=10), so it is part of the API.
        if cfg.scheme == "guo":
            ux_f = ux + velocity_correction(cfg.gx, rho)
            uy_f = uy + velocity_correction(cfg.gy, rho)
        else:
            ux_f, uy_f = ux, uy

        feq = equilibrium(rho, ux_f, uy_f)
        f_post = (1.0 - omega) * self.f + omega * feq
        if cfg.scheme == "edm":
            f_post = forcing_edm(f_post, cfg.gx, cfg.gy, ux, uy)
        else:
            f_post = forcing_guo_bgk(f_post, cfg.gx, cfg.gy, ux_f, uy_f, omega)

        # streaming: f_i(x) <- f_i(x - c_i), periodic in x (periodic y too, but
        # the wall rows are overwritten below before the next collision)
        f = f_post.copy()
        for i in range(1, 9):
            f[i] = np.roll(f_post[i], (int(C[i, 0]), int(C[i, 1])), axis=(0, 1))

        if cfg.wall.startswith("force_corrected_nebb"):
            rho_policy = "recover" if cfg.wall.endswith("_recovered") else None
            f = apply_force_corrected_nebb(f, "bottom", gx=cfg.gx, gy=cfg.gy, rho_wall=rho_policy)
            f = apply_force_corrected_nebb(f, "top", gx=cfg.gx, gy=cfg.gy, rho_wall=rho_policy)
        elif cfg.wall == "nonequilibrium_nebb":
            for loc in ("bottom", "top"):
                spec = wall_spec(loc)
                idx = 0 if loc == "bottom" else -1
                wall = f[:, :, idx]
                rho_wall = wall.sum(axis=0)
                uxw, uyw = corrected_wall_velocity(0.0, 0.0, cfg.gx, cfg.gy, rho_wall)
                f[:, :, idx] = nonequilibrium_nebb_row(wall, spec, rho_wall, uxw, uyw)
        else:
            # halfway bounce-back: the unknown populations on a wall node are
            # the *post-collision* opposites of that same node
            for loc, idx in (("bottom", 0), ("top", -1)):
                spec = wall_spec(loc)
                f[list(spec.in_dirs), :, idx] = f_post[list(spec.out_dirs), :, idx]
        self.f = f

    # -- driver ---------------------------------------------------------- #
    def _probe(self) -> tuple[np.ndarray, float]:
        rho, ux, uy = macroscopic(self.f)
        # corrected physical velocity on the mid column
        uxc = ux + velocity_correction(self.config.gx, rho)
        return uxc.mean(axis=0), float(self.f.sum())

    def run(self, steps: int | None = None) -> ChannelResult:
        cfg = self.config
        n = cfg.steps if steps is None else int(steps)
        steps_used = n
        residual = float("nan")
        prev = None
        for k in range(n):
            self.step()
            if cfg.check_every and (k % cfg.check_every == 0 or k == n - 1):
                prof, mass = self._probe()
                self.history["step"].append(k + 1)
                self.history["mass"].append(mass)
                self.history["u_mid"].append(float(prof[self.config.ny // 2]))
                if prev is not None:
                    res = float(np.abs(prof - prev).max())
                    self.history["residual"].append(res)
                    if res < cfg.steady_tol:
                        steps_used, residual = k + 1, res
                        return self._result(steps_used, residual)
                prev = prof
                residual = self.history["residual"][-1] if self.history["residual"] else residual
        return self._result(steps_used, residual)

    def _result(self, steps_used: int, residual: float) -> ChannelResult:
        cfg = self.config
        rho, ux, uy = macroscopic(self.f)
        uxc = ux + velocity_correction(cfg.gx, rho)
        uyc = uy + velocity_correction(cfg.gy, rho)
        return ChannelResult(
            config=cfg,
            ux=ux,
            uy=uy,
            rho=rho,
            ux_corrected=uxc,
            steps_used=steps_used,
            steady_residual=residual,
            mass_initial=self.mass_initial,
            mass_final=float(self.f.sum()),
            wall_velocity_raw=float(ux[:, 0].mean()),
            wall_velocity_corrected=float(uxc[:, 0].mean()),
            value_at_walls=float(np.abs(uyc[0, :]).max()),
            history={k: list(v) for k, v in self.history.items()},
        )


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #
def _cli_table(rows: Sequence[dict[str, Any]]) -> str:
    head = f"{'scheme':<8}{'wall':<24}{'ny':>4}{'umax':>12}{'ana(o=0)':>12}{'err(o=0)':>11}{'err(fit o)':>12}{'o_fit':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['scheme']:<8}{r['wall']:<24}{r['ny']:>4}{r['umax']:>12.6f}"
            f"{r['umax_analytic']:>12.6f}{r['err_global_offset0']:>11.5f}"
            f"{r['err_global_offset_fit']:>12.3e}{r['wall_offset']:>8.3f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=10, help="rows; rows 0 and ny-1 are wall nodes")
    ap.add_argument("--nu", type=float, default=0.2)
    ap.add_argument("--gx", type=float, default=1e-3)
    ap.add_argument("--scheme", default="edm", choices=("edm", "guo"))
    ap.add_argument(
        "--wall",
        default="force_corrected_nebb",
        choices=WALL_MODES,
    )
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--steady-tol", type=float, default=1e-13)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cfg = ChannelConfig(
        nx=args.nx,
        ny=args.ny,
        nu=args.nu,
        gx=args.gx,
        scheme=args.scheme,
        wall=args.wall,
        steps=args.steps,
        steady_tol=args.steady_tol,
    )
    res = BodyForceChannel(cfg).run()
    summ = res.summary()
    summ["profile_analytic_offset0"] = analytic_poiseuille(cfg.ny, cfg.gx, cfg.nu, 0.0).tolist()
    row = {
        "scheme": cfg.scheme,
        "wall": cfg.wall,
        "ny": cfg.ny,
        "umax": summ["umax"],
        "umax_analytic": float(analytic_poiseuille(cfg.ny, cfg.gx, cfg.nu, 0.0).max()),
        "err_global_offset0": summ["errors_offset0"]["max_rel_global"],
        "err_global_offset_fit": summ["errors_offset_fitted"]["max_rel_global"],
        "wall_offset": summ["effective_wall_offset"],
    }
    print(_cli_table([row]))
    print(
        f"steps_used={res.steps_used} steady_residual={res.steady_residual:.3e} "
        f"mass_drift={res.mass_drift:.3e} u_wall_raw={res.wall_velocity_raw:.3e} "
        f"u_wall_corrected={res.wall_velocity_corrected:.3e}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summ, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())