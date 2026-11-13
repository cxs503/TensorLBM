#!/usr/bin/env python3
"""XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION -- cross-validation bench.

Question
--------
``hietwll/LBM_Taichi`` prescribes boundary populations with the *equilibrium
difference* rule ``f(b) = f_eq(b) - f_eq(1) + f(1)`` (``lbm_solver.py:130``,
inventoried in ``docs/oss_inventory_lbm_taichi.md``; its OSS-SOL slug is
``equilibrium-difference-extrapolation``).  This bench asks, measurably:

  1. is the rule as ported into TensorLBM
     (:mod:`tensorlbm.boundary.equilibrium_extrapolation`) *the same rule* --
     proven algebraically and against an independent NumPy transcription in the
     OSS's array-of-structs layout, not by inspection;
  2. what accuracy does it deliver against closed-form solutions (plane
     Poiseuille, plane Couette) and against the OSS repo's own validation case
     (lid-driven cavity vs Ghia 1982 Re = 100) -- inside the owner's 15 % gate?
  3. which of the OSS modelling choices cost accuracy: single-node
     :math:`f^{neq}` extrapolation (its ``order=1``) or the copied wall density
     (``rho_mode='copy'``)?

Arms
----
``oss1``  the OSS rule exactly as documented (``order=1, rho_mode='copy'``)
``guo2``  Guo/Zheng/Shi (2002) linear :math:`f^{neq}` extrapolation from two
          inner nodes + extrapolated wall density (``order=2, rho_mode='extrapolate'``)
``bb``    control arm: half-way bounce-back (TensorLBM's standard wall) -- a
          different wall model at a different geometric location, i.e. the
          alternative the OSS rule has to beat to be worth adopting

Predeclared decision (fixed before any result was inspected)
------------------------------------------------------------
* Ship ``oss1`` as a documented, low-cost boundary option **iff** every case is
  inside 15 % of its reference and the algebra identity holds to 1e-12.
* Prefer ``guo2`` as the default extrapolation variant **iff** its observed
  convergence order on the Poiseuille ladder exceeds ``oss1``'s by >= 0.7
  (``oss1`` should be first order there: :math:`f^{neq}` varies linearly across
  the wall-normal direction, exactly what a single-node copy discards).
* Couette is the *falsification* case: there :math:`f^{neq}` is constant in y, so
  ``oss1`` must reproduce the linear profile to well under 1 %.  If it does not,
  the port is wrong rather than merely low order.

Everything in ``docs/XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION.md`` and in
``figures/xval_equilibrium_difference_extrapolation/`` is rendered from
``runs/xval_equilibrium_difference_extrapolation/result.json`` by this script,
so no number in the write-up is typed by hand.

Run::

    python benchmarks/xval_equilibrium_difference_extrapolation.py            # all cases + docs
    python benchmarks/xval_equilibrium_difference_extrapolation.py --quick    # CI-sized
    python benchmarks/xval_equilibrium_difference_extrapolation.py --case poiseuille --no-docs
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.boundary.equilibrium_extrapolation import (  # noqa: E402
    EquilibriumExtrapolationSpec,
    apply_cavity_extrapolation,
    apply_channel_extrapolation,
    apply_equilibrium_extrapolation,
    oss_equilibrium_difference,
)
from tensorlbm.d2q9 import C, OPPOSITE, W_EXACT, equilibrium, macroscopic  # noqa: E402
from tensorlbm.lid_driven_cavity import (  # noqa: E402
    GHIA_RE100,
    centreline_coordinates,
    compare_ghia,
)

RUN_DIR = ROOT / "runs" / "xval_equilibrium_difference_extrapolation"
ARTIFACT_DIR = ROOT / "artifacts" / "xval_equilibrium_difference_extrapolation"
DOC_PATH = ROOT / "docs" / "XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION.md"
FIG_DIR = ROOT / "figures" / "xval_equilibrium_difference_extrapolation"

#: owner precision standard: reference deviation > 15 % is not acceptable
GATE_PCT = 15.0
#: algebra-identity tolerance for the OSS form vs the modular implementation
IDENTITY_TOL = 1e-12
ARM_INFO = {
    "oss1": {"spec": EquilibriumExtrapolationSpec(order=1, rho_mode="copy"), "wall_offset": 0.0,
             "description": "OSS equilibrium-difference rule (single inner node, rho copied)"},
    "guo2": {"spec": EquilibriumExtrapolationSpec(order=2, rho_mode="extrapolate"), "wall_offset": 0.0,
             "description": "Guo(2002) 2-node f_neq extrapolation + extrapolated rho"},
    "bb": {"spec": None, "wall_offset": -0.5,
           "description": "half-way bounce-back control arm (solid node rows/cols)"},
}
ARM_ORDER = ("oss1", "guo2", "bb")


# --------------------------------------------------------------------------- #
# lattice kernels (D2Q9 BGK + Guo forcing), float64, (9, ny, nx)
# --------------------------------------------------------------------------- #
_CX = [int(v) for v in C[:, 0].tolist()]
_CY = [int(v) for v in C[:, 1].tolist()]
_OPP = [int(v) for v in OPPOSITE.tolist()]
_W = list(W_EXACT)
_STREAM_IDX: dict = {}


def _weights(f: torch.Tensor) -> torch.Tensor:
    return torch.tensor(_W, dtype=f.dtype, device=f.device).view(9, 1, 1)


def _stream_index(ny: int, nx: int, periodic_x: bool) -> torch.Tensor:
    """Flat source-index map for pull streaming, cached per geometry."""
    key = (ny, nx, periodic_x)
    if key not in _STREAM_IDX:
        idx = np.zeros((9, ny * nx), dtype=np.int64)
        for i in range(9):
            for y in range(ny):
                for x in range(nx):
                    sy, sx = y - _CY[i], x - _CX[i]
                    if periodic_x:
                        sx %= nx
                    # out-of-domain pulls (only wall rows/columns need them) land on
                    # the node itself; each such entry is overwritten by the boundary
                    # prescription or by bounce-back before it can be read back.
                    idx[i, y * nx + x] = sy * nx + sx if (0 <= sy < ny and 0 <= sx < nx) else y * nx + x
        _STREAM_IDX[key] = torch.from_numpy(idx)
    return _STREAM_IDX[key]


def stream_pull(f: torch.Tensor, periodic_x: bool) -> torch.Tensor:
    """Pull streaming ``f_new[i, y, x] = f[i, y - cy, x - cx]`` (single gather)."""
    ny, nx = f.shape[1], f.shape[2]
    idx = _stream_index(ny, nx, periodic_x)
    return f.reshape(9, -1).gather(1, idx).reshape(9, ny, nx)


def guo_forcing(rho, ux, uy, fx, fy, tau, f: torch.Tensor) -> torch.Tensor:
    """Guo (2002) second-order forcing source term ``S_i`` for a force density ``F``."""
    w = _weights(f)
    cxt = torch.tensor(_CX, dtype=f.dtype, device=f.device).view(9, 1, 1)
    cyt = torch.tensor(_CY, dtype=f.dtype, device=f.device).view(9, 1, 1)
    cu = cxt * ux + cyt * uy
    cF = cxt * fx + cyt * fy
    uF = ux * fx + uy * fy
    return (1.0 - 0.5 / tau) * w * (3.0 * (cF - uF) + 9.0 * cu * cF)


def collide(f, tau, force=None):
    """BGK collision with optional constant body-force density ``force=(fx, fy)``."""
    rho, ux, uy = macroscopic(f)
    if force is not None:
        fx, fy = force
        ux = ux + 0.5 * fx / rho
        uy = uy + 0.5 * fy / rho
    feq = equilibrium(rho, ux, uy)
    out = f - (f - feq) / tau
    if force is not None:
        out = out + guo_forcing(rho, ux, uy, force[0], force[1], tau, f)
    return out


def momentum(f, force=None):
    """Velocity field including the half-force correction (Guo forcing)."""
    rho, ux, uy = macroscopic(f)
    if force is not None:
        ux = ux + 0.5 * force[0] / rho
        uy = uy + 0.5 * force[1] / rho
    return rho, ux, uy


def _bb_directions(wall: str) -> list[int]:
    """Reflected (outgoing) directions of a wall: those pointing *away* from it.

    Half-way bounce-back sets the outgoing population from the post-collision
    population in the opposite (incident) direction,
    ``f_j(x_f) = f_{opp j}^post(x_f) + 6 w_j rho (c_j.u_w)``; the ``+6 w rho (c.u_w)``
    sign follows from requiring ``f = f_eq`` to be reproduced for a fluid moving
    with the wall (``f_eq_{opp j} - f_eq_j = -6 w_j rho (c_j.u)``).
    """
    if wall == "bottom":
        return [i for i in range(9) if _CY[i] > 0]
    if wall == "top":
        return [i for i in range(9) if _CY[i] < 0]
    if wall == "left":
        return [i for i in range(9) if _CX[i] > 0]
    if wall == "right":
        return [i for i in range(9) if _CX[i] < 0]
    raise ValueError(wall)


def bounce_back(f_new, f_post, walls, u_wall: dict[str, float], rho_w=1.0):
    """Half-way bounce-back with moving walls, applied to the fluid-node strip."""
    ny, nx = f_new.shape[1], f_new.shape[2]
    strips = {
        "bottom": (slice(1, 2), slice(None)),
        "top": (slice(ny - 2, ny - 1), slice(None)),
        "left": (slice(None), slice(1, 2)),
        "right": (slice(None), slice(nx - 2, nx - 1)),
    }
    out = f_new
    w = _weights(f_new)
    for wall in walls:
        sy, sx = strips[wall]
        uw = u_wall.get(wall, 0.0)
        for i in _bb_directions(wall):
            j = _OPP[i]
            # +6 w_i rho (c_i . u_w) pulls the wall's tangential momentum into the fluid
            out[i, sy, sx] = f_post[j, sy, sx] + 6.0 * w[i] * rho_w * uw * _CX[i]
    return out


# --------------------------------------------------------------------------- #
# independent NumPy transcription of the OSS rule (array-of-structs layout)
# --------------------------------------------------------------------------- #
def oss_rule_numpy(f_row: np.ndarray, rho_b: float, ux_b: float, uy_b: float,
                   rho_1: float, ux_1: float, uy_1: float) -> np.ndarray:
    """Textbook transcription of ``f(b) = feq(b) - feq(1) + f(1)`` in plain NumPy.

    ``f_row`` is the 9-component population vector of the *inner* node (AoS, the
    layout the Taichi implementation uses); the equilibrium is expanded by hand
    from the D2Q9 weights, i.e. no code is shared with the production kernel.
    Used only as an independent implementation check.
    """

    def feq(rho, ux, uy):
        cu = np.array([cx * ux + cy * uy for cx, cy in zip(_CX, _CY)])
        u2 = ux * ux + uy * uy
        return np.array(_W) * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)

    return feq(rho_b, ux_b, uy_b) - feq(rho_1, ux_1, uy_1) + f_row


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    case: str
    arm: str
    params: dict
    metrics: dict
    steps: int
    wall_time_s: float
    extra: dict = field(default_factory=dict)


def _residual(ux_now: torch.Tensor, ux_prev: torch.Tensor, scale: float) -> float:
    return float((ux_now - ux_prev).abs().max().item() / scale)


def run_poiseuille(ny: int, nx: int, u_max: float, re_h: float, arm: str,
                   tol: float = 1e-9, max_steps: int = 400_000) -> RunRecord:
    """Force-driven plane Poiseuille, ``Re_H = u_max*H/nu`` held fixed along the ladder."""
    ghost = arm != "bb"
    H = float(ny - 1) if ghost else float(ny - 2)
    nu = u_max * H / re_h
    tau = 3.0 * nu + 0.5
    g0 = 8.0 * nu * u_max / (H * H)  # u(y) = g0 y (H - y) / (2 nu)
    force = (g0, 0.0)
    zero = torch.zeros((ny, nx), dtype=torch.float64)
    f = equilibrium(torch.ones((ny, nx), dtype=torch.float64), zero, zero)
    spec = ARM_INFO[arm]["spec"]
    inner = torch.zeros((ny, nx), dtype=torch.bool)
    inner[1:-1, :] = True

    t0 = time.time()
    steps, residual, converged = 0, float("nan"), False
    u_prev = torch.zeros_like(zero)
    while steps < max_steps:
        f_post = collide(f, tau, force)
        if ghost:
            f_post = apply_channel_extrapolation(f_post, 0.0, 0.0, spec)
            f = stream_pull(f_post, periodic_x=True)
        else:
            f = bounce_back(stream_pull(f_post, periodic_x=True), f_post, ("bottom", "top"), {}, 1.0)
        steps += 1
        if steps % 200 == 0:
            _, ux, _ = momentum(f, force)
            residual = _residual(ux[inner], u_prev[inner], u_max)
            if steps > 400 and residual < tol:
                converged = True
                break
            u_prev = ux.clone()
    wall_s = time.time() - t0

    rho, ux, _ = momentum(f, force)
    prof = ux[:, nx // 2].numpy()
    y_phys = np.arange(ny, dtype=float) if ghost else np.arange(ny, dtype=float) - 0.5
    u_exact = g0 * y_phys * (H - y_phys) / (2.0 * nu)
    u_num, u_ref = prof[1:-1], u_exact[1:-1]
    l2 = float(np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref))
    dudy_num = (-3.0 * prof[1] + 4.0 * prof[2] - prof[3]) / 2.0
    dudy_exact = g0 * (H - 2.0 * y_phys[1]) / (2.0 * nu)
    metrics = {
        "l2_rel_pct": l2 * 100.0,
        "max_abs_dev_pct_of_umax": float(np.abs(u_num - u_ref).max() / u_max * 100.0),
        "wall_shear_rel_pct": float(abs(dudy_num - dudy_exact) / abs(dudy_exact) * 100.0),
        "residual": residual,
        "converged": converged,
        "rho_min": float(rho[inner].min()),
        "rho_max": float(rho[inner].max()),
    }
    return RunRecord("poiseuille", arm,
                     {"ny": ny, "nx": nx, "H": H, "u_max": u_max, "Re_H": re_h, "nu": nu, "tau": tau,
                      "g0": g0, "Ma": u_max * math.sqrt(3.0)},
                     metrics, steps, wall_s,
                     {"profile": prof.tolist(), "y_phys": y_phys.tolist(), "u_exact": u_exact.tolist()})


def run_couette(ny: int, nx: int, U: float, re_h: float, arm: str,
                tol: float = 1e-9, max_steps: int = 400_000) -> RunRecord:
    """Plane Couette: bottom wall moves at U, top wall at rest -- f_neq is constant in y."""
    ghost = arm != "bb"
    H = float(ny - 1) if ghost else float(ny - 2)
    nu = U * H / re_h
    tau = 3.0 * nu + 0.5
    zero = torch.zeros((ny, nx), dtype=torch.float64)
    f = equilibrium(torch.ones((ny, nx), dtype=torch.float64), zero, zero)
    spec = ARM_INFO[arm]["spec"]
    inner = torch.zeros((ny, nx), dtype=torch.bool)
    inner[1:-1, :] = True

    t0 = time.time()
    steps, residual, converged = 0, float("nan"), False
    u_prev = torch.zeros_like(zero)
    while steps < max_steps:
        f_post = collide(f, tau, None)
        if ghost:
            f_post = apply_channel_extrapolation(f_post, U, 0.0, spec)
            f = stream_pull(f_post, periodic_x=True)
        else:
            f = bounce_back(stream_pull(f_post, periodic_x=True), f_post, ("bottom", "top"),
                            {"bottom": U}, 1.0)
        steps += 1
        if steps % 200 == 0:
            _, ux, _ = momentum(f)
            residual = _residual(ux[inner], u_prev[inner], U)
            if steps > 400 and residual < tol:
                converged = True
                break
            u_prev = ux.clone()
    wall_s = time.time() - t0

    rho, ux, _ = momentum(f)
    prof = ux[:, nx // 2].numpy()
    y_phys = np.arange(ny, dtype=float) if ghost else np.arange(ny, dtype=float) - 0.5
    u_exact = U * (1.0 - y_phys / H)
    u_num, u_ref = prof[1:-1], u_exact[1:-1]
    dudy_num = (-3.0 * prof[1] + 4.0 * prof[2] - prof[3]) / 2.0
    metrics = {
        "l2_rel_pct": float(np.linalg.norm(u_num - u_ref) / np.linalg.norm(u_ref)) * 100.0,
        "max_abs_dev_pct_of_U": float(np.abs(u_num - u_ref).max() / U * 100.0),
        "wall_shear_rel_pct": float(abs(dudy_num + U / H) / (U / H) * 100.0),
        "residual": residual,
        "converged": converged,
        "rho_min": float(rho[inner].min()),
        "rho_max": float(rho[inner].max()),
    }
    return RunRecord("couette", arm,
                     {"ny": ny, "nx": nx, "H": H, "U": U, "Re_H": re_h, "nu": nu, "tau": tau,
                      "Ma": U * math.sqrt(3.0)},
                     metrics, steps, wall_s,
                     {"profile": prof.tolist(), "y_phys": y_phys.tolist(), "u_exact": u_exact.tolist()})


def run_cavity(n: int, u_lid: float, re: float, arm: str, tol: float = 1e-8,
               max_steps: int = 400_000) -> RunRecord:
    """Lid-driven cavity at ``Re = u_lid*L/nu``; compared with Ghia (1982) Re = 100."""
    ghost = arm != "bb"
    L = float(n - 1) if ghost else float(n - 2)
    nu = u_lid * L / re
    tau = 3.0 * nu + 0.5
    zero = torch.zeros((n, n), dtype=torch.float64)
    f = equilibrium(torch.ones((n, n), dtype=torch.float64), zero, zero)
    spec = ARM_INFO[arm]["spec"]
    inner = torch.zeros((n, n), dtype=torch.bool)
    inner[1:-1, 1:-1] = True

    t0 = time.time()
    steps, residual, converged = 0, float("nan"), False
    u_prev = torch.zeros_like(zero)
    m_hist: list[tuple[int, float, float]] = []
    while steps < max_steps:
        f_post = collide(f, tau, None)
        if ghost:
            f_post = apply_cavity_extrapolation(f_post, u_lid, spec)
            f = stream_pull(f_post, periodic_x=False)
        else:
            f = bounce_back(stream_pull(f_post, periodic_x=False), f_post,
                            ("bottom", "top", "left", "right"), {"top": u_lid}, 1.0)
        steps += 1
        if steps % 500 == 0:
            _, ux, _ = momentum(f)
            residual = _residual(ux[inner], u_prev[inner], u_lid)
            if steps % 1000 == 0:
                rho_now = f.sum(dim=0)
                m_hist.append((steps, float(rho_now[inner].mean()), float(rho_now.mean())))
            if steps > 1000 and residual < tol:
                converged = True
                break
            u_prev = ux.clone()
    wall_s = time.time() - t0

    rho, ux, uy = momentum(f)
    offset = ARM_INFO[arm]["wall_offset"]
    ghia = compare_ghia(ux, uy, u_lid, GHIA_RE100, wall_offset=offset, exclude_walls=True)
    ux_np = ux.numpy() / u_lid
    y_pos = np.asarray(centreline_coordinates(n, offset))
    ref_y = np.asarray(GHIA_RE100["y"], dtype=float)
    ref_u = np.asarray(GHIA_RE100["u"], dtype=float)
    keep = (ref_y > 0.0) & (ref_y < 1.0)
    got_u = np.interp(ref_y, y_pos, ux_np[:, n // 2])
    dev = np.abs(got_u - ref_u)[keep]
    x_pos = np.asarray(centreline_coordinates(n, offset))
    ref_x = np.asarray(GHIA_RE100["x"], dtype=float)
    ref_v = np.asarray(GHIA_RE100["v"], dtype=float)
    got_v = np.interp(ref_x, x_pos, (uy.numpy() / u_lid)[n // 2, :])
    metrics = {
        "rmse_u_pct_of_lid": ghia["rmse_u"] * 100.0,
        "rmse_v_pct_of_lid": ghia["rmse_v"] * 100.0,
        "max_abs_dev_pct_of_lid": float(dev.max() * 100.0),
        "max_abs_dev_v_pct_of_lid": float(np.abs(got_v - ref_v).max() * 100.0),
        "n_scored_u": ghia["n_scored_u"],
        "u_min_pct_of_lid": float(ux_np.min() * 100.0),
        "mass_drift_rel": float(abs(rho.sum().item() - float(n * n)) / float(n * n)),
        "mass_drift_interior_rel": (
            abs(m_hist[-1][1] - m_hist[0][1]) / m_hist[0][1] if len(m_hist) > 1 else 0.0),
        "mass_rate_per_1000": (
            (m_hist[-1][1] - m_hist[0][1]) / m_hist[0][1] / (m_hist[-1][0] - m_hist[0][0]) * 1000.0
            if len(m_hist) > 1 else 0.0),
        "residual": residual,
        "converged": converged,
    }
    return RunRecord("cavity_re100", arm,
                     {"n": n, "L": L, "u_lid": u_lid, "re": re, "nu": nu, "tau": tau,
                      "Ma": u_lid * math.sqrt(3.0), "wall_offset": offset},
                     metrics, steps, wall_s,
                     {"mass_hist": m_hist,
                      "ghia_y": GHIA_RE100["y"], "ghia_u": GHIA_RE100["u"],
                      "ghia_x": GHIA_RE100["x"], "ghia_v": GHIA_RE100["v"],
                      "got_u": got_u.tolist(), "got_v": got_v.tolist()})


def fit_order(runs: list[RunRecord], key: str = "l2_rel_pct") -> dict:
    """Least-squares slope of log(error) vs log(H) over a grid ladder."""
    hs = np.array([r.params["H"] for r in runs], dtype=float)
    errs = np.array([max(r.metrics[key] / 100.0, 1e-16) for r in runs], dtype=float)
    slope, intercept = np.polyfit(np.log(hs), np.log(errs), 1)
    return {"arm": runs[0].arm, "order": float(slope), "intercept": float(intercept),
            "H": hs.tolist(), "err": errs.tolist()}


def identity_check(seed: int = 12345) -> dict:
    """Port vs literal OSS form (torch) and vs an independent NumPy transcription."""
    gen = torch.Generator().manual_seed(seed)
    ny, nx = 6, 5
    rho = 1.0 + 1e-3 * torch.rand((ny, nx), generator=gen, dtype=torch.float64)
    ux = 0.05 * torch.rand((ny, nx), generator=gen, dtype=torch.float64)
    uy = 0.05 * torch.rand((ny, nx), generator=gen, dtype=torch.float64)
    f = equilibrium(rho, ux, uy) * (1.0 + 0.01 * torch.rand((9, ny, nx), generator=gen, dtype=torch.float64))
    dst = (slice(None), slice(1, 2), slice(None))   # row 1 plays "boundary"
    src1 = (slice(None), slice(2, 3), slice(None))  # row 2 = first inner node
    src2 = (slice(None), slice(3, 4), slice(None))  # row 3 = second inner node
    mod = apply_equilibrium_extrapolation(
        f, dst, src1, src2, 0.0, 0.0, EquilibriumExtrapolationSpec(order=1, rho_mode="copy")
    )
    mod_plane = mod[:, 1, :]
    oss_plane = oss_equilibrium_difference(f, dst, src1, 0.0, 0.0)
    dev_torch = float((mod_plane - oss_plane[:, 0, :]).abs().max().item())

    rho_s, ux_s, uy_s = macroscopic(f)
    dev_np = 0.0
    for j in range(nx):
        ref = oss_rule_numpy(f[:, 2, j].numpy(), float(rho_s[2, j]), 0.0, 0.0,
                             float(rho_s[2, j]), float(ux_s[2, j]), float(uy_s[2, j]))
        dev_np = max(dev_np, float(np.abs(ref - mod_plane[:, j].numpy()).max()))
    # order=2 must differ from order=1 on the same input (otherwise the switch is inert)
    guo_plane = apply_equilibrium_extrapolation(
        f, dst, src1, src2, 0.0, 0.0, EquilibriumExtrapolationSpec(order=2, rho_mode="extrapolate")
    )[:, 1, :]
    return {"torch_modular_vs_oss_form_max_abs": dev_torch,
            "numpy_transcription_vs_torch_max_abs": dev_np,
            "order2_vs_order1_max_abs": float((guo_plane - mod_plane).abs().max().item()),
            "tol": IDENTITY_TOL,
            "pass": bool(dev_torch < IDENTITY_TOL and dev_np < IDENTITY_TOL)}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def collect(quick: bool = False, cases=("identity", "poiseuille", "couette", "cavity")) -> dict:
    ladders = [17, 33] if quick else [17, 33, 65]
    cavity_ns = [33] if quick else [65]
    arms_json = {
        arm: {"description": info["description"], "wall_offset": info["wall_offset"],
              "spec": None if info["spec"] is None else asdict(info["spec"])}
        for arm, info in ARM_INFO.items()
    }
    result = {
        "spec": {"gate_pct": GATE_PCT, "identity_tol": IDENTITY_TOL, "arms": arms_json, "quick": quick,
                 "ladders": ladders, "cavity_n": cavity_ns,
                 "re_h_poiseuille": 10.0, "re_h_couette": 10.0, "u_max": 0.02, "U_couette": 0.02,
                 "cavity_re": 100.0, "cavity_u_lid": 0.1},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                "platform": platform.platform(), "dtype": "float64", "device": "cpu"},
        "runs": [], "orders": {}, "identity": None,
    }
    if "identity" in cases:
        result["identity"] = identity_check()
        print("identity:", result["identity"], flush=True)
    if "poiseuille" in cases:
        for arm in ARM_ORDER:
            recs = []
            for ny in ladders:
                r = run_poiseuille(ny=ny, nx=4, u_max=0.02, re_h=10.0, arm=arm)
                recs.append(r)
                print(f"poiseuille {arm} H={r.params['H']:.0f}: steps={r.steps} conv={r.metrics['converged']} "
                      f"l2={r.metrics['l2_rel_pct']:.3e}% shear={r.metrics['wall_shear_rel_pct']:.3e}% "
                      f"({r.wall_time_s:.1f}s)", flush=True)
            result["runs"].extend(recs)
            result["orders"][f"poiseuille_{arm}"] = fit_order(recs)
    if "couette" in cases:
        for arm in ARM_ORDER:
            recs = []
            for ny in ladders:
                r = run_couette(ny=ny, nx=4, U=0.02, re_h=10.0, arm=arm)
                recs.append(r)
                print(f"couette {arm} H={r.params['H']:.0f}: steps={r.steps} conv={r.metrics['converged']} "
                      f"l2={r.metrics['l2_rel_pct']:.3e}% ({r.wall_time_s:.1f}s)", flush=True)
            result["runs"].extend(recs)
            result["orders"][f"couette_{arm}"] = fit_order(recs)
    if "cavity" in cases:
        for arm in ARM_ORDER:
            for n in cavity_ns:
                r = run_cavity(n=n, u_lid=0.1, re=100.0, arm=arm)
                print(f"cavity {arm} n={n}: steps={r.steps} conv={r.metrics['converged']} "
                      f"rmse_u={r.metrics['rmse_u_pct_of_lid']:.3f}% "
                      f"max={r.metrics['max_abs_dev_pct_of_lid']:.3f}% ({r.wall_time_s:.1f}s)", flush=True)
                result["runs"].append(r)
    return result


def _jsonable(result: dict) -> dict:
    return {"spec": result["spec"], "env": result["env"], "identity": result["identity"],
            "orders": result["orders"], "runs": [asdict(r) for r in result["runs"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short ladders for CI")
    ap.add_argument("--case", default="all", choices=["all", "poiseuille", "couette", "cavity", "identity"])
    ap.add_argument("--no-docs", action="store_true", help="skip figure/report rendering")
    args = ap.parse_args(argv)
    cases = ("identity", "poiseuille", "couette", "cavity") if args.case == "all" else (args.case,)
    result = collect(args.quick, cases)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(result)
    (RUN_DIR / "result.json").write_text(json.dumps(payload, indent=2))
    print("wrote", RUN_DIR / "result.json", flush=True)
    if not args.no_docs:
        from xval_eq_extrap_docs import render  # local helper module next to this file
        render(ROOT, RUN_DIR / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())