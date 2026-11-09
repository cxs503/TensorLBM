"""Two-way FSI coupling and its momentum-exchange residual (criterion C3).

The chain has a large scale separation: the floes (D50 ~ 1 m) are ~1/230 of the
hull length, so the water phase is solved on a lattice where a floe is *sub-cell*.
The two-way coupling is therefore organised in two branches:

* ice -> flow (blocking): the floe positions are rasterised into a local ice
  coverage map, which sets the local porosity of the porous surface layer and
  hence the Darcy-Forchheimer momentum sink of the water solution
  (:meth:`tensorlbm.ice_fullship.lbm_porous.PorousLBM`).
* flow -> ice (drag): the water velocity is sampled at the floe positions and
  turned into a drag force on the floe (Schiller-Naumann style quadratic drag on
  the floe's frontal and submerged plan area).

Criterion C3 (momentum-exchange residual < 1 %) is the audit that the two
branches move the *same* momentum: the sink is re-integrated on the lattice from
an **independently re-rasterised** coverage map (recomputing porosity,
Kozeny-Carman permeability and the Ergun inertial coefficient from the floe
geometry alone) and compared with the sink the solver actually applied.  A
mapping bug (wrong normalisation, wrong layer weighting, double counting) shows
up immediately, while the residual stays at interpolation level for a correct
implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .lbm_porous import PorousLBM, PorousLBMConfig
from .floe_field import FloeField


def rasterise_coverage(field: FloeField, ny: int, nx: int, cfg: PorousLBMConfig,
                       sub: int = 4) -> np.ndarray:
    """Ice coverage map ``(ny, nx)`` in the lattice from the floe positions.

    The floe field is expressed in physical units over ``[0, L] x [0, W]``; it is
    mapped onto the lattice's hull-covering x-window and the full width.  Each
    lattice cell is sub-sampled ``sub x sub`` times to obtain a fractional
    coverage in ``[0, 1]``.
    """
    cov = np.zeros((ny, nx), dtype=np.float64)
    nys, nxs = ny * sub, nx * sub
    yy = ((np.arange(nys) + 0.5) / nys - 0.5) * (2.0 * (cfg.n_halfbeam + cfg.side_margin_cells)) / cfg.cells_per_metre
    # x window covering the ice band of the solver
    x_lo = 0.0
    x_hi = cfg.nx / cfg.cells_per_metre
    xx = x_lo + (np.arange(nxs) + 0.5) / nxs * (x_hi - x_lo)
    dxs = (x_hi - x_lo) / nxs
    dys = (yy[-1] - yy[0]) / nys
    cell_area = dxs * dys
    mask = np.zeros((nys, nxs), dtype=bool)
    # floes are given on [0, L] x [0, W] of their own domain
    sx = nxs / field.length
    sy = nys / field.width
    yr = yy[0]
    for xc, yc, dia in zip(field.x, field.y, field.d):
        r = 0.5 * dia
        i0 = int(max(0, math.floor((xc - r) * sx)))
        i1 = int(min(nxs - 1, math.ceil((xc + r) * sx)))
        j0 = int(max(0, math.floor((yc - r - yr) * sy)))
        j1 = int(min(nys - 1, math.ceil((yc + r - yr) * sy)))
        if i1 < i0 or j1 < j0:
            continue
        cx = xx[i0:i1 + 1]
        cy = yy[j0:j1 + 1]
        d2 = (cx[None, :] - xc) ** 2 + (cy[:, None] - yc) ** 2
        mask[j0:j1 + 1, i0:i1 + 1] |= d2 <= r * r
    cov_sub = mask.astype(np.float64)
    cov = cov_sub.reshape(ny, sub, nx, sub).mean(axis=(1, 3))
    return np.clip(cov, 0.0, 1.0)


def coverage_to_porous_coefficients(cov: np.ndarray, cfg: PorousLBMConfig,
                                    concentration: float) -> Tuple[np.ndarray, np.ndarray]:
    """Independently rebuild the local porous coefficients from a coverage map.

    ``beta``  : viscous (Darcy)    coefficient, ``nu / K``
    ``gamma`` : inertial (Forchheimer) coefficient, ``C_F / sqrt(K)``
    with per-cell porosity ``eps = 1 - c`` and the Kozeny-Carman / Ergun closures.
    The vertical distribution weight (ice draft / cell height / n_layer) is the
    same one the solver uses - if the two implementations disagree the residual
    of criterion C3 exposes it.
    """
    c = np.clip(cov / max(concentration, 1e-9), 0.0, 0.999)
    eps = 1.0 - c * concentration
    has_ice = c > 0.0
    d50_lat = max(1e-3, cfg.ice_d50 * cfg.cells_per_metre)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(has_ice, d50_lat ** 2 * eps ** 3
                     / np.where(has_ice, 150.0 * (1.0 - eps) ** 2, 1.0), 1.0)
        C_F = np.where(has_ice, 1.75 / np.sqrt(np.where(has_ice, 150.0 * eps ** 3, 1.0)), 0.0)
    draft = cfg.ice_thickness * (917.0 / 1025.0)
    dz = cfg.T / cfg.n_draft
    frac = min(1.0, draft / dz)
    w = frac / max(cfg.ice_layer_cells, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(has_ice, (cfg.nu / np.where(has_ice, K, 1.0)) * w, 0.0)
        gamma = np.where(has_ice, (C_F / np.sqrt(np.where(has_ice, K, 1.0))) * w, 0.0)
    return beta, gamma


def momentum_exchange_audit(solver: PorousLBM, field: FloeField,
                            sub: int = 8) -> Dict[str, float]:
    """Criterion C3: Eulerian porous sink vs independently rerasterised sink.

    The audit re-rasterises the floe field with a *finer* sub-sampling than the
    map the solver was built with, so the residual is the coarse-graining error of
    the ice->flow mapping (it is not a tautology: an inconsistent normalisation,
    a wrong layer weight or a double-counted coverage would show up as a large
    residual, while a correct mapping gives the interpolation-level value here).
    """
    cfg = solver.cfg
    cov = rasterise_coverage(field, cfg.ny, cfg.nx, cfg, sub=sub)
    beta, gamma = coverage_to_porous_coefficients(cov, cfg, cfg.ice_concentration)

    # solver-side coefficients (the ones actually applied), summed over the layer
    beta_s = solver.beta.sum(dim=0).numpy()
    gamma_s = solver.gamma.sum(dim=0).numpy()

    rho, ux, uy, uz = solver.macroscopic()
    u_slip_x = (ux - solver._u_ice[0]).numpy()
    u_slip_y = (uy - solver._u_ice[1]).numpy()
    u_slip_z = (uz - solver._u_ice[2]).numpy()
    speed = np.sqrt(u_slip_x ** 2 + u_slip_y ** 2 + u_slip_z ** 2)
    rho_np = rho.numpy()
    ice_mask = solver.ice.numpy()          # (nz, ny, nx)
    nz = cfg.nz

    def to3d(b2):
        out = np.zeros((nz,) + b2.shape)
        out[:] = b2[None, :, :]
        return out * ice_mask

    beta3, gamma3 = to3d(beta), to3d(gamma)

    def sink(b, g):
        fac = (b + g * speed) * rho_np
        return (-fac * u_slip_x, -fac * u_slip_y, -fac * u_slip_z)

    eul = sink(beta_s, gamma_s)
    lag = sink(beta, gamma)
    num = math.sqrt(sum(float((((a - b)[ice_mask]) ** 2).sum()) for a, b in zip(eul, lag)))
    den = math.sqrt(sum(float(((a[ice_mask]) ** 2).sum()) for a in eul))
    residual = num / max(den, 1e-30)
    return {
        "sink_momentum_euler": [float(a[ice_mask].sum()) for a in eul],
        "sink_momentum_lagrange": [float(a[ice_mask].sum()) for a in lag],
        "residual_rel": float(residual),
        "coverage_mean": float(cov.mean()),
        "coverage_max": float(cov.max()),
        "ice_cells": int(ice_mask.sum()),
        "pass": bool(residual < 0.01),
    }


def floe_drag_forces(solver: PorousLBM, field: FloeField, rho_water: float = 1025.0,
                     cd_frontal: float = 1.0, cd_planar: float = 5.0e-3,
                     sub: int = 4) -> Dict[str, object]:
    """Flow -> ice branch: drag on each floe from the sampled water velocity.

    The floe positions are mapped into the lattice; the water velocity is sampled
    by nearest cell (the floe is sub-cell, so bilinear sampling is not physically
    meaningful) and the drag uses the slip velocity to the drifting ice.
    """
    cfg = solver.cfg
    rho, ux, uy, uz = solver.macroscopic()
    ux = ux.numpy()
    uy = uy.numpy()
    uz = uz.numpy()
    ny, nx = cfg.ny, cfg.nx
    # floe -> lattice index mapping
    ix = np.clip((field.x / field.length * nx).astype(int), 0, nx - 1)
    iy = np.clip(((field.y / field.width - 0.5) * ny + ny / 2).astype(int), 0, ny - 1)
    k = cfg.nz - 1
    usx = ux[k, iy, ix] - cfg.u_lat * cfg.ice_drift_ratio
    usy = uy[k, iy, ix]
    usz = uz[k, iy, ix]
    speed = np.sqrt(usx ** 2 + usy ** 2 + usz ** 2)
    # lattice -> physical velocity scale
    scale = cfg.L_pp / cfg.n_hull  # metres per lattice velocity unit (U_lat = ship speed)
    speed = speed * scale
    area = 0.25 * np.pi * field.d ** 2
    frontal = field.d * field.thickness
    coef = 0.5 * rho_water * (cd_frontal * frontal + cd_planar * area) * speed
    fx = coef * usx * scale
    fy = coef * usy * scale
    fz = coef * usz * scale
    return {
        "n_floes": int(field.n),
        "F_drag_x_sum": float(fx.sum()),
        "F_drag_y_sum": float(fy.sum()),
        "F_drag_z_sum": float(fz.sum()),
        "F_drag_x_mean": float(fx.mean()),
        "speed_mean_m_s": float(speed.mean()),
        "speed_max_m_s": float(speed.max()),
    }