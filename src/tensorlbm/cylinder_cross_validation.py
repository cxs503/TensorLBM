"""D2Q9 cylinder cross-validation: collision × turbulence Cd matrix.

Runs a 2-D cylinder flow at Re=100 on a small grid (100×50, 200 steps) for
every combination of D2Q9 collision family (BGK, MRT, TRT, RLBM) and LES
turbulence model (none, Smagorinsky, WALE, Vreman), producing a
machine-readable Cd comparison matrix.

Design notes
------------
* **No solver hot-path changes.**  Per-cell ``tau_eff`` wrappers for MRT,
  TRT, and RLBM live here, not in :mod:`tensorlbm.solver`.
* Turbulence eddy-viscosity helpers are reused from
  :mod:`tensorlbm.turbulence` (``_smagorinsky_tau``, ``_wale_nu_t_2d``,
  ``_vreman_nu_t_2d``, ``_nu_t_to_tau_eff``).
* ``status="diagnostic_only"`` and ``physical_validation=False`` on every
  result — this is a numerical cross-check, not a physical validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import torch

from .boundaries import (
    apply_simple_channel_boundaries,
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
    make_channel_wall_mask,
)
from .d2q9 import equilibrium, macroscopic
from .solver import (
    OPPOSITE_2D,
    _get_d2q9_mrt_matrices,
    collide_bgk,
    collide_mrt,
    collide_rlbm,
    collide_trt,
    stream,
)
from .turbulence import (
    _neq_stress_norm_2d,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _vreman_nu_t_2d,
    _wale_nu_t_2d,
)

CollisionFamily = Literal["BGK", "MRT", "TRT", "RLBM"]
TurbulenceModel = Literal["none", "Smagorinsky", "WALE", "Vreman"]

D2Q9_COLLISION_FAMILIES: list[str] = ["BGK", "MRT", "TRT", "RLBM"]
D2Q9_TURBULENCE_MODELS: list[str] = ["none", "Smagorinsky", "WALE", "Vreman"]


# ---------------------------------------------------------------------------
# Per-cell tau_eff collision wrappers
# ---------------------------------------------------------------------------

def _compute_tau_eff_field(
    f: torch.Tensor,
    tau: float,
    turbulence_model: str,
) -> torch.Tensor | None:
    """Compute per-cell effective relaxation time for a turbulence model.

    Returns ``None`` for ``"none"`` (use scalar *tau*).
    """
    if turbulence_model == "none":
        return None

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq

    if turbulence_model == "Smagorinsky":
        pi_norm = _neq_stress_norm_2d(f_neq)
        return _smagorinsky_tau(tau, pi_norm, rho, C_s=0.1)

    if turbulence_model == "WALE":
        nu_t = _wale_nu_t_2d(ux, uy, C_w=0.5)
        return _nu_t_to_tau_eff(tau, nu_t)

    if turbulence_model == "Vreman":
        nu_t = _vreman_nu_t_2d(ux, uy, C_V=0.025)
        return _nu_t_to_tau_eff(tau, nu_t)

    raise ValueError(f"Unknown turbulence model: {turbulence_model}")


def _collide_bgk_field(
    f: torch.Tensor,
    tau: float,
    tau_eff: torch.Tensor | None,
) -> torch.Tensor:
    """BGK collision with optional per-cell tau_eff field."""
    if tau_eff is None:
        return collide_bgk(f, tau)
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq
    return f - f_neq / tau_eff.unsqueeze(0)


def _collide_mrt_field(
    f: torch.Tensor,
    tau: float,
    tau_eff: torch.Tensor | None,
    s_e: float = 1.64,
    s_eps: float = 1.54,
    s_q: float = 1.7,
) -> torch.Tensor:
    """MRT collision with optional per-cell tau_eff field.

    When *tau_eff* is None, delegates to :func:`collide_mrt`.
    Otherwise the stress modes (rows 7, 8) use the per-cell
    ``1/tau_eff`` rate; all other rates are fixed.
    """
    if tau_eff is None:
        return collide_mrt(f, tau, s_e=s_e, s_eps=s_eps, s_q=s_q)

    device = f.device
    matrix, matrix_inv = _get_d2q9_mrt_matrices(device, f.dtype)

    s_nu_field = 1.0 / tau_eff  # (ny, nx)
    ny, nx = f.shape[1], f.shape[2]

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)

    f_flat = f.reshape(9, -1)
    feq_flat = feq.reshape(9, -1)
    s_nu_flat = s_nu_field.reshape(-1)

    s_fixed = torch.tensor(
        [0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, 0.0],
        dtype=f.dtype,
        device=device,
    )

    m = matrix @ f_flat
    m_eq = matrix @ feq_flat
    dm = m - m_eq

    m_star = m - s_fixed.unsqueeze(1) * dm
    for k in (7, 8):
        m_star[k] = m[k] - s_nu_flat * dm[k]

    return (matrix_inv @ m_star).reshape(9, ny, nx)


def _collide_trt_field(
    f: torch.Tensor,
    tau: float,
    tau_eff: torch.Tensor | None,
    lambda_trt: float = 3.0 / 16.0,
) -> torch.Tensor:
    """TRT collision with optional per-cell tau_eff field.

    When *tau_eff* is None, delegates to :func:`collide_trt`.
    Otherwise both tau_plus and the derived tau_minus become per-cell.
    """
    if tau_eff is None:
        return collide_trt(f, tau_plus=tau, lambda_trt=lambda_trt)

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)

    tau_plus = tau_eff  # (ny, nx)
    tau_minus = 0.5 + lambda_trt / (tau_plus - 0.5)

    opp = OPPOSITE_2D.to(f.device)
    f_plus = 0.5 * (f + f[opp])
    f_minus = 0.5 * (f - f[opp])
    feq_plus = 0.5 * (feq + feq[opp])
    feq_minus = 0.5 * (feq - feq[opp])

    return (
        f
        - (f_plus - feq_plus) / tau_plus.unsqueeze(0)
        - (f_minus - feq_minus) / tau_minus.unsqueeze(0)
    )


def _collide_rlbm_field(
    f: torch.Tensor,
    tau: float,
    tau_eff: torch.Tensor | None,
) -> torch.Tensor:
    """Regularized BGK (RLBM) collision with optional per-cell tau_eff field.

    When *tau_eff* is None, delegates to :func:`collide_rlbm`.
    Otherwise the relaxation rate ``1/tau`` is replaced by the per-cell
    ``1/tau_eff`` in the post-collision expression.
    """
    if tau_eff is None:
        return collide_rlbm(f, tau)

    from .d2q9 import _c_on, _w_on

    device = f.device
    c = _c_on(device).to(f.dtype)
    w = _w_on(device).to(f.dtype)

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    fneq = f - feq

    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    pi_xx = (cx * cx * fneq).sum(dim=0)
    pi_yy = (cy * cy * fneq).sum(dim=0)
    pi_xy = (cx * cy * fneq).sum(dim=0)

    cs2 = 1.0 / 3.0
    h_xx = cx * cx - cs2
    h_yy = cy * cy - cs2
    h_xy = cx * cy
    w_view = w.view(9, 1, 1)
    fneq_reg = (9.0 / 2.0) * w_view * (
        h_xx * pi_xx + h_yy * pi_yy + 2.0 * h_xy * pi_xy
    )

    inv_tau_eff = (1.0 / tau_eff).unsqueeze(0)  # (1, ny, nx)
    return feq + (1.0 - inv_tau_eff) * fneq_reg


_COLLIDE_DISPATCH: dict[str, Any] = {
    "BGK": _collide_bgk_field,
    "MRT": _collide_mrt_field,
    "TRT": _collide_trt_field,
    "RLBM": _collide_rlbm_field,
}


# ---------------------------------------------------------------------------
# Cylinder flow runner
# ---------------------------------------------------------------------------

def run_single_combination(
    collision_family: str,
    turbulence_model: str,
    re: float = 100.0,
    nx: int = 100,
    ny: int = 50,
    steps: int = 200,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one collision × turbulence combination and return a result dict.

    The result dict always contains the keys:
        collision_family, turbulence_model, Cd, finite,
        steps_completed, status, physical_validation
    """
    collide_fn = _COLLIDE_DISPATCH[collision_family]

    radius = 6.0
    u_in = 0.06
    nu = u_in * 2.0 * radius / re
    tau = 3.0 * nu + 0.5

    dev = torch.device(device)
    mask = cylinder_mask(nx, ny, nx // 3, ny // 2, radius, device=dev)
    wall_mask = make_channel_wall_mask(ny, nx, mask, device=dev)

    rho0 = torch.ones(ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0), device=dev)

    cd_list: list[float] = []
    steps_completed = 0
    crashed = False

    for step in range(1, steps + 1):
        try:
            tau_eff = _compute_tau_eff_field(f, tau, turbulence_model)
            f = collide_fn(f, tau, tau_eff)
            f = stream(f)
            fx, fy = compute_obstacle_forces(f, mask)
            f = apply_simple_channel_boundaries(
                f,
                u_in=u_in,
                wall_mask=wall_mask,
                obstacle_mask=torch.zeros_like(mask),
            )
            f = bounce_back_cells(f, mask)

            if not torch.isfinite(fx).item() or not torch.isfinite(f).all().item():
                crashed = True
                break

            if step > steps // 3:
                cd = float(fx.item()) / (0.5 * u_in**2 * 2.0 * radius)
                cd_list.append(cd)

            steps_completed = step
        except Exception:
            crashed = True
            break

    if cd_list and not crashed:
        cd_mean = sum(cd_list) / len(cd_list)
        finite = math.isfinite(cd_mean)
    else:
        cd_mean = float("nan")
        finite = False

    return {
        "collision_family": collision_family,
        "turbulence_model": turbulence_model,
        "Cd": cd_mean,
        "finite": finite,
        "steps_completed": steps_completed,
        "status": "diagnostic_only",
        "physical_validation": False,
    }


def run_cross_validation_matrix(
    re: float = 100.0,
    nx: int = 100,
    ny: int = 50,
    steps: int = 200,
    artifact_path: str | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run the full 4×4 collision × turbulence matrix.

    If *artifact_path* is given, writes a JSON artifact with the full
    matrix and metadata.
    """
    results: list[dict[str, Any]] = []

    for cf in D2Q9_COLLISION_FAMILIES:
        for tm in D2Q9_TURBULENCE_MODELS:
            result = run_single_combination(
                cf, tm, re=re, nx=nx, ny=ny, steps=steps, device=device
            )
            results.append(result)

    if artifact_path is not None:
        artifact = {
            "description": "D2Q9 cylinder cross-validation: collision × turbulence Cd matrix",
            "lattice": "D2Q9",
            "reynolds_number": re,
            "grid": {"nx": nx, "ny": ny},
            "steps": steps,
            "status": "diagnostic_only",
            "physical_validation": False,
            "collision_families": D2Q9_COLLISION_FAMILIES,
            "turbulence_models": D2Q9_TURBULENCE_MODELS,
            "matrix": results,
        }
        Path(artifact_path).write_text(json.dumps(artifact, indent=2, default=str))

    return results


__all__ = [
    "D2Q9_COLLISION_FAMILIES",
    "D2Q9_TURBULENCE_MODELS",
    "run_single_combination",
    "run_cross_validation_matrix",
]
