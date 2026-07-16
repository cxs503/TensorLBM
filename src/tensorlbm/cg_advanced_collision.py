"""Thin Color-Gradient adapters over reusable D3Q19 stress kernels.

Only recoloring, phase splitting, body-force velocity shift, and solid bounce-back
are CG-specific.  The shared collision is a second-order stress projection.

Maturity disclosure
-------------------
``collide_cg_cumulant_3d`` is a backward-compatible alias for an experimental
regularized-stress CG adapter, not a cumulant transform.  The legacy
``cascaded`` name is a second-order central-stress approximation, not a full
cascaded central-moment scheme.  The legacy ``kbc`` name has no H-entropy,
gamma solve, or positivity proof and is therefore withheld as KBC.  All legacy
names emit ``DeprecationWarning`` and are retained only for API compatibility.
"""
from __future__ import annotations

import warnings
import torch

from .collision_d3q19_advanced import (
    collide_central_stress_d3q19,
    collide_regularized_stress_d3q19,
    reconstruct_second_order_stress_d3q19,
    second_order_stress_d3q19,
)
from .d3q19 import C as C3D, OPPOSITE as OPP, equilibrium3d
from .multiphase3d import _grad_phase_field_3d
from .turbulence import (
    _neq_stress_norm_3d,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _vreman_nu_t_3d,
    _wale_nu_t_3d,
)

_VALID_SGS_MODELS = ("smagorinsky", "wale", "vreman")


def _views(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # CG legacy arithmetic uses fixed float32 lattice constants even for
    # float64 populations; preserving this is required for bitwise replay.
    c = C3D.to(device=f.device, dtype=torch.float32)
    return (c[:, 0].view(19, 1, 1, 1), c[:, 1].view(19, 1, 1, 1),
            c[:, 2].view(19, 1, 1, 1))


# Compatibility-private names now delegate to the single common implementation.
def _stress_tensor_d3q19(f_neq: torch.Tensor, device: torch.device | None = None):
    del device
    return second_order_stress_d3q19(f_neq)


def _reconstruct_fneq_d3q19(
    pi_xx: torch.Tensor, pi_yy: torch.Tensor, pi_zz: torch.Tensor,
    pi_xy: torch.Tensor, pi_xz: torch.Tensor, pi_yz: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Private compatibility adapter; accepts the historical positional device."""
    del device
    return reconstruct_second_order_stress_d3q19(pi_xx, pi_yy, pi_zz, pi_xy, pi_xz, pi_yz)


def _recolor(
    f_total: torch.Tensor, rho_r: torch.Tensor, rho_b: torch.Tensor, rho: torch.Tensor,
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor, device: torch.device | None = None,
    A: float = 0.01, beta: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CG-specific phase recoloring (``A`` remains accepted for compatibility)."""
    del device, A
    cx, cy, cz = _views(f_total)
    rho_safe = rho.clamp(min=1e-12)
    phi, grad_mag, nx, ny, nz = _grad_phase_field_3d(rho_r, rho_b)
    del phi, grad_mag
    feq_r = equilibrium3d(rho_r, ux, uy, uz)
    feq_b = equilibrium3d(rho_b, ux, uy, uz)
    fneq = f_total - (feq_r + feq_b)
    w_r = (0.5 + beta * (cx * nx.unsqueeze(0) + cy * ny.unsqueeze(0) + cz * nz.unsqueeze(0))).clamp(0.0, 1.0)
    return feq_r + w_r * fneq, feq_b + (1.0 - w_r) * fneq


def _phase_state(f_r: torch.Tensor, f_b: torch.Tensor, tau: float, gx: float, gy: float, gz: float):
    f_total = f_r + f_b
    rho_r, rho_b = f_r.sum(0), f_b.sum(0)
    rho = rho_r + rho_b
    cx, cy, cz = _views(f_total)
    rho_safe = rho.clamp(min=1e-12)
    return (f_total, rho_r, rho_b, rho,
            (f_total * cx).sum(0) / rho_safe + tau * gx,
            (f_total * cy).sum(0) / rho_safe + tau * gy,
            (f_total * cz).sum(0) / rho_safe + tau * gz)


def _cg_tau_eff(
    tau: float,
    sgs_model: str,
    f_total: torch.Tensor,
    feq: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    C_s: float = 0.0,
    C_w: float = 0.5,
    C_V: float = 0.025,
) -> torch.Tensor | float:
    """Per-cell effective relaxation time for the CG stress adapters.

    Returns either a scalar ``tau`` (no SGS) or a per-cell tensor
    ``tau_eff(x)`` depending on the selected sub-grid model.

    * ``smagorinsky`` with ``C_s > 0``: Frobenius-norm non-equilibrium stress
      → per-cell ``tau_eff`` (same formula as :func:`collide_smagorinsky_bgk3d`).
      ``C_s = 0`` is a no-op returning the scalar ``tau`` (waLBerla/OpenLB
      pattern).
    * ``wale``: WALE eddy viscosity from velocity gradients →
      ``tau_eff = tau + 3*nu_t`` (always active; ``C_w=0`` ⇒ no-op).
    * ``vreman``: Vreman eddy viscosity → ``tau_eff = tau + 3*nu_t``.

    The velocity-gradient models (WALE, Vreman) use the macroscopic velocity
    field; the uniform body-force shift in ``_phase_state`` is spatially
    constant, so its gradient vanishes and does not bias the eddy viscosity.
    """
    if sgs_model == "smagorinsky":
        if C_s <= 0.0:
            return tau
        pi_norm = _neq_stress_norm_3d(f_total - feq)
        return _smagorinsky_tau(tau, pi_norm, rho, C_s)
    if sgs_model == "wale":
        nu_t = _wale_nu_t_3d(ux, uy, uz, C_w)
        return _nu_t_to_tau_eff(tau, nu_t)
    if sgs_model == "vreman":
        nu_t = _vreman_nu_t_3d(ux, uy, uz, C_V)
        return _nu_t_to_tau_eff(tau, nu_t)
    raise ValueError(
        f"Unknown sgs_model: {sgs_model!r}. "
        f"Expected one of {_VALID_SGS_MODELS}."
    )


def _collide_cg_stress(
    f_r: torch.Tensor, f_b: torch.Tensor, tau: float, A: float, beta: float,
    gx: float, gy: float, gz: float, solid_mask: torch.Tensor | None, s_bulk: float | None,
    sgs_model: str = "smagorinsky", C_s: float = 0.0,
    C_w: float = 0.5, C_V: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    f_total, rho_r, rho_b, rho, ux, uy, uz = _phase_state(f_r, f_b, tau, gx, gy, gz)
    feq = equilibrium3d(rho, ux, uy, uz)
    tau_eff = _cg_tau_eff(
        tau, sgs_model, f_total, feq, rho, ux, uy, uz, C_s, C_w, C_V)
    post = (
        collide_regularized_stress_d3q19(f_total, feq, tau_eff)
        if s_bulk is None
        else collide_central_stress_d3q19(f_total, feq, tau_eff, s_bulk=s_bulk)
    )
    red, blue = _recolor(post, rho_r, rho_b, rho, ux, uy, uz, A=A, beta=beta)
    if solid_mask is not None:
        opp = OPP.to(f_r.device)
        red[:, solid_mask] = red[opp, solid_mask]
        blue[:, solid_mask] = blue[opp, solid_mask]
    return red, blue


def collide_cg_regularized_stress_3d(
    f_r: torch.Tensor, f_b: torch.Tensor, tau: float = 1.0, A: float = 0.01, beta: float = 0.7,
    gx: float = 0.0, gy: float = 0.0, gz: float = 0.0, solid_mask: torch.Tensor | None = None,
    sgs_model: str = "smagorinsky", C_s: float = 0.0, C_w: float = 0.5, C_V: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CG adapter for the verified D3Q19 regularized second-order stress kernel.

    When *sgs_model* is set to ``'wale'`` or ``'vreman'`` (or
    ``'smagorinsky'`` with ``C_s > 0``), a per-cell effective relaxation time
    is computed from the corresponding sub-grid eddy viscosity and passed to
    the stress kernel in place of the scalar *tau*.  The default
    (``sgs_model='smagorinsky', C_s=0``) applies no SGS and is bitwise
    identical to the pre-SGS path.
    """
    return _collide_cg_stress(
        f_r, f_b, tau, A, beta, gx, gy, gz, solid_mask, s_bulk=None,
        sgs_model=sgs_model, C_s=C_s, C_w=C_w, C_V=C_V)


def collide_cg_central_stress_3d(
    f_r: torch.Tensor, f_b: torch.Tensor, tau: float = 1.0, A: float = 0.01, beta: float = 0.7,
    gx: float = 0.0, gy: float = 0.0, gz: float = 0.0, solid_mask: torch.Tensor | None = None,
    s_bulk: float | None = None, C_s: float = 0.0,
    sgs_model: str = "smagorinsky", C_w: float = 0.5, C_V: float = 0.025,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental CG adapter for second-order central-stress relaxation only.

    Accepts the same *sgs_model* / SGS-constant parameters as
    :func:`collide_cg_regularized_stress_3d`.
    """
    # The legacy cascaded operation always used its trace/deviatoric path,
    # including when no explicit bulk rate was supplied.
    effective_bulk = 1.0 / tau if s_bulk is None else s_bulk
    return _collide_cg_stress(
        f_r, f_b, tau, A, beta, gx, gy, gz, solid_mask, s_bulk=effective_bulk,
        sgs_model=sgs_model, C_s=C_s, C_w=C_w, C_V=C_V)


def collide_cg_cumulant_3d(*args, **kwargs):
    """WITHHELD legacy alias; this is not a D3Q19 cumulant collision."""
    warnings.warn("WITHHELD: legacy 'cumulant' is regularized-stress only, not a cumulant implementation", DeprecationWarning, stacklevel=2)
    return collide_cg_regularized_stress_3d(*args, **kwargs)


def collide_cg_cascaded_3d(*args, **kwargs):
    """WITHHELD legacy alias; this is not a full cascaded collision."""
    warnings.warn("WITHHELD: legacy 'cascaded' is second-order central-stress only, not full cascaded CM", DeprecationWarning, stacklevel=2)
    return collide_cg_central_stress_3d(*args, **kwargs)


def collide_cg_kbc_3d(*args, **kwargs):
    """WITHHELD legacy alias; no entropy/gamma/positivity KBC mechanism exists."""
    warnings.warn("WITHHELD: legacy 'kbc' has no H-entropy/gamma/positivity solve and is not KBC", DeprecationWarning, stacklevel=2)
    kwargs.pop("C_s", None)
    return collide_cg_regularized_stress_3d(*args, **kwargs)


__all__ = [
    "collide_cg_regularized_stress_3d", "collide_cg_central_stress_3d",
    "collide_cg_cumulant_3d", "collide_cg_cascaded_3d", "collide_cg_kbc_3d",
]
