"""Inference helpers that inject the trained AI turbulence model into LBM.

Given a velocity field, predict a per-cell eddy viscosity ``ν_t`` with the
network and convert it into an effective relaxation time

    τ_eff(x) = τ_0 + 3 ν_t(x)

which is then used by a BGK collision step.  The interface mirrors
:func:`tensorlbm.turbulence.collide_smagorinsky_bgk` so a user can switch
between classical and AI-based LES closures with a one-line change.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..d2q9 import equilibrium, macroscopic
from .dataset import strain_rate_tensor_2d

if TYPE_CHECKING:
    from .model import EddyViscosityMLP


def predict_nu_t_2d(
    model: EddyViscosityMLP,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """Predict the eddy viscosity field for a 2-D velocity snapshot.

    Args:
        model: A trained :class:`EddyViscosityMLP`.
        ux, uy: Velocity fields of shape ``(ny, nx)``.

    Returns:
        A non-negative tensor of shape ``(ny, nx)``.
    """
    s_xx, s_yy, s_xy = strain_rate_tensor_2d(ux, uy)
    feats = torch.stack([s_xx, s_yy, s_xy], dim=-1)  # (ny, nx, 3)
    ny, nx, _ = feats.shape
    flat = feats.reshape(-1, 3)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(flat)
    if was_training:
        model.train()
    return out.reshape(ny, nx).clamp_min(0.0)


def predict_tau_eff_2d(
    model: EddyViscosityMLP,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute the per-cell effective relaxation time ``τ + 3 ν_t``.

    The returned tensor is clamped below at 0.5 + 1e-3 to keep the BGK
    operator stable even if the network briefly under-predicts.
    """
    nu_t = predict_nu_t_2d(model, ux, uy)
    return torch.clamp_min(float(tau) + 3.0 * nu_t, 0.5 + 1e-3)


def collide_ai_les_bgk(
    f: torch.Tensor,
    tau: float,
    model: EddyViscosityMLP,
) -> torch.Tensor:
    """D2Q9 BGK collision with a neural-network LES sub-grid closure.

    Drop-in replacement for
    :func:`tensorlbm.turbulence.collide_smagorinsky_bgk` that uses
    ``model`` to predict the local eddy viscosity instead of the
    Smagorinsky algebraic formula.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Molecular (baseline) relaxation time ``τ_0 > 0.5``.
        model: Trained :class:`EddyViscosityMLP`.

    Returns:
        Updated distribution tensor of the same shape as *f*.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq
    tau_eff = predict_tau_eff_2d(model, ux, uy, tau)  # (ny, nx)
    return f - f_neq / tau_eff.unsqueeze(0)
