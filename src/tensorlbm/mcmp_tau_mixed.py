"""Tau-mixed (viscosity-weighted) MCMP velocity: REF-OSS formula + analytic limits.

Purpose
-------
This library's two-component Shan-Chen collision
(:func:`tensorlbm.multiphase.collide_sc_two_component`) offers
``u_eq="viscous"``, which relaxes both components towards the
**viscosity-weighted** common velocity

    u_visc = Σ_k (ρ_k u_k / τ_k) / Σ_k (ρ_k / τ_k) .

The same construction is used by external LBM implementations, so the mode is
cross-validated here against a concrete OSS reference.  The module supplies
(a) the reference formula as an independent, explicitly-transcribed function,
(b) the continuum limit of that formula (the mixture viscosity a solver must
reproduce), and (c) a black-box probe that recovers the mixed velocity the
library collision *actually* used, so parity with the reference can be measured
rather than assumed.  Nothing here modifies the solver.

REF-OSS provenance (no hand transcription of numbers)
-----------------------------------------------------
Platform ``oss_solutions`` entry id=35 (evidence: report 9555), repo
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``, feature
``GAP-LBM-MCMP-POISEUILLE``, tag ``shan-chen-mcmp-tau-mixed-velocity``:

    ``vel_mix = Σ(Vel*Rho/tau)/Σ(Rho/tau)``;  source lines ``Domain.py:70-88``,
    ``Domain.py:143-176`` (the mixing itself at ``Domain.py:151-153``); each
    component's τ taken from the per-component viscosity list
    (``Domain.py:77-78``).

The inventory additionally records the OSS draft's limitations, which this
module's docstrings keep as the *difference list* rather than borrowing:
``Rhoref=200.0``/``Psi=4.0`` are placeholder constants and the draft has **no
non-local Shan-Chen force**, hence no phase separation.

Continuum limit (used as the quantitative reference)
----------------------------------------------------
For the ideal (``G_12 = 0``) co-moving two-component mixture the mixed velocity
equals the barycentric velocity, and Chapman-Enskog on the τ-mixed equilibrium
gives a mixture kinematic viscosity that is the density-weighted **arithmetic**
mean of the component viscosities:

    ν_eff = Σ_k ρ_k ν_k / ρ ,   ν_k = c_s² (τ_k − ½),  ρ = Σ_k ρ_k .

(This is derived in the module docstring of the cross-validation report; the
discriminating power of the check comes from the alternative readings of the
same formula — ``ν(τ̄)`` or the harmonic mean — which differ by ≫ the 15 %
acceptance band for τ pairs away from equal.)

All functions are plain-tensor, dtype-preserving, and return tensors that can be
dumped straight into an artifact.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .d2q9 import macroscopic
from .multiphase import collide_sc_two_component

CS2 = 1.0 / 3.0

__all__ = [
    "CS2",
    "oss_vel_mix",
    "tau_mixed_velocity",
    "mixture_effective_viscosity",
    "mixture_effective_tau",
    "channel_poiseuille_profile",
    "collision_momentum_probe",
]


def oss_vel_mix(
    rho: Sequence[torch.Tensor] | torch.Tensor,
    vel: Sequence[torch.Tensor] | torch.Tensor,
    tau: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """REF-OSS viscosity-weighted common velocity ``Σ(Vel·Rho/τ)/Σ(Rho/τ)``.

    Args:
        rho: per-component densities, either a sequence of ``K`` tensors or a
            stacked tensor with the component axis first.
        vel: per-component velocities, same layout as ``rho``.
        tau: ``K`` relaxation times (one per component).

    Returns:
        Tensor of the same per-component shape holding ``vel_mix``.
    """
    rho_list = list(rho) if isinstance(rho, (list, tuple)) else list(rho.unbind(0))
    vel_list = list(vel) if isinstance(vel, (list, tuple)) else list(vel.unbind(0))
    if len(rho_list) != len(vel_list) or len(rho_list) != len(tau):
        raise ValueError("oss_vel_mix: rho/vel/tau must have the same component count")
    if len(rho_list) == 0:
        raise ValueError("oss_vel_mix: at least one component is required")
    w_sum = None
    num = None
    for r_k, u_k, t_k in zip(rho_list, vel_list, tau):
        if float(t_k) <= 0.0:
            raise ValueError(f"oss_vel_mix: tau must be positive, got {t_k}")
        w = r_k / float(t_k)
        contrib = w * u_k
        w_sum = w if w_sum is None else w_sum + w
        num = contrib if num is None else num + contrib
    if bool((w_sum <= 0).any()):
        raise ValueError("oss_vel_mix: total weight Σρ/τ must be positive everywhere")
    return num / w_sum


def tau_mixed_velocity(
    rho1: torch.Tensor,
    ux1: torch.Tensor,
    uy1: torch.Tensor,
    rho2: torch.Tensor,
    ux2: torch.Tensor,
    uy2: torch.Tensor,
    tau1: float,
    tau2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-component form of the REF-OSS mixture velocity (x, y components)."""
    if rho1.shape != rho2.shape or ux1.shape != ux2.shape or uy1.shape != uy2.shape:
        raise ValueError("tau_mixed_velocity: component shapes must match")
    if tau1 <= 0.0 or tau2 <= 0.0:
        raise ValueError("tau_mixed_velocity: tau must be positive")
    w1 = rho1 / tau1
    w2 = rho2 / tau2
    ux = (w1 * ux1 + w2 * ux2) / (w1 + w2)
    uy = (w1 * uy1 + w2 * uy2) / (w1 + w2)
    return ux, uy


def mixture_effective_viscosity(
    tau1: float, tau2: float, rho1: float = 1.0, rho2: float = 1.0
) -> float:
    """Density-weighted arithmetic mean of the component viscosities.

    ``ν_eff = (ρ1 ν1 + ρ2 ν2)/(ρ1 + ρ2)`` with ``ν_k = c_s²(τ_k − ½)`` — the
    continuum limit of the τ-mixed equilibrium (see module docstring).
    """
    if tau1 <= 0.5 or tau2 <= 0.5:
        raise ValueError("mixture_effective_viscosity: tau must exceed 1/2")
    if rho1 < 0 or rho2 < 0 or rho1 + rho2 <= 0:
        raise ValueError("mixture_effective_viscosity: densities must be non-negative")
    nu1 = CS2 * (tau1 - 0.5)
    nu2 = CS2 * (tau2 - 0.5)
    return (rho1 * nu1 + rho2 * nu2) / (rho1 + rho2)


def mixture_effective_tau(
    tau1: float, tau2: float, rho1: float = 1.0, rho2: float = 1.0
) -> float:
    """Equivalent single relaxation time implied by :func:`mixture_effective_viscosity`."""
    return 0.5 + mixture_effective_viscosity(tau1, tau2, rho1, rho2) / CS2


def channel_poiseuille_profile(
    ny: int,
    gx: float,
    nu: float,
    *,
    nx: int = 1,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Analytic plane-Poiseuille profile between two no-slip walls.

    Solid rows are ``y = 0`` and ``y = ny − 1``.  This library's post-stream
    bounce-back reflects inside the wall cells, which places the no-slip planes
    at the wall *node centres* ``y = 0`` and ``y = ny − 1`` (measured: quadratic
    fit of a driven channel gives wall offsets 0.004/0.003 lattice units at
    ny = 33/17, see the XVAL report), so ``H = ny − 1`` and

        u(y) = gx / (2 ν) · y · (H − y).

    Returns ``(ny, nx)`` (``x``-uniform), zero on the two solid rows.
    """
    if ny < 4:
        raise ValueError("channel_poiseuille_profile: ny must be >= 4")
    if nu <= 0.0:
        raise ValueError("channel_poiseuille_profile: nu must be positive")
    h = float(ny - 1)
    ys = torch.arange(ny, dtype=dtype, device=device).view(ny, 1)
    u = gx / (2.0 * nu) * ys * (h - ys)
    u = u.expand(ny, nx).clone()
    u[0, :] = 0.0
    u[-1, :] = 0.0
    return u


def collision_momentum_probe(
    f1: torch.Tensor,
    f2: torch.Tensor,
    tau1: float,
    tau2: float,
) -> dict[str, float]:
    """Recover the mixed velocity the library collision actually used.

    One force-free collision with ``G_12 = 0``, ``force_scheme="shift"`` and
    ``u_eq="viscous"`` changes each component's momentum by
    ``ΔJ_k = −(J_k − ρ_k u_visc)/τ_k``; inverting gives the effective
    ``u_visc`` the collision imposed.  Comparing that with
    :func:`oss_vel_mix` on the same pre-collision fields measures parity of the
    library path against the REF-OSS formula as a black box (round-off level).

    Returns a dict with the maximum absolute and maximum relative deviation over
    the two velocity components.
    """
    rho1, ux1, uy1 = macroscopic(f1)
    rho2, ux2, uy2 = macroscopic(f2)
    post1, post2 = collide_sc_two_component(
        f1, f2, G_12=0.0, tau1=tau1, tau2=tau2, u_eq="viscous", force_scheme="shift"
    )
    _, px1, py1 = macroscopic(post1)
    _, px2, py2 = macroscopic(post2)
    jx1, jy1 = rho1 * ux1, rho1 * uy1
    jx2, jy2 = rho2 * ux2, rho2 * uy2
    # ΔJ_k = J_k^post − J_k = −(J_k − ρ_k u_visc)/τ_k  ⇒  ρ_k u_visc = J_k + τ_k ΔJ_k
    d_x1, d_y1 = rho1 * px1 - jx1, rho1 * py1 - jy1
    d_x2, d_y2 = rho2 * px2 - jx2, rho2 * py2 - jy2
    ux_rec = (jx1 + tau1 * d_x1) / rho1
    uy_rec = (jy1 + tau1 * d_y1) / rho1
    ux_rec2 = (jx2 + tau2 * d_x2) / rho2
    uy_rec2 = (jy2 + tau2 * d_y2) / rho2
    ref_ux = oss_vel_mix([rho1, rho2], [ux1, ux2], [tau1, tau2])
    ref_uy = oss_vel_mix([rho1, rho2], [uy1, uy2], [tau1, tau2])
    scale = max(float(ref_ux.abs().max()), float(ref_uy.abs().max()), 1e-300)
    scale2 = max(float(ux_rec2.abs().max()), float(uy_rec2.abs().max()), 1e-300)
    return {
        "max_abs_dev": max(
            float((ux_rec - ref_ux).abs().max()),
            float((uy_rec - ref_uy).abs().max()),
            float((ux_rec2 - ref_ux).abs().max()),
            float((uy_rec2 - ref_uy).abs().max()),
        ),
        "max_rel_dev": max(
            float((ux_rec - ref_ux).abs().max()) / scale,
            float((uy_rec - ref_uy).abs().max()) / scale,
            float((ux_rec2 - ref_ux).abs().max()) / scale2,
            float((uy_rec2 - ref_uy).abs().max()) / scale2,
        ),
        "scale": scale,
    }