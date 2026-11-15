r"""Allen-Cahn / CSF hydrodynamic *force chain* with a density-deviation pressure.

Why this module exists
----------------------
:mod:`tensorlbm.allen_cahn_lbm` already solves the Allen-Cahn (AC) transport
equation and already computes the Korteweg capillary force ``F = mu grad(phi)``
with the isotropic D3Q19 operators of
:mod:`tensorlbm.phasefield.isotropic_fd`.  XVAL round 1
(``docs/XVAL_M17410_ALLEN_CAHN_CSF_FORCE_CHAIN.md``) proved that this *force
evaluation* is bit-identical to ``lbmpy``'s; it explicitly deferred the
**dynamic** part of the chain - how the force enters the hydrodynamic step - and
left ``allen_cahn_step`` WITHHELD because that step diverges.

This module closes that gap.  Two things are missing in the old stepper.

1. **The pressure is pinned to ``p = cs^2 rho(phi)``.**
   The old stepper builds ``f_eq`` with ``rho = rho_L + phi (rho_H - rho_L)``,
   so the zeroth moment of ``f`` is re-imposed every step.  A capillary
   equilibrium needs the pressure to *deviate* from the bulk equation of state
   by the Young-Laplace jump ``sigma/R``; a pinned density forbids that, so the
   Korteweg force is never balanced and the fluid accumulates momentum linearly
   (measured: ``max|u|`` grows ``8.3e-2`` per step on a 10:1 droplet and the run
   is non-finite by step 14).  The fix is the *density-deviation*
   (pressure-evolution) formulation of lbmpy's chain: ``f`` carries
   ``rho_dev = sum_i f_i`` (i.e. ``p/cs^2`` up to a constant), and
   ``f_i^eq = w_i [ rho_dev + rho_0 (3 c.u + (9/2)(c.u)^2 - (3/2) u^2) ]`` with
   the constant reference density ``rho_0 = 1``.

2. **Two of the three force terms of the lbmpy chain are missing.**
   ``lbmpy.phasefield_allen_cahn.kernel_equations.hydrodynamic_force`` returns

   .. math::

       F = \underbrace{\mu \nabla\phi}_{F_{st}}
         + \underbrace{-\tfrac{1}{3}\rho_{dev}\,\Delta\rho\,\nabla\phi}_{F_p}
         + \underbrace{\left(\tfrac{1}{2}-\tau\right)\Delta\rho\,
                       (\Pi^{neq}\cdot\nabla\phi)}_{F_v}
         + F_{body},

   with ``Delta rho = rho_H - rho_L`` and
   ``Pi^{neq}_{ab} = sum_i c_ia c_ib (f_i - f_i^eq)``.  ``F_p`` carries the
   pressure / density coupling and ``F_v`` the variable-viscosity cross term;
   :data:`tensorlbm.phasefield.free_energy.force_mu_grad_phi` implements only
   ``F_st``.  Which of the two extra terms is *necessary* on a static droplet is
   exactly what the ablation in
   ``scripts/xval_ac_csf_force_chain_m22329.py`` measures.

Provenance and fidelity
-----------------------
* Reference: ``lbmpy 2.0``, ``lbmpy/phasefield_allen_cahn/kernel_equations.py``
  (``pressure_force``, ``viscous_force``, ``surface_tension_force``,
  ``hydrodynamic_force_assignments`` - the sub-iterated velocity shift), plus
  ``lbmpy/forcemodels.py:Guo``.
* Everything here is written against the *published formulas* above; nothing is
  copied.  The mapping between the two double-well conventions
  (``phi in [-1,1]`` here, ``phi in [0,1]`` in lbmpy) is the one already
  established in XVAL round 1: ``W_here = W_lbmpy / 2``.
* The operators are the isotropic D3Q19 pair from
  :mod:`tensorlbm.phasefield.isotropic_fd`; round 1 measured them bit-identical
  to lbmpy's, so operator error cannot be confused with chain error here.

Honest status
-------------
REAL: the force terms, the density-deviation equilibrium and the coupled stepper
are exercised, and graded against the analytic Young-Laplace law, by
``scripts/xval_ac_csf_force_chain_m22329.py`` and
``tests/test_ac_csf_force_chain_m22329.py``.  Where a claim is only bounded
rather than accurate, the test asserts the *bound*, not a wished-for value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import torch

from ..d3q19 import C as _C3D
from ..d3q19 import W as _W3D
from .isotropic_fd import isotropic_gradient_3d, isotropic_laplacian_3d

__all__ = [
    "ForceChainConfig",
    "ac_csf_step",
    "chemical_potential_isotropic",
    "density_from_phase_field",
    "hydrodynamic_force_chain",
    "laplace_pressure_jump_2phase",
    "pressure_force",
    "relaxation_time_from_phase_field",
    "second_moment_neq",
    "surface_tension_force",
    "viscous_force",
]

Boundary = Literal["periodic", "no_flux"]

_CS2 = 1.0 / 3.0
_CHAIN_TERMS = ("st", "p", "v")


@dataclass(frozen=True)
class ForceChainConfig:
    """Which pieces of the lbmpy Allen-Cahn/CSF chain are switched on.

    Attributes:
        terms: subset of ``("st", "p", "v")`` - surface tension, pressure,
            viscous.  ``("st",)`` reproduces the pre-existing TensorLBM chain.
        formulation: ``"density_deviation"`` (lbmpy: ``f`` stores ``p/cs^2`` and
            the pressure is free) or ``"full_density"`` (legacy: ``f`` stores
            the phase-field density, ``p = cs^2 rho(phi)`` is pinned).
        sub_iterations: velocity-shift sub-iterations for the viscous force,
            matching ``hydrodynamic_force_assignments(..., sub_iterations=2)``.
        bc: phase-field boundary policy for the isotropic operators.
        momentum_density: normalisation of the velocity recovered from the
            populations, ``u = (sum_i c_i f_i + F/2) / rho_mom``.
            ``"reference"`` uses the constant incompressible reference density
            ``rho_0 = 1`` (pressure-evolution form: the *variable* inertia is
            carried by ``F_p``/``F_v`` instead); ``"physical"`` uses the local
            mixture density ``rho(phi)``.  ``"physical"`` was measured to be
            linearly unstable on a 10:1 droplet (non-finite by step 10-13) - it
            is kept only so the ablation can report it honestly.
    """

    terms: tuple[str, ...] = ("st", "p", "v")
    formulation: Literal["density_deviation", "full_density"] = "density_deviation"
    sub_iterations: int = 2
    bc: Boundary = "periodic"
    momentum_density: Literal["reference", "physical"] = "reference"

    def __post_init__(self) -> None:
        unknown = [t for t in self.terms if t not in _CHAIN_TERMS]
        if unknown:
            raise ValueError(f"unknown force chain term(s) {unknown}; allowed {_CHAIN_TERMS}")
        if self.sub_iterations < 1:
            raise ValueError("sub_iterations must be >= 1")
        if self.formulation not in ("density_deviation", "full_density"):
            raise ValueError(f"unknown formulation {self.formulation!r}")
        if self.momentum_density not in ("reference", "physical"):
            raise ValueError(f"unknown momentum_density {self.momentum_density!r}")


# --------------------------------------------------------------------------- #
# phase-field material fields
# --------------------------------------------------------------------------- #
def density_from_phase_field(
    phi: torch.Tensor, rho_h: float, rho_l: float
) -> torch.Tensor:
    """Mixture density ``rho = rho_L + phi (rho_H - rho_L)`` for ``phi in [-1,1]``.

    This is the ``phi in [-1,1]`` image of lbmpy's
    ``rho_L + phi (rho_H - rho_L)`` with ``phi in [0,1]``; the two agree once the
    order parameter is mapped, and the mapping keeps ``rho(+1) = rho_H``.
    """
    return rho_l + (rho_h - rho_l) * (1.0 + phi) / 2.0


def relaxation_time_from_phase_field(
    phi: torch.Tensor, nu_h: float, nu_l: float
) -> torch.Tensor:
    """``tau = 1/2 + 3 nu(phi)`` with ``nu`` interpolated on the phase field.

    lbmpy writes the same rule as ``tau = 1/2 + tau_L + phi (tau_H - tau_L)``
    with ``tau`` the *kinematic viscosity*; here ``tau`` is the relaxation time,
    hence the ``3 nu`` (``cs^2 = 1/3``).
    """
    nu = nu_l + (nu_h - nu_l) * (1.0 + phi) / 2.0
    return 0.5 + 3.0 * nu


def chemical_potential_isotropic(
    phi: torch.Tensor,
    beta: float,
    kappa: float,
    *,
    bc: Boundary = "periodic",
) -> torch.Tensor:
    """``mu = beta phi (phi^2 - 1) - kappa lap(phi)`` with the isotropic Laplacian."""
    return beta * phi * (phi * phi - 1.0) - kappa * isotropic_laplacian_3d(phi, boundary=bc)


# --------------------------------------------------------------------------- #
# the three force terms
# --------------------------------------------------------------------------- #
def surface_tension_force(
    phi: torch.Tensor, mu: torch.Tensor, *, bc: Boundary = "periodic"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``F_st = mu grad(phi)`` (Korteweg form; lbmpy ``surface_tension_force``)."""
    gx, gy, gz = isotropic_gradient_3d(phi, boundary=bc)
    return mu * gx, mu * gy, mu * gz


def pressure_force(
    phi: torch.Tensor,
    rho_dev: torch.Tensor,
    rho_h: float,
    rho_l: float,
    *,
    bc: Boundary = "periodic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``F_p = -(1/3) rho_dev (rho_H - rho_L) grad(phi)`` (lbmpy ``pressure_force``)."""
    factor = -(1.0 / 3.0) * rho_dev * (rho_h - rho_l)
    gx, gy, gz = isotropic_gradient_3d(phi, boundary=bc)
    return factor * gx, factor * gy, factor * gz


def second_moment_neq(neq: torch.Tensor) -> torch.Tensor:
    """``Pi^neq_ab = sum_i c_ia c_ib neq_i`` for a ``(19, nz, ny, nx)`` field.

    Returns a ``(3, 3, nz, ny, nx)`` tensor.
    """
    if neq.ndim != 4 or neq.shape[0] != 19:
        raise ValueError("neq must have shape (19, nz, ny, nx)")
    c = _C3D.to(device=neq.device, dtype=neq.dtype)
    out = torch.empty((3, 3) + tuple(neq.shape[1:]), device=neq.device, dtype=neq.dtype)
    for a in range(3):
        for b in range(3):
            w_ab = c[:, a] * c[:, b]
            out[a, b] = (w_ab.view(19, 1, 1, 1) * neq).sum(dim=0)
    return out


def viscous_force(
    phi: torch.Tensor,
    pis_neq: torch.Tensor,
    tau: torch.Tensor,
    rho_h: float,
    rho_l: float,
    *,
    bc: Boundary = "periodic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``F_v = (1/2 - tau) (rho_H - rho_L) (Pi^neq . grad(phi))``.

    ``Pi^neq . grad(phi)`` is the contraction ``sum_b Pi^neq_ab d_b phi``; the
    ``(1/2 - tau)`` prefactor is lbmpy's ``(half - tau)`` and is *negative* for
    ``tau > 1/2``.
    """
    grad = isotropic_gradient_3d(phi, boundary=bc)
    d_rho = rho_h - rho_l
    factor = (0.5 - tau) * d_rho
    comps = []
    for a in range(3):
        contracted = (
            pis_neq[a, 0] * grad[0] + pis_neq[a, 1] * grad[1] + pis_neq[a, 2] * grad[2]
        )
        comps.append(factor * contracted)
    return comps[0], comps[1], comps[2]


# --------------------------------------------------------------------------- #
# chain assembly
# --------------------------------------------------------------------------- #
def hydrodynamic_force_chain(
    phi: torch.Tensor,
    f_pre: torch.Tensor,
    f_eq: torch.Tensor,
    *,
    beta: float,
    kappa: float,
    rho_h: float,
    rho_l: float,
    tau: torch.Tensor,
    rho_dev: torch.Tensor | None = None,
    config: ForceChainConfig = ForceChainConfig(),
    body_force: Sequence[torch.Tensor] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    """Assemble ``F = F_st + F_p + F_v + F_body`` and return it with its parts.

    Args:
        phi: phase field, ``(z, y, x)``.
        f_pre: pre-collision populations, used for ``Pi^neq``.
        f_eq: equilibrium of ``f_pre`` (same shape), used for ``Pi^neq``.
        beta, kappa: double-well coefficients (see
            :func:`tensorlbm.allen_cahn_lbm.free_energy_parameters`).
        rho_h, rho_l: bulk densities.
        tau: relaxation-time field.
        rho_dev: density deviation used by ``F_p``; for the
            ``"density_deviation"`` formulation that is ``sum_i f_i``, for
            ``"full_density"`` it is ``sum_i f_i`` as well (the physical
            pressure deviation is then ``cs^2 rho_dev`` up to a constant).
        config: which terms are active.
        body_force: optional extra body force triple (e.g. ``rho g``).

    Returns:
        ``((Fx, Fy, Fz), parts)`` where ``parts`` holds each term and ``mu``.
    """
    mu = chemical_potential_isotropic(phi, beta, kappa, bc=config.bc)
    parts: dict[str, torch.Tensor] = {"mu": mu}
    total = [torch.zeros_like(phi) for _ in range(3)]

    st = surface_tension_force(phi, mu, bc=config.bc)
    parts["st"] = st[0]
    if "st" in config.terms:
        total = [total[i] + st[i] for i in range(3)]

    if "p" in config.terms:
        if rho_dev is None:
            raise ValueError("the pressure force term needs rho_dev")
        fp = pressure_force(phi, rho_dev, rho_h, rho_l, bc=config.bc)
        parts["p"] = fp[0]
        total = [total[i] + fp[i] for i in range(3)]

    if "v" in config.terms:
        neq = f_pre - f_eq
        pis = second_moment_neq(neq)
        fv = viscous_force(phi, pis, tau, rho_h, rho_l, bc=config.bc)
        parts["v"] = fv[0]
        total = [total[i] + fv[i] for i in range(3)]

    if body_force is not None:
        total = [total[i] + body_force[i] for i in range(3)]

    return (total[0], total[1], total[2]), parts


def _equilibrium_density_deviation(
    rho_dev: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Incompressible (density-deviation) D3Q19 equilibrium.

    ``f_i^eq = w_i [ rho_dev + rho_0 (3 c.u + (9/2)(c.u)^2 - (3/2) u^2) ]`` with
    the constant reference density ``rho_0 = 1``.  The first moment is therefore
    ``rho_0 u`` and the pressure is carried by ``rho_dev`` alone, so the
    Young-Laplace jump no longer has to be produced by a density jump.
    """
    if device is None:
        device = rho_dev.device
    if dtype is None:
        dtype = rho_dev.dtype
    c = _C3D.to(device=device, dtype=dtype)
    w = _W3D.to(device=device, dtype=dtype)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * uy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * uz.unsqueeze(0)
    )
    u_sq = ux * ux + uy * uy + uz * uz
    return w.view(19, 1, 1, 1) * (
        rho_dev.unsqueeze(0) + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0)
    )


def _equilibrium_full_density(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Standard compressible D3Q19 equilibrium ``f_i^eq = w_i rho (1 + ...)``."""
    if device is None:
        device = rho.device
    if dtype is None:
        dtype = rho.dtype
    c = _C3D.to(device=device, dtype=dtype)
    w = _W3D.to(device=device, dtype=dtype)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * uy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * uz.unsqueeze(0)
    )
    u_sq = ux * ux + uy * uy + uz * uz
    return w.view(19, 1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0)
    )


def _stream(f_post: torch.Tensor) -> torch.Tensor:
    """Pull-scheme streaming on the ``(z, y, x)`` layout."""
    c = _C3D.to(device=f_post.device, dtype=torch.int64)
    return torch.stack(
        [
            torch.roll(f_post[q], shifts=(int(c[q, 2]), int(c[q, 1]), int(c[q, 0])), dims=(0, 1, 2))
            for q in range(19)
        ]
    )


def _guo_source(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Guo source ``S_i = w_i (1 - 1/(2 tau)) [3(c.F - u.F) + 9 (c.u)(c.F)]``."""
    c = _C3D.to(device=ux.device, dtype=ux.dtype)
    w = _W3D.to(device=ux.device, dtype=ux.dtype).view(19, 1, 1, 1)
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * uy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * uz.unsqueeze(0)
    )
    cf = (
        c[:, 0].view(19, 1, 1, 1) * fx.unsqueeze(0)
        + c[:, 1].view(19, 1, 1, 1) * fy.unsqueeze(0)
        + c[:, 2].view(19, 1, 1, 1) * fz.unsqueeze(0)
    )
    uf = ux * fx + uy * fy + uz * fz
    return w * (1.0 - 0.5 / tau).unsqueeze(0) * (3.0 * (cf - uf.unsqueeze(0)) + 9.0 * cu * cf)


def ac_csf_step(
    f: torch.Tensor,
    phi: torch.Tensor,
    *,
    rho_h: float = 1.0,
    rho_l: float = 0.1,
    nu_h: float = 0.1,
    nu_l: float = 0.02,
    sigma: float = 1e-2,
    W: float = 4.0,
    mobility: float = 5e-3,
    config: ForceChainConfig = ForceChainConfig(),
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    advance_phase_field: bool = True,
    beta: float | None = None,
    kappa: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One coupled Allen-Cahn / CSF timestep with the full lbmpy force chain.

    The step is, in order: chemical potential and ``F_st``/``F_p``/``F_v`` ->
    sub-iterated Guo velocity shift ``u_eq = u + F/(2 rho(phi))`` -> Guo-forced
    BGK collision -> pull streaming -> Allen-Cahn update advected by ``u_eq``.

    Args:
        f: hydrodynamic populations ``(19, nz, ny, nx)``.
        phi: phase field ``(z, y, x)``, ``phi = +1`` heavy.
        rho_h, rho_l, nu_h, nu_l: bulk densities and kinematic viscosities.
        sigma, W: surface tension and interface thickness.
        mobility: Allen-Cahn mobility ``M``.
        config: force chain / formulation switch (see :class:`ForceChainConfig`).
        gravity: body acceleration triple (multiplied by ``rho(phi)``).
        advance_phase_field: ``False`` freezes ``phi`` (hydrodynamics-only probe;
            used by the ablation to isolate the chain from AC transport).
        beta, kappa: optional explicit double-well coefficients; when both are
            given they override the ``(sigma, W)`` mapping, which lets the
            ablation use lbmpy's own ``(beta, kappa)`` for the same interface.

    Returns:
        ``(f_new, phi_new, diag)``; ``diag`` holds ``u``, ``rho``, ``rho_dev``,
        the chain parts and ``max|F|``.
    """
    if f.ndim != 4 or f.shape[0] != 19:
        raise ValueError("f must have shape (19, nz, ny, nx)")
    if phi.shape != f.shape[1:]:
        raise ValueError("phi must have shape (nz, ny, nx) matching f")
    if rho_l <= 0.0 or rho_h <= 0.0:
        raise ValueError("densities must be positive")

    dtype = f.dtype
    device = f.device
    phi = phi.to(dtype=dtype, device=device)

    if beta is None or kappa is None:
        from ..allen_cahn_lbm import free_energy_parameters

        params = free_energy_parameters(sigma, W)
        beta = params.beta if beta is None else beta
        kappa = params.kappa if kappa is None else kappa

    rho = density_from_phase_field(phi, rho_h, rho_l)
    tau = relaxation_time_from_phase_field(phi, nu_h, nu_l)
    rho_safe = rho.clamp(min=1e-12)

    # --- moments of the incoming populations -------------------------------- #
    c = _C3D.to(device=device, dtype=dtype)
    rho_sum = f.sum(dim=0)
    mom = [
        (f * c[:, a].view(19, 1, 1, 1)).sum(dim=0) for a in range(3)
    ]
    rho_dev = rho_sum
    if config.momentum_density == "reference":
        # Pressure-evolution bookkeeping: the populations transport momentum
        # per unit *reference* density rho_0 = 1 (== rho_sum in the deviation
        # formulation), so the velocity is sum(c f) / rho_0 + F/(2 rho_0).
        rho_mom = (
            torch.ones_like(rho)
            if config.formulation == "density_deviation"
            else rho_sum.clamp(min=1e-12)
        )
    else:
        rho_mom = rho_safe
    u_raw = [mom[a] / rho_mom for a in range(3)]

    body = None
    if any(g != 0.0 for g in gravity):
        body = (rho * gravity[0], rho * gravity[1], rho * gravity[2])

    # --- chain: F depends on u only through F_v, so sub-iterate the shift --- #
    u_eq = list(u_raw)
    parts: dict[str, torch.Tensor] = {}
    force = (torch.zeros_like(phi), torch.zeros_like(phi), torch.zeros_like(phi))
    for _ in range(config.sub_iterations):
        if config.formulation == "density_deviation":
            f_eq = _equilibrium_density_deviation(rho_dev, u_eq[0], u_eq[1], u_eq[2])
        else:
            f_eq = _equilibrium_full_density(rho, u_eq[0], u_eq[1], u_eq[2])
        force, parts = hydrodynamic_force_chain(
            phi,
            f,
            f_eq,
            beta=beta,
            kappa=kappa,
            rho_h=rho_h,
            rho_l=rho_l,
            tau=tau,
            rho_dev=rho_dev,
            config=config,
            body_force=body,
        )
        u_eq = [u_raw[a] + 0.5 * force[a] / rho_mom for a in range(3)]

    if config.formulation == "density_deviation":
        f_eq = _equilibrium_density_deviation(rho_dev, u_eq[0], u_eq[1], u_eq[2])
    else:
        f_eq = _equilibrium_full_density(rho, u_eq[0], u_eq[1], u_eq[2])

    f_post = f - (f - f_eq) / tau.unsqueeze(0) + _guo_source(
        u_eq[0], u_eq[1], u_eq[2], force[0], force[1], force[2], tau
    )
    f_new = _stream(f_post)

    diag: dict[str, torch.Tensor] = {
        "u": torch.stack(u_eq),
        "rho": rho,
        "rho_dev": rho_dev,
        "force": torch.stack(force),
        "max_abs_force": max(float(x.abs().max()) for x in force),
        "max_abs_u": max(float(x.abs().max()) for x in u_eq),
        "max_abs_phi": float(phi.abs().max()),
    }
    for key, value in parts.items():
        diag[f"part_{key}"] = value
    diag["tau_range"] = torch.tensor([float(tau.min()), float(tau.max())])

    phi_new = phi
    if advance_phase_field:
        from ..allen_cahn_lbm import step_allen_cahn_field

        phi_new = step_allen_cahn_field(
            phi,
            beta=beta,
            kappa=kappa,
            mobility=mobility,
            u=(u_eq[0], u_eq[1], u_eq[2]),
            boundary=config.bc,
        )
    return f_new, phi_new, diag


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
def laplace_pressure_jump_2phase(
    f: torch.Tensor | None,
    phi: torch.Tensor,
    *,
    radius: float,
    sigma: float,
    rho_h: float,
    rho_l: float,
    formulation: Literal["density_deviation", "full_density"] = "density_deviation",
    rho_dev: torch.Tensor | None = None,
    geometry: Literal["sphere", "cylinder"] = "sphere",
) -> dict[str, float]:
    """Young-Laplace ``Delta p`` of a static droplet, valid at any density ratio.

    The pre-existing :func:`tensorlbm.allen_cahn_lbm.laplace_pressure_jump` uses
    ``p = rho cs^2`` with ``rho`` the *total* density, which at ``rho_H != rho_L``
    reports the bulk density difference ``(rho_H - rho_L)/3`` (e.g. ``0.3`` at
    10:1) as if it were the capillary jump, i.e. a ``4e4 %`` error that has
    nothing to do with the force chain.  The pressure that must satisfy
    ``Delta p = sigma/R`` is the *deviation* from each phase's own equation of
    state:

    * ``"density_deviation"``: ``p = cs^2 rho_dev``, so the jump is
      ``(rho_dev_in - rho_dev_out)/3``;
    * ``"full_density"``: ``p = cs^2 rho``, so the jump is
      ``[(rho_in - rho_H) - (rho_out - rho_L)]/3``.

    The reference value is ``2 sigma/R`` for a sphere (two equal principal
    curvatures) and ``sigma/R`` for a cylinder/2-D droplet; the pre-existing
    helper used ``sigma/R`` unconditionally, which is a 2x error on the very
    geometry it builds (``initialize_allen_cahn``/the XVAL droplet are 3-D).

    Args:
        f: populations (needed when ``rho_dev`` is not supplied).
        phi: phase field used to build the bulk masks.
        radius: nominal droplet radius (the analytic ``sigma/R`` scale).
        sigma: surface tension.
        rho_h, rho_l: bulk densities.
        formulation: which hydro formulation produced the state.
        rho_dev: optional precomputed density deviation.

    Returns:
        Dict with ``dp_measured``, ``dp_laplace``, ``relative_error``,
        ``p_in``, ``p_out`` and the mask cell counts.
    """
    if rho_dev is None:
        if f is None:
            raise ValueError("either f or rho_dev must be given")
        rho_dev = f.sum(dim=0)
    inside = phi > 0.9
    outside = phi < -0.9
    if not bool(inside.any()) or not bool(outside.any()):
        raise ValueError("droplet and bulk masks must both be non-empty")

    if formulation == "density_deviation":
        p_in = float(rho_dev[inside].mean()) * _CS2
        p_out = float(rho_dev[outside].mean()) * _CS2
    else:
        rho = rho_dev
        p_in = (float(rho[inside].mean()) - rho_h) * _CS2
        p_out = (float(rho[outside].mean()) - rho_l) * _CS2

    dp = p_in - p_out
    curvature = 2.0 / radius if geometry == "sphere" else 1.0 / radius
    expected = sigma * curvature
    return {
        "p_in": p_in,
        "p_out": p_out,
        "dp_measured": dp,
        "dp_laplace": expected,
        "relative_error": abs(dp - expected) / expected,
        "curvature_coefficient": curvature,
        "n_inside": int(inside.sum()),
        "n_outside": int(outside.sum()),
    }