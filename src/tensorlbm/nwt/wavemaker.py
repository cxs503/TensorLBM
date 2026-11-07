"""Wave generation / absorption machinery for the numerical wave tank (NWT).

Two ingredients, both solver-agnostic (they only touch the free-surface state
carried by :mod:`tensorlbm.free_surface_lbm_2d` — the independent mass ledger
``mass``, the population ``f`` and the cell-type ``flags``):

1. **Relaxation-zone wavemaker** (Mayer et al. 1998; Jacobsen et al. 2012,
   "Dirichlet-type" relaxation).  Inside the generation zone the analytic
   second-order Stokes solution is blended into the numerical solution with a
   smooth weight ``gamma(x)`` that is 1 at the offshore boundary and 0 at the
   end of the zone.  Two quantities are relaxed:

   * the liquid column inventory -> enforces the analytic elevation ``eta``;
   * the populations -> toward local equilibrium at the analytic velocity.

2. **Relaxation-zone absorber** (sponge).  Same blending, target = still water
   (``eta = 0``, ``u = 0``).  Used at the leeward end to suppress re-reflection
   of the outgoing wave train.

The column inventory used here is the *conserved* quantity of the Körner
free-surface model,

.. math::

    V(x) = \\frac{1}{\\rho_l}\\Big[\\sum_{y\\in LIQUID}\\rho(y,x)
            + \\sum_{y\\in INTERFACE} m(y,x)\\Big],

so ``eta(x) = V(x) - h`` is a boundary-consistent free-surface elevation
estimate (no geometric reconstruction, no adjustable threshold).
"""

from __future__ import annotations

import math

import torch

from ..d2q9 import equilibrium
from ..d2q9 import C as _C_LATTICE
from ..free_surface_lbm_2d import GAS, INTERFACE, LIQUID, SOLID

_CX = _C_LATTICE[:, 0].float().view(9, 1, 1)
_CY = _C_LATTICE[:, 1].float().view(9, 1, 1)

__all__ = [
    "relaxation_weight",
    "column_free_surface",
    "relax_column_inventory",
    "relax_velocity_to_target",
    "damp_velocity",
    "ramp_factor",
]


def ramp_factor(t: float, t_ramp: float) -> float:
    """C1-smooth start-up ramp in ``[0, 1]``: 0 at ``t=0``, 1 at ``t >= t_ramp``."""
    if t_ramp <= 0.0:
        return 1.0
    if t <= 0.0:
        return 0.0
    if t >= t_ramp:
        return 1.0
    u = t / t_ramp
    return 0.5 * (1.0 - math.cos(math.pi * u))


def relaxation_weight(x: torch.Tensor, x0: float, x1: float) -> torch.Tensor:
    """Smooth blending weight, 1 at ``x0`` and 0 at ``x1`` (cosine, C1).

    ``x0``/``x1`` may be given in either order (the profile is symmetric), which
    lets the same helper build both the offshore generator (1 at the wall) and
    the leeward absorber (1 at the wall, 0 at the sponge entrance).
    """
    length = x1 - x0
    if length == 0:
        return torch.zeros_like(x)
    s = ((x - x0) / length).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * s))


def column_free_surface(
    f: torch.Tensor,
    mass: torch.Tensor,
    flags: torch.Tensor,
    rho_liquid: float = 1.0,
) -> torch.Tensor:
    """Per-column free-surface elevation estimator ``V(x) = eta(x) + h``.

    ``V`` is the **geometric** liquid column height in lattice cells,

    .. math::

        V(x) = N_{LIQUID}(x) + \sum_{y\in INTERFACE} \frac{m(y,x)}{\rho_{local}},

    i.e. one cell per bulk-liquid cell plus the tracked fill of the interface
    cells (exactly the model's own fill definition).  Using the *geometric* fill rather than the conserved compressible
    inventory (``sum rho`` over LIQUID) matters: the hydrostatic density
    stratification of the LBM column is ``g h / c_s^2`` (a few percent here), so
    a density-weighted column average carries a constant offset *and* a wave
    pressure contamination of order ``g eta / c_s^2`` per cell.  The geometric
    measure is free of both and is the quantity the analytic ``eta`` refers to.

    The conserved (compressible) inventory remains the right mass-conservation
    diagnostic and is reported separately by the driver.
    """
    n_liquid = (flags == LIQUID).sum(dim=0).to(mass.dtype)
    # the model defines the interface fill as ``mass / rho_local`` (see
    # ``free_surface_step_2d``): using rho_local instead of rho_liquid removes the
    # hydrostatic-compressibility bias of the interface band from eta
    rho = f.sum(dim=0).clamp(min=1e-6)
    iface_fill = torch.where(flags == INTERFACE, mass / rho, torch.zeros_like(mass)).sum(dim=0)
    return n_liquid + iface_fill


def relax_column_inventory(
    mass: torch.Tensor,
    flags: torch.Tensor,
    target_inventory: torch.Tensor,
    weight: torch.Tensor,
    current_inventory: torch.Tensor | None = None,
    rho_liquid: float = 1.0,
    f: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend each column's liquid inventory toward ``target_inventory``.

    ``Delta V = weight * (V_target - V_current)`` is credited to the *topmost
    INTERFACE cell* of the column — the only cell whose tracked mass is under
    the control of the mass ledger (LIQUID cells are governed by their
    populations, GAS cells carry no mass).  A per-step addition is capped at
    ``rho_l - m`` so that the cell converts to LIQUID by the regular Körner
    I->L rule instead of provoking an excess-mass redistribution; any deferred
    remainder is applied on the following steps.  Removal is allowed to drive the
    fill slightly negative (bounded by ``-0.5``), which is exactly the state the
    model needs to convert the cell to GAS; the LIQUID below is then demoted to
    INTERFACE by the model's interface envelope, so the surface can continue to
    fall on subsequent steps.

    Returns the updated ``mass`` field (the input tensor is not modified).
    """
    ny, nx = mass.shape
    if f is None:
        raise ValueError("f is required for the current column inventory")
    if current_inventory is None:
        current_inventory = column_free_surface(f, mass, flags, rho_liquid)
    y_idx = torch.arange(ny, device=mass.device).view(ny, 1).expand(ny, nx)
    iface = flags == INTERFACE
    top_y = torch.where(iface, y_idx, torch.full_like(y_idx, -1)).max(dim=0).values
    valid = top_y >= 0
    idx = top_y.clamp(min=0).view(1, nx)
    cur = mass.gather(0, idx).squeeze(0)
    delta = torch.where(valid, weight * (target_inventory - current_inventory), torch.zeros_like(cur))
    room_up = (rho_liquid - cur).clamp(min=0.0)
    # Additions stop at the full-fill line so the regular I->L rule converts the
    # cell without an excess-mass redistribution.  Removals are allowed to go
    # *below* zero: the model triggers I->G only for ``fill < -kappa`` (its own
    # drainage mechanism) and redistributes the negative excess conservatively —
    # clamping the removal at m >= 0 would block the surface from going down at
    # all and turn the wavemaker into a mass pump.
    step = torch.where(
        delta > 0.0,
        torch.minimum(delta, room_up),
        torch.maximum(delta, torch.full_like(delta, -0.5)),
    )
    new = cur + step
    new = torch.where(valid, new, cur)
    out = mass.clone()
    out.scatter_(0, idx, new.view(1, nx))
    return out


def relax_velocity_to_target(
    f: torch.Tensor,
    flags: torch.Tensor,
    ux_target: torch.Tensor,
    uy_target: torch.Tensor,
    weight: torch.Tensor,
    rho_liquid: float = 1.0,
) -> torch.Tensor:
    """Blend fluid populations toward equilibrium at the analytic velocity.

    ``f <- (1 - gamma) f + gamma f_eq(rho, u_target)`` on LIQUID/INTERFACE
    cells, with the *local* density ``rho = sum_q f_q`` retained so the blend
    stays compressibility-consistent and introduces no density source.  GAS and
    SOLID cells are untouched.
    """
    fluid = (flags == LIQUID) | (flags == INTERFACE)
    rho = f.sum(dim=0).clamp(min=1e-6)
    feq = equilibrium(rho, ux_target, uy_target)
    w = (weight.view(1, *weight.shape) * fluid.view(1, *fluid.shape)).clamp(0.0, 1.0)
    return (1.0 - w) * f + w * feq

def damp_velocity(
    f: torch.Tensor,
    flags: torch.Tensor,
    weight: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Viscous sponge: shrink the non-equilibrium part of the populations.

    ``f <- f_eq + (1 - s(x)) (f - f_eq)`` on LIQUID/INTERFACE cells with
    ``s = strength * weight``.  The density of every cell is untouched, so this
    is a pure momentum sink — the standard way to build a dissipative sponge
    without a mass source/sink at the boundary.  ``weight`` is the relaxation
    profile and ``strength`` (<= ~0.1) the maximum damping per time step.
    """
    fluid = (flags == LIQUID) | (flags == INTERFACE)
    rho = f.sum(dim=0).clamp(min=1e-6)
    ux = (f * _CX).sum(dim=0) / rho
    uy = (f * _CY).sum(dim=0) / rho
    feq = equilibrium(rho, ux, uy)
    factor = (1.0 - strength * weight).clamp(0.0, 1.0)
    factor = torch.where(fluid, factor, torch.ones_like(factor))
    return feq + factor.view(1, *factor.shape) * (f - feq)
