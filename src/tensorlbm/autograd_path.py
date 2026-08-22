"""Differentiable step chain for solver-in-the-loop autograd workflows.

`docs/differentiable_path.md` established that the eager operators
(``tensorlbm.solver3d``) are differentiable one by one.  This module turns
that property into a *usable composition contract*: a single autograd-clean
D3Q19 step — collision (fluid cells only) -> periodic gather streaming ->
full-way bounce-back on a solid mask — plus a rollout loop with optional
per-step gradient checkpointing and a differentiable momentum-exchange force
probe on the obstacle.

Everything is built from out-of-place torch ops that keep the autograd graph:

* collision is plain arithmetic (``f - (f - feq) / tau``), with ``tau``
  accepted as a graph-connected 0-dim tensor;
* streaming reuses the audited gather implementation ``solver3d.stream3d``
  (index tensors are cached constants);
* the solid is applied with ``torch.where`` selections only — no in-place
  boundary overwrites (the pattern documented as gradient-hostile in
  ``docs/differentiable_path.md``);
* the collision operator is a slot: the default is single-component BGK
  (:func:`tensorlbm.solver3d.collide_bgk3d`); any callable
  ``f, tau -> f`` built from differentiable ops drops in, e.g.
  :func:`tensorlbm.turbulence.collide_smagorinsky_bgk3d` to learn the
  Smagorinsky constant ``C_s`` through the solver.

This is the TensorLBM counterpart of the solver-in-the-loop paradigm of
Autodesk XLB (JAX, Apache-2.0, Ataei & Salehipour, CPC 300:109187, 2024) and
Um et al., NeurIPS 2020: the discrete time-stepping operator itself is in the
backward graph, so ``d(loss)/d(tau)``, ``d(loss)/d(C_s)`` and ``d(loss)/df0``
are exact derivatives of the *simulated* observable — no frozen-field
surrogate (``tensorlbm/adjoint.py``) and no hand-derived adjoint.

Scope (deliberate): single-component D3Q19 BGK-family collision, periodic
lattice units, stationary solid mask.  Multi-component/free-surface/distributed
paths and memory-format optimisations are out of scope for this module.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from .d3q19 import OPPOSITE
from .d3q19 import C as C3D
from .solver3d import collide_bgk3d, stream3d

CollideFn = Callable[..., torch.Tensor]

__all__ = ["differentiable_step", "obstacle_force", "rollout"]


def _collide_skip_solid(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    mask: torch.Tensor | None,
    collide: CollideFn | None,
) -> torch.Tensor:
    """Collision step, skipped inside the solid (NoDynamics obstacle cells).

    Solid cells carry the populations reflected by the previous bounce-back;
    relaxing them toward equilibrium would stream contaminated values into
    the fluid on the next :func:`differentiable_step`.  ``torch.where`` keeps
    the branch differentiable (the solid branch passes ``f`` through, and the
    reflected populations themselves carry gradients from earlier steps).
    """
    f_col = collide_bgk3d(f, tau) if collide is None else collide(f, tau)
    if mask is None:
        return f_col
    return torch.where(mask.unsqueeze(0), f, f_col)


def differentiable_step(
    f: torch.Tensor,
    tau: float | torch.Tensor = 0.9,
    mask: torch.Tensor | None = None,
    *,
    collide: CollideFn | None = None,
    return_probe: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One autograd-clean D3Q19 step: collide (fluid) -> stream -> bounce-back.

    Composition (the production SUBOFF phase order, memory note "force sample
    post-stream / pre-bounce-back"):

    1. BGK collision with relaxation time *tau*, skipped inside the solid
       mask (NoDynamics), see :func:`_collide_skip_solid`;
    2. periodic gather streaming (:func:`tensorlbm.solver3d.stream3d`);
    3. full-way bounce-back on the solid: ``f_new = where(mask, f_str[opp],
       f_str)`` — the reflected populations leave the solid on the next
       streaming step.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` (the state
            returned by a previous :func:`differentiable_step`, i.e.
            post-bounce-back).
        tau: BGK relaxation time; a 0-dim tensor with ``requires_grad=True``
            stays connected to the autograd graph.
        mask: Boolean solid mask of shape ``(nz, ny, nx)``; ``None`` gives
            the plain periodic collide->stream chain (identical operator
            order to the rollouts in ``tests/test_autograd.py``).
        collide: Optional replacement collision operator ``f, tau -> f``.
            Must be built from differentiable ops (e.g.
            ``functools.partial(collide_smagorinsky_bgk3d, C_s=cs)`` with a
            tensor ``C_s``).
        return_probe: Additionally return the post-stream / pre-bounce-back
            state, the sampling point for :func:`obstacle_force`.

    Returns:
        The stepped distribution ``f_new``; with ``return_probe=True`` a
        tuple ``(f_new, f_probe)``.  Macroscopic observables computed on
        ``f_new`` should exclude the solid cells (they hold reflected
        populations): mask with ``~mask``.
    """
    f_str = stream3d(_collide_skip_solid(f, tau, mask, collide))
    if mask is None:
        return (f_str, f_str) if return_probe else f_str
    probe = f_str
    f_new = torch.where(mask.unsqueeze(0), f_str[OPPOSITE.to(f_str.device)], f_str)
    return (f_new, probe) if return_probe else f_new


def obstacle_force(f_probe: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Momentum-exchange force on the bounce-back obstacle (Ladd, wet-node).

    Evaluate on the post-stream / pre-bounce-back state returned by
    ``differentiable_step(..., return_probe=True)``:

    .. math:: F_\\alpha = 2 \\sum_{x \\in \\mathrm{solid}} \\sum_q
        c_{q,\\alpha}\\, f[q, x]

    (house convention: factor 2 from the bounce-back momentum exchange,
    lattice units, force on the full obstacle).  The result carries
    autograd gradients w.r.t. everything ``f_probe`` depends on.

    Args:
        f_probe: Post-stream / pre-bounce-back distribution, shape
            ``(19, nz, ny, nx)``.
        mask: Boolean solid mask of shape ``(nz, ny, nx)``.

    Returns:
        Force vector ``(fx, fy, fz)`` as a tensor of shape ``(3,)``.
    """
    c = C3D.to(device=f_probe.device, dtype=f_probe.dtype)
    momentum = (f_probe * mask.to(dtype=f_probe.dtype)).sum(dim=(1, 2, 3))
    return 2.0 * torch.matmul(momentum, c)


def rollout(
    f: torch.Tensor,
    n_steps: int,
    tau: float | torch.Tensor = 0.9,
    mask: torch.Tensor | None = None,
    *,
    collide: CollideFn | None = None,
    checkpoint: bool = False,
    return_probes: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Roll out :func:`differentiable_step` for *n_steps*, keeping the graph.

    The plain mode stores every step's activations (memory grows linearly,
    ~9 ``(19, nz, ny, nx)`` tensors per step for BGK); ``checkpoint=True``
    wraps each step in ``torch.utils.checkpoint`` (``use_reentrant=False``)
    so activation memory stays near-flat while gradients remain identical
    (the strategy quantified in ``examples/differentiable_lbm.py`` and the
    transparent analogue of XLB's segmented checkpoint-replay adjoint).

    Args:
        f: Initial distribution, shape ``(19, nz, ny, nx)``.
        n_steps: Number of solver steps to apply.
        tau: Relaxation time (0-dim tensor allowed, keeps gradients).
        mask: Boolean solid mask ``(nz, ny, nx)`` or ``None`` for periodic.
        collide: Optional replacement collision operator (see
            :func:`differentiable_step`).
        checkpoint: Wrap every step in gradient checkpointing.
        return_probes: Collect the per-step post-stream / pre-bounce-back
            states for :func:`obstacle_force` (usable together with
            ``checkpoint``; the probes stay in the autograd graph).

    Returns:
        The final distribution; with ``return_probes=True`` a tuple
        ``(f_final, probes)`` with one probe per step.
    """
    probes: list[torch.Tensor] | None = [] if return_probes else None
    for _ in range(n_steps):
        if checkpoint:
            out = _checkpoint(
                differentiable_step,
                f,
                tau,
                mask,
                use_reentrant=False,
                collide=collide,
                return_probe=return_probes,
            )
        else:
            out = differentiable_step(f, tau, mask, collide=collide, return_probe=return_probes)
        if return_probes:
            f, probe = out
            probes.append(probe)
        else:
            f = out
    if return_probes:
        return f, probes
    return f
