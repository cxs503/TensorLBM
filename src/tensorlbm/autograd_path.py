"""Differentiable step chain for solver-in-the-loop autograd workflows.

`docs/differentiable_path.md` established that the eager operators
(``tensorlbm.solver3d``) are differentiable one by one.  This module turns
that property into a *usable composition contract*: a single autograd-clean
D3Q19 step — collision (fluid cells only) -> periodic gather streaming ->
optional boundary conditions on the x planes -> full-way bounce-back on a
solid mask — plus a rollout loop with optional per-step gradient
checkpointing and a differentiable momentum-exchange force probe on the
obstacle.

Everything is built from out-of-place torch ops that keep the autograd graph:

* collision is plain arithmetic (``f - (f - feq) / tau``), with ``tau``
  accepted as a graph-connected 0-dim tensor;
* streaming reuses the audited gather implementation ``solver3d.stream3d``
  (index tensors are cached constants);
* the solid is applied with ``torch.where`` selections only — no in-place
  boundary overwrites (the pattern documented as gradient-hostile in
  ``docs/differentiable_path.md``);
* the boundary conditions follow the same discipline: the inlet and outlet
  planes are *reconstructed* from candidate tensors selected with
  ``torch.where`` and re-assembled with ``torch.cat``, so gradients cross
  the boundary overwrites exactly (see :class:`InletSpec` /
  :class:`OutletSpec`);
* the collision operator is a slot: the default is single-component BGK
  (:func:`tensorlbm.solver3d.collide_bgk3d`); any callable
  ``f, tau -> f`` built from differentiable ops drops in, e.g.
  :func:`tensorlbm.turbulence.collide_smagorinsky_bgk3d` to learn the
  Smagorinsky constant ``C_s`` through the solver.

This is the TensorLBM counterpart of the solver-in-the-loop paradigm of
Autodesk XLB (JAX, Apache-2.0, Ataei & Salehipour, CPC 300:109187, 2024) and
Um et al., NeurIPS 2020: the discrete time-stepping operator itself is in the
backward graph, so ``d(loss)/d(tau)``, ``d(loss)/d(C_s)``, ``d(loss)/df0`` and
``d(loss)/du_in`` are exact derivatives of the *simulated* observable — no
frozen-field surrogate (``tensorlbm/adjoint.py``) and no hand-derived adjoint.

Scope (deliberate): single-component D3Q19 BGK-family collision, stationary
solid mask.  The default lattice stays fully periodic; passing
``inlet``/``outlet`` replaces the periodic wrap on the x planes with a
differentiable velocity inlet and a zero-gradient outlet (the SUBOFF
production phase order) — the lateral y/z planes remain periodic in this
first version, which is acceptable for wake-type campaigns on wide domains.
Multi-component/free-surface/distributed paths and memory-format
optimisations are out of scope for this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from .d3q19 import OPPOSITE, equilibrium3d
from .d3q19 import C as C3D
from .solver3d import collide_bgk3d, stream3d

CollideFn = Callable[..., torch.Tensor]

# Direction sets on the streamwise (x) boundary planes.  Layout is
# ``(19, nz, ny, nx)`` with ``c[q, 0]`` the shift along the *last* axis;
# ``stream3d`` gathers ``f_new[q, x] = f[q, x - c_qx]``, so after streaming
# the inlet plane x = 0 holds, for the c_x = +1 directions, values wrapped
# around from the outlet — those are the unknowns the inlet closure must
# supply.  Symmetrically the c_x = -1 directions on the outlet plane x = nx-1
# wrapped around from the inlet.  ``OPPOSITE`` maps one set onto the other
# element-wise.
_INCOMING_X = (1, 7, 9, 11, 13)  # c_x = +1: unknown at the inlet plane
_OUTGOING_X = (2, 8, 10, 12, 14)  # c_x = -1: unknown at the outlet plane
_NO_SHIFT_X = (0, 3, 4, 5, 6, 15, 16, 17, 18)  # c_x = 0

__all__ = [
    "InletSpec",
    "OutletSpec",
    "differentiable_step",
    "obstacle_force",
    "rollout",
]


@dataclass(frozen=True)
class InletSpec:
    """Differentiable velocity inlet on the plane x = 0.

    Two closures are available (``method``):

    ``"equilibrium"`` (default) — the whole inlet plane is (re)set to the
    Dirichlet equilibrium ``f_eq(rho0, u)``.  Cheap, exactly reproduces
    ``(rho0, u)`` on the plane and robust for driving a bounded campaign;
    the outgoing populations at the plane are absorbed rather than leaving
    the domain, which adds a small acoustic reflection.

    ``"zouhe"`` — only the five unknown (incoming, c_x = +1) populations are
    reconstructed with the Zou/He non-equilibrium correction

    .. math:: f_i = f_i^{eq}(\\rho_{in}, u) + f_{\\bar i} -
        f_{\\bar i}^{eq}(\\rho_{in}, u), \\qquad i \\in \\text{incoming},

    with the plane density obtained self-consistently from the known
    (streamed, i.e. post-collision) populations,

    .. math:: \\rho_{in} = \\big(\\textstyle\\sum_{c_x = 0} f +
        2 \\sum_{c_x = -1} f\\big) / (1 - u_x).

    The outgoing populations keep their streamed values, and the closure
    reproduces the prescribed *normal* velocity ``u_x`` and the Zou/He plane
    density exactly.  Two approximations are inherited from this standard
    non-equilibrium bounce-back form: the plane density differs from the
    exact Zou/He moment solution by the outgoing non-equilibrium, and the
    *tangential* momentum on the plane is whatever the ``c_x = 0``
    populations streamed in from the interior (it is not pinned to
    ``u_y``/``u_z``) — appropriate for streamwise inflow ``u = (u_in, 0, 0)``
    where the tangential contamination at the inlet is negligible.
    ``u_x`` must stay below 1 in lattice units (division by ``1 - u_x``).

    All components accept floats or 0-dim tensors; tensors with
    ``requires_grad=True`` keep the boundary condition in the autograd graph
    (e.g. calibrating the free-stream velocity against measured drag).
    """

    ux: float | torch.Tensor = 0.0
    uy: float | torch.Tensor = 0.0
    uz: float | torch.Tensor = 0.0
    rho0: float | torch.Tensor = 1.0
    method: str = "equilibrium"

    def __post_init__(self) -> None:
        if self.method not in ("equilibrium", "zouhe"):
            raise ValueError(f"inlet method must be 'equilibrium' or 'zouhe', got {self.method!r}")


@dataclass(frozen=True)
class OutletSpec:
    """Differentiable zero-gradient (copy) outlet on the plane x = nx - 1.

    After streaming, the five unknown (outgoing, c_x = -1) populations on
    the outlet plane would wrap around from the inlet; they are replaced by
    the values of the fully interior neighbour plane x = nx - 2, i.e. a
    zero-gradient extrapolation of the distribution.  The known (c_x >= 0)
    populations keep their streamed interior values.  The condition carries
    gradients (the copied plane is graph-connected); a convective outlet is
    a possible future extension of this marker.
    """


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
    the branch differentiable (the solid branch passes ``f`` through, and
    the reflected populations themselves carry gradients from earlier steps).
    """
    f_col = collide_bgk3d(f, tau) if collide is None else collide(f, tau)
    if mask is None:
        return f_col
    return torch.where(mask.unsqueeze(0), f, f_col)


def _scalar(value: float | torch.Tensor, dtype: torch.dtype, device: torch.device):
    """Coerce a Python float or 0-dim tensor to *dtype/device* (graph kept)."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _dir_selector(indices: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Boolean (19, 1, 1, 1) selector, True on *indices*."""
    sel = torch.zeros((19, 1, 1, 1), dtype=torch.bool, device=device)
    sel[list(indices), 0, 0, 0] = True
    return sel


def _apply_inlet(f_str: torch.Tensor, inlet: InletSpec) -> torch.Tensor:
    """Replace the inlet plane x = 0 of the post-stream state (out-of-place).

    Returns ``cat([plane_new, f_str[..., 1:]], dim=-1)`` — a new tensor, so
    the discarded wrapped values simply drop out of the autograd graph and
    every kept entry keeps its gradient.
    """
    device, dtype = f_str.device, f_str.dtype
    nz, ny = f_str.shape[1], f_str.shape[2]
    ux = _scalar(inlet.ux, dtype, device)
    uy = _scalar(inlet.uy, dtype, device)
    uz = _scalar(inlet.uz, dtype, device)
    plane = f_str[..., :1]  # (19, nz, ny, 1) post-stream inlet slice

    if inlet.method == "equilibrium":
        rho = _scalar(inlet.rho0, dtype, device)
        # 0-dim inputs broadcast to the whole (19, nz, ny, 1) plane
        plane_new = equilibrium3d(rho, ux, uy, uz, device).expand(19, nz, ny, 1)
    else:  # "zouhe"
        rest = plane[list(_NO_SHIFT_X)].sum(dim=0)
        outgoing = plane[list(_OUTGOING_X)].sum(dim=0)
        rho_in = (rest + 2.0 * outgoing) / (1.0 - ux)  # (nz, ny, 1)
        ones = torch.ones_like(rho_in)  # broadcast the 0-dim velocity to the plane
        feq = equilibrium3d(rho_in, ux * ones, uy * ones, uz * ones, device)  # (19, nz, ny, 1)
        opp = OPPOSITE.to(device)
        # candidate for every q: feq + (f_opp - feq_opp); masked to the
        # five unknown incoming directions by the selector below
        cand = feq + (plane[opp] - feq[opp])
        plane_new = torch.where(_dir_selector(_INCOMING_X, device), cand, plane)

    return torch.cat([plane_new, f_str[..., 1:]], dim=-1)


def _apply_outlet(f_str: torch.Tensor) -> torch.Tensor:
    """Zero-gradient outlet on x = nx - 1 of the post-stream state (out-of-place).

    Only the unknown outgoing (c_x = -1) populations are copied from x = nx-2;
    the known ones keep their streamed interior values.
    """
    plane_new = torch.where(
        _dir_selector(_OUTGOING_X, f_str.device),
        f_str[..., -2:-1],
        f_str[..., -1:],
    )
    return torch.cat([f_str[..., :-1], plane_new], dim=-1)


def differentiable_step(
    f: torch.Tensor,
    tau: float | torch.Tensor = 0.9,
    mask: torch.Tensor | None = None,
    *,
    collide: CollideFn | None = None,
    return_probe: bool = False,
    inlet: InletSpec | None = None,
    outlet: OutletSpec | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One autograd-clean D3Q19 step: collide (fluid) -> stream -> BC -> bounce-back.

    Composition (the production SUBOFF phase order, memory note "force sample
    post-stream / pre-bounce-back"):

    1. BGK collision with relaxation time *tau*, skipped inside the solid
       mask (NoDynamics), see :func:`_collide_skip_solid`;
    2. periodic gather streaming (:func:`tensorlbm.solver3d.stream3d`);
    3. if given, the inlet/outlet conditions on the x planes
       (:func:`_apply_inlet`, :func:`_apply_outlet`);
    4. full-way bounce-back on the solid: ``f_new = where(mask, f_bc[opp],
       f_bc)`` — the reflected populations leave the solid on the next
       streaming step.

    Boundary phase — why post-stream / pre-bounce-back: streaming is the
    operator whose periodic wrap injects the unphysical cross-domain data, so
    the correction belongs immediately after it; at that moment the known
    populations on the boundary planes still carry the freshest post-collision
    interior information, which is exactly what the Zou/He non-equilibrium
    correction requires (non-equilibrium of *post-collision* populations —
    the phase is self-consistent with the collision that just ran).  Applying
    the boundaries before bounce-back also keeps the force probe (below) on
    the boundary-conditioned state, matching the production sampling phase.

    With ``inlet=None, outlet=None`` (default) the operator is bit-for-bit
    the original periodic chain.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)`` (the state
            returned by a previous :func:`differentiable_step`, i.e.
            post-bounce-back).
        tau: BGK relaxation time; a 0-dim tensor with ``requires_grad=True``
            stays connected to the autograd graph.
        mask: Boolean solid mask of shape ``(nz, ny, nx)``; ``None`` gives
            the plain periodic collide->stream chain (identical operator
            order to the rollouts in ``tests/test_autograd.py``).  The
            inlet/outlet planes are assumed fluid; if the mask is True
            there, bounce-back (applied last) wins over the boundary value.
        collide: Optional replacement collision operator ``f, tau -> f``.
            Must be built from differentiable ops (e.g.
            ``functools.partial(collide_smagorinsky_bgk3d, C_s=cs)`` with a
            tensor ``C_s``).
        return_probe: Additionally return the post-stream /
            post-boundary-condition / pre-bounce-back state, the sampling
            point for :func:`obstacle_force`.
        inlet: :class:`InletSpec` velocity inlet on x = 0 (``None``: the
            plane stays periodic).
        outlet: :class:`OutletSpec` zero-gradient outlet on x = nx - 1
            (``None``: the plane stays periodic).

    Returns:
        The stepped distribution ``f_new``; with ``return_probe=True`` a
        tuple ``(f_new, f_probe)``.  Macroscopic observables computed on
        ``f_new`` should exclude the solid cells (they hold reflected
        populations): mask with ``~mask``.
    """
    f_str = stream3d(_collide_skip_solid(f, tau, mask, collide))
    if inlet is not None:
        f_str = _apply_inlet(f_str, inlet)
    if outlet is not None:
        f_str = _apply_outlet(f_str)
    probe = f_str
    if mask is None:
        return (probe, probe) if return_probe else probe
    f_new = torch.where(mask.unsqueeze(0), probe[OPPOSITE.to(probe.device)], probe)
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
    inlet: InletSpec | None = None,
    outlet: OutletSpec | None = None,
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
        inlet: :class:`InletSpec` velocity inlet on x = 0 (default periodic).
        outlet: :class:`OutletSpec` zero-gradient outlet on x = nx - 1
            (default periodic).

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
                inlet=inlet,
                outlet=outlet,
            )
        else:
            out = differentiable_step(
                f,
                tau,
                mask,
                collide=collide,
                return_probe=return_probes,
                inlet=inlet,
                outlet=outlet,
            )
        if return_probes:
            f, probe = out
            probes.append(probe)
        else:
            f = out
    if return_probes:
        return f, probes
    return f
