"""Per-step setup hooks + mask-blend boundary composition.

Formulation taken from ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
(``LatFlow/Domain.py`` ``Solve`` / ``CollideSC`` and its ``lid_cavity.py`` driver),
ported onto TensorLBM's torch conventions:

* **per-step setup hook** -- upstream's ``Domain.Solve(sess, Tf, initialize_step,
  setup_step, save_step, save_interval)`` is a plain Python driver loop whose
  per-step pre-processing (moving lid velocity, inlet value, ...) and output
  cadence are *injected as callbacks* rather than baked into the solver.  Here
  the same shape is provided by :func:`solve_with_hooks`: ``initialize_step``
  once, then per step ``save_step`` (on the save tick) -> ``setup_step`` ->
  one lattice step.  Hooks may mutate the ``state`` dict in place and return it
  (or return ``None``); the return value of ``save_step`` is ignored.
* **mask blend** -- boundary conditions are *not* branches over cell indices but
  one tensor expression over a float mask field: ``f = f*(1 - m) + f_bc*m``.
  Upstream hard-codes a single mask (``self.boundary``, a constant) and a single
  BC family (stationary bounce).  :func:`compose_masked_boundaries` generalises
  that to a *partition* of masks, i.e. several BC families superposed over
  arbitrary geometry::

      f_new = f_post * (1 - M) + sum_k m_k * f_bc_k,        M = sum_k m_k

  with ``M <= 1`` enforced (a cell belongs to at most one BC).

Two porting decisions are deliberate and are measured in
``benchmarks/xval_per_step_hook_mask_blend``:

1. Upstream's bounce blend uses the *pre*-collision populations and injects no
   wall momentum, so the mask can only express a **stationary** wall.  Moving
   walls are made possible here by :func:`moving_bounce_layer`, which adds the
   standard momentum term ``6 w_i (c_i . u_wall)`` to the reversed population.
   With ``u_wall = 0`` it is bit-wise identical to the stationary layer.
2. Upstream's per-step hook can only *overwrite solver variables* (see
   ``lid_cavity.py``: ``Vel[:, :1] += value``).  A velocity that is never fed
   back into the populations does not drive the flow; :func:`equilibrium_layer`
   supplies the missing piece when a Dirichlet velocity BC is wanted at a mask.

Everything is torch and dtype/shape generic: fields are ``(q, ny, nx)`` or
``(q, nz, ny, nx)`` and masks ``(ny, nx)`` / ``(nz, ny, nx)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

__all__ = [
    "BoundaryLayer",
    "HookLog",
    "TableSet",
    "blend_coverage",
    "compose_masked_boundaries",
    "equilibrium_layer",
    "layer_populations",
    "masked_step",
    "moving_bounce_layer",
    "solve_with_hooks",
    "stationary_bounce_layer",
    "tables_for",
    "velocity_ramp_hook",
]

KINDS = ("bounce", "moving", "equilibrium")

# --------------------------------------------------------------------- tables
@dataclass(frozen=True)
class TableSet:
    """Velocity / weight / direction-reversal tables for one lattice."""

    lveloc: torch.Tensor  # (q, dim)
    opposite: torch.Tensor  # (q,)
    weights: torch.Tensor  # (q,)

    @property
    def dim(self) -> int:
        return int(self.lveloc.shape[1])


def tables_for(f: torch.Tensor, *, exact_weights: bool = True) -> TableSet:
    """Default lattice tables for a field of rank 3 (2D) or rank 4 (3D).

    The library weight tables (``d2q9.W`` / ``d3q19.W``) are float32.  For a
    float64 field the *exact* weights are used instead, otherwise the momentum
    injected by a moving-wall layer (``6 w_i c_i u``) carries the float32
    rounding of ``1/9`` (~7.5e-9 relative) into the wall flux.
    """
    if f.ndim == 3:
        from .d2q9 import C, OPPOSITE, W, W_EXACT

        weights = torch.as_tensor(W_EXACT, dtype=torch.float64) if f.dtype == torch.float64 else W
        return TableSet(C.to(f.dtype), OPPOSITE.to(torch.int64), weights.to(f.dtype))
    if f.ndim == 4:
        from .d3q19 import C, OPPOSITE, W, W_EXACT64

        weights = W_EXACT64 if f.dtype == torch.float64 else W
        return TableSet(C.to(f.dtype), OPPOSITE.to(torch.int64), weights.to(f.dtype))
    raise ValueError(f"expected a (q, ny, nx) or (q, nz, ny, nx) field, got rank {f.ndim}")


# --------------------------------------------------------------------- layers
@dataclass(frozen=True)
class BoundaryLayer:
    """One boundary family carried by a float mask.

    ``mask`` is 1.0 on the cells that belong to the BC (partial values are
    allowed and mean a partial blend, exactly like upstream's float ``boundary``).
    ``velocity`` is required for ``kind="moving"``/``"equilibrium"`` and may be

    * a length-``dim`` sequence / tensor (uniform velocity), or
    * a tensor with trailing axis ``dim`` and the field's spatial shape, or
    * a callable ``state -> velocity`` (this is how a per-step setup hook feeds
      a time-dependent wall/lid velocity into the BC).
    """

    name: str
    mask: torch.Tensor
    kind: str = "bounce"
    velocity: Any = None
    density: Any = 1.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown boundary kind {self.kind!r}; expected one of {KINDS}")
        if self.kind in ("moving", "equilibrium") and self.velocity is None:
            raise ValueError(f"boundary layer {self.name!r} of kind {self.kind!r} needs a velocity")
        m = self.mask
        if not isinstance(m, torch.Tensor):
            raise TypeError(f"mask of layer {self.name!r} must be a torch.Tensor")
        if m.is_floating_point() or m.is_complex():
            lo = float(m.min())
            hi = float(m.max())
        else:
            lo = float(m.min().item())
            hi = float(m.max().item())
        if lo < -1e-9 or hi > 1.0 + 1e-9:
            raise ValueError(
                f"mask of layer {self.name!r} must lie in [0, 1], got [{lo}, {hi}]"
            )


def stationary_bounce_layer(name: str, mask: torch.Tensor) -> BoundaryLayer:
    """Upstream ``CollideSC`` wall: reversed populations from the pre-collision state."""
    return BoundaryLayer(name=name, mask=mask, kind="bounce")


def moving_bounce_layer(name: str, mask: torch.Tensor, velocity: Any) -> BoundaryLayer:
    """Bounce with the momentum term ``6 w_i (c_i . u_wall)`` added.

    This is the piece upstream lacks; it makes a mask-blended wall a *moving*
    no-slip wall.  ``velocity`` may be a callable so a per-step hook can ramp it.
    """
    return BoundaryLayer(name=name, mask=mask, kind="moving", velocity=velocity)


def equilibrium_layer(
    name: str, mask: torch.Tensor, velocity: Any, density: Any = 1.0
) -> BoundaryLayer:
    """Dirichlet velocity BC by equilibrium substitution on the masked cells."""
    return BoundaryLayer(name=name, mask=mask, kind="equilibrium", velocity=velocity, density=density)


# ------------------------------------------------------------------ internals
def _resolve(value: Any, state: dict[str, Any] | None, like: torch.Tensor) -> Any:
    if callable(value):
        return value(state if state is not None else {}, like)
    return value


def _mask_like(mask: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Broadcast a spatial mask to ``(1, *spatial)`` in ``f``'s dtype/device."""
    m = mask.to(dtype=f.dtype, device=f.device)
    if m.ndim == f.ndim - 1:
        m = m.unsqueeze(0)
    elif m.ndim == f.ndim and m.shape[0] == 1:
        pass
    elif m.ndim == f.ndim and m.shape[0] == f.shape[0]:
        pass  # per-direction mask, used verbatim
    else:
        raise ValueError(
            f"mask rank {m.ndim} does not fit field shape {tuple(f.shape)} "
            f"(expected {f.ndim - 1} or {f.ndim})"
        )
    return m


def _as_velocity_tensor(value: Any, f: torch.Tensor, dim: int) -> torch.Tensor:
    u = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    u = u.to(dtype=f.dtype, device=f.device)
    if u.ndim == 0 or (u.ndim == 1 and u.numel() < dim):
        raise ValueError(f"velocity needs {dim} components, got shape {tuple(u.shape)}")
    if u.ndim >= 1 and u.shape[-1] > dim:
        u = u[..., :dim]
    return u


def _velocity_kernel(velocity: Any, tables: TableSet, f: torch.Tensor) -> torch.Tensor:
    """``c . u`` for a uniform velocity (shape ``(q,)``) or a velocity field (``(q, *spatial)``)."""
    u = _as_velocity_tensor(velocity, f, tables.dim)
    c = tables.lveloc.to(dtype=f.dtype, device=f.device)[:, : tables.dim]
    if u.ndim == 1:
        return c @ u
    return torch.einsum("qd,...d->q...", c, u)


def layer_populations(
    layer: BoundaryLayer,
    f_pre: torch.Tensor,
    *,
    tables: TableSet | None = None,
    state: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Candidate populations ``f_bc`` that this layer would install on its mask."""
    tables = tables or tables_for(f_pre)
    reversed_pop = f_pre[tables.opposite.to(f_pre.device)]
    if layer.kind == "bounce":
        return reversed_pop
    w = tables.weights.to(dtype=f_pre.dtype, device=f_pre.device)
    wshape = [w.shape[0]] + [1] * (f_pre.ndim - 1)
    velocity = _resolve(layer.velocity, state, f_pre)
    cu = _velocity_kernel(velocity, tables, f_pre)
    if layer.kind == "moving":
        return reversed_pop + 6.0 * w.view(*wshape) * cu.reshape(wshape)
    # kind == "equilibrium": feq(rho, u) with cs^2 = 1/3 and dx = dt = 1
    u = _as_velocity_tensor(velocity, f_pre, tables.dim)
    uu = (u * u).sum(dim=-1) if u.ndim > 1 else (u * u).sum()
    rho = _resolve(layer.density, state, f_pre)
    rho_t = rho if isinstance(rho, torch.Tensor) else torch.as_tensor(rho, dtype=f_pre.dtype, device=f_pre.device)
    rho_t = rho_t.to(dtype=f_pre.dtype, device=f_pre.device)
    return w.view(*wshape) * rho_t * (1.0 + 3.0 * cu.reshape(wshape)
                                      + 4.5 * cu.reshape(wshape) ** 2 - 1.5 * uu)


def blend_coverage(layers: Sequence[BoundaryLayer], f: torch.Tensor) -> torch.Tensor:
    """Total mask ``M = sum_k m_k`` (the fraction of each cell covered by a BC)."""
    total = None
    for layer in layers:
        m = _mask_like(layer.mask, f)
        total = m.clone() if total is None else total + m
    if total is None:
        raise ValueError("no boundary layers given")
    return total


def compose_masked_boundaries(
    f_post_collision: torch.Tensor,
    f_pre_collision: torch.Tensor,
    layers: Iterable[BoundaryLayer],
    *,
    tables: TableSet | None = None,
    state: dict[str, Any] | None = None,
    strict: bool = True,
    overlap_tolerance: float = 1e-6,
) -> torch.Tensor:
    """``f = f_post*(1 - M) + sum_k m_k * f_bc_k`` over a partition of masks.

    Upstream implements the single-layer case (one mask, stationary bounce).
    The multi-layer form lets wall / moving wall / inlet-outlet coexist on an
    arbitrary geometry; ``strict`` enforces that the masks do not overlap, so a
    cell can never receive two conflicting BCs.
    """
    layers = list(layers)
    if not layers:
        return f_post_collision
    tables = tables or tables_for(f_pre_collision)
    coverage = blend_coverage(layers, f_post_collision)
    if strict:
        worst = float(coverage.max()) if coverage.numel() else 0.0
        if worst > 1.0 + overlap_tolerance:
            raise ValueError(
                "boundary masks overlap: sum of masks reaches "
                f"{worst:.6f} > 1 (overlap_tolerance={overlap_tolerance}); "
                "give each cell to at most one boundary layer or pass strict=False"
            )
    out = f_post_collision * (1.0 - coverage)
    for layer in layers:
        m = _mask_like(layer.mask, f_post_collision)
        out = out + m * layer_populations(
            layer, f_pre_collision, tables=tables, state=state
        )
    return out


# ------------------------------------------------------------------- stepping
@dataclass
class HookLog:
    """Record of what the driver actually called (assertable in tests)."""

    initialize: int = 0
    setup: list[int] = field(default_factory=list)
    save: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"initialize": self.initialize, "setup": list(self.setup), "save": list(self.save)}


def _apply(hook: Callable[..., Any] | None, state: dict[str, Any], *args: Any) -> dict[str, Any]:
    if hook is None:
        return state
    out = hook(state, *args)
    return state if out is None else out


def solve_with_hooks(
    state: dict[str, Any],
    n_steps: int,
    *,
    operator: Callable[[dict[str, Any], float], dict[str, Any]],
    initialize_step: Callable[..., Any] | None = None,
    setup_step: Callable[..., Any] | None = None,
    save_step: Callable[..., Any] | None = None,
    save_interval: int = 0,
    t0: float = 0.0,
    dt: float = 1.0,
) -> tuple[dict[str, Any], HookLog]:
    """Upstream ``Domain.Solve`` as a torch driver.

    Order per step: ``save_step`` (on the save tick) -> ``setup_step`` ->
    ``operator`` (one lattice step).  ``setup_step`` therefore lands in the same
    place as upstream's ``sess.run(setup_step)`` immediately before
    ``collide``/``stream``, and a velocity written there is visible to the BC of
    that very step.  ``state`` is a dict; ``state["f"]`` holds the populations.
    """
    log = HookLog()
    if initialize_step is not None:
        state = _apply(initialize_step, state, t0)
        log.initialize += 1
    t = t0
    for i in range(int(n_steps)):
        if save_step is not None and save_interval and i % int(save_interval) == 0:
            _apply(save_step, state, t, i)
            log.save.append(i)
        state = _apply(setup_step, state, t, i)
        log.setup.append(i)
        out = operator(state, t)
        if out is not None:
            state = out
        t += dt
    if save_step is not None:
        _apply(save_step, state, t, int(n_steps))
        log.save.append(int(n_steps))
    return state, log


def velocity_ramp_hook(
    key: str = "wall_velocity",
    velocity: Any = None,
    ramp_steps: int = 0,
    *,
    dim: int = 2,
    component: int = 0,
    device: torch.device | None = None,
) -> Callable[..., dict[str, Any]]:
    """``setup_step`` hook that ramps a uniform velocity from 0 to ``velocity``.

    The layer then reads ``state[key]`` through a callable ``velocity``, which is
    the torch analogue of upstream's ``setup_step`` writing ``domain.Vel``.
    Instant onset (``ramp_steps <= 0``) reproduces upstream's step change.
    """

    def hook(state: dict[str, Any], t: float, i: int) -> dict[str, Any]:
        frac = 1.0 if ramp_steps <= 0 else min(1.0, float(i + 1) / float(ramp_steps))
        u = [0.0] * dim
        if velocity is None:
            raise ValueError("velocity_ramp_hook needs a target velocity")
        if isinstance(velocity, torch.Tensor):
            u = [float(v) * frac for v in velocity.tolist()][:dim]
        else:
            u = [float(v) * frac for v in list(velocity)[:dim]]
        state[key] = torch.tensor(u, dtype=state["f"].dtype, device=state["f"].device)
        return state

    return hook


def masked_step(
    *,
    tau: float | Callable[[dict[str, Any]], float],
    layers: Sequence[BoundaryLayer] = (),
    transport: Callable[[torch.Tensor], torch.Tensor] | None = None,
    collide: Callable[[torch.Tensor, float], torch.Tensor] | None = None,
    source: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor] | None = None,
    tables: TableSet | None = None,
    strict: bool = True,
) -> Callable[[dict[str, Any], float], dict[str, Any]]:
    """Build the lattice-step operator ``state -> state`` used by :func:`solve_with_hooks`.

    Ordering is ``stream -> collide -> mask blend``: the blend consumes the
    post-*stream* populations as its "pre-collision" state, which is the same
    pairing upstream uses (``CollideSC`` bounces ``self.F`` as it stands before
    that step's collision).
    """
    if transport is None or collide is None:
        from .solver import collide_bgk, stream

        transport = transport or stream
        collide = collide or collide_bgk

    def operator(state: dict[str, Any], t: float) -> dict[str, Any]:
        f = state["f"]
        f_streamed = transport(f)
        tau_now = tau(state) if callable(tau) else float(tau)
        f_post = collide(f_streamed, tau_now)
        if source is not None:
            f_post = f_post + source(f_streamed, state)
        if layers:
            f_post = compose_masked_boundaries(
                f_post,
                f_streamed,
                layers,
                tables=tables,
                state=state,
                strict=strict,
            )
        state["f"] = f_post
        state["t"] = t
        return state

    return operator