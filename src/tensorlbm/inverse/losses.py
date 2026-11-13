"""Composable loss functions for inverse-problem optimisation.

Each loss is a callable ``(sim, obs) -> torch.Tensor`` that returns a scalar
loss value.  Losses can be combined with :class:`LossComposition` to form
weighted sums, enabling multi-objective inverse problems (e.g. field-MSE +
integral constraint + regularisation).

Design
------
- All losses accept ``(sim, obs)`` where ``sim`` is the simulated observable
  (tensor or sequence of tensors) and ``obs`` is the reference (same shape or
  a dict with ``"fields"`` / ``"times"`` keys).
- Losses are stateless pure functions — the optimiser loop owns the state.
- ``LossComposition`` supports named components so the convergence log can
  report per-component contributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

__all__ = [
    "LossComposition",
    "LossFn",
    "l1_loss",
    "relative_mse_loss",
    "weighted_mse_loss",
]

#: Signature for a loss function: (simulated, observed) -> scalar tensor.
LossFn = Callable[[Any, Any], torch.Tensor]


# ---------------------------------------------------------------------------
# Primitive losses
# ---------------------------------------------------------------------------


def relative_mse_loss(
    sim: Sequence[tuple[torch.Tensor, torch.Tensor]],
    obs: Any,
) -> torch.Tensor:
    """Relative mean-squared error over a sequence of velocity-field pairs.

    Each time step contributes one unit of weight::

        L = (1/N) sum_t [ ||u_sim - u_obs||^2 + ||v_sim - v_obs||^2 ]
                            / [ ||u_obs||^2 + ||v_obs||^2 ]

    This is the default loss for the Taylor-Green tau-recovery demo.
    """
    fields = obs.fields if hasattr(obs, "fields") else obs
    total: torch.Tensor | None = None
    n = 0
    for (sx, sy), (ox, oy) in zip(sim, fields):
        num = ((sx - ox) ** 2).mean() + ((sy - oy) ** 2).mean()
        den = (ox**2).mean() + (oy**2).mean()
        term = num / den
        total = term if total is None else total + term
        n += 1
    if total is None or n == 0:
        return torch.tensor(0.0, dtype=sim[0][0].dtype if sim else torch.float64)
    return total / n


def weighted_mse_loss(
    sim: Sequence[tuple[torch.Tensor, torch.Tensor]],
    obs: Any,
    *,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Weighted MSE over field pairs.

    If ``weights`` is None, uniform weighting is used (equivalent to
    :func:`relative_mse_loss` but with absolute, not relative, error).
    """
    fields = obs.fields if hasattr(obs, "fields") else obs
    if weights is None:
        weights = [1.0] * len(fields)
    total: torch.Tensor | None = None
    w_sum = 0.0
    for w, (sx, sy), (ox, oy) in zip(weights, sim, fields):
        term = w * (((sx - ox) ** 2).mean() + ((sy - oy) ** 2).mean())
        total = term if total is None else total + term
        w_sum += w
    if total is None or w_sum == 0:
        return torch.tensor(0.0, dtype=sim[0][0].dtype if sim else torch.float64)
    return total / w_sum


def l1_loss(
    sim: Sequence[tuple[torch.Tensor, torch.Tensor]],
    obs: Any,
) -> torch.Tensor:
    """Mean absolute error (L1) over field pairs."""
    fields = obs.fields if hasattr(obs, "fields") else obs
    total: torch.Tensor | None = None
    n = 0
    for (sx, sy), (ox, oy) in zip(sim, fields):
        term = (sx - ox).abs().mean() + (sy - oy).abs().mean()
        total = term if total is None else total + term
        n += 1
    if total is None or n == 0:
        return torch.tensor(0.0, dtype=sim[0][0].dtype if sim else torch.float64)
    return total / n


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossComponent:
    """A named loss with a weight."""

    name: str
    fn: LossFn
    weight: float = 1.0


@dataclass(frozen=True)
class LossComposition:
    """Weighted sum of named loss components.

    The total loss is ``sum(w_i * L_i(sim, obs))``.  Each component's
    contribution is recorded separately in the convergence log.
    """

    components: tuple[LossComponent, ...]

    def __call__(self, sim: Any, obs: Any) -> torch.Tensor:
        total: torch.Tensor | None = None
        for comp in self.components:
            term = comp.fn(sim, obs)
            contribution = comp.weight * term
            total = contribution if total is None else total + contribution
        if total is None:
            return torch.tensor(0.0)
        return total

    def component_losses(self, sim: Any, obs: Any) -> dict[str, float]:
        """Evaluate each component separately (for convergence logging)."""
        result: dict[str, float] = {}
        for comp in self.components:
            val = comp.fn(sim, obs)
            result[comp.name] = float(val.detach())
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components)


def default_relative_mse() -> LossComposition:
    """Single-component loss: relative MSE (the Taylor-Green demo default)."""
    return LossComposition((LossComponent("rel_mse", relative_mse_loss, 1.0),))
