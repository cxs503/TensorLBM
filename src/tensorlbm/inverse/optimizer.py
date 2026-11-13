"""Generic optimiser loop for differentiable inverse problems.

Provides :func:`optimise_loop` — a reusable Adam + optional L-BFGS polish
loop with cosine learning-rate decay, box constraints, convergence detection,
and per-iteration convergence logging.  The loop is parameter-agnostic: it
operates on any list of ``nn.Parameter`` s and any differentiable loss
callable.

This is the core of ``tensorlbm.inverse``: the Taylor-Green tau-recovery demo
and any future inverse problem (geometry, boundary condition, material
parameter) all use this same loop.

Usage
-----
::

    from tensorlbm.inverse import optimise_loop, LossComposition, default_relative_mse

    # params = [log_nu, amplitude]  # nn.Parameters with requires_grad=True
    result = optimise_loop(
        params=params,
        loss_fn=default_relative_mse(),
        forward_fn=lambda: simulate(tau, amplitude),  # returns (sim, obs) pair
        iters=20,
        lr=0.05,
        polish=10,
        box_constraints={"log_nu": (lo, hi)},
        convergence_log=log,  # optional ConvergenceLog
    )
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from .convergence import ConvergenceLog, IterationRecord
from .losses import LossComposition, LossFn, default_relative_mse

__all__ = [
    "OptimiseResult",
    "optimise_loop",
]


@dataclass
class OptimiseResult:
    """Outcome of one optimisation run."""

    converged: bool
    n_iterations: int
    wall_s: float
    final_loss: float
    initial_loss: float
    params: dict[str, float]
    history: list[IterationRecord] = field(default_factory=list)

    @property
    def loss_reduction(self) -> float:
        return self.initial_loss - self.final_loss if self.initial_loss > 0 else 0.0


def _extract_param_values(
    params: list[torch.Tensor],
    param_names: Sequence[str],
) -> dict[str, float]:
    return {n: float(p.detach().item()) for n, p in zip(param_names, params)}


def _extract_gradients(
    params: list[torch.Tensor],
    param_names: Sequence[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for n, p in zip(param_names, params):
        if p.grad is not None:
            result[n] = float(p.grad.item())
        else:
            result[n] = float("nan")
    return result


def optimise_loop(
    *,
    params: list[torch.Tensor],
    param_names: Sequence[str],
    loss_fn: LossComposition | LossFn,
    forward_fn: Callable[[], tuple[Any, Any]],
    iters: int = 20,
    lr: float = 0.05,
    polish: int = 0,
    box_constraints: Mapping[str, tuple[float, float]] | None = None,
    convergence_log: ConvergenceLog | None = None,
    tol: float = 1e-10,
    problem_name: str = "inverse",
    seed: int = 0,
    extra_per_iter: Callable[[], dict[str, float]] | None = None,
) -> OptimiseResult:
    """Run the Adam + optional L-BFGS polish optimisation loop.

    Args:
        params: List of ``nn.Parameter`` tensors to optimise.
        param_names: Names for each parameter (for logging).
        loss_fn: Loss function or composition. Called as ``loss_fn(sim, obs)``.
        forward_fn: Zero-arg callable that runs the forward model and returns
            ``(sim, obs)``.  Must be differentiable w.r.t. ``params``.
        iters: Number of Adam iterations.
        lr: Adam learning rate.
        polish: Number of L-BFGS polish iterations (0 = skip).
        box_constraints: ``{param_name: (lo, hi)}`` applied after each step.
        convergence_log: Optional log to record per-iteration state.
        tol: Convergence tolerance on loss change (relative).
        problem_name: Label for the convergence log.
        seed: Seed for reproducibility tracking.
        extra_per_iter: Optional callable returning extra metrics per iteration.

    Returns:
        :class:`OptimiseResult` with final state and history.
    """
    t_start = time.time()
    box_constraints = box_constraints or {}

    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters))

    history: list[IterationRecord] = []
    loss_value = float("inf")
    loss_prev = float("inf")
    converged = False
    initial_loss: float | None = None

    for it in range(iters):
        opt.zero_grad()
        sim, obs = forward_fn()

        if isinstance(loss_fn, LossComposition):
            loss = loss_fn(sim, obs)
            comp_losses = loss_fn.component_losses(sim, obs)
        else:
            loss = loss_fn(sim, obs)
            comp_losses = {}

        loss.backward()
        grads = _extract_gradients(params, param_names)
        opt.step()
        sched.step()

        # Apply box constraints
        with torch.no_grad():
            for name, p in zip(param_names, params):
                if name in box_constraints:
                    lo, hi = box_constraints[name]
                    p.clamp_(min=lo, max=hi)

        loss_value = float(loss.detach())
        if initial_loss is None:
            initial_loss = loss_value

        param_values = _extract_param_values(params, param_names)
        extra = extra_per_iter() if extra_per_iter else {}

        rec = IterationRecord(
            iter=it,
            phase="adam",
            loss=loss_value,
            params=param_values,
            gradients=grads,
            lr=float(sched.get_last_lr()[0]),
            extra={**comp_losses, **extra},
        )
        history.append(rec)
        if convergence_log:
            convergence_log.record(rec)

        # Convergence check
        if it > 2 and abs(loss_prev - loss_value) < tol * max(1.0, abs(loss_prev)):
            converged = True
            break
        loss_prev = loss_value

    # L-BFGS polish
    if polish > 0 and loss_value == loss_value:  # not NaN
        opt_l = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=int(polish),
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
        )

        def closure() -> torch.Tensor:
            opt_l.zero_grad()
            sim_t, obs_t = forward_fn()
            if isinstance(loss_fn, LossComposition):
                value = loss_fn(sim_t, obs_t)
            else:
                value = loss_fn(sim_t, obs_t)
            value.backward()
            param_values = _extract_param_values(params, param_names)
            rec = IterationRecord(
                iter=len(history),
                phase="lbfgs_eval",
                loss=float(value.detach()),
                params=param_values,
                gradients=_extract_gradients(params, param_names),
                lr=1.0,
            )
            history.append(rec)
            if convergence_log:
                convergence_log.record(rec)
            return value

        opt_l.step(closure)

        # Final constraint clamp
        with torch.no_grad():
            for name, p in zip(param_names, params):
                if name in box_constraints:
                    lo, hi = box_constraints[name]
                    p.clamp_(min=lo, max=hi)

        # Final evaluation
        sim_eval, obs_eval = forward_fn()
        if isinstance(loss_fn, LossComposition):
            loss_value = float(loss_fn(sim_eval, obs_eval).detach())
        else:
            loss_value = float(loss_fn(sim_eval, obs_eval).detach())

        param_values = _extract_param_values(params, param_names)
        rec = IterationRecord(
            iter=len(history),
            phase="final",
            loss=loss_value,
            params=param_values,
        )
        history.append(rec)
        if convergence_log:
            convergence_log.record(rec)

    wall_s = time.time() - t_start
    param_values = _extract_param_values(params, param_names)

    if convergence_log:
        convergence_log.finalise(
            wall_s=wall_s,
            converged=converged,
            final_loss=loss_value,
            initial_loss=initial_loss if initial_loss is not None else loss_value,
            params=param_values,
        )

    return OptimiseResult(
        converged=converged,
        n_iterations=len(history),
        wall_s=wall_s,
        final_loss=loss_value,
        initial_loss=initial_loss if initial_loss is not None else loss_value,
        params=param_values,
        history=history,
    )
