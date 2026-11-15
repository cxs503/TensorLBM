"""Horizon-free discrete adjoint for the flow-response shape derivative.

Why this module exists
----------------------
``tensorlbm.adjoint`` is a *frozen-field* surrogate (``shape_derivative_only=True``)
and is blocked from optimisation (see the verdict ``GAP-DIFFLBM-ADJOINT-VERDICT``).
The admissible replacement is a *flow-response* derivative: the geometric
parameter stays inside the autograd graph of the LBM step chain.

``tensorlbm.shape_gradient.shape_gradient_through_flow`` delivers that by
back-propagating through the last ``n_unroll`` steps of a converged run (a
truncated BPTT / "unrolled discrete adjoint").  That answer depends on the
horizon ``n_unroll`` and, more importantly, it is verified only against finite
differences of *the same finite-horizon problem*.  This module adds a second,
structurally different algorithm for the *same* total derivative, so the
gradient is anchored twice:

* **no tape, no horizon** -- the adjoint is solved as a linear fixed-point
  problem in the converged state, ``(I - S_f)^T lambda = (dJ/df)^T`` with one
  LBM step map ``S``, by matrix-free GMRES (transposed Krylov);
* **implicit-function theorem** -- ``dJ/dp = dJ/dp|_f + lambda^T dS/dp`` where
  the *flow response* is carried entirely by ``lambda``.

Both routes agree with central finite differences of the *converged* solver in
``benchmarks/adjoint_verdict_fixed_point.py``; the finite-difference reference
is a different numerical route again, and the mirror symmetry of the channel
``J(y_c) = -J(H - y_c)`` / ``dJ/dy_c(y_c) = dJ/dy_c(H - y_c)`` is an analytic
invariant that no tuning can satisfy by accident.

Derivation
----------
Let ``f*`` be the converged state, i.e. the fixed point of one step,
``R(f, p) = f - S(f, p) = 0``, and ``J(f, p)`` the lift measured from that
state.  Differentiating ``J(f*(p), p)``::

    R_f f_p + R_p = 0          ->   f_p = -(I - S_f)^{-1} (-S_p) = (I - S_f)^{-1} S_p
    dJ/dp = J_p + J_f f_p      ->   dJ/dp = J_p + lambda^T S_p,
                                    (I - S_f)^T lambda = J_f^T

``lambda`` is the discrete adjoint state at the fixed point.  There is no
truncation in time and no stored tape: each Krylov mat-vec is one step plus one
reverse-mode sweep over that single step.

Contract
--------
The returned dict carries the same explicit flags as
:func:`tensorlbm.shape_gradient.shape_gradient_through_flow`
(``shape_derivative_only=False`` / ``includes_flow_response=True``) and its
``optimization_path_eligible`` is decided by
:func:`tensorlbm.adjoint.optimization_path_eligible` -- the single gate, not a
locally invented one.  Diagnostics (fixed-point residual, Krylov iteration
count and true relative residual of the solved system) travel with the result
so the caller can see whether the linear solve actually converged.

Usage
-----
>>> from tensorlbm.shape_gradient import make_channel, run_to_steady
>>> from tensorlbm.fixed_point_adjoint import fixed_point_adjoint
>>> geom = make_channel(nx=48, ny=32, diameter=6.0, center_y=17.0)
>>> steady = run_to_steady(geom, max_steps=4000)
>>> out = fixed_point_adjoint(steady)
>>> out["shape_derivative_only"], out["includes_flow_response"]
(False, True)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from .adjoint import optimization_path_eligible
from .shape_gradient import (
    ChannelDisk,
    SteadySolution,
    disk_fluid_weight,
    lift_from_probe,
    step,
)

__all__ = ["fixed_point_adjoint", "fixed_point_residual", "steady_lift"]


# ---------------------------------------------------------------------------
# state-space pieces of the fixed-point problem
# ---------------------------------------------------------------------------


def _parameter_value(geom: ChannelDisk, name: str) -> float:
    return {
        "y_c": float(geom.center_y),
        "x_c": float(geom.center_x),
        "diameter": float(geom.diameter),
    }[name]


def _geometry_kwargs(geom: ChannelDisk, live: dict[str, torch.Tensor]) -> dict:
    return {
        "center_y": live.get("y_c", float(geom.center_y)),
        "center_x": live.get("x_c", float(geom.center_x)),
        "diameter": live.get("diameter", float(geom.diameter)),
    }


def _solid_weight(geom: ChannelDisk, live: dict[str, torch.Tensor]) -> torch.Tensor:
    w = disk_fluid_weight(geom, **_geometry_kwargs(geom, live))
    return torch.where(geom.wall_mask(), torch.zeros_like(w), 1.0 - w)


def state_step(
    f: torch.Tensor, geom: ChannelDisk, live: dict[str, torch.Tensor]
) -> torch.Tensor:
    """One LBM step map ``S(f; p)`` (state to state)."""
    return step(f, geom, **_geometry_kwargs(geom, live))[0]


def state_lift(
    f: torch.Tensor, geom: ChannelDisk, live: dict[str, torch.Tensor], scale: float
) -> torch.Tensor:
    """Objective ``J(f; p) = C_L`` measured from the state ``f`` (one step ahead)."""
    _, probe = step(f, geom, **_geometry_kwargs(geom, live))
    return lift_from_probe(probe, _solid_weight(geom, live))[1] / scale


def steady_lift(steady: SteadySolution) -> float:
    """``C_L`` of the converged solution, the quantity finite differences use."""
    geom = steady.geom
    u_ref = geom.bulk_velocity()
    scale = 0.5 * u_ref * u_ref * float(geom.diameter)
    return float(state_lift(steady.f.detach(), geom, {}, scale))


def fixed_point_residual(steady: SteadySolution) -> float:
    """``max|f - S(f)|`` at the delivered state (how converged the anchor is)."""
    with torch.no_grad():
        return float((state_step(steady.f.detach(), steady.geom, {}) - steady.f).abs().max())


# ---------------------------------------------------------------------------
# the adjoint solve
# ---------------------------------------------------------------------------


def fixed_point_adjoint(
    steady: SteadySolution,
    *,
    differentiate: Sequence[str] = ("y_c",),
    u_ref: float | None = None,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int = 200,
    max_matvecs: int = 1000,
) -> dict:
    """Total (flow-response) shape derivative at the converged state.

    Solves ``(I - S_f)^T lambda = (dJ/df)^T`` matrix-free by GMRES and evaluates
    ``dJ/dp = dJ/dp|_f + lambda^T dS/dp`` for each requested parameter.

    Args:
        steady: converged solution (:func:`tensorlbm.shape_gradient.run_to_steady`).
        differentiate: subset of ``("y_c", "x_c", "diameter")``.
        u_ref: reference velocity of the ``C_L`` normalisation (default: the
            analytic bulk velocity of the empty channel).
        rtol, atol: Krylov stopping tolerances on the true residual.  Measured on
            the study case: rtol=1e-4 -> 157 mat-vecs, rtol=1e-6 -> 265 mat-vecs, and
            the resulting gradient moves by 0.3 % while the converged FD limit is
            reproduced to 8.5e-5 relative; 1e-8 buys nothing but wall time.
        restart, max_matvecs: GMRES restart length and mat-vec budget.

    Returns:
        dict with ``dJ_dy_c``, ``grad``, ``grad_vector`` (ordered as
        ``differentiate``), the objective value, the contract flags
        ``shape_derivative_only=False`` / ``includes_flow_response=True`` /
        ``optimization_path_eligible`` (decided by
        :func:`tensorlbm.adjoint.optimization_path_eligible`), the diagnostics
        ``fixed_point_residual``, ``krylov_iterations``,
        ``krylov_relative_residual`` and ``adjoint_norm``.
    """
    try:
        from scipy.sparse.linalg import LinearOperator, gmres
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise RuntimeError("scipy is required for the fixed-point adjoint") from exc

    geom = steady.geom
    dtype = geom.dtype
    u = u_ref if u_ref is not None else geom.bulk_velocity()
    scale = 0.5 * u * u * float(geom.diameter)

    valid = ("y_c", "x_c", "diameter")
    for key in differentiate:
        if key not in valid:
            raise ValueError(f"differentiate must be subset of {valid}, got {key!r}")
    differentiate = tuple(differentiate)

    f_star = steady.f.detach().clone()
    base_live: dict[str, torch.Tensor] = {
        key: torch.tensor(_parameter_value(geom, key), dtype=dtype) for key in valid
    }
    residual = float((state_step(f_star, geom, {}) - f_star).abs().max())
    objective = float(state_lift(f_star, geom, {}, scale))

    # -- right-hand side: adjoint of the objective w.r.t. the state ----------
    f_leaf = f_star.clone().requires_grad_(True)
    J = state_lift(f_leaf, geom, {}, scale)
    (j_f,) = torch.autograd.grad(J, f_leaf)

    # -- matrix-free operator M v = v - (S_f)^T v ----------------------------
    counters = {"matvec": 0}

    def matvec(v):
        v_t = torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=dtype).reshape(f_star.shape)
        f_lin = f_star.clone().requires_grad_(True)
        s = state_step(f_lin, geom, {})
        (s_t_v,) = torch.autograd.grad(s, f_lin, grad_outputs=v_t)
        counters["matvec"] += 1
        return (v_t - s_t_v).reshape(-1).numpy().astype(np.float64)

    n_dof = int(f_star.numel())
    op = LinearOperator((n_dof, n_dof), matvec=matvec, dtype=np.float64)
    rhs = j_f.reshape(-1).numpy().astype(np.float64)
    lam_flat, info = gmres(op, rhs, rtol=rtol, atol=atol, restart=restart, maxiter=max_matvecs)
    true_res = op.matvec(lam_flat) - rhs
    lam_rel = float(
        (true_res @ true_res) ** 0.5 / max((rhs @ rhs) ** 0.5, 1e-300)
    )
    lam = torch.as_tensor(np.asarray(lam_flat, dtype=np.float64), dtype=dtype).reshape(f_star.shape)

    # -- total derivative for each parameter --------------------------------
    grad: dict[str, float] = {}
    explicit: dict[str, float] = {}
    flow_response: dict[str, float] = {}
    for key in differentiate:
        live = dict(base_live)
        live[key] = base_live[key].clone().requires_grad_(True)
        p = live[key]
        # dJ/dp at fixed state (explicit + metric term)
        j_p = torch.autograd.grad(state_lift(f_star, geom, live, scale), p)[0]
        # lambda^T dS/dp (flow response)
        s_p = torch.autograd.grad(
            state_step(f_star, geom, live), p, grad_outputs=lam
        )[0]
        explicit[key] = float(j_p)
        flow_response[key] = float(s_p)
        grad[key] = float(j_p) + float(s_p)

    result = {
        "dJ_dy_c": grad.get("y_c"),
        "grad": grad,
        "grad_vector": [grad[key] for key in differentiate],
        "differentiate": list(differentiate),
        "objective": objective,
        "y_c": float(geom.center_y),
        "reynolds": geom.reynolds(),
        "explicit_term": explicit,
        "flow_response_term": flow_response,
        "shape_derivative_only": False,
        "includes_flow_response": True,
        "method": "fixed-point discrete adjoint (transposed Krylov, implicit-function theorem)",
        "fixed_point_residual": residual,
        "krylov_iterations": int(counters["matvec"]),
        "krylov_info": int(info),
        "krylov_relative_residual": lam_rel,
        "adjoint_norm": float(lam.norm()),
        "krylov_rtol": rtol,
    }
    result["optimization_path_eligible"] = optimization_path_eligible(result)
    return result


def cosine(a, b) -> float:
    """Cosine of two (possibly 1-element) gradient vectors."""
    a = list(a) if isinstance(a, (list, tuple)) else [a]
    b = list(b) if isinstance(b, (list, tuple)) else [b]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)