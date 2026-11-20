"""Adjoint sensitivity analysis for TensorLBM -- FROZEN-FIELD SURROGATE ONLY.

.. warning::

   **Verdict GAP-DIFFLBM-ADJOINT-VERDICT (2026-11-14): this module is a
   frozen-field surrogate and must NOT be used to drive a shape optimisation.**
   Every result it returns is flagged ``shape_derivative_only=True`` and is
   rejected by :func:`optimization_path_eligible`.

   Measured on the reference case of
   ``benchmarks/adjoint_verdict_diff_lbm.py`` (plane channel, SDF disk,
   Re = 40, objective = lift, parameter = disk centre ``y_c``):

   ===========================  =========  ==========================
   quantity                     value      meaning
   ===========================  =========  ==========================
   cos(this module, unrolled AD)  -1.000    exactly anti-parallel
   |g_surrogate| / |g_true|        2.289   amplitude also wrong
   optimisation step (y_c + 2)    21.0 -> 22.5   target moved the wrong way
   same step with the true grad.  21.0 -> 19.5   (``C_L`` values, scaled)
   ===========================  =========  ==========================

   The reason is structural, not numerical: this module differentiates the
   *objective only*, on a flow field produced by an ordinary forward solve, so
   the derivative contains the explicit (mask/metric) term of the shape
   derivative but **not the flow response** ``∂J/∂f · ∂f/∂y_c``.  For lift the
   flow response is not a correction, it is the whole signal (the wake and the
   wall-induced pressure asymmetry *are* the lift), which is why the surrogate
   comes out anti-parallel.

   Use :func:`tensorlbm.shape_gradient.shape_gradient_through_flow` instead:
   it keeps the geometric parameter inside the autograd graph of the LBM step
   chain and returns ``shape_derivative_only=False`` /
   ``includes_flow_response=True`` (verified against finite differences of the
   converged solver in the same benchmark).

What this module still is good for
----------------------------------
A cheap, mesh-agnostic *diagnostic*: "which boundary nodes does the objective
feel most strongly", i.e. a distribution/attention map over a frozen snapshot.
It answers no question of the form "how would the flow have evolved".

Implementation note
-------------------
The sensitivity is obtained with PyTorch autograd applied to the objective
functional of the *frozen* fields ``(rho, ux, uy)`` and propagated to the
boundary nodes with a central field difference; ``perturbation_scale`` is a
uniform factor on that product.  Because a uniform factor cannot change a
cosine similarity, the recorded verdict does not depend on its value.

Objectives supported:
  - drag coefficient C_D
  - lift coefficient C_L
  - total pressure loss
  - mixing uniformity index

Usage
-----
>>> from tensorlbm.adjoint import adjoint_sensitivity
>>> result = adjoint_sensitivity(rho, ux, uy, obstacle_mask, objective="drag")
>>> result["shape_derivative_only"]
True
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

ObjectiveType = Literal["drag", "lift", "pressure_loss", "mixing_uniformity"]

#: Contract flag of the whole module (GAP-DIFFLBM-ADJOINT-VERDICT).
SHAPE_DERIVATIVE_ONLY: bool = True

#: Measured evidence behind :data:`SHAPE_DERIVATIVE_ONLY`
#: (``benchmarks/adjoint_verdict_diff_lbm.py``, artifact
#: ``artifacts/adjoint_verdict_diff_lbm.json``).
FROZEN_FIELD_VERDICT: dict = {
    "anchor": "GAP-DIFFLBM-ADJOINT-VERDICT",
    "objective": "lift",
    "parameter": "y_c (disk centre, transverse)",
    "cos_vs_unrolled_ad": -1.000,
    "amplitude_ratio_vs_true": 2.289,
    "optimization_trajectory_reversed": True,
    "shape_derivative_only": True,
    "includes_flow_response": False,
    "optimization_path_eligible": False,
    "replacement": "tensorlbm.shape_gradient.shape_gradient_through_flow",
}


def optimization_path_eligible(result: dict | None = None) -> bool:
    """Gate: may this gradient enter a shape-optimisation loop?

    ``adjoint.py`` answers **False** by construction.  The gate is written so
    that any future gradient source has to *claim* both flags ---
    ``shape_derivative_only is False`` **and** ``includes_flow_response is
    True`` --- before it is accepted, instead of being trusted by default::

        >>> optimization_path_eligible(adjoint_sensitivity(rho, ux, uy, mask))
        False
    """
    if result is None:
        return False
    return bool(
        result.get("shape_derivative_only") is False
        and result.get("includes_flow_response") is True
    )


# ---------------------------------------------------------------------------
# Objective functions (differentiable w.r.t. field tensors)
# ---------------------------------------------------------------------------


def _objective_drag(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    mask: torch.Tensor,
    cs2: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Drag proxy: integral of pressure × surface-normal-x over obstacle."""
    p = cs2 * (rho - rho.mean())
    # surface cells: solid neighboured by fluid in x-direction
    surf_right = mask & (~F.pad(mask, (0, 1, 0, 0))[:, 1:])
    surf_left = mask & (~F.pad(mask, (1, 0, 0, 0))[:, :-1])
    drag = (p * surf_right.float()).sum() - (p * surf_left.float()).sum()
    return drag


def _objective_lift(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    mask: torch.Tensor,
    cs2: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Lift proxy: integral of pressure × surface-normal-y over obstacle."""
    p = cs2 * (rho - rho.mean())
    # roll is periodic; the top/bottom rows are one-sided in a bounded domain
    surf_top = mask & ~(torch.roll(mask, -1, dims=0))
    surf_bottom = mask & ~(torch.roll(mask, 1, dims=0))
    lift = (p * surf_top.float()).sum() - (p * surf_bottom.float()).sum()
    return lift


def _objective_pressure_loss(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    mask: torch.Tensor,
    cs2: float = 1.0 / 3.0,
) -> torch.Tensor:
    """Total pressure loss: inlet stagnation pressure minus outlet mean."""
    ny, nx = rho.shape
    p = cs2 * rho
    p_in = p[:, : max(1, nx // 10)].mean()
    p_out = p[:, min(nx - nx // 10, nx - 1) :].mean()
    return p_in - p_out


def _objective_mixing(
    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, mask: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Mixing uniformity: negative coefficient of variation of |u| at outlet."""
    ny, nx = rho.shape
    u_mag = torch.sqrt(ux**2 + uy**2 + 1e-12)
    outlet = u_mag[:, -max(1, nx // 10) :]
    cv = outlet.std() / (outlet.mean() + 1e-12)
    return cv  # minimise → negative objective


_OBJECTIVES = {
    "drag": _objective_drag,
    "lift": _objective_lift,
    "pressure_loss": _objective_pressure_loss,
    "mixing_uniformity": _objective_mixing,
}


# ---------------------------------------------------------------------------
# Boundary node extraction
# ---------------------------------------------------------------------------


def _extract_boundary_nodes(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return y, x indices of obstacle surface nodes."""
    # surface = solid & has ≥1 fluid neighbour
    fluid = (~mask).float()
    neigh = (
        torch.roll(fluid, 1, 0)
        + torch.roll(fluid, -1, 0)
        + torch.roll(fluid, 1, 1)
        + torch.roll(fluid, -1, 1)
    )
    surface = mask & (neigh > 0)
    ys, xs = torch.where(surface)
    return ys, xs


# ---------------------------------------------------------------------------
# Adjoint sensitivity
# ---------------------------------------------------------------------------


def adjoint_sensitivity(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    obstacle_mask: torch.Tensor,
    objective: ObjectiveType = "drag",
    perturbation_scale: float = 1e-3,
    finite_diff_check: bool = False,
) -> dict:
    """Frozen-field shape sensitivity (``shape_derivative_only=True``).

    .. warning::

       **Diagnostic only -- not a shape derivative.**  The returned
       ``sensitivity_x`` / ``sensitivity_y`` are ``∂J/∂(x_i, y_i)`` computed
       with the flow field *frozen*; the flow response ``∂J/∂f · ∂f/∂y_c`` is
       absent, and for a lift objective that term dominates and even flips the
       sign (measured cos = -1.000 against the unrolled AD gradient, see
       :data:`FROZEN_FIELD_VERDICT`).  The result therefore carries
       ``shape_derivative_only=True`` and ``optimization_path_eligible=False``;
       :func:`optimization_path_eligible` rejects it, and a shape-optimisation
       loop must use :func:`tensorlbm.shape_gradient.shape_gradient_through_flow`.

    For each boundary node the gradient ∂J/∂(x_i, y_i) is estimated by
    computing ∂J/∂rho · ∂rho/∂x_i + ∂J/∂u · ∂u/∂x_i through the AD graph
    evaluated on the frozen fields.

    Parameters
    ----------
    rho, ux, uy : Tensor
        Flow fields (2-D, same shape).
    obstacle_mask : Tensor
        Boolean mask (True = solid).
    objective : str
        Objective function key.
    perturbation_scale : float
        Scale of implicit surface perturbation.
    finite_diff_check : bool
        If True, also return a finite-difference verification for a subset.

    Returns
    -------
    dict with keys:
        shape_derivative_only : bool  – always ``True`` (no flow response)
        includes_flow_response : bool – always ``False``
        optimization_path_eligible : bool – always ``False``; see the gate
        frozen_field_verdict : dict – measured evidence for the flag
        objective_value : float
        sensitivity_x : list[float]  – ∂J/∂x for each boundary node
        sensitivity_y : list[float]  – ∂J/∂y for each boundary node
        node_x : list[int]
        node_y : list[int]
        most_sensitive_node : dict
        gradient_norm : float
    """
    obj_fn = _OBJECTIVES.get(objective)
    if obj_fn is None:
        raise ValueError(f"Unknown objective '{objective}'. Choose from {list(_OBJECTIVES)}")

    # Make fields differentiable
    rho_d = rho.clone().detach().requires_grad_(True)
    ux_d = ux.clone().detach().requires_grad_(True)
    uy_d = uy.clone().detach().requires_grad_(True)

    J = obj_fn(rho_d, ux_d, uy_d, obstacle_mask)
    J_val = float(J.item())

    # Compute gradients w.r.t. fields
    J.backward()

    dJ_drho = rho_d.grad  # (ny, nx)
    dJ_dux = ux_d.grad
    dJ_duy = uy_d.grad

    if dJ_drho is None:
        dJ_drho = torch.zeros_like(rho)
    if dJ_dux is None:
        dJ_dux = torch.zeros_like(ux)
    if dJ_duy is None:
        dJ_duy = torch.zeros_like(uy)

    # Boundary nodes
    ys, xs = _extract_boundary_nodes(obstacle_mask)
    n_nodes = len(ys)

    if n_nodes == 0:
        return {
            "objective_value": J_val,
            "objective": objective,
            "n_boundary_nodes": 0,
            "sensitivity_x": [],
            "sensitivity_y": [],
            "node_x": [],
            "node_y": [],
            "gradient_norm": 0.0,
            "most_sensitive_node": {},
            "shape_derivative_only": True,
            "includes_flow_response": False,
            "optimization_path_eligible": False,
        }

    # Shape sensitivity via implicit differentiation:
    # ∂J/∂x_i ≈ dJ/drho * (rho at x_i+1 - rho at x_i-1)/(2) * scale
    # This is the one-sided shape derivative in a lattice-unit sense.
    sens_x = []
    sens_y = []

    for i in range(n_nodes):
        y_i, x_i = int(ys[i]), int(xs[i])
        ny, nx = rho.shape
        # x-sensitivity: finite difference of field gradient at node
        rho_xp = float(rho[y_i, min(x_i + 1, nx - 1)])
        rho_xm = float(rho[y_i, max(x_i - 1, 0)])
        drho_dx = (rho_xp - rho_xm) / 2.0

        rho_yp = float(rho[min(y_i + 1, ny - 1), x_i])
        rho_ym = float(rho[max(y_i - 1, 0), x_i])
        drho_dy = (rho_yp - rho_ym) / 2.0

        dJ_rho_i = float(dJ_drho[y_i, x_i])

        sx = dJ_rho_i * drho_dx * perturbation_scale
        sy = dJ_rho_i * drho_dy * perturbation_scale

        # Add velocity gradient contribution
        ux_xp = float(ux[y_i, min(x_i + 1, nx - 1)])
        ux_xm = float(ux[y_i, max(x_i - 1, 0)])
        sx += float(dJ_dux[y_i, x_i]) * (ux_xp - ux_xm) / 2.0 * perturbation_scale

        uy_yp = float(uy[min(y_i + 1, ny - 1), x_i])
        uy_ym = float(uy[max(y_i - 1, 0), x_i])
        sy += float(dJ_duy[y_i, x_i]) * (uy_yp - uy_ym) / 2.0 * perturbation_scale

        sens_x.append(sx)
        sens_y.append(sy)

    grad_norm = float((sum(s**2 for s in sens_x) + sum(s**2 for s in sens_y)) ** 0.5)

    # Most sensitive node
    magnitudes = [math.sqrt(sx**2 + sy**2) for sx, sy in zip(sens_x, sens_y)]
    if magnitudes:
        idx_max = max(range(len(magnitudes)), key=lambda i: magnitudes[i])
        most_sensitive = {
            "node_x": int(xs[idx_max]),
            "node_y": int(ys[idx_max]),
            "sensitivity_magnitude": magnitudes[idx_max],
            "sx": sens_x[idx_max],
            "sy": sens_y[idx_max],
        }
    else:
        most_sensitive = {}

    result = {
        "objective_value": J_val,
        "objective": objective,
        "n_boundary_nodes": n_nodes,
        "sensitivity_x": sens_x,
        "sensitivity_y": sens_y,
        "node_x": xs.tolist(),
        "node_y": ys.tolist(),
        "gradient_norm": grad_norm,
        "most_sensitive_node": most_sensitive,
        "shape_derivative_only": True,
        "includes_flow_response": False,
        "optimization_path_eligible": False,
        "frozen_field_verdict": dict(FROZEN_FIELD_VERDICT),
    }

    if finite_diff_check and n_nodes > 0:
        # Verify first node with FD
        i_check = 0
        y_c, x_c = int(ys[i_check]), int(xs[i_check])
        eps = perturbation_scale
        rho_plus = rho.clone()
        rho_plus[y_c, min(x_c + 1, rho.shape[1] - 1)] += eps
        J_plus = float(obj_fn(rho_plus.detach(), ux.detach(), uy.detach(), obstacle_mask))
        fd_grad = (J_plus - J_val) / eps
        result["fd_check"] = {
            "node": (x_c, y_c),
            "adjoint_sx": sens_x[i_check],
            "fd_approx": fd_grad * perturbation_scale,
        }

    return result
