"""Density-based topology optimisation for LBM flow problems.

Implements the classic **SIMP** (Solid Isotropic Material with Penalisation)
topology optimisation method combined with the adjoint LBM sensitivity.  This
is analogous to the shape/topology optimisation capability in PowerFlow (design
sensitivity maps) and more advanced 3DS Tosca Fluid.

Method
------
The optimisation minimises (or maximises) a flow objective subject to a volume
fraction constraint:

    min  J(u, ρ_mat)
    s.t. Σ ρ_mat / V ≤ V_f     (volume fraction ≤ target)
         ρ_mat ∈ [0, 1]

where ρ_mat is the design variable (material density; 1 = solid, 0 = fluid).

The solid–fluid interface is represented by a **Brinkman penalisation** term
added to the LBM body force:

    F_Brinkman = −α(ρ_mat) u

    α(ρ_mat) = α_max q (1 − ρ_mat) / (q + ρ_mat)     (SIMP interpolation)

where q is a convexity parameter (typically 0.01–0.1).

Sensitivity computation
-----------------------
The design sensitivity dJ/dρ_mat is obtained via the **adjoint method**
(PyTorch autograd through the Brinkman-penalised equilibrium):

    dJ/dρ_mat = ∂J/∂α × dα/dρ_mat

The SIMP update is applied with the **optimality criteria (OC)** method:

    ρ_mat^(k+1) = ρ_mat^(k) × (−dJ/dρ_mat / λ)^η

where η = 0.5 is the step exponent and λ is a Lagrange multiplier found by
bisection to satisfy the volume constraint.

Filter
------
A density filter (convolution with a circular kernel of radius r_min) is
applied to prevent checkerboard instabilities:

    ρ_filtered = H * ρ_mat / Σ H

References
----------
Bendsøe, M. P. & Kikuchi, N. (1988). Generating optimal topologies in
    structural design using a homogenisation method. *CMAME* 71, 197–224.
Borrvall, T. & Petersson, J. (2003). Topology optimisation of fluids in
    Stokes flow. *Int. J. Numer. Methods Fluids* 41, 77–107.
Sigmund, O. (2007). Morphology-based black and white filters for topology
    optimisation. *Struct. Multidisc. Optim.* 33, 401–424.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "TopOptConfig",
    "TopOptResult",
    "brinkman_alpha",
    "density_filter",
    "compute_sensitivity",
    "oc_update",
    "run_topology_optimisation",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TopOptConfig:
    """Configuration for density-based topology optimisation."""
    # Grid
    nx: int = 80
    ny: int = 40
    # Volume fraction target
    vf_target: float = 0.4             # max fraction of domain that may be solid
    # Brinkman penalisation
    alpha_max: float = 2.5e4           # large penalisation for solid
    q_simp: float = 0.01               # SIMP convexity parameter
    # Density filter
    r_min: float = 2.0                 # filter radius in grid cells
    # Optimisation
    n_iter: int = 80
    oc_eta: float = 0.5                # OC move exponent
    move_limit: float = 0.2            # max change per iteration
    # Tolerance for convergence
    tol: float = 1e-3
    # Objective
    objective: str = "pressure_drop"   # "pressure_drop" | "flow_uniformity"
    # Re (used to set LBM τ)
    re: float = 100.0
    nu_lb: float | None = None         # if None, inferred from Re and nx


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TopOptResult:
    """Output of a topology optimisation run."""
    density: list[list[float]]         # final design (ny, nx) ∈ [0, 1]
    objective_history: list[float]     # objective value vs iteration
    volume_fraction_history: list[float]
    converged: bool
    n_iterations: int
    final_objective: float
    final_volume_fraction: float


# ---------------------------------------------------------------------------
# Brinkman penalisation
# ---------------------------------------------------------------------------

def brinkman_alpha(
    rho_mat: torch.Tensor,
    alpha_max: float = 2.5e4,
    q: float = 0.01,
) -> torch.Tensor:
    """SIMP Brinkman coefficient α(ρ_mat) [lattice units⁻¹].

    α → 0 for fluid (ρ_mat → 0); α → α_max for solid (ρ_mat → 1).
    """
    return alpha_max * q * (1.0 - rho_mat) / (q + rho_mat + 1e-12)


# ---------------------------------------------------------------------------
# Density filter
# ---------------------------------------------------------------------------

def density_filter(
    rho_mat: torch.Tensor,
    r_min: float = 2.0,
) -> torch.Tensor:
    """Apply a circular density filter to prevent checkerboard instability.

    Uses a 2-D convolution with a cone-shaped kernel of radius r_min.
    """
    # Build the kernel
    r = int(math.ceil(r_min))
    kernel_size = 2 * r + 1
    kernel = torch.zeros(kernel_size, kernel_size, dtype=rho_mat.dtype, device=rho_mat.device)
    cx, cy = r, r
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = math.sqrt((i - cx)**2 + (j - cy)**2)
            if dist <= r_min:
                kernel[i, j] = max(r_min - dist, 0.0)

    kernel = kernel / kernel.sum()
    k = kernel.unsqueeze(0).unsqueeze(0)

    rho_f = rho_mat.float().unsqueeze(0).unsqueeze(0)   # (1, 1, ny, nx)
    filtered = F.conv2d(rho_f, k.float(), padding=r)
    return filtered.squeeze().to(rho_mat.dtype)


# ---------------------------------------------------------------------------
# Sensitivity computation (adjoint via autograd)
# ---------------------------------------------------------------------------

def _lbm_objective_proxy(
    ux: torch.Tensor,
    uy: torch.Tensor,
    rho: torch.Tensor,
    rho_mat: torch.Tensor,
    alpha_fn,
    objective: str,
) -> torch.Tensor:
    """Differentiable proxy objective for autograd-based sensitivity.

    Returns a scalar objective value.
    """
    alpha = alpha_fn(rho_mat)

    # Brinkman drag: F = −α u, so power = Σ α |u|² (proxy for pressure drop)
    drag_proxy = (alpha * (ux**2 + uy**2)).sum()

    if objective == "pressure_drop":
        cs2 = 1.0 / 3.0
        p = cs2 * rho
        ny, nx = p.shape
        p_in  = p[:, :max(1, nx // 10)].mean()
        p_out = p[:, max(0, nx - nx // 10):].mean()
        return (p_in - p_out) + 0.01 * drag_proxy

    elif objective == "flow_uniformity":
        # Maximise flow uniformity at outlet: minimise CoV
        u_out = ux[:, -1]
        mean_u = u_out.mean().abs() + 1e-12
        cov = u_out.std() / mean_u
        return cov + 0.01 * drag_proxy

    # Default
    return drag_proxy


def compute_sensitivity(
    ux: torch.Tensor,
    uy: torch.Tensor,
    rho: torch.Tensor,
    rho_mat: torch.Tensor,
    cfg: TopOptConfig,
) -> torch.Tensor:
    """Compute dJ/dρ_mat via autograd through the Brinkman objective proxy.

    Returns a (ny, nx) sensitivity tensor.
    """
    rho_var = rho_mat.detach().requires_grad_(True).clone()
    alpha_fn = lambda r: brinkman_alpha(r, cfg.alpha_max, cfg.q_simp)   # noqa: E731

    J = _lbm_objective_proxy(ux, uy, rho, rho_var, alpha_fn, cfg.objective)
    J.backward()

    sens = rho_var.grad.detach().clone()
    return sens


# ---------------------------------------------------------------------------
# Optimality criteria update
# ---------------------------------------------------------------------------

def oc_update(
    rho_mat: torch.Tensor,
    sensitivity: torch.Tensor,
    vf_target: float,
    move: float = 0.2,
    eta: float = 0.5,
) -> torch.Tensor:
    """Optimality criteria (OC) density update with bisection for λ.

    Minimisation: ρ_new = ρ × (−sens / λ)^η
    """
    rho = rho_mat.clone()

    def _oc_apply(lam: float) -> torch.Tensor:
        ratio = (-sensitivity / (lam + 1e-30)) ** eta
        rho_new = rho * ratio
        rho_new = torch.clamp(rho_new, min=1e-3, max=1.0)
        rho_new = torch.clamp(rho_new, min=rho - move, max=rho + move)
        return rho_new

    # Bisection to find λ satisfying volume constraint
    lam_lo, lam_hi = 1e-30, 1e10
    for _ in range(50):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        rho_trial = _oc_apply(lam_mid)
        vf = rho_trial.mean().item()
        if vf > vf_target:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return _oc_apply(0.5 * (lam_lo + lam_hi))


# ---------------------------------------------------------------------------
# Simple LBM flow solver for optimisation
# ---------------------------------------------------------------------------

def _lbm_step_brinkman(
    f: torch.Tensor,
    rho_mat: torch.Tensor,
    tau: float,
    cfg: TopOptConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One BGK collision + streaming step with Brinkman penalisation."""
    from tensorlbm.d2q9 import equilibrium, macroscopic

    rho, ux, uy = macroscopic(f)

    # Brinkman body force: reduce velocity toward zero in solid regions
    alpha = brinkman_alpha(rho_mat, cfg.alpha_max, cfg.q_simp)
    denom = 1.0 + alpha / (1.0 / tau - 0.5)
    ux = ux / denom
    uy = uy / denom

    f_eq = equilibrium(rho, ux, uy)
    f_out = f - (f - f_eq) / tau

    # Streaming (roll each velocity direction)
    _CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    _CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    f_stream = torch.zeros_like(f_out)
    for q in range(9):
        f_stream[q] = torch.roll(
            torch.roll(f_out[q], _CX[q], dims=1),
            _CY[q], dims=0,
        )

    return f_stream, rho, ux, uy


def _run_lbm_for_topopt(
    rho_mat: torch.Tensor,
    cfg: TopOptConfig,
    n_steps: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a short LBM simulation with Brinkman penalisation.

    Returns (rho, ux, uy) at the final state.
    """
    ny, nx = cfg.ny, cfg.nx
    if cfg.nu_lb is not None:
        nu_lb = cfg.nu_lb
    else:
        u_lb = 0.05   # lattice inlet velocity
        nu_lb = u_lb * nx / cfg.re
        nu_lb = max(min(nu_lb, 0.3), 1e-4)

    tau = 0.5 + 3.0 * nu_lb

    # Initialise with uniform flow
    f = torch.zeros(9, ny, nx, dtype=torch.float64)
    f[0] = 1.0 / 9.0
    rho_init = torch.ones(ny, nx, dtype=torch.float64)
    u_inlet = 0.05
    ux_init = torch.full((ny, nx), u_inlet, dtype=torch.float64)
    uy_init = torch.zeros(ny, nx, dtype=torch.float64)
    from tensorlbm.d2q9 import equilibrium
    f = equilibrium(rho_init, ux_init, uy_init)

    for _ in range(n_steps):
        f, rho, ux, uy = _lbm_step_brinkman(f, rho_mat.detach(), tau, cfg)

        # Zou-He inlet BC (left)
        ux[:, 0] = u_inlet
        uy[:, 0] = 0.0

        # Outflow BC (right) – copy
        f[:, :, -1] = f[:, :, -2]

        # No-slip top/bottom (periodic-like)
        ux[0, :] = 0.0
        ux[-1, :] = 0.0
        uy[0, :] = 0.0
        uy[-1, :] = 0.0

    return rho, ux, uy


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def run_topology_optimisation(cfg: TopOptConfig) -> TopOptResult:
    """Run the full density-based topology optimisation.

    Uses a simplified Brinkman-LBM solver coupled with OC updates.
    """
    ny, nx = cfg.ny, cfg.nx

    # Initialise density uniformly at volume fraction target
    rho_mat = torch.full((ny, nx), cfg.vf_target, dtype=torch.float64)

    obj_history: list[float] = []
    vf_history: list[float] = []
    converged = False

    for it in range(cfg.n_iter):
        # 1. Run LBM flow
        rho, ux, uy = _run_lbm_for_topopt(rho_mat, cfg)

        # 2. Apply density filter
        rho_filt = density_filter(rho_mat, cfg.r_min)

        # 3. Compute objective
        alpha_fn = lambda r: brinkman_alpha(r, cfg.alpha_max, cfg.q_simp)   # noqa: E731
        rho_var = rho_filt.requires_grad_(True)
        J = _lbm_objective_proxy(ux, uy, rho, rho_var, alpha_fn, cfg.objective)
        J.backward()
        sens = rho_var.grad.detach()

        obj_val = float(J.detach())
        vf_val = float(rho_mat.mean())
        obj_history.append(obj_val)
        vf_history.append(vf_val)

        # 4. Filter sensitivity
        sens_filt = density_filter(sens, cfg.r_min)

        # 5. OC update
        rho_new = oc_update(rho_mat, sens_filt, cfg.vf_target, cfg.move_limit, cfg.oc_eta)

        # 6. Check convergence
        change = (rho_new - rho_mat).abs().max().item()
        rho_mat = rho_new

        if it > 5 and change < cfg.tol:
            converged = True
            break

    return TopOptResult(
        density=rho_mat.tolist(),
        objective_history=obj_history,
        volume_fraction_history=vf_history,
        converged=converged,
        n_iterations=len(obj_history),
        final_objective=obj_history[-1] if obj_history else 0.0,
        final_volume_fraction=vf_history[-1] if vf_history else 0.0,
    )
