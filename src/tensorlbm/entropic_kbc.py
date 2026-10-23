"""Complete entropic KBC (Karlin–Bösch–Chikatamarla) collision for D3Q19 / D3Q27.

This module implements the *full* H-theorem / entropy-minimisation KBC collision,
upgrading the previous KBC-style blend in :mod:`advanced_collision` (which used a
caller-supplied ``beta``) to a per-cell nonlinear entropy solve.

Key components
---------------
1. **Discrete entropy functional** ``H(f) = Σ_i f_i ln(f_i / w_i)``
2. **KBC decomposition** ``f = f_eq + k + s + h``
   * *s* — second-order deviatoric (shear / traceless) stress projection
   * *k* — second-order trace (bulk) projection  (kinetic ghost mode)
   * *h* — higher-order residual (third-order and above)
3. **Entropy-condition beta-solve** — minimise ``H(f_eq + (1 − 1/τ)·s + β·h)``
   over the admissibility domain intersected with ``β ∈ [0, 1]``
4. **Positivity / admissibility-domain** enforcement along the solved direction
5. **Per-cell nonlinear bisection** solver (vectorised over all cells)

Post-collision state (Karlin–Bösch–Chikatamarla construction)::

    f* = f_eq + (1 − 1/τ)·s + β·h

where *k* is fully relaxed (removed), the shear projection *s* is relaxed by the
exact BGK factor ``(1 − 1/τ)``, and the higher-order ghost residual *h* is
relaxed by the per-cell entropy-optimal ``β``.  The shear viscosity is therefore
controlled *exactly* by *τ*: ``nu = c_s^2 (τ − 1/2)``, identical to BGK.  The
entropy condition only fixes the non-hydrodynamic ghost relaxation ``β``, which
cannot feed back into the shear stress because *h* carries no second-order
moments.  Restricting ``β`` to ``[0, 1]`` and to the positivity domain keeps
``H(f*) ≤ H(f)`` (H-theorem behaviour) without disturbing the viscosity.

References
----------
Karlin, Bösch, Chikatamarla (2014).
*Multi-relaxation time lattice Boltzmann model for high Reynolds number flows.*
"""

from __future__ import annotations

import functools

import torch

from .d3q19 import W_EXACT64 as W19
from .d3q19 import C as C19
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import W_EXACT64 as W27
from .d3q27 import C as C27
from .d3q27 import equilibrium27, macroscopic27

_CS2 = 1.0 / 3.0  # lattice speed of sound squared
_FACTOR = 9.0 / 2.0  # Hermite projection factor = 1 / (2 * cs^4)


# ---------------------------------------------------------------------------
# 1. Discrete entropy functional
# ---------------------------------------------------------------------------


def discrete_entropy(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute the discrete Boltzmann entropy ``H(f) = Σ_i f_i ln(f_i / w_i)``.

    Args:
        f:  Distribution tensor of shape ``(Q, nz, ny, nx)``.
        w:  Lattice weights of shape ``(Q, 1, 1, 1)`` (or broadcastable).

    Returns:
        Per-cell entropy tensor of shape ``(nz, ny, nx)``.
    """
    f_safe = torch.clamp(f, min=1e-30)
    return (f_safe * torch.log(f_safe / w)).sum(dim=0)


# ---------------------------------------------------------------------------
# 2. KBC decomposition helpers
# ---------------------------------------------------------------------------


@functools.cache
def _lattice_constants(
    c_tensor: torch.Tensor,
    w_tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Pre-compute lattice views and Hermite basis tensors (cached)."""
    c = c_tensor.to(device=device, dtype=dtype if dtype.is_floating_point else torch.float32)
    w = w_tensor.to(device=device, dtype=dtype if dtype.is_floating_point else torch.float32)
    Q = c.shape[0]
    cx = c[:, 0].view(Q, 1, 1, 1)
    cy = c[:, 1].view(Q, 1, 1, 1)
    cz = c[:, 2].view(Q, 1, 1, 1)
    w_v = w.view(Q, 1, 1, 1)

    c_sq = cx * cx + cy * cy + cz * cz  # |c_i|^2, shape (Q, 1, 1, 1)

    # Full second-order Hermite basis: H_αβ = c_α c_β - cs^2 δ_αβ
    H_xx = cx * cx - _CS2
    H_yy = cy * cy - _CS2
    H_zz = cz * cz - _CS2
    H_xy = cx * cy
    H_xz = cx * cz
    H_yz = cy * cz

    # Deviatoric (traceless) Hermite basis: H_dev_αβ = c_α c_β - (1/3) |c|^2 δ_αβ
    Hd_xx = cx * cx - c_sq / 3.0
    Hd_yy = cy * cy - c_sq / 3.0
    Hd_zz = cz * cz - c_sq / 3.0
    # Off-diagonal components are the same (already traceless)
    Hd_xy = H_xy
    Hd_xz = H_xz
    Hd_yz = H_yz

    return {
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "w": w_v,
        "c_sq": c_sq,
        "H_xx": H_xx,
        "H_yy": H_yy,
        "H_zz": H_zz,
        "H_xy": H_xy,
        "H_xz": H_xz,
        "H_yz": H_yz,
        "Hd_xx": Hd_xx,
        "Hd_yy": Hd_yy,
        "Hd_zz": Hd_zz,
        "Hd_xy": Hd_xy,
        "Hd_xz": Hd_xz,
        "Hd_yz": Hd_yz,
    }


def _kbc_decompose(
    f_neq: torch.Tensor,
    p: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose ``f_neq`` into ``(s, k, h)``: shear, kinetic (bulk), higher-order.

    * **s** — second-order deviatoric (traceless) stress projection
    * **k** — second-order trace (bulk) projection
    * **h** — residual higher-order (third-order and above)

    ``s + k + h = f_neq`` exactly.
    """
    cx, cy, cz, w = p["cx"], p["cy"], p["cz"], p["w"]

    # Second-order stress tensor Π_αβ = Σ_i c_iα c_iβ f_neq,i
    pi_xx = (cx * cx * f_neq).sum(0)
    pi_yy = (cy * cy * f_neq).sum(0)
    pi_zz = (cz * cz * f_neq).sum(0)
    pi_xy = (cx * cy * f_neq).sum(0)
    pi_xz = (cx * cz * f_neq).sum(0)
    pi_yz = (cy * cz * f_neq).sum(0)

    pi_tr = pi_xx + pi_yy + pi_zz

    # Deviatoric (traceless) stress
    pi_dev_xx = pi_xx - pi_tr / 3.0
    pi_dev_yy = pi_yy - pi_tr / 3.0
    pi_dev_zz = pi_zz - pi_tr / 3.0
    # Off-diagonal unchanged
    pi_dev_xy = pi_xy
    pi_dev_xz = pi_xz
    pi_dev_yz = pi_yz

    # Shear (deviatoric) projection: s = (9/2) w [Hd · Π_dev]
    s = (
        _FACTOR
        * w
        * (
            p["Hd_xx"] * pi_dev_xx.unsqueeze(0)
            + p["Hd_yy"] * pi_dev_yy.unsqueeze(0)
            + p["Hd_zz"] * pi_dev_zz.unsqueeze(0)
            + 2.0 * p["Hd_xy"] * pi_dev_xy.unsqueeze(0)
            + 2.0 * p["Hd_xz"] * pi_dev_xz.unsqueeze(0)
            + 2.0 * p["Hd_yz"] * pi_dev_yz.unsqueeze(0)
        )
    )

    # Full second-order projection: f_neq^(2) = (9/2) w [H · Π]
    f_neq_2 = (
        _FACTOR
        * w
        * (
            p["H_xx"] * pi_xx.unsqueeze(0)
            + p["H_yy"] * pi_yy.unsqueeze(0)
            + p["H_zz"] * pi_zz.unsqueeze(0)
            + 2.0 * p["H_xy"] * pi_xy.unsqueeze(0)
            + 2.0 * p["H_xz"] * pi_xz.unsqueeze(0)
            + 2.0 * p["H_yz"] * pi_yz.unsqueeze(0)
        )
    )

    # Kinetic (bulk/trace) part: k = f_neq^(2) - s
    k = f_neq_2 - s

    # Higher-order residual: h = f_neq - f_neq^(2)
    h = f_neq - f_neq_2

    return s, k, h


def kbc_decompose_d3q19(f_neq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KBC decomposition for D3Q19: ``f_neq → (s, k, h)``."""
    p = _lattice_constants(C19, W19, f_neq.device, f_neq.dtype)
    return _kbc_decompose(f_neq, p)


def kbc_decompose_d3q27(f_neq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KBC decomposition for D3Q27: ``f_neq → (s, k, h)``."""
    p = _lattice_constants(C27, W27, f_neq.device, f_neq.dtype)
    return _kbc_decompose(f_neq, p)


# ---------------------------------------------------------------------------
# 3. Entropy-condition 1-D minimisation solvers (vectorised bisection)
# ---------------------------------------------------------------------------


def solve_gamma_entropy(
    feq: torch.Tensor,
    s: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    gamma_init: torch.Tensor,
    max_iter: int = 28,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Solve for the per-cell entropy-optimal ``γ`` by bisection.

    Minimises ``H(f_eq + γ·s + h) = Σ_i (f_eq + γ·s + h)_i ln((f_eq + γ·s + h)_i / w_i)``.

    The entropy ``H`` is convex in ``γ`` (``d²H/dγ² = Σ s_i² / f_i > 0``),
    so the minimum is unique and bisection on ``dH/dγ`` converges.

    Args:
        feq:        Equilibrium distribution ``(Q, nz, ny, nx)``.
        s:          Shear (deviatoric) non-equilibrium ``(Q, nz, ny, nx)``.
        h:          Higher-order non-equilibrium ``(Q, nz, ny, nx)``.
        w:          Lattice weights ``(Q, 1, 1, 1)``.
        gamma_init: Initial guess per cell ``(nz, ny, nx)`` (typically ``1 − 1/τ``).
        max_iter:   Maximum bisection iterations.
        tol:        Convergence tolerance on ``|dH/dγ|``.

    Returns:
        Per-cell optimal ``γ`` tensor of shape ``(nz, ny, nx)``.
    """
    # f_base = f at γ=0: f_eq + h
    f_base = feq + h

    # --- Admissibility domain (positivity): f_i = f_base_i + γ·s_i > 0 ---
    # For s_i > 0:  γ > -f_base_i / s_i   → lower bound
    # For s_i < 0:  γ < -f_base_i / s_i   → upper bound
    # For s_i = 0:  no constraint
    eps_s = 1e-30
    s_safe = torch.where(s.abs() > eps_s, s, torch.full_like(s, eps_s))
    ratio = -f_base / s_safe  # (Q, nz, ny, nx)

    neg_inf = torch.full_like(gamma_init, -1e6)
    pos_inf = torch.full_like(gamma_init, 1e6)

    # Lower bound: max over Q of ratio where s > 0
    pos_mask = s > eps_s
    ratio_pos = torch.where(pos_mask, ratio, neg_inf.unsqueeze(0).expand_as(ratio))
    gamma_lower = ratio_pos.amax(dim=0)

    # Upper bound: min over Q of ratio where s < 0
    neg_mask = s < -eps_s
    ratio_neg = torch.where(neg_mask, ratio, pos_inf.unsqueeze(0).expand_as(ratio))
    gamma_upper = ratio_neg.amin(dim=0)

    # If no constraint from either side (all s ≈ 0), return gamma_init
    no_constraint = (gamma_lower <= -1e5) & (gamma_upper >= 1e5)
    gamma_lower = torch.where(no_constraint, gamma_init, gamma_lower)
    gamma_upper = torch.where(no_constraint, gamma_init, gamma_upper)

    # Clamp gamma_init to the admissibility domain [lower, upper]
    gamma_init = torch.clamp(gamma_init, gamma_lower, gamma_upper)
    # Guarantee lower < upper
    gamma_lower = torch.minimum(gamma_lower, gamma_upper - 1e-10)

    # --- Bisection on dH/dγ ---
    # dH/dγ = Σ_i s_i [1 + ln(f_i / w_i)]
    # At γ_lower: dH/dγ < 0 (need to increase γ)
    # At γ_upper: dH/dγ > 0 (need to decrease γ)
    for _ in range(max_iter):
        gamma_mid = 0.5 * (gamma_lower + gamma_upper)
        f_mid = feq + gamma_mid.unsqueeze(0) * s + h
        f_safe = torch.clamp(f_mid, min=1e-30)
        dH = (s * (1.0 + torch.log(f_safe / w))).sum(dim=0)

        # Convergence check
        if dH.abs().max().item() < tol:
            break

        # dH > 0 → minimum is to the left → shrink upper
        # dH ≤ 0 → minimum is to the right → shrink lower
        upper_mask = dH > 0
        gamma_upper = torch.where(upper_mask, gamma_mid, gamma_upper)
        gamma_lower = torch.where(~upper_mask, gamma_mid, gamma_lower)

    return 0.5 * (gamma_lower + gamma_upper)


def solve_beta_entropy(
    feq: torch.Tensor,
    s_relaxed: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    beta_init: torch.Tensor,
    max_iter: int = 28,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Solve for the per-cell entropy-optimal ghost coefficient ``β`` by bisection.

    Minimises ``H(f_eq + s_relaxed + β·h) = Σ_i (f_eq + s_relaxed + β·h)_i
    ln((f_eq + s_relaxed + β·h)_i / w_i)`` over the admissibility domain
    (population positivity along the ``h`` direction) intersected with the
    physical ghost-relaxation bracket ``β ∈ [0, 1]``.

    The entropy ``H`` is convex in ``β`` (``d²H/dβ² = Σ h_i² / f_i > 0``), so
    the minimum is unique and bisection on ``dH/dβ`` converges.  If the
    positivity domain does not intersect ``[0, 1]``, the positivity domain
    wins (admissibility has priority over the bracket).

    Convergence is measured on the bracket *width* (the ``β`` resolution),
    which is amplitude-independent.  A legacy ``|dH/dβ|`` criterion would be
    amplitude-dependent (``dH`` scales with the perturbation) and truncates
    the solve prematurely at small amplitudes, freezing ``β`` near the
    initial bracket midpoint and biasing the measured viscosity by up to
    ~2.3% per the W7-A shear-wave audit.

    Args:
        feq:        Equilibrium distribution ``(Q, nz, ny, nx)``.
        s_relaxed:  Fixed shear offset ``(1 − 1/τ)·s`` already applied ``(Q, nz, ny, nx)``.
        h:          Higher-order non-equilibrium direction ``(Q, nz, ny, nx)``.
        w:          Lattice weights ``(Q, 1, 1, 1)``.
        beta_init:  Initial guess per cell ``(nz, ny, nx)`` (typically ``clamp(1 − 1/τ, 0, 1)``).
        max_iter:   Maximum bisection iterations.
        tol:        Convergence tolerance on the bracket width (β resolution).

    Returns:
        Per-cell optimal ``β`` tensor of shape ``(nz, ny, nx)``, admissible
        (positive populations along the search ray) and — whenever the
        admissible interval allows — confined to ``[0, 1]``.
    """
    # f_base = f at β=0: f_eq + s_relaxed
    f_base = feq + s_relaxed

    # --- Admissibility domain (positivity): f_i = f_base_i + β·h_i > 0 ---
    eps_h = 1e-30
    h_safe = torch.where(h.abs() > eps_h, h, torch.full_like(h, eps_h))
    ratio = -f_base / h_safe  # (Q, nz, ny, nx)

    neg_inf = torch.full_like(beta_init, -1e6)
    pos_inf = torch.full_like(beta_init, 1e6)

    # Lower bound: max over Q of ratio where h > 0
    pos_mask = h > eps_h
    ratio_pos = torch.where(pos_mask, ratio, neg_inf.unsqueeze(0).expand_as(ratio))
    h_lower = ratio_pos.amax(dim=0)

    # Upper bound: min over Q of ratio where h < 0
    neg_mask = h < -eps_h
    ratio_neg = torch.where(neg_mask, ratio, pos_inf.unsqueeze(0).expand_as(ratio))
    h_upper = ratio_neg.amin(dim=0)

    # If no constraint from either side (all h ≈ 0), return beta_init
    no_constraint = (h_lower <= -1e5) & (h_upper >= 1e5)
    h_lower = torch.where(no_constraint, beta_init, h_lower)
    h_upper = torch.where(no_constraint, beta_init, h_upper)

    # --- Intersect the admissibility domain with the physical bracket [0, 1] ---
    zero = torch.zeros_like(h_lower)
    one = torch.ones_like(h_upper)
    beta_lower = torch.maximum(h_lower, zero)
    beta_upper = torch.minimum(h_upper, one)
    # If the intersection is empty, keep the full positivity domain
    empty_bracket = beta_lower > beta_upper
    beta_lower = torch.where(empty_bracket, h_lower, beta_lower)
    beta_upper = torch.where(empty_bracket, h_upper, beta_upper)

    # Clamp beta_init to the search interval
    beta_init = torch.clamp(beta_init, beta_lower, beta_upper)
    # Guarantee lower < upper
    beta_lower = torch.minimum(beta_lower, beta_upper - 1e-10)

    # --- Bisection on dH/dβ ---
    # dH/dβ = Σ_i h_i [1 + ln(f_i / w_i)]
    # At beta_lower: dH/dβ < 0 (need to increase β)
    # At beta_upper: dH/dβ > 0 (need to decrease β)
    # Convergence: bracket width (amplitude-independent β resolution).  An
    # absolute |dH| stop would truncate the solve at small perturbation
    # amplitudes because dH itself scales with the amplitude.
    for _ in range(max_iter):
        if float((beta_upper - beta_lower).max().item()) < tol:
            break

        beta_mid = 0.5 * (beta_lower + beta_upper)
        f_mid = feq + s_relaxed + beta_mid.unsqueeze(0) * h
        f_safe = torch.clamp(f_mid, min=1e-30)
        dH = (h * (1.0 + torch.log(f_safe / w))).sum(dim=0)

        # dH > 0 → minimum is to the left → shrink upper
        # dH ≤ 0 → minimum is to the right → shrink lower
        upper_mask = dH > 0
        beta_upper = torch.where(upper_mask, beta_mid, beta_upper)
        beta_lower = torch.where(~upper_mask, beta_mid, beta_lower)

    return 0.5 * (beta_lower + beta_upper)


# ---------------------------------------------------------------------------
# 4. Full entropic KBC collision operators
# ---------------------------------------------------------------------------


def collide_kbc_d3q19(
    f: torch.Tensor,
    tau: float,
    *,
    max_iter: int = 28,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Complete entropic KBC collision for D3Q19.

    Implements the full H-theorem / entropy-minimisation KBC collision
    (Karlin–Bösch–Chikatamarla construction — viscosity on the shear modes,
    entropy on the ghost modes):

    1. Decompose ``f = f_eq + k + s + h`` (kinetic / shear / higher-order)
    2. Remove *k* (fully relaxed kinetic ghost modes)
    3. Relax the shear projection *s* by the exact BGK factor:
       ``s* = (1 − 1/τ)·s`` — this pins the shear viscosity to
       ``nu = c_s^2 (τ − 1/2)`` exactly, independent of the entropy solve
    4. Find per-cell ``β = argmin H(f_eq + (1 − 1/τ)·s + β·h)`` restricted to
       ``[0, 1]`` and the positivity domain
    5. Post-collision: ``f* = f_eq + (1 − 1/τ)·s + β·h``

    Because *h* carries no second-order moments, the entropy-optimal ``β``
    cannot feed back into the viscous stress: the shear viscosity is set by
    *τ* alone, while the H-theorem behaviour (``H(f*) ≤ H(f)``) is retained
    through the ghost-mode entropy minimisation.

    Args:
        f:        Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau:      Shear relaxation time (τ > 0.5).
        max_iter: Maximum bisection iterations for the β-solve.
        tol:      Convergence tolerance on the bracket width (β resolution).

    Returns:
        Post-collision distribution of the same shape.
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")

    device = f.device
    dtype = f.dtype
    p = _lattice_constants(C19, W19, device, dtype)

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq

    s, k, h = _kbc_decompose(f_neq, p)

    # Shear retention factor: exact BGK relaxation sigma = 1 - 1/tau
    sigma = 1.0 - 1.0 / tau
    s_relaxed = sigma * s

    # Ghost retention initial guess: sigma clamped to the physical bracket [0, 1]
    beta_init = torch.full(
        rho.shape,
        min(max(sigma, 0.0), 1.0),
        device=device,
        dtype=dtype,
    )

    w = p["w"]
    beta = solve_beta_entropy(feq, s_relaxed, h, w, beta_init, max_iter=max_iter, tol=tol)

    # Post-collision: f* = f_eq + (1-1/tau)·s + β·h  (k removed, shear exact-τ)
    return feq + s_relaxed + beta.unsqueeze(0) * h


def collide_kbc_d3q27(
    f: torch.Tensor,
    tau: float,
    *,
    max_iter: int = 28,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Complete entropic KBC collision for D3Q27.

    Same algorithm as :func:`collide_kbc_d3q19` but for the 27-velocity lattice:
    ``f* = f_eq + (1 − 1/τ)·s + β·h`` with ``β = argmin H`` per cell, so the
    shear viscosity is ``nu = c_s^2 (τ − 1/2)`` exactly.

    Args:
        f:        Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau:      Shear relaxation time (τ > 0.5).
        max_iter: Maximum bisection iterations for the β-solve.
        tol:      Convergence tolerance on the bracket width (β resolution).

    Returns:
        Post-collision distribution of the same shape.
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")

    device = f.device
    dtype = f.dtype
    p = _lattice_constants(C27, W27, device, dtype)

    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq

    s, k, h = _kbc_decompose(f_neq, p)

    # Shear retention factor: exact BGK relaxation sigma = 1 - 1/tau
    sigma = 1.0 - 1.0 / tau
    s_relaxed = sigma * s

    # Ghost retention initial guess: sigma clamped to the physical bracket [0, 1]
    beta_init = torch.full(
        rho.shape,
        min(max(sigma, 0.0), 1.0),
        device=device,
        dtype=dtype,
    )

    w = p["w"]
    beta = solve_beta_entropy(feq, s_relaxed, h, w, beta_init, max_iter=max_iter, tol=tol)

    return feq + s_relaxed + beta.unsqueeze(0) * h


def _natural_kbc_gamma(
    f: torch.Tensor,
    feq: torch.Tensor,
    shear_delta: torch.Tensor,
    higher_delta: torch.Tensor,
    beta: float | torch.Tensor,
) -> torch.Tensor:
    """Approximate KBC gamma with an admissible positivity-domain clamp."""
    inverse_equilibrium = feq.clamp_min(1.0e-30).reciprocal()
    numerator = (shear_delta * higher_delta * inverse_equilibrium).sum(dim=0)
    denominator = (higher_delta.square() * inverse_equilibrium).sum(dim=0)
    gamma = torch.where(
        denominator > 1.0e-30,
        1.0 / beta - (2.0 - 1.0 / beta) * numerator / denominator.clamp_min(1.0e-30),
        torch.full_like(denominator, 2.0),
    )

    base = f - 2.0 * beta * shear_delta
    direction = -beta * higher_delta
    epsilon = torch.finfo(f.dtype).eps
    ratio = (epsilon - base) / torch.where(
        direction.abs() > 1.0e-30,
        direction,
        torch.ones_like(direction),
    )
    negative_infinity = torch.full_like(denominator, -torch.inf)
    positive_infinity = torch.full_like(denominator, torch.inf)
    lower = torch.where(
        direction > 1.0e-30,
        ratio,
        negative_infinity.unsqueeze(0),
    ).amax(dim=0)
    upper = torch.where(
        direction < -1.0e-30,
        ratio,
        positive_infinity.unsqueeze(0),
    ).amin(dim=0)
    return torch.minimum(torch.maximum(gamma, lower), upper)


def _collide_natural_kbc_d3q19_unchecked(
    f: torch.Tensor,
    tau: float | torch.Tensor,
) -> torch.Tensor:
    """Natural-KBC tensor kernel after caller-side contract validation.

    Keeping the relaxation time as a tensor-compatible argument allows a
    compiled executor to reuse one graph throughout a viscosity ramp instead
    of specializing and recompiling once per Python float value.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    shear_delta, bulk_delta, higher_delta = _kbc_decompose(
        f - feq,
        _lattice_constants(C19, W19, f.device, f.dtype),
    )
    remaining_delta = bulk_delta + higher_delta
    beta = 1.0 / (2.0 * tau)
    gamma = _natural_kbc_gamma(
        f,
        feq,
        shear_delta,
        remaining_delta,
        beta,
    )
    return f - 2.0 * beta * shear_delta - beta * gamma.unsqueeze(0) * remaining_delta


def collide_natural_kbc_d3q19(
    f: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Experimental natural-moment KBC collision with fixed shear viscosity.

    This follows ``f' = f - 2*beta*Delta_s - beta*gamma*Delta_h`` with
    ``beta = 1/(2*tau)``. The deviatoric second-order projection is the shear
    part; bulk and higher-order residuals form ``Delta_h``. The analytical
    entropic scalar-product approximation supplies gamma and is clamped to the
    population positivity interval.

    The kernel remains experimental until independent viscosity, entropy and
    flow benchmarks pass; it does not replace :func:`collide_kbc_d3q19`.
    """
    if not isinstance(f, torch.Tensor) or f.ndim != 4 or f.shape[0] != 19:
        raise ValueError("f must have shape (19,nz,ny,nx)")
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    return _collide_natural_kbc_d3q19_unchecked(f, tau)


__all__ = [
    "discrete_entropy",
    "kbc_decompose_d3q19",
    "kbc_decompose_d3q27",
    "solve_gamma_entropy",
    "solve_beta_entropy",
    "collide_kbc_d3q19",
    "collide_kbc_d3q27",
    "collide_natural_kbc_d3q19",
]
