"""Entropic lattice Boltzmann method (ELBM) with an exact entropy condition.

The module implements the *entropic* (H-theorem based) lattice-Boltzmann
collision of Karlin, Ferrante & Öttinger (1999) and Ansumali & Karlin (2002)
for quadratically-minimal lattices, together with the entropy functionals it
relies on:

``H_G(f) = Σ_i f_i ln(f_i / w_i)``            (Gauss / Burg entropy, default)
``H_BE(f)= Σ_i [ f_i ln(f_i/w_i) - (f_i + w_i) ln(1 + f_i/w_i) ]``  (Bose-Einstein)

Both are convex in ``f`` on the positive octant, so the *entropic equilibrium*
(their constrained minimiser at fixed density and momentum) is unique.  It is
computed in two independent ways:

* :func:`entropic_equilibrium_gauss_d3q27` -- the exact closed-form tensor
  product equilibrium of Ansumali & Karlin (2002), specialised to D3Q27 (the
  lattice whose weights *are* a product of three 1-D D1Q3 weights, which is
  what makes a closed form possible);
* :func:`entropic_equilibrium_dual` -- a vectorised Newton solve of the
  4-unknown Lagrange-dual problem ``∇H = λ·(1, c)``, valid on any lattice and
  for both entropies.  It is the reference used by the test-suite to certify
  the closed form and it is the production path for D3Q19.

Collision (the ELBM part)
-------------------------
BGK relaxes with the rate ``1/τ``::

    f* = f + (1/τ) (f^eq - f),        ν = c_s² (τ - 1/2)

ELBM replaces the fixed rate by an entropy-conditioned one::

    f* = f + (α/τ) (f^eq - f),        H(f*) = H(f)

with ``α`` solved *per cell and per step* (Newton with a positivity bracket and
an ``α ≤ 2`` admissibility cap).  Consequences, all machine-checked in
``tests/test_elbm.py``:

* ``H`` never increases in the collision (``H(f*) ≤ H(f)``) and is exactly
  invariant under periodic streaming, hence the discrete H-theorem holds for
  the full stream+collide step with no time-step restriction -- this is the
  "unconditional stability" of ELBM;
* every population stays positive (the solve is bracketed by the positivity
  domain), so the scheme cannot produce negative densities at any ``τ``;
* in near-equilibrium states the entropy root of the quadratic entropy model
  is ``α/(τ) → 2``, i.e. ``α ≈ 2τ ≈ 1``: the BGK viscosity
  ``ν = c_s²(τ/α - 1/2)`` is recovered to ``O((α-1)²)`` and only *strongly*
  non-equilibrium cells (``α < 1``) receive extra dissipation.  ELBM is thus a
  natural implicit LES: no eddy-viscosity model is required for stability.

The effective viscosity of a collision is exposed through
:func:`effective_viscosity` so a solver can report the state-dependent
transport coefficient instead of assuming ``ν(α = 1)``.

References
----------
* S. Ansumali, I. V. Karlin, *Phys. Rev. E* 65, 056312 (2002) -- entropic
  equilibria for the D1Q3/D3Q27 lattices and the H-theorem.
* I. V. Karlin, A. Ferrante, H. C. Öttinger, *Europhys. Lett.* 47, 182 (1999)
  -- entropy condition and the discrete H-theorem.
* S. S. Chikatamarla, I. V. Karlin, *Phys. Rev. Lett.* 97, 190601 (2006) --
  H-theorem, Galilean invariance and the ``α ≤ 2`` admissibility bound.
"""

from __future__ import annotations

from typing import Literal

import torch

from .d3q19 import C as C19
from .d3q19 import W_EXACT64 as W19
from .d3q27 import C as C27
from .d3q27 import W_EXACT64 as W27

__all__ = [
    "ALPHA_CAP",
    "EntropyConditionInfo",
    "LATTICES",
    "KINDS",
    "bose_einstein_entropy",
    "collide_elbm_d3q19",
    "collide_elbm_d3q27",
    "effective_viscosity",
    "entropic_equilibrium_dual",
    "entropic_equilibrium_gauss_d3q27",
    "gauss_entropy",
    "relative_entropy",
    "solve_entropy_condition",
]

CS2 = 1.0 / 3.0
KINDS = ("gauss", "bose-einstein")
LATTICES = ("d3q19", "d3q27")

#: Admissibility cap of the entropic relaxation factor.
#:
#: Convention used here and in the whole module: ``f* = f + (α/τ)(f^eq - f)``,
#: so ``α = 1`` is *exactly* BGK (rate ``1/τ``) and ``α < 1`` under-relaxes,
#: i.e. ``ν(α) = c_s²(τ/α − 1/2) > ν_BGK``.  The classical ELBM papers write the
#: same collision as ``f* = f + α_lit β (f^eq − f)`` with ``β = 1/(2τ)`` and cap
#: ``α_lit ≤ 2``; the dictionary is ``α_lit = 2α``, so ``α_lit ≤ 2 ⇔ α ≤ 1``.
#:
#: The cap matters physically: along the line ``f → f^eq`` the discrete entropy
#: is *convex and symmetric to second order*, so the entropy root of a
#: near-equilibrium state sits at the mirror point (rate 2, i.e. ``α ≈ 2τ``),
#: which is **not** the BGK rate.  The entropy condition alone therefore cannot
#: fix the transport coefficient; the cap does.  With ``α ≤ 1``:
#:
#: * resolved (near-equilibrium) cells relax at ``α = 1`` ⇒ BGK viscosity is
#:   reproduced exactly and the H-condition is satisfied with slack;
#: * strongly non-equilibrium cells, where the entropy root drops below 1, are
#:   under-relaxed ⇒ extra dissipation ⇒ positivity and H-monotonicity are
#:   preserved at any ``τ``.  This is the ELBM stabilisation mechanism.
ALPHA_CAP = 1.0

#: Floor applied to populations before taking logarithms.
_FLOOR = 1e-30

#: Relative tolerance of the entropy-condition root (|H(f*) - H(f)| <= tol * (|H|+1)).
_ENTROPY_TOL = 1e-6


def _lattice(lattice: str) -> tuple[torch.Tensor, torch.Tensor]:
    if lattice == "d3q19":
        return C19, W19
    if lattice == "d3q27":
        return C27, W27
    raise ValueError(f"unsupported lattice {lattice!r}; expected one of {LATTICES}")


def _views(
    c: torch.Tensor, w: torch.Tensor, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = c.shape[0]
    cc = c.to(device=device, dtype=dtype).view(q, 3, 1, 1, 1)
    ww = w.to(device=device, dtype=dtype).view(q, 1, 1, 1)
    return cc, ww, c.to(device=device, dtype=torch.int64)


# ---------------------------------------------------------------------------
# 1. Entropy functionals
# ---------------------------------------------------------------------------


def gauss_entropy(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Discrete Gauss (Burg) entropy ``H = Σ_i f_i ln(f_i / w_i)`` per cell."""
    f_safe = torch.clamp(f, min=_FLOOR)
    return (f_safe * torch.log(f_safe / w)).sum(dim=0)


def bose_einstein_entropy(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Discrete Bose-Einstein entropy per cell.

    ``H = Σ_i [ f_i ln(f_i/w_i) - (f_i + w_i) ln(1 + f_i/w_i) ]``
    (Ansumali & Karlin 2002).  Convex in ``f``; its constrained minimiser is
    the Bose-Einstein distribution of :func:`entropic_equilibrium_dual`.
    """
    f_safe = torch.clamp(f, min=_FLOOR)
    x = f_safe / w
    return (f_safe * torch.log(x) - (f_safe + w) * torch.log1p(x)).sum(dim=0)


def relative_entropy(
    f: torch.Tensor, feq: torch.Tensor, *, floor: float = _FLOOR
) -> torch.Tensor:
    """Kullback-Leibler excess ``Σ_i f_i ln(f_i / f_i^eq)`` per cell (>= 0)."""
    f_safe = torch.clamp(f, min=floor)
    feq_safe = torch.clamp(feq, min=floor)
    return (f_safe * torch.log(f_safe / feq_safe)).sum(dim=0)


# ---------------------------------------------------------------------------
# 2. Entropic equilibria
# ---------------------------------------------------------------------------


def _axis_factors(u: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """1-D entropic (Gauss) factors ``g(+1), g(0), g(-1)`` for one axis.

    Solves the D1Q3 entropy minimisation in closed form.  With
    ``S = sqrt(1 + 3u²)``, ``s = (S - 2u)/(1 + u)`` and ``P = 6/(4 + s + 1/s)``
    the three factors are ``g(c) = P s^{-c}``.  ``P`` is exactly the
    exponential prefactor ``e^{-θ}`` of the Lagrange multiplier.
    """
    s_squared = 1.0 + 3.0 * u * u
    s_root = torch.sqrt(s_squared)
    ratio = (s_root - 2.0 * u) / (1.0 + u)
    pref = 6.0 / (4.0 + ratio + 1.0 / ratio)
    return pref / ratio, pref, pref * ratio


def entropic_equilibrium_gauss_d3q27(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact closed-form Gauss-entropy equilibrium on D3Q27.

    ``f_i^eq = w_i ρ Π_α g_α(c_iα)`` with the per-axis factors of
    :func:`_axis_factors`.  The product form is exact on D3Q27 because its
    weights are the tensor product of three D1Q3 weight sets
    ``(2/3, 1/6, 1/6)``; it reproduces density and momentum exactly and is the
    unique constrained minimiser of :func:`gauss_entropy` (certified against
    :func:`entropic_equilibrium_dual` to <=1e-11 in the test-suite).

    Args:
        rho, ux, uy, uz: fields of shape ``(nz, ny, nx)``.
        out: optional pre-allocated ``(27, nz, ny, nx)`` buffer.

    Returns:
        Distribution of shape ``(27, nz, ny, nx)``.
    """
    if not (rho.shape == ux.shape == uy.shape == uz.shape):
        raise ValueError("rho/ux/uy/uz must share one shape")
    dtype = rho.dtype if rho.is_floating_point() else torch.float32
    device = rho.device
    c = C27.to(device=device, dtype=torch.int64)
    w = W27.to(device=device, dtype=dtype).view(27, 1, 1, 1)

    gx = _axis_factors(ux.to(dtype), dtype=dtype)
    gy = _axis_factors(uy.to(dtype), dtype=dtype)
    gz = _axis_factors(uz.to(dtype), dtype=dtype)

    # Broadcast-select the per-axis factor by velocity component.
    def _select(factors: tuple[torch.Tensor, ...], comp: torch.Tensor) -> torch.Tensor:
        cv = comp.view(27, 1, 1, 1)
        return torch.where(cv > 0, factors[0], torch.where(cv == 0, factors[1], factors[2]))

    fx = _select(gx, c[:, 0])
    fy = _select(gy, c[:, 1])
    fz = _select(gz, c[:, 2])
    result = w * rho.unsqueeze(0) * fx * fy * fz
    if out is not None:
        out.copy_(result)
        return out
    return result


def entropic_equilibrium_dual(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    *,
    lattice: str = "d3q27",
    entropy: Literal["gauss", "bose-einstein"] = "gauss",
    n_newton: int = 24,
    tol: float = 1e-13,
    theta_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entropic equilibrium by Newton iteration on the Lagrange dual.

    Stationarity of ``H`` subject to ``Σ f = ρ``, ``Σ c f = ρ u`` gives

    ``gauss``: ``f_i = w_i exp(-θ - a·c_i)``
    ``bose-einstein``: ``f_i = w_i / (exp(θ + a·c_i) - 1)``

    with a 4-vector of unknowns ``(θ, a_x, a_y, a_z)``.  The residual system
    ``R = (Σ f - ρ, Σ c f - ρ u)`` is solved by Newton; the Jacobian
    ``J = -Σ f g (1, c)(1, c)^T`` with ``g = 1`` (Gauss) or ``g = 1 + f/w``
    (Bose-Einstein) is symmetric positive definite, so the solve is stable.

    This path works on *any* lattice (D3Q19 included, whose weights are not a
    tensor product and therefore admit no product closed form) and for both
    entropies.  Its downside is cost, so the production D3Q27 collision uses
    the closed form.

    Args:
        rho, ux, uy, uz: fields of shape ``(nz, ny, nx)``.
        lattice: ``"d3q19"`` or ``"d3q27"``.
        entropy: ``"gauss"`` or ``"bose-einstein"``.
        n_newton: Newton iterations (fixed count keeps the kernel static).
        tol: informational only (no host sync / early exit).
        theta_init: optional ``(4, nz, ny, nx)`` warm start, e.g. the previous
            step's multipliers.

    Returns:
        Distribution of shape ``(Q, nz, ny, nx)``.
    """
    if entropy not in KINDS:
        raise ValueError(f"unsupported entropy {entropy!r}; expected one of {KINDS}")
    c, w = _lattice(lattice)
    dtype = rho.dtype if rho.is_floating_point() else torch.float32
    device = rho.device
    q = c.shape[0]
    cv = c.to(device=device, dtype=dtype).view(q, 1, 1, 1, 3)
    wv = w.to(device=device, dtype=dtype).view(q, 1, 1, 1)
    shape = tuple(rho.shape)

    if theta_init is None:
        theta = torch.empty((4,) + shape, dtype=dtype, device=device)
        if entropy == "gauss":
            # u = 0 solution: f = w exp(-θ) = w ρ  ->  θ = -ln ρ
            theta[0] = -torch.log(rho.to(dtype))
        else:
            # u = 0 solution: f = w/(exp(θ)-1) = w ρ -> θ = ln(1+ρ)
            theta[0] = torch.log1p(rho.to(dtype))
        theta[1] = 0.0
        theta[2] = 0.0
        theta[3] = 0.0
    else:
        theta = theta_init.to(dtype=dtype, device=device).clone()

    target = torch.stack([rho.to(dtype), rho.to(dtype) * ux.to(dtype),
                          rho.to(dtype) * uy.to(dtype), rho.to(dtype) * uz.to(dtype)])
    ones = torch.ones((q, 1, 1, 1), dtype=dtype, device=device)
    basis = torch.cat([ones.unsqueeze(-1), cv], dim=-1)  # (q,1,1,1,4)

    for _ in range(int(n_newton)):
        lin = theta[0].unsqueeze(0) + (cv * theta[1:].permute(1, 2, 3, 0).unsqueeze(0)).sum(dim=-1)
        if entropy == "gauss":
            f = wv * torch.exp(-lin)
            g = ones.expand_as(f)
        else:
            f = wv / torch.expm1(lin)
            g = 1.0 + f / wv
        moment = torch.stack([f.sum(dim=0)] + [(f * cv[..., j]).sum(dim=0) for j in range(3)])
        resid = moment - target
        # J_{ab} = -Σ f g b_a b_b ; solve  J Δ = -R  ->  Δ = (Σ f g b b^T)^{-1} R
        weight = (f * g).unsqueeze(-1) * basis  # (q,...,4)
        jac = torch.einsum("q...a,q...b->...ab", weight, basis)
        jac = jac + 1e-30 * torch.eye(4, dtype=dtype, device=device)
        # J = dR/dtheta = -jac, so Newton is theta <- theta + jac^{-1} R
        delta = torch.linalg.solve(jac, resid.permute(1, 2, 3, 0).unsqueeze(-1)).squeeze(-1)
        theta = theta + delta.permute(3, 0, 1, 2)
        if float(resid.abs().max()) < tol:
            break

    lin = theta[0].unsqueeze(0) + (cv * theta[1:].permute(1, 2, 3, 0).unsqueeze(0)).sum(dim=-1)
    if entropy == "gauss":
        return wv * torch.exp(-lin)
    return wv / torch.expm1(lin)


def equilibrium_dispatch(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    *,
    lattice: str = "d3q27",
    entropy: str = "gauss",
    n_newton: int = 6,
    theta_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cheapest available entropic equilibrium for the requested lattice."""
    if entropy == "gauss" and lattice == "d3q27":
        return entropic_equilibrium_gauss_d3q27(rho, ux, uy, uz)
    return entropic_equilibrium_dual(
        rho, ux, uy, uz, lattice=lattice, entropy=entropy,  # type: ignore[arg-type]
        n_newton=n_newton, theta_init=theta_init,
    )


# ---------------------------------------------------------------------------
# 3. Entropy condition: per-cell alpha solve
# ---------------------------------------------------------------------------


class EntropyConditionInfo:
    """Diagnostics of one :func:`solve_entropy_condition` call."""

    __slots__ = ("alpha", "residual", "hit_cap", "positivity_bound", "rate")

    def __init__(
        self,
        alpha: torch.Tensor,
        residual: torch.Tensor,
        hit_cap: torch.Tensor,
        positivity_bound: torch.Tensor,
        rate: torch.Tensor,
    ) -> None:
        self.alpha = alpha
        self.residual = residual
        self.hit_cap = hit_cap
        self.positivity_bound = positivity_bound
        self.rate = rate

    def summary(self) -> dict[str, float]:
        """Host-side scalars (one synchronisation) for logging/gates."""
        a = self.alpha.detach()
        return {
            "alpha_mean": float(a.mean()),
            "alpha_min": float(a.min()),
            "alpha_max": float(a.max()),
            "alpha_p01": float(a.flatten().kthvalue(max(1, a.numel() // 100)).values),
            "residual_max": float(self.residual.detach().abs().max()),
            "hit_cap_fraction": float(self.hit_cap.to(torch.float64).mean()),
            "positivity_limited_fraction": float(
                (self.positivity_bound < self.hit_cap).to(torch.float64).mean()
            ),
        }


def solve_entropy_condition(
    f: torch.Tensor,
    feq: torch.Tensor,
    tau: float | torch.Tensor,
    *,
    w: torch.Tensor,
    alpha_init: torch.Tensor | None = None,
    n_iter: int = 8,
    alpha_cap: float = ALPHA_CAP,
    floor: float = _FLOOR,
) -> EntropyConditionInfo:
    """Solve ``H(f + (α/τ)(f^eq - f)) = H(f)`` cell by cell.

    The map ``α ↦ H`` is convex along the straight line between ``f`` and
    ``f^eq`` (its second derivative is ``Σ_i (Δf_i)²/f_i > 0``), so the root is
    unique and a safeguarded Newton iteration with a positivity bracket
    converges quadratically.  ``α`` is clipped to ``[0, alpha_cap]``:

    * the upper bracket is ``min(alpha_cap, τ · t_pos)`` where ``t_pos`` is the
      largest rate keeping every population positive;
    * the trivial root ``α = 0`` is excluded by bracketing on the descending
      branch (``H`` decreases from ``α = 0`` up to the root).

    Args:
        f: pre-collision populations ``(Q, ...)``.
        feq: entropic equilibrium of ``f`` (same shape).
        tau: BGK relaxation time (scalar or per-cell tensor).
        w: lattice weights, shape ``(Q, 1, ...)`` or ``(Q,)``.
        alpha_init: optional per-cell warm start (e.g. previous step).
        n_iter: Newton iterations (fixed; the kernel stays static).
        alpha_cap: admissibility cap on ``α``.
        floor: positivity floor used for the logarithms.

    Returns:
        :class:`EntropyConditionInfo` with the per-cell ``α`` and diagnostics.
    """
    if f.shape != feq.shape:
        raise ValueError(f"f and feq shapes differ: {tuple(f.shape)} vs {tuple(feq.shape)}")
    shape = tuple(f.shape[1:])
    dtype = f.dtype
    out_dtype = f.dtype  # alpha/residual are returned in the caller's precision
    device = f.device
    wv = w.to(device=device, dtype=dtype)
    if wv.dim() == 1:
        wv = wv.view(-1, *([1] * len(shape)))

    delta = feq - f
    tau_t = torch.as_tensor(tau, dtype=dtype, device=device)

    # Entropy arithmetic in float64 even when the fields are float32: the
    # entropy-condition test compares H at the mirror point, where the exact
    # difference is zero to all orders in float32 (see the tolerance comment).
    if dtype == torch.float32:
        f = f.to(torch.float64)
        feq = feq.to(torch.float64)
        wv = wv.to(torch.float64)
        delta = delta.to(torch.float64)
        tau_t = tau_t.to(torch.float64)
        dtype = torch.float64

    # positivity bound on the rate t = α/τ (per cell)
    neg = delta < 0
    if bool(neg.any()):
        bound = torch.where(neg, -f / torch.where(neg, delta, torch.ones_like(delta)), torch.full_like(f, torch.inf))
        t_pos = bound.amin(dim=0)
    else:  # pragma: no cover - only when feq >= f everywhere
        t_pos = torch.full(shape, torch.inf, dtype=dtype, device=device)
    alpha_pos = tau_t * torch.clamp(t_pos, min=1e-12)
    alpha_hi = torch.minimum(alpha_pos * (1.0 - 1e-6), torch.full_like(alpha_pos, alpha_cap))
    alpha_hi = torch.clamp(alpha_hi, min=1e-9)

    f_floor = torch.clamp(f, min=floor)
    h0 = (f_floor * torch.log(f_floor / wv)).sum(dim=0)

    if alpha_init is None:
        alpha = torch.clamp(torch.as_tensor(1.0, dtype=dtype, device=device).expand(shape).contiguous(), min=1e-9)
    else:
        alpha = torch.clamp(alpha_init.to(dtype=dtype, device=device), min=1e-9, max=1.0)
        alpha = torch.minimum(alpha, alpha_hi)

    # Bracketing: H is convex in alpha with H(0) = H(f) and H'(0) <= 0, so a
    # root inside [0, alpha_hi] (if any) is unique; when no root exists the
    # Newton iteration is confined by the bracket and returns the cap.
    lo = torch.zeros_like(alpha)
    hi = alpha_hi.clone()
    beyond = torch.zeros_like(alpha, dtype=torch.bool)
    alpha = torch.minimum(alpha, alpha_hi)

    # Tolerance of the entropy test.  It is tight on purpose: with tau ~ 0.5 the
    # BGK rate 1/tau is ~2, i.e. the *mirror point* of the entropy parabola where
    # dH vanishes, so the test is only meaningful if the entropy arithmetic is
    # done in double precision.  For float32 inputs the populations are therefore
    # upcast (see below) -- the same defect was found in the C kernel, where
    # storing f^eq in float32 produced a spurious dH of +1e-5 on an exactly
    # equilibrated state and collapsed alpha to 0.
    tol = _ENTROPY_TOL * (h0.abs() + 1.0)

    # Stage 1: does the admissibility bound alpha_hi already satisfy the entropy
    # condition (H not increasing)?  This is the usual near-equilibrium case and
    # it is also what makes the solve immune to floating-point noise in H: a
    # degenerate state (f == f^eq) takes alpha_hi instead of a noise-driven
    # Newton step that would collapse alpha towards zero.
    f_hi = torch.clamp(f + (alpha_hi / tau_t) * delta, min=floor)
    h_hi = (f_hi * torch.log(f_hi / wv)).sum(dim=0)
    root_inside = (h_hi - h0) > tol
    alpha = torch.where(root_inside, alpha, alpha_hi)

    # Stage 2: a root exists inside [0, alpha_hi] -> safeguarded Newton/bisection.
    # The returned value is `lo`, the last iterate whose entropy was *verified*
    # not to increase, so the H-theorem holds by construction even when the
    # bracket has not fully collapsed.
    lo = torch.zeros_like(alpha)
    hi = torch.where(root_inside, alpha_hi.clone(), alpha_hi.clone())
    start = torch.where(
        root_inside & (alpha_init is not None) & (alpha > 1e-9) & (alpha < alpha_hi),
        alpha,
        0.5 * alpha_hi,
    )
    alpha = torch.where(root_inside, torch.clamp(start, min=1e-9, max=alpha_cap), alpha)
    active = root_inside.clone()
    h_err = torch.zeros_like(alpha)
    for _ in range(int(n_iter)):
        ft = torch.clamp(f + (alpha / tau_t) * delta, min=floor)
        ht = (ft * torch.log(ft / wv)).sum(dim=0)
        dht = (delta * (torch.log(ft / wv) + 1.0)).sum(dim=0) / tau_t
        h_err = ht - h0
        upper = h_err > 0
        hi = torch.where(upper & active, alpha, hi)
        lo = torch.where((~upper) & active, alpha, lo)
        denom = torch.where(dht.abs() < 1e-30, torch.full_like(dht, 1e-30), dht)
        step = alpha - h_err / denom
        bad = (step <= lo) | (step >= hi) | ~torch.isfinite(step)
        step = torch.where(bad, 0.5 * (lo + hi), step)
        alpha = torch.clamp(torch.where(active, step, alpha), min=1e-9, max=alpha_cap)
    alpha = torch.where(root_inside & (lo > 1e-9), lo, alpha)
    beyond = ~root_inside
    hit_cap = alpha > alpha_cap * (1.0 - 1e-6)
    rate = alpha / tau_t
    return EntropyConditionInfo(
        alpha=alpha.to(out_dtype),
        residual=h_err.to(out_dtype),
        hit_cap=hit_cap | beyond,
        positivity_bound=alpha_pos.to(out_dtype),
        rate=rate.to(out_dtype),
    )


# ---------------------------------------------------------------------------
# 4. ELBM collision
# ---------------------------------------------------------------------------


def effective_viscosity(tau: float | torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """State-dependent viscosity of the entropic collision ``ν = c_s²(τ/α - 1/2)``."""
    return CS2 * (torch.as_tensor(tau, dtype=alpha.dtype, device=alpha.device) / alpha - 0.5)


def _collide_elbm_generic(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    *,
    lattice: str,
    entropy: str,
    alpha_init: torch.Tensor | None,
    n_newton: int,
    n_iter: int,
    alpha_cap: float,
    return_info: bool,
    velocity_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor | tuple[torch.Tensor, EntropyConditionInfo]:
    from .d3q19 import W_EXACT64 as _w19
    from .d3q19 import macroscopic3d as _mac19
    from .d3q27 import W_EXACT64 as _w27
    from .d3q27 import macroscopic27 as _mac27

    if f.shape[0] != (19 if lattice == "d3q19" else 27):
        raise ValueError(f"{lattice} collision got {f.shape[0]} populations")
    mac = _mac19 if lattice == "d3q19" else _mac27
    rho, ux, uy, uz = mac(f)
    # The entropy condition is solved against the equilibrium at the *measured*
    # velocity, while the collision relaxes towards the half-force shifted one.
    # Reason: the Guo velocity shift adds external work, and dH along the shifted
    # direction is +O(F) *by construction* (H of the equilibrium family grows
    # with |u|).  Testing the shifted direction therefore reports an entropy
    # increase of magnitude ~ rho*|u|*F/2 even for a perfectly equilibrated
    # state, and the safeguarded Newton then collapses to alpha -> 0.  Measured
    # on an exact entropic equilibrium with h = 8, u_tau = 0.02: dH = +1.6e-5
    # against a tolerance of 1.1e-6 -> alpha = 0 in every fluid cell.
    feq_entropy = equilibrium_dispatch(
        rho, ux, uy, uz, lattice=lattice, entropy=entropy, n_newton=n_newton
    )
    if velocity_shift != (0.0, 0.0, 0.0):
        # Guo forcing: build the collision target at the half-force shifted
        # velocity so that collision + source deliver exactly rho*F per step.
        ux = ux + velocity_shift[0]
        uy = uy + velocity_shift[1]
        uz = uz + velocity_shift[2]
        feq = equilibrium_dispatch(
            rho, ux, uy, uz, lattice=lattice, entropy=entropy, n_newton=n_newton
        )
    else:
        feq = feq_entropy
    w = _w19 if lattice == "d3q19" else _w27
    info = solve_entropy_condition(
        f, feq_entropy, tau, w=w, alpha_init=alpha_init, n_iter=n_iter, alpha_cap=alpha_cap
    )
    tau_t = torch.as_tensor(tau, dtype=f.dtype, device=f.device)
    out = f + (info.alpha / tau_t) * (feq - f)
    if return_info:
        return out, info
    return out


def collide_elbm_d3q27(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    *,
    alpha_init: torch.Tensor | None = None,
    n_iter: int = 8,
    alpha_cap: float = ALPHA_CAP,
    return_info: bool = False,
    velocity_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor | tuple[torch.Tensor, EntropyConditionInfo]:
    """One ELBM collision on D3Q27 (Gauss entropy, closed-form equilibrium).

    ``velocity_shift`` is added to the macroscopic velocity *inside* the
    equilibrium only; it carries the Guo half-force shift for body-forced
    flows (see :func:`tensorlbm.elbm_channel._guo_force`).
    """
    return _collide_elbm_generic(
        f, tau, lattice="d3q27", entropy="gauss", alpha_init=alpha_init,
        n_newton=2, n_iter=n_iter, alpha_cap=alpha_cap, return_info=return_info,
        velocity_shift=velocity_shift,
    )


def collide_elbm_d3q19(
    f: torch.Tensor,
    tau: float | torch.Tensor,
    *,
    entropy: str = "gauss",
    alpha_init: torch.Tensor | None = None,
    n_newton: int = 6,
    n_iter: int = 8,
    alpha_cap: float = ALPHA_CAP,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, EntropyConditionInfo]:
    """One ELBM collision on D3Q19 (equilibrium solved by dual Newton).

    D3Q19 weights are not a tensor product, so no product closed form exists;
    the equilibrium comes from :func:`entropic_equilibrium_dual`
    (``n_newton`` iterations, warm-startable through a caller-held state).
    """
    return _collide_elbm_generic(
        f, tau, lattice="d3q19", entropy=entropy, alpha_init=alpha_init,
        n_newton=n_newton, n_iter=n_iter, alpha_cap=alpha_cap, return_info=return_info,
    )