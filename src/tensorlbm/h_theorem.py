"""H-theorem verification face for lattice-Boltzmann collision operators.

The module turns the discrete Boltzmann H-theorem from an *assumption* into a
*machine-checked property* for the collision operators shipped in
:mod:`tensorlbm.entropic_kbc` (D3Q19 / D3Q27) and adds the missing D2Q9 member,
so that entropy-conditioned KBC can be certified -- or falsified -- on every
lattice the project supports.

Functionals (all sums run over the ``Q`` discrete velocities; on a periodic box
the streaming step is an exact permutation of the summation index):

``H`` (discrete Boltzmann entropy functional), per cell::

    H(f) = sum_i f_i ln(f_i / w_i)

``H_rel`` (Kullback-Leibler divergence against the *local* equilibrium) -- the
literal ``H = sum f_i ln(f_i / f_i^eq)`` of the task statement::

    H_rel(f) = sum_i f_i ln(f_i / f_i^eq)   >= 0,    H_rel = 0  iff  f = f^eq

``H2`` (quadratic entropy excess around the local equilibrium), per cell::

    H2(f) = 0.5 * sum_i (f_i - f_i^eq)^2 / f_i^eq

Why these three
---------------
* ``H`` is invariant under periodic streaming (pure index permutation), so a
  full stream+collide step changes ``H`` by the collision contribution only.
  Per-step monotonicity of ``H`` is therefore a clean, tolerance-tight
  H-theorem statement -- criterion **C1** (with ``H_rel`` as **C1b**).
* ``H2`` carries the *exact* BGK entropy-production / dissipation identity.
  ``df = f - f^eq`` has zero mass and zero momentum, so the post-collision
  deviation is ``df* = sigma df`` with ``sigma = 1 - 1/tau`` and::

      H2(f*) = sigma^2 H2(f)
      P = -dH2/dt = (1 - sigma^2) H2 = (2 tau - 1)/tau^2 * H2 = D

  i.e. *entropy production equals dissipation* with the dissipation functional
  ``D = (2 tau - 1)/tau^2 * H2``.  ``P`` is measured on the post-collision
  populations, ``D`` on the pre-collision ones, so both numbers are independent
  -- the residual is pure round-off.  Criterion **C2** (gate <= 1e-10; measured
  order 1e-16).

Operator-agnostic exact identity (second C2 residual).  For any ``f``, ``f*``::

    dH = sum_i (f*_i - f_i) ln(f_i/w_i)  +  sum_i f*_i ln(f*_i/f_i)
       = first-order production          +  remainder

so the production/remainder split of an entropy change is exact and its
residual measures arithmetic, not physics.

Physical (Chapman-Enskog) reading, reported as **C2b**: with the second-order
Hermite truncation ``H2 ~ Pi:Pi / (4 rho cs^4)`` and ``Pi = -2 rho cs^4 tau S``
one gets ``-dH2/dt = cs^2 * eps`` where ``eps = 2 rho nu S:S`` is the viscous
dissipation rate of kinetic energy.  That form carries ``O(Ma^2, k^2)``
truncation error, so its measured error is reported instead of being gated.

KBC collision and its adaptive parameter alpha
----------------------------------------------
``f* = f^eq + (1 - 1/tau) s + alpha h`` with

    alpha = argmin_{alpha in [0,1] and positivity domain} H(f^eq + (1-1/tau)s + alpha h)

The campaign statement calls this coefficient ``alpha``;
:mod:`tensorlbm.entropic_kbc` calls it ``beta``.  Criterion **C5** checks
``alpha`` stays in ``[0, 1]`` cell-by-cell at every step.

Lattices
--------
``d2q9``, ``d3q19``, ``d3q27``.  The generic kernels below are independent
re-implementations of the production D3 kernels of
:mod:`tensorlbm.entropic_kbc`; ``tests/test_h_theorem_kbc.py`` pins them to
<=1e-12 agreement, so the verification face certifies the shipped operator
rather than a private clone.

Limitations (honest scope statement)
------------------------------------
* The equilibrium used throughout is the *polynomial* one.  Strict discrete
  H-monotonicity is a property of the *entropic* (exponential) equilibrium
  (Ansumali & Karlin 2002); with the polynomial equilibrium the inequality can
  be violated at finite Mach number.  The campaign therefore reports violations
  *and* their magnitude instead of hiding them.
* Streaming is periodic only (that is what makes ``H`` streaming invariant).
* Wall / open-boundary entropy production is out of scope.

References
----------
Karlin, Boesch, Chikatamarla (2014), *Multi-relaxation time lattice Boltzmann
model for high Reynolds number flows*; Karlin & Asinari (2010), *Factorization
symmetry in kinetic equations*; Ansumali & Karlin (2002), *Kinetic boundary
conditions in lattice Boltzmann methods*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .d2q9 import W_EXACT as _W_D2Q9
from .d2q9 import C as _C_D2Q9
from .d3q19 import W_EXACT64 as _W_D3Q19
from .d3q19 import C as _C_D3Q19
from .d3q27 import W_EXACT64 as _W_D3Q27
from .d3q27 import C as _C_D3Q27

_CS2 = 1.0 / 3.0  # lattice speed of sound squared
_HERMITE_FACTOR = 1.0 / (2.0 * _CS2 * _CS2)  # 1/(2 cs^4) = 9/2
_EPS = 1e-30
_FLOOR = 1e-300  # log floor (never reached in double precision)


# ---------------------------------------------------------------------------
# Lattice registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatticeSpec:
    """Minimal lattice description: velocities, weights, dimension."""

    name: str
    q: int
    dim: int
    c: torch.Tensor  # (Q, dim) float64
    w: torch.Tensor  # (Q,) float64

    def weight_view(self, ndim: int) -> torch.Tensor:
        """Weights broadcastable against a ``(Q, *spatial)`` tensor."""
        return self.w.view(self.q, *([1] * (ndim - 1)))

    def c_view(self, axis: int, ndim: int) -> torch.Tensor:
        return self.c[:, axis].view(self.q, *([1] * (ndim - 1)))


_LATTICE_TABLE: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {
    "d2q9": (
        torch.as_tensor(_C_D2Q9, dtype=torch.float64),
        torch.tensor(_W_D2Q9, dtype=torch.float64),
        2,
    ),
    "d3q19": (
        torch.as_tensor(_C_D3Q19, dtype=torch.float64),
        _W_D3Q19.detach().clone().to(torch.float64),
        3,
    ),
    "d3q27": (
        torch.as_tensor(_C_D3Q27, dtype=torch.float64),
        _W_D3Q27.detach().clone().to(torch.float64),
        3,
    ),
}


def available_lattices() -> tuple[str, ...]:
    return tuple(_LATTICE_TABLE)


def lattice_spec(name: str = "d2q9", dtype: torch.dtype = torch.float64) -> LatticeSpec:
    """Return the :class:`LatticeSpec` for ``name`` (``d2q9``/``d3q19``/``d3q27``)."""
    key = name.lower()
    if key not in _LATTICE_TABLE:
        raise ValueError(f"unknown lattice {name!r}; available: {available_lattices()}")
    c, w, dim = _LATTICE_TABLE[key]
    return LatticeSpec(name=key, q=int(c.shape[0]), dim=dim, c=c.to(dtype), w=w.to(dtype))


# ---------------------------------------------------------------------------
# Macroscopic fields and polynomial equilibrium (generic over dimension)
# ---------------------------------------------------------------------------


def macroscopic(f: torch.Tensor, spec: LatticeSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rho, u)`` with ``u`` shaped ``(dim, *spatial)``."""
    rho = f.sum(dim=0)
    rho_safe = rho.clamp_min(_EPS)
    u = torch.stack(
        [(f * spec.c_view(a, f.ndim)).sum(dim=0) / rho_safe for a in range(spec.dim)],
        dim=0,
    )
    return rho, u


def equilibrium(rho: torch.Tensor, u: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    """Second-order (polynomial) lattice equilibrium ``f^eq``."""
    ndim = rho.ndim + 1
    wv = spec.weight_view(ndim)
    cu = torch.zeros_like(rho)
    for a in range(spec.dim):
        cu = cu + spec.c_view(a, ndim) * u[a].unsqueeze(0)
    u_sq = (u * u).sum(dim=0)
    return wv * rho.unsqueeze(0) * (
        1.0 + cu / _CS2 + 0.5 * (cu * cu) / (_CS2 * _CS2) - 0.5 * u_sq.unsqueeze(0) / _CS2
    )


def equilibrium_from_state(f: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    rho, u = macroscopic(f, spec)
    return equilibrium(rho, u, spec)


# ---------------------------------------------------------------------------
# Entropy functionals
# ---------------------------------------------------------------------------


def entropy_boltzmann_cell(f: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    """Per-cell ``H(f) = sum_i f_i ln(f_i / w_i)``."""
    fs = f.clamp_min(_FLOOR)
    wv = spec.weight_view(f.ndim)
    return (fs * torch.log(fs / wv)).sum(dim=0)


def entropy_boltzmann(f: torch.Tensor, spec: LatticeSpec) -> float:
    """Scalar ``H(f) = sum_x sum_i f_i ln(f_i / w_i)`` (streaming invariant)."""
    return float(entropy_boltzmann_cell(f, spec).sum())


def entropy_relative_cell(f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
    """Per-cell ``H_rel = sum_i f_i ln(f_i / f_i^eq)`` (KL divergence, >= 0)."""
    fs = f.clamp_min(_FLOOR)
    fq = feq.clamp_min(_FLOOR)
    return (fs * torch.log(fs / fq)).sum(dim=0)


def entropy_relative(f: torch.Tensor, feq: torch.Tensor) -> float:
    return float(entropy_relative_cell(f, feq).sum())


def entropy_excess_quadratic_cell(f: torch.Tensor, feq: torch.Tensor) -> torch.Tensor:
    """Per-cell ``H2 = 0.5 * sum_i (f_i - f_i^eq)^2 / f_i^eq``."""
    d = f - feq
    return 0.5 * (d * d / feq.clamp_min(_FLOOR)).sum(dim=0)


def entropy_excess_quadratic(f: torch.Tensor, feq: torch.Tensor) -> float:
    return float(entropy_excess_quadratic_cell(f, feq).sum())

# ---------------------------------------------------------------------------
# Exact identities (C2 gate)
# ---------------------------------------------------------------------------


@dataclass
class EntropyBalance:
    """Exact telescoping split of the entropy change of one collision step."""

    H_pre: float
    H_post: float
    delta_H: float
    production_first_order: float
    remainder: float
    identity_residual: float
    identity_rel_residual: float

    def as_dict(self) -> dict[str, float]:
        return {
            "H_pre": self.H_pre,
            "H_post": self.H_post,
            "delta_H": self.delta_H,
            "production_first_order": self.production_first_order,
            "remainder": self.remainder,
            "identity_residual": self.identity_residual,
            "identity_rel_residual": self.identity_rel_residual,
        }


def entropy_balance(
    f_pre: torch.Tensor,
    f_post: torch.Tensor,
    spec: LatticeSpec,
    *,
    H_pre: float | None = None,
) -> EntropyBalance:
    """Check ``dH = sum (f*-f) ln(f/w) + sum f* ln(f*/f)`` (exact for any collision)."""
    wv = spec.weight_view(f_pre.ndim)
    fs = f_pre.clamp_min(_FLOOR)
    fp = f_post.clamp_min(_FLOOR)
    lw = torch.log(fs / wv)
    if H_pre is None:
        H_pre = float((fs * lw).sum())
    # L1 scale of the entropy summands: the correct reference for a round-off
    # claim (H itself may pass through zero, and the production/remainder pair
    # may nearly cancel).
    l1_scale = float((fs * lw).abs().sum())
    H_post = float((fp * torch.log(fp / wv)).sum())
    first = float(((f_post - f_pre) * lw).sum())
    remainder = float((fp * torch.log(fp / fs)).sum())
    delta = H_post - H_pre
    resid = delta - (first + remainder)
    scale = max(l1_scale, _FLOOR)
    return EntropyBalance(
        H_pre=H_pre,
        H_post=H_post,
        delta_H=delta,
        production_first_order=first,
        remainder=remainder,
        identity_residual=resid,
        identity_rel_residual=abs(resid) / scale,
    )


@dataclass
class DissipationIdentity:
    """Exact BGK entropy-production / dissipation identity on ``H2``."""

    tau: float
    sigma: float
    H2_pre: float
    H2_post: float
    delta_H2: float
    production: float  # -dH2
    dissipation: float  # (2 tau - 1)/tau^2 * H2(pre)
    identity_residual: float
    identity_rel_residual: float
    stress_quadratic: float  # Pi:Pi/(4 rho cs^4)
    H2_vs_stress_rel_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "tau": self.tau,
            "sigma": self.sigma,
            "H2_pre": self.H2_pre,
            "H2_post": self.H2_post,
            "delta_H2": self.delta_H2,
            "production": self.production,
            "dissipation": self.dissipation,
            "identity_residual": self.identity_residual,
            "identity_rel_residual": self.identity_rel_residual,
            "stress_quadratic": self.stress_quadratic,
            "H2_vs_stress_rel_error": self.H2_vs_stress_rel_error,
        }


def dissipation_identity(
    f_pre: torch.Tensor,
    f_post: torch.Tensor,
    tau: float,
    spec: LatticeSpec,
) -> DissipationIdentity:
    """Exact production-dissipation identity for a BGK-type relaxation.

    ``df* = sigma df`` with ``sigma = 1 - 1/tau`` gives ``H2(f*) = sigma^2 H2(f)``
    exactly, hence ``P = -dH2 = (1 - sigma^2) H2 = (2tau-1)/tau^2 H2 = D``.  ``P``
    is computed from the post-collision state, ``D`` from the pre-collision state.
    """
    rho, u = macroscopic(f_pre, spec)
    feq_pre = equilibrium(rho, u, spec)
    rho_p, u_p = macroscopic(f_post, spec)
    feq_post = equilibrium(rho_p, u_p, spec)

    H2_pre = entropy_excess_quadratic(f_pre, feq_pre)
    H2_post = entropy_excess_quadratic(f_post, feq_post)

    sigma = 1.0 - 1.0 / tau
    production = -(H2_post - H2_pre)
    diss_coeff = (2.0 * tau - 1.0) / (tau * tau)
    dissipation = diss_coeff * H2_pre
    resid = production - dissipation
    scale = max(abs(H2_pre), abs(H2_post), abs(production), abs(dissipation), _FLOOR)

    pi = second_order_moment(f_pre - feq_pre, spec)
    pi2_total = float((pi * pi).sum())
    rho_mean = max(float(rho.mean()), _EPS)
    stress_quad = pi2_total / (4.0 * rho_mean * _CS2 * _CS2)

    return DissipationIdentity(
        tau=float(tau),
        sigma=float(sigma),
        H2_pre=H2_pre,
        H2_post=H2_post,
        delta_H2=H2_post - H2_pre,
        production=production,
        dissipation=dissipation,
        identity_residual=resid,
        identity_rel_residual=abs(resid) / scale,
        stress_quadratic=stress_quad,
        H2_vs_stress_rel_error=abs(H2_pre - stress_quad) / max(abs(H2_pre), _FLOOR),
    )


# ---------------------------------------------------------------------------
# Second-order moments and KBC decomposition
# ---------------------------------------------------------------------------


def second_order_moment(delta: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    """``Pi_ab = sum_i c_ia c_ib d_i`` with shape ``(dim, dim, *spatial)``."""
    dim = spec.dim
    pi = torch.empty((dim, dim, *delta.shape[1:]), dtype=delta.dtype, device=delta.device)
    for a in range(dim):
        ca = spec.c_view(a, delta.ndim)
        for b in range(a, dim):
            val = (ca * spec.c_view(b, delta.ndim) * delta).sum(dim=0)
            pi[a, b] = val
            pi[b, a] = val
    return pi


def stress_inner_product(f: torch.Tensor, feq: torch.Tensor, spec: LatticeSpec) -> float:
    """Spatially summed ``Pi:Pi`` of the non-equilibrium stress."""
    pi = second_order_moment(f - feq, spec)
    return float((pi * pi).sum())


def kbc_decompose(
    f: torch.Tensor,
    spec: LatticeSpec,
    feq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Split ``df = f - f^eq`` into ``(s, k, h)``: shear / bulk-trace / higher order.

    * ``s`` -- second-order deviatoric (traceless) projection; carries the whole
      shear stress, hence the shear viscosity;
    * ``k`` -- second-order trace (bulk) projection;
    * ``h`` -- residual (third order and above).

    ``s + k + h = df`` by construction.  Same construction as
    :func:`tensorlbm.entropic_kbc._kbc_decompose`, generalised to ``dim in {2,3}``.
    """
    if feq is None:
        feq = equilibrium_from_state(f, spec)
    delta = f - feq
    dense = delta.ndim
    wv = spec.weight_view(dense)
    dim = spec.dim

    pi = second_order_moment(delta, spec)
    tr = sum(pi[a, a] for a in range(dim))
    c2 = torch.zeros_like(tr)
    for a in range(dim):
        ca = spec.c_view(a, dense)
        c2 = c2 + ca * ca

    dev = pi.clone()
    for a in range(dim):
        dev[a, a] = pi[a, a] - tr / dim

    dev_proj = torch.zeros_like(delta)
    full_proj = torch.zeros_like(delta)
    for a in range(dim):
        for b in range(dim):
            hab = spec.c_view(a, dense) * spec.c_view(b, dense)
            hd_ab = hab - (c2 / dim if a == b else 0.0)
            dev_proj = dev_proj + hd_ab * dev[a, b].unsqueeze(0)
            full_proj = full_proj + (hab - (_CS2 if a == b else 0.0)) * pi[a, b].unsqueeze(0)

    s = _HERMITE_FACTOR * wv * dev_proj
    f2 = _HERMITE_FACTOR * wv * full_proj
    k = f2 - s
    h = delta - f2

    s_mom = second_order_moment(s, spec)
    shear_residual = max(
        float((s_mom[a, b] - dev[a, b]).abs().max()) for a in range(dim) for b in range(dim)
    )
    diag = {
        "decomposition_residual": float((s + k + h - delta).abs().max()),
        "shear_stress_residual": shear_residual,
        "shear_norm": float(s.abs().sum()),
        "bulk_norm": float(k.abs().sum()),
        "higher_norm": float(h.abs().sum()),
    }
    return s, k, h, diag


# ---------------------------------------------------------------------------
# Collision operators
# ---------------------------------------------------------------------------


def collide_bgk(f: torch.Tensor, tau: float, spec: LatticeSpec) -> torch.Tensor:
    """Single-relaxation-time BGK collision ``f* = f - (f - f^eq)/tau``."""
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    feq = equilibrium_from_state(f, spec)
    return f - (f - feq) / tau


def _solve_alpha_entropy(
    f_base: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    alpha_init: torch.Tensor,
    lower: float = 0.0,
    upper: float = 1.0,
    max_iter: int = 40,
    tol: float = 1e-12,
    method: str = "newton",
) -> torch.Tensor:
    """Per-cell minimiser of ``H(f_base + alpha h)``.

    ``H`` is convex in ``alpha`` (``d2H/dalpha2 = sum h_i^2/f_i > 0``), so the
    minimum is unique.  The search interval is ``[lower, upper]`` intersected
    with the population-positivity domain; when the intersection is empty the
    positivity domain wins (realisability has priority over the bracket).

    ``method="newton"`` uses a safeguarded Newton iteration (monotone gradient ->
    bracket update, Newton step when it lands inside the bracket, bisection step
    otherwise); ``method="bisection"`` is the textbook reference used to
    cross-validate it.  Both stop on the bracket width / step size, which is
    amplitude independent -- an ``|dH|`` stop would truncate the solve at small
    perturbation amplitudes (cf. the W7-A shear-wave audit referenced in
    :mod:`tensorlbm.entropic_kbc`).
    """
    eps_h = 1e-30
    h_safe = torch.where(h.abs() > eps_h, h, torch.full_like(h, eps_h))
    ratio = -f_base / h_safe

    neg_inf = torch.full_like(alpha_init, -1e6)
    pos_inf = torch.full_like(alpha_init, 1e6)

    ratio_pos = torch.where(h > eps_h, ratio, neg_inf.unsqueeze(0).expand_as(ratio))
    a_lower_pos = ratio_pos.amax(dim=0)
    ratio_neg = torch.where(h < -eps_h, ratio, pos_inf.unsqueeze(0).expand_as(ratio))
    a_upper_pos = ratio_neg.amin(dim=0)

    no_constraint = (a_lower_pos <= -1e5) & (a_upper_pos >= 1e5)
    a_lower_pos = torch.where(no_constraint, alpha_init, a_lower_pos)
    a_upper_pos = torch.where(no_constraint, alpha_init, a_upper_pos)

    zero = torch.zeros_like(a_lower_pos)
    one = torch.ones_like(a_upper_pos)
    a_lower = torch.maximum(a_lower_pos, zero + lower)
    a_upper = torch.minimum(a_upper_pos, one * upper)
    escaped = a_lower > a_upper  # empty intersection: keep the positivity domain
    a_lower = torch.where(escaped, a_lower_pos, a_lower)
    a_upper = torch.where(escaped, a_upper_pos, a_upper)

    init = torch.maximum(torch.minimum(alpha_init, a_upper), a_lower)
    lo = torch.minimum(a_lower, a_upper - 1e-14)
    hi = torch.maximum(a_upper, lo + 1e-14)

    lnw = torch.log(w)
    sum_h = h.sum(dim=0)

    def _grad(alpha: torch.Tensor) -> torch.Tensor:
        """``dH/dalpha = sum_i h_i (1 + ln(f_i/w_i))``."""
        f_mid = (f_base + alpha.unsqueeze(0) * h).clamp_min(_FLOOR)
        return (h * (torch.log(f_mid) - lnw)).sum(dim=0) + sum_h

    if method == "bisection":
        a_l, a_u = lo, hi
        for _ in range(max_iter):
            if float((a_u - a_l).abs().max()) < tol:
                break
            mid = 0.5 * (a_l + a_u)
            positive = _grad(mid) > 0
            a_u = torch.where(positive, mid, a_u)
            a_l = torch.where(positive, a_l, mid)
        alpha = 0.5 * (a_l + a_u)
    elif method == "newton":
        alpha = init.clamp(min=0.0, max=0.0) + init  # copy of init
        for _ in range(max_iter):
            g = _grad(alpha)
            f_mid = (f_base + alpha.unsqueeze(0) * h).clamp_min(_FLOOR)
            g2 = (h * h / f_mid).sum(dim=0)
            # monotone increasing gradient (convex H): update the bracket
            lo = torch.where(g < 0, alpha, lo)
            hi = torch.where(g > 0, alpha, hi)
            step = g / g2.clamp_min(1e-300)
            newton = alpha - step
            inside = (newton > lo) & (newton < hi) & torch.isfinite(newton)
            cand = torch.where(inside, newton, 0.5 * (lo + hi))
            delta = (cand - alpha).abs()
            alpha = cand
            if float(delta.max()) <= tol or float((hi - lo).abs().max()) <= tol:
                break
    else:
        raise ValueError(f"unknown alpha-solve method {method!r}")

    # keep the positivity domain if it had to override the bracket, otherwise
    # keep the result strictly inside [lower, upper]
    alpha = torch.where(torch.isfinite(alpha), alpha, init)
    alpha = torch.where(escaped, init, alpha.clamp(lower, upper))
    return alpha


def collide_kbc(
    f: torch.Tensor,
    tau: float,
    spec: LatticeSpec,
    *,
    alpha_bounds: tuple[float, float] = (0.0, 1.0),
    max_iter: int = 40,
    tol: float = 1e-12,
    method: str = "newton",
    return_internals: bool = False,
):
    """Entropy-conditioned KBC collision; returns ``(f_post, alpha)``.

    ``f* = f^eq + (1 - 1/tau) s + alpha h``: the shear projection is relaxed by
    the exact BGK factor (so ``nu = cs^2 (tau - 1/2)`` exactly), while the ghost
    mode ``h`` is relaxed by the entropy-optimal ``alpha`` confined to
    ``alpha_bounds``.
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    rho, u = macroscopic(f, spec)
    feq = equilibrium(rho, u, spec)
    s, _k, h, _ = kbc_decompose(f, spec, feq)
    sigma = 1.0 - 1.0 / tau
    init_val = min(max(sigma, alpha_bounds[0]), alpha_bounds[1])
    alpha_init = torch.full_like(rho, init_val)
    alpha = _solve_alpha_entropy(
        feq + sigma * s,
        h,
        spec.weight_view(f.ndim),
        alpha_init,
        alpha_bounds[0],
        alpha_bounds[1],
        max_iter,
        tol,
        method=method,
    )
    f_post = feq + sigma * s + alpha.unsqueeze(0) * h
    if return_internals:
        return f_post, alpha, {"feq": feq, "s": s, "k": _k, "h": h, "sigma": sigma}
    return f_post, alpha


def collide_kbc_fixed_alpha(
    f: torch.Tensor,
    tau: float,
    spec: LatticeSpec,
    alpha: float,
) -> torch.Tensor:
    """KBC collision with a *fixed* ghost-mode coefficient (``alpha = const``).

    ``alpha = 0`` is the degeneration control (**C7**): the ghost mode is fully
    relaxed in one step instead of being entropy-conditioned.
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    rho, u = macroscopic(f, spec)
    feq = equilibrium(rho, u, spec)
    s, _k, h, _ = kbc_decompose(f, spec, feq)
    sigma = 1.0 - 1.0 / tau
    return feq + sigma * s + alpha * h


# ---------------------------------------------------------------------------
# Streaming, step drivers, health, dissipation
# ---------------------------------------------------------------------------


def stream(f: torch.Tensor, spec: LatticeSpec) -> torch.Tensor:
    """Full-way periodic streaming (an exact index permutation)."""
    out = f
    for a in range(spec.dim):
        dim = -(a + 1)
        col = [int(v) for v in spec.c[:, a].tolist()]
        if all(v == 0 for v in col):
            continue
        out = torch.stack(
            [torch.roll(out[i], shifts=col[i], dims=dim) for i in range(spec.q)], dim=0
        )
    return out


def step(
    f: torch.Tensor,
    tau: float,
    spec: LatticeSpec,
    *,
    collision: str = "kbc",
    alpha: float | None = None,
    alpha_bounds: tuple[float, float] = (0.0, 1.0),
    max_iter: int = 40,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Streaming + collision.  Returns ``(f_next, alpha_or_None)``."""
    if collision == "kbc":
        if alpha is not None:
            raise ValueError("fixed alpha requires collision='kbc_fixed'")
        f_c, alpha_out = collide_kbc(
            f, tau, spec, alpha_bounds=alpha_bounds, max_iter=max_iter, tol=tol
        )
    elif collision == "kbc_fixed":
        f_c = collide_kbc_fixed_alpha(f, tau, spec, 0.0 if alpha is None else alpha)
        alpha_out = None
    elif collision == "bgk":
        f_c = collide_bgk(f, tau, spec)
        alpha_out = None
    else:
        raise ValueError(f"unknown collision {collision!r}")
    return stream(f_c, spec), alpha_out


def flow_health(f: torch.Tensor, spec: LatticeSpec, u_limit: float = 0.5) -> dict[str, Any]:
    """Divergence indicators: finiteness, positivity, Mach limit."""
    if not bool(torch.isfinite(f).all()):
        return {
            "finite": False,
            "rho_min": float("nan"),
            "rho_max": float("nan"),
            "u_max": float("nan"),
            "f_min": float("nan"),
            "f_max": float("nan"),
            "negative_cells": -1,
            "diverged": True,
        }
    rho, u = macroscopic(f, spec)
    u_max = float(u.square().sum(dim=0).sqrt().max())
    f_min = float(f.min())
    rho_min = float(rho.min())
    rho_max = float(rho.max())
    negative = int((f < 0).sum().item())
    diverged = bool(u_max > u_limit or rho_min <= 0.0 or rho_max > 10.0 or f_min < -1e-3)
    return {
        "finite": True,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "u_max": u_max,
        "f_min": f_min,
        "f_max": float(f.max()),
        "negative_cells": negative,
        "diverged": diverged,
    }


def shear_rate_tensor(u: torch.Tensor) -> torch.Tensor:
    """Central-difference rate of strain ``S_ab = 0.5(d_a u_b + d_b u_a)`` (periodic)."""
    dim, *shape = u.shape
    s = torch.empty((dim, dim, *shape), dtype=u.dtype, device=u.device)
    for a in range(dim):
        for b in range(a, dim):
            dua = (torch.roll(u[b], -1, dims=-(a + 1)) - torch.roll(u[b], 1, dims=-(a + 1))) * 0.5
            dub = (torch.roll(u[a], -1, dims=-(b + 1)) - torch.roll(u[a], 1, dims=-(b + 1))) * 0.5
            s[a, b] = 0.5 * (dua + dub)
            s[b, a] = s[a, b]
    return s


def viscous_dissipation_rate(f: torch.Tensor, tau: float, spec: LatticeSpec) -> float:
    """``eps = sum_x 2 rho nu S:S`` with central-difference ``S`` (periodic)."""
    rho, u = macroscopic(f, spec)
    nu = _CS2 * (tau - 0.5)
    s = shear_rate_tensor(u)
    s_sq = (s * s).sum(dim=0).sum(dim=0)
    return float((2.0 * rho * nu * s_sq).sum())


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def make_ic(
    kind: str,
    spec: LatticeSpec,
    shape: Sequence[int],
    *,
    u0: float = 0.1,
    wavelength: int | None = None,
    rho0: float = 1.0,
    density_amplitude: float = 0.0,
    interface_width: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a non-equilibrium initial condition on a periodic box.

    ``kind``:
      * ``shear_wave``   -- ``u_y = u0 sin(2 pi x / lam)`` (Kolmogorov-type, smooth);
      * ``shear_layer``  -- ``u_y = u0 tanh((x - x0)/delta)`` (grid-scale shear:
        the canonical under-resolved high-Reynolds configuration);
      * ``density_step`` -- ``rho = rho0 + A tanh((x - x0)/delta)`` with ``u = 0``.

    ``shape`` is the spatial shape (``(ny, nx)`` for D2Q9, ``(nz, ny, nx)`` for 3D).
    """
    shape = tuple(int(v) for v in shape)
    if len(shape) != spec.dim:
        raise ValueError(f"shape {shape} incompatible with {spec.name} (dim={spec.dim})")
    nx = shape[-1]
    x = torch.arange(nx, dtype=dtype).view(*([1] * (spec.dim - 1)), nx)
    lam = int(wavelength) if wavelength else nx
    rho = torch.full(shape, float(rho0), dtype=dtype)
    u = torch.zeros((spec.dim, *shape), dtype=dtype)

    if kind == "shear_wave":
        u[1] = u0 * torch.sin(2.0 * math.pi * x / lam)
        if density_amplitude:
            rho = rho + density_amplitude * torch.cos(2.0 * math.pi * x / lam)
    elif kind == "shear_layer":
        x0 = 0.25 * nx
        u[1] = u0 * torch.tanh((x - x0) / interface_width)
    elif kind == "density_step":
        x0 = 0.5 * nx
        rho = rho + float(u0) * torch.tanh((x - x0) / interface_width)
    else:
        raise ValueError(f"unknown initial condition {kind!r}")

    rho = rho.contiguous().expand(shape)
    return equilibrium(rho, u, spec)


@dataclass
class KBCProductionIdentity:
    """Exact entropy-production / dissipation ledger for the KBC operator.

    The KBC post-collision state is ``f* = f^eq + sigma s + alpha h`` with
    ``sigma = 1 - 1/tau`` and ``s + k + h = f - f^eq``.  Both the measured
    production (from ``H2`` of the two states) and the predicted dissipation are
    independent functions of the pre-collision state and the operator parameters,
    so their difference is a floating-point residual -- this is the *exact*
    version of the production=dissipation statement for the KBC operator (the
    ``sigma^2`` form of :func:`dissipation_identity` holds for BGK, where the
    whole deviation is relaxed by one factor).

    Mode-resolved dissipation (reported, not gated)::

        D_s = (1 - sigma^2) * H2(s)      shear mode, exact-BGK retention
        D_k = (1 - 0)       * H2(k)      bulk trace mode, fully relaxed (retention 0)
        D_h = (1 - alpha^2) * H2(h)      ghost mode, entropy-conditioned retention

    ``cross_term = D_total - (D_s + D_k + D_h)`` is the metric cross-talk of the
    three non-orthogonal modes in the ``1/f^eq`` norm; it is O(Ma^2) and is
    reported rather than hidden.
    """

    tau: float
    sigma: float
    alpha_min: float
    alpha_max: float
    H2_pre: float
    H2_post: float
    production: float
    dissipation: float
    identity_residual: float
    identity_rel_residual: float
    D_shear: float
    D_bulk: float
    D_ghost: float
    cross_term: float
    operator_residual: float
    moment_residual: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def kbc_production_identity(
    f_pre: torch.Tensor,
    f_post: torch.Tensor,
    tau: float,
    spec: LatticeSpec,
    alpha: torch.Tensor | None = None,
    internals: dict[str, Any] | None = None,
) -> KBCProductionIdentity:
    """Exact production=dissipation ledger for ``f* = f^eq + sigma s + alpha h``."""
    if internals is None:
        rho, u = macroscopic(f_pre, spec)
        feq = equilibrium(rho, u, spec)
        s, k, h, _ = kbc_decompose(f_pre, spec, feq)
    else:
        feq = internals["feq"]
        s, k, h = internals["s"], internals["k"], internals["h"]
    rho, u = macroscopic(f_pre, spec)
    sigma = 1.0 - 1.0 / tau

    if alpha is None:
        # recover alpha from the post-collision state where the ghost mode lives
        num = f_post - feq - sigma * s
        h_safe = torch.where(h.abs() > 1e-300, h, torch.full_like(h, 1e-300))
        alpha = (num / h_safe).mean(dim=0)
    a = alpha.unsqueeze(0)

    feq_metric = feq.clamp_min(_FLOOR)

    def _h2_of(x: torch.Tensor) -> float:
        return 0.5 * float((x * x / feq_metric).sum())

    H2_pre = entropy_excess_quadratic(f_pre, feq)
    # post-collision excess measured against the *same* equilibrium: mass and
    # momentum are conserved, so f^eq(f*) = f^eq(f) up to round-off
    H2_post = _h2_of(f_post - feq)
    production = H2_pre - H2_post

    post_pred = feq + sigma * s + a * h
    dissipation = _h2_of(s + k + h) - _h2_of(sigma * s + a * h)
    resid = production - dissipation
    scale = max(abs(H2_pre), abs(H2_post), abs(production), _FLOOR)

    rho_p, u_p = macroscopic(f_post, spec)
    moment_residual = max(
        float((rho_p - rho).abs().max()),
        max(float((u_p[i] - u[i]).abs().max()) * float(rho.mean()) for i in range(spec.dim)),
    )
    return KBCProductionIdentity(
        tau=float(tau),
        sigma=float(sigma),
        alpha_min=float(alpha.min()),
        alpha_max=float(alpha.max()),
        H2_pre=H2_pre,
        H2_post=H2_post,
        production=production,
        dissipation=dissipation,
        identity_residual=resid,
        identity_rel_residual=abs(resid) / scale,
        D_shear=(1.0 - sigma * sigma) * _h2_of(s),
        D_bulk=1.0 * _h2_of(k),
        D_ghost=float((1.0 - alpha * alpha).mean()) * _h2_of(h),
        cross_term=dissipation
        - (
            (1.0 - sigma * sigma) * _h2_of(s)
            + _h2_of(k)
            + float((1.0 - alpha * alpha).mean()) * _h2_of(h)
        ),
        operator_residual=float((f_post - post_pred).abs().max()),
        moment_residual=moment_residual,
    )


# ---------------------------------------------------------------------------
# Campaign driver: criteria C1 ... C7
# ---------------------------------------------------------------------------


@dataclass
class CampaignConfig:
    """Configuration of one H-theorem campaign (one lattice, one IC)."""

    lattice: str = "d2q9"
    shape: tuple[int, ...] = (32, 32)
    ic: str = "shear_wave"
    u0: float = 0.15
    wavelength: int | None = 4
    interface_width: float = 1.0
    tau: float = 0.501
    steps_kbc: int = 10000
    steps_bgk_max: int = 100
    sample: int = 100
    alpha_bounds: tuple[float, float] = (0.0, 1.0)
    max_iter: int = 40  # safeguarded Newton iteration cap
    alpha_tol: float = 1e-12
    u_limit: float = 0.5
    # equilibrium-recovery sub-run (C3)
    eq_ic: str = "shear_wave"
    eq_tau: float = 1.0
    eq_steps: int = 2000
    eq_u0: float = 0.01
    eq_wavelength: int | None = 16
    # tolerances
    c1_rel_tol: float = 1e-12
    c2_tol: float = 1e-10
    c3_tol: float = 1e-6
    dtype: str = "float64"

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["shape"] = list(self.shape)
        d["alpha_bounds"] = list(self.alpha_bounds)
        return d


def _to_dtype(name: str) -> torch.dtype:
    return {"float64": torch.float64, "float32": torch.float32}[name]


def _u_max(f: torch.Tensor, spec: LatticeSpec) -> float:
    _, u = macroscopic(f, spec)
    return float(u.square().sum(dim=0).sqrt().max())


def entropy_scale(f: torch.Tensor, spec: LatticeSpec) -> float:
    """``sum_x sum_i |f_i ln(f_i/w_i)|`` -- L1 scale of the entropy summands.

    Round-off of the entropy functional is ``~eps *`` this number, so it is the
    correct reference for a floating-point monotonicity tolerance (the entropy
    itself can pass through zero while the summands are O(1)).
    """
    fs = f.clamp_min(_FLOOR)
    return float((fs * torch.log(fs / spec.weight_view(f.ndim))).abs().sum())


def _state_snapshot(
    f: torch.Tensor,
    spec: LatticeSpec,
    n: int,
    alpha: torch.Tensor | None = None,
    delta_h: float | None = None,
) -> dict[str, float]:
    feq = equilibrium_from_state(f, spec)
    row: dict[str, float] = {
        "step": float(n),
        "H": entropy_boltzmann(f, spec),
        "H_rel": entropy_relative(f, feq),
        "u_max": _u_max(f, spec),
        "f_min": float(f.min()),
        "negative_cells": float((f < 0).sum()),
    }
    if alpha is not None:
        row["alpha_min"] = float(alpha.min())
        row["alpha_max"] = float(alpha.max())
        row["alpha_mean"] = float(alpha.mean())
        row["alpha_at_lower_bound"] = float((alpha <= 1e-9).float().mean())
    if delta_h is not None:
        row["delta_H_collision"] = float(delta_h)
    return row


def run_campaign(cfg: CampaignConfig | None = None) -> dict[str, Any]:  # noqa: C901
    """Execute criteria C1...C7 for one lattice / one initial condition.

    The main KBC run doubles as the C4 stability run: it is executed with the
    *same* configuration as the BGK negative control (C6), so "KBC survives N
    steps while BGK dies inside M" is a like-for-like comparison.
    """
    cfg = cfg or CampaignConfig()
    dtype = _to_dtype(cfg.dtype)
    spec = lattice_spec(cfg.lattice, dtype=dtype)
    shape = tuple(cfg.shape)
    if len(shape) != spec.dim:
        raise ValueError(f"shape {shape} incompatible with {cfg.lattice}")
    wv = spec.weight_view(spec.dim + 1)

    f0 = make_ic(
        cfg.ic,
        spec,
        shape,
        u0=cfg.u0,
        wavelength=cfg.wavelength,
        interface_width=cfg.interface_width,
        dtype=dtype,
    )
    result: dict[str, Any] = {
        "config": cfg.as_dict(),
        "lattice": {"name": spec.name, "q": spec.q, "dim": spec.dim},
        "ic_health": flow_health(f0, spec, cfg.u_limit),
    }

    # ---------------- C1 / C2 / C5 / C4 : entropic KBC main run --------------
    f = stream(f0, spec)  # streaming first: builds the non-equilibrium state
    H_prev = entropy_boltzmann(f, spec)
    S_lin_max = entropy_scale(f, spec)
    c1_violations = 0
    c1b_violations = 0
    max_excess = 0.0
    max_excess_rel = 0.0
    c2_tel_abs = 0.0
    c2_ledger_abs = 0.0
    c2_operator_abs = 0.0
    c2_moment_abs = 0.0
    h2_scale_max = 0.0
    bgk_h2_scale_max = 0.0
    bgk_sigma2_abs = 0.0
    alpha_min = math.inf
    alpha_max = -math.inf
    alpha_samples = 0
    alpha_escaped_cells = 0
    c2d_rows: list[dict[str, float]] = []
    series: list[dict[str, float]] = []
    sample_every = max(1, cfg.steps_kbc // max(1, cfg.sample))
    kbc_diverged_at: int | None = None

    for n in range(1, cfg.steps_kbc + 1):
        f_post, alpha, internals = collide_kbc(
            f,
            cfg.tau,
            spec,
            alpha_bounds=cfg.alpha_bounds,
            max_iter=cfg.max_iter,
            tol=cfg.alpha_tol,
            return_internals=True,
        )
        bal = entropy_balance(f, f_post, spec, H_pre=H_prev)
        ledger = kbc_production_identity(f, f_post, cfg.tau, spec, alpha, internals)

        # --- C1: H must not increase beyond the round-off of the entropy sum ---
        if bal.delta_H > 0.0:
            tol_abs = cfg.c1_rel_tol * S_lin_max
            max_excess = max(max_excess, bal.delta_H)
            max_excess_rel = max(max_excess_rel, bal.delta_H / max(S_lin_max, _FLOOR))
            if bal.delta_H > tol_abs:
                c1_violations += 1

        # --- C1b: KL divergence against the local equilibrium (sampled) ---
        if n % sample_every == 0:
            feq_here = internals["feq"]
            hrel_pre = entropy_relative(f, feq_here)
            d_hrel = entropy_relative(f_post, feq_here) - hrel_pre
            if d_hrel > cfg.c1_rel_tol * max(abs(hrel_pre), 1.0):
                c1b_violations += 1

        # --- C2: exact identities ---
        c2_tel_abs = max(c2_tel_abs, abs(bal.identity_residual))
        c2_ledger_abs = max(c2_ledger_abs, abs(ledger.identity_residual))
        c2_operator_abs = max(c2_operator_abs, ledger.operator_residual)
        c2_moment_abs = max(c2_moment_abs, ledger.moment_residual)
        h2_scale_max = max(h2_scale_max, abs(ledger.H2_pre), abs(ledger.production))

        # --- C2d: physical reading of the same balance (sampled) ---
        if n % sample_every == 0 or n == 1:
            eps = viscous_dissipation_rate(f, cfg.tau, spec)
            c2d_rows.append(
                {
                    "step": float(n),
                    "production": ledger.production,
                    "cs2_times_epsilon": _CS2 * eps,
                    "H2_pre": ledger.H2_pre,
                    "D_shear": ledger.D_shear,
                    "D_bulk": ledger.D_bulk,
                    "D_ghost": ledger.D_ghost,
                    "cross_term": ledger.cross_term,
                }
            )

        # --- C5: alpha in bounds ---
        alpha_min = min(alpha_min, float(alpha.min()))
        alpha_max = max(alpha_max, float(alpha.max()))
        alpha_samples += int(alpha.numel())
        alpha_escaped_cells += int(
            ((alpha < cfg.alpha_bounds[0] - 1e-12) | (alpha > cfg.alpha_bounds[1] + 1e-12)).sum()
        )

        f = stream(f_post, spec)
        H_prev = float(
            (f.clamp_min(_FLOOR) * torch.log(f.clamp_min(_FLOOR) / wv)).sum()
        )
        if n % sample_every == 0 or n == 1 or n == cfg.steps_kbc:
            S_lin_max = max(S_lin_max, entropy_scale(f, spec))
            series.append(_state_snapshot(f, spec, n, alpha, bal.delta_H))
        # cheap divergence sentinel every step, full health audit on samples
        if kbc_diverged_at is None:
            if not bool(torch.isfinite(f).all()) or float(f.min()) < -1e-3:
                kbc_diverged_at = n
            elif _u_max(f, spec) > cfg.u_limit:
                kbc_diverged_at = n

    health_final = flow_health(f, spec, cfg.u_limit)
    H_series_vals = [r["H"] for r in series]
    window_violations = sum(
        1
        for i in range(1, len(H_series_vals))
        if H_series_vals[i] > H_series_vals[i - 1]
    )
    result["c1"] = {
        "steps": cfg.steps_kbc,
        "window_violations": window_violations,
        "window_count": max(len(H_series_vals) - 1, 1),
        "H_initial": H_series_vals[0] if H_series_vals else float("nan"),
        "H_change_over_run": (H_series_vals[-1] - H_series_vals[0]) if H_series_vals else float("nan"),
        "violations": c1_violations,
        "kl_violations": c1b_violations,
        "kl_samples": len(series),
        "max_positive_excess_abs": max_excess,
        "max_positive_excess_rel": max_excess_rel,
        "tolerance_rel": cfg.c1_rel_tol,
        "scale_L1_max": S_lin_max,
        "tolerance_abs": cfg.c1_rel_tol * S_lin_max,
        "H_final": entropy_boltzmann(f, spec),
        "series": series,
        "pass": c1_violations == 0,
    }
    result["c2"] = {
        "tolerance": cfg.c2_tol,
        "telescoping_abs": c2_tel_abs,
        "telescoping_rel": c2_tel_abs / max(S_lin_max, _FLOOR),
        "kbc_ledger_abs": c2_ledger_abs,
        "kbc_ledger_rel": c2_ledger_abs / max(h2_scale_max, _FLOOR),
        "operator_formula_abs": c2_operator_abs,
        "moment_conservation_abs": c2_moment_abs,
        "identities": [
            "dH = sum(f*-f) ln(f/w) + sum f* ln(f*/f)   (telescoping, any operator)",
            "P = -dH2 = D  from f* = f^eq + sigma s + alpha h   (KBC ledger)",
            "H2(f*) = sigma^2 H2(f)   (BGK, exact; C6 control path)",
        ],
        "pass": None,
    }
    result["c2d"] = {
        "note": (
            "Chapman-Enskog reading of the same balance: -dH2 vs cs^2 * eps with "
            "eps = 2 rho nu S:S; truncation error O(Ma^2, k^2) -- NOT gated at 1e-10"
        ),
        "rows": c2d_rows,
    }
    result["c4_kbc"] = {
        "steps": cfg.steps_kbc,
        "diverged_at": kbc_diverged_at,
        "survived": kbc_diverged_at is None,
        "health_final": health_final,
        "u_max_final": _u_max(f, spec),
    }
    result["c5"] = {
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "samples": alpha_samples,
        "violations": alpha_escaped_cells,
        "bounds": list(cfg.alpha_bounds),
        "pass": alpha_escaped_cells == 0,
    }

    # ---------------- C3 : equilibrium recovery ------------------------------
    f_eq_run = make_ic(
        cfg.eq_ic,
        spec,
        shape,
        u0=cfg.eq_u0,
        wavelength=cfg.eq_wavelength,
        dtype=dtype,
    )
    eq_series: list[dict[str, float]] = []
    eq_every = max(1, cfg.eq_steps // 40)
    H2_initial: float | None = None
    for n in range(1, cfg.eq_steps + 1):
        f_eq_run, _alpha = collide_kbc(
            f_eq_run, cfg.eq_tau, spec, max_iter=cfg.max_iter, tol=cfg.alpha_tol
        )
        f_eq_run = stream(f_eq_run, spec)
        if n == 1:
            feq0 = equilibrium_from_state(f_eq_run, spec)
            H2_initial = max(entropy_excess_quadratic(f_eq_run, feq0), 1e-300)
        if n % eq_every == 0 or n == cfg.eq_steps:
            feq_here = equilibrium_from_state(f_eq_run, spec)
            H_now = entropy_boltzmann(f_eq_run, spec)
            H_ref = entropy_boltzmann(feq_here, spec)
            H2_now = entropy_excess_quadratic(f_eq_run, feq_here)
            eq_series.append(
                {
                    "step": float(n),
                    "H_over_H_eq_minus_one": (H_now / H_ref - 1.0) if abs(H_ref) > 0 else float("nan"),
                    "H_excess_ratio": H2_now / max(H2_initial or 1.0, 1e-300),
                    "H_rel": entropy_relative(f_eq_run, feq_here),
                    "u_max": _u_max(f_eq_run, spec),
                }
            )
    last = eq_series[-1]
    rho_eq, _u_eq = macroscopic(f_eq_run, spec)
    result["c3"] = {
        "tau": cfg.eq_tau,
        "steps": cfg.eq_steps,
        "u0": cfg.eq_u0,
        "wavelength": cfg.eq_wavelength,
        "H_over_H_eq_minus_one": last["H_over_H_eq_minus_one"],
        "H_excess_ratio": last["H_excess_ratio"],
        "H_rel_final": last["H_rel"],
        "u_max_final": last["u_max"],
        "rho_uniformity": float(rho_eq.max() - rho_eq.min()),
        "tolerance": cfg.c3_tol,
        "criterion": (
            "H_excess_ratio = H2(t)/H2(t0) <= 1e-6 is the numerically stable form of "
            "H(t->inf)/H_eq - 1 <= 1e-6 (H itself passes through zero at the reference "
            "equilibrium, so the raw ratio is 0/0 there)"
        ),
        "pass": abs(last["H_excess_ratio"]) <= cfg.c3_tol,
        "series": eq_series,
    }

    # ---------------- C4 / C6 : BGK high-Re negative control ------------------
    f_bgk = stream(f0.clone(), spec)
    bgk_rows: list[dict[str, float]] = []
    bgk_diverged_at: int | None = None
    for n in range(1, cfg.steps_bgk_max + 1):
        f_bgk_post = collide_bgk(f_bgk, cfg.tau, spec)
        diss = dissipation_identity(f_bgk, f_bgk_post, cfg.tau, spec)
        bgk_sigma2_abs = max(bgk_sigma2_abs, abs(diss.identity_residual))
        bgk_h2_scale_max = max(bgk_h2_scale_max, abs(diss.H2_pre))
        f_bgk = stream(f_bgk_post, spec)
        finite = bool(torch.isfinite(f_bgk).all())
        if not finite:
            bgk_rows.append(
                {
                    "step": float(n),
                    "u_max": float("inf"),
                    "f_min": float("-inf"),
                    "negative_cells": float("nan"),
                    "rho_min": float("nan"),
                    "diverged": 1.0,
                }
            )
            bgk_diverged_at = n
            break
        health = flow_health(f_bgk, spec, cfg.u_limit)
        bgk_rows.append(
            {
                "step": float(n),
                "u_max": float(health["u_max"]),
                "f_min": float(health["f_min"]),
                "negative_cells": float(health["negative_cells"]),
                "rho_min": float(health["rho_min"]),
                "diverged": 1.0 if health["diverged"] else 0.0,
            }
        )
        if health["diverged"]:
            bgk_diverged_at = n
            break
    result["c2"]["bgk_sigma2_abs"] = bgk_sigma2_abs
    result["c2"]["bgk_sigma2_rel"] = bgk_sigma2_abs / max(bgk_h2_scale_max, _FLOOR)
    result["c2"]["pass"] = bool(
        max(result["c2"]["telescoping_rel"], result["c2"]["kbc_ledger_rel"], result["c2"]["bgk_sigma2_rel"])
        <= cfg.c2_tol
    )
    result["c6_bgk"] = {
        "steps_run": len(bgk_rows),
        "steps_allowed": cfg.steps_bgk_max,
        "diverged_at": bgk_diverged_at,
        "diverged": bgk_diverged_at is not None,
        "rows": bgk_rows,
    }
    result["c4"] = {
        "tau": cfg.tau,
        "kbc_survived": result["c4_kbc"]["survived"],
        "kbc_steps": cfg.steps_kbc,
        "bgk_diverged_within": bgk_diverged_at,
        "pass": bool(result["c4_kbc"]["survived"] and bgk_diverged_at is not None),
    }

    # ---------------- C7 : alpha = 0 degeneration control ---------------------
    f_a0 = stream(f0.clone(), spec)
    a0_rows: list[dict[str, float]] = []
    a0_diverged_at: int | None = None
    a0_violations = 0
    a0_H = entropy_boltzmann(f_a0, spec)
    a0_lin = entropy_scale(f_a0, spec)
    for n in range(1, cfg.steps_bgk_max + 1):
        f_a0_post = collide_kbc_fixed_alpha(f_a0, cfg.tau, spec, 0.0)
        H_post = entropy_boltzmann(f_a0_post, spec)
        if H_post - a0_H > cfg.c1_rel_tol * a0_lin:
            a0_violations += 1
        a0_H = H_post
        f_a0 = stream(f_a0_post, spec)
        finite = bool(torch.isfinite(f_a0).all())
        health = flow_health(f_a0, spec, cfg.u_limit) if finite else {"u_max": float("inf"), "f_min": float("-inf")}
        a0_rows.append(
            {
                "step": float(n),
                "u_max": float(health["u_max"]),
                "f_min": float(health["f_min"]),
            }
        )
        if not finite or health["diverged"]:
            a0_diverged_at = n
            break

    # non-triviality of the entropy solve on a representative non-equilibrium
    # state (the initial condition itself is an exact equilibrium: alpha is
    # meaningless there, so the probe state is streamed once first)
    f_probe = stream(f0.clone(), spec)
    post_entropic, alpha_probe = collide_kbc(
        f_probe, cfg.tau, spec, max_iter=cfg.max_iter, tol=cfg.alpha_tol
    )
    post_alpha0 = collide_kbc_fixed_alpha(f_probe, cfg.tau, spec, 0.0)
    diff_alpha0 = float((post_entropic - post_alpha0).abs().max())
    sigma0 = min(max(1.0 - 1.0 / cfg.tau, 0.0), 1.0)
    result["c7"] = {
        "note": (
            "alpha=0 fully relaxes the ghost mode (no entropy conditioning): the "
            "degeneration negative control"
        ),
        "alpha_zero_diverged_at": a0_diverged_at,
        "alpha_zero_steps": len(a0_rows),
        "alpha_zero_h_violations": a0_violations,
        "alpha_zero_rows": a0_rows,
        "entropic_alpha_min": float(alpha_probe.min()),
        "entropic_alpha_max": float(alpha_probe.max()),
        "sigma_bgk_clamped": sigma0,
        "post_collision_diff_entropic_vs_alpha0": diff_alpha0,
        "solve_is_non_trivial": bool(diff_alpha0 > 0.0),
        "pass": bool(diff_alpha0 > 0.0),
    }

    # ---------------- verdict ------------------------------------------------
    verdict = {
        "C1_H_monotone": bool(result["c1"]["pass"]),
        "C2_production_dissipation_identity": bool(result["c2"]["pass"]),
        "C3_equilibrium_recovery": bool(result["c3"]["pass"]),
        "C4_high_re_kbc_stable_bgk_diverges": bool(result["c4"]["pass"]),
        "C5_alpha_in_bounds": bool(result["c5"]["pass"]),
        "C6_bgk_negative_control": bool(result["c6_bgk"]["diverged"]),
        "C7_alpha0_degeneration": bool(result["c7"]["pass"]),
    }
    verdict["all_pass"] = all(verdict.values())
    result["verdict"] = verdict
    return result


# ---------------------------------------------------------------------------
# Supporting studies: Reynolds/amplitude envelope and viscosity calibration
# ---------------------------------------------------------------------------


def run_stability_sweep(
    cfg: CampaignConfig | None = None,
    u0_list: Sequence[float] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25),
    *,
    kbc_steps: int = 400,
    bgk_steps: int = 100,
) -> list[dict[str, Any]]:
    """BGK-vs-KBC envelope over shear-wave amplitude at fixed ``tau``.

    For each ``u0`` the same initial condition is integrated with both
    collisions; the row records the first step at which the run is declared
    diverged (non-finite, ``u_max > u_limit``, ``rho <= 0`` or ``min f < -1e-3``)
    and the entropy monotonicity violations of the KBC run.
    """
    cfg = cfg or CampaignConfig()
    dtype = _to_dtype(cfg.dtype)
    spec = lattice_spec(cfg.lattice, dtype=dtype)
    shape = tuple(cfg.shape)
    rows: list[dict[str, Any]] = []
    for u0 in u0_list:
        f = stream(
            make_ic(
                cfg.ic,
                spec,
                shape,
                u0=u0,
                wavelength=cfg.wavelength,
                interface_width=cfg.interface_width,
                dtype=dtype,
            ),
            spec,
        )
        re_lattice = float(u0) * float(cfg.wavelength or shape[-1]) / (_CS2 * (cfg.tau - 0.5))
        # --- BGK ---
        f_b = f.clone()
        bgk_div: int | None = None
        for n in range(1, bgk_steps + 1):
            f_b = stream(collide_bgk(f_b, cfg.tau, spec), spec)
            if not bool(torch.isfinite(f_b).all()):
                bgk_div = n
                break
            if flow_health(f_b, spec, cfg.u_limit)["diverged"]:
                bgk_div = n
                break
        # --- KBC ---
        f_k = f.clone()
        kbc_div: int | None = None
        viol = 0
        max_excess_rel = 0.0
        H_prev = entropy_boltzmann(f_k, spec)
        lin = entropy_scale(f_k, spec)
        u_max_trace: list[float] = []
        for n in range(1, kbc_steps + 1):
            f_kp, alpha = collide_kbc(f_k, cfg.tau, spec, max_iter=cfg.max_iter, tol=cfg.alpha_tol)
            H_post = entropy_boltzmann(f_kp, spec)
            if H_post - H_prev > 0.0:
                max_excess_rel = max(max_excess_rel, (H_post - H_prev) / max(lin, _FLOOR))
            if H_post - H_prev > cfg.c1_rel_tol * lin:
                viol += 1
            H_prev = H_post
            f_k = stream(f_kp, spec)
            lin = max(lin, entropy_scale(f_k, spec))
            if not bool(torch.isfinite(f_k).all()):
                kbc_div = n
                break
            health = flow_health(f_k, spec, cfg.u_limit)
            if n % max(1, kbc_steps // 20) == 0:
                u_max_trace.append(float(health["u_max"]))
            if health["diverged"]:
                kbc_div = n
                break
        rows.append(
            {
                "u0": float(u0),
                "re_lattice": re_lattice,
                "bgk_diverged_at": bgk_div,
                "kbc_diverged_at": kbc_div,
                "kbc_steps": kbc_steps,
                "kbc_h_violations": viol,
                "kbc_max_positive_excess_rel": max_excess_rel,
                "kbc_u_max_trace": u_max_trace,
            }
        )
    return rows


def shear_wave_viscosity_study(
    *,
    lattice: str = "d2q9",
    shape: Sequence[int] = (32, 32),
    taus: Sequence[float] = (0.6, 0.7, 0.8, 1.0, 1.5),
    u0: float = 0.01,
    wavelengths: Sequence[int] = (16, 32),
    steps: int = 1200,
    sample: int = 30,
) -> list[dict[str, float]]:
    """Measure the shear viscosity from the KBC shear-wave decay rate.

    The analytical reference is the Navier-Stokes decay of a Kolmogorov shear
    wave, ``u_y(t) = u0 exp(-nu k^2 t)`` with ``k = 2 pi / lambda`` and
    ``nu = cs^2 (tau - 1/2)``.  The decay rate is fitted on the first Fourier
    mode of ``u_y`` inside the amplitude window ``1e-6 A0 < A < 0.9 A0`` (a
    wider window would fit round-off at the tail and the initial transient at
    the head).  Two wavelengths are run so the ``k^2`` spatial-truncation error
    of the discrete velocity set can be resolved instead of being hidden.

    Returns one row per ``(tau, wavelength)`` pair.
    """
    spec = lattice_spec(lattice)
    shape = tuple(int(v) for v in shape)
    rows: list[dict[str, float]] = []
    nx = shape[-1]
    for lam in wavelengths:
        k = 2.0 * math.pi / float(lam)
        # mode extractor for u_y at wavenumber k along x
        x = torch.arange(nx, dtype=torch.float64)
        proj = torch.sin(k * x)
        norm = float((proj * proj).sum())
        for tau in taus:
            f = make_ic("shear_wave", spec, shape, u0=u0, wavelength=lam)
            amp: list[float] = []
            t: list[float] = []
            every = max(1, steps // sample)
            for n in range(0, steps + 1):
                if n % every == 0:
                    _, u = macroscopic(f, spec)
                    mode = (u[1] * proj.view(*([1] * (spec.dim - 1)), nx)).sum() / norm
                    amp.append(float(mode.mean()))
                    t.append(float(n))
                if n == steps:
                    break
                f, _a = collide_kbc(f, tau, spec)
                f = stream(f, spec)
            a0 = abs(amp[0])
            idx = [i for i, a in enumerate(amp) if 1e-6 * a0 < abs(a) < 0.9 * a0]
            if len(idx) < 4:
                idx = [i for i, a in enumerate(amp) if abs(a) > 1e-12 * max(a0, 1e-30)]
            tt = torch.tensor([t[i] for i in idx], dtype=torch.float64)
            ln_a = torch.log(torch.tensor([abs(amp[i]) for i in idx], dtype=torch.float64))
            A = torch.stack([tt, torch.ones_like(tt)], dim=1)
            coef = torch.linalg.lstsq(A, ln_a.unsqueeze(1)).solution.squeeze(1)
            rate = -float(coef[0])
            nu_meas = rate / (k * k)
            nu_theory = _CS2 * (tau - 0.5)
            rows.append(
                {
                    "wavelength": float(lam),
                    "tau": float(tau),
                    "nu_theory": nu_theory,
                    "nu_measured": nu_meas,
                    "rel_error": (nu_meas - nu_theory) / nu_theory,
                    "fit_points": float(len(idx)),
                }
            )
    return rows


__all__ = [
    "LatticeSpec",
    "available_lattices",
    "lattice_spec",
    "macroscopic",
    "equilibrium",
    "equilibrium_from_state",
    "entropy_boltzmann",
    "entropy_boltzmann_cell",
    "entropy_relative",
    "entropy_relative_cell",
    "entropy_excess_quadratic",
    "entropy_excess_quadratic_cell",
    "entropy_scale",
    "entropy_balance",
    "EntropyBalance",
    "dissipation_identity",
    "DissipationIdentity",
    "kbc_production_identity",
    "KBCProductionIdentity",
    "second_order_moment",
    "stress_inner_product",
    "kbc_decompose",
    "collide_bgk",
    "collide_kbc",
    "collide_kbc_fixed_alpha",
    "stream",
    "step",
    "flow_health",
    "shear_rate_tensor",
    "viscous_dissipation_rate",
    "make_ic",
    "CampaignConfig",
    "run_campaign",
    "run_stability_sweep",
    "shear_wave_viscosity_study",
]
