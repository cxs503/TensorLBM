"""Shan–Chen (SC) equation-of-state coexistence analysis — G3 ledger utilities.

Why this module exists
----------------------
``benchmarks/gaps.md`` entry **G3** records that the cavitation module
(``tensorlbm/cavitation.py``) shipped with a Shan–Chen coupling default of
``G = -5.5`` and reference densities ``ρ_l = 2.65 / ρ_v = 0.038`` annotated as
*"SC at G ≈ -5.5"*, although the SC equation of state has **no liquid/vapour
coexistence** at that coupling.  Auditing that entry needs three pieces of
machinery the library did not have:

1. the **mechanical pressure** of the lattice SC scheme, in the sign convention
   the implemented force actually uses (:func:`mechanical_pressure`);
2. a **spinodal** locator — the band where ``∂p/∂ρ < 0`` (:func:`spinodal_bounds`);
3. a **Maxwell equal-area** solver that either returns the coexistence pair or
   states numerically *why* no pair exists (:func:`maxwell_coexistence`).

Sign convention (locked by ledger entry G1, verified numerically in M4)
-----------------------------------------------------------------------
``multiphase.sc_single_component_force`` gathers the **backward** neighbour
``ψ(x − c_i)``, i.e.

    F = −G ψ(x) Σ_i w_i c_i ψ(x − c_i)  =  (G cs²/2) ∇(ψ²) + O(h²),

so **G > 0 is attractive** in this library and the mechanical pressure is

    p(ρ) = ρ cs² − (G/2) cs² ψ².                                            (1)

The classical Shan–Chen literature writes ``F = −G_c ψ ∇ψ`` with ``G_c < 0``
attractive and ``p = ρ cs² + (G_c/2) cs² ψ²``; the two conventions are related
by ``G_c = −G``.  Every function here takes the **library** convention ``G``
(attractive ⇒ ``G > 0``) and also reports ``G_c = −G``, so ledger numbers
written in the classic convention compare directly: classic ``G = −5.5`` ⇔
``G = +5.5`` here, and both give the ledger's ``p = ρ/3 − 5.5 ψ²/6``.

The equal-area rule used is the **correct** one, in specific-volume space,

    ∫_{v_l}^{v_v} [p(v) − p_sat] dv = 0,   v = 1/ρ,                        (2)

which is algebraically equivalent to equality of the chemical potentials
``μ(ρ_v) = μ(ρ_l)`` with ``μ(ρ) = ∫ dp/ρ`` (see :func:`chemical_potential`,
used as an independent correctness check in the tests).  A frequently seen
variant integrates ``p − p_sat`` over ``dρ`` instead of ``dv``; that is
dimensionally inconsistent and changes the coexistence pair by orders of
magnitude in the vapour density (quantified in
``docs/g_ledger_g3_cavitation_coexistence.md``).

No SciPy dependency: the solver uses a log-spaced EOS table, vectorised
bisection and trapezoidal quadrature, so the runtime dependency set
(torch / numpy / matplotlib) is unchanged.

All functions are pure-torch-compatible: ``psi_fn`` receives and returns a
``torch.Tensor`` (so the library pseudopotentials, or ``functools.partial`` of
them, work unchanged) and the results are plain floats, ready to be injected
into JSON artefacts without hand-copying.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

__all__ = [
    "CS2",
    "MaxwellResult",
    "chemical_potential",
    "classic_to_library",
    "FlatInterfaceResult",
    "coexistence_exists",
    "flat_interface_coexistence",
    "library_to_classic",
    "maxwell_coexistence",
    "mechanical_pressure",
    "relax_scmp",
    "spinodal_bounds",
]

CS2 = 1.0 / 3.0
"""Lattice speed of sound squared for the D2Q9/D3Q19 stencils."""


def classic_to_library(G_classic: float) -> float:
    """Convert a classic Shan–Chen coupling (``G < 0`` attractive) to this library's."""
    return -float(G_classic)


def library_to_classic(G: float) -> float:
    """Convert this library's coupling (``G > 0`` attractive) to the classic sign."""
    return -float(G)


def mechanical_pressure(
    rho: torch.Tensor,
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Mechanical pressure of the SC lattice scheme, Eq. (1), library convention.

    ``p = ρ cs² − (G/2) cs² ψ(ρ)²`` — identical to the classic expression
    ``p = ρ/3 + G_c ψ²/6`` with ``G_c = −G``.

    Args:
        rho: Density field (any shape).
        G: SC coupling in this library's sign convention (``G > 0`` attractive).
        psi_fn: Pseudopotential ψ(ρ).

    Returns:
        Pressure field, same shape as ``rho``.
    """
    psi = psi_fn(rho)
    return rho * CS2 - 0.5 * G * CS2 * psi**2


def chemical_potential(
    rho: torch.Tensor,
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    rho_ref: float = 1.0,
    n: int = 20000,
) -> torch.Tensor:
    """Chemical potential ``μ(ρ) = ∫_{ρ_ref}^{ρ} (1/ρ') dp/dρ' dρ'`` (up to a constant).

    Integrated as ``μ(ρ) = p(ρ)/ρ − p(ρ_ref)/ρ_ref + ∫_{ρ_ref}^{ρ} p/ρ'² dρ'``
    (log-spaced trapezoid), which is an independent route to the coexistence
    condition: two phases coexist iff ``p`` **and** ``μ`` match.  Used by the
    test suite to validate :func:`maxwell_coexistence` without re-using its
    equal-area integral.

    Args:
        rho: Target density (any shape).
        G: SC coupling, library convention.
        psi_fn: Pseudopotential.
        rho_ref: Reference density of the integration.
        n: Quadrature points between ``rho_ref`` and ``rho`` (upper bound).

    Returns:
        Chemical potential relative to ``rho_ref``, same shape as ``rho``.
    """
    rho = torch.as_tensor(rho, dtype=torch.float64)

    def mu_scalar(r: float) -> float:
        r = float(r)
        lo, hi = (min(r, rho_ref), max(r, rho_ref))
        u = torch.linspace(math.log(lo), math.log(hi), n, dtype=torch.float64)
        rr = torch.exp(u)
        p = mechanical_pressure(rr, G, psi_fn).to(torch.float64)
        integral = torch.trapezoid(p / rr, u)  # ∫ p/ρ² dρ = ∫ p/ρ dlnρ
        p_r = float(mechanical_pressure(torch.tensor(r), G, psi_fn))
        p_ref = float(mechanical_pressure(torch.tensor(rho_ref), G, psi_fn))
        sign = 1.0 if r >= rho_ref else -1.0
        # μ(r) − μ(ρ_ref) = [p/ρ](r) − [p/ρ](ρ_ref) + ∫_{ρ_ref}^{r} p/ρ² dρ
        return (p_r / r - p_ref / rho_ref) + sign * float(integral)

    flat = rho.flatten()
    out = torch.tensor([mu_scalar(float(r)) for r in flat], dtype=torch.float64)
    return out.reshape(rho.shape)


def _grid(rho_lo: float, rho_hi: float, n: int) -> torch.Tensor:
    if rho_lo <= 0 or rho_hi <= rho_lo or n < 64:
        msg = f"need 0 < rho_lo < rho_hi and n >= 64 (got {rho_lo}, {rho_hi}, {n})"
        raise ValueError(msg)
    return torch.exp(torch.linspace(math.log(rho_lo), math.log(rho_hi), n))


def spinodal_bounds(
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    rho_lo: float = 1e-3,
    rho_hi: float = 2e2,
    n: int = 8000,
) -> tuple[float, float] | None:
    """Return ``(ρ_spinodal_vapour, ρ_spinodal_liquid)`` or ``None``.

    The spinodal is the band where ``∂p/∂ρ < 0``; inside it the homogeneous
    state is mechanically unstable (the middle branch of the EOS loop).  A
    single van der Waals loop is assumed: the first and last negative-slope
    samples of the log-spaced density grid delimit the band.

    Args:
        G: SC coupling, library convention.
        psi_fn: Pseudopotential.
        rho_lo: Lower density of the scan.
        rho_hi: Upper density of the scan.
        n: Number of log-spaced samples.

    Returns:
        ``(rho_v, rho_l)`` spinodal densities, or ``None`` when ``∂p/∂ρ ≥ 0``
        everywhere on the scan (no phase separation possible).
    """
    rs = _grid(rho_lo, rho_hi, n)
    p = mechanical_pressure(rs, G, psi_fn)
    dp = torch.gradient(p, spacing=(rs,))[0]
    neg = torch.nonzero(dp < 0, as_tuple=False).flatten()
    if neg.numel() == 0:
        return None
    return float(rs[int(neg[0])]), float(rs[int(neg[-1])])


@dataclass(frozen=True)
class MaxwellResult:
    """Outcome of an equal-area (Maxwell) coexistence construction.

    ``found=False`` is a *physical* answer (the EOS admits no two-phase state at
    this coupling), not an error; ``reason`` always carries the numeric evidence
    for the decision, so the verdict can be audited without re-running.
    """

    found: bool
    G: float
    G_classic: float
    reason: str
    p_sat: float | None = None
    rho_vapour: float | None = None
    rho_liquid: float | None = None
    density_ratio: float | None = None
    rho_spinodal_vapour: float | None = None
    rho_spinodal_liquid: float | None = None
    area_residual: float | None = None
    p_min: float | None = None
    p_min_at_rho: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view (floats rounded, ``None`` preserved as ``null``)."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            out[key] = round(value, 12) if isinstance(value, float) else value
        return out


def maxwell_coexistence(
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    rho_lo: float = 1e-5,
    rho_hi: float = 2e2,
    n: int = 12000,
    n_psat: int = 200,
    n_quad: int = 400,
) -> MaxwellResult:
    """Maxwell equal-area coexistence for the SC equation of state.

    Solves ``p(ρ_v) = p(ρ_l) = p_sat`` together with the equal-area rule
    Eq. (2) on the two mechanically stable branches that bracket the spinodal.

    Args:
        G: SC coupling, library convention (``G > 0`` attractive).
        psi_fn: Pseudopotential ψ(ρ).
        rho_lo: Lower density of the EOS scan.
        rho_hi: Upper density of the EOS scan.
        n: Number of log-spaced EOS samples.
        n_psat: Number of trial saturation pressures used to bracket ``p_sat``.
        n_quad: Quadrature points per loop for the area integral.

    Returns:
        :class:`MaxwellResult`; ``found=False`` with ``reason`` when the EOS has
        no two-phase solution (no spinodal, no overlapping branch pressures, or
        no sign change of the equal-area integral).
    """
    rs = _grid(rho_lo, rho_hi, n)
    ps = mechanical_pressure(rs, G, psi_fn).to(torch.float64)
    dps = torch.gradient(ps, spacing=(rs,))[0]
    neg = torch.nonzero(dps < 0, as_tuple=False).flatten()
    p_min_idx = int(torch.argmin(ps))
    base: dict[str, Any] = {
        "G": float(G),
        "G_classic": library_to_classic(G),
        "p_min": float(ps[p_min_idx]),
        "p_min_at_rho": float(rs[p_min_idx]),
    }
    if neg.numel() == 0:
        return MaxwellResult(
            found=False,
            reason=(
                "no spinodal: dp/drho >= 0 on the whole scan "
                f"[{rho_lo:g}, {rho_hi:g}] — the EOS is monotone, so no "
                "liquid/vapour coexistence exists at this coupling"
            ),
            **base,
        )

    i_v, i_l = int(neg[0]), int(neg[-1])
    r_sp_v, r_sp_l = float(rs[i_v]), float(rs[i_l])
    base |= {"rho_spinodal_vapour": r_sp_v, "rho_spinodal_liquid": r_sp_l}

    # Branch pressure ranges: vapour branch r ∈ (rho_lo, r_sp_v], liquid branch
    # r ∈ [r_sp_l, rho_hi).  A common p_sat must lie in their overlap.
    p_vap = ps[: i_v + 1]
    p_liq = ps[i_l:]
    lo = max(float(p_vap.min()), float(p_liq.min()))
    hi = min(float(p_vap.max()), float(p_liq.max()))
    if not hi > lo:
        return MaxwellResult(
            found=False,
            reason=(
                "no pressure overlap between the vapour branch "
                f"[p ∈ {float(p_vap.min()):.6g}, {float(p_vap.max()):.6g}] and the liquid "
                f"branch [p ∈ {float(p_liq.min()):.6g}, {float(p_liq.max()):.6g}]: the loop "
                "is too deep to be closed by any saturation pressure"
            ),
            **base,
        )

    # --- Pre-computed tables: monotone branch inverses + cumulative ∫p/ρ² dρ ---
    # The equal-area integral Eq. (2) written in ρ-space is
    #     A(p_sat) = ∫_{ρ_v}^{ρ_l} p/ρ² dρ − p_sat (1/ρ_v − 1/ρ_l),
    # so with a cumulative table C(ρ) = ∫_{ρ_lo}^{ρ} p/ρ² dρ and the branch
    # inverses ρ_v(p_sat), ρ_l(p_sat) the whole construction is table lookups:
    # no per-trial quadrature and no per-trial root finding.
    log_rs = torch.log(rs)
    dln = log_rs[1:] - log_rs[:-1]
    p_over_rho = ps / rs
    d_c = 0.5 * (p_over_rho[1:] + p_over_rho[:-1]) * dln  # ∫ p/ρ² dρ = ∫ (p/ρ) dln ρ
    cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(d_c, 0)])

    def _monotone(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keep = torch.cat([torch.ones(1, dtype=torch.bool), x[1:] > x[:-1]])
        return x[keep], y[keep]

    p_vap_b, rho_vap_b = _monotone(ps[: i_v + 1].double(), rs[: i_v + 1].double())
    p_liq_b, rho_liq_b = _monotone(ps[i_l:].double(), rs[i_l:].double())

    def _interp(x: torch.Tensor, y: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of ``y(x)`` (both increasing) at ``xq``, clamped."""
        idx = torch.searchsorted(x, xq).clamp(1, x.numel() - 1)
        x0, x1 = x[idx - 1], x[idx]
        w = ((xq - x0) / (x1 - x0)).clamp(0.0, 1.0)
        return y[idx - 1] * (1.0 - w) + y[idx] * w

    def area(p_sat: torch.Tensor) -> torch.Tensor:
        """Equal-area integral A(p_sat) = ∫_{v_l}^{v_v} (p − p_sat) dv, vectorised."""
        r_v = _interp(p_vap_b, rho_vap_b, p_sat)
        r_l = _interp(p_liq_b, rho_liq_b, p_sat)
        c_v = _interp(rs.double(), cumulative, r_v)
        c_l = _interp(rs.double(), cumulative, r_l)
        return (c_l - c_v) - p_sat * (1.0 / r_v - 1.0 / r_l)

    trials = torch.linspace(lo, hi, n_psat, dtype=torch.float64)
    vals = area(trials)
    jump = torch.nonzero((vals[:-1] > 0) != (vals[1:] > 0), as_tuple=False).flatten()
    if jump.numel() == 0:
        return MaxwellResult(
            found=False,
            reason=(
                "no equal-area solution: the area integral Eq. (2) has no sign change over "
                f"the branch pressure overlap [{lo:.6g}, {hi:.6g}] (area in "
                f"[{float(vals.min()):.6e}, {float(vals.max()):.6e}]) — the liquid branch "
                "never becomes the stable coexistence state"
            ),
            **base,
        )

    k = int(jump[0])
    a_p, b_p = float(trials[k]), float(trials[k + 1])
    f_a = float(vals[k])
    for _ in range(100):
        mid = 0.5 * (a_p + b_p)
        f_mid = float(area(torch.tensor([mid], dtype=torch.float64))[0])
        if (f_a > 0.0) == (f_mid > 0.0):
            a_p, f_a = mid, f_mid
        else:
            b_p = mid
        if b_p - a_p <= 1e-14 * max(1.0, abs(mid)):
            break
    p_sat = 0.5 * (a_p + b_p)

    # Final branch roots from the exact EOS (not the table), for reporting accuracy.
    r_v = _bisect_scalar(
        lambda r: float(mechanical_pressure(torch.tensor(r), G, psi_fn)) - p_sat,
        rho_lo,
        r_sp_v,
    )
    r_l = _bisect_scalar(
        lambda r: float(mechanical_pressure(torch.tensor(r), G, psi_fn)) - p_sat,
        r_sp_l,
        rho_hi,
    )
    residual = float(area(torch.tensor([p_sat], dtype=torch.float64))[0])
    return MaxwellResult(
        found=True,
        G=float(G),
        G_classic=library_to_classic(G),
        reason="equal-area construction converged",
        p_sat=p_sat,
        rho_vapour=r_v,
        rho_liquid=r_l,
        density_ratio=r_l / r_v,
        rho_spinodal_vapour=r_sp_v,
        rho_spinodal_liquid=r_sp_l,
        area_residual=residual,
        p_min=base["p_min"],
        p_min_at_rho=base["p_min_at_rho"],
    )


def _bisect_scalar(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    max_iter: int = 200,
) -> float:
    """Scalar bisection requiring ``f(lo)`` and ``f(hi)`` to bracket a root."""
    f_lo = f(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo <= 1e-15 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def coexistence_exists(
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    **kwargs: Any,
) -> bool:
    """``True`` iff :func:`maxwell_coexistence` finds a two-phase solution."""
    return maxwell_coexistence(G, psi_fn, **kwargs).found


def relax_scmp(
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    nx: int = 64,
    ny: int = 64,
    rho0: float = 2.0,
    noise: float = 0.02,
    steps: int = 2000,
    tau: float = 1.0,
    seed: int = 0,
    forcing: str = "edm",
    device: str = "cpu",
) -> dict[str, Any]:
    """Relax a noisy uniform SCMP state and report the measured density extrema.

    Periodic single-component Shan–Chen relaxation: a uniform density plus 2 %
    noise either phase-separates (attractive coupling inside the coexistence
    window) or stays homogeneous / diverges.  Used by the G3 audit to test the
    *dynamic* prediction of the EOS analysis.

    Args:
        G: SC coupling, library convention (``G > 0`` attractive).
        psi_fn: Pseudopotential.
        nx, ny: Domain size.
        rho0: Mean density.
        noise: Relative amplitude of the uniform random initial perturbation.
        steps: Number of time steps.
        tau: BGK relaxation time.
        seed: RNG seed for the perturbation.
        forcing: Forcing scheme forwarded to
            :func:`tensorlbm.multiphase.collide_sc_single_component`.
        device: Torch device string.

    Returns:
        Dictionary with ``rho_max``, ``rho_min``, ``rho_mean``, ``density_ratio``,
        ``mass_drift``, ``diverged`` and the echo of the configuration.
    """
    from . import d2q9, solver  # noqa: PLC0415
    from .multiphase import collide_sc_single_component  # noqa: PLC0415

    gen = torch.Generator().manual_seed(seed)
    rho = rho0 * (1.0 + noise * (torch.rand((ny, nx), generator=gen) - 0.5))
    rho = rho.to(device)
    zeros = torch.zeros((ny, nx), device=device)
    f = d2q9.equilibrium(rho, zeros, zeros)
    mass0 = float(rho.sum().item())
    for _ in range(steps):
        f = collide_sc_single_component(f, G=G, tau=tau, psi_fn=psi_fn, forcing=forcing)
        f = solver.stream(f)
    rho_f, _, _ = d2q9.macroscopic(f)
    diverged = bool(torch.isnan(rho_f).any() or torch.isinf(rho_f).any())
    out: dict[str, Any] = {
        "G": float(G),
        "G_classic": library_to_classic(G),
        "nx": nx,
        "ny": ny,
        "rho0": rho0,
        "steps": steps,
        "diverged": diverged,
        "rho_max": None,
        "rho_min": None,
        "rho_mean": None,
        "density_ratio": None,
        "mass_drift": None,
    }
    if diverged:
        return out
    rho_max = float(rho_f.max().item())
    rho_min = float(rho_f.min().item())
    mass1 = float(rho_f.sum().item())
    out |= {
        "rho_max": rho_max,
        "rho_min": rho_min,
        "rho_mean": float(rho_f.mean().item()),
        "density_ratio": rho_max / max(rho_min, 1e-12),
        "mass_drift": abs(mass1 - mass0) / max(abs(mass0), 1e-12),
    }
    return out

@dataclass(frozen=True)
class FlatInterfaceResult:
    """Coexisting densities measured from a relaxed periodic flat interface.

    ``diverged=False`` with a density pair is the calibration result; a
    divergence is reported with the step at which the state became non-finite so
    the failure is auditable instead of silently dropped.
    """

    G: float
    G_classic: float
    forcing: str
    ny: int
    nx: int
    tau: float
    steps: int
    diverged: bool
    diverged_at_step: int | None = None
    rho_liquid: float | None = None
    rho_vapour: float | None = None
    rho_liquid_seed: float | None = None
    rho_vapour_seed: float | None = None
    p_liquid: float | None = None
    p_vapour: float | None = None
    spurious_u_max: float | None = None
    mass_drift: float | None = None
    spurious_u_history: Any = None
    spurious_u_profile: Any = None
    profile: Any = None

    def as_dict(self, keep_profile: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if key in ("profile", "spurious_u_profile") and not keep_profile:
                continue
            out[key] = round(value, 12) if isinstance(value, float) else value
        return out


def flat_interface_coexistence(
    G: float,
    psi_fn: Callable[[torch.Tensor], torch.Tensor],
    ny: int = 64,
    nx: int = 2,
    forcing: str = "edm",
    tau: float = 1.0,
    steps: int = 20000,
    seed_width: float = 4.0,
    seed_rho_liquid: float | None = None,
    seed_rho_vapour: float | None = None,
    average_over: int = 5000,
    u_probe_every: int = 0,
    rho_floor: float = 1e-6,
    device: str = "cpu",
) -> FlatInterfaceResult:
    """Measure the *scheme's own* coexisting densities on a periodic flat interface.

    This is the coexistence calibration the R4 route calls for (Shan & Chen 1993/
    1994): a fully periodic strip holds two planar interfaces, no walls and no
    body force, so a relaxed state has zero Laplace pressure and the plateaux are
    the coexistence densities of the *discretised* model.  Two things decide
    whether the run is informative, and both are done here because round 1 of
    ``XVAL-LBM-R4-COEXISTENCE`` got them wrong:

    1. **The initial state is the Maxwell binodal of the same EOS**, not a nominal
       pair.  Starting from an arbitrary ``(rho_l, rho_v)`` either shocks the
       interface (blow-up at large ``G``) or relaxes on a diffusion time that is
       much longer than the run, so the measured plateaux are initial-state
       artefacts.
    2. **The library collision operator is used** (:func:`tensorlbm.multiphase.
       collide_sc_single_component`) with its momentum-exact forcings
       (``edm``/``guo``/``velocity_shift``) rather than a hand-rolled velocity
       shift.  The hand-rolled variant in the round-1 driver divided the force by
       ``2*rho`` evaluated on the *unfloored* density and produced NaN once
       ``rho_v <~ 0.25``; the floor is applied inside the pseudopotential here.

    Args:
        G: SC coupling, library convention (> 0 attractive).
        psi_fn: Pseudopotential ψ(ρ).
        ny, nx: Domain size (``nx`` only needs to be wide enough for the 3-D
            stencil roll; 2 is enough).
        forcing: Forwarded to the collision operator (``edm``, ``guo`` or
            ``velocity_shift``).
        tau: BGK relaxation time.
        steps: Number of time steps.
        seed_width: tanh width of the initial interface, in cells.
        seed_rho_liquid, seed_rho_vapour: Seed plateaux; ``None`` takes the
            Maxwell binodal of the same EOS.
        average_over: Number of trailing steps averaged for the plateaux.
        u_probe_every: If > 0, record ``max|u|`` every that many steps (the
            spurious-current decay is the evidence that the state is static; a
            non-decaying ``max|u|`` means the plateaux are still moving).
        rho_floor: Density floor applied *inside* the pseudopotential.
        device: Torch device string.

    Returns:
        :class:`FlatInterfaceResult`.  ``diverged=True`` reports
        ``diverged_at_step`` and no densities.
    """
    from . import d2q9, solver  # noqa: PLC0415
    from .multiphase import collide_sc_single_component  # noqa: PLC0415

    if seed_rho_liquid is None or seed_rho_vapour is None:
        maxwell = maxwell_coexistence(G=G, psi_fn=psi_fn, rho_lo=1e-5, rho_hi=50.0, n=12000)
        if not maxwell.found:
            raise ValueError(
                f"no Maxwell coexistence at G={G} ({maxwell.reason}); pass explicit seeds"
            )
        seed_rho_liquid = float(maxwell.rho_liquid)  # type: ignore[arg-type]
        seed_rho_vapour = float(maxwell.rho_vapour)  # type: ignore[arg-type]

    def _psi(clipped: torch.Tensor) -> torch.Tensor:
        return psi_fn(torch.clamp(clipped, min=rho_floor))

    y = torch.arange(ny).view(-1, 1).double()
    prof = seed_rho_vapour + 0.5 * (seed_rho_liquid - seed_rho_vapour) * (
        torch.tanh((y - ny / 4) / seed_width) - torch.tanh((y - 3 * ny / 4) / seed_width)
    )
    rho = prof.expand(ny, nx).contiguous().to(device)
    zeros = torch.zeros_like(rho)
    f = d2q9.equilibrium(rho, zeros, zeros)
    mass0 = float(rho.sum().item())
    acc = torch.zeros_like(rho)
    nacc = 0
    u_hist: list[dict[str, float]] = []
    diverged_at: int | None = None
    want_avg = max(1, int(average_over))
    for step in range(steps):
        f = collide_sc_single_component(f, G=G, tau=tau, psi_fn=_psi, forcing=forcing)
        f = solver.stream(f)
        if not bool(torch.isfinite(f).all()):
            diverged_at = step
            break
        if u_probe_every and (step % u_probe_every == 0 or step == steps - 1):
            _, ux_p, uy_p = d2q9.macroscopic(f)
            u_hist.append({"step": step + 1,
                           "u_max": float(torch.sqrt(ux_p**2 + uy_p**2).max())})
        if step >= steps - want_avg:
            rho_t, _, _ = d2q9.macroscopic(f)
            acc += rho_t
            nacc += 1

    def _empty(**kw: Any) -> FlatInterfaceResult:
        return FlatInterfaceResult(G=G, G_classic=library_to_classic(G), forcing=forcing,
                                   ny=ny, nx=nx, tau=tau, steps=steps,
                                   diverged=True, diverged_at_step=diverged_at,
                                   rho_liquid_seed=seed_rho_liquid,
                                   rho_vapour_seed=seed_rho_vapour, **kw)

    if diverged_at is not None or nacc == 0:
        return _empty()

    rho_avg = acc / nacc
    rho_f, ux, uy = d2q9.macroscopic(f)
    liq = slice(ny // 2 - ny // 8, ny // 2 + ny // 8)
    vap = slice(0, max(2, ny // 8))
    rho_l = float(rho_avg[liq, :].mean())
    rho_v = float(rho_avg[vap, :].mean())
    p_l = float(mechanical_pressure(rho_avg[liq, :], G=G, psi_fn=_psi).mean())
    p_v = float(mechanical_pressure(rho_avg[vap, :], G=G, psi_fn=_psi).mean())
    return FlatInterfaceResult(
        G=G,
        G_classic=library_to_classic(G),
        forcing=forcing,
        ny=ny,
        nx=nx,
        tau=tau,
        steps=steps,
        diverged=False,
        rho_liquid=rho_l,
        rho_vapour=rho_v,
        rho_liquid_seed=seed_rho_liquid,
        rho_vapour_seed=seed_rho_vapour,
        p_liquid=p_l,
        p_vapour=p_v,
        spurious_u_max=float(torch.sqrt(ux**2 + uy**2).max()),
        mass_drift=abs(float(rho_f.sum().item()) - mass0) / max(abs(mass0), 1e-12),
        spurious_u_history=u_hist or None,
        spurious_u_profile=[float(v) for v in torch.sqrt(ux**2 + uy**2).max(dim=1).values],
        profile=[float(v) for v in rho_avg[:, 0]],
    )
