"""Verification face for D3Q27 collision operators: BGK vs KBC vs cumulant.

This module is *measurement*, not a new solver.  It provides the rigs behind the
mission ``p246w2-n01`` acceptance criteria so that "which collision operator"
becomes a measured quantity instead of folklore:

``A1``  moment identities of the discrete equilibrium
        ``Σf^eq = ρ``, ``Σf^eq c_i = ρu_i``,
        ``Σf^eq c_i c_j = ρu_i u_j + ρc_s²δ_ij``

``A2``  invertibility of the moment ↔ cumulant transforms and exact
        mass/momentum conservation of a collision

``A3``  Galilean-invariance defect of the deviatoric stress: a shear wave
        advected at uniform speed ``U`` must have the *same* shear-stress mode
        amplitude as the same wave at rest.  The observable is the Fourier
        amplitude of ``Π_xy^neq`` (phase invariant, so the uniform translation
        of the pattern is factored out analytically)

``A4``  realizability: covariance matrix ``κ_ij = Π_ij − ρu_iu_j`` of the
        post-collision distribution must stay positive semi-definite
        (``λ_min ≥ 0``) and the density positive

``A5``  high-Reynolds stability domain: ``(Ma, τ, operator) →`` steps survived
        on a fixed Taylor–Green grid

``A6``  physics fidelity: force-driven Poiseuille profile against the
        parabolic analytic solution

``A7``  negative control: single-rate cumulant (all relaxation rates equal to
        ``1/τ``, the moment-space degenerate limit of BGK) must *recover* the
        BGK defect scaling

Everything runs in float64.  Nothing here is tuned per operator; the same grid,
the same initial data and the same number of steps are used for all operators so
the comparison is controlled (see ``docs/CUMULANT_D3Q27_VERIFICATION.md``).

Definitions used (stated once, used everywhere)
-----------------------------------------------
**Galilean defect.**  Let ``Π_xy^neq = Σ_i f_i c_ix c_iy − ρ u_x u_y`` and let
``Â(t) = |FFT_k[Π_xy^neq]|`` be the amplitude of the ``k``-th Fourier mode of a
shear wave ``u_y = A sin kx`` riding on a uniform stream ``u_x = U``.  The defect
at Mach number ``Ma = U/c_s`` is

    Δ(Ma) = max_t |Â_Ma(t) − Â_0(t)| / Â_0(0)

A Galilean-invariant scheme gives ``Δ = O(ε_machine)``; BGK gives
``Δ ∝ Ma²``.  ``Â`` is a Fourier amplitude, so translating the pattern changes
only its phase: the defect measures the scheme, not the shift.

**Realizability.**  For a discrete velocity distribution the covariance
``κ_ij = Σ_i f_i (c_i − u)_i (c_i − u)_j`` must be positive semi-definite for *f*
to be representable by any continuous velocity density; a negative eigenvalue
means the state has already left the realizable cone, which is the earliest
detectable precursor of a blow-up.

Theory anchors
--------------
* Geier, Schönherr, Pasquali, Krafczyk, *Computers & Fluids* 108 (2015) — the
  cumulant transform implemented in :mod:`tensorlbm.cumulant`.
* Frapolli, Chikatamarla, Karlin, *Phys. Rev. E* 90, 043306 (2014) — entropic /
  KBC stabilisation and the role of the higher-order modes.
* Qian & Orszag (1993); Chen & Doolen (1998) — the O(Ma²) advection defect of
  the truncated-equilibrium BGK model.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch

from .cumulant import (
    _central_to_cumulant_geier,
    _cumulant_to_central_geier,
    _get_matrices,
    _to_central,
    _to_raw,
    collide_cumulant_d3q27,
    collide_cumulant_geier_d3q27,
)
from .d3q27 import C as C27
from .d3q27 import W as W27
from .d3q27 import collide_bgk27, equilibrium27, macroscopic27, stream27_roll
from .entropic_kbc import collide_kbc_d3q27

__all__ = [
    "CS2",
    "OPERATOR_NAMES",
    "GalileanResult",
    "StabilityResult",
    "collision_conservation_error",
    "covariance_spectrum",
    "fit_power_law_slope",
    "galilean_defect",
    "kinetic_energy",
    "make_operator",
    "moment_identity_errors",
    "poiseuille_profile",
    "shear_wave_amplitude_series",
    "single_collision_stress_defect",
    "stability_run",
    "stability_summary",
    "taylor_green_setup",
    "transform_roundtrip_error",
]

CS2 = 1.0 / 3.0
_C = C27.to(torch.float64)
_W = W27.to(torch.float64)
_OPP = torch.tensor(
    [int((C27 == -C27[q]).all(dim=1).nonzero()[0, 0]) for q in range(27)], dtype=torch.long
)


# ---------------------------------------------------------------------------
# Operator registry — one name, one closure, identical calling convention
# ---------------------------------------------------------------------------
def make_operator(
    name: str,
    tau: float,
    *,
    omega_b: float = 1.0,
    omega_odd: float = 1.0,
    omega_even: float = 1.0,
    kbc_max_iter: int = 24,
    kbc_tol: float = 1e-10,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``f -> f*`` for one of the collision operators under test.

    Names
    -----
    ``bgk``                  single-relaxation-time BGK (baseline)
    ``kbc``                  entropic KBC (``entropic_kbc.collide_kbc_d3q27``)
    ``cumulant``             Geier 2015 cumulant, per-order rates
                             (defaults ``ω_b = ω_3 = ω_4 = 1``)
    ``cumulant_single_rate`` negative control (A7): every rate ``= 1/τ``, the
                             moment-space degenerate limit of BGK
    ``cumulant_legacy``      legacy "cumulant" = central-moment collision, kept
                             as a second reference point
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5, got {tau}")
    if name == "bgk":
        return lambda f: collide_bgk27(f, tau)
    if name == "kbc":
        return lambda f: collide_kbc_d3q27(f, tau, max_iter=kbc_max_iter, tol=kbc_tol)
    if name == "cumulant":
        return lambda f: collide_cumulant_geier_d3q27(
            f, tau, omega_b=omega_b, omega_odd=omega_odd, omega_even=omega_even
        )
    if name == "cumulant_single_rate":
        w = 1.0 / tau
        return lambda f: collide_cumulant_geier_d3q27(
            f, tau, omega_b=w, omega_odd=w, omega_even=w
        )
    if name == "cumulant_legacy":
        return lambda f: collide_cumulant_d3q27(
            f, tau, omega_b=omega_b, omega_odd=omega_odd, omega_even=omega_even
        )
    if name == "no_collision":
        # Control: pure streaming.  Used only by the Galilean rig to show which
        # part of the defect survives when the collision is switched off.
        return lambda f: f
    raise ValueError(f"unknown operator {name!r}")


OPERATOR_NAMES = ("bgk", "kbc", "cumulant", "cumulant_single_rate")


# ---------------------------------------------------------------------------
# A1 — equilibrium moment identities
# ---------------------------------------------------------------------------
def _rel_err(num: torch.Tensor, den: torch.Tensor, floor: float = 1e-12) -> float:
    """Max relative error ``|num - den| / max(|den|, floor)``."""
    scale = torch.clamp(den.abs(), min=floor)
    return float(((num - den).abs() / scale).max().item())


def _rel_norm_err(num: torch.Tensor, den: torch.Tensor) -> float:
    """``max|num - den| / max|den|`` — relative error on the tensor scale.

    Used for transform round-trips: an element-wise relative error would divide
    by individual cumulants that are legitimately ~1e-3 of the scale and would
    report conditioning, not transform accuracy.
    """
    scale = float(den.abs().max().item())
    return float((num - den).abs().max().item() / max(scale, 1e-300))


def moment_identity_errors(
    n_samples: int = 20,
    *,
    seed: int = 20260211,
    rho_range: tuple[float, float] = (0.6, 1.6),
    u_max: float = 0.3,
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    """A1: verify ``Σf^eq = ρ``, ``Σf^eq c_i = ρu_i``, ``Σf^eq c_ic_j`` exactly.

    ``n_samples`` random ``(ρ, u)`` draws with ``|u| ≤ u_max`` (``Ma ≤ 0.52`` for
    ``c_s = 1/√3``).  Relative errors use a 1e-12 floor so a zero right-hand
    side (e.g. ``i ≠ j`` with ``u_y = 0``) is not reported as ``inf``.
    """
    gen = torch.Generator().manual_seed(seed)
    n = int(n_samples)
    rho = rho_range[0] + (rho_range[1] - rho_range[0]) * torch.rand(n, generator=gen, dtype=dtype)
    u = torch.randn(3, n, generator=gen, dtype=dtype)
    u = u / u.norm(dim=0, keepdim=True).clamp(min=1e-12)
    u = u * (u_max * torch.rand(n, generator=gen, dtype=dtype) ** (1.0 / 3.0))

    rho3 = rho.view(1, 1, n)
    feq = equilibrium27(
        rho3, u[0].view(1, 1, n), u[1].view(1, 1, n), u[2].view(1, 1, n)
    ).reshape(27, n)  # (27, n)
    c = _C.to(dtype).view(27, 3, 1)

    m0 = feq.sum(dim=0)
    m1 = (feq.unsqueeze(1) * c).sum(dim=0)  # (3, n)
    cc = _C.to(dtype)
    m2 = (
        feq.view(27, 1, 1, n) * cc[:, :, None, None] * cc[:, None, :, None]
    ).sum(dim=0)  # (3,3,n)

    uu = u.unsqueeze(1) * u.unsqueeze(0)  # (3,3,n)
    eye = torch.eye(3, dtype=dtype).view(3, 3, 1)
    rhs = rho.view(1, 1, n) * (uu + CS2 * eye)
    return {
        "n_samples": float(n),
        "u_max": float(u_max),
        "mass_rel_err": _rel_err(m0, rho),
        "momentum_rel_err": _rel_err(m1, rho.view(1, n) * u),
        "second_moment_rel_err": _rel_err(m2, rhs),
    }


# ---------------------------------------------------------------------------
# A2 — transform invertibility and collision conservation
# ---------------------------------------------------------------------------
def _random_positive_state(
    shape: tuple[int, ...] = (1, 1, 64),
    *,
    seed: int = 7,
    noise: float = 0.35,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Equilibrium + bounded random perturbation, guaranteed strictly positive."""
    gen = torch.Generator().manual_seed(seed)
    rho = torch.ones(shape, dtype=dtype)
    ux = 0.05 * torch.randn(shape, generator=gen, dtype=dtype)
    uy = 0.05 * torch.randn(shape, generator=gen, dtype=dtype)
    uz = 0.05 * torch.randn(shape, generator=gen, dtype=dtype)
    feq = equilibrium27(rho, ux, uy, uz)
    w = _W.to(dtype).view(27, *([1] * len(shape)))
    pert = noise * w * torch.randn((27, *shape), generator=gen, dtype=dtype)
    return torch.clamp(feq + pert, min=1e-6)  # type: ignore[return-value]


def transform_roundtrip_error(
    n_states: int = 8,
    *,
    seed: int = 11,
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    """A2 part 1: ``m → κ → C → κ → m → f`` round-trip error on random states.

    The three legs are reported separately so a regression can be localised:
    cumulant inverse, velocity unshift, population recovery.
    """
    M, M_inv = _get_matrices(torch.device("cpu"), dtype)
    err_round = 0.0
    err_unshift = 0.0
    err_cumulant = 0.0
    for s in range(n_states):
        f = _random_positive_state(seed=seed + s, dtype=dtype)
        rho, ux, uy, uz = macroscopic27(f)
        m = M @ f.reshape(27, -1)
        k = _to_central(m, ux, uy, uz)
        C = _central_to_cumulant_geier(k)
        k2 = _cumulant_to_central_geier(C)
        err_cumulant = max(err_cumulant, _rel_norm_err(k2, k))
        m2 = _to_raw(k2, ux, uy, uz)
        err_unshift = max(err_unshift, _rel_norm_err(m2, m))
        f2 = (M_inv @ m2).reshape(f.shape)
        err_round = max(err_round, _rel_norm_err(f2, f))
    return {
        "n_states": float(n_states),
        "cumulant_roundtrip_rel_err": err_cumulant,
        "unshift_rel_err": err_unshift,
        "population_roundtrip_rel_err": err_round,
    }


def collision_conservation_error(
    operator: str,
    tau: float,
    *,
    seed: int = 13,
    dtype: torch.dtype = torch.float64,
) -> dict[str, float]:
    """A2 part 2: mass and momentum drift of one collision on a random state."""
    f = _random_positive_state(seed=seed, dtype=dtype)
    f2 = make_operator(operator, tau)(f)
    m0a, m0b = f.sum(dim=0), f2.sum(dim=0)
    c = _C.to(dtype).view(27, 3, 1, 1, 1)
    ja = (f.unsqueeze(1) * c).sum(dim=0)
    jb = (f2.unsqueeze(1) * c).sum(dim=0)
    rho_scale = float(m0a.abs().max().item())
    return {
        "operator": operator,  # type: ignore[dict-item]
        "tau": float(tau),
        "mass_rel_err": _rel_err(m0b, m0a),
        "momentum_rel_err": _rel_err(jb, ja),
        # Momentum is O(Ma·ρ) while the moment transforms are O(ρ), so a
        # round-off of 1e-16·ρ shows up as 1e-16/Ma when normalised by |j|.
        # Both normalisations are reported so the criterion is unambiguous.
        "mass_abs_err_over_rho": float((m0b - m0a).abs().max().item() / rho_scale),
        "momentum_abs_err_over_rho": float((jb - ja).abs().max().item() / rho_scale),
    }


# ---------------------------------------------------------------------------
# A3 / A7 — Galilean invariance of the shear stress
# ---------------------------------------------------------------------------
def _neq_shear_stress_mode(f: torch.Tensor, k_index: int) -> float:
    """Fourier amplitude of ``Π_xy^neq`` at ``k_index`` along x.

    Population tensor axis order is ``(27, nz, ny, nx)``.  ``Π_xy^eq = ρu_xu_y``
    exactly for the second-order discrete equilibrium, so the deviatoric part is
    ``Σ f c_xc_y − ρu_xu_y``.
    """
    rho, ux, uy, _uz = macroscopic27(f)
    cxy = (_C[:, 0] * _C[:, 1]).to(f.dtype).view(27, 1, 1, 1)
    pi_neq = (f * cxy).sum(dim=0) - rho * ux * uy
    spec = torch.fft.rfft(pi_neq.mean(dim=(0, 1)), dim=-1)
    return float(spec[k_index].abs().item())


def shear_wave_amplitude_series(
    operator: str,
    tau: float,
    *,
    mach: float = 0.1,
    amp: float = 0.02,
    n: int = 32,
    nsteps: int = 200,
    k_index: int = 1,
    dtype: torch.dtype = torch.float64,
) -> dict[str, object]:
    """Advected shear wave: ``u = (U, A sin kx, 0)`` started on the CE manifold.

    The initial nonequilibrium is the Chapman–Enskog first-order response to the
    shear rate ``s = ∂_x u_y = A k cos kx``::

        f_i^(1) = K w_i c_ix c_iy s,     K = −2 (τ − ½) / c_s²

    which gives exactly ``Π_xy^neq = −2ρc_s²(τ − ½)s`` (the viscous stress the
    operator is supposed to produce), zero stress in the other components and no
    mass/momentum contamination.  Starting on the hydrodynamic manifold keeps
    the test about the *operator* rather than about the initial transient.
    """
    x = torch.arange(n, dtype=dtype) * (2.0 * math.pi / n)
    k = 2.0 * math.pi * k_index / n
    shape = (1, 1, n)
    uy = amp * torch.sin(k * x).view(shape)
    ux = torch.full(shape, float(mach) * math.sqrt(CS2), dtype=dtype)
    rho = torch.ones(shape, dtype=dtype)

    feq = equilibrium27(rho, ux, uy, torch.zeros_like(ux))
    s_rate = amp * k * torch.cos(k * x).view(shape)
    cxy = (_C[:, 0] * _C[:, 1]).to(dtype).view(27, 1, 1, 1)
    w = _W.to(dtype).view(27, 1, 1, 1)
    norm = float((w * cxy * cxy).sum().item())  # = c_s⁴ = 1/9
    k_coef = -2.0 * (tau - 0.5) / CS2
    f = feq + k_coef * w * cxy * s_rate.unsqueeze(0)

    pi_target = float(
        ((f - feq) * cxy).sum(dim=0).mean().item()
    )
    op = make_operator(operator, tau)
    amps = [_neq_shear_stress_mode(f, k_index)]
    ok = True
    for _ in range(nsteps):
        f = stream27_roll(op(f))
        if not torch.isfinite(f).all():
            ok = False
            break
        if float(f.min().item()) < -1e-3:
            ok = False
            break
        amps.append(_neq_shear_stress_mode(f, k_index))
    return {
        "operator": operator,
        "tau": float(tau),
        "mach": float(mach),
        "amp": float(amp),
        "amplitudes": amps,
        "finite": ok,
        "cs4_check": norm,
        "pi_xy_mean_initial": pi_target,
    }


def _decay_rate(series: Sequence[float]) -> float:
    """Least-squares decay rate ``λ > 0`` of ``ln A(t)`` over a positive series."""
    if len(series) < 3:
        return float("nan")
    a = torch.tensor(series, dtype=torch.float64)
    if float(a.min().item()) <= 0.0:
        return float("nan")
    t = torch.arange(len(series), dtype=torch.float64)
    y = torch.log(a)
    xm, ym = t.mean(), y.mean()
    return -float(((t - xm) * (y - ym)).sum().item() / ((t - xm) ** 2).sum().item())


def fit_power_law_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of ``ln y`` vs ``ln x`` (the ``Ma²`` check)."""
    pairs = [(xx, yy) for xx, yy in zip(xs, ys) if xx > 0 and yy > 0]
    if len(pairs) < 2:
        return float("nan")
    x = torch.log(torch.tensor([p[0] for p in pairs], dtype=torch.float64))
    y = torch.log(torch.tensor([p[1] for p in pairs], dtype=torch.float64))
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum().item())
    if denom == 0.0:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum().item() / denom)


@dataclass
class GalileanResult:
    """A3/A7 outcome for one operator."""

    operator: str
    tau: float
    machs: list[float] = field(default_factory=list)
    defects: list[float] = field(default_factory=list)
    decay_rates: list[float] = field(default_factory=list)
    slope: float = float("nan")
    peak_defect: float = float("nan")
    finite: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "operator": self.operator,
            "tau": self.tau,
            "machs": self.machs,
            "defects": self.defects,
            "decay_rates": self.decay_rates,
            "loglog_slope": self.slope,
            "defect_at_max_mach": self.peak_defect,
            "finite": self.finite,
        }


def single_collision_stress_defect(
    operator: str,
    tau: float,
    *,
    mach: float = 0.1,
    amp: float = 0.02,
    n: int = 32,
    k_index: int = 1,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Definition D1: Galilean defect of **one collision** on a uniform state.

    Same shear-wave state as :func:`shear_wave_amplitude_series` but the
    observable is the post-collision ``Π_xy^neq`` at a *single* cell-averaged
    level, compared between ``Ma`` and ``Ma = 0``.  No streaming is involved, so
    this isolates the collision operator itself.
    """
    def build(ma: float) -> torch.Tensor:
        x = torch.arange(n, dtype=dtype) * (2.0 * math.pi / n)
        k = 2.0 * math.pi * k_index / n
        shape = (1, 1, n)
        uy = amp * torch.sin(k * x).view(shape)
        ux = torch.full(shape, ma * math.sqrt(CS2), dtype=dtype)
        rho = torch.ones(shape, dtype=dtype)
        feq = equilibrium27(rho, ux, uy, torch.zeros_like(ux))
        s_rate = amp * k * torch.cos(k * x).view(shape)
        cxy = (_C[:, 0] * _C[:, 1]).to(dtype).view(27, 1, 1, 1)
        w = _W.to(dtype).view(27, 1, 1, 1)
        return feq + (-2.0 * (tau - 0.5) / CS2) * w * cxy * s_rate.unsqueeze(0)

    cxy = (_C[:, 0] * _C[:, 1]).to(dtype).view(27, 1, 1, 1)
    op = make_operator(operator, tau)
    out = []
    for ma in (0.0, mach):
        f = build(ma)
        f_post = op(f)
        rho, ux, uy, _uz = macroscopic27(f_post)
        out.append((f_post * cxy).sum(dim=0).mean() - (rho * ux * uy).mean())
    ref = float(out[0].abs().item())
    return float((out[1] - out[0]).abs().item() / max(ref, 1e-300))


def galilean_defect(
    operator: str,
    *,
    tau: float = 0.55,
    machs: Iterable[float] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3),
    amp: float = 0.02,
    n: int = 32,
    nsteps: int = 200,
    window: int = 0,
) -> GalileanResult:
    """A3/A7: defect ``Δ(Ma)`` of the shear-stress mode amplitude in a stream.

    ``window`` skips the first ``window`` steps.  This matters: the *transient*
    of the initial condition relaxes at a rate that differs between operators
    (measured: 14 % between BGK and cumulant over the first 400 steps), while
    the *transport* defect — the quantity that means "the moving frame has a
    different viscosity" — is a late-time property (3e-4 relative).  Report which
    window was used or the number is meaningless.
    """
    res = GalileanResult(operator=operator, tau=tau)
    ref = shear_wave_amplitude_series(operator, tau, mach=0.0, amp=amp, n=n, nsteps=nsteps)
    ref_series: list[float] = list(ref["amplitudes"])  # type: ignore[arg-type]
    ref0 = ref_series[0]
    res.decay_rates.append(_decay_rate(ref_series[window:]))
    for ma in machs:
        run = shear_wave_amplitude_series(operator, tau, mach=ma, amp=amp, n=n, nsteps=nsteps)
        series: list[float] = list(run["amplitudes"])  # type: ignore[arg-type]
        res.finite = res.finite and bool(run["finite"])
        m = min(len(series), len(ref_series))
        lo = max(0, min(window, m - 1))
        dev = max(abs(series[i] - ref_series[i]) for i in range(lo, m))
        res.machs.append(float(ma))
        res.defects.append(dev / ref0 if ref0 > 0 else float("nan"))
        res.decay_rates.append(_decay_rate(series[window:]))
    res.slope = fit_power_law_slope(res.machs, res.defects)
    res.peak_defect = res.defects[-1] if res.defects else float("nan")
    return res


# ---------------------------------------------------------------------------
# A4 — realizability (positive density, PSD covariance)
# ---------------------------------------------------------------------------
def covariance_spectrum(f: torch.Tensor) -> dict[str, float]:
    """Eigenvalues of the covariance ``κ_ij = Π_ij − ρu_iu_j`` (min over cells).

    ``λ_min < 0`` means the state cannot be a velocity distribution: the
    earliest algebraic precursor of a blow-up, visible long before ``f`` itself
    goes non-finite.
    """
    rho, ux, uy, uz = macroscopic27(f)
    cc = _C.to(f.dtype)
    nd = f.dim() + 2  # (pop, vel_a, vel_b, spatial...)
    pi = (
        f.unsqueeze(1).unsqueeze(2)
        * cc.view(27, 3, *([1] * (nd - 2)))
        * cc.view(27, 1, 3, *([1] * (nd - 3)))
    ).sum(dim=0)  # (3,3,nz,ny,nx)
    u = torch.stack([ux, uy, uz], dim=0)
    kappa = pi - rho.unsqueeze(0).unsqueeze(0) * u.unsqueeze(1) * u.unsqueeze(0)
    kappa = 0.5 * (kappa + kappa.transpose(0, 1))
    flat = kappa.reshape(3, 3, -1).permute(2, 0, 1)
    eig = torch.linalg.eigvalsh(flat)
    return {
        "lambda_min": float(eig[:, 0].min().item()),
        "lambda_max": float(eig[:, -1].max().item()),
        "rho_min": float(rho.min().item()),
        "f_min": float(f.min().item()),
        "trace_min": float(eig.sum(dim=1).min().item()),
    }


# ---------------------------------------------------------------------------
# A5 — high-Re stability domain on Taylor–Green
# ---------------------------------------------------------------------------
def taylor_green_setup(
    n: int = 32,
    *,
    u0: float = 0.05,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """3-D Taylor–Green vortex on a periodic ``[0, 2π]³`` box (stability case)."""
    x = torch.arange(n, dtype=dtype) * (2.0 * math.pi / n)
    z_, y_, x_ = torch.meshgrid(x, x, x, indexing="ij")
    ux = u0 * torch.sin(x_) * torch.cos(y_) * torch.cos(z_)
    uy = -u0 * torch.cos(x_) * torch.sin(y_) * torch.cos(z_)
    uz = torch.zeros_like(ux)
    rho = torch.ones_like(ux)
    return equilibrium27(rho, ux, uy, uz).contiguous()  # type: ignore[return-value]


def kinetic_energy(f: torch.Tensor) -> float:
    """Mean kinetic energy density ``½ρ|u|²``."""
    rho, ux, uy, uz = macroscopic27(f)
    return float((0.5 * rho * (ux * ux + uy * uy + uz * uz)).mean().item())


@dataclass
class StabilityResult:
    """A5 outcome for one ``(operator, τ)`` point."""

    operator: str
    tau: float
    re: float
    mach: float
    steps_requested: int = 0
    steps_completed: int = 0
    diverged: bool = False
    fail_step: int = -1
    kelvin_initial: float = float("nan")
    kelvin_final: float = float("nan")
    lambda_min: float = float("nan")
    rho_min: float = float("nan")
    realizability_violated: bool = False
    first_violation_step: int = -1
    elapsed_s: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def stability_run(
    operator: str,
    tau: float,
    *,
    n: int = 32,
    u0: float = 0.05,
    nsteps: int = 2000,
    check_every: int = 25,
    ke_blowup: float = 10.0,
    realizability_tol: float = -1e-12,
    f: torch.Tensor | None = None,
    track_realizability: bool = True,
    op_kwargs: dict | None = None,
    dtype: torch.dtype = torch.float64,
) -> StabilityResult:
    """A5/A4: run one ``(operator, τ)`` Taylor–Green case and record when it dies.

    Divergence is declared when ``f`` stops being finite, the density goes
    negative by more than ``1e-6``, or the mean kinetic energy exceeds
    ``ke_blowup`` times its initial value (TG kinetic energy decays in exact
    Navier–Stokes, so 10× growth is unconditionally a numerical event).

    ``Re = u0·2π/ν``, ``ν = c_s²(τ − ½)`` and ``Ma = u0/c_s``: lattice
    definitions, identical across operators on the same grid.
    """
    t0 = time.perf_counter()
    f_work = f.clone() if f is not None else taylor_green_setup(n, u0=u0, dtype=dtype)
    nu = CS2 * (tau - 0.5)
    res = StabilityResult(
        operator=operator,
        tau=float(tau),
        re=float(u0 * 2.0 * math.pi / nu),
        mach=float(u0 / math.sqrt(CS2)),
        steps_requested=nsteps,
        lambda_min=float("inf"),
        rho_min=float("inf"),
    )
    ke0 = kinetic_energy(f_work)
    res.kelvin_initial = ke0
    op = make_operator(operator, tau, **(op_kwargs or {}))
    lam_min = float("inf")
    rho_min = float("inf")
    for step in range(1, nsteps + 1):
        f_work = stream27_roll(op(f_work))
        if not torch.isfinite(f_work).all():
            res.diverged, res.fail_step = True, step
            break
        if step % check_every == 0 or step == 1:
            rho = macroscopic27(f_work)[0]
            rmin = float(rho.min().item())
            rho_min = min(rho_min, rmin)
            if rmin <= -1e-6:
                res.diverged, res.fail_step = True, step
                break
            if track_realizability:
                lam = covariance_spectrum(f_work)["lambda_min"]
                lam_min = min(lam_min, lam)
                if lam < realizability_tol and not res.realizability_violated:
                    res.realizability_violated = True
                    res.first_violation_step = step
            if kinetic_energy(f_work) > ke_blowup * ke0:
                res.diverged, res.fail_step = True, step
                break
        res.steps_completed = step
    res.lambda_min = lam_min if math.isfinite(lam_min) else float("nan")
    res.rho_min = rho_min if math.isfinite(rho_min) else float("nan")
    res.kelvin_final = kinetic_energy(f_work) if bool(torch.isfinite(f_work).all()) else float("nan")
    res.elapsed_s = time.perf_counter() - t0
    return res


def stability_summary(rows: Sequence[dict]) -> dict[str, dict]:
    """Collapse ``(operator, τ)`` stability rows into a per-operator domain edge.

    The selection criterion used by the report is:
    ``Re_max(op) = max Re over rows that completed every requested step``.
    Rows that died are kept as evidence of where the wall is.  No fitting: the
    sweep is the measurement.
    """
    out: dict[str, dict] = {}
    for row in rows:
        op = str(row["operator"])
        slot = out.setdefault(
            op,
            {
                "re_max_survived": 0.0,
                "re_min_died": float("inf"),
                "n_rows": 0,
                "died_after_realizability_violation": 0,
                "died_first": 0,
                "survived": 0,
            },
        )
        slot["n_rows"] += 1
        survived = (not row["diverged"]) and row["steps_completed"] >= row["steps_requested"]
        if survived:
            slot["survived"] += 1
            slot["re_max_survived"] = max(slot["re_max_survived"], float(row["re"]))
        else:
            slot["re_min_died"] = min(slot["re_min_died"], float(row["re"]))
            if row["realizability_violated"] and row["first_violation_step"] > 0:
                slot["died_after_realizability_violation"] += 1
            else:
                slot["died_first"] += 1
    for slot in out.values():
        if not math.isfinite(slot["re_min_died"]):
            slot["re_min_died"] = None
    base = out.get("bgk", {}).get("re_max_survived") or 0.0
    for op, slot in out.items():
        slot["re_max_ratio_vs_bgk"] = (
            slot["re_max_survived"] / base if base > 0 else float("nan")
        )
    return out


# ---------------------------------------------------------------------------
# A6 — force-driven Poiseuille channel vs the parabolic analytic profile
# ---------------------------------------------------------------------------
def _bounce_back_y(f_post: torch.Tensor, f_stream: torch.Tensor) -> torch.Tensor:
    """Half-way bounce-back on the two ``y`` walls; axis order ``(27, nz, ny, nx)``.

    The wall sits half a cell outside the first/last fluid node, so the incoming
    population that streaming brought in from outside is replaced by the
    post-collision opposite population of the *same* node:
    ``f_i(x_w, t+1) = f_{ī}^post(x_w, t)``.
    """
    out = f_stream
    lo_mask = _C[:, 1] > 0  # arrived from y = −1
    hi_mask = _C[:, 1] < 0  # arrived from y = ny
    out[lo_mask, :, 0, :] = f_post[_OPP[lo_mask], :, 0, :]
    out[hi_mask, :, -1, :] = f_post[_OPP[hi_mask], :, -1, :]
    return out


def poiseuille_profile(
    operator: str,
    tau: float,
    *,
    ny: int = 32,
    nx: int = 4,
    nz: int = 4,
    force: float = 2.0e-6,
    nsteps: int = 6000,
    sample_every: int = 100,
    conv_tol: float = 1e-11,
    dtype: torch.dtype = torch.float64,
) -> dict[str, object]:
    """A6: plane Poiseuille flow (body force along x, bounce-back ``y`` walls).

    Analytic steady solution with no-slip planes half a cell outside the first
    and last node, channel height ``H = ny``::

        u_x(Y) = g/(2ν) · Y (H − Y),    Y = y + ½

    The reported error is the max deviation normalised by the analytic peak —
    the standard LBM check that the operator reproduces the correct viscosity.
    """
    shape = (nz, ny, nx)
    zeros = torch.zeros(shape, dtype=dtype)
    f = equilibrium27(torch.ones(shape, dtype=dtype), zeros, zeros, zeros)
    op = make_operator(operator, tau)
    w = _W.to(dtype).view(27, 1, 1, 1)
    cx = _C[:, 0].to(dtype).view(27, 1, 1, 1)
    half_force = 0.5 * 3.0 * force * w * cx
    prof: torch.Tensor | None = None
    prev: torch.Tensor | None = None
    converged = False
    for step in range(1, nsteps + 1):
        f = f + half_force
        f_post = op(f)  # collide
        f_stream = stream27_roll(f_post)  # stream
        f = _bounce_back_y(f_post, f_stream) + half_force
        if sample_every and step % sample_every == 0:
            prof = macroscopic27(f)[1].mean(dim=(0, 2))
            if prev is not None and float((prof - prev).abs().max().item()) < conv_tol:
                converged = True
                break
            prev = prof.clone()
    if prof is None:
        prof = macroscopic27(f)[1].mean(dim=(0, 2))
    nu = CS2 * (tau - 0.5)
    y = torch.arange(ny, dtype=dtype) + 0.5
    h = float(ny)
    analytic = (force / (2.0 * nu)) * y * (h - y)
    scale = max(float(analytic.abs().max().item()), 1e-30)
    # Wall-convention-independent accuracy statement: fit a parabola to the
    # nodes and read the viscosity off its curvature, ν = g / (2 |c₂|).
    A = torch.stack([torch.ones_like(y), y, y * y], dim=1)
    coef = torch.linalg.lstsq(A, prof.unsqueeze(1)).solution.squeeze(1)
    c2 = float(coef[2].item())
    nu_fit = -force / (2.0 * c2) if c2 != 0.0 else float("nan")
    fit = A @ coef
    return {
        "operator": operator,
        "tau": float(tau),
        "nu": nu,
        "nu_from_curvature": nu_fit,
        "nu_rel_err": abs(nu_fit / nu - 1.0) if c2 != 0.0 else float("nan"),
        "force": float(force),
        "profile": [float(v) for v in prof],
        "analytic": [float(v) for v in analytic],
        "rel_err": float((prof - analytic).abs().max().item() / scale),
        "rel_err_vs_parabola_fit": float((prof - fit).abs().max().item() / scale),
        "rms_rel_err": float(((prof - analytic) ** 2).mean().sqrt().item() / scale),
        "converged": converged,
        "steps": step,
        "u_max": float(prof.max().item()),
        "u_max_analytic": float(analytic.max().item()),
    }