"""Usability criteria for coarse-mesh wall-modelled LES (node IN61-N06).

A wall model is only worth using at a given resolution if five things hold, and
each of them has to be *measured*, not asserted:

1. **Inversion self-consistency** — the friction-velocity inversion that turns a
   sampled off-wall velocity into ``u_tau`` must return the manufactured value
   to solver precision (target ``< 1e-8`` relative).
2. **Wall-stress self-consistency** — the ``tau_w`` the model reconstructs must
   agree with the ``(nu du/dn + tau_sgs)`` stress budget of the wall node
   (target ``< 2 %``).  Two independent readings are provided:
   :func:`wall_stress_budget` (the resolved-stencil budget the model applies,
   i.e. the finite-difference bias of the sampling interval) and
   :func:`dns_wall_stress_recovery` (an *a-priori* check: invert the velocity of
   a published DNS profile and compare the recovered ``tau_w`` with the DNS wall
   stress).
3. **Channel validation** — the log-layer slope of the modelled ``u+`` profile
   against the MKM1999 / Moser DNS tables at ``Re_tau = 180`` and ``590``
   (target ``< 5 %``).
4. **Grid convergence** — a Richardson convergence ratio ``R < 1`` on a
   three-grid ladder, or a ``< 5 %`` change between the finest pair.
5. **Negative controls** — wrong ``kappa``, a one-cell sampling offset and a
   disabled wall law must each break the gate (3/3).

Everything here is domain-side physics; no platform semantics.  The composite
wall laws implemented are Spalding's unified law (``kappa = 0.4``,
``A = 0.1108``) and Reichardt's correlation (``kappa = 0.41``), both written as
``y+ = f(u+)`` / ``u+ = f(y+)`` pairs so that inversion is unambiguous.

The empirical finding that motivated the default choice (see
``docs/wall_model_usability_in61_n06.md``) is that Spalding's classic constants
reproduce the DNS ``u+`` *level* to 1–3 % but are 8–19 % too steep across the
buffer/overlap band, while Reichardt's ``kappa = 0.41`` correlation meets the
5 % slope target at ``Re_tau = 587`` and the 2 % stress target at both
Reynolds numbers.  ``DEFAULT_LAW`` therefore exposes both and documents the
choice instead of hiding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

WALL_LAWS = ("spalding", "reichardt")
KAPPA_DNS = 0.41
SPALDING_KAPPA = 0.4
SPALDING_A = 0.1108
REICHARDT_KAPPA = 0.41

_I64 = torch.float64


def _as_tensor(value, dtype=_I64) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


@dataclass(frozen=True)
class WallLawParams:
    """Parameters of a composite wall law ``u+ = f(y+)``."""

    name: str
    kappa: float
    a: float | None = None

    def __post_init__(self) -> None:
        if self.name not in WALL_LAWS:
            raise ValueError(f"unknown wall law {self.name!r}; expected one of {WALL_LAWS}")
        if not (0.2 < self.kappa < 0.8):
            raise ValueError("kappa must lie in (0.2, 0.8)")
        if self.name == "spalding" and (self.a is None or not 0.0 < self.a < 1.0):
            raise ValueError("Spalding's law needs the damping constant a in (0, 1)")
        if self.name == "reichardt" and self.a is not None:
            raise ValueError("Reichardt's law has no damping constant")

    def to_dict(self) -> dict[str, float | str | None]:
        return {"name": self.name, "kappa": self.kappa, "a": self.a}


SPALDING = WallLawParams("spalding", SPALDING_KAPPA, SPALDING_A)
REICHARDT = WallLawParams("reichardt", REICHARDT_KAPPA)
DEFAULT_LAW = REICHARDT


# ---------------------------------------------------------------------------
# wall laws: y+ = f(u+) and u+ = f(y+)
# ---------------------------------------------------------------------------
def _spalding_y_plus(u_plus: torch.Tensor, law: WallLawParams) -> torch.Tensor:
    ku = law.kappa * u_plus
    return u_plus + law.a * (
        torch.exp(ku) - 1.0 - ku - 0.5 * ku.square() - ku.pow(3) / 6.0
    )


def _spalding_dy_plus_du_plus(u_plus: torch.Tensor, law: WallLawParams) -> torch.Tensor:
    ku = law.kappa * u_plus
    return 1.0 + law.a * (
        law.kappa * torch.exp(ku)
        - law.kappa
        - law.kappa**2 * u_plus
        - 0.5 * law.kappa**3 * u_plus.square()
    )


def _reichardt_u_plus(y_plus: torch.Tensor, law: WallLawParams) -> torch.Tensor:
    k = law.kappa
    return torch.log1p(k * y_plus) / k + 7.8 * (
        1.0 - torch.exp(-y_plus / 11.0) - (y_plus / 11.0) * torch.exp(-y_plus / 3.0)
    )


def _reichardt_du_plus_dy_plus(y_plus: torch.Tensor, law: WallLawParams) -> torch.Tensor:
    """Analytic ``du+/dy+`` of Reichardt's correlation."""
    k = law.kappa
    return (
        1.0 / (1.0 + k * y_plus)
        + 7.8
        * (
            torch.exp(-y_plus / 11.0) / 11.0
            - torch.exp(-y_plus / 3.0) / 11.0
            + (y_plus / 33.0) * torch.exp(-y_plus / 3.0)
        )
    )


def wall_law_y_plus(u_plus, law: WallLawParams = SPALDING) -> torch.Tensor:
    """Return ``y+ = f(u+)``; both laws are evaluated by inversion if needed."""
    u_plus = _as_tensor(u_plus).clamp_min(0.0)
    if law.name == "spalding":
        return _spalding_y_plus(u_plus, law)
    # Reichardt is given as u+(y+): invert the monotone map on ln(y+).
    lo = torch.full_like(u_plus, -60.0)
    hi = torch.full_like(u_plus, 20.0)
    target = u_plus
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        value = _reichardt_u_plus(torch.exp(mid), law)
        below = value < target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return torch.exp(0.5 * (lo + hi))


def wall_law_u_plus(y_plus, law: WallLawParams = SPALDING) -> torch.Tensor:
    """Return ``u+ = f(y+)``.

    Reichardt's correlation is explicit in ``y+``.  Spalding's law is given as
    ``y+ = Y(u+)`` with ``Y`` strictly increasing, so it is inverted by Newton
    on ``u+`` (analytic derivative) inside a ``[0, 90]`` bracket that keeps
    ``exp(kappa u+)`` in double range for ``y+ <= 1e5``.  The iteration is
    driven to a *relative* ``y+`` residual of ``1e-15``: stopping on the step
    size is not enough, because ``dy+/du+ ~ 4e5`` at ``y+ = 1e5`` amplifies a
    ``1e-14`` step error into a ``1e-8`` friction-velocity error.
    """
    y_plus = _as_tensor(y_plus).clamp_min(0.0)
    if law.name == "reichardt":
        return _reichardt_u_plus(y_plus, law)
    low = torch.zeros_like(y_plus)
    high = torch.full_like(y_plus, 90.0)
    u_plus = torch.where(
        y_plus > 1.0,
        (torch.log(y_plus.clamp_min(1.0)) / law.kappa + 5.2).clamp(1.0, 80.0),
        y_plus.clamp(0.0, 80.0),
    )
    scale = y_plus.clamp_min(1e-300)
    for _ in range(80):
        value = _spalding_y_plus(u_plus, law)
        slope = _spalding_dy_plus_du_plus(u_plus, law)
        step = (value - y_plus) / slope.clamp_min(1e-300)
        candidate = (u_plus - step).clamp(0.0, 90.0)
        below = value < y_plus
        low = torch.where(below, u_plus, low)
        high = torch.where(below, high, u_plus)
        outside = (candidate <= low) | (candidate >= high)
        candidate = torch.where(outside, 0.5 * (low + high), candidate)
        u_plus = candidate
        residual = (_spalding_y_plus(u_plus, law) - y_plus).abs() / scale
        if bool((residual <= 1e-15).all()):
            break
    return u_plus


def wall_law_du_plus_dy_plus(y_plus, law: WallLawParams = SPALDING) -> torch.Tensor:
    """Return ``du+/dy+`` at ``y+``."""
    y_plus = _as_tensor(y_plus).clamp_min(1e-300)
    if law.name == "reichardt":
        return _reichardt_du_plus_dy_plus(y_plus, law)
    u_plus = wall_law_u_plus(y_plus, law)
    return 1.0 / _spalding_dy_plus_du_plus(u_plus, law)


# ---------------------------------------------------------------------------
# friction-velocity inversion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrictionVelocitySolution:
    """Result of the wall-law inversion (all diagnostics kept for the log)."""

    u_tau: torch.Tensor
    iterations: int
    residual: float
    converged: bool
    method: str
    law: WallLawParams

    def to_dict(self) -> dict[str, object]:
        return {
            "u_tau_mean": float(self.u_tau.mean().item()),
            "iterations": self.iterations,
            "residual_max": self.residual,
            "converged": self.converged,
            "method": self.method,
            "law": self.law.to_dict(),
        }


def _inversion_residual(
    u_tau: torch.Tensor,
    tangential: torch.Tensor,
    distance: torch.Tensor,
    nu: float,
    law: WallLawParams,
) -> torch.Tensor:
    """``g(u_tau) = u+(y+ ) - u_t/u_tau``; strictly increasing in ``u_tau``."""
    y_plus = distance * u_tau / nu
    return wall_law_u_plus(y_plus, law) - tangential / u_tau


def _inversion_system(
    u_plus: torch.Tensor,
    tangential: torch.Tensor,
    distance: torch.Tensor,
    nu: torch.Tensor,
    law: WallLawParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(g, dg/du+)`` with ``g`` increasing in ``u+`` and ``g = 0`` at the root.

    The inversion is written in the ``u+`` variable.  With
    ``y+ = y u_t /(nu u+)``:

    * Spalding: ``g = Y(u+) - y+`` (increasing, since the subtracted term falls
      as ``1/u+``);
    * Reichardt: ``g = u+ - u+_R(y+)`` (the law is explicit in ``y+``, and this
      form is decreasing, so the sign is flipped to keep ``g`` increasing).

    Solving in ``u+`` rather than in ``u_tau`` costs two cheap law evaluations
    per Newton step and never needs the inner ``y+ -> u+`` inversion, which is
    what makes the per-step wall model affordable.
    """
    y_plus = distance * tangential / (nu * u_plus)
    if law.name == "spalding":
        g = _spalding_y_plus(u_plus, law) - y_plus
        dg = _spalding_dy_plus_du_plus(u_plus, law) + y_plus / u_plus
        return g, dg
    g = u_plus - _reichardt_u_plus(y_plus, law)
    dy_plus_du_plus = -y_plus / u_plus
    dg = 1.0 - _reichardt_du_plus_dy_plus(y_plus, law) * dy_plus_du_plus
    return g, dg


def solve_friction_velocity(
    tangential_speed,
    wall_distance,
    nu: float,
    *,
    law: WallLawParams = DEFAULT_LAW,
    tol: float = 1e-13,
    max_iter: int = 60,
) -> FrictionVelocitySolution:
    """Invert the composite wall law for ``u_tau`` (safeguarded Newton in ``u+``).

    Solves ``u+ (y u_tau / nu) = u_t / u_tau``, i.e. the single monotone
    equation :func:`_inversion_system`, and returns ``u_tau = u_t / u+``.  The
    bracket ``u+ in (0, 90]`` is kept as a safeguard and a bisection fallback
    catches any Newton overshoot, so the solve is overflow-free even for
    velocities that are four decades away from the log layer.  Manufactured
    wall-law states are recovered to ``~1e-15`` relative error.
    """
    nu_t = nu if torch.is_tensor(nu) else torch.as_tensor(nu, dtype=_I64)
    if bool((nu_t <= 0.0).any()):
        raise ValueError("nu must be positive")
    u_t = _as_tensor(tangential_speed).clamp_min(0.0)
    y = _as_tensor(wall_distance)
    if y.ndim and y.shape != u_t.shape:
        y = y.expand_as(u_t)
    nu_t = nu_t.to(dtype=u_t.dtype)
    if nu_t.ndim and nu_t.shape != u_t.shape:
        nu_t = nu_t.expand_as(u_t)
    if bool((y <= 0.0).any()):
        raise ValueError("wall_distance must be positive")
    active = u_t > 0.0
    safe_t = torch.where(active, u_t, torch.ones_like(u_t))

    # initial guess: three sweeps of the fully turbulent log law
    u_plus = torch.full_like(u_t, 12.0)
    for _ in range(3):
        y_guess = y * safe_t / (nu_t * u_plus)
        u_plus = (torch.log(y_guess.clamp_min(1e-12)) / law.kappa + 5.2).clamp(1.0, 60.0)

    low = torch.full_like(u_t, 1e-12)
    high = torch.full_like(u_t, 90.0)
    residual = math.inf
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        g, dg = _inversion_system(u_plus, safe_t, y, nu_t, law)
        g = torch.where(active, g, torch.zeros_like(g))
        step = torch.where(active, g / dg.clamp_min(1e-300), torch.zeros_like(g))
        candidate = (u_plus - step).clamp(1e-12, 90.0)
        below = g < 0.0
        low = torch.where(below, torch.maximum(low, u_plus), low)
        high = torch.where(below, high, torch.minimum(high, u_plus))
        outside = (candidate <= low) | (candidate >= high) | ~torch.isfinite(candidate)
        candidate = torch.where(outside, 0.5 * (low + high), candidate)
        change = ((candidate - u_plus) / u_plus).abs()
        u_plus = candidate
        residual = float(change.max().item())
        if residual < tol:
            converged = True
            break
    u_tau = torch.where(active, u_t / u_plus, torch.zeros_like(u_t))
    return FrictionVelocitySolution(
        u_tau=u_tau,
        iterations=iteration,
        residual=residual,
        converged=converged or residual < 10.0 * tol,
        method="safeguarded_newton_in_u_plus",
        law=law,
    )


def solve_friction_velocity_bisection(
    tangential_speed,
    wall_distance,
    nu: float,
    *,
    law: WallLawParams = DEFAULT_LAW,
    iterations: int = 200,
) -> torch.Tensor:
    """Non-vectorised-in-bracket reference solver (cross-check for the Newton)."""
    nu_b = nu if torch.is_tensor(nu) else torch.as_tensor(nu, dtype=_I64)
    if bool((nu_b <= 0.0).any()):
        raise ValueError("nu must be positive")
    u_t = _as_tensor(tangential_speed).clamp_min(0.0)
    y = _as_tensor(wall_distance)
    if y.ndim and y.shape != u_t.shape:
        y = y.expand_as(u_t)
    lo = torch.full_like(u_t, 1e-300)
    hi = torch.full_like(u_t, 0.0)
    scale = torch.where(u_t > 0.0, u_t, torch.ones_like(u_t))
    hi = scale * 10.0
    # make sure the bracket contains the root
    for _ in range(20):
        g = _inversion_residual(hi, u_t, y, nu_b, law)
        hi = torch.where(g < 0.0, hi * 10.0, hi)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        g = _inversion_residual(mid, u_t, y, nu_b, law)
        lo = torch.where(g < 0.0, mid, lo)
        hi = torch.where(g < 0.0, hi, mid)
    u_tau = 0.5 * (lo + hi)
    return torch.where(u_t > 0.0, u_tau, torch.zeros_like(u_t))


def manufactured_wall_state(
    u_tau_true: float,
    y_plus,
    nu: float,
    *,
    law: WallLawParams = DEFAULT_LAW,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact wall-law state: ``(u_t = u_tau u+, y = y+ nu/u_tau)``."""
    y_plus = _as_tensor(y_plus)
    u_plus = wall_law_u_plus(y_plus, law)
    u_t = u_tau_true * u_plus
    y = y_plus * nu / u_tau_true
    return u_t, y


def inversion_relative_error(u_tau_recovered, u_tau_true) -> torch.Tensor:
    u_tau_recovered = _as_tensor(u_tau_recovered)
    u_tau_true = _as_tensor(u_tau_true)
    return (u_tau_recovered - u_tau_true).abs() / u_tau_true


# ---------------------------------------------------------------------------
# criterion 2: wall-stress budget
# ---------------------------------------------------------------------------
def wall_stress_residual(tau_w, viscous_stress, sgs_stress) -> torch.Tensor:
    """``|tau_w - (nu du/dn + tau_sgs)| / tau_w`` (elementwise)."""
    tau_w = _as_tensor(tau_w)
    viscous = _as_tensor(viscous_stress)
    sgs = _as_tensor(sgs_stress)
    if bool((tau_w <= 0.0).any()):
        raise ValueError("tau_w must be positive")
    return (tau_w - (viscous + sgs)).abs() / tau_w


def wall_stress_budget(
    u1,
    u2,
    y1,
    y2,
    tau_w,
    nu: float,
    *,
    law: WallLawParams = DEFAULT_LAW,
) -> dict[str, object]:
    """Resolved-stencil stress budget at the wall node.

    The wall model applies ``tau_w`` and reconstructs the near-wall profile with
    the same law, so the stress the model attributes to the resolved stencil is
    ``(nu + nu_t) (u2 - u1)/(y2 - y1)`` with the *law-consistent* eddy viscosity
    ``nu_t = tau_w / (du/dn)|_law(y_mid) - nu``.  The residual therefore
    measures the finite-difference bias of the sampling interval — the quantity
    a one-cell stencil at coarse ``dy+`` cannot keep below 2 % — and any
    sampling/bookkeeping slip (offset node, mismatched distance) shows up as a
    large residual.
    """
    u1 = _as_tensor(u1)
    u2 = _as_tensor(u2)
    y1 = _as_tensor(y1)
    y2 = _as_tensor(y2)
    tau_w = _as_tensor(tau_w)
    if bool((y2 <= y1).any()):
        raise ValueError("sampling requires y2 > y1 > 0")
    gradient_fd = (u2 - u1) / (y2 - y1)
    y_mid = 0.5 * (y1 + y2)
    u_tau = tau_w.sqrt()
    gradient_law = wall_law_du_plus_dy_plus(y_mid * u_tau / nu, law) * u_tau.square() / nu
    nu_t = tau_w / gradient_law.clamp_min(1e-300) - nu
    viscous = nu * gradient_fd
    sgs = nu_t * gradient_fd
    residual = wall_stress_residual(tau_w, viscous, sgs)
    return {
        "gradient_fd": gradient_fd,
        "gradient_law": gradient_law,
        "nu_t_law": nu_t,
        "residual": residual,
        "max_residual": float(residual.max().item()) if residual.numel() else 0.0,
        "fd_bias": float((gradient_fd / gradient_law - 1.0).abs().max().item()),
    }


def dns_wall_stress_recovery(
    profile: "MkmProfile",
    law: WallLawParams,
    *,
    sample_y_plus=(30.0, 50.0, 100.0, 200.0),
) -> dict[str, object]:
    """A-priori ``tau_w`` recovery: invert the DNS velocity, compare to DNS.

    The DNS profile is normalised by ``u_tau``, so the inversion is run in wall
    units (``nu = 1``); a perfect wall law returns ``u_tau = 1`` and the
    residual ``|u_tau - 1|`` is exactly the ``tau_w`` error ``|tau_w -
    (nu du/dn + tau_sgs)|/tau_w`` the model would carry at that sample.
    """
    if profile.re_tau <= 0.0:
        raise ValueError("profile has no Re_tau")
    samples = []
    for y_plus in sample_y_plus:
        if y_plus > 0.6 * profile.re_tau:
            continue
        index = int(torch.argmin((profile.y_plus - y_plus).abs()).item())
        yp = float(profile.y_plus[index].item())
        up = float(profile.u_plus[index].item())
        solution = solve_friction_velocity(up, yp, 1.0, law=law)
        u_tau = float(solution.u_tau.item())
        samples.append(
            {
                "y_plus": yp,
                "u_plus": up,
                "u_tau_recovered": u_tau,
                "tau_w_rel_error": abs(u_tau - 1.0),
            }
        )
    worst = max((s["tau_w_rel_error"] for s in samples), default=float("nan"))
    return {
        "re_tau": profile.re_tau,
        "law": law.to_dict(),
        "samples": samples,
        "max_tau_w_rel_error": worst,
        "mean_tau_w_rel_error": (
            sum(s["tau_w_rel_error"] for s in samples) / len(samples) if samples else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# criterion 3: channel validation against MKM1999 DNS
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MkmProfile:
    """Verbatim Moser/Kim/Mansour 1999 channel mean profile."""

    re_tau: float
    y_plus: torch.Tensor
    u_plus: torch.Tensor
    du_plus_dy_plus: torch.Tensor
    source: str = ""

    def band(self, lower: float, upper: float) -> torch.Tensor:
        return (self.y_plus >= lower) & (self.y_plus <= upper)


_MEANS_COLUMNS = 7


def load_mkm_profile(path: str | Path) -> MkmProfile:
    """Parse a MKM1999 ``chanXXX.means`` table (verbatim public file).

    Columns: ``y/h, y+, Umean+, dUmean/dy, Wmean, dWmean/dy, Pmean`` with the
    DNS normalisation ``U_tau, h``; the shear column is converted to
    ``dU+/dy+ = (dUmean/dy) / Re_tau``.
    """
    path = Path(path)
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != _MEANS_COLUMNS:
            continue
        try:
            rows.append([float(value) for value in parts])
        except ValueError:
            continue
    if len(rows) < 8:
        raise ValueError(f"{path} does not look like an MKM1999 mean-profile table")
    data = torch.tensor(rows, dtype=_I64)
    y_plus = data[:, 1]
    re_tau = float(y_plus.max().item())
    return MkmProfile(
        re_tau=re_tau,
        y_plus=y_plus,
        u_plus=data[:, 2],
        du_plus_dy_plus=data[:, 3] / re_tau,
        source=str(path),
    )


def log_layer_slope(
    y_plus,
    u_plus,
    *,
    lower: float = 30.0,
    upper: float = 100.0,
    estimator: str = "median_local",
    du_plus_dy_plus=None,
) -> float:
    """Estimate the von Karman constant in a log-layer band.

    ``estimator="median_local"``: ``kappa = median(1 / (y+ du+/dy+))`` — local
    slopes, immune to the curvature bias that a single straight-line fit
    absorbs at the band edges.  ``estimator="band_fit"``: least squares of
    ``u+`` against ``ln y+`` (the classical construction, reported as a
    sensitivity).  ``du_plus_dy_plus`` is required for the local estimator.
    """
    y_plus = _as_tensor(y_plus)
    u_plus = _as_tensor(u_plus)
    band = (y_plus >= lower) & (y_plus <= upper)
    if int(band.sum().item()) < 4:
        raise ValueError(
            f"log-layer band [{lower}, {upper}] has fewer than 4 DNS nodes; "
            "the channel has no resolved overlap layer there"
        )
    yb = y_plus[band]
    ub = u_plus[band]
    if estimator == "band_fit":
        x = torch.log(yb)
        x_mean = x.mean()
        y_mean = ub.mean()
        slope = ((x - x_mean) * (ub - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        return float(1.0 / slope)
    if estimator != "median_local":
        raise ValueError("estimator must be 'median_local' or 'band_fit'")
    if du_plus_dy_plus is None:
        raise ValueError("median_local needs du_plus_dy_plus")
    gradient = _as_tensor(du_plus_dy_plus)[band]
    local = 1.0 / (yb * gradient)
    return float(torch.median(local))


def log_layer_band(profile: MkmProfile, *, lower: float = 30.0, fraction: float = 0.3) -> tuple[float, float]:
    """Band ``[30, fraction*Re_tau]`` — the widest overlap band below the wake."""
    return lower, fraction * profile.re_tau


def channel_slope_deviation(
    profile: MkmProfile,
    law: WallLawParams,
    *,
    lower: float = 30.0,
    upper_fraction: float = 0.3,
    estimator: str = "median_local",
) -> dict[str, object]:
    """Log-layer slope deviation of ``law`` against a DNS profile."""
    upper = upper_fraction * profile.re_tau
    band = profile.band(lower, upper)
    if int(band.sum().item()) < 4:
        raise ValueError(
            f"Re_tau={profile.re_tau:.2f} has no log-layer band below {upper:.1f}"
        )
    yb = profile.y_plus[band]
    kappa_dns = log_layer_slope(
        profile.y_plus,
        profile.u_plus,
        lower=lower,
        upper=upper,
        estimator=estimator,
        du_plus_dy_plus=profile.du_plus_dy_plus,
    )
    kappa_law = log_layer_slope(
        yb,
        wall_law_u_plus(yb, law),
        lower=lower,
        upper=upper,
        estimator=estimator,
        du_plus_dy_plus=wall_law_du_plus_dy_plus(yb, law),
    )
    return {
        "re_tau": profile.re_tau,
        "band": [lower, upper],
        "nodes": int(band.sum().item()),
        "estimator": estimator,
        "kappa_dns": kappa_dns,
        "kappa_law": kappa_law,
        "slope_deviation": abs(kappa_law - kappa_dns) / kappa_dns,
        "law": law.to_dict(),
    }


def channel_level_deviation(
    profile: MkmProfile,
    law: WallLawParams,
    *,
    lower: float = 30.0,
    upper_fraction: float = 0.3,
) -> dict[str, object]:
    """Relative ``u+`` level deviation of the law over the same band."""
    upper = upper_fraction * profile.re_tau
    band = profile.band(lower, upper)
    if int(band.sum().item()) < 4:
        raise ValueError("band too thin for a level comparison")
    yb = profile.y_plus[band]
    ub = profile.u_plus[band]
    modelled = wall_law_u_plus(yb, law)
    rel = (modelled - ub).abs() / ub
    return {
        "re_tau": profile.re_tau,
        "band": [lower, upper],
        "rms_relative": float(torch.sqrt((rel**2).mean()).item()),
        "max_relative": float(rel.max().item()),
        "law": law.to_dict(),
    }


# ---------------------------------------------------------------------------
# criterion 3b: equilibrium wall-modelled channel (a-posteriori, 1-D)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WallModelledChannel:
    """Equilibrium wall-modelled channel solution (wall units)."""

    re_tau: float
    law: WallLawParams
    van_driest_a: float
    y_plus: torch.Tensor
    u_plus: torch.Tensor
    du_plus_dy_plus: torch.Tensor
    tau_plus: torch.Tensor

    def interpolate_u_plus(self, y_plus) -> torch.Tensor:
        query = _as_tensor(y_plus)
        nodes = self.y_plus
        clipped = query.clamp(nodes[0], nodes[-1])
        index = torch.searchsorted(nodes, clipped).clamp(1, len(nodes) - 1)
        left = nodes[index - 1]
        right = nodes[index]
        weight = (clipped - left) / (right - left).clamp_min(1e-300)
        return self.u_plus[index - 1] * (1.0 - weight) + self.u_plus[index] * weight


def equilibrium_wall_modelled_channel(
    re_tau: float,
    *,
    law: WallLawParams = DEFAULT_LAW,
    cells: int = 20000,
    van_driest_a: float = 26.0,
) -> WallModelledChannel:
    """Integrate the equilibrium channel momentum balance with a mixing length.

    ``tau+(y) = 1 - y/H`` (wall units, ``H = Re_tau``), the eddy viscosity is
    Prandtl's mixing length with van Driest damping
    ``l+ = kappa y+ (1 - exp(-y+/A+))``, and the gradient follows from the
    quadratic ``tau+ = du+/dy+ + l+^2 (du+/dy+)^2``:

        du+/dy+ = (-1 + sqrt(1 + 4 l+^2 tau+)) / (2 l+^2)

    The wall model is *not* imposed here: this is the resolved-variable
    equilibrium solution that a wall-modelled LES is supposed to reproduce, so
    comparing it with the DNS profile is a genuine (if 1-D) validation rather
    than a statement about the wall law's own shape.  ``kappa`` is taken from
    the wall-law parameters, which is what couples the two.
    """
    if re_tau <= 0.0:
        raise ValueError("re_tau must be positive")
    if cells < 100:
        raise ValueError("cells must be >= 100")
    height = float(re_tau)
    dy = height / cells
    y = torch.arange(cells + 1, dtype=_I64) * dy
    y_mid = y[:-1] + 0.5 * dy
    damping = 1.0 - torch.exp(-y_mid / van_driest_a)
    mixing = law.kappa * y_mid * damping
    tau_mid = (1.0 - y_mid / height).clamp_min(0.0)
    nu_t = mixing.square()
    gradient = (-1.0 + torch.sqrt(1.0 + 4.0 * nu_t * tau_mid)) / (2.0 * nu_t)
    # sublayer start: the 0/0 limit at y+ -> 0 is du+/dy+ = tau+
    gradient = torch.where(nu_t > 0.0, gradient, tau_mid)
    u = torch.zeros_like(y)
    u[1:] = torch.cumsum(gradient * dy, dim=0)
    y_eval = y_mid.clamp_min(1e-300)
    y_out = torch.cat((torch.zeros(1, dtype=_I64), y_eval))
    return WallModelledChannel(
        re_tau=height,
        law=law,
        van_driest_a=van_driest_a,
        y_plus=y_out,
        u_plus=torch.cat((torch.zeros(1, dtype=_I64), u[1:])),
        du_plus_dy_plus=torch.cat((tau_mid[:1], gradient)),
        tau_plus=torch.cat((torch.ones(1, dtype=_I64), tau_mid)),
    )


def channel_profile_slope(
    channel: WallModelledChannel,
    *,
    lower: float = 30.0,
    upper_fraction: float = 0.3,
    estimator: str = "band_fit",
) -> float:
    """Log-layer slope of the wall-modelled channel profile."""
    upper = upper_fraction * channel.re_tau
    return log_layer_slope(
        channel.y_plus,
        channel.u_plus,
        lower=lower,
        upper=upper,
        estimator=estimator,
        du_plus_dy_plus=channel.du_plus_dy_plus,
    )


def channel_wall_stress_consistency(
    channel: WallModelledChannel,
    *,
    sample_y_plus=(30.0, 50.0, 100.0, 200.0),
) -> dict[str, object]:
    """Invert the channel profile velocity and compare with ``tau_w+ = 1``."""
    samples = []
    for y_plus in sample_y_plus:
        if y_plus > 0.6 * channel.re_tau:
            continue
        u_plus = float(channel.interpolate_u_plus(y_plus).item())
        solution = solve_friction_velocity(u_plus, y_plus, 1.0, law=channel.law)
        u_tau = float(solution.u_tau.item())
        samples.append(
            {
                "y_plus": float(y_plus),
                "u_plus": u_plus,
                "u_tau_recovered": u_tau,
                "tau_w_rel_error": abs(u_tau - 1.0),
            }
        )
    worst = max((s["tau_w_rel_error"] for s in samples), default=float("nan"))
    return {
        "re_tau": channel.re_tau,
        "law": channel.law.to_dict(),
        "samples": samples,
        "max_tau_w_rel_error": worst,
    }


def channel_slope_deviation(
    profile: MkmProfile,
    channel: WallModelledChannel,
    *,
    lower: float = 30.0,
    upper_fraction: float = 0.3,
    estimator: str = "band_fit",
) -> dict[str, object]:
    """Log-layer slope deviation between the wall-modelled channel and DNS."""
    upper = upper_fraction * profile.re_tau
    band = profile.band(lower, upper)
    if int(band.sum().item()) < 4:
        raise ValueError(
            f"Re_tau={profile.re_tau:.2f} has no overlap band below {upper:.1f}"
        )
    kappa_dns = log_layer_slope(
        profile.y_plus,
        profile.u_plus,
        lower=lower,
        upper=upper,
        estimator=estimator,
        du_plus_dy_plus=profile.du_plus_dy_plus,
    )
    kappa_model = channel_profile_slope(
        channel, lower=lower, upper_fraction=upper_fraction, estimator=estimator
    )
    return {
        "re_tau": profile.re_tau,
        "band": [lower, upper],
        "nodes_dns": int(band.sum().item()),
        "estimator": estimator,
        "kappa_dns": kappa_dns,
        "kappa_model": kappa_model,
        "slope_deviation": abs(kappa_model - kappa_dns) / kappa_dns,
        "law": channel.law.to_dict(),
    }


def channel_level_deviation(
    profile: MkmProfile,
    channel: WallModelledChannel,
    *,
    lower: float = 30.0,
    upper_fraction: float = 0.3,
) -> dict[str, object]:
    """Relative ``u+`` deviation of the channel solution over the DNS band."""
    upper = upper_fraction * profile.re_tau
    band = profile.band(lower, upper)
    if int(band.sum().item()) < 4:
        raise ValueError("band too thin for a level comparison")
    yb = profile.y_plus[band]
    ub = profile.u_plus[band]
    modelled = channel.interpolate_u_plus(yb)
    rel = (modelled - ub).abs() / ub
    return {
        "re_tau": profile.re_tau,
        "band": [lower, upper],
        "rms_relative": float(torch.sqrt((rel**2).mean()).item()),
        "max_relative": float(rel.max().item()),
    }


# ---------------------------------------------------------------------------
# criterion 4: grid convergence
# ---------------------------------------------------------------------------
def richardson_ratio(coarse: float, medium: float, fine: float) -> float:
    """Convergence ratio ``R = |f_f - f_m| / |f_m - f_c|``.

    ``R < 1`` means the change per grid level *shrinks* with refinement, i.e. the
    ladder is monotonically converging and the Richardson extrapolation is
    admissible; ``R > 1`` flags a diverging or oscillating ladder (the state the
    parent N04 node found for ``C_X``).
    """
    coarse_change = abs(medium - coarse)
    fine_change = abs(fine - medium)
    if coarse_change == 0.0:
        return 0.0 if fine_change == 0.0 else math.inf
    return fine_change / coarse_change


def grid_ladder_verdict(
    coarse: float,
    medium: float,
    fine: float,
    *,
    ratio: float = 2.0,
    richardson_limit: float = 1.0,
    fine_pair_limit: float = 0.05,
) -> dict[str, object]:
    """Richardson / GCI verdict for a three-grid ladder.

    Passes when the ladder converges monotonically (``R < 1``, so the
    Richardson extrapolation is admissible) **or** when the two finest grids
    already agree within ``fine_pair_limit``.
    """
    r = richardson_ratio(coarse, medium, fine)
    fine_pair = abs(fine - medium) / abs(fine) if fine != 0.0 else math.inf
    d1, d2 = abs(medium - coarse), abs(fine - medium)
    if d2 == 0.0 or d1 == 0.0:
        order = float("nan")
    else:
        order = math.log(d1 / d2) / math.log(ratio)
    extrapolated = fine + (fine - medium) / max(ratio ** order - 1.0, 1e-6) if math.isfinite(order) else float("nan")
    return {
        "values": {"coarse": coarse, "medium": medium, "fine": fine},
        "richardson_ratio": r,
        "observed_order": order,
        "fine_pair_relative_change": fine_pair,
        "extrapolated": extrapolated,
        "monotone": r < richardson_limit,
        "fine_pair_within_limit": fine_pair < fine_pair_limit,
        "passed": bool(r < richardson_limit or fine_pair < fine_pair_limit),
    }


# ---------------------------------------------------------------------------
# criterion gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UsabilityCriteria:
    inversion_limit: float = 1e-8
    stress_limit: float = 0.02
    slope_limit: float = 0.05
    richardson_limit: float = 1.0
    fine_pair_limit: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return {
            "inversion_limit": self.inversion_limit,
            "stress_limit": self.stress_limit,
            "slope_limit": self.slope_limit,
            "richardson_limit": self.richardson_limit,
            "fine_pair_limit": self.fine_pair_limit,
        }


DEFAULT_CRITERIA = UsabilityCriteria()


@dataclass(frozen=True)
class CriterionResult:
    name: str
    value: float
    limit: float
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "limit": self.limit,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True)
class UsabilityReport:
    criteria: UsabilityCriteria
    results: tuple[CriterionResult, ...]
    negative_controls: tuple[CriterionResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results) and all(r.passed for r in self.negative_controls)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(r.name for r in (*self.results, *self.negative_controls) if not r.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "criteria": self.criteria.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "negative_controls": [r.to_dict() for r in self.negative_controls],
        }


def assess_wall_model_usability(
    *,
    inversion_max_rel_error: float,
    stress_max_rel_error: float,
    channel_slope_deviation: dict[float, float],
    richardson: dict[str, object],
    negative_controls: tuple[CriterionResult, ...] = (),
    criteria: UsabilityCriteria = DEFAULT_CRITERIA,
    notes: dict[str, str] | None = None,
) -> UsabilityReport:
    """Assemble the five-criterion verdict from measured evidence.

    ``channel_slope_deviation`` maps ``Re_tau -> slope deviation`` and must
    contain both channel Reynolds numbers (a missing leg is a failure, not a
    silent pass).
    """
    notes = notes or {}

    inversion = CriterionResult(
        "inversion_rel_error",
        float(inversion_max_rel_error),
        criteria.inversion_limit,
        float(inversion_max_rel_error) < criteria.inversion_limit,
        notes.get("inversion_rel_error", "manufactured wall-law state, u_tau recovered from u(y2)"),
    )
    stress = CriterionResult(
        "stress_rel_error",
        float(stress_max_rel_error),
        criteria.stress_limit,
        float(stress_max_rel_error) < criteria.stress_limit,
        notes.get("stress_rel_error", "|tau_w - (nu du/dn + tau_sgs)|/tau_w at the wall node"),
    )
    slope_value = max(channel_slope_deviation.values()) if channel_slope_deviation else math.inf
    legs = sorted(float(key) for key in channel_slope_deviation)
    # both channel Reynolds numbers must be covered: one leg in the low-Re
    # regime (Re_tau < 300) and one in the developed-log-layer regime
    missing = [] if (len(legs) >= 2 and legs[0] < 300.0 <= legs[-1]) else ["Re_tau<300", "Re_tau>=300"]
    slope = CriterionResult(
        "channel_slope_deviation",
        float(slope_value),
        criteria.slope_limit,
        bool(channel_slope_deviation) and not missing and slope_value < criteria.slope_limit,
        notes.get(
            "channel_slope_deviation",
            "log-layer slope vs MKM1999 DNS"
            + (f"; missing Re_tau legs: {missing}" if missing else ""),
        ),
    )
    rich = CriterionResult(
        "grid_convergence",
        float(richardson.get("richardson_ratio", math.inf)),
        criteria.richardson_limit,
        bool(richardson.get("passed", False)),
        notes.get(
            "grid_convergence",
            f"R<{criteria.richardson_limit} or fine-pair<{criteria.fine_pair_limit}; "
            f"fine pair change {richardson.get('fine_pair_relative_change')}",
        ),
    )
    return UsabilityReport(
        criteria=criteria,
        results=(inversion, stress, slope, rich),
        negative_controls=tuple(negative_controls),
    )


def assert_wall_model_usable(report: UsabilityReport) -> UsabilityReport:
    """Raise ``AssertionError`` unless every criterion and control passes."""
    if not report.passed:
        details = "; ".join(
            f"{r.name}={r.value:.3e} (limit {r.limit:.3e})"
            for r in (*report.results, *report.negative_controls)
            if not r.passed
        )
        raise AssertionError(f"wall-model usability gate failed: {details}")
    return report


def negative_control(
    name: str,
    *,
    measured: float,
    limit: float,
    expected: bool = False,
    note: str = "",
) -> CriterionResult:
    """A negative control passes when the perturbed model *fails* its criterion."""
    return CriterionResult(
        name=f"nc:{name}",
        value=float(measured),
        limit=float(limit),
        passed=bool(measured >= limit if not expected else measured < limit),
        note=note,
    )


__all__ = [
    "DEFAULT_CRITERIA",
    "DEFAULT_LAW",
    "KAPPA_DNS",
    "MkmProfile",
    "REICHARDT",
    "SPALDING",
    "UsabilityCriteria",
    "UsabilityReport",
    "WallLawParams",
    "assert_wall_model_usable",
    "assess_wall_model_usability",
    "channel_level_deviation",
    "channel_profile_slope",
    "channel_slope_deviation",
    "dns_wall_stress_recovery",
    "grid_ladder_verdict",
    "inversion_relative_error",
    "load_mkm_profile",
    "log_layer_band",
    "log_layer_slope",
    "equilibrium_wall_modelled_channel",
    "manufactured_wall_state",
    "negative_control",
    "richardson_ratio",
    "solve_friction_velocity",
    "solve_friction_velocity_bisection",
    "wall_law_du_plus_dy_plus",
    "wall_law_u_plus",
    "wall_law_y_plus",
    "wall_stress_budget",
    "wall_stress_residual",
    "WallModelledChannel",
    "channel_wall_stress_consistency",
]