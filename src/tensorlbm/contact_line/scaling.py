"""Cox-Voinov / Tanner scaling analysis of the driven contact line.

Physics
-------
For a contact line moving at capillary number ``Ca = mu U / sigma`` the
*advancing* dynamic angle of a partially wetting system obeys the Cox-Voinov
law (Cox 1986, Voinov 1976)

    theta_d^3 = theta_e^3 + 9 Ca ln(L / l_m),                                (1)

where ``L`` is the macroscopic length (droplet radius / wetting width) and
``l_m`` the microscopic cutoff at which the continuum description is cut off
(slip length, interface thickness, ...).  Three consequences are used as
machine-checkable criteria of this node:

* ``Ca -> 0``: ``theta_d -> theta_e`` (criterion C1);
* as ``Ca`` grows with ``theta_d >> theta_e`` the ``theta_e`` term drops out and

      theta_d ~ (9 ln(L/l_m))^(1/3) Ca^(1/3)                                 (2)

  the **Tanner law** — a log-log slope of exactly ``1/3`` with respect to
  ``Ca`` (criterion C2);
* the coefficient of the linear-in-``Ca`` form (1) determines
  ``K = 9 ln(L/l_m)``, hence ``l_m = L exp(-K/9)``, which can be compared with
  an *independently measured* microscopic length (criterion C3).

Everything here is a least-squares/statistical post-processing of numbers that
:mod:`tensorlbm.contact_line.capillary_driven` measured on the lattice; no
asymptotic result is assumed, each fit reports slope, coefficient, ``R^2``,
degrees of freedom and a bootstrap confidence interval.

Resolvability gate (criterion C5)
---------------------------------
The low-``Ca`` end of a pseudopotential-LBM sweep is limited by the residual
(spurious) current of the model, not by the drive: a data point is only usable
for a scaling fit when the *angle change* it produces exceeds the measurement
uncertainty, ``|theta_d - theta_e| > k sqrt(sigma_meas^2 + sigma_ref^2)`` with
``k = 2`` here.  Points that fail the gate are still reported (they are the
evidence for the limit) but are excluded from the fit, and the number of
decades actually covered by the *admitted* points is reported separately so
that a fit can never quietly rest on a sub-decade range.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch

__all__ = [
    "CoxVoinovFit",
    "ScalingPoint",
    "TannerFit",
    "cox_voinov_fit",
    "decades_covered",
    "evaluate_c1",
    "evaluate_c2",
    "evaluate_c3",
    "microscopic_length",
    "power_law_fit",
    "resolvability_gate",
]

#: Radians per degree.
DEG = math.pi / 180.0


@dataclass
class ScalingPoint:
    """One driven-droplet run reduced to the quantities the fits need."""

    gx: float
    capillary: float
    velocity: float
    velocity_stderr: float
    theta_a_deg: float
    theta_a_unc_deg: float
    theta_r_deg: float
    theta_r_unc_deg: float
    theta_e_deg: float
    theta_e_unc_deg: float
    wall_fraction: float
    resolvable: bool = True
    gate_reason: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def theta_a_rad(self) -> float:
        return self.theta_a_deg * DEG

    @property
    def theta_e_rad(self) -> float:
        return self.theta_e_deg * DEG

    @property
    def delta_deg(self) -> float:
        """Angle change produced by the drive (advancing side), degrees."""
        return self.theta_a_deg - self.theta_e_deg


def decades_covered(values: list[float]) -> float:
    """Number of decades spanned by ``values`` (``log10(max/min)``)."""
    v = [float(x) for x in values if x > 0.0]
    if len(v) < 2:
        return 0.0
    return math.log10(max(v) / min(v))


def resolvability_gate(
    points: list[ScalingPoint],
    *,
    k: float = 2.0,
    min_decorrelation: float = 0.5,
) -> list[ScalingPoint]:
    """Mark the points whose angle change is resolved above the noise.

    A point is admitted when *both* of the following hold:

    ``angle``
        ``|theta_d - theta_e| > k * sqrt(unc_dyn^2 + unc_ref^2)`` — the dynamic
        angle differs from the static one by more than the combined measurement
        uncertainty (``k = 2`` by default);
    ``velocity``
        the fitted drift is at least ``min_decorrelation`` lattice units, i.e.
        resolvable at all over the sampling window.

    The returned list contains the same objects with ``resolvable`` and
    ``gate_reason`` refreshed (the objects are mutated in place and returned for
    convenience).

    Args:
        points:            Sweep points.
        k:                 Uncertainty multiplier of the angle gate.
        min_decorrelation: Minimum centroid drift (lattice units).

    Returns:
        The input list, updated.
    """
    for p in points:
        reasons = []
        unc = math.hypot(p.theta_a_unc_deg, p.theta_e_unc_deg)
        if abs(p.delta_deg) <= float(k) * unc:
            reasons.append(f"angle |dtheta|={abs(p.delta_deg):.2f}<{k}*{unc:.2f}")
        drift = float(p.extra.get("drift", 0.0) or 0.0)
        if abs(drift) < float(min_decorrelation):
            reasons.append(f"drift={drift:.2f}<{min_decorrelation}")
        p.resolvable = not reasons
        p.gate_reason = "; ".join(reasons)
    return points


def power_law_fit(
    x: list[float], y: list[float], *, bootstrap: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Least-squares fit of ``log y = exponent * log x + log A`` with a CI.

    Returns:
        Dict with ``exponent``, ``prefactor``, ``r2``, ``stderr`` and the 95%
        bootstrap confidence interval ``ci_low``/``ci_high`` of the exponent.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    lx = torch.log(torch.tensor([float(v) for v in x], dtype=torch.float64))
    ly = torch.log(torch.tensor([float(v) for v in y], dtype=torch.float64))
    if lx.numel() < 3:
        raise ValueError("need at least 3 points for a power-law fit")
    n = lx.numel()
    A = torch.stack([lx, torch.ones_like(lx)], dim=1)
    sol = torch.linalg.lstsq(A, ly.unsqueeze(1)).solution.squeeze(1)
    exponent = float(sol[0])
    prefactor = math.exp(float(sol[1]))
    resid = ly - (A @ sol)
    ss_res = float((resid**2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(n - 2, 1)
    s2 = ss_res / dof
    stderr = math.sqrt(max(s2 / float(((lx - lx.mean()) ** 2).sum()), 0.0))

    g = torch.Generator().manual_seed(int(seed))
    boots = []
    for _ in range(int(bootstrap)):
        idx = torch.randint(0, n, (n,), generator=g)
        try:
            s = torch.linalg.lstsq(A[idx], ly[idx].unsqueeze(1)).solution.squeeze(1)
        except Exception:  # pragma: no cover
            continue
        boots.append(float(s[0]))
    if len(boots) > 10:
        bt = torch.tensor(boots, dtype=torch.float64)
        lo = float(torch.quantile(bt, 0.025))
        hi = float(torch.quantile(bt, 0.975))
    else:  # pragma: no cover
        lo = hi = float("nan")
    return {
        "exponent": exponent,
        "prefactor": prefactor,
        "r2": r2,
        "stderr": stderr,
        "ci_low": lo,
        "ci_high": hi,
        "n": float(n),
    }


@dataclass
class TannerFit:
    """Tanner-law regression ``theta_d ~ A Ca^(1/3)``."""

    exponent: float
    prefactor: float
    r2: float
    stderr: float
    ci_low: float
    ci_high: float
    decades: float
    n_points: int
    degrees: list[float] = field(default_factory=list)
    capillary: list[float] = field(default_factory=list)
    passes: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_c2(
    points: list[ScalingPoint],
    *,
    use: str = "A",
    tolerance: float = 0.10,
    min_decades: float = 1.5,
    bootstrap: int = 2000,
) -> TannerFit:
    """Tanner-law check (criterion C2): log-log slope ``= 1/3``.

    Args:
        points:      Sweep points (only ``resolvable`` ones are used).
        use:         ``"A"`` (advancing) or ``"mean"`` (mean of A/R).
        tolerance:   Allowed absolute deviation of the exponent from ``1/3``.
        min_decades: Minimum decade span of the admitted ``Ca`` values.

    Returns:
        :class:`TannerFit`; ``passes`` is the machine verdict.
    """
    pts = [p for p in points if p.resolvable and p.capillary > 0.0]
    if len(pts) < 3:
        return TannerFit(
            exponent=float("nan"),
            prefactor=float("nan"),
            r2=float("nan"),
            stderr=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            decades=0.0,
            n_points=len(pts),
            passes=False,
            note="fewer than 3 resolvable points",
        )
    ca = [p.capillary for p in pts]
    if use == "A":
        th = [p.theta_a_rad for p in pts]
    elif use == "mean":
        th = [0.5 * (p.theta_a_rad + p.theta_r_deg * DEG) for p in pts]
    else:  # pragma: no cover
        raise ValueError("use must be 'A' or 'mean'")
    fit = power_law_fit(ca, th, bootstrap=bootstrap)
    dec = decades_covered(ca)
    ok = (
        abs(fit["exponent"] - 1.0 / 3.0) <= float(tolerance)
        and dec >= float(min_decades)
    )
    note = ""
    if dec < float(min_decades):
        note = f"only {dec:.2f} decades admitted (< {min_decades})"
    elif not ok:
        note = (
            f"exponent {fit['exponent']:.3f} deviates from 1/3 by "
            f"{abs(fit['exponent'] - 1.0 / 3.0):.3f} > {tolerance}"
        )
    return TannerFit(
        exponent=fit["exponent"],
        prefactor=fit["prefactor"],
        r2=fit["r2"],
        stderr=fit["stderr"],
        ci_low=fit["ci_low"],
        ci_high=fit["ci_high"],
        decades=dec,
        n_points=len(pts),
        degrees=[math.degrees(v) for v in th],
        capillary=ca,
        passes=bool(ok),
        note=note,
    )


@dataclass
class CoxVoinovFit:
    """Cox-Voinov regression ``theta_d^3 - theta_e^3 = K Ca``."""

    slope: float
    intercept: float
    r2: float
    stderr: float
    n_points: int
    decades: float
    log_term: float = float("nan")
    micro_length: float = float("nan")
    macro_length: float = float("nan")
    passes: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cox_voinov_fit(
    points: list[ScalingPoint],
    *,
    macro_length: float,
    use: str = "A",
) -> CoxVoinovFit:
    """Fit ``theta_d^3 - theta_e^3 = K Ca`` and extract ``l_m``.

    ``intercept`` is reported as a consistency check: the law predicts zero, so a
    large intercept relative to the fitted range flags a systematic error of the
    measurement (e.g. the static angle and the dynamic angle not being measured
    with the same estimator).

    Args:
        points:       Sweep points (resolvable only).
        macro_length: ``L`` in ``K = 9 ln(L/l_m)`` — the measured wetting radius.
        use:          ``"A"`` (advancing) or ``"mean"``.

    Returns:
        :class:`CoxVoinovFit`; ``passes`` is True when the fit is admissible
        (non-negative slope, ``R^2 > 0.9``, at least three points).
    """
    pts = [p for p in points if p.resolvable and p.capillary > 0.0]
    if len(pts) < 3:
        return CoxVoinovFit(
            slope=float("nan"),
            intercept=float("nan"),
            r2=float("nan"),
            stderr=float("nan"),
            n_points=len(pts),
            decades=0.0,
            note="fewer than 3 resolvable points",
        )
    ca = torch.tensor([p.capillary for p in pts], dtype=torch.float64)
    if use == "A":
        th = [p.theta_a_rad for p in pts]
    elif use == "mean":
        th = [0.5 * (p.theta_a_rad + p.theta_r_deg * DEG) for p in pts]
    else:  # pragma: no cover
        raise ValueError("use must be 'A' or 'mean'")
    cube = torch.tensor(
        [t**3 - p.theta_e_rad**3 for t, p in zip(th, pts)], dtype=torch.float64
    )
    A = torch.stack([ca, torch.ones_like(ca)], dim=1)
    sol = torch.linalg.lstsq(A, cube.unsqueeze(1)).solution.squeeze(1)
    slope = float(sol[0])
    intercept = float(sol[1])
    resid = cube - (A @ sol)
    ss_res = float((resid**2).sum())
    ss_tot = float(((cube - cube.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(len(pts) - 2, 1)
    stderr = math.sqrt(max(ss_res / dof / float(((ca - ca.mean()) ** 2).sum()), 0.0))
    log_term = slope / 9.0
    l_m = microscopic_length(slope, macro_length)
    ok = slope > 0.0 and r2 > 0.9 and math.isfinite(l_m)
    note = ""
    if slope <= 0.0:
        note = "slope <= 0: the dynamic angle does not grow with Ca"
    elif r2 <= 0.9:
        note = f"R^2 {r2:.3f} <= 0.9"
    return CoxVoinovFit(
        slope=slope,
        intercept=intercept,
        r2=r2,
        stderr=stderr,
        n_points=len(pts),
        decades=decades_covered([p.capillary for p in pts]),
        log_term=log_term,
        micro_length=l_m,
        macro_length=float(macro_length),
        passes=bool(ok),
        note=note,
    )


def microscopic_length(slope: float, macro_length: float) -> float:
    """Invert ``K = 9 ln(L/l_m)`` for the microscopic cutoff ``l_m``."""
    if macro_length <= 0.0:
        raise ValueError("macro_length must be positive")
    return float(macro_length) * math.exp(-float(slope) / 9.0)


def evaluate_c3(
    fit: CoxVoinovFit,
    *,
    micro_measured: float,
    tolerance: float = 0.20,
    min_distinctness: float = 3.0,
) -> dict[str, object]:
    """Criterion C3: the fitted log term must be resolvable and consistent.

    Three conditions:

    ``distinct``   ``L / l_m >= min_distinctness`` — otherwise ``ln(L/l_m)`` is
                   not a logarithm of *distinct* scales and the fit is not
                   informative (the log could be absorbed into the prefactor);
    ``positive``   the extracted ``l_m`` is finite and positive;
    ``consistent`` ``|l_m/l_micro - 1| <= tolerance`` with ``l_micro`` measured
                   independently (interface thickness / slip length).

    Returns:
        Dict with the numbers, the individual flags and ``passes``.
    """
    l_m = float(fit.micro_length)
    ratio = l_m / float(micro_measured) if micro_measured > 0 else float("nan")
    distinct = math.isfinite(l_m) and l_m > 0.0 and (fit.macro_length / l_m) >= float(
        min_distinctness
    )
    consistent = math.isfinite(ratio) and abs(ratio - 1.0) <= float(tolerance)
    return {
        "macro_length": float(fit.macro_length),
        "micro_length_fit": l_m,
        "micro_length_measured": float(micro_measured),
        "ratio_fit_over_measured": ratio,
        "log_term_ln_L_over_l": float(fit.log_term),
        "distinct_scales": bool(distinct),
        "consistent": bool(consistent),
        "passes": bool(distinct and consistent and fit.passes),
        "tolerance": float(tolerance),
        "min_distinctness": float(min_distinctness),
    }


def evaluate_c1(
    points: list[ScalingPoint],
    *,
    tolerance_deg: float = 3.0,
    use: str = "mean",
) -> dict[str, object]:
    """Criterion C1: the ``Ca -> 0`` limit of the dynamic angle is ``theta_e``.

    The lowest-capillary *resolvable-or-not* point is extrapolated to ``Ca = 0``
    with the Cox-Voinov form evaluated on the resolved points (i.e. the measured
    ``K`` is used to remove the finite-``Ca`` contribution), and the residual is
    compared with ``theta_e``.

    Returns:
        Dict with ``theta_limit_deg``, ``theta_e_deg``, ``error_deg`` and
        ``passes``.
    """
    pts = sorted(points, key=lambda p: p.capillary)
    admitted = [p for p in pts if p.resolvable and p.capillary > 0.0]
    if len(admitted) < 3 or not pts:
        return {
            "theta_limit_deg": float("nan"),
            "theta_e_deg": float("nan"),
            "error_deg": float("nan"),
            "passes": False,
            "note": "not enough admitted points to extrapolate",
        }
    fit = cox_voinov_fit(admitted, macro_length=admitted[0].extra.get("macro_length", 1.0))
    slope = fit.slope if math.isfinite(fit.slope) else 0.0
    lowest = admitted[0]
    if use == "mean":
        theta = 0.5 * (lowest.theta_a_rad + lowest.theta_r_deg * DEG)
    else:
        theta = lowest.theta_a_rad
    correction = slope * lowest.capillary
    cube = theta**3 - correction
    theta_limit = math.degrees(math.copysign(abs(cube) ** (1.0 / 3.0), cube))
    err = theta_limit - lowest.theta_e_deg
    return {
        "theta_limit_deg": theta_limit,
        "theta_e_deg": lowest.theta_e_deg,
        "error_deg": err,
        "abs_error_deg": abs(err),
        "ca_of_lowest_point": lowest.capillary,
        "applied_correction_rad3": correction,
        "tolerance_deg": float(tolerance_deg),
        "passes": bool(abs(err) <= float(tolerance_deg)),
    }