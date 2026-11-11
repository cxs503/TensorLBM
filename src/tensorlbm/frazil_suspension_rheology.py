"""Frazil suspension power-law rheology: volume fraction -> effective viscosity.

Node anchor: ``H2-FRAZIL-RHEOLOGY`` (frazil power-law suspension rheology).

What this module adds on top of the **W9-1215 physical root**
(:mod:`tensorlbm.frazil_rheology`, which owns the closure algebra) is the
*validation face* the root declared missing ("``mu_r(phi)`` curve: literature
ranges only ... not pointwise comparable", ``docs/frazil_rheology_lbm.md`` §4):
a pointwise, threshold-checked comparison of the implemented chain

    phi --(Krieger-Dougherty mu_r)--> K(phi) = mu_w * mu_r(phi)
        --(flow index n(phi))-------> mu_app(gdot) = K(phi) * gdot^(n(phi)-1)

against independent references, plus the hand-off of ``(K, n)`` into the D2Q9
power-law channel solver (:func:`tensorlbm.frazil_lbm.run_frazil_poiseuille`)
whose velocity profile is checked against the exact Ostwald-de Waele
(Poiseuille) similarity solution.

Criteria implemented here (machine-checked, thresholds predeclared below):

``c1_einstein_dilute``
    Model chain vs Einstein's analytic dilute limit ``mu_r = 1 + [eta] phi``
    (:math:`[\\eta] = 2.5`), ``phi in [0, 0.05]``, tolerance 10 %.

``c2_krieger_dougherty_mid``
    Model chain vs the Krieger-Dougherty closed form at mid concentration
    ``phi in [0.10, 0.30]``, tolerance 10 % -- reported together with a
    **non-self-referential** cross-check of the same closure against
    Thomas (1965), an empirical correlation fitted to published suspension
    viscosity data (so the mid-concentration claim does not rest on comparing
    the closure with itself).

``c3_w9_1215_root``
    The W9-1215 physical root's own criteria, cited rather than re-invented:
    ``n(0) = 1``, ``n -> n_packed`` at the packed limit, ``K = mu_w mu_r``,
    ``mu_app(gdot=1) = K``, log-log slope ``= n - 1``, monotonicity, guards,
    and the root's documented **2 %** channel criterion on the LBM slurry
    channel.

``c4_shear_rate_sweep``
    The shear-rate scan must be a power law: fitted log-log slope equals
    ``n(phi) - 1`` and the ``phi = 0`` case is Newtonian (slope 0).

References
----------
Einstein, A. (1906) Ann. Phys. 19, 289 -- dilute-limit intrinsic viscosity.
Batchelor, G. K. & Green, J. T. (1972) J. Fluid Mech. 56, 401 -- O(phi^2) term.
Krieger, I. M. & Dougherty, T. J. (1959) Trans. Soc. Rheol. 3, 137.
Thomas, D. G. (1965) J. Colloid Sci. 20, 267 -- empirical suspension
    correlation ``mu_r = 1 + 2.5 phi + 10.05 phi^2 + 0.00273 exp(16.6 phi)``.

Scope note
----------
This module validates the *closure chain* and the *generalised-Newtonian
channel solution*.  It does **not** claim a particle-resolved measurement of
``mu_r`` from a lattice suspension cell: that route was audited and falsified
in ``docs/AUDIT_m14882_H2_FRAZIL_RHEOLOGY.md`` (statistical noise, 2.37x
shear-rate calibration defect) and is deliberately out of scope here.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch

from . import frazil_rheology as root
from .frazil_rheology import (
    DEFAULT_INTRINSIC_VISCOSITY,
    DEFAULT_N_PACKED,
    DEFAULT_PHI_MAX,
    FrazilRheologyError,
    frazil_apparent_viscosity,
    frazil_consistency_index,
    frazil_flow_index,
    relative_viscosity_krieger_dougherty,
)

__all__ = [
    "SCHEMA_VERSION",
    "EINSTEIN_INTRINSIC",
    "BATCHELOR_K2",
    "THOMAS_C2",
    "THOMAS_A",
    "THOMAS_B",
    "DILUTE_LIMIT",
    "MID_CONCENTRATION_RANGE",
    "DILUTE_TOLERANCE",
    "MID_TOLERANCE",
    "CHANNEL_TOLERANCE",
    "DEFAULT_SHEAR_RATES",
    "FrazilSuspensionRheologyError",
    "einstein_relative_viscosity",
    "thomas_relative_viscosity",
    "krieger_dougherty_relative_viscosity",
    "effective_viscosity",
    "relative_viscosity",
    "flow_index",
    "consistency_index",
    "shear_rate_sweep",
    "dilute_limit_benchmark",
    "mid_concentration_benchmark",
    "root_chain_benchmark",
    "slurry_channel_benchmark",
    "evaluate_acceptance",
    "build_report",
]

SCHEMA_VERSION = "h2-frazil-rheology/1.0"

#: Einstein intrinsic viscosity of rigid spheres, [eta] = 5/2.
EINSTEIN_INTRINSIC = 2.5
#: Second-order (O(phi^2)) coefficient, Batchelor & Green (1972).
BATCHELOR_K2 = 6.2
#: Thomas (1965) empirical correlation coefficients.
THOMAS_C2 = 10.05
THOMAS_A = 0.00273
THOMAS_B = 16.6
#: Upper edge of the dilute regime in which Einstein's first-order law applies.
DILUTE_LIMIT = 0.10
#: Mid-concentration window used by criterion ``c2``.
MID_CONCENTRATION_RANGE = (0.10, 0.30)
#: Predeclared tolerances (relative error).
DILUTE_TOLERANCE = 0.10
MID_TOLERANCE = 0.10
#: The W9-1215 root's documented channel criterion (docs/frazil_rheology_lbm.md).
CHANNEL_TOLERANCE = 0.02

DEFAULT_SHEAR_RATES = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e2, 1.0e3)


class FrazilSuspensionRheologyError(ValueError):
    """Raised when a suspension rheology input is outside its admitted range."""


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _as_float(value, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise FrazilSuspensionRheologyError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise FrazilSuspensionRheologyError(f"{name} must be finite, got {value!r}")
    return out


def _check_phi(phi, phi_max: float = DEFAULT_PHI_MAX) -> float:
    """Validate a volume fraction: finite, ``0 <= phi < phi_max``."""
    phi = _as_float(phi, "phi")
    phi_max = _as_float(phi_max, "phi_max")
    if not (0.0 < phi_max <= 1.0):
        raise FrazilSuspensionRheologyError(f"phi_max must be in (0, 1], got {phi_max!r}")
    if phi < 0.0:
        raise FrazilSuspensionRheologyError(f"phi must be >= 0, got {phi!r}")
    if phi >= phi_max:
        raise FrazilSuspensionRheologyError(
            f"phi must stay below the packed limit phi_max={phi_max}, got {phi!r}"
            " (a suspension past phi_max does not flow; no extrapolation)"
        )
    return phi


def _check_positive(value, name: str) -> float:
    value = _as_float(value, name)
    if value <= 0.0:
        raise FrazilSuspensionRheologyError(f"{name} must be > 0, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Reference closures (independent of the platform chain)
# ---------------------------------------------------------------------------


def einstein_relative_viscosity(
    phi: float,
    *,
    order: int = 1,
    k2: float = BATCHELOR_K2,
    phi_limit: float = DILUTE_LIMIT,
) -> float:
    """Einstein (1906) dilute-limit relative viscosity of a suspension.

    ``order=1``: ``mu_r = 1 + [eta] phi`` (Einstein 1906).
    ``order=2``: ``mu_r = 1 + [eta] phi + k2 phi^2`` (Batchelor & Green 1972).

    The ``phi_limit`` guard is the point of the criterion: the expansion is a
    dilute-limit result and is refused outside ``phi <= phi_limit``.
    """
    if order not in (1, 2):
        raise FrazilSuspensionRheologyError(f"order must be 1 or 2, got {order!r}")
    phi = _check_phi(phi)
    if phi > phi_limit:
        raise FrazilSuspensionRheologyError(
            f"Einstein expansion is only admitted for phi <= {phi_limit}, got {phi!r}"
        )
    mu_r = 1.0 + EINSTEIN_INTRINSIC * phi
    if order == 2:
        mu_r += k2 * phi * phi
    return mu_r


def thomas_relative_viscosity(phi: float, *, phi_limit: float = 0.50) -> float:
    """Thomas (1965) empirical suspension correlation (independent anchor).

    ``mu_r = 1 + 2.5 phi + 10.05 phi^2 + 0.00273 exp(16.6 phi)``, fitted to
    published suspension viscosity data up to ``phi ~ 0.5``.  It is used as a
    *literature* reference for the mid-concentration window: a curve that was
    not derived from the Krieger-Dougherty form.
    """
    phi = _check_phi(phi)
    if phi > phi_limit:
        raise FrazilSuspensionRheologyError(
            f"Thomas correlation is only admitted for phi <= {phi_limit}, got {phi!r}"
        )
    return 1.0 + EINSTEIN_INTRINSIC * phi + THOMAS_C2 * phi * phi + THOMAS_A * math.exp(THOMAS_B * phi)


def krieger_dougherty_relative_viscosity(
    phi: float,
    *,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
) -> float:
    """Krieger-Dougherty closed form, delegated to the W9-1215 root.

    Delegation (rather than a second copy of the formula) keeps one physical
    source of truth:
    :func:`tensorlbm.frazil_rheology.relative_viscosity_krieger_dougherty`.
    """
    phi = _check_phi(phi, phi_max)
    return float(relative_viscosity_krieger_dougherty(phi, phi_max, intrinsic))


# ---------------------------------------------------------------------------
# The model chain (the thing under test)
# ---------------------------------------------------------------------------


def effective_viscosity(
    phi,
    shear_rate,
    *,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    n_packed: float = DEFAULT_N_PACKED,
    index_exponent: float = root.DEFAULT_INDEX_EXPONENT,
    shear_rate_floor: float = 1e-12,
) -> dict:
    """Power-law effective viscosity of a frazil suspension at ``(phi, gdot)``.

    This is the public entry point of the node: ice volume fraction in,
    apparent viscosity out.

    Args:
        phi: ice volume fraction, scalar or array, ``0 <= phi < phi_max``.
        shear_rate: shear-rate magnitude in the working unit system.
        mu_water: water viscosity in the same unit system.

    Returns:
        dict with ``phi``, ``shear_rate``, ``mu_app`` (apparent viscosity),
        ``mu_relative`` (``K/mu_water``, the concentration factor alone),
        ``consistency`` (``K``) and ``flow_index`` (``n``), as numpy arrays
        broadcast to the common shape.
    """
    mu_water = _check_positive(mu_water, "mu_water")
    phi_t = torch.as_tensor(np.asarray(phi, dtype=np.float64), dtype=torch.float64)
    if phi_t.numel() and float(phi_t.min()) < 0.0:
        raise FrazilSuspensionRheologyError("phi must be >= 0")
    if phi_t.numel() and float(phi_t.max()) >= phi_max:
        raise FrazilSuspensionRheologyError(f"phi must stay below phi_max={phi_max}")
    g_t = torch.as_tensor(np.asarray(shear_rate, dtype=np.float64), dtype=torch.float64)
    if g_t.numel() and float(g_t.min()) < 0.0:
        raise FrazilSuspensionRheologyError("shear_rate must be >= 0")
    mu_app, n = frazil_apparent_viscosity(
        phi_t,
        g_t,
        mu_water=mu_water,
        phi_max=phi_max,
        intrinsic=intrinsic,
        n_packed=n_packed,
        index_exponent=index_exponent,
        shear_rate_floor=shear_rate_floor,
    )
    K = frazil_consistency_index(phi_t, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
    if isinstance(K, float):
        K = torch.as_tensor(K, dtype=torch.float64)
    phi_b, g_b = torch.broadcast_tensors(phi_t, g_t)
    return {
        "phi": phi_b.detach().cpu().numpy(),
        "shear_rate": g_b.detach().cpu().numpy(),
        "mu_app": mu_app.detach().cpu().numpy(),
        "mu_relative": (K / mu_water).detach().cpu().numpy(),
        "consistency": K.detach().cpu().numpy(),
        "flow_index": n.detach().cpu().numpy(),
    }


def relative_viscosity(
    phi: float,
    *,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
) -> float:
    """``mu_r(phi)`` read off the **model chain** (not the closure directly).

    Evaluated as ``mu_app(phi, gdot=1) / mu_water``, which is exactly how the
    solver sees the concentration factor: ``K(phi)`` is the apparent viscosity
    at the reference shear rate ``gdot = 1``.  Routing the criterion through
    this path (instead of calling the closure) means a break anywhere in the
    chain -- mu_water plumbing, exponent, broadcast -- surfaces as a criterion
    failure rather than hiding behind a tautology.
    """
    out = effective_viscosity(phi, 1.0, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
    return float(np.asarray(out["mu_app"]).reshape(-1)[0] / mu_water)


def flow_index(phi: float, *, phi_max: float = DEFAULT_PHI_MAX, n_packed: float = DEFAULT_N_PACKED) -> float:
    """``n(phi)``: the W9-1215 volume-fraction dependent power-law index."""
    phi = _check_phi(phi, phi_max)
    return float(frazil_flow_index(phi, phi_max=phi_max, n_packed=n_packed))


def consistency_index(
    phi: float,
    *,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
) -> float:
    """``K(phi) = mu_water * mu_r(phi)`` of the W9-1215 root."""
    phi = _check_phi(phi, phi_max)
    return float(frazil_consistency_index(phi, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic))


# ---------------------------------------------------------------------------
# Shear-rate sweep
# ---------------------------------------------------------------------------


def shear_rate_sweep(
    phis: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    gamma_dots: Sequence[float] = DEFAULT_SHEAR_RATES,
    *,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    n_packed: float = DEFAULT_N_PACKED,
) -> dict:
    """Apparent viscosity vs shear rate for a set of ice fractions.

    For every ``phi`` the log-log slope of ``mu_app(gdot)`` is fitted and
    compared with the root's ``n(phi) - 1``: a power law must recover its own
    exponent, and the Newtonian case ``phi = 0`` must come out flat.

    Returns:
        dict with ``gamma_dots`` and ``curves`` -- one entry per ``phi`` with
        ``mu_app``, ``mu_relative``, ``n``, ``K``, ``fitted_slope``,
        ``expected_slope``, ``slope_error``, ``fit_r2`` and
        ``shear_thinning_ratio`` -- plus ``max_slope_error`` and
        ``newtonian_slope`` diagnostics.
    """
    g = np.asarray([_check_positive(v, "shear_rate") for v in gamma_dots], dtype=np.float64)
    if g.size < 2:
        raise FrazilSuspensionRheologyError("a log-log slope needs at least two shear rates")
    curves = []
    max_slope_error = 0.0
    newtonian_slope = None
    for phi in phis:
        phi = _check_phi(phi, phi_max)
        out = effective_viscosity(
            phi,
            g,
            mu_water=mu_water,
            phi_max=phi_max,
            intrinsic=intrinsic,
            n_packed=n_packed,
        )
        mu = np.asarray(out["mu_app"], dtype=np.float64)
        n = float(np.asarray(out["flow_index"]).reshape(-1)[0])
        slope, intercept = np.polyfit(np.log(g), np.log(mu), 1)
        pred = slope * np.log(g) + intercept
        ss_res = float(np.sum((np.log(mu) - pred) ** 2))
        ss_tot = float(np.sum((np.log(mu) - np.log(mu).mean()) ** 2))
        r2 = 1.0 if ss_tot == 0.0 else max(0.0, 1.0 - ss_res / ss_tot)
        slope_error = abs(float(slope) - (n - 1.0))
        max_slope_error = max(max_slope_error, slope_error)
        if phi == 0.0:
            newtonian_slope = float(slope)
        curves.append(
            {
                "phi": phi,
                "mu_app": mu.tolist(),
                "mu_relative": float(np.asarray(out["mu_relative"]).reshape(-1)[0]),
                "n": n,
                "K": float(np.asarray(out["consistency"]).reshape(-1)[0]),
                "fitted_slope": float(slope),
                "expected_slope": n - 1.0,
                "slope_error": slope_error,
                "fit_r2": r2,
                "shear_thinning_ratio": float(mu[0] / mu[-1]),
            }
        )
    return {
        "gamma_dots": g.tolist(),
        "curves": curves,
        "max_slope_error": max_slope_error,
        "newtonian_slope": newtonian_slope,
        "phi_max": phi_max,
        "intrinsic_viscosity": intrinsic,
        "n_packed": n_packed,
    }


# ---------------------------------------------------------------------------
# Criterion c1: dilute limit vs Einstein
# ---------------------------------------------------------------------------


def dilute_limit_benchmark(
    phis: Iterable[float] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
    *,
    threshold: float = DILUTE_TOLERANCE,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
) -> dict:
    """Criterion ``c1``: model chain vs Einstein's analytic dilute limit."""
    rows = []
    for phi in phis:
        phi = _check_phi(phi, phi_max)
        if phi > DILUTE_LIMIT:
            raise FrazilSuspensionRheologyError(
                f"dilute benchmark admits phi <= {DILUTE_LIMIT}, got {phi!r}"
            )
        ref = einstein_relative_viscosity(phi)
        got = relative_viscosity(phi, mu_water=mu_water, phi_max=phi_max)
        err = abs(got - ref) / ref
        rows.append(
            {
                "phi": phi,
                "ref_einstein": ref,
                "got_model": got,
                "abs_error": abs(got - ref),
                "rel_error": err,
                "threshold": threshold,
                "passed": bool(err < threshold),
            }
        )
    max_err = max((r["rel_error"] for r in rows), default=float("inf"))
    return {
        "criterion": "c1_einstein_dilute",
        "reference": "Einstein (1906): mu_r = 1 + 2.5 phi",
        "rows": rows,
        "max_rel_error": max_err,
        "threshold": threshold,
        "n_pass": sum(1 for r in rows if r["passed"]),
        "n_defined": len(rows),
        "passed": bool(rows) and max_err < threshold,
    }


# ---------------------------------------------------------------------------
# Criterion c2: mid concentration vs Krieger-Dougherty (+ independent Thomas)
# ---------------------------------------------------------------------------


def mid_concentration_benchmark(
    phis: Iterable[float] = (0.10, 0.15, 0.20, 0.25, 0.30),
    *,
    threshold: float = MID_TOLERANCE,
    mu_water: float = 1.0,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
) -> dict:
    """Criterion ``c2``: model chain vs Krieger-Dougherty, plus Thomas (1965).

    Two comparisons per point:

    ``kd``
        ``got`` = model chain, ``ref`` = closed form.  This is a *code-path*
        check and must agree to round-off; the 10 % window of the criterion is
        therefore trivially met here, which is stated plainly instead of being
        presented as an independent success.
    ``thomas``
        ``got`` = model chain, ``ref`` = Thomas (1965) empirical correlation --
        an independent functional form fitted to published suspension data.
        This is the non-self-referential content of ``c2``.
    """
    lo, hi = MID_CONCENTRATION_RANGE
    rows = []
    for phi in phis:
        phi = _check_phi(phi, phi_max)
        if not (lo - 1e-12 <= phi <= hi + 1e-12):
            raise FrazilSuspensionRheologyError(
                f"mid-concentration benchmark admits phi in [{lo}, {hi}], got {phi!r}"
            )
        kd = krieger_dougherty_relative_viscosity(phi, phi_max=phi_max, intrinsic=intrinsic)
        thomas = thomas_relative_viscosity(phi)
        got = relative_viscosity(phi, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
        err_kd = abs(got - kd) / kd
        err_th = abs(got - thomas) / thomas
        rows.append(
            {
                "phi": phi,
                "got_model": got,
                "ref_krieger_dougherty": kd,
                "rel_error_kd": err_kd,
                "ref_thomas_1965": thomas,
                "rel_error_thomas": err_th,
                "threshold": threshold,
                "passed": bool(err_kd < threshold and err_th < threshold),
            }
        )
    max_kd = max((r["rel_error_kd"] for r in rows), default=float("inf"))
    max_th = max((r["rel_error_thomas"] for r in rows), default=float("inf"))
    return {
        "criterion": "c2_krieger_dougherty_mid",
        "reference": "Krieger & Dougherty (1959); independent cross-check Thomas (1965)",
        "kd_parameterization": {"phi_max": phi_max, "intrinsic_viscosity": intrinsic},
        "rows": rows,
        "max_rel_error_kd": max_kd,
        "max_rel_error_thomas": max_th,
        "threshold": threshold,
        "n_pass": sum(1 for r in rows if r["passed"]),
        "n_defined": len(rows),
        "passed": bool(rows) and max_kd < threshold and max_th < threshold,
    }


# ---------------------------------------------------------------------------
# Criterion c3: the W9-1215 physical root, quoted criteria
# ---------------------------------------------------------------------------


def root_chain_benchmark(
    *,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    n_packed: float = DEFAULT_N_PACKED,
    mu_water: float = 1.0,
    phi_probe: float = 0.20,
    slope_phi: float = 0.30,
) -> dict:
    """Criterion ``c3`` (analytic part): the W9-1215 root's own criteria.

    Every check is taken from the root's documented contract
    (``docs/frazil_rheology_lbm.md`` §3-§4) rather than invented here.
    """
    checks = []

    def add(name, ref, got, rel_err, threshold, note=""):
        checks.append(
            {
                "name": name,
                "ref": ref,
                "got": got,
                "rel_error": rel_err,
                "threshold": threshold,
                "passed": bool(rel_err <= threshold),
                "note": note,
            }
        )

    # n(0) = 1 : clear water is Newtonian.
    n0 = flow_index(0.0, phi_max=phi_max, n_packed=n_packed)
    add("n(phi=0)=1", 1.0, n0, abs(n0 - 1.0), 1e-12)

    # n -> n_packed at the packed limit.
    n_hi = flow_index(phi_max * (1.0 - 1e-6), phi_max=phi_max, n_packed=n_packed)
    add("n(phi->phi_max)=n_packed", n_packed, n_hi, abs(n_hi - n_packed) / n_packed, 1e-3)

    # K(phi) = mu_water * mu_r(phi).
    k = consistency_index(phi_probe, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
    mu_r = krieger_dougherty_relative_viscosity(phi_probe, phi_max=phi_max, intrinsic=intrinsic)
    add("K(phi)=mu_w*mu_r(phi)", mu_water * mu_r, k, abs(k - mu_water * mu_r) / (mu_water * mu_r), 1e-12)

    # mu_app(gdot=1) = K : the consistency index is the reference-shear viscosity.
    app = effective_viscosity(phi_probe, 1.0, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
    mu_at_one = float(np.asarray(app["mu_app"]).reshape(-1)[0])
    add("mu_app(gdot=1)=K", k, mu_at_one, abs(mu_at_one - k) / k, 1e-12)

    # log-log slope = n - 1 (a power law reproduces its own exponent).
    g = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0])
    out = effective_viscosity(slope_phi, g, mu_water=mu_water, phi_max=phi_max, intrinsic=intrinsic)
    mu = np.asarray(out["mu_app"], dtype=np.float64)
    n = float(np.asarray(out["flow_index"]).reshape(-1)[0])
    slope = float(np.polyfit(np.log(g), np.log(mu), 1)[0])
    add(
        "dln(mu_app)/dln(gdot)=n-1",
        n - 1.0,
        slope,
        abs(slope - (n - 1.0)) / max(abs(n - 1.0), 1e-30),
        1e-9,
    )

    # Monotonicity: mu_r increasing and n decreasing in phi.
    phis = np.linspace(0.0, phi_max * 0.99, 60)
    mu_rs = np.array(
        [krieger_dougherty_relative_viscosity(p, phi_max=phi_max, intrinsic=intrinsic) for p in phis]
    )
    ns = np.array([flow_index(p, phi_max=phi_max, n_packed=n_packed) for p in phis])
    d_mu = float(np.min(np.diff(mu_rs)))
    d_n = float(np.max(np.diff(ns)))
    checks.append(
        {
            "name": "mu_r(phi) strictly increasing",
            "ref": "monotone closure",
            "got": d_mu,
            "rel_error": 0.0,
            "threshold": 0.0,
            "passed": bool(d_mu > 0.0),
            "note": "min forward difference of mu_r over phi in [0, 0.99 phi_max]",
        }
    )
    checks.append(
        {
            "name": "n(phi) monotonically decreasing",
            "ref": "monotone closure",
            "got": d_n,
            "rel_error": 0.0,
            "threshold": 0.0,
            "passed": bool(d_n <= 0.0),
            "note": "max forward difference of n over phi in [0, 0.99 phi_max]",
        }
    )

    # Guards: non-physical input is refused, never extrapolated.
    guards = {}
    for label, call in (
        ("phi=phi_max", lambda: effective_viscosity(phi_max, 1.0, phi_max=phi_max)),
        ("phi<0", lambda: effective_viscosity(-0.01, 1.0, phi_max=phi_max)),
        ("mu_water=0", lambda: effective_viscosity(0.1, 1.0, mu_water=0.0, phi_max=phi_max)),
        ("einstein beyond dilute", lambda: einstein_relative_viscosity(0.5)),
        ("thomas beyond 0.5", lambda: thomas_relative_viscosity(0.6)),
        ("non-positive shear rate", lambda: shear_rate_sweep((0.1,), (0.0, 1.0))),
    ):
        raised = False
        try:
            call()
        except (FrazilSuspensionRheologyError, FrazilRheologyError):
            raised = True
        guards[label] = raised
    checks.append(
        {
            "name": "guards raise on non-physical input",
            "ref": "all guarded",
            "got": float(sum(guards.values())) / float(len(guards)),
            "rel_error": 0.0,
            "threshold": 0.0,
            "passed": bool(all(guards.values())),
            "note": ",".join(sorted(k for k, v in guards.items() if not v)) or "all guards raised",
        }
    )

    n_pass = sum(1 for c in checks if c["passed"])
    return {
        "criterion": "c3_w9_1215_root",
        "reference": "docs/frazil_rheology_lbm.md (W9-1215 physical root criteria)",
        "checks": checks,
        "guards": guards,
        "n_pass": n_pass,
        "n_defined": len(checks),
        "passed": n_pass == len(checks),
    }


# ---------------------------------------------------------------------------
# Criterion c3 (numerical part): LBM slurry channel vs the exact solution
# ---------------------------------------------------------------------------


def slurry_channel_benchmark(
    phis: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.40),
    *,
    H: int = 16,
    nx: int = 2,
    mu_water: float = 1.0e-3,
    u_ref: float = 0.03,
    wall_viscosity_target: float | None = 2.0e-2,
    max_steps: int = 20000,
    tol: float = 1.0e-4,
    check_every: int = 200,
    tolerance: float = CHANNEL_TOLERANCE,
    phi_max: float = DEFAULT_PHI_MAX,
    run: bool = True,
    return_profiles: bool = False,
) -> dict:
    """Run the slurry channel through the D2Q9 power-law solver.

    ``ref``: the exact Ostwald-de Waele Poiseuille similarity solution for the
    **same** ``(K, n, a)`` pair handed to the solver
    (:func:`tensorlbm.frazil_lbm.poiseuille_analytic_profile`).
    ``got``: the LBM velocity profile.

    The W9-1215 root's documented node criterion is 2 % on this channel.  The
    more concentrated cases are lattice-rescaled (``K -> lambda K``,
    ``a -> lambda a``) when the physical Krieger-Dougherty consistency would
    push the relaxation time far above 1; the velocity field is exactly
    invariant under that rescaling (the momentum balance is linear in both), so
    the verification target is unchanged and ``lattice_rescale`` is recorded
    for every case.

    ``run=False`` returns the criterion skeleton with ``n_defined = 0`` (an
    unrun check is *unknown*, never "passed").
    """
    from .frazil_lbm import run_frazil_poiseuille

    rows = []
    for phi in phis:
        phi = _check_phi(phi, phi_max)
        entry = {"phi": phi, "ran": False}
        if run:
            res = run_frazil_poiseuille(
                phi,
                mu_water=mu_water,
                u_ref=u_ref,
                H=H,
                nx=nx,
                phi_max=phi_max,
                wall_viscosity_target=wall_viscosity_target,
                max_steps=max_steps,
                tol=tol,
                check_every=check_every,
            )
            fr = res["frazil"]
            err = res["error"]
            entry.update(
                {
                    "ran": True,
                    "l2_relative": float(err["l2_relative"]),
                    "peak_relative": float(err["peak_relative"]),
                    "u_max_numeric": float(err["u_max_numeric"]),
                    "u_max_exact": float(err["u_max_exact"]),
                    "flow_index": float(fr["flow_index"]),
                    "consistency_index": float(fr["consistency_index"]),
                    "viscosity_wall_physical": float(fr["viscosity_wall_physical"]),
                    "viscosity_wall": float(fr["viscosity_wall"]),
                    "lattice_rescale": float(fr["lattice_rescale"]),
                    "tau_relaxation_max": float(fr["tau_relaxation_max"]),
                    "forcing": float(fr["forcing"]),
                    "relative_viscosity": float(fr["relative_viscosity"]),
                    "steps": int(res.get("steps", -1)),
                    "H": int(H),
                    "nx": int(nx),
                    "threshold": tolerance,
                    "passed": bool(float(err["l2_relative"]) < tolerance),
                    # integration check: the (n, mu_r) the *solver* actually used
                    # must be the ones this module's public API returns, i.e. the
                    # solver route and the node API share one closure chain.
                    "route_api_flow_index": flow_index(phi, phi_max=phi_max),
                    "route_api_relative_viscosity": krieger_dougherty_relative_viscosity(
                        phi, phi_max=phi_max
                    ),
                    "route_consistent": bool(
                        abs(float(fr["flow_index"]) - flow_index(phi, phi_max=phi_max)) <= 1e-12
                        and abs(
                            float(fr["relative_viscosity"])
                            - krieger_dougherty_relative_viscosity(phi, phi_max=phi_max)
                        )
                        <= 1e-12
                    ),
                }
            )
            if return_profiles:
                entry["profile"] = np.asarray(res["profile"], dtype=np.float64).tolist()
                entry["u_exact"] = np.asarray(res["u_exact"], dtype=np.float64).tolist()
                entry["y_phys"] = np.asarray(res["y_phys"], dtype=np.float64).tolist()
        rows.append(entry)
    ran = [r for r in rows if r["ran"]]
    worst = max((r["l2_relative"] for r in ran), default=float("inf"))
    return {
        "criterion": "c3b_lbm_slurry_channel",
        "reference": "exact Ostwald-de Waele Poiseuille similarity solution",
        "solver": "tensorlbm.frazil_lbm.run_frazil_poiseuille",
        "rows": rows,
        "worst_l2_relative": worst,
        "threshold": tolerance,
        "n_pass": sum(1 for r in ran if r["passed"]),
        "n_defined": len(ran),
        "passed": bool(ran)
        and worst < tolerance
        and len(ran) == len(rows)
        and all(r.get("route_consistent", False) for r in ran),
    }


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def evaluate_acceptance(
    *,
    dilute: dict,
    mid: dict,
    root_chain: dict,
    channels: dict,
    sweep: dict | None = None,
    sweep_slope_tolerance: float = 1e-6,
    newtonian_slope_tolerance: float = 1e-9,
) -> dict:
    """Aggregate the four criteria into a machine-checkable verdict."""
    criteria = {
        "c1_einstein_dilute": {
            "passed": bool(dilute["passed"]),
            "threshold": dilute["threshold"],
            "measured": dilute["max_rel_error"],
            "metric": "max |mu_r - Einstein| / Einstein over phi <= 0.05",
        },
        "c2_krieger_dougherty_mid": {
            "passed": bool(mid["passed"]),
            "threshold": mid["threshold"],
            "measured": max(mid["max_rel_error_kd"], mid["max_rel_error_thomas"]),
            "metric": "max relative error of mu_r vs KD and vs Thomas (1965) over 0.10 <= phi <= 0.30",
        },
        # A channel run that was not performed is *unknown*, not passed: the
        # criterion is then reported as undefined and excluded from the
        # denominator instead of being silently counted as a failure or a pass.
        "c3_w9_1215_root": {
            "passed": (
                None
                if channels["n_defined"] == 0
                else bool(root_chain["passed"] and channels["passed"])
            ),
            "threshold": channels["threshold"],
            "measured": {
                "analytic_checks_passed": root_chain["n_pass"],
                "analytic_checks_defined": root_chain["n_defined"],
                "channel_runs": channels["n_defined"],
                "channel_worst_l2_relative": (
                    channels["worst_l2_relative"] if channels["n_defined"] else "not run"
                ),
            },
            "metric": "W9-1215 root identities + LBM channel profile vs exact solution",
        },
    }
    if sweep is not None:
        newtonian_ok = True
        if sweep.get("newtonian_slope") is not None:
            newtonian_ok = abs(float(sweep["newtonian_slope"])) < newtonian_slope_tolerance
        thinning_ok = all(
            c["shear_thinning_ratio"] > 1.0 + 1e-9 for c in sweep["curves"] if c["phi"] > 0.0
        )
        criteria["c4_shear_rate_sweep"] = {
            "passed": bool(sweep["max_slope_error"] < sweep_slope_tolerance and newtonian_ok and thinning_ok),
            "threshold": sweep_slope_tolerance,
            "measured": {
                "max_slope_error": sweep["max_slope_error"],
                "newtonian_slope": sweep["newtonian_slope"],
                "thinning_all_positive": thinning_ok,
            },
            "metric": "fitted log-log slope vs n(phi)-1 over 1e-3 <= gdot <= 1e3",
        }
    n_pass = sum(1 for c in criteria.values() if c["passed"] is True)
    n_defined = sum(1 for c in criteria.values() if c["passed"] is not None)
    undefined = sorted(cid for cid, c in criteria.items() if c["passed"] is None)
    return {
        "criteria": criteria,
        "n_pass": n_pass,
        "n_defined": n_defined,
        "n_failed": n_defined - n_pass,
        "undefined": undefined,
        "all_passed": bool(n_defined) and n_pass == n_defined,
        "verdict": "PASS" if (n_defined and n_pass == n_defined) else "FAIL",
    }


def build_report(
    *,
    phis_sweep: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    phis_channel: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.40),
    gamma_dots: Sequence[float] = DEFAULT_SHEAR_RATES,
    mu_water: float = 1.0e-3,
    run_channels: bool = True,
    return_profiles: bool = False,
    H: int = 16,
    nx: int = 2,
    max_steps: int = 20000,
    tol: float = 1.0e-4,
    check_every: int = 200,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    n_packed: float = DEFAULT_N_PACKED,
) -> dict:
    """Build the full machine-readable H2-FRAZIL-RHEOLOGY report.

    The sweep uses the physical water viscosity ``mu_water`` (Pa s) so the
    apparent-viscosity curves are in SI; the closure *ratios* are
    mu_water-independent by construction and are also recorded.
    """
    sweep = shear_rate_sweep(
        phis_sweep,
        gamma_dots,
        mu_water=mu_water,
        phi_max=phi_max,
        intrinsic=intrinsic,
        n_packed=n_packed,
    )
    dilute = dilute_limit_benchmark(phis=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05), phi_max=phi_max)
    mid = mid_concentration_benchmark(phi_max=phi_max, intrinsic=intrinsic)
    root_chain = root_chain_benchmark(
        phi_max=phi_max, intrinsic=intrinsic, n_packed=n_packed, mu_water=mu_water
    )
    channels = slurry_channel_benchmark(
        phis_channel,
        H=H,
        nx=nx,
        mu_water=mu_water,
        max_steps=max_steps,
        tol=tol,
        check_every=check_every,
        phi_max=phi_max,
        run=run_channels,
        return_profiles=return_profiles,
    )
    acceptance = evaluate_acceptance(
        dilute=dilute, mid=mid, root_chain=root_chain, channels=channels, sweep=sweep
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "node": "H2-FRAZIL-RHEOLOGY",
        "anchor": "frazil power-law suspension rheology (phi -> mu_eff)",
        "physical_root": "W9-1215 (tensorlbm.frazil_rheology)",
        "closures": {
            "relative_viscosity": "Krieger-Dougherty (1959), phi_max=%.3f, [eta]=%.3f"
            % (phi_max, intrinsic),
            "flow_index": "n(phi) = 1 - (1 - n_packed) (phi/phi_max)^1.5, n_packed=%.3f" % n_packed,
            "consistency": "K(phi) = mu_water * mu_r(phi)",
            "apparent_viscosity": "mu_app = K(phi) * gdot^(n(phi)-1)",
        },
        "anchors": {
            "c1": "Einstein (1906) Ann. Phys. 19, 289: mu_r = 1 + 2.5 phi",
            "c2": "Krieger & Dougherty (1959) Trans. Soc. Rheol. 3, 137",
            "c2_independent": "Thomas (1965) J. Colloid Sci. 20, 267 (empirical correlation)",
            "c3": "W9-1215 physical root criteria, docs/frazil_rheology_lbm.md",
            "c3_numeric": "exact Ostwald-de Waele Poiseuille similarity solution",
            "second_order": "Batchelor & Green (1972) J. Fluid Mech. 56, 401 (6.2 phi^2)",
        },
        "parameters": {
            "mu_water": mu_water,
            "phi_max": phi_max,
            "intrinsic_viscosity": intrinsic,
            "n_packed": n_packed,
            "shear_rates": list(gamma_dots),
            "channel_H": H,
            "channel_nx": nx,
            "channel_max_steps": max_steps,
            "channel_tol": tol,
        },
        "shear_rate_sweep": sweep,
        "dilute_limit": dilute,
        "mid_concentration": mid,
        "root_chain": root_chain,
        "channels": channels,
        "acceptance": acceptance,
    }
