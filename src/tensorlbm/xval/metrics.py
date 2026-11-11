"""Metric definitions for XVAL-LBM-TAYLORGREEN.

Everything here is a pure function of raw tool output (histories of
``K(t)``/``|u|_max(t)`` and dumped velocity fields), so the numbers in the
cross-validation table are computed from files, never typed by hand.

Metrics
-------
``gamma_K``      slope of ln K(t) vs t (sign-flipped), with the constant boost
                 offset removed for the Galilean-boosted variant.
``field_l2_pct`` ||u_sim - u_exact||_2 / ||u_exact||_2  * 100, at reference
                 steps, using the closed-form Taylor-Green field.
``phase_deg``    phase of the (k, k) Fourier mode of u; for the boosted case its
                 time derivative must equal ``-k * U_boost`` exactly.
``peak_drift_pct`` drift of the position of max |u| from its t = 0 position, in
                 percent of the box length (stationary case: must be ~0).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .case_spec import TG2DSpec

TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------- analytic
def exact_velocity(spec: TG2DSpec, step: int, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Taylor-Green velocity at lattice time ``step``.

    Cell centres are at ``x_i = i + 0.5`` (units of dx = 1) exactly as in every
    tool's initial condition.
    """
    n = spec.n if n is None else int(n)
    jj, ii = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="ij")
    x = ii + 0.5
    y = jj + 0.5
    t = float(step)
    decay = math.exp(-2.0 * spec.nu * spec.k * spec.k * t)
    shift = spec.boost * t  # Galilean translation of the vortex pattern
    xs = x - shift
    ux = spec.boost - spec.u0 * np.cos(spec.k * xs) * np.sin(spec.k * y) * decay
    uy = spec.u0 * np.sin(spec.k * xs) * np.cos(spec.k * y) * decay
    return ux, uy


def exact_K(spec: TG2DSpec, step: float) -> float:
    """Closed-form kinetic energy: 0.5*U_b^2 + 0.25 U0^2 e^{-4 nu k^2 t}."""
    return 0.5 * spec.boost**2 + spec.k0_theory * math.exp(-spec.gamma_k_theory * float(step))


def exact_phase_rate(spec: TG2DSpec) -> float:
    """d(phase)/dt of the (k, k) mode, rad/step: -k * U_boost."""
    return -spec.k * spec.boost


# -------------------------------------------------------------------- fits
def fit_exp_rate(
    t: Iterable[float],
    values: Iterable[float],
    *,
    offset: float = 0.0,
) -> dict[str, Any]:
    """Least-squares fit of ``ln(values - offset)`` against ``t``.

    Returns the decay rate ``gamma = -slope``, the coefficient of
    determination and the sample count.  ``offset`` removes a constant (the
    uniform boost adds ``0.5*U_b^2`` to K without decaying).
    """
    ta = np.asarray(list(t), dtype=np.float64)
    va = np.asarray(list(values), dtype=np.float64) - float(offset)
    if ta.size < 3:
        raise ValueError("need at least 3 samples to fit a decay rate")
    if np.any(va <= 0.0):
        raise ValueError("decayed quantity must stay positive for a log fit")
    lnv = np.log(va)
    slope, intercept = np.polyfit(ta, lnv, 1)
    resid = lnv - (intercept + slope * ta)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((lnv - lnv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"gamma": -float(slope), "r2": r2, "n_samples": int(ta.size)}


def relative_error_pct(value: float, reference: float) -> float:
    """Signed relative deviation in percent."""
    if reference == 0.0:
        raise ValueError("reference must be non-zero")
    return (float(value) / float(reference) - 1.0) * 100.0


def cross_difference_pct(value: float, others: Iterable[float]) -> float:
    """Worst relative difference of ``value`` against a set of peer values."""
    worst = 0.0
    for other in others:
        denom = 0.5 * (abs(float(value)) + abs(float(other)))
        if denom == 0.0:
            continue
        worst = max(worst, abs(float(value) - float(other)) / denom * 100.0)
    return worst


# ------------------------------------------------------------------ fields
def field_l2_error_pct(
    ux: np.ndarray, uy: np.ndarray, ex: np.ndarray, ey: np.ndarray
) -> float:
    """Relative L2 norm of the velocity error, in percent."""
    num = math.sqrt(float(np.mean((ux - ex) ** 2 + (uy - ey) ** 2)))
    den = math.sqrt(float(np.mean(ex**2 + ey**2)))
    if den == 0.0:
        raise ValueError("exact field is identically zero")
    return num / den * 100.0


def mode_phase_deg(spec: TG2DSpec, ux: np.ndarray, n: int | None = None) -> float:
    """Phase of the (k, k) Fourier mode of the x-velocity field, in degrees.

    ``c = sum_ij ux_ij exp(-i k (x_i + y_j))`` with the same cell-centre
    convention as the initial condition.
    """
    n = ux.shape[0] if n is None else int(n)
    ii = np.arange(n, dtype=np.float64) + 0.5
    phase = np.cos(spec.k * ii) - 1j * np.sin(spec.k * ii)
    # c = sum_ij ux_ij exp(-i k (x_i + y_j)); the kernel is separable and the
    # x- and y-kernels are identical (same k, same cell-centre convention).
    c = complex(np.dot(ux @ phase, phase))
    if abs(c) == 0.0:
        raise ValueError("mode amplitude is zero; phase undefined")
    return math.degrees(math.atan2(c.imag, c.real))


def phase_rate(spec: TG2DSpec, steps: Iterable[int], phases_deg: Iterable[float]) -> dict[str, Any]:
    """Unwrapped linear fit d(phase)/dt in rad/step (deg input, rad output)."""
    t = np.asarray(list(steps), dtype=np.float64)
    phi = np.deg2rad(np.asarray(list(phases_deg), dtype=np.float64))
    if t.size < 3:
        raise ValueError("need at least 3 phase samples")
    phi = np.unwrap(phi)
    slope = float(np.polyfit(t, phi, 1)[0])
    resid = phi - (np.mean(phi) + slope * (t - np.mean(t)))
    ss_tot = float(np.sum((phi - phi.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 1.0
    return {"phase_rate_rad_per_step": slope, "r2": r2, "n_samples": int(t.size)}


def peak_position_fraction(spec: TG2DSpec, ux: np.ndarray, uy: np.ndarray) -> tuple[float, float]:
    """Position of max |u| as a fraction of the box length (sub-cell free)."""
    idx = int(np.argmax(ux * ux + uy * uy))
    j, i = divmod(idx, ux.shape[1])
    return (i + 0.5) / ux.shape[1], (j + 0.5) / ux.shape[0]


def periodic_distance_fraction(a: float, b: float) -> float:
    """Minimum-image distance in units of a periodic unit interval."""
    d = abs(float(a) - float(b)) % 1.0
    return min(d, 1.0 - d)


# ------------------------------------------------------------------- table
def build_xval_table(
    reference_value: float,
    per_tool: dict[str, dict[str, float]],
    *,
    metric_order: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble ``result.xval_table`` rows.

    ``per_tool[tool][metric]`` holds the measured value; every row carries the
    deviation from the analytic reference (``vs_ref_err``, percent) and the
    worst deviation from the peer tools (``cross_diff``, percent).
    """
    if metric_order is not None:
        metrics = list(metric_order)
    elif isinstance(reference_value, dict):
        # only metrics with a declared reference enter the table, so a metric
        # that is meaningless for the case variant (e.g. the phase *rate* of a
        # stationary vortex) cannot leak in as a spurious 200 % difference.
        metrics = list(reference_value)
    else:
        metrics = sorted({m for values in per_tool.values() for m in values})
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        got = {tool: float(vals[metric]) for tool, vals in per_tool.items() if metric in vals}
        if not got:
            continue
        for tool, value in got.items():
            peers = [v for other, v in got.items() if other != tool]
            if isinstance(reference_value, dict):
                ref = reference_value.get(metric)
            else:
                ref = reference_value
            # ``ref is None`` marks an absolute metric whose exact value is 0
            # (e.g. a field error or a phase error): the deviation is the value
            # itself rather than a relative deviation.
            err = value if ref is None else relative_error_pct(value, ref)
            rows.append(
                {
                    "tool": tool,
                    "metric": metric,
                    "got": value,
                    "ref": ref,
                    "vs_ref_err": err,
                    "cross_diff": cross_difference_pct(value, peers),
                }
            )
    return rows


def evaluate_criteria(
    table: list[dict[str, Any]],
    criteria: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn the table into the machine-checkable acceptance record.

    ``criteria`` is a predeclared list of
    ``{"metric": .., "kind": "vs_exact" | "cross_tool", "tol_pct": ..}``.

    Two kinds of metric are deliberately kept apart:

    * *measurement* metrics (decay rates, phase rate) carry both a vs-exact and
      a cross-tool criterion -- they are physical quantities whose value every
      tool must reproduce;
    * *error* metrics (field L2 error, phase error) only carry a vs-exact
      criterion, because their exact value is zero: a percentage difference
      between two near-zero errors (0.46 % vs 3.09 %) is not a physical
      disagreement and must not be reported as one.
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for crit in criteria:
        metric = crit["metric"]
        kind = crit["kind"]
        key = (metric, kind)
        if key in seen:
            continue
        seen.add(key)
        rows = [r for r in table if r["metric"] == metric]
        if not rows:
            continue
        if kind == "vs_exact":
            worst = max(abs(r["vs_ref_err"]) for r in rows)
            item = f"vs_exact_{metric}"
        elif kind == "cross_tool":
            worst = max(abs(r["cross_diff"]) for r in rows)
            item = f"cross_tool_{metric}"
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown criterion kind {kind!r}")
        out.append(
            {
                "item": item,
                "kind": kind,
                "metric": metric,
                "worst_abs_err_pct": float(worst),
                "tol_pct": float(crit["tol_pct"]),
                "tools_compared": len({r["tool"] for r in rows}),
                "PASS": bool(worst <= float(crit["tol_pct"])),
            }
        )
    return out
