"""Shared machinery for the dam-break verification face (``benchmarks/dam_break``).

The face has two jobs:

1. **Analytic anchor** -- the Ritter (1892) self-similar solution of the
   shallow-water equations is the reference depth profile
   ``h(x, t) = h0/9 * (2 - (x - x_dam)/(c0 t))**2`` with ``c0 = sqrt(g h0)``;
2. **Audit** -- every route reports the same three numbers: the depth-profile
   deviation from Ritter, the mass drift of the whole run, and the observed
   grid-convergence order over three (or more) grids.

Both the shallow-water route (:mod:`tensorlbm.shallow_water2d`) and the
lattice-Boltzmann free-surface route (:mod:`tensorlbm.free_surface_lbm`, Körner
model) are driven through the same metric code so the numbers are comparable.

Nondimensionalisation
---------------------
Lattice units: the water depth is ``h0`` cells, ``g`` is chosen so that the
gravity-wave celerity ``c0 = sqrt(g h0)`` matches a fixed lattice value (0.08,
i.e. Mach 0.14 for ``c_s = 1/sqrt(3)``), and time is reported as
``t* = t sqrt(g/h0) = t c0 / h0``.  In these units the analytic anchor of a
column of width ``a`` against a wall is

.. math::

    h(x, t) = \\frac{h_0}{9}\\left(2 - \\frac{x - a}{c_0 t}\\right)^2,
    \\qquad a - c_0 t \\le x \\le a + 2 c_0 t,

valid while ``c_0 t <= a`` (the rarefaction head has not reached the wall).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Predeclared gates (fixed BEFORE any production run; see
# docs/verification/DAM_BREAK_LBM.md §2 "pre-registration")
# ---------------------------------------------------------------------------
PROFILE_REL_ERROR_MAX = 0.05
"""P1: relative L1 depth-profile deviation from Ritter on the wet region."""

FRONT_REL_ERROR_MAX = 0.05
"""P2: relative front-position deviation from the Ritter front x_dam + 2 c0 t."""

MASS_DRIFT_MAX = 1.0e-3
"""P3: max |M(t) - M(0)| / M(0) over the whole run."""

CONVERGENCE_ORDER_MIN = 1.5
"""P4: observed order p = log(e_coarse/e_fine)/log(r) on the P1 metric."""


def gates() -> dict[str, float]:
    """The predeclared gate thresholds (single source of truth for the JSON)."""
    return {
        "profile_rel_error_max": PROFILE_REL_ERROR_MAX,
        "front_rel_error_max": FRONT_REL_ERROR_MAX,
        "mass_drift_max": MASS_DRIFT_MAX,
        "convergence_order_min": CONVERGENCE_ORDER_MIN,
    }


# ---------------------------------------------------------------------------
# Ritter (1892) analytic anchor
# ---------------------------------------------------------------------------
def ritter_depth(x: np.ndarray, t: float, h0: float, g: float, x_dam: float) -> np.ndarray:
    """Ritter self-similar depth at positions ``x`` and time ``t`` (dry bed).

    ``x_dam`` is the initial dam face; the reservoir lies behind it (``x < x_dam``),
    the dry bed ahead.  At ``t = 0`` the solution is the initial step.
    """
    x = np.asarray(x, dtype=float)
    c0 = math.sqrt(g * h0)
    if t <= 0.0:
        return np.where(x <= x_dam, h0, 0.0)
    xi = (x - x_dam) / (c0 * t)
    rarefaction = h0 / 9.0 * np.maximum(2.0 - xi, 0.0) ** 2
    return np.where(xi <= -1.0, h0, np.where(xi >= 2.0, 0.0, rarefaction))


def ritter_front(t: float, h0: float, g: float, x_dam: float) -> float:
    """Ritter front position ``x_dam + 2 c0 t``."""
    return x_dam + 2.0 * math.sqrt(g * h0) * t


def ritter_head(t: float, h0: float, g: float, x_dam: float) -> float:
    """Ritter rarefaction-head position ``x_dam - c0 t``."""
    return x_dam - math.sqrt(g * h0) * t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def profile_metrics(
    x: np.ndarray,
    h_sim: np.ndarray,
    t_star: float,
    h0: float,
    g: float,
    x_dam: float,
    *,
    front_window: float = 0.0,
) -> dict[str, float]:
    """Depth-profile deviation from Ritter (relative L1 and L2, wet region).

    The wet region is ``x_head <= x <= x_sim_front - front_window``; the
    analytic head/front bounds keep the metric independent of how far a
    numerical front has smeared.  ``front_window`` (in cells) trims the last
    few cells of the numerical front, where the analytic depth vanishes and a
    pointwise relative error is dominated by round-off of a zero reference.
    """
    t = t_star  # lattice time; the caller passes t in lattice units
    x = np.asarray(x, dtype=float)
    h_ref = ritter_depth(x, t, h0, g, x_dam)
    wet = (x >= ritter_head(t, h0, g, x_dam)) & (x <= ritter_front(t, h0, g, x_dam))
    if front_window > 0.0:
        wet &= x <= ritter_front(t, h0, g, x_dam) - front_window
    if not np.any(wet):
        return {"rel_l1": float("nan"), "rel_l2": float("nan"), "n_cells": 0}
    diff = np.abs(h_sim[wet] - h_ref[wet])
    denom_l1 = float(np.mean(np.abs(h_ref[wet])))
    rel_l1 = float(np.mean(diff) / denom_l1)
    rel_l2 = float(math.sqrt(float(np.mean(diff**2)) / float(np.mean(h_ref[wet] ** 2))))
    return {"rel_l1": rel_l1, "rel_l2": rel_l2, "n_cells": int(np.count_nonzero(wet))}


def front_position(x: np.ndarray, h: np.ndarray, threshold_frac: float = 1e-3) -> float:
    """Outermost wet position (sub-cell interpolated) of a depth profile."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    wet = np.nonzero(h > threshold_frac * max(float(np.max(h)), 1e-30))[0]
    if wet.size == 0:
        return float(x[0])
    last = int(wet[-1])
    if last + 1 >= h.size:
        return float(x[last])
    h_last, h_next = float(h[last]), float(h[last + 1])
    if h_last <= 0.0:
        return float(x[last])
    frac = h_last / (h_last + max(-h_next, 0.0)) if h_next < 0.0 else 0.0
    return float(x[last] + frac * (x[last + 1] - x[last]))


def front_metrics(
    x: np.ndarray,
    h: np.ndarray,
    t: float,
    h0: float,
    g: float,
    x_dam: float,
) -> dict[str, float]:
    """Front-position deviation from the Ritter front, relative to its advance."""
    x_front = front_position(x, h)
    x_ref = ritter_front(t, h0, g, x_dam)
    advance = x_ref - x_dam
    return {
        "x_front": x_front,
        "x_front_analytic": x_ref,
        "rel_error": float(abs(x_front - x_ref) / advance),
    }


def convergence_order(errors: list[float], ratios: list[float]) -> dict[str, Any]:
    """Observed order(s) ``p = log(e_i/e_{i+1}) / log(r_i)`` and their mean."""
    orders = []
    for i in range(len(errors) - 1):
        e0, e1 = errors[i], errors[i + 1]
        if e0 > 0.0 and e1 > 0.0:
            orders.append(float(math.log(e0 / e1) / math.log(ratios[i])))
    return {
        "errors": errors,
        "refinement_ratios": ratios,
        "orders": orders,
        "order_mean": float(np.mean(orders)) if orders else float("nan"),
    }


def mass_drift_series(drift: list[float]) -> dict[str, float]:
    """Summary of a recorded relative-mass-drift time series."""
    arr = np.asarray(drift, dtype=float)
    return {
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "final": float(arr[-1]) if arr.size else 0.0,
        "n_samples": int(arr.size),
    }


def json_ready(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as pretty JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_profiles(
    path: str | Path,
    analytic_x: np.ndarray,
    analytic: np.ndarray,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    xlabel: str = "x [lattice cells]",
    ylabel: str = "depth h [cells]",
) -> Path:
    """Depth profiles of every run against the Ritter profile.

    ``profiles`` maps a label to ``(x, h)`` so grids of different resolution can
    be overlaid without interpolation artefacts.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
    ax.plot(analytic_x, analytic, "k--", lw=2.0, label="Ritter (1892) analytic")
    for label, (prof_x, prof) in profiles.items():
        ax.plot(prof_x, prof, lw=1.4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_mass_drift(
    path: str | Path, series: dict[str, list[float]], *, title: str, limit: float
) -> Path:
    """Relative mass drift of every route against the gate limit."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=140)
    for label, drift in series.items():
        steps = np.arange(len(drift))
        ax.plot(steps, np.asarray(drift) * 100.0, lw=1.4, label=label)
    ax.axhline(limit * 100.0, color="r", ls=":", label=f"gate {limit * 100:g}%")
    ax.axhline(-limit * 100.0, color="r", ls=":")
    ax.set_xlabel("time step")
    ax.set_ylabel("relative mass drift [%]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_convergence(
    path: str | Path, series: dict[str, tuple[list[float], list[float]]], *, title: str
) -> Path:
    """Log-log error vs grid spacing with the measured orders in the legend."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for label, (spacing, errors) in series.items():
        spacing_arr = np.asarray(spacing, dtype=float)
        errors_arr = np.asarray(errors, dtype=float)
        ax.loglog(spacing_arr, errors_arr, "o-", lw=1.4, label=label)
        if spacing_arr.size >= 2:
            order = math.log(errors_arr[0] / errors_arr[-1]) / math.log(
                spacing_arr[0] / spacing_arr[-1]
            )
            ref = errors_arr[0] * (spacing_arr / spacing_arr[0]) ** order
            ax.loglog(spacing_arr, ref, ":", lw=1.0, color="grey")
    ax.set_xlabel("grid spacing dx / h0")
    ax.set_ylabel("relative profile error P1")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


@dataclass
class GridCase:
    """Description of one grid in a refinement study."""

    label: str
    nx: int
    ny: int
    dx: float
    n_steps: int
    t_star: float
