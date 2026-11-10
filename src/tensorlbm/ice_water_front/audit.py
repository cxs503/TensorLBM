"""Metrics for the ice/water interface verification surface.

Three families, one per acceptance criterion:

``interface_thickness_cells``
    How many grid cells the phase indicator needs to switch from (almost)
    fully solid to (almost) fully liquid -- criterion **C4** (must be <= 2 for
    the sharp representation).

``energy_audit``
    Latent heat released/absorbed vs the enthalpy change implied by the
    computed temperature field, both against the boundary heat input --
    criterion **C3** (residual <= 2 %).

``temperature_continuity``
    Temperature on the two sides of the interface, extrapolated from interior
    nodes only, vs the melting point -- criterion **C5** (<= 0.1 K).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "interface_thickness_cells",
    "vof_circle_field",
    "energy_audit",
    "temperature_continuity",
]


def _ray_thickness(profile: np.ndarray, low: float, high: float) -> int:
    """Number of consecutive cells whose value lies strictly inside (low, high).

    The maximum over all contiguous runs is returned; a profile that
    transitions sharply has one or two such cells, a diffuse one has many.
    """
    inside = (profile > low) & (profile < high)
    best = run = 0
    for flag in inside:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def interface_thickness_cells(
    field: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
) -> dict:
    """Interface thickness in grid cells for a phase-indicator field.

    Parameters
    ----------
    field : ndarray
        1-D or 2-D phase indicator with ``1`` = solid (ice) and ``0`` = liquid
        (water), e.g. a volume fraction or ``1 - liquid_fraction``.
    low, high : float
        The transition band; cells whose indicator lies strictly between the
        two thresholds are counted as interface cells.

    Returns
    -------
    dict with ``max_cells`` (worst ray, what C4 is judged on), ``mean_cells``
    and the per-ray counts.
    """
    f = np.asarray(field, dtype=float)
    if f.ndim == 1:
        profiles = [f, f[::-1]]
    elif f.ndim == 2:
        # Radial rays through the centre only: a ray that grazes the interface
        # tangentially would count a long chord as "interface", which is not a
        # thickness measurement.  Axis and diagonal rays are used instead.
        n = f.shape[0]
        c = n // 2
        diag = np.diag(f)
        anti = np.diag(np.fliplr(f))
        profiles = []
        for ray in (f[c, :], f[:, c], diag, anti):
            if ray.max() - ray.min() > 0.5:
                profiles.extend([ray, ray[::-1]])
    else:  # pragma: no cover
        raise ValueError("field must be 1-D or 2-D")
    if not profiles:  # pragma: no cover
        return {"max_cells": 0, "mean_cells": 0.0, "counts": []}
    counts = [_ray_thickness(p, low, high) for p in profiles]
    return {
        "max_cells": int(max(counts)),
        "mean_cells": float(np.mean(counts)),
        "counts": counts,
        "n_rays": len(counts),
        "low": low,
        "high": high,
    }


def vof_circle_field(X: np.ndarray, Y: np.ndarray, radius: float, cell: float,
                     sub: int = 8) -> np.ndarray:
    """Cell-averaged volume fraction of the disc ``r <= radius``.

    Sub-cell sampling (``sub x sub`` points per cell) is used, which reproduces
    the area fraction of a sharp circular front to O(1/sub^2).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    frac = np.zeros_like(X)
    for dx in offsets:
        for dy in offsets:
            frac += (np.hypot(X + dx * cell, Y + dy * cell) <= radius)
    return frac / float(sub * sub)


def energy_audit(result, latent_key: str = "latent_energy") -> dict:
    """Latent-heat / energy conservation audit for a front-tracking run.

    Compares the heat that actually entered the domain through the driven
    boundary (``q_wall_integral``) with the sum of the latent heat associated
    with the front motion and the sensible enthalpy change of the computed
    temperature field.  The residual is normalised by the latent term, which is
    the quantity the criterion constrains.
    """
    latent = float(getattr(result, latent_key))
    sensible = float(getattr(result, "sensible_energy"))
    wall = float(getattr(result, "q_wall_integral"))
    far = float(getattr(result, "q_far_integral", 0.0))
    lhs = wall - far
    melting = bool(getattr(result, "melting", True))
    rhs = sensible + (latent if melting else -latent)
    scale = abs(latent) if latent != 0 else 1.0
    return {
        "boundary_heat_J": lhs,
        "latent_heat_J": latent,
        "melting": melting,
        "sensible_heat_J": sensible,
        "sum_J": rhs,
        "residual_J": lhs - rhs,
        "relative_residual": abs(lhs - rhs) / scale,
        "closure_ratio": rhs / lhs if lhs != 0 else float("nan"),
    }


def temperature_continuity(result, t_melt: float) -> dict:
    """Interface temperature continuity in K (criterion C5)."""
    inner = float(getattr(result, "continuity_inner", float("nan")))
    outer = float(getattr(result, "continuity_outer", float("nan")))
    return {
        "t_melt_K": t_melt,
        "inner_side_offset_K": inner,
        "outer_side_offset_K": outer,
        "max_offset_K": max(inner, outer),
        "jump_K": abs(inner - outer),
    }