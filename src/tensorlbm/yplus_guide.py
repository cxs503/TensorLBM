"""Pre-run y+ estimator and real-time y+ monitor for wall-modelled LBM.

Adds to :class:`DragMonitor` for real-time y+ tracking, and provides
a static estimator for pre-run grid sizing.

Usage — pre-run estimation::

    from tensorlbm.yplus_guide import estimate_yplus
    yp = estimate_yplus(nx=200, hull_length=80.0, u_in=0.06, re=2e6)
    print(f"Expected y+ ≈ {yp:.0f}")
    # → y+ ≈ 590  (deep in log-law, good)
    # → y+ ≈ 30   (buffer layer, wall function may lose accuracy)
    # → y+ ≈ 5    (viscous sublayer, wall function NOT recommended)

Usage — real-time monitoring::

    monitor = DragMonitor(warmup=500, track_yplus=True)
    for step in range(1, n_steps + 1):
        f, drag_fric, drag_pres, u_tau, near_mask = wall_fn(...)
        monitor.add(step, drag_fric, drag_pres)
        if step % 200 == 0:
            yp = 0.5 * u_tau / nu
            monitor.add_yplus(yp[near_mask])
            print(monitor.yplus_summary())  # → "y+ median=590 >30=100%"
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .hydrodynamics import ittc57_friction_coefficient


def estimate_exchange_yplus(
    *,
    physical_reynolds: float,
    characteristic_length_cells: float,
    exchange_distance_cells: float,
) -> float:
    """Estimate wall-model exchange-location y+ in lattice units.

    The estimate uses ITTC-1957 to obtain ``u_tau/U`` and the exact lattice
    similarity relation ``nu = U L_cells / Re``.  Unlike a first-cell helper,
    this API accepts the actual finest-level body resolution and the declared
    wall-normal exchange distance, so it remains valid under local refinement.
    """
    if min(
        physical_reynolds,
        characteristic_length_cells,
        exchange_distance_cells,
    ) <= 0.0:
        raise ValueError("Reynolds number, length and exchange distance must be positive")
    cf = ittc57_friction_coefficient(physical_reynolds)
    return (
        exchange_distance_cells
        / characteristic_length_cells
        * physical_reynolds
        * math.sqrt(cf / 2.0)
    )


def estimate_bfl_exchange_yplus_bounds(
    *,
    physical_reynolds: float,
    characteristic_length_cells: float,
    requested_exchange_distance_cells: float,
    minimum_bfl_wall_distance_cells: float = 0.0,
    maximum_bfl_wall_distance_cells: float = 1.0,
) -> dict[str, float]:
    """Bound y+ after the BFL sampler enforces clearance from the wall.

    The exchange sampler uses ``y2=max(requested, y1+0.5)`` at each curved
    boundary node.  A requested one-cell height therefore does not imply that
    every sample lies exactly one cell from the analytical wall.  This helper
    exposes the corresponding a-priori range instead of reporting the nominal
    value as a guaranteed maximum.
    """
    if requested_exchange_distance_cells <= 0.0:
        raise ValueError("requested exchange distance must be positive")
    if not (
        0.0 <= minimum_bfl_wall_distance_cells
        <= maximum_bfl_wall_distance_cells
    ):
        raise ValueError("BFL wall-distance bounds must be ordered and non-negative")
    minimum_effective_distance = max(
        requested_exchange_distance_cells,
        minimum_bfl_wall_distance_cells + 0.5,
    )
    maximum_effective_distance = max(
        requested_exchange_distance_cells,
        maximum_bfl_wall_distance_cells + 0.5,
    )
    common = {
        "physical_reynolds": physical_reynolds,
        "characteristic_length_cells": characteristic_length_cells,
    }
    return {
        "requested_exchange_distance_cells": requested_exchange_distance_cells,
        "minimum_effective_exchange_distance_cells": minimum_effective_distance,
        "maximum_effective_exchange_distance_cells": maximum_effective_distance,
        "minimum_exchange_y_plus_estimate": estimate_exchange_yplus(
            **common,
            exchange_distance_cells=minimum_effective_distance,
        ),
        "maximum_exchange_y_plus_estimate": estimate_exchange_yplus(
            **common,
            exchange_distance_cells=maximum_effective_distance,
        ),
    }


def plan_exchange_yplus_refinement(
    *,
    physical_reynolds: float,
    characteristic_length_cells: float,
    minimum_exchange_distance_cells: float = 1.0,
    target_maximum_yplus: float = 1000.0,
    refinement_ratio: int = 2,
) -> dict:
    """Plan extra wall-normal refinement needed to meet an exchange y+ target.

    ``characteristic_length_cells`` is the body's current finest-level
    resolution.  Each additional local level multiplies that resolution by
    ``refinement_ratio`` while retaining a one-cell (or caller-supplied)
    minimum interpolation distance.  The result is a planning diagnostic, not
    an accuracy claim; the final wall model still requires measured y+ and
    force-convergence evidence.
    """
    if target_maximum_yplus <= 0.0:
        raise ValueError("target maximum y+ must be positive")
    if refinement_ratio < 2:
        raise ValueError("refinement ratio must be at least two")
    current_minimum_yplus = estimate_exchange_yplus(
        physical_reynolds=physical_reynolds,
        characteristic_length_cells=characteristic_length_cells,
        exchange_distance_cells=minimum_exchange_distance_cells,
    )
    required_characteristic_cells = (
        characteristic_length_cells
        * current_minimum_yplus
        / target_maximum_yplus
    )
    resolution_ratio = required_characteristic_cells / characteristic_length_cells
    additional_levels = max(
        0,
        math.ceil(math.log(max(resolution_ratio, 1.0), refinement_ratio)),
    )
    planned_characteristic_cells = (
        characteristic_length_cells * refinement_ratio**additional_levels
    )
    return {
        "physical_reynolds": physical_reynolds,
        "current_characteristic_length_cells": characteristic_length_cells,
        "minimum_exchange_distance_cells": minimum_exchange_distance_cells,
        "target_maximum_y_plus": target_maximum_yplus,
        "current_minimum_exchange_y_plus_estimate": current_minimum_yplus,
        "required_characteristic_length_cells": required_characteristic_cells,
        "refinement_ratio": refinement_ratio,
        "additional_refinement_levels": additional_levels,
        "planned_characteristic_length_cells": planned_characteristic_cells,
        "planned_exchange_y_plus_estimate": estimate_exchange_yplus(
            physical_reynolds=physical_reynolds,
            characteristic_length_cells=planned_characteristic_cells,
            exchange_distance_cells=minimum_exchange_distance_cells,
        ),
    }


def estimate_yplus(
    *,
    nx: int,
    hull_length: float,
    u_in: float = 0.06,
    re: float = 2e6,
    y_val: float = 0.5,
) -> float:
    """Estimate the first-cell y+ BEFORE running the simulation.

    Uses the ITTC-1957 friction line to estimate u_tau analytically::

        Cf = 0.075 / (log10(Re) - 2)^2
        u_tau = u_in * sqrt(Cf / 2)
        y+ = (y_val * dx) * u_tau / nu

    where ``dx = hull_length / nx``, ``nu = u_in * hull_length / re``.

    This gives a reasonable pre-run estimate.  The actual y+ will differ
    because the wall function modifies the near-wall velocity gradient,
    but the estimate tells you which flow regime you are in:

    =======  =========================================================
    y+        regime
    =======  =========================================================
    > 100    log-law — wall function is in its design range (good)
    30–100   buffer layer — wall function marginal
    5–30     transition — wall function accuracy degrades rapidly
    < 5      viscous sublayer — wall function is WRONG, need wall-resolved
    =======  =========================================================

    Returns:
        Estimated y+ (float).
    """
    nu = u_in * hull_length / re
    dx = hull_length / nx
    y = y_val * dx  # first off-wall cell centre distance

    # ITTC-1957 friction coefficient
    if re < 1e5:
        raise ValueError(f"Re={re:.0f} too low for ITTC-1957 (needs Re≥1e5)")
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    u_tau = u_in * math.sqrt(max(cf, 1e-12) / 2.0)

    return y * u_tau / nu


def yplus_recommendation(
    *,
    nx: int,
    hull_length: float,
    u_in: float = 0.06,
    re: float = 2e6,
) -> dict:
    """Pre-run recommendation for wall-function applicability.

    Returns a dict with ``y_plus_est`` and a ``status`` string:
    ``"log_law"``, ``"buffer"``, ``"transition"``, or ``"viscous"``.
    Also returns the recommended action.
    """
    yp = estimate_yplus(nx=nx, hull_length=hull_length, u_in=u_in, re=re)
    if yp > 100:
        status, action = "log_law", "wall function recommended, good accuracy expected"
    elif yp > 30:
        status, action = "buffer", "wall function marginal, consider finer grid"
    elif yp > 5:
        status, action = "transition", "wall function accuracy poor, switch to wall-resolved"
    else:
        status, action = "viscous", "wall function invalid, must wall-resolve"

    return {
        "y_plus_estimate": yp,
        "regime": status,
        "recommendation": action,
        "parameters": {"nx": nx, "hull_length": hull_length, "u_in": u_in, "re": re},
    }


# ---------------------------------------------------------------------------
# Grid quality metrics — blockage ratio, domain scales, pressure convergence
# ---------------------------------------------------------------------------

def grid_quality_metrics(
    *,
    nx: int, ny: int, nz: int,
    hull_length: float,
    u_in: float = 0.06,
    re: float = 2e6,
    hull_radius: float | None = None,
) -> dict:
    """Pre-run grid quality assessment for external-flow wall-modelled LBM.

    Returns a dict of metrics that together predict whether a given grid
    will produce accurate drag predictions.  y+ alone is NOT sufficient
    — you also need adequate domain size and manageable pressure-field
    behaviour.

    Metrics returned
    ----------------
    *y_plus_est*: Estimated first-cell y+ (ITTC-1957 → u_tau)
    *y_plus_regime*: ``log_law`` / ``buffer`` / ``transition`` / ``viscous``
    *blockage_ratio*: hull cross-section / (ny × nz), smaller is better
    *blockage_ok*: True if blockage < 2%
    *domain_aspect*: nx / max(ny, nz), ≥ 2 recommended for external flow
    *cells_per_hull_length*: hull length in lattice cells
    *pressure_settle_time*: estimated steps for pressure to reach steady state
    *quality_tier*: ``recommended`` / ``acceptable`` / ``marginal`` / ``poor``

    Example
    -------
    >>> grid_quality_metrics(nx=200, ny=80, nz=80, hull_length=80.0)
    {'y_plus_est': 225, 'y_plus_regime': 'log_law',
     'blockage_ratio': 0.009, 'blockage_ok': True,
     'quality_tier': 'recommended'}
    """
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("hull_length", hull_length),
        ("u_in", u_in),
        ("re", re),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if re <= 100.0:
        raise ValueError("re must exceed 100 for the ITTC-1957 estimate")
    if hull_radius is not None and (
        not math.isfinite(hull_radius) or hull_radius <= 0.0
    ):
        raise ValueError("hull_radius must be finite and positive")
    nu = u_in * hull_length / re
    # All geometric inputs are already lattice-cell counts.  The lattice
    # spacing is one; ``nx`` is the domain length, not the hull resolution.
    # The previous hull_length/nx factor mixed these two meanings and
    # underpredicted first-cell y+ while overpredicting blockage.
    dx = 1.0
    y = 0.5

    # y+
    cf = 0.075 / (math.log10(re) - 2.0) ** 2
    u_tau = u_in * math.sqrt(max(cf, 1e-12) / 2.0)
    y_plus_est = y * u_tau / nu

    if y_plus_est > 100:
        yp_regime = "log_law"
    elif y_plus_est > 30:
        yp_regime = "buffer"
    elif y_plus_est > 5:
        yp_regime = "transition"
    else:
        yp_regime = "viscous"

    # Blockage ratio: hull cross-section / domain cross-section
    # SUBOFF diameter ≈ hull_length / 8.57
    r_hull = hull_radius if hull_radius is not None else hull_length / (2 * 8.57)
    hull_area = math.pi * r_hull ** 2
    domain_area = ny * nz
    blockage = hull_area / max(domain_area, 1e-12)

    # Domain aspect ratio
    domain_aspect = nx / max(ny, nz)

    # Pressure settle time: domain_length / sound_speed, in steps
    sound_speed = 1.0 / math.sqrt(3.0)  # cs = 1/sqrt(3) in lattice units
    domain_crossings = 3  # pressure waves need ~3 domain crossings to settle
    pressure_steps = domain_crossings * nx / sound_speed

    # Quality tier combines the wall-model regime and domain adequacy.  The
    # two-percent blockage target is a preflight gate, not an empirical drag
    # correction; formal domain convergence still needs multiple CFD boxes.
    if (
        yp_regime == "log_law"
        and domain_aspect >= 2.0
        and blockage < 0.02
    ):
        tier = "recommended"
    elif yp_regime == "log_law" and blockage < 0.05:
        tier = "acceptable"
    elif yp_regime == "buffer" or blockage < 0.10:
        tier = "marginal"
    else:
        tier = "poor"

    return {
        "y_plus_est": y_plus_est,
        "y_plus_regime": yp_regime,
        "blockage_ratio": blockage,
        "blockage_pct": blockage * 100,
        "blockage_ok": blockage < 0.02,
        "domain_aspect": domain_aspect,
        "domain_aspect_ok": domain_aspect >= 2.0,
        "cells_per_hull_length": hull_length,
        "streamwise_domain_lengths": nx / hull_length,
        "transverse_domain_diameters": min(ny, nz) / (2.0 * r_hull),
        "pressure_settle_steps_est": int(pressure_steps),
        "quality_tier": tier,
        "parameters": {"nx": nx, "ny": ny, "nz": nz, "hull_length": hull_length,
                       "u_in": u_in, "re": re, "dx": dx, "y_first": y,
                       "dx_lu": dx, "y_first_lu": y,
                       "nu": nu, "u_tau_est": u_tau},
    }


def recommend_grid(
    *,
    re: float = 2e6,
    u_in: float = 0.06,
    hull_length: float | None = None,
    target_y_plus_min: float = 100,
    target_y_plus_max: float = 1000,
    max_blockage: float = 0.02,
    min_aspect: float = 2.0,
    nx_candidates: list[int] | None = None,
) -> list[dict]:
    """Scan candidate grid sizes and return those meeting quality criteria.

    Returns a list of viable ``(nx, ny, nz)`` tuples with their metrics,
    sorted by quality score.
    """
    if nx_candidates is None:
        nx_candidates = [128, 160, 200, 256, 320, 384, 448, 512]

    results = []
    for nx in nx_candidates:
        if hull_length is not None:
            hl = hull_length
        else:
            hl = nx * 0.4  # default: hull = 40% of domain

        ny = nz = max(32, nx * 2 // 5)  # keep 2.5:1:1 aspect
        m = grid_quality_metrics(nx=nx, ny=ny, nz=nz, hull_length=hl, u_in=u_in, re=re)

        passes = (
            target_y_plus_min <= m["y_plus_est"] <= target_y_plus_max
            and m["blockage_ratio"] <= max_blockage
            and m["domain_aspect"] >= min_aspect
        )
        m["passes"] = passes
        if passes:
            results.append(m)

    results.sort(key=lambda m: -m["y_plus_est"])  # prefer higher y+ (coarser)
    return results


# ---------------------------------------------------------------------------
# Extend DragMonitor with y+ tracking
# ---------------------------------------------------------------------------

@dataclass
class DragMonitor:
    """Running-average drag convergence monitor with optional y+ tracking.

    Attributes:
        warmup: Number of initial steps to discard.
        window_frac: Fraction used for recent-window convergence check.
        track_yplus: If True, collect y+ statistics for real-time monitoring.
    """

    warmup: int = 500
    window_frac: float = 0.20
    track_yplus: bool = False

    _steps: list[int] = field(default_factory=list, init=False)
    _fric: list[float] = field(default_factory=list, init=False)
    _pres: list[float] = field(default_factory=list, init=False)
    _yplus_samples: list[float] = field(default_factory=list, init=False)

    def add(self, step: int, drag_fric: float, drag_pres: float) -> None:
        if step <= self.warmup:
            return
        self._steps.append(step)
        self._fric.append(drag_fric)
        self._pres.append(drag_pres)

    def add_yplus(self, yplus_tensor: torch.Tensor) -> None:
        """Record y+ values from near-wall cells.  Batched: keeps last 100k values."""
        if not self.track_yplus:
            return
        vals = yplus_tensor.cpu().tolist() if yplus_tensor.is_cuda else yplus_tensor.tolist()
        self._yplus_samples.extend(vals)
        # Keep bounded
        if len(self._yplus_samples) > 100_000:
            self._yplus_samples = self._yplus_samples[-50_000:]

    def yplus_summary(self) -> str:
        """Return a one-line y+ status string."""
        if not self._yplus_samples or len(self._yplus_samples) < 10:
            return "y+: no data"
        yp = torch.tensor(self._yplus_samples[-10000:])
        median = float(yp.median().item())
        pct_gt100 = float((yp > 100).float().mean().item()) * 100
        pct_gt30 = float((yp > 30).float().mean().item()) * 100
        regime = "log-law" if pct_gt30 > 90 else ("buffer" if pct_gt30 > 50 else "transition")
        return f"y+ median={median:.0f} >30={pct_gt30:.0f}% >100={pct_gt100:.0f}% [{regime}]"

    def summary(self) -> dict:
        n = len(self._fric)
        if n == 0:
            return {"n": 0, "converged": False}

        af = _mean(self._fric)
        ap = _mean(self._pres)
        at = af + ap

        sf = _std(self._fric, af) if n > 1 else 0.0
        sp = _std(self._pres, ap) if n > 1 else 0.0
        st = math.sqrt(
            sum((self._fric[i] + self._pres[i] - at) ** 2 for i in range(n)) / (n - 1)
        ) if n > 1 else 0.0

        win_n = max(1, int(n * self.window_frac))
        if win_n < 2:
            change = 0.0
        else:
            early_af = _mean(self._fric[:-win_n]) if n > win_n else af
            early_ap = _mean(self._pres[:-win_n]) if n > win_n else ap
            late_af = _mean(self._fric[-win_n:])
            late_ap = _mean(self._pres[-win_n:])
            early_total = early_af + early_ap
            late_total = late_af + late_ap
            if abs(early_total) > 1e-15:
                change = (late_total - early_total) / abs(early_total)
            else:
                change = late_total - early_total

        result: dict = {
            "n": n,
            "Ct_fric_avg": af,
            "Ct_pres_avg": ap,
            "Ct_total_avg": at,
            "Ct_fric_std": sf,
            "Ct_pres_std": sp,
            "Ct_total_std": st,
            "Ct_change_window": change,
            "converged": abs(change) < 0.005,
        }
        if self.track_yplus and self._yplus_samples:
            result["yplus_summary"] = self.yplus_summary()
        return result

    @property
    def n(self) -> int:
        return len(self._fric)

    @property
    def ct_total_avg(self) -> float:
        return (sum(self._fric) + sum(self._pres)) / max(len(self._fric), 1)

    def sliding_mean(self, window: int) -> dict:
        """Return Ct components averaged over the *last* `window` samples (sliding window)."""
        n = min(window, len(self._fric))
        if n == 0:
            return {"Ct_fric_slide": 0.0, "Ct_pres_slide": 0.0,
                    "Ct_total_slide": 0.0, "n_slide": 0}
        af = sum(self._fric[-n:]) / n
        ap = sum(self._pres[-n:]) / n
        return {
            "Ct_fric_slide": af,
            "Ct_pres_slide": ap,
            "Ct_total_slide": af + ap,
            "n_slide": n,
        }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: Sequence[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
