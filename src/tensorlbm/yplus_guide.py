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
from dataclasses import dataclass, field
from typing import Sequence

import torch


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
    nx: int,
    ny: int,
    nz: int,
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
    *blockage_ok*: True if blockage < 5%
    *domain_aspect*: nx / max(ny, nz), ≥ 2 recommended for external flow
    *cells_per_hull_length*: nx / (hull_length / dx) ≡ nx, coarser → smaller y+
    *pressure_settle_time*: estimated steps for pressure to reach steady state
    *quality_tier*: ``recommended`` / ``acceptable`` / ``marginal`` / ``poor``

    Example
    -------
    >>> grid_quality_metrics(nx=200, ny=80, nz=80, hull_length=80.0)
    {'y_plus_est': 225, 'y_plus_regime': 'log_law',
     'blockage_ratio': 0.009, 'blockage_ok': True,
     'quality_tier': 'recommended'}
    """
    nu = u_in * hull_length / re
    dx = hull_length / nx
    y = 0.5 * dx

    # y+
    if re < 1e5:
        cf = 0.0
    else:
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
    hull_area = math.pi * r_hull**2
    domain_area = (ny * dx) * (nz * dx)
    blockage = hull_area / max(domain_area, 1e-12)

    # Domain aspect ratio
    domain_aspect = nx / max(ny, nz)

    # Pressure settle time: domain_length / sound_speed, in steps
    sound_speed = 1.0 / math.sqrt(3.0)  # cs = 1/sqrt(3) in lattice units
    domain_crossings = 3  # pressure waves need ~3 domain crossings to settle
    pressure_steps = domain_crossings * nx / sound_speed

    # Quality tier — primary: y+ regime, secondary: domain adequacy
    # For slender bodies (L/D > 5), blockage up to 10% is acceptable
    if yp_regime == "log_law" and domain_aspect >= 2.0:
        tier = "recommended"
    elif yp_regime == "log_law":
        tier = "acceptable"
    elif yp_regime == "buffer":
        tier = "marginal"
    else:
        tier = "poor"

    return {
        "y_plus_est": y_plus_est,
        "y_plus_regime": yp_regime,
        "blockage_ratio": blockage,
        "blockage_pct": blockage * 100,
        "blockage_ok": blockage < 0.05,
        "domain_aspect": domain_aspect,
        "domain_aspect_ok": domain_aspect >= 2.0,
        "cells_per_hull_length": nx,
        "pressure_settle_steps_est": int(pressure_steps),
        "quality_tier": tier,
        "parameters": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "hull_length": hull_length,
            "u_in": u_in,
            "re": re,
            "dx": dx,
            "y_first": y,
            "nu": nu,
            "u_tau_est": u_tau,
        },
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
        st = (
            math.sqrt(sum((self._fric[i] + self._pres[i] - at) ** 2 for i in range(n)) / (n - 1))
            if n > 1
            else 0.0
        )

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
            return {"Ct_fric_slide": 0.0, "Ct_pres_slide": 0.0, "Ct_total_slide": 0.0, "n_slide": 0}
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
