"""Calibration-based contact-angle inversion for SC-LBM.

Core algorithm
--------------
1. **Sweep**: For each ``f_w`` in a prescribed set, relax a sessile drop and
   measure the equilibrium contact angle ``theta``.
2. **Fit**: Fit a monotone piecewise-linear curve to ``(f_w, cos(theta))``.
   Using ``cos`` keeps the curve smooth across the full 0-180 deg range.
3. **Invert**: Given ``theta_target``, compute ``cos_target`` and use
   linear interpolation to find ``f_w``.

Why ``cos(theta)`` and not ``theta`` directly?
The physical relation between ``f_w`` and ``theta`` is monotone but highly
nonlinear near the endpoints (0 deg and 180 deg).  In ``cos`` space the
curve is much closer to linear, which makes interpolation more accurate
and inversion more stable.

Huang 2007 connection
---------------------
Huang et al. (2007) showed that the Young relation
``cos(theta) = (psi_w - psi_v) / (psi_l - psi_v)`` fails because the wall
pseudopotential only affects one lattice row while the fluid-fluid
interaction spans the entire diffuse interface.  Their corrected formula
introduces a factor that depends on the interaction strength and the
density ratio.  The calibration approach here is the *empirical*
realisation of that correction: instead of deriving the factor
analytically, we measure it on the actual grid and use it for inversion.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from ..contact_line.engine import (
    RHO_L_DEFAULT,
    RHO_V_DEFAULT,
    SIGMA_DEFAULT,
    Channel,
    WallStepper,
    sessile_drop_state,
)
from ..sc_surface_tension import contact_angle_from_mask, psi_exp

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CalibrationPoint:
    """One point on the calibration curve."""
    f_w: float
    theta_measured: float
    cos_measured: float
    status: str  # "ok", "no_interface", "unconverged"
    n_contour: int
    drift_deg: float


@dataclass
class CalibrationResult:
    """Result of a full calibration sweep.

    Attributes:
        nx, ny:       Grid resolution used for calibration.
        radius:       Droplet radius used.
        steps:        Relaxation steps per point.
        wall_rows:    Number of solid wall rows.
        psi_l:        Liquid pseudopotential at coexistence.
        psi_v:        Vapour pseudopotential at coexistence.
        points:       List of calibration points.
        fw_values:    Sorted f_w values for valid points.
        cos_values:   Corresponding cos(theta) values.
        valid_range:  (f_w_min, f_w_max) where the fit is valid.
        young_slope:  Slope of the Young relation (for comparison).
        measured_slope: Slope of the measured curve (linear fit).
        slope_ratio:  measured_slope / young_slope (Huang 2007 correction factor).
    """
    nx: int
    ny: int
    radius: float
    steps: int
    wall_rows: int
    psi_l: float
    psi_v: float
    points: List[CalibrationPoint] = field(default_factory=list)
    fw_values: List[float] = field(default_factory=list)
    cos_values: List[float] = field(default_factory=list)
    valid_range: Tuple[float, float] = (0.0, 1.0)
    young_slope: float = 0.0
    measured_slope: float = 0.0
    slope_ratio: float = 0.0
    neutral_fw_young: float = 0.0
    neutral_fw_measured: float = 0.0

    def to_dict(self) -> dict:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "radius": self.radius,
            "steps": self.steps,
            "wall_rows": self.wall_rows,
            "psi_l": self.psi_l,
            "psi_v": self.psi_v,
            "points": [
                {"f_w": p.f_w, "theta_measured": p.theta_measured,
                 "cos_measured": p.cos_measured, "status": p.status,
                 "n_contour": p.n_contour, "drift_deg": p.drift_deg}
                for p in self.points
            ],
            "fw_values": self.fw_values,
            "cos_values": self.cos_values,
            "valid_range": list(self.valid_range),
            "young_slope": self.young_slope,
            "measured_slope": self.measured_slope,
            "slope_ratio": self.slope_ratio,
            "neutral_fw_young": self.neutral_fw_young,
            "neutral_fw_measured": self.neutral_fw_measured,
        }


# ---------------------------------------------------------------------------
# Young relation (for comparison)
# ---------------------------------------------------------------------------


def young_prediction(f_w: float, psi_l: float, psi_v: float) -> float:
    """Contact angle (degrees) predicted by the Young relation.

    ``cos(theta) = (f_w * psi_l - psi_v) / (psi_l - psi_v)``
    """
    if abs(psi_l - psi_v) < 1e-12:
        return float("nan")
    cos_t = (f_w * psi_l - psi_v) / (psi_l - psi_v)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


# ---------------------------------------------------------------------------
# Linear interpolation utilities
# ---------------------------------------------------------------------------


def _linear_interp(x: List[float], y: List[float], xq: float) -> float:
    """Piecewise linear interpolation.

    Args:
        x: Sorted x values.
        y: Corresponding y values.
        xq: Query point.

    Returns:
        Interpolated y value, or NaN if out of range.
    """
    if len(x) < 2:
        return float("nan")

    # Clamp to range
    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]

    # Find interval
    i = bisect.bisect_right(x, xq) - 1
    i = max(0, min(i, len(x) - 2))

    # Linear interpolation
    t = (xq - x[i]) / (x[i + 1] - x[i])
    return y[i] + t * (y[i + 1] - y[i])


def _linear_interp_inverse(x: List[float], y: List[float], yq: float) -> float:
    """Inverse piecewise linear interpolation: find x such that interp(x) = yq.

    Assumes y is monotone (decreasing for cos vs f_w in wetting).

    Args:
        x: Sorted x values.
        y: Corresponding y values (should be monotone).
        yq: Target y value.

    Returns:
        Interpolated x value, or NaN if out of range.
    """
    if len(x) < 2:
        return float("nan")

    # Check if yq is in range
    y_min = min(y)
    y_max = max(y)
    if yq < y_min - 1e-6 or yq > y_max + 1e-6:
        return float("nan")

    # Clamp to range
    if yq <= y_min:
        idx = y.index(y_min)
        return x[idx]
    if yq >= y_max:
        idx = y.index(y_max)
        return x[idx]

    # Find interval (y is decreasing, so we search differently)
    # We need to find i such that y[i] >= yq >= y[i+1] (or vice versa)
    for i in range(len(y) - 1):
        if (y[i] - yq) * (y[i + 1] - yq) <= 0:
            # yq is between y[i] and y[i+1]
            dy = y[i + 1] - y[i]
            if abs(dy) < 1e-12:
                return x[i]
            t = (yq - y[i]) / dy
            return x[i] + t * (x[i + 1] - x[i])

    return float("nan")


# ---------------------------------------------------------------------------
# Contact angle measurement (reuses existing infrastructure)
# ---------------------------------------------------------------------------


def _measure_theta(
    rho: torch.Tensor,
    solid: torch.Tensor,
    y_wall: float,
    rho_l: float,
    rho_v: float,
) -> dict:
    """Measure contact angle from density field."""
    mid = 0.5 * (rho_l + rho_v)
    result = contact_angle_from_mask(rho, solid, mid=mid)
    return result


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------


def calibrate(
    fw_values: Optional[List[float]] = None,
    nx: int = 128,
    ny: int = 64,
    radius: float = 15.0,
    steps: int = 8000,
    wall_rows: int = 2,
    sample_interval: int = 500,
    device: str = "cpu",
    threads: int = 4,
) -> CalibrationResult:
    """Run a calibration sweep: measure theta(f_w) on the target grid.

    Args:
        fw_values: Wall fractions to scan. Default: 11 points from 0.30 to 0.80.
        nx, ny: Grid resolution.
        radius: Droplet radius (lattice units).
        steps: Relaxation steps per point.
        wall_rows: Number of solid wall rows.
        sample_interval: How often to sample during relaxation.
        device: Torch device.
        threads: Number of torch threads.

    Returns:
        CalibrationResult with fitted curve and diagnostics.
    """
    torch.set_num_threads(threads)

    if fw_values is None:
        fw_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    channel = Channel(nx=nx, ny=ny, wall_rows=wall_rows, device=device)
    solid = channel.solid_mask()
    y_wall = channel.y_wall()

    psi_l = float(psi_exp(torch.tensor(RHO_L_DEFAULT)))
    psi_v = float(psi_exp(torch.tensor(RHO_V_DEFAULT)))

    points: List[CalibrationPoint] = []

    for fw in fw_values:
        # Initialize sessile drop (90 deg initial shape as neutral starting point)
        f, _ = sessile_drop_state(channel, radius=radius, height=0.0)

        stepper = WallStepper(
            channel,
            G=5.0,
            tau=1.0,
            rho_l=RHO_L_DEFAULT,
            rho_v=RHO_V_DEFAULT,
            psi_fn=psi_exp,
            wall_fraction=fw,
        )

        theta_history = []
        for step in range(1, steps + 1):
            f = stepper.step(f)

            if step % sample_interval == 0:
                rho = stepper.density(f)
                meas = _measure_theta(rho, solid, y_wall, RHO_L_DEFAULT, RHO_V_DEFAULT)
                if meas.get("status") == "ok":
                    theta_history.append((step, meas["theta_deg"]))

        # Final measurement
        rho = stepper.density(f)
        meas = _measure_theta(rho, solid, y_wall, RHO_L_DEFAULT, RHO_V_DEFAULT)

        # Compute drift
        drift = float("nan")
        if len(theta_history) >= 4:
            tail = theta_history[-4:]
            drift = abs(tail[-1][1] - tail[0][1])

        status = "ok"
        theta_meas = float("nan")
        n_contour = 0
        if meas.get("status") == "ok":
            theta_meas = meas["theta_deg"]
            n_contour = meas.get("n_contour", 0)
            if drift > 1.0:
                status = "unconverged"
        elif meas.get("status") == "no_contour" or meas.get("status") == "too_few_points":
            status = "no_interface"

        cos_meas = math.cos(math.radians(theta_meas)) if math.isfinite(theta_meas) else float("nan")

        points.append(CalibrationPoint(
            f_w=fw,
            theta_measured=theta_meas,
            cos_measured=cos_meas,
            status=status,
            n_contour=n_contour,
            drift_deg=drift,
        ))

    # Filter valid points for fitting
    valid = [p for p in points if p.status == "ok" and math.isfinite(p.cos_measured)]

    # Sort by f_w
    valid.sort(key=lambda p: p.f_w)

    fw_vals = [p.f_w for p in valid]
    cos_vals = [p.cos_measured for p in valid]

    # Compute slopes for comparison
    if len(valid) >= 2:
        # Young slope: d(cos)/d(f_w) = psi_l / (psi_l - psi_v)
        young_slope = psi_l / (psi_l - psi_v) if abs(psi_l - psi_v) > 1e-12 else 0.0

        # Measured slope: linear fit of cos vs f_w
        n = len(valid)
        fw_mean = sum(fw_vals) / n
        cos_mean = sum(cos_vals) / n
        ss_xx = sum((fw - fw_mean) ** 2 for fw in fw_vals)
        ss_xy = sum((fw - fw_mean) * (cos - cos_mean) for fw, cos in zip(fw_vals, cos_vals))
        measured_slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
        slope_ratio = measured_slope / young_slope if young_slope != 0 else 0.0

        # Neutral wall (where cos = 0, i.e., theta = 90 deg)
        neutral_fw_young = psi_v / psi_l if psi_l > 0 else 0.0
        # For measured: solve linear fit cos = measured_slope * (fw - fw0)
        neutral_fw_measured = fw_mean - cos_mean / measured_slope if measured_slope != 0 else 0.5
    else:
        young_slope = 0.0
        measured_slope = 0.0
        slope_ratio = 0.0
        neutral_fw_young = 0.0
        neutral_fw_measured = 0.5

    valid_range = (fw_vals[0], fw_vals[-1]) if fw_vals else (0.0, 1.0)

    return CalibrationResult(
        nx=nx,
        ny=ny,
        radius=radius,
        steps=steps,
        wall_rows=wall_rows,
        psi_l=psi_l,
        psi_v=psi_v,
        points=points,
        fw_values=fw_vals,
        cos_values=cos_vals,
        valid_range=valid_range,
        young_slope=young_slope,
        measured_slope=measured_slope,
        slope_ratio=slope_ratio,
        neutral_fw_young=neutral_fw_young,
        neutral_fw_measured=neutral_fw_measured,
    )


# ---------------------------------------------------------------------------
# Forward and inverse
# ---------------------------------------------------------------------------


def forward(cal: CalibrationResult, f_w: float) -> float:
    """Predict contact angle (degrees) for a given wall fraction.

    Uses linear interpolation in cos space if f_w is in the valid range,
    otherwise falls back to the Young relation.
    """
    if cal.fw_values and cal.valid_range[0] <= f_w <= cal.valid_range[1]:
        cos_pred = _linear_interp(cal.fw_values, cal.cos_values, f_w)
        cos_pred = max(-1.0, min(1.0, cos_pred))
        return math.degrees(math.acos(cos_pred))
    else:
        return young_prediction(f_w, cal.psi_l, cal.psi_v)


def invert(
    cal: CalibrationResult,
    theta_target: float,
    *,
    fallback_to_young: bool = False,
) -> float:
    """Find wall fraction f_w that produces the target contact angle.

    Args:
        cal: Calibration result from :func:`calibrate`.
        theta_target: Target contact angle in degrees.
        fallback_to_young: If True and inversion fails, return Young prediction.

    Returns:
        Wall fraction f_w, or NaN if inversion fails.
    """
    cos_target = math.cos(math.radians(theta_target))
    cos_target = max(-1.0, min(1.0, cos_target))

    if not cal.fw_values:
        if fallback_to_young:
            return (cal.psi_v + cos_target * (cal.psi_l - cal.psi_v)) / cal.psi_l
        return float("nan")

    # Check if target is in range
    cos_min = min(cal.cos_values)
    cos_max = max(cal.cos_values)

    if cos_target < cos_min - 0.01 or cos_target > cos_max + 0.01:
        if fallback_to_young:
            return (cal.psi_v + cos_target * (cal.psi_l - cal.psi_v)) / cal.psi_l
        return float("nan")

    fw = _linear_interp_inverse(cal.fw_values, cal.cos_values, cos_target)
    if not math.isfinite(fw) and fallback_to_young:
        return (cal.psi_v + cos_target * (cal.psi_l - cal.psi_v)) / cal.psi_l
    return fw
