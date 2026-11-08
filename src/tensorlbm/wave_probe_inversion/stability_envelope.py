"""Viscosity-stability envelope for wave tank simulations.

Scans kinematic viscosity ν and determines the maximum stable number
of time steps N_max(ν) before numerical instability or excessive damping.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@dataclass
class StabilityPoint:
    """Single point in stability scan.

    Attributes
    ----------
    nu : float
        Kinematic viscosity.
    N_max : int
        Maximum stable number of time steps.
    stable : bool
        Whether the simulation remained stable.
    damping_ratio : float
        Ratio of measured to theoretical amplitude (1.0 = no damping).
    """
    nu: float
    N_max: int
    stable: bool
    damping_ratio: float


@dataclass
class StabilityEnvelope:
    """Result of stability scan.

    Attributes
    ----------
    points : list of StabilityPoint
        All scan points.
    fit_coefficients : tuple of float
        Polynomial fit coefficients for N_max(ν).
    fit_r_squared : float
        R² of the fit.
    """
    points: list = field(default_factory=list)
    fit_coefficients: Optional[tuple] = None
    fit_r_squared: float = 0.0

    def to_dict(self) -> dict:
        return {
            "points": [
                {
                    "nu": p.nu,
                    "N_max": p.N_max,
                    "stable": p.stable,
                    "damping_ratio": p.damping_ratio,
                }
                for p in self.points
            ],
            "fit_coefficients": list(self.fit_coefficients) if self.fit_coefficients else None,
            "fit_r_squared": self.fit_r_squared,
        }


def run_stability_scan(
    nu_values: Optional[list[float]] = None,
    kh: float = 1.5,
    eps: float = 0.05,
    max_steps: int = 10000,
) -> StabilityEnvelope:
    """Run viscosity-stability scan.

    Parameters
    ----------
    nu_values : list of float, optional
        Viscosity values to scan. Default: [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02].
    kh : float
        Non-dimensional wavenumber.
    eps : float
        Wave steepness.
    max_steps : int
        Maximum number of steps to test.

    Returns
    -------
    StabilityEnvelope
    """
    from tensorlbm.wave_tank.tank import WaveTankConfig, run_case

    if nu_values is None:
        nu_values = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02]

    DEPTH = {0.7: 12.0, 1.5: 24.0, 3.0: 40.0}
    h = DEPTH.get(kh, 24.0)
    g = 6.5e-3

    envelope = StabilityEnvelope()

    for nu in nu_values:
        print(f"Testing nu = {nu}")

        # Binary search for N_max
        lo, hi = 100, max_steps
        N_max = lo

        while lo <= hi:
            mid = (lo + hi) // 2

            cfg = WaveTankConfig(
                kh=kh,
                eps=eps,
                h=h,
                g=g,
                nu=nu,
                absorb=True,
                warm_periods=2.0,
                meas_periods=mid / cfg.wave.period if hasattr(cfg, 'wave') else 10.0,
                sample_every=2,
            )

            try:
                # Run simulation
                result = run_case(cfg, quiet=True)

                # Check for NaN/Inf
                eta_final = result.snapshot["eta"]
                if np.any(np.isnan(eta_final)) or np.any(np.isinf(eta_final)):
                    hi = mid - 1
                    continue

                # Check amplitude (should not grow unboundedly)
                eta_max = np.max(np.abs(eta_final))
                if eta_max > 10.0 * eps * h:  # amplitude should be ~eps*h
                    hi = mid - 1
                    continue

                # Stable at this step count
                N_max = mid
                lo = mid + 1

            except Exception as e:
                print(f"  Failed at {mid} steps: {e}")
                hi = mid - 1
                continue

        # Run at N_max to measure damping
        cfg = WaveTankConfig(
            kh=kh,
            eps=eps,
            h=h,
            g=g,
            nu=nu,
            absorb=True,
            warm_periods=2.0,
            meas_periods=max(1.0, N_max / 100.0),
            sample_every=2,
        )

        try:
            result = run_case(cfg, quiet=True)
            probe_eta = result.series["probe_eta"]
            t = result.series["t"]

            # Measure amplitude at middle probe
            mid_probe = probe_eta.shape[0] // 2
            eta = probe_eta[mid_probe]

            # Skip warm-up, measure in measurement window
            n_warm = int(0.3 * len(eta))
            eta_meas = eta[n_warm:]
            amp_measured = 0.5 * (np.max(eta_meas) - np.min(eta_meas))

            # Theoretical amplitude
            amp_theory = eps * h / kh

            damping_ratio = amp_measured / amp_theory if amp_theory > 0 else 1.0

            stable = True
        except Exception as e:
            print(f"  Damping measurement failed: {e}")
            damping_ratio = 0.0
            stable = False

        point = StabilityPoint(
            nu=nu,
            N_max=N_max,
            stable=stable,
            damping_ratio=damping_ratio,
        )
        envelope.points.append(point)
        print(f"  N_max = {N_max}, damping = {damping_ratio:.3f}")

    # Fit N_max(ν)
    envelope.fit_coefficients, envelope.fit_r_squared = _fit_stability_curve(envelope)

    return envelope


def _fit_stability_curve(envelope: StabilityEnvelope) -> tuple[Optional[tuple], float]:
    """Fit N_max(ν) curve.

    Uses polynomial fit: N_max = a + b*ν + c*ν²
    Returns (coefficients, R²).
    """
    if len(envelope.points) < 3:
        return None, 0.0

    nu = np.array([p.nu for p in envelope.points])
    N_max = np.array([p.N_max for p in envelope.points])

    # Fit polynomial
    try:
        coeffs = np.polyfit(nu, N_max, 2)
        N_fit = np.polyval(coeffs, nu)
        ss_res = np.sum((N_max - N_fit) ** 2)
        ss_tot = np.sum((N_max - np.mean(N_max)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return tuple(coeffs), r_squared
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, 0.0
