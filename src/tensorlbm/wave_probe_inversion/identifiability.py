"""Identifiability scan for wave probe inversion.

Scans over protocol parameters (Δx/λ, s/λ, L/λ) and maps the error
in estimated R, k_r, α, and c. Determines the minimum protocol envelope
to achieve target accuracy (3% wave speed, |ΔR|<0.03, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .synthetic_tank import SyntheticWaveConfig, SyntheticWaveTank
from .inversion import (
    TwoPointInversion,
    ThreePointGodaInversion,
    MultiProbeLeastSquares,
    InversionResult,
    fit_harmonic,
)


@dataclass
class IdentifiabilityPoint:
    """Single point in the identifiability scan.

    Attributes
    ----------
    dx_over_lambda : float
        Grid resolution Δx/λ.
    s_over_lambda : float
        Probe spacing s/λ.
    L_over_lambda : float
        Propagation length L/λ.
    R_error : float
        Absolute error in reflection coefficient |R̂ - R*|.
    kr_error : float
        Relative error in wavenumber |k̂_r - k*| / k*.
    alpha_error : float
        Relative error in attenuation |α̂ - α*| / α* (or abs if α*=0).
    c_error : float
        Relative error in phase speed |ĉ - c*| / c*.
    method : str
        Inversion method used.
    """
    dx_over_lambda: float
    s_over_lambda: float
    L_over_lambda: float
    R_error: float
    kr_error: float
    alpha_error: float
    c_error: float
    method: str


@dataclass
class IdentifiabilityResult:
    """Result of identifiability scan.

    Attributes
    ----------
    points : list of IdentifiabilityPoint
        All scan points.
    target_R_error : float
        Target absolute error for R.
    target_kr_error : float
        Target relative error for k_r.
    target_alpha_error : float
        Target relative error for α.
    target_c_error : float
        Target relative error for c.
    min_protocol : dict or None
        Minimum protocol (dx/λ, s/λ, L/λ) achieving all targets.
    """
    points: list = field(default_factory=list)
    target_R_error: float = 0.03
    target_kr_error: float = 0.01
    target_alpha_error: float = 0.10
    target_c_error: float = 0.03
    min_protocol: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "points": [
                {
                    "dx_over_lambda": p.dx_over_lambda,
                    "s_over_lambda": p.s_over_lambda,
                    "L_over_lambda": p.L_over_lambda,
                    "R_error": p.R_error,
                    "kr_error": p.kr_error,
                    "alpha_error": p.alpha_error,
                    "c_error": p.c_error,
                    "method": p.method,
                }
                for p in self.points
            ],
            "target_R_error": self.target_R_error,
            "target_kr_error": self.target_kr_error,
            "target_alpha_error": self.target_alpha_error,
            "target_c_error": self.target_c_error,
            "min_protocol": self.min_protocol,
        }


class IdentifiabilityScan:
    """Identifiability scan for wave probe inversion.

    Scans over:
    - Δx/λ ∈ [1/40, 1/8]: spatial resolution (affects temporal sampling)
    - s/λ ∈ [0.1, 0.75]: probe spacing
    - L/λ ∈ [0.5, 4.0]: propagation length (distance from wavemaker to last probe)

    For each protocol, synthesizes probe time series and runs inversion,
    comparing estimated parameters to ground truth.
    """

    def __init__(
        self,
        base_config: Optional[SyntheticWaveConfig] = None,
        method: str = "multi_probe_ls",
    ):
        """Initialize scan.

        Parameters
        ----------
        base_config : SyntheticWaveConfig, optional
            Base wave configuration. If None, uses defaults.
        method : str
            Inversion method: "two_point", "three_point_goda", "multi_probe_ls".
        """
        if base_config is None:
            base_config = SyntheticWaveConfig()
        self.base_config = base_config
        self.method = method

    def _generate_probe_positions(
        self,
        s_over_lambda: float,
        L_over_lambda: float,
        wavelength: float,
        n_probes: int = 5,
    ) -> list[float]:
        """Generate probe positions.

        Parameters
        ----------
        s_over_lambda : float
            Probe spacing as fraction of wavelength.
        L_over_lambda : float
            Total propagation length as fraction of wavelength.
        wavelength : float
            Wavelength.
        n_probes : int
            Number of probes.

        Returns
        -------
        list of float
            Probe positions.
        """
        s = s_over_lambda * wavelength
        L = L_over_lambda * wavelength

        # Place probes starting from x = 0.5λ (after wavemaker zone)
        x_start = 0.5 * wavelength
        positions = [x_start + i * s for i in range(n_probes)]

        # Ensure last probe is within L
        while positions[-1] > x_start + L and len(positions) > 3:
            positions.pop()

        return positions

    def _compute_samples_per_period(
        self,
        dx_over_lambda: float,
    ) -> int:
        """Compute samples per period from spatial resolution.

        For a wave propagating at speed c, temporal resolution Δt should
        resolve the wave period. With spatial resolution Δx, the CFL-like
        condition gives Δt ~ Δx / c, so samples_per_period ~ λ / Δx.
        """
        # samples_per_period = λ / Δx = 1 / (Δx/λ)
        return max(20, int(1.0 / dx_over_lambda))

    def run_single(
        self,
        dx_over_lambda: float,
        s_over_lambda: float,
        L_over_lambda: float,
        R_star: float = 0.1,
        alpha_star: float = 0.001,
        n_probes: int = 5,
    ) -> IdentifiabilityPoint:
        """Run inversion for one protocol configuration.

        Parameters
        ----------
        dx_over_lambda : float
            Spatial resolution.
        s_over_lambda : float
            Probe spacing.
        L_over_lambda : float
            Propagation length.
        R_star : float
            True reflection coefficient.
        alpha_star : float
            True attenuation coefficient.
        n_probes : int
            Number of probes.

        Returns
        -------
        IdentifiabilityPoint
        """
        cfg = self.base_config
        wavelength = cfg.wavelength

        # Update config for this scan point
        samples_per_period = self._compute_samples_per_period(dx_over_lambda)
        cfg = SyntheticWaveConfig(
            wavelength=wavelength,
            amplitude=cfg.amplitude,
            depth=cfg.depth,
            gravity=cfg.gravity,
            R_star=R_star,
            alpha_star=alpha_star,
            n_harmonics=cfg.n_harmonics,
            n_periods=cfg.n_periods,
            samples_per_period=samples_per_period,
            warm_periods=cfg.warm_periods,
        )

        # Generate probe positions
        x_probes = self._generate_probe_positions(
            s_over_lambda, L_over_lambda, wavelength, n_probes
        )

        # Synthesize probe time series
        tank = SyntheticWaveTank(cfg)
        t, etas = tank.synthesize_probes(x_probes, x_origin=0.0)

        # Extract complex amplitudes
        omega = cfg.omega
        c_probes = np.array([fit_harmonic(t, eta, omega) for eta in etas])

        # Run inversion
        if self.method == "two_point":
            inv = TwoPointInversion(omega, cfg.k_r)
            result = inv.invert(x_probes[0], x_probes[1], c_probes[0], c_probes[1])
        elif self.method == "three_point_goda":
            inv = ThreePointGodaInversion(omega, cfg.k_r)
            result = inv.invert(
                x_probes[0], x_probes[1], x_probes[2],
                c_probes[0], c_probes[1], c_probes[2]
            )
        elif self.method == "multi_probe_ls":
            inv = MultiProbeLeastSquares(omega, cfg.k_r, alpha_star)
            result = inv.invert(np.array(x_probes), c_probes)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute errors
        k_r_true = cfg.k_r
        c_true = cfg.celerity

        R_error = abs(result.R_hat - R_star)
        kr_error = abs(result.k_r_hat - k_r_true) / k_r_true
        if alpha_star > 1e-10:
            alpha_error = abs(result.alpha_hat - alpha_star) / alpha_star
        else:
            alpha_error = abs(result.alpha_hat - alpha_star)
        c_error = abs(result.c_hat - c_true) / c_true

        return IdentifiabilityPoint(
            dx_over_lambda=dx_over_lambda,
            s_over_lambda=s_over_lambda,
            L_over_lambda=L_over_lambda,
            R_error=R_error,
            kr_error=kr_error,
            alpha_error=alpha_error,
            c_error=c_error,
            method=self.method,
        )

    def run_scan(
        self,
        dx_values: Optional[list[float]] = None,
        s_values: Optional[list[float]] = None,
        L_values: Optional[list[float]] = None,
        R_star: float = 0.1,
        alpha_star: float = 0.001,
        n_probes: int = 5,
    ) -> IdentifiabilityResult:
        """Run full identifiability scan.

        Parameters
        ----------
        dx_values : list of float, optional
            Δx/λ values to scan. Default: [1/40, 1/30, 1/20, 1/15, 1/10, 1/8].
        s_values : list of float, optional
            s/λ values to scan. Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75].
        L_values : list of float, optional
            L/λ values to scan. Default: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0].
        R_star : float
            True reflection coefficient.
        alpha_star : float
            True attenuation coefficient.
        n_probes : int
            Number of probes.

        Returns
        -------
        IdentifiabilityResult
        """
        if dx_values is None:
            dx_values = [1/40, 1/30, 1/20, 1/15, 1/10, 1/8]
        if s_values is None:
            s_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
        if L_values is None:
            L_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

        result = IdentifiabilityResult()

        for dx in dx_values:
            for s in s_values:
                for L in L_values:
                    try:
                        point = self.run_single(
                            dx, s, L, R_star, alpha_star, n_probes
                        )
                        result.points.append(point)
                    except Exception as e:
                        # Skip failed configurations
                        print(f"Failed dx={dx}, s={s}, L={L}: {e}")
                        continue

        # Find minimum protocol achieving targets
        result.min_protocol = self._find_min_protocol(result)

        return result

    def _find_min_protocol(self, result: IdentifiabilityResult) -> Optional[dict]:
        """Find minimum protocol achieving all targets.

        Returns the configuration with smallest L/λ that meets all criteria,
        breaking ties by largest s/λ (fewer probes needed).
        """
        passing = []
        for p in result.points:
            if (
                p.R_error < result.target_R_error
                and p.kr_error < result.target_kr_error
                and p.alpha_error < result.target_alpha_error
                and p.c_error < result.target_c_error
            ):
                passing.append(p)

        if not passing:
            return None

        # Sort by L/λ (ascending), then s/λ (descending)
        passing.sort(key=lambda p: (p.L_over_lambda, -p.s_over_lambda))

        best = passing[0]
        return {
            "dx_over_lambda": best.dx_over_lambda,
            "s_over_lambda": best.s_over_lambda,
            "L_over_lambda": best.L_over_lambda,
            "R_error": best.R_error,
            "kr_error": best.kr_error,
            "alpha_error": best.alpha_error,
            "c_error": best.c_error,
        }
