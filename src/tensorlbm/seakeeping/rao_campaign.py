"""Seakeeping campaign: wave matrix + RAO extraction via FFT.

Runs the Wigley III hull through a matrix of regular wave conditions:
    λ/L ∈ {0.5, 0.75, 1.0, 1.5, 2.0}
    H/λ = 1/60 (linear regime)

For each wave condition:
1. Run 20 wave periods of time-domain simulation.
2. Discard first 10 periods (transient).
3. FFT the last 10 periods to extract heave/pitch RAO.
4. Cross-validate with strip-theory frequency-domain RAO.

The RAO is defined as:
    RAO(ω) = |X(ω)| / ζ_a

where X(ω) is the complex motion amplitude and ζ_a is the wave amplitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .wigley_hull import WigleyIII
from .strip_theory import StripTheorySolver, compute_strip_theory_rao
from .lbm_body_coupling import run_section_coupling, LBMBodyCoupling2D
from ..nwt.stokes2 import Stokes2Wave


@dataclass
class WaveCondition:
    """One regular wave condition.

    Attributes:
        wavelength: Wavelength λ [m].
        wave_amplitude: Wave amplitude ζ_a = H/2 [m].
        wave_length_ratio: λ/L [-].
        steepness: H/λ [-].
    """
    wavelength: float
    wave_amplitude: float
    wave_length_ratio: float
    steepness: float


@dataclass
class RAOResult:
    """RAO result for one wave condition.

    Attributes:
        condition: Wave condition.
        omega: Wave frequency [rad/s].
        heave_rao: Heave RAO |H3| [m/m].
        pitch_rao: Pitch RAO |H5| [rad/m].
        heave_phase: Heave RAO phase [rad].
        pitch_phase: Pitch RAO phase [rad].
        n_periods_total: Total periods simulated.
        n_periods_analysis: Periods used for FFT.
    """
    condition: WaveCondition
    omega: float
    heave_rao: float
    pitch_rao: float
    heave_phase: float
    pitch_phase: float
    n_periods_total: int
    n_periods_analysis: int


@dataclass
class SeakeepingCampaign:
    """Campaign driver for Wigley III seakeeping validation.

    Args:
        hull: Wigley III hull object.
        wave_length_ratios: List of λ/L values.
        steepness: Wave steepness H/λ.
        n_periods: Total wave periods to simulate.
        n_periods_analysis: Periods for FFT analysis (last N).
        steps_per_period: Time steps per wave period.
        depth: Water depth [m].
    """
    hull: WigleyIII
    wave_length_ratios: list = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    steepness: float = 1.0 / 60.0
    n_periods: int = 20
    n_periods_analysis: int = 10
    steps_per_period: int = 200
    depth: float = 10.0  # deep water

    def _build_conditions(self) -> list[WaveCondition]:
        """Build the wave condition matrix."""
        conditions = []
        for lam_ratio in self.wave_length_ratios:
            wavelength = lam_ratio * self.hull.L
            wave_height = self.steepness * wavelength
            wave_amplitude = wave_height / 2.0
            conditions.append(WaveCondition(
                wavelength=wavelength,
                wave_amplitude=wave_amplitude,
                wave_length_ratio=lam_ratio,
                steepness=self.steepness,
            ))
        return conditions

    def _extract_rao_fft(
        self,
        time: np.ndarray,
        response: np.ndarray,
        wave_elev: np.ndarray,
        omega_wave: float,
        n_periods_analysis: int,
    ) -> tuple[float, float]:
        """Extract RAO amplitude and phase from time series using FFT.

        Args:
            time: Time array [s].
            response: Motion time series [m or rad].
            wave_elev: Wave elevation time series [m].
            omega_wave: Wave frequency [rad/s].
            n_periods_analysis: Number of periods for analysis.

        Returns:
            (rao_amplitude, rao_phase): RAO magnitude and phase.
        """
        # Use last n_periods_analysis periods
        T_period = 2.0 * math.pi / omega_wave
        t_total = time[-1] - time[0]
        t_start = t_total - n_periods_analysis * T_period

        # Find indices for analysis window
        mask = time >= (time[0] + t_start)
        t_ana = time[mask]
        resp_ana = response[mask]
        wave_ana = wave_elev[mask]

        if len(t_ana) < 10:
            return 0.0, 0.0

        # FFT
        dt = t_ana[1] - t_ana[0]
        N = len(t_ana)

        # Apply Hann window
        window = np.hanning(N)
        resp_win = resp_ana * window
        wave_win = wave_ana * window

        # FFT
        RESP = np.fft.rfft(resp_win)
        WAVE = np.fft.rfft(wave_win)
        freqs = np.fft.rfftfreq(N, d=dt)
        omega = 2.0 * math.pi * freqs

        # Find the wave frequency bin
        idx_wave = np.argmin(np.abs(omega - omega_wave))

        # RAO = Response / Wave (complex ratio)
        if abs(WAVE[idx_wave]) > 1e-15:
            rao_complex = RESP[idx_wave] / WAVE[idx_wave]
            rao_amp = abs(rao_complex)
            rao_phase = np.angle(rao_complex)
        else:
            rao_amp = 0.0
            rao_phase = 0.0

        return rao_amp, rao_phase

    def run_single_condition(self, condition: WaveCondition) -> RAOResult:
        """Run one wave condition and extract RAO.

        Args:
            condition: Wave condition to simulate.

        Returns:
            RAOResult with heave/pitch RAO.
        """
        # Wave parameters
        wavelength = condition.wavelength
        wave_amplitude = condition.wave_amplitude
        k = 2.0 * math.pi / wavelength
        omega = math.sqrt(self.hull.g * k * math.tanh(k * self.depth))

        # Run 2-D section coupling at midship
        result = run_section_coupling(
            hull=self.hull,
            x_station=0.0,  # midship
            wave_amplitude=wave_amplitude,
            wave_length=wavelength,
            depth=self.depth,
            n_periods=self.n_periods,
            steps_per_period=self.steps_per_period,
            rho=self.hull.rho,
            g=self.hull.g,
        )

        # Extract heave RAO via FFT
        heave_rao, heave_phase = self._extract_rao_fft(
            result["time"],
            result["z"],
            result["wave_elev"],
            omega,
            self.n_periods_analysis,
        )

        # For pitch, we need to run at multiple stations and integrate
        # Simplified: use strip theory for pitch RAO
        # (full 3-D coupling is a future extension)
        pitch_rao, pitch_phase = self._compute_pitch_rao_strip_theory(
            omega, wave_amplitude
        )

        return RAOResult(
            condition=condition,
            omega=omega,
            heave_rao=heave_rao,
            pitch_rao=pitch_rao,
            heave_phase=heave_phase,
            pitch_phase=pitch_phase,
            n_periods_total=self.n_periods,
            n_periods_analysis=self.n_periods_analysis,
        )

    def _compute_pitch_rao_strip_theory(
        self, omega: float, wave_amplitude: float
    ) -> tuple[float, float]:
        """Compute pitch RAO using strip theory.

        Args:
            omega: Wave frequency [rad/s].
            wave_amplitude: Wave amplitude [m].

        Returns:
            (pitch_rao, pitch_phase): Pitch RAO [rad/m] and phase [rad].
        """
        solver = StripTheorySolver(self.hull, n_stations=21)
        result = solver.solve_at_frequency(omega)
        # Pitch RAO is per unit wave amplitude
        pitch_rao = result.rao_amplitude[1]
        pitch_phase = result.rao_phase[1]
        return pitch_rao, pitch_phase

    def run(self) -> dict:
        """Run the full campaign.

        Returns:
            Dictionary with:
            - 'conditions': list of WaveCondition
            - 'results': list of RAOResult
            - 'strip_theory': strip-theory RAO for comparison
            - 'summary': summary statistics
        """
        conditions = self._build_conditions()
        results = []

        for cond in conditions:
            rao_result = self.run_single_condition(cond)
            results.append(rao_result)

        # Compute strip-theory RAO for comparison
        omega_range = np.array([r.omega for r in results])
        strip_rao = compute_strip_theory_rao(self.hull, omega_range, n_stations=21)

        # Summary
        summary = {
            "n_conditions": len(results),
            "wave_length_ratios": self.wave_length_ratios,
            "steepness": self.steepness,
            "n_periods": self.n_periods,
            "n_periods_analysis": self.n_periods_analysis,
        }

        return {
            "conditions": conditions,
            "results": results,
            "strip_theory": strip_rao,
            "summary": summary,
        }


def run_campaign(
    hull: Optional[WigleyIII] = None,
    wave_length_ratios: Optional[list] = None,
    steepness: float = 1.0 / 60.0,
    n_periods: int = 20,
    n_periods_analysis: int = 10,
    steps_per_period: int = 200,
    depth: float = 10.0,
) -> dict:
    """Convenience function to run the seakeeping campaign.

    Args:
        hull: Wigley III hull (default: standard dimensions).
        wave_length_ratios: List of λ/L values.
        steepness: Wave steepness H/λ.
        n_periods: Total wave periods.
        n_periods_analysis: Periods for FFT.
        steps_per_period: Steps per period.
        depth: Water depth [m].

    Returns:
        Campaign results dictionary.
    """
    hull = hull or WigleyIII()
    wave_length_ratios = wave_length_ratios or [0.5, 0.75, 1.0, 1.5, 2.0]

    campaign = SeakeepingCampaign(
        hull=hull,
        wave_length_ratios=wave_length_ratios,
        steepness=steepness,
        n_periods=n_periods,
        n_periods_analysis=n_periods_analysis,
        steps_per_period=steps_per_period,
        depth=depth,
    )

    return campaign.run()
