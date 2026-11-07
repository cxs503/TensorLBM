"""2-D numerical wave-tank benchmark built on the Körner free-surface LBM.

Components
----------
:mod:`~tensorlbm.wave_tank2d.stokes2`
    Analytical Stokes second-order wave theory (dispersion solver, surface
    elevation, kinematics, bound second harmonic) — the reference solution.
:mod:`~tensorlbm.wave_tank2d.goda`
    Goda–Suzuki two-point and least-squares multi-probe incident/reflected
    separation, phasor/harmonic fitting and wavenumber measurement.
:mod:`~tensorlbm.wave_tank2d.periodic`
    Periodic-domain propagation rig: verifies dispersion, wave harmonics and
    liquid-mass conservation without wavemaker or absorber artefacts.
:mod:`~tensorlbm.wave_tank2d.tank`
    Relaxation-zone wavemaker and sponge layer on the same free-surface
    solver, with a probe array and the reflection-coefficient workflow.

All lengths are lattice units.  The lattice gravity ``g`` sets the phase speed
through ``c^2 = (g/k) tanh(k h)``; the campaign keeps ``c/c_s ~ 0.07``
(``c_s = 1/sqrt(3)``) so the compressibility error of the LBM stays below the
1 % dispersion tolerance.
"""

from __future__ import annotations

from .goda import (
    apparent_reflection,
    conditioning,
    fit_harmonics,
    least_squares_reflection,
    phase_speed_from_phasors,
    phasor_from_series,
    recommended_spacing,
    spatial_decay_rate,
    spatiotemporal_harmonics,
    two_point_reflection,
    wave_number_from_phasors,
)
from .periodic import PeriodicWaveConfig, run_periodic, surface_elevation
from .stokes2 import (
    GRAVITY,
    Stokes2Wave,
    dispersion_k,
    dispersion_omega,
    kinematic_residual,
    phase_speed,
    second_order_amplitude,
    stokes_validity,
)

__all__ = [
    "GRAVITY",
    "PeriodicWaveConfig",
    "Stokes2Wave",
    "apparent_reflection",
    "conditioning",
    "dispersion_k",
    "dispersion_omega",
    "fit_harmonics",
    "kinematic_residual",
    "least_squares_reflection",
    "phase_speed",
    "phase_speed_from_phasors",
    "phasor_from_series",
    "recommended_spacing",
    "run_periodic",
    "second_order_amplitude",
    "spatial_decay_rate",
    "spatiotemporal_harmonics",
    "stokes_validity",
    "surface_elevation",
    "two_point_reflection",
    "wave_number_from_phasors",
]