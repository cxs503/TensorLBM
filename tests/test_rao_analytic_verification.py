"""Analytic verification of ``tensorlbm.rao_analysis`` (seakeeping RAO / SRA).

Every assertion below is against a closed form that does not share code with the
module under test:

* single-DOF damped oscillator  |H(ω)| = |F| / √[(C − (m+A)ω²)² + (Bω)²]
  (Faltinsen 1990, ch. 3; the module documents the same equation but solved as a
  complex 6x6 system, so the closed form is an independent reference);
* decoupled natural frequency ω_n = √(C_ii / (M_ii + A_ii)) and the coupled
  version checked through det(C − ω²M) = 0 (no eigenvalue routine reused);
* Bretschneider/ISSC wave-spectrum moments in closed form
  m_n = (A/4)·B^{(n−4)/4}·Γ(1 − n/4) for n < 4, with S(ω) = (A/ω⁵)exp(−B/ω⁴);
* Rayleigh extreme-value factor for N zero-crossing cycles,
  E[max amp] / (2√m0) = √(2 ln N)/2 (Longuet-Higgins 1952).

The transfer-function check is leakage-free by construction (integer DFT bin),
so the cross-spectrum estimator can be held to 1e-3 relative.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.rao_analysis import (
    RAOData,
    compare_rao_with_experiment,
    compute_natural_frequencies,
    compute_rao_from_hydrodynamics,
    compute_rao_from_timeseries,
    rao_peak_detection,
    spectral_response_analysis,
)

HEAVE = 2  # DOF index used throughout (Surge, Sway, Heave, ...)

M_BODY = 1000.0  # heave mass [kg]  (light body: strong m+A vs C cancellation)
A_ADDED = 500.0  # added mass A(0) [kg]
B_DAMP = 2000.0  # radiation damping [kg/s]
C_STIFF = 1.0e5  # hydrostatic stiffness [N/m]
F_UNIT = 1.0  # excitation amplitude [N]


def _heave_inputs(omega: torch.Tensor, f0: float = F_UNIT):
    """Single modelled DOF (heave) on a 6-DOF grid."""
    n = omega.numel()
    mass = torch.zeros(6, 6, dtype=torch.float64)
    mass[HEAVE, HEAVE] = M_BODY
    added = torch.zeros(n, 6, 6, dtype=torch.float64)
    added[:, HEAVE, HEAVE] = A_ADDED
    damping = torch.zeros(n, 6, 6, dtype=torch.float64)
    damping[:, HEAVE, HEAVE] = B_DAMP
    stiffness = torch.zeros(6, 6, dtype=torch.float64)
    stiffness[HEAVE, HEAVE] = C_STIFF
    excitation = torch.zeros(n, 6, dtype=torch.float64)
    excitation[:, HEAVE] = f0
    return added, damping, stiffness, mass, excitation


def _closed_form_heave(omega: torch.Tensor, f0: float = F_UNIT) -> torch.Tensor:
    return f0 / torch.sqrt(
        (C_STIFF - (M_BODY + A_ADDED) * omega**2) ** 2 + (B_DAMP * omega) ** 2
    )


# --------------------------------------------------------------- RAO from Z(ω)
def test_heave_rao_matches_closed_form_damped_oscillator() -> None:
    """|H(ω)| of the complex-matrix solve vs the scalar oscillator solution."""
    omega_n = math.sqrt(C_STIFF / (M_BODY + A_ADDED))
    # 0.2 ω_n … 5 ω_n, deliberately straddling the resonance peak and the
    # post-resonance cancellation dip where cancellation error is worst.
    omega = torch.linspace(0.2 * omega_n, 5.0 * omega_n, 601, dtype=torch.float64)
    added, damping, stiffness, mass, excitation = _heave_inputs(omega)
    rao = compute_rao_from_hydrodynamics(
        omega, added, damping, stiffness, mass, excitation
    )

    reference = _closed_form_heave(omega)
    rel_error = (rao.rao_amplitude[:, HEAVE] - reference).abs() / reference

    assert rao.rao_amplitude.shape == (601, 6)
    assert rel_error.max().item() < 1e-12
    # other DOFs are unmodelled -> exactly zero response, not NaN / inf
    assert torch.equal(rao.rao_amplitude[:, 0], torch.zeros(601, dtype=torch.float64))
    assert rao.inactive_dofs == (0, 1, 3, 4, 5)


def test_rao_resonance_amplitude_and_phase_match_closed_form() -> None:
    """At ω = ω_n: |H| = |F|/(B ω_n) and the phase lags by exactly 90°."""
    omega_n = math.sqrt(C_STIFF / (M_BODY + A_ADDED))
    omega = torch.tensor([omega_n], dtype=torch.float64)
    added, damping, stiffness, mass, excitation = _heave_inputs(omega)
    rao = compute_rao_from_hydrodynamics(
        omega, added, damping, stiffness, mass, excitation
    )

    reference = F_UNIT / (B_DAMP * omega_n)
    assert rao.rao_amplitude[0, HEAVE].item() == pytest.approx(reference, rel=1e-12)
    assert rao.rao_phase[0, HEAVE].item() == pytest.approx(-math.pi / 2, abs=1e-10)


def test_rao_static_and_deep_water_limits() -> None:
    """ω → 0 gives the static stiffness response; ω → ∞ decays as F/((m+A)ω²)."""
    omega = torch.tensor([1e-4, 5.0e3], dtype=torch.float64)
    added, damping, stiffness, mass, excitation = _heave_inputs(omega)
    rao = compute_rao_from_hydrodynamics(
        omega, added, damping, stiffness, mass, excitation
    )
    assert rao.rao_amplitude[0, HEAVE].item() == pytest.approx(
        F_UNIT / C_STIFF, rel=1e-8
    )
    assert rao.rao_amplitude[1, HEAVE].item() == pytest.approx(
        F_UNIT / ((M_BODY + A_ADDED) * 5000.0**2), rel=1e-8
    )


def test_rao_is_float64_and_does_not_raise_on_partial_dof_model() -> None:
    """Regression: the 6x6 solve used to raise _LinAlgError for a heave-only model."""
    omega = torch.linspace(0.1, 2.0, 7, dtype=torch.float64)
    added, damping, stiffness, mass, excitation = _heave_inputs(omega)
    rao = compute_rao_from_hydrodynamics(
        omega, added, damping, stiffness, mass, excitation
    )
    assert isinstance(rao, RAOData)
    assert rao.rao_complex.dtype == torch.complex128
    assert torch.isfinite(rao.rao_amplitude).all()


# ------------------------------------------------------------ natural frequency
def test_natural_frequencies_match_decoupled_closed_form() -> None:
    mass = torch.diag(torch.tensor([2.0e3, 2.0e3, M_BODY, 4.0e3, 5.0e3, 6.0e3], dtype=torch.float64))
    added = torch.diag(torch.tensor([1.0e2, 1.0e2, A_ADDED, 1.0e3, 1.0e3, 1.0e3], dtype=torch.float64))
    stiffness = torch.diag(
        torch.tensor([1.0e3, 1.0e3, C_STIFF, 5.0e5, 5.0e5, 5.0e5], dtype=torch.float64)
    )
    omega_n = compute_natural_frequencies(mass, added, stiffness)

    reference = torch.sqrt(torch.diagonal(stiffness) / torch.diagonal(mass + added))
    assert torch.allclose(omega_n, reference, rtol=1e-12)
    assert omega_n[HEAVE].item() == pytest.approx(
        math.sqrt(C_STIFF / (M_BODY + A_ADDED)), rel=1e-12
    )


def test_coupled_natural_frequencies_satisfy_characteristic_polynomial() -> None:
    """Heave–pitch coupling: det(C − ω²M) = 0 at every returned ω_n."""
    mass = torch.zeros(6, 6, dtype=torch.float64)
    mass[HEAVE, HEAVE] = 1000.0
    mass[4, 4] = 8.0e5  # pitch inertia
    mass[HEAVE, 4] = mass[4, HEAVE] = 1.2e4  # inertial coupling
    added = torch.zeros(6, 6, dtype=torch.float64)
    added[HEAVE, HEAVE] = 500.0
    added[4, 4] = 2.0e5
    stiffness = torch.zeros(6, 6, dtype=torch.float64)
    stiffness[HEAVE, HEAVE] = 1.0e5
    stiffness[4, 4] = 6.0e7
    stiffness[HEAVE, 4] = stiffness[4, HEAVE] = 1.5e6

    omega_n = compute_natural_frequencies(mass, added, stiffness)
    coupled = torch.tensor([HEAVE, 4])
    values = omega_n[coupled]
    assert torch.isfinite(values).all()

    total_mass = mass + added
    for value in values:
        det = torch.linalg.det(
            stiffness[coupled][:, coupled] - value.item() ** 2 * total_mass[coupled][:, coupled]
        )
        assert abs(det.item()) < 1e-6 * 1.0e5 * 6.0e7  # |det| small vs C scale

    # Independent 2x2 closed form for the same generalized problem.
    m11 = total_mass[HEAVE, HEAVE].item()
    m12 = total_mass[HEAVE, 4].item()
    m22 = total_mass[4, 4].item()
    k11 = stiffness[HEAVE, HEAVE].item()
    k12 = stiffness[HEAVE, 4].item()
    k22 = stiffness[4, 4].item()
    a = m11 * m22 - m12**2
    b = -(k11 * m22 + k22 * m11 - 2.0 * k12 * m12)
    c = k11 * k22 - k12**2
    disc = math.sqrt(max(b * b - 4.0 * a * c, 0.0))
    roots = sorted(
        [(-b - disc) / (2.0 * a), (-b + disc) / (2.0 * a)]
    )
    assert sorted(v.item() ** 2 for v in values) == pytest.approx(roots, rel=1e-9)


def test_natural_frequencies_report_nan_for_unmodelled_dof() -> None:
    """Regression: singular mass matrix used to raise _LinAlgError."""
    mass = torch.zeros(6, 6, dtype=torch.float64)
    mass[HEAVE, HEAVE] = M_BODY
    added = torch.zeros(6, 6, dtype=torch.float64)
    added[HEAVE, HEAVE] = A_ADDED
    stiffness = torch.zeros(6, 6, dtype=torch.float64)
    stiffness[HEAVE, HEAVE] = C_STIFF

    omega_n = compute_natural_frequencies(mass, added, stiffness)
    assert omega_n[HEAVE].item() == pytest.approx(
        math.sqrt(C_STIFF / (M_BODY + A_ADDED)), rel=1e-12
    )
    others = [i for i in range(6) if i != HEAVE]
    assert torch.isnan(omega_n[others]).all()


# ------------------------------------------------------------- wave spectrum
def _bretschneider(omega: torch.Tensor, a_coef: float, b_coef: float) -> torch.Tensor:
    safe = torch.clamp(omega, min=1e-9)
    return a_coef / safe**5 * torch.exp(-b_coef / safe**4)


def test_bretschneider_moments_match_gamma_closed_form() -> None:
    """Spectral moments m_n = (A/4)·B^{(n−4)/4}·Γ(1−n/4) for n = 0, 2."""
    hs, t1 = 6.0, 10.0
    a_coef = 173.0 * hs**2 / t1**4
    b_coef = 691.0 / t1**4
    omega = torch.linspace(0.02, 40.0, 60001, dtype=torch.float64)
    spectrum = _bretschneider(omega, a_coef, b_coef)

    result = spectral_response_analysis(omega, spectrum, torch.ones_like(omega))
    m0_ref = a_coef / 4.0 * b_coef ** (-1.0) * math.gamma(1.0)
    m2_ref = a_coef / 4.0 * b_coef ** (-0.5) * math.gamma(0.5)

    assert result["m0"].item() == pytest.approx(m0_ref, rel=1e-5)
    assert result["m2"].item() == pytest.approx(m2_ref, rel=1e-3)
    # ISSC coefficient set (173, 691) is Hs²/16 rounded: 0.14 % literature check.
    assert m0_ref / (hs**2 / 16.0) == pytest.approx(1.0014, abs=2e-3)
    assert m0_ref / (hs**2 / 16.0) == pytest.approx(1.0, rel=5e-3)


def test_sra_statistics_match_closed_form_for_unit_rao() -> None:
    """|RAO| = 1: sa = 2√m0 = Hs/2, Tz = 2π(πB)^{−1/4}, E[max]/sa = √(2 ln 1000)/2."""
    hs, t1 = 6.0, 10.0
    a_coef = 173.0 * hs**2 / t1**4
    b_coef = 691.0 / t1**4
    omega = torch.linspace(0.02, 40.0, 60001, dtype=torch.float64)
    spectrum = _bretschneider(omega, a_coef, b_coef)

    result = spectral_response_analysis(omega, spectrum, torch.ones_like(omega))
    m0 = a_coef / (4.0 * b_coef)
    m2 = a_coef / 4.0 * b_coef ** (-0.5) * math.gamma(0.5)

    assert result["significant_amplitude"].item() == pytest.approx(2.0 * math.sqrt(m0), rel=1e-5)
    assert result["significant_amplitude"].item() == pytest.approx(hs / 2.0, rel=5e-3)
    assert result["mean_period"].item() == pytest.approx(
        2.0 * math.pi * math.sqrt(m0 / m2), rel=1e-3
    )
    rayleigh = math.sqrt(2.0 * math.log(1000)) / 2.0
    assert result["max_response_estimate"].item() / result["significant_amplitude"].item() == pytest.approx(
        rayleigh, rel=1e-10
    )
    assert rayleigh == pytest.approx(1.8586, abs=2e-4)  # 1.86 in the module docstring


def test_sra_sorts_descending_frequency_grid() -> None:
    """Regression: a descending ω grid used to give m0 < 0 and a NaN amplitude."""
    omega = torch.linspace(3.0, 0.05, 200, dtype=torch.float64)
    spectrum = _bretschneider(omega, 173.0 * 36.0 / 1e4, 691.0 / 1e4)
    result = spectral_response_analysis(omega, spectrum, torch.ones_like(omega))

    assert result["m0"].item() > 0.0
    assert torch.isfinite(result["significant_amplitude"])
    assert result["significant_amplitude"].item() > 0.0
    assert bool(torch.all(result["omega"][1:] > result["omega"][:-1]))


# -------------------------------------------------------------- cross-spectrum
def test_timeseries_cross_spectrum_recovers_transfer_function() -> None:
    """x = 0.5·η with a 0.3 rad lag must come back as |RAO| = 0.5, φ = −0.3."""
    n, dt = 4096, 0.125
    time = torch.arange(n, dtype=torch.float64) * dt
    omega_true = 2.0 * math.pi * 128 / (n * dt)  # exactly DFT bin 128
    elevation = torch.cos(omega_true * time)
    response = 0.5 * torch.cos(omega_true * time - 0.3)

    omega, amplitude, phase = compute_rao_from_timeseries(time, elevation, response)
    index = int(torch.argmin(torch.abs(omega - omega_true)))

    assert amplitude[index].item() == pytest.approx(0.5, rel=1e-3)
    assert phase[index].item() == pytest.approx(-0.3, abs=1e-3)
    # Regression: empty bins used to return O(1/clamp) garbage that dominated
    # argmax (observed 1.0e2 at a spurious frequency before the fix).
    assert amplitude.max().item() < 0.55
    assert int(torch.argmax(amplitude)) == pytest.approx(index, abs=3)


def test_rao_peak_detection_reports_resonance_period() -> None:
    omega = torch.linspace(0.1, 3.0, 300, dtype=torch.float64)
    amplitude = 1.0 / torch.sqrt((1.0 - omega**2) ** 2 + (0.05 * omega) ** 2)
    peaks = rao_peak_detection(omega, amplitude)

    assert peaks["n_peaks"] == 1
    assert peaks["dominant_omega"] == pytest.approx(1.0, abs=0.02)
    assert peaks["dominant_period"] == pytest.approx(2.0 * math.pi, abs=0.15)


def test_compare_rao_with_experiment_error_metrics() -> None:
    omega = torch.linspace(0.5, 2.5, 41, dtype=torch.float64)
    reference = _closed_form_heave(omega)
    identical = compare_rao_with_experiment(omega, reference, omega, reference)
    assert identical["mean_relative_error"].item() == pytest.approx(0.0, abs=1e-15)
    assert identical["pass"] is True

    offset = compare_rao_with_experiment(
        omega, reference * 1.10, omega, reference, tolerance_relative=0.05
    )
    assert offset["mean_relative_error"].item() == pytest.approx(0.10, rel=1e-6)
    assert offset["max_relative_error"].item() == pytest.approx(0.10, rel=1e-6)
    assert offset["pass"] is False