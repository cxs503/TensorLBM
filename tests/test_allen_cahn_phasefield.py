"""Analytic verification of the Allen-Cahn phase-field kernel (m6394).

Anchors (all lattice-unit, no calibration constants):

* A1 - the equilibrium profile ``tanh(s/W)`` is stationary;
* A2 - the thickness relaxes to ``W = sqrt(2 kappa/beta)`` from a 2x-too-thick start;
* A3 - the driven travelling wave moves at ``v = (3/2) M h xi``.

Evidence: ``docs/allen_cahn_phasefield_r1_20261108.md`` and
``artifacts/allen_cahn_r1_20261108/results.json``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.allen_cahn_lbm import (
    allen_cahn_step,
    free_energy_parameters,
    initialize_allen_cahn,
    interface_profile_1d,
    measure_interface_width,
    step_allen_cahn_field,
)


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _crossing(phi: torch.Tensor) -> float:
    values = phi[0, 0].to(torch.float64)
    idx = int(torch.nonzero(values >= 0.0).max())
    return idx + (0.0 - float(values[idx])) / (float(values[idx + 1]) - float(values[idx]))


def test_free_energy_parameters_reproduce_sigma_and_thickness():
    """The (sigma, W) -> (beta, kappa) map must be an exact inverse."""
    params = free_energy_parameters(0.01, 4.0)
    assert params.beta == pytest.approx(3.0 * 0.01 / (2.0 * 4.0))
    assert params.kappa == pytest.approx(3.0 * 0.01 * 4.0 / 4.0)
    sigma_back = (2.0 * math.sqrt(2.0) / 3.0) * math.sqrt(params.beta * params.kappa)
    thickness_back = math.sqrt(2.0 * params.kappa / params.beta)
    assert sigma_back == pytest.approx(0.01, rel=1e-12)
    assert thickness_back == pytest.approx(4.0, rel=1e-12)


def test_interface_width_measurement_recovers_nominal_width():
    """The +0.5/-0.5 crossing estimator must be unbiased to sub-cell accuracy."""
    for W in (2.0, 3.0, 4.0, 6.0):
        measured = measure_interface_width(interface_profile_1d(96, W, center=48.0))
        # lattice sampling of tanh biases the crossing distance by O(h^2/W)
        assert abs(measured - W) / W < 0.03


def test_equilibrium_profile_is_stationary():
    """A1: the analytic tanh profile must not drift (no spurious interface motion)."""
    params = free_energy_parameters(0.01, 4.0)
    reference = interface_profile_1d(96, 4.0, center=48.0)
    phi = reference.clone()
    for _ in range(2001):
        phi = step_allen_cahn_field(
            phi, beta=params.beta, kappa=params.kappa, mobility=1.0, boundary="no_flux"
        )
    assert abs(measure_interface_width(phi) - 4.0) / 4.0 < 0.01
    assert float((phi - reference).abs().max()) < 0.01
    assert abs(_crossing(phi) - _crossing(reference)) < 0.05


def test_thickness_relaxes_to_equilibrium_width():
    """A2: starting at twice the equilibrium thickness must converge to W."""
    params = free_energy_parameters(0.01, 4.0)
    phi = interface_profile_1d(96, 8.0, center=48.0)
    assert measure_interface_width(phi) > 7.5
    for _ in range(6001):
        phi = step_allen_cahn_field(
            phi, beta=params.beta, kappa=params.kappa, mobility=1.0, boundary="no_flux"
        )
    assert abs(measure_interface_width(phi) - 4.0) / 4.0 < 0.01


def test_travelling_wave_speed_matches_analytic_asymptote():
    """A3: constant chemical-potential offset h drives v = (3/2) M h xi.

    The 10% band (measured 7.4%) reflects the first-order transient of the
    discrete travelling wave, which is not exactly self-similar at finite
    lattice spacing; the anchor is a band, not a tight equality.
    """
    params = free_energy_parameters(0.01, 4.0)
    xi = math.sqrt(2.0 * params.kappa / params.beta)
    mobility, driving, steps = 1.0, 5.0e-4, 600
    phi = interface_profile_1d(160, 4.0, center=60.0)
    x0 = _crossing(phi)
    for _ in range(steps):
        phi = step_allen_cahn_field(
            phi,
            beta=params.beta,
            kappa=params.kappa,
            mobility=mobility,
            boundary="no_flux",
            driving=driving,
        )
    speed = (_crossing(phi) - x0) / steps
    analytic = 1.5 * mobility * driving * xi
    assert speed > 0.0, "a positive h must drive the interface in +x"
    assert abs(speed - analytic) / analytic < 0.10


def test_phase_field_stays_bounded_and_conserves_phase_volume():
    """The clamp and the no-flux AC flux must not create or destroy phase volume."""
    params = free_energy_parameters(0.01, 4.0)
    phi = interface_profile_1d(96, 4.0, center=48.0)
    before = float(phi.sum())
    for _ in range(500):
        phi = step_allen_cahn_field(
            phi, beta=params.beta, kappa=params.kappa, mobility=1.0, boundary="no_flux"
        )
    assert float(phi.min()) >= -1.0 and float(phi.max()) <= 1.0
    assert abs(float(phi.sum()) - before) / abs(before) < 1e-6


def test_unverified_hydro_coupling_is_gated():
    """The coupled variable-density BGK step is WITHHELD: it must fail loudly.

    Measured divergence (``artifacts/allen_cahn_r1_20261108/results.json``):
    step 5 at the default stratified setup, step 5 for a 1000:1 and step 12 for
    a 10:1 static droplet.  A silent NaN return is a trap, so the default path
    raises instead.
    """
    f, g, phi = initialize_allen_cahn(8, 8, 8)
    with pytest.raises(RuntimeError, match="UNVERIFIED"):
        allen_cahn_step(f, g, phi)
    # the documented escape hatch must actually run and expose the defect
    f2, g2, phi2 = initialize_allen_cahn(8, 8, 8)
    finite_at = None
    for step in range(1, 30):
        f2, g2, phi2 = allen_cahn_step(f2, g2, phi2, allow_unverified_hydro=True)
        if not torch.isfinite(f2).all():
            finite_at = step
            break
    assert finite_at is not None and finite_at <= 30