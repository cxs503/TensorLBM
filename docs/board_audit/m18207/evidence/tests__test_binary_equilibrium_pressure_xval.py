"""Numeric verification of the free-energy binary-fluid equilibrium pressure.

Anchor: XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE.
Every assertion below binds a measured number to a closed-form free-energy
prediction (no hard-coded measured values):

* coexistence pressure ``p_coex = A^2/(4B)`` — independent Maxwell construction
  (Newton on equal ``mu`` and equal ``p0``) plus the field-measured bulk values;
* ``P_nn`` uniformity across a flat periodic interface (mechanical equilibrium,
  ``div P = 0``);
* ``sigma = int (P_nn - P_tt)`` vs ``(2 sqrt2/3) sqrt(kappa A^3)/B``;
* the time-stepped production free-energy LBM keeps the total normal pressure
  uniform and the residual velocity in the low-Mach band.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.free_energy_stress import FreeEnergyParams, interface_width, surface_tension_analytic
from tensorlbm.phasefield.binary_equilibrium_pressure import (
    EquilibriumPressureConfig,
    diagnose_ch_phase_flux_gap,
    maxwell_coexistence,
    periodic_planar_profile,
    planar_pressure_diagnostics,
    run_planar_equilibrium_lbm,
)


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    torch.set_num_threads(1)


def test_maxwell_coexistence_matches_closed_form() -> None:
    """Independent numerical Maxwell construction reproduces ``A^2/(4B)``."""
    for A, B in ((1.0, 1.0), (0.1, 0.4), (2.5, 0.7)):
        params = FreeEnergyParams(A=A, B=B, kappa=0.5 * A)
        state = maxwell_coexistence(params)
        assert state.residual < 1e-10
        assert state.p_coex == pytest.approx(A * A / (4.0 * B), rel=1e-9)
        assert state.phi_heavy == pytest.approx(math.sqrt(A / B), rel=1e-8)
        assert state.phi_light == pytest.approx(-math.sqrt(A / B), rel=1e-8)


def test_periodic_planar_profile_is_periodic_and_saturated() -> None:
    params = FreeEnergyParams(A=1.0, B=1.0, kappa=2.0)
    field = periodic_planar_profile(8, 128, params)
    assert field.wrap_residual < 1e-5  # periodic to exp(-L/2xi)
    phi = field.phi
    assert float(phi.max()) == pytest.approx(field.phi_eq, rel=1e-6)
    assert float(phi.min()) == pytest.approx(-field.phi_eq, rel=1e-6)
    # exactly two zero crossings per period (two flat interfaces)
    positive = phi[0] > 0
    crossings = int((positive[1:] != positive[:-1]).sum())
    assert crossings == 2


def test_pressure_tensor_predictions_at_production_resolution() -> None:
    """Planar equilibrium: uniform normal pressure and analytic surface tension."""
    params = FreeEnergyParams(A=1.0, B=1.0, kappa=2.0)  # xi = 2 cells
    field = periodic_planar_profile(16, 128, params)
    diag = planar_pressure_diagnostics(field, params, scheme="iso9")

    # C1: bulk pressure equals the closed-form coexistence pressure
    assert diag.p_coex_closed_form == pytest.approx(0.25, rel=1e-12)
    assert diag.coexistence_relative_error < 1e-4
    assert diag.p0_bulk[0] == pytest.approx(diag.p_coex_closed_form, rel=1e-4)
    assert diag.p0_bulk[1] == pytest.approx(diag.p_coex_closed_form, rel=1e-4)

    # C2: normal pressure is uniform to within the pre-declared 25% band
    assert diag.p_nn_max_deviation_relative < 0.25

    # C3: capillary integral matches the closed form inside the 15% standard
    assert diag.sigma_analytic == pytest.approx(4.0 / 3.0, rel=1e-12)
    assert abs(diag.sigma_relative_error) < 0.15


def test_surface_tension_converges_with_interface_resolution() -> None:
    """Refining ``xi/h`` must reduce the capillary error monotonically."""
    errors = []
    for cells in (1.5, 2.0, 3.0):
        params = FreeEnergyParams(A=1.0, B=1.0, kappa=0.5 * cells**2)
        field = periodic_planar_profile(8, 128, params)
        diag = planar_pressure_diagnostics(field, params, scheme="iso9")
        assert diag.sigma_analytic == pytest.approx(surface_tension_analytic(params), rel=1e-12)
        assert interface_width(params) == pytest.approx(cells, rel=1e-12)
        errors.append(abs(diag.sigma_relative_error))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.05


def test_lbm_run_holds_mechanical_equilibrium() -> None:
    """Production Korteweg-force LBM: uniform total normal pressure, low Mach."""
    config = EquilibriumPressureConfig(A=1.0, B=1.0, interface_cells=2.0, nx=64, ny=8, steps=300, sample_every=100)
    run, phi = run_planar_equilibrium_lbm(config)
    assert len(run.max_velocity_history) == 3
    assert run.final_p_nn_total_deviation_relative < 0.25
    assert run.final_max_velocity < 0.15 * math.sqrt(1.0 / 3.0)
    # residual velocity decreases from the impulsive start
    assert run.max_velocity_history[-1] < run.max_velocity_history[0]
    # density stays in the incompressible band the model assumes
    assert 0.9 < run.rho_min < run.rho_max < 1.1
    assert phi.shape == (8, 64)


def test_ch_phase_flux_gap_is_measured_not_assumed() -> None:
    """Guards the documented defect: production ``g`` dissolves an equilibrium droplet."""
    gap = diagnose_ch_phase_flux_gap(nx=32, ny=32, steps=200)
    assert gap["peak_decay_ratio"] < 0.9  # equilibrium peak is NOT preserved
    assert gap["numerical_diffusivity_cs2_tau_minus_half"] == pytest.approx(1.0 / 3.0 * 0.2, rel=1e-12)
    assert gap["diffusion_length_estimate"] > 5.0  # comparable to the droplet radius


def test_config_rejects_unresolved_interface() -> None:
    with pytest.raises(ValueError):
        EquilibriumPressureConfig(interface_cells=0.4)
    with pytest.raises(ValueError):
        EquilibriumPressureConfig(nx=16)