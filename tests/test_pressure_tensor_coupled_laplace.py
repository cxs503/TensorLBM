"""Gate tests for the *coupled* free-energy droplet arm (Laplace law).

The frozen-phi arms in ``test_pressure_tensor_equilibrium_xval.py`` verify that
one equilibrium is a fixed point.  These tests advance **both** distributions,
so the interface relaxes to its own mechanical equilibrium and the Laplace law
``p_in - p_out = sigma / R`` becomes a falsifiable number.

Run: ``python -m pytest tests/test_pressure_tensor_coupled_laplace.py -q``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.phasefield.pressure_tensor_equilibrium import (  # noqa: E402
    analytic_sigma,
    coupled_step,
    droplet_profile,
    chemical_potential,
    fluid_equilibrium,
    moments,
    order_param_equilibrium,
    run_coupled_droplet,
)

A, KAPPA, GAMMA = 0.1, 0.2, 0.2


@pytest.fixture(scope="module")
def coupled():
    return run_coupled_droplet(
        A=A, kappa=KAPPA, radius=16.0, steps=2000, tau_f=1.0, tau_phi=1.0, Gamma=GAMMA
    )


def test_coupled_step_preserves_mass_and_order_parameter():
    n = 32
    phi = droplet_profile(n, n, A=A, kappa=KAPPA, radius=8.0)
    rho = torch.ones_like(phi)
    z = torch.zeros_like(phi)
    f = fluid_equilibrium(rho, z, z, phi, A=A, kappa=KAPPA)
    g = order_param_equilibrium(phi, z, z, chemical_potential(phi, A=A, kappa=KAPPA), Gamma=GAMMA)
    f1, g1 = coupled_step(f, g, A=A, kappa=KAPPA, Gamma=GAMMA, tau_f=1.0, tau_phi=1.0)
    rho1, _, _ = moments(f1)
    assert abs(float(rho1.sum()) - float(f.sum())) < 1e-12
    assert abs(float(g1.sum()) - float(g.sum())) < 1e-12
    assert torch.isfinite(f1).all() and torch.isfinite(g1).all()


def test_laplace_law_within_owner_band(coupled):
    """``dp * R / sigma = 1`` to better than 15 % (owner precision standard)."""
    assert abs(coupled.laplace_tail_rel_error) < 0.15, coupled.laplace_tail_rel_error
    assert coupled.dp_tail_mean > 0.0  # inner pressure higher, as Laplace demands


def test_surface_tension_measured_from_populations(coupled):
    """sigma from the measured tensor ``int (P_rr - P_tt) dr`` vs analytic."""
    assert abs(coupled.sigma_rel_error) < 0.15, coupled.sigma_rel_error
    assert coupled.sigma_rel_error < 0.0  # discretisation bias is systematic (negative)


def test_stability_rules_and_conservation(coupled):
    """No blow-up, spurious current inside the declared band, both masses conserved."""
    assert math.isfinite(coupled.max_speed) and coupled.max_speed < 1e-2
    assert 0.9 < coupled.rho_min < 1.05 and 0.95 < coupled.rho_max < 1.1
    assert coupled.mass_err_rho < 1e-10
    assert coupled.mass_err_phi < 1e-10


def test_series_converges_towards_the_analytic_jump():
    """The jump grows from ~0 towards sigma/R; the last sample is not an outlier."""
    r = run_coupled_droplet(
        A=A, kappa=KAPPA, radius=16.0, steps=1200, tau_f=1.0, tau_phi=1.0, Gamma=GAMMA,
        record_every=200,
    )
    first = r.series[1]["dp_over_sigma_R"]
    last = r.series[-1]["dp_over_sigma_R"]
    assert first < 0.75 and last > 0.7
    assert last > first


def test_low_mobility_is_unstable_and_reported():
    """``tau_phi = 0.8`` (M ~ 0.06) diverges: an honest boundary of the transplant."""
    with pytest.raises(RuntimeError):
        run_coupled_droplet(
            A=A, kappa=KAPPA, radius=10.0, steps=400, tau_f=1.0, tau_phi=0.8, Gamma=GAMMA
        )