import math

import pytest
import torch

from tensorlbm.phasefield.free_energy import DoubleWellFreeEnergy
from tensorlbm.phasefield.static_droplet import (
    diagnose_static_droplet,
    estimate_droplet_radius,
    initialize_static_droplet,
    periodic_chemical_potential_and_korteweg_force,
)


def test_uniform_phase_has_zero_periodic_korteweg_force():
    phi = torch.full((5, 6, 7), 0.25, dtype=torch.float64)
    model = DoubleWellFreeEnergy(A=0.1, B=0.1, kappa=0.02)

    mu, (force_x, force_y, force_z) = periodic_chemical_potential_and_korteweg_force(phi, model)

    assert torch.allclose(mu, torch.full_like(phi, -0.1 * 0.25 + 0.1 * 0.25**3))
    assert torch.count_nonzero(force_x) == 0
    assert torch.count_nonzero(force_y) == 0
    assert torch.count_nonzero(force_z) == 0


def test_initializer_places_tanh_droplet_at_center_and_uses_periodic_geometry():
    shape = (25, 25, 25)
    phi = initialize_static_droplet(
        shape, radius=4.0, interface_width=1.0, center=(12.0, 12.0, 0.0), dtype=torch.float64
    )

    phi_width = initialize_static_droplet(
        shape, radius=4.0, width=1.0, center=(12.0, 12.0, 0.0), dtype=torch.float64
    )
    assert torch.equal(phi, phi_width)
    assert phi.shape == shape
    assert phi.dtype == torch.float64
    assert phi[12, 12, 0] > 0.99
    # The periodic image of a center at x=0 is one cell away at x=-1.
    assert phi[12, 12, -1] > 0.9
    assert phi[12, 12, 5] < 0.0
    assert estimate_droplet_radius(phi).item() == pytest.approx(4.0, abs=0.35)


def test_initializer_validates_tensor_geometry_parameters():
    with pytest.raises(ValueError, match="three positive"):
        initialize_static_droplet((8, 8), radius=2.0, interface_width=1.0)
    with pytest.raises(ValueError, match="radius"):
        initialize_static_droplet((8, 8, 8), radius=0.0, interface_width=1.0)
    with pytest.raises(ValueError, match="interface_width"):
        initialize_static_droplet((8, 8, 8), radius=2.0, interface_width=0.0)
    with pytest.raises(ValueError, match="center"):
        initialize_static_droplet((8, 8, 8), radius=2.0, interface_width=1.0, center=(1.0, 2.0))
    with pytest.raises(ValueError, match="three positive"):
        initialize_static_droplet((True, 8, 8), radius=2.0, width=1.0)
    with pytest.raises(ValueError, match="radius"):
        initialize_static_droplet((8, 8, 8), radius=True, width=1.0)
    with pytest.raises(ValueError, match="interface_width"):
        initialize_static_droplet((8, 8, 8), radius=2.0, width=True)
    with pytest.raises(ValueError, match="center"):
        initialize_static_droplet((8, 8, 8), radius=2.0, width=1.0, center=(True, 2.0, 2.0))


def test_laplace_style_diagnostic_is_explicitly_withheld_without_pressure_field():
    phi = initialize_static_droplet((25, 25, 25), radius=4.0, interface_width=1.0, dtype=torch.float64)
    result = diagnose_static_droplet(phi, DoubleWellFreeEnergy(A=0.1, B=0.1, kappa=0.02))

    assert result.status == "diagnostic_only"
    assert result.physical_acceptance is False
    assert result.geometry.center_zyx == pytest.approx((12.0, 12.0, 12.0), abs=1e-12)
    assert result.geometry.equivalent_radius == pytest.approx(4.0, abs=0.35)
    assert result.geometry.candidate_mean_curvature == pytest.approx(2.0 / 4.0, abs=0.05)
    assert result.force.net_force == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert result.laplace.status == "withheld"
    assert result.laplace.observed_pressure_jump is None
    assert result.laplace.expected_pressure_jump is None
    assert "pressure" in result.laplace.reason.lower()
    assert "mu" in result.laplace.reason.lower()
    assert "force" in result.laplace.reason.lower()
    assert math.isfinite(result.force.l2_norm)
