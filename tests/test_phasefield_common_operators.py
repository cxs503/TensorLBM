import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.multiphase3d import free_energy_step_3d
from tensorlbm.phasefield.diagnostics import phase_volume_smoothed, phase_volume_threshold
from tensorlbm.phasefield.free_energy import DoubleWellFreeEnergy, force_minus_phi_grad_mu, force_mu_grad_phi
from tensorlbm.phasefield.operators import central_gradient_3d, laplacian_3d


def test_periodic_central_operators_are_second_order_stencils():
    field = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6)
    grad_x, grad_y, grad_z = central_gradient_3d(field, boundary="periodic")
    expected_x = 0.5 * (torch.roll(field, -1, 2) - torch.roll(field, 1, 2))
    expected_y = 0.5 * (torch.roll(field, -1, 1) - torch.roll(field, 1, 1))
    expected_z = 0.5 * (torch.roll(field, -1, 0) - torch.roll(field, 1, 0))
    expected_lap = sum(torch.roll(field, shift, dim) for dim in range(3) for shift in (-1, 1)) - 6 * field
    assert torch.equal(grad_x, expected_x)
    assert torch.equal(grad_y, expected_y)
    assert torch.equal(grad_z, expected_z)
    assert torch.equal(laplacian_3d(field, boundary="periodic"), expected_lap)


def test_no_flux_and_wall_mask_do_not_wrap_through_opposite_boundary():
    field = torch.zeros((3, 3, 4), dtype=torch.float32)
    field[:, :, -1] = 8.0
    grad_x, _, _ = central_gradient_3d(field, boundary="no_flux")
    assert torch.all(grad_x[:, :, 0] == 0.0)
    assert torch.all(laplacian_3d(field, boundary="no_flux")[:, :, 0] == 0.0)

    mask = torch.zeros_like(field, dtype=torch.bool)
    mask[:, :, 1] = True
    grad_x_masked, _, _ = central_gradient_3d(field, boundary="no_flux", solid_mask=mask)
    assert torch.all(grad_x_masked[:, :, 0] == 0.0)


def test_operators_require_3d_scalar_and_explicit_policy():
    field = torch.zeros((2, 2, 2), dtype=torch.float32)
    with pytest.raises(TypeError):
        central_gradient_3d(field)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        laplacian_3d(torch.zeros((2, 2)), boundary="periodic")
    with pytest.raises(ValueError):
        central_gradient_3d(field, boundary="bad")


def test_free_energy_forces_and_volume_diagnostics_are_distinct():
    phi = torch.tensor([[[-1.0, 0.0, 1.0]]], dtype=torch.float32)
    model = DoubleWellFreeEnergy(A=0.2, B=0.3, kappa=0.1)
    mu = model.chemical_potential(phi, boundary="no_flux")
    expected_mu = -0.2 * phi + 0.3 * phi**3 - 0.1 * laplacian_3d(phi, boundary="no_flux")
    assert torch.allclose(mu, expected_mu)
    minus_phi = force_minus_phi_grad_mu(phi, mu, boundary="no_flux")
    mu_grad = force_mu_grad_phi(phi, mu, boundary="no_flux")
    assert not torch.equal(minus_phi[0], mu_grad[0])
    assert phase_volume_smoothed(phi).item() == pytest.approx(1.5)
    assert phase_volume_threshold(phi, threshold=0.0).item() == pytest.approx(1.0)


@pytest.mark.parametrize("force", [force_minus_phi_grad_mu, force_mu_grad_phi])
def test_free_energy_force_rejects_nonmatching_or_broadcastable_field_shapes(force):
    phi = torch.zeros((2, 2, 2), dtype=torch.float32)
    mu = torch.zeros_like(phi)
    with pytest.raises(ValueError, match="3-D scalar"):
        force(torch.zeros((2, 2)), mu, boundary="periodic")
    with pytest.raises(ValueError, match="3-D scalar"):
        force(phi, torch.zeros((2, 2)), boundary="periodic")
    with pytest.raises(ValueError, match="same shape"):
        force(torch.zeros((1, 1, 1)), mu, boundary="periodic")


def test_free_energy_step_matches_pre_common_operator_d3q19_formula():
    torch.manual_seed(7)
    rho = torch.ones((3, 4, 5), dtype=torch.float32)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, zero, zero, zero)
    rho, ux, uy, uz = macroscopic3d(f)
    phi = torch.rand_like(rho) * 2.0 - 1.0
    g = equilibrium3d(phi, zero, zero, zero)
    # The production step obtains the order parameter from g's zeroth moment.
    phi = g.sum(dim=0)
    kwargs = dict(tau_f=0.9, tau_g=0.8, A=0.11, B=0.13, kappa=0.02, Gamma=0.4, gx=0.001, gy=-0.002, gz=0.003)
    got_f, got_g = free_energy_step_3d(f, g, **kwargs)

    mu = -kwargs["A"] * phi + kwargs["B"] * phi**3 - kwargs["kappa"] * (
        torch.roll(phi, 1, 2) + torch.roll(phi, -1, 2) + torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1) + torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0) - 6.0 * phi
    )
    grad_x = 0.5 * (torch.roll(mu, -1, 2) - torch.roll(mu, 1, 2))
    grad_y = 0.5 * (torch.roll(mu, -1, 1) - torch.roll(mu, 1, 1))
    grad_z = 0.5 * (torch.roll(mu, -1, 0) - torch.roll(mu, 1, 0))
    expected_feq = equilibrium3d(
        rho,
        ux + kwargs["tau_f"] * (-phi * grad_x + rho * kwargs["gx"]) / rho.clamp(min=1e-12),
        uy + kwargs["tau_f"] * (-phi * grad_y + rho * kwargs["gy"]) / rho.clamp(min=1e-12),
        uz + kwargs["tau_f"] * (-phi * grad_z + rho * kwargs["gz"]) / rho.clamp(min=1e-12),
    )
    expected_f = f - (f - expected_feq) / kwargs["tau_f"]

    from tensorlbm.d3q19 import C, W
    cv = C.to(phi.device).float()
    csq = (cv[:, 0] ** 2 + cv[:, 1] ** 2 + cv[:, 2] ** 2).view(19, 1, 1, 1)
    w = W.to(phi.device).view(19, 1, 1, 1)
    cu = cv[:, 0].view(19, 1, 1, 1) * ux + cv[:, 1].view(19, 1, 1, 1) * uy + cv[:, 2].view(19, 1, 1, 1) * uz
    u_sq = (ux**2 + uy**2 + uz**2).unsqueeze(0)
    expected_g_eq = w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq) + w * kwargs["Gamma"] * (csq / (1.0 / 3.0) - 3.0) * mu.unsqueeze(0)
    expected_g = g - (g - expected_g_eq) / kwargs["tau_g"]
    assert torch.equal(got_f, expected_f)
    assert torch.equal(got_g, expected_g)
