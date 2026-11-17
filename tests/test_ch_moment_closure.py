"""Analytic moments, CH dispersion, conservation and negative controls."""

import math

import pytest
import torch

from tensorlbm.phasefield.ch_benchmarks import run_linear_mode, run_planar_interface
from tensorlbm.phasefield.ch_equilibrium import ch_equilibrium


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_phase_moments_do_not_contain_ordinary_phi_diffusion(dimension, dtype):
    if dimension == 2:
        from tensorlbm.d2q9 import C
    else:
        from tensorlbm.d3q19 import C
    rng = torch.Generator().manual_seed(741)
    shape = (4,) * dimension
    phi = torch.rand(shape, generator=rng, dtype=dtype) * 2 - 1
    mu = torch.rand(shape, generator=rng, dtype=dtype) * 0.1
    velocity = tuple(torch.rand(shape, generator=rng, dtype=dtype) * 0.02 for _ in range(dimension))
    Gamma = 0.37
    g = ch_equilibrium(phi, velocity, mu, Gamma)
    atol = 2e-7 if dtype == torch.float32 else 3e-15
    torch.testing.assert_close(g.sum(0), phi, atol=atol, rtol=0)
    c = C.to(dtype=dtype)
    for a in range(dimension):
        ca = c[:, a].view((-1,) + (1,) * dimension)
        torch.testing.assert_close((ca * g).sum(0), phi * velocity[a], atol=atol, rtol=0)
        for b in range(dimension):
            cb = c[:, b].view((-1,) + (1,) * dimension)
            expected = phi * velocity[a] * velocity[b] + (Gamma * mu if a == b else 0)
            torch.testing.assert_close((ca * cb * g).sum(0), expected, atol=atol, rtol=0)


@pytest.mark.parametrize("dimension,axis", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("background", [0.0, 1.0])
def test_growth_and_decay_against_continuum_in_each_direction(dimension, axis, background):
    result = run_linear_mode(dimension=dimension, axis=axis, background=background)
    assert result["passed"], result
    assert result["relative_rate_error"] < 0.005


@pytest.mark.parametrize("dimension", [2, 3])
def test_diffusive_grid_refinement_is_second_order(dimension):
    results = [
        run_linear_mode(dimension=dimension, n=n, steps=100 * (n // 16) ** 2) for n in (16, 32, 64)
    ]
    assert all(r["passed"] for r in results)
    errors = [r["relative_rate_error"] for r in results]
    for coarse, fine in zip(errors, errors[1:]):
        assert 1.8 < math.log2(coarse / fine) < 2.2


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("tau_g,Gamma", [(0.8, 0.2), (1.2, 0.4), (1.0, 0.0)])
def test_mobility_scaling_and_zero_mobility(dimension, tau_g, Gamma):
    result = run_linear_mode(dimension=dimension, tau_g=tau_g, Gamma=Gamma)
    assert result["passed"], result


@pytest.mark.parametrize("dimension", [2, 3])
def test_legacy_control_is_rejected_by_the_same_zero_mobility_gate(dimension):
    result = run_linear_mode(dimension=dimension, Gamma=0.0, geq_convention="swift")
    assert not result["passed"]
    assert not result["checks"]["rate_or_zero_mobility"]
    assert result["max_amplitude_relative_change"] > 0.5


def test_evolved_planar_interface_retains_surface_tension_and_phase():
    result = run_planar_interface(steps=1000)
    assert result["passed"], result
    assert result["sigma_relative_error"] < 0.03
    legacy = run_planar_interface(steps=1000, geq_convention="swift")
    assert not legacy["passed"]
    assert legacy["sigma_relative_error"] > 0.5


def test_closed_tank_corrected_phase_is_conserved_and_solid_storage_is_untouched():
    from tensorlbm.boundary.tank import stream_tank
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.multiphase import free_energy_step, init_free_energy_g

    phi = torch.linspace(-0.1, 0.1, 20, dtype=torch.float64).expand(16, 20).clone()
    zero = torch.zeros_like(phi)
    mask = torch.zeros_like(phi, dtype=torch.bool)
    mask[-1] = True
    mask[:, [0, -1]] = True
    mask[8:10, 8:12] = True
    f = equilibrium(torch.ones_like(phi), zero, zero)
    g = init_free_energy_g(phi, geq_convention="lee_lin")
    initial = g.clone()
    mass = g[:, ~mask].sum()
    for _ in range(200):
        f, g = free_energy_step(
            f, g, tau_g=1.0, geq_convention="lee_lin", boundary="no_flux", solid_mask=mask
        )
        f, g = stream_tank(f, g, mask)
    torch.testing.assert_close(g[:, ~mask].sum(), mass, atol=1e-12, rtol=0)
    torch.testing.assert_close(g[:, mask], initial[:, mask], atol=0, rtol=0)


@pytest.mark.parametrize("Gamma", [-1, float("nan"), float("inf")])
def test_invalid_mobility_rejected(Gamma):
    phi = torch.zeros((4, 5))
    with pytest.raises(ValueError, match="Gamma"):
        ch_equilibrium(phi, (phi, phi), Gamma=Gamma)


def test_broadcastable_chemical_potential_is_rejected():
    phi = torch.zeros((4, 5))
    with pytest.raises(ValueError, match="shape"):
        ch_equilibrium(phi, (phi, phi), torch.zeros((1, 5)))


@pytest.mark.parametrize("dimension", [2, 3])
def test_corrected_collision_advects_with_physical_half_force_velocity(dimension):
    if dimension == 2:
        from tensorlbm.d2q9 import C, equilibrium
        from tensorlbm.multiphase import free_energy_step, init_free_energy_g
    else:
        from tensorlbm.d3q19 import C
        from tensorlbm.d3q19 import equilibrium3d as equilibrium
        from tensorlbm.multiphase3d import free_energy_step_3d as free_energy_step
        from tensorlbm.multiphase3d import init_free_energy_g_3d as init_free_energy_g

    rho = torch.ones((4,) * dimension, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    phi = rho * 0.4
    f = equilibrium(rho, *([zero] * dimension))
    g = init_free_energy_g(phi, geq_convention="lee_lin")
    _, g = free_energy_step(f, g, tau_g=1, Gamma=0, gy=0.002, geq_convention="lee_lin")
    cy = C[:, 1].view((-1,) + (1,) * dimension)
    torch.testing.assert_close((cy * g).sum(0), phi * 0.001, atol=1e-15, rtol=0)


def test_failed_cli_preserves_strict_json_before_returning_nonzero(tmp_path, monkeypatch):
    import json

    from benchmarks.phasefield_ch_core import run

    failed = run_linear_mode(Gamma=0, geq_convention="swift")
    report = {
        "passed": False,
        "status": "failed_ch_core",
        "cases": [failed],
        "invalid_diagnostic": float("inf"),
    }
    monkeypatch.setattr(run, "run_suite", lambda convention: report)
    path = tmp_path / "failure.json"
    assert run.main(["--output", str(path)]) == 1
    saved = json.loads(path.read_text())
    assert saved["invalid_diagnostic"] is None
    assert not saved["cases"][0]["passed"]


def test_benchmark_rejects_nonfinite_collision(monkeypatch):
    import tensorlbm.phasefield.ch_benchmarks as benchmark

    def broken(f, g, **kwargs):
        return f, g * float("nan")

    monkeypatch.setattr(benchmark, "free_energy_step", broken)
    result = benchmark.run_linear_mode()
    assert not result["passed"]
    assert not result["checks"]["completed_finite"]
    assert result["measured_rate"] is None


def test_impact_calibration_matches_free_energy_and_corrected_equilibrium():
    from examples.droplet_impact import _resolve_phasefield_parameters, parser

    args = parser().parse_args([])
    _resolve_phasefield_parameters(args)
    assert args.geq_convention == "lee_lin"
    assert args.kappa == pytest.approx(args.A * args.interface_width**2 / 2)
    args = parser().parse_args(["--kappa", "0.02", "--interface-width", "3"])
    with pytest.raises(ValueError, match="interface-width"):
        _resolve_phasefield_parameters(args)
