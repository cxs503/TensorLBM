import torch

from tensorlbm.phasefield.ch_validation import (
    FreeEnergyCHValidationConfig,
    run_closed_periodic_free_energy_diagnostic,
    uniform_phase_capillary_force,
)


def test_uniform_phase_has_zero_periodic_capillary_force():
    phi = torch.full((3, 4, 5), 0.25, dtype=torch.float32)

    force_x, force_y, force_z = uniform_phase_capillary_force(
        phi, A=0.1, B=0.1, kappa=0.02
    )

    assert torch.count_nonzero(force_x) == 0
    assert torch.count_nonzero(force_y) == 0
    assert torch.count_nonzero(force_z) == 0


def test_closed_periodic_runner_calls_real_step_with_f_and_g_for_multiple_steps(monkeypatch):
    import tensorlbm.phasefield.ch_validation as validation

    real_step = validation.free_energy_step_3d
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recording_step(f, g, **kwargs):
        calls.append((f, g))
        return real_step(f, g, **kwargs)

    monkeypatch.setattr(validation, "free_energy_step_3d", recording_step)
    config = FreeEnergyCHValidationConfig(shape=(3, 4, 5), steps=2, seed=7)
    result = run_closed_periodic_free_energy_diagnostic(config)

    assert len(calls) == 2
    assert all(f.shape == (19, 3, 4, 5) and g.shape == (19, 3, 4, 5) for f, g in calls)
    assert len(result.series) == 3
    assert [sample.step for sample in result.series] == [0, 1, 2]
    assert result.status == "diagnostic_only"
    assert result.physical_acceptance is False
    assert all(sample.phase_is_finite for sample in result.series)
    assert all(sample.f_is_finite for sample in result.series)
    assert all(sample.g_is_finite for sample in result.series)
    assert all(sample.phase_min <= sample.phase_max for sample in result.series)
    assert all(isinstance(sample.phase_integral, float) for sample in result.series)
    assert all(isinstance(sample.f_mass, float) for sample in result.series)


def test_runner_distinguishes_phase_integral_from_f_mass_and_does_not_claim_conservation():
    result = run_closed_periodic_free_energy_diagnostic(
        FreeEnergyCHValidationConfig(shape=(3, 3, 4), steps=2, seed=3)
    )

    assert result.phase_integral_name == "phase_integral=sum(phi), where phi=sum_i(g_i)"
    assert result.f_mass_name == "f_mass=sum_i,x(f_i)"
    assert "g_mass" not in result.__dict__
    assert "phase_volume" not in result.__dict__
    assert result.conservation_claim is False
