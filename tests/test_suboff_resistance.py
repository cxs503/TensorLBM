from __future__ import annotations

from tensorlbm import SuboffResistanceBenchmarkConfig, run_suboff_resistance_benchmark
from tensorlbm import suboff_resistance
from tensorlbm.suboff_resistance import run_suboff_resistance_runtime


def test_suboff_resistance_benchmark_reaches_target() -> None:
    cfg = SuboffResistanceBenchmarkConfig(
        hull_type="full",
        base_length_lu=40.0,
        max_iterations=3,
        target_error_pct=3.0,
    )
    out = run_suboff_resistance_benchmark(cfg)
    assert out["name"] == "suboff_resistance"
    assert out["target_met"] is True
    assert float(out["final_error_pct"]) <= 3.0
    assert len(out["iterations"]) >= 1
    for it in out["iterations"]:
        assert "drag_lu" in it
        assert "lbm" in it
        assert float(it["lbm"]["tau"]) > 0.5


def test_suboff_resistance_cli_quick_profile_reaches_target() -> None:
    cfg = SuboffResistanceBenchmarkConfig(
        hull_type="full",
        base_length_lu=48.0,
        max_iterations=3,
        target_error_pct=3.0,
    )
    out = run_suboff_resistance_benchmark(cfg)
    assert out["target_met"] is True
    assert float(out["final_error_pct"]) <= 3.0


def test_suboff_resistance_iteration_error_drops() -> None:
    cfg = SuboffResistanceBenchmarkConfig(
        hull_type="full",
        base_length_lu=48.0,
        max_iterations=4,
        target_error_pct=0.5,  # force all iterations
    )
    out = run_suboff_resistance_benchmark(cfg)
    errs = [float(i["error_pct"]) for i in out["iterations"] if i["error_pct"] is not None]
    assert errs
    assert min(errs) <= 3.0


def test_suboff_resistance_adaptive_mesh_quantitative_metrics() -> None:
    cfg = SuboffResistanceBenchmarkConfig(
        hull_type="full",
        base_length_lu=40.0,
        max_iterations=2,
        target_error_pct=5.0,
        lbm_steps=30,
        lbm_warmup_steps=10,
        use_adaptive_mesh=True,
    )
    out = run_suboff_resistance_benchmark(cfg)
    mesh = out["adaptive_mesh"]
    assert mesh["enabled"] is True
    assert float(mesh["active_cells_mean"]) > 0.0
    assert float(mesh["finest_uniform_cells_mean"]) > float(mesh["active_cells_mean"])
    for it in out["iterations"]:
        assert bool(it["mesh"]["adaptive"]) is True


def test_suboff_runtime_reports_closed_per_step_momentum_operator_budget() -> None:
    observation = run_suboff_resistance_runtime(SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
        lbm_steps=10, lbm_warmup_steps=0, lbm_sample_interval=2,
        momentum_budget_diagnostic=True, momentum_budget_interval=1,
    ))
    budget = observation["conservation"]["source_attribution"]["momentum"]["per_step_budget"]
    assert budget["status"] == "measured"
    assert budget["units"] == "lattice momentum per time step (rho_lu * dx_lu^4 / dt_lu)"
    boundary_flux = budget["boundary_flux"]
    assert boundary_flux["status"] == "measured"
    assert boundary_flux["kind"] == "face_integrated_population_momentum_flux"
    assert boundary_flux["sign_convention"] == (
        "outward control-volume transport: inlet normal is -x, "
        "outlet normal is +x; net_outward=inlet_outward+outlet_outward"
    )
    assert boundary_flux["closure"] == {
        "status": "withheld",
        "reason": "face_flux_is_not_a_bc_population_delta",
    }
    # Face transport is observed separately; these remain BC operator deltas.
    assert "face_flux" in budget["samples"][0]
    assert "inlet_boundary" in budget["samples"][0]
    assert budget["body_force"]["status"] == "unavailable"
    assert budget["coverage"] == "full_per_step"
    assert budget["sampled_step_indices"] == list(range(1, 11))
    assert budget["sample_count"] == 10
    assert len(budget["samples"]) == 10
    assert all(len(step["fluid_momentum_delta"]) == 3 for step in budget["samples"])
    assert all(len(step["unexplained_residual"]) == 3 for step in budget["samples"])
    ledger = budget["samples"][0]["operator_domain_ledger"]
    assert ledger["domain"] == "entire retained D3Q19 population array"
    assert ledger["operator_identity"]["status"] == "measured"
    assert ledger["operator_identity"]["meaning"].endswith("not a physical control-volume closure")
    assert ledger["streaming"]["implementation"] == "periodic torch.roll permutation"
    assert ledger["streaming"]["expected"] == "zero_global_population_momentum_delta"
    assert max(abs(value) for value in budget["samples"][0]["streaming"]) < 1.0e-9
    assert ledger["wall_impulse"]["fluid_momentum_change"] == budget["samples"][0]["wall_exchange"]
    assert ledger["solid_impulse"]["fluid_momentum_change"] == budget["samples"][0]["solid_exchange"]
    for impulse in (ledger["wall_impulse"], ledger["solid_impulse"]):
        reaction = impulse["reaction_on_wall"] if "reaction_on_wall" in impulse else impulse["reaction_on_solid"]
        assert reaction == [-value for value in impulse["fluid_momentum_change"]]
    cumulative = budget["cumulative_sampled"]
    explained = [sum(cumulative[name][axis] for name in (
        "collision", "streaming", "inlet_boundary", "outlet_boundary", "wall_exchange",
        "solid_exchange", "unexplained_residual",
    )) for axis in range(3)]
    assert max(abs(cumulative["fluid_momentum_delta"][axis] - explained[axis])
               for axis in range(3)) < 1.0e-9
    attribution = observation["conservation"]["source_attribution"]["momentum"]["operator_attribution"]
    assert attribution["status"] == "measured"
    assert attribution["dominant_operator"] in {
        "collision", "inlet_boundary", "outlet_boundary", "wall_exchange", "solid_exchange",
    }


def test_suboff_runtime_default_disables_operator_budget() -> None:
    observation = run_suboff_resistance_runtime(SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
        lbm_steps=10, lbm_warmup_steps=0, lbm_sample_interval=2,
    ))
    momentum = observation["conservation"]["source_attribution"]["momentum"]
    assert momentum["per_step_budget"] is None
    assert momentum["operator_budget"]["status"] == "disabled"
    assert momentum["operator_attribution"] == {
        "status": "withheld", "reason": "operator_budget_disabled"}


def test_suboff_runtime_operator_budget_samples_without_claiming_closure() -> None:
    observation = run_suboff_resistance_runtime(SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
        lbm_steps=10, lbm_warmup_steps=0, lbm_sample_interval=2,
        momentum_budget_diagnostic=True, momentum_budget_interval=3,
    ))
    momentum = observation["conservation"]["source_attribution"]["momentum"]
    budget = momentum["operator_budget"]
    assert budget["coverage"] == "sampled"
    assert budget["sampled_step_indices"] == [1, 4, 7, 10]
    assert budget["sample_count"] == 4
    assert len(budget["samples"]) == 4
    assert budget["closure"]["status"] == "withheld"
    assert momentum["per_step_budget"] is None
    assert momentum["operator_attribution"]["status"] == "sampled"


def test_suboff_operator_budget_reductions_only_run_for_samples(monkeypatch) -> None:
    calls = 0
    original = suboff_resistance._lattice_momentum

    def counted(f):
        nonlocal calls
        calls += 1
        return original(f)

    monkeypatch.setattr(suboff_resistance, "_lattice_momentum", counted)
    config = SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
        lbm_steps=10, lbm_warmup_steps=0, lbm_sample_interval=2,
        momentum_budget_diagnostic=True, momentum_budget_interval=3,
    )
    run_suboff_resistance_benchmark(config)
    assert calls == 1 + 8 * 4  # initial snapshot plus eight reductions/sample

    calls = 0
    run_suboff_resistance_benchmark(SuboffResistanceBenchmarkConfig(
        base_length_lu=20.0, max_length_lu=20.0, max_iterations=1,
        lbm_steps=10, lbm_warmup_steps=0, lbm_sample_interval=2,
    ))
    assert calls == 0
