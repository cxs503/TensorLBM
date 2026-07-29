from __future__ import annotations

from tensorlbm import SuboffResistanceBenchmarkConfig, run_suboff_resistance_benchmark


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
    assert float(mesh["cell_saving_pct_mean"]) > 0.0
    for it in out["iterations"]:
        assert bool(it["mesh"]["adaptive"]) is True
