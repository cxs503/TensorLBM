"""Tests for the unified reporter/callback protocol (tensorlbm.reporters).

Covers:

* interval semantics — ``steps=100, interval=25`` fires exactly 4 times;
* the zero-reporter red line — ``run`` output is bit-identical to chained
  ``step`` calls and 1e-6-close to the standalone function pipeline;
* EarlyStop trigger logic (threshold, patience, streak reset, NaN bail,
  min_step, diag-key monitors);
* ThroughputReporter MLUPS arithmetic (deterministic fake clock) and a
  real executor run sanity check;
* FieldSampleReporter end-to-end: HDF5 snapshots + catalog registration
  queryable through the R2 product path;
* the same hook on the Triton fused solver (CUDA+Triton hosts only).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import tensorlbm
from tensorlbm import dispatch as report_dispatch
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.lbm_step import LBMStepExecutor
from tensorlbm.reporters import (
    CallbackReporter,
    EarlyStopReporter,
    FieldSampleReporter,
    Reporter,
    StepContext,
    ThroughputReporter,
)
from tensorlbm.solver3d import collide_bgk3d, stream3d

DEVICE = torch.device("cpu")
DTYPE = torch.float32
ATOL = 1e-6
NZ, NY, NX = 16, 16, 16
N_CELLS = NZ * NY * NX
CODE_SHA = "a" * 40


def _make_f(seed: int = 42) -> torch.Tensor:
    """Valid distribution: equilibrium at rho=1 with a small perturbation."""
    torch.manual_seed(seed)
    rho0 = torch.ones(NZ, NY, NX, device=DEVICE, dtype=DTYPE)
    ux0 = torch.full_like(rho0, 0.05)
    uy0 = torch.full_like(rho0, 0.02)
    uz0 = torch.full_like(rho0, 0.01)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=DEVICE)
    return f + 0.0005 * torch.randn_like(f)


def _make_mask() -> torch.Tensor:
    zz, yy, xx = torch.meshgrid(torch.arange(NZ), torch.arange(NY), torch.arange(NX), indexing="ij")
    r = NZ // 6
    return ((xx - NX // 2) ** 2 + (yy - NY // 2) ** 2 + (zz - NZ // 2) ** 2) < r * r


def _make_executor(**overrides) -> LBMStepExecutor:
    kwargs: dict = {
        "lattice": "D3Q19",
        "collide_fn": "bgk",
        "device": DEVICE,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "tau": 0.6,
        "boundary_fn": bounce_back_cells_3d,
        "boundary_kwargs": {"mask": _make_mask()},
    }
    kwargs.update(overrides)
    return LBMStepExecutor(**kwargs)


def _fake_ctx(step: int, **overrides) -> StepContext:
    """Minimal CPU context for reporter unit tests (no host loop)."""
    kwargs: dict = {
        "step": step,
        "f": torch.zeros(19, 2, 2, 2),
        "lattice": "D3Q19",
        "num_cells": 8,
        "num_steps": 100,
    }
    kwargs.update(overrides)
    return StepContext(**kwargs)


# ---------------------------------------------------------------------------
# 1. Interval semantics
# ---------------------------------------------------------------------------


class TestIntervalSemantics:
    def test_interval_25_over_100_steps_fires_exactly_4_times(self):
        seen: list[int] = []
        reporter = CallbackReporter(lambda ctx: seen.append(ctx.step), interval=25)
        executor = _make_executor()
        executor.run(_make_f(), 100, reporters=[reporter])
        assert seen == [25, 50, 75, 100]

    def test_interval_1_fires_every_step_including_last(self):
        seen: list[int] = []
        reporter = CallbackReporter(lambda ctx: seen.append(ctx.step), interval=1)
        executor = _make_executor()
        executor.run(_make_f(), 10, reporters=[reporter])
        assert seen == list(range(1, 11))

    def test_no_fire_at_step_zero_and_none_for_zero_steps(self):
        seen: list[int] = []
        reporter = CallbackReporter(lambda ctx: seen.append(ctx.step), interval=25)
        executor = _make_executor()
        f_out, diags = executor.run(_make_f(), 0, reporters=[reporter])
        assert seen == []
        assert diags == []
        assert f_out.shape == (19, NZ, NY, NX)

    def test_step_counter_persists_across_run_calls(self):
        seen: list[int] = []
        reporter = CallbackReporter(lambda ctx: seen.append(ctx.step), interval=25)
        executor = _make_executor(reporters=[reporter])
        f = _make_f()
        executor.run(f, 60)
        executor.run(f, 40)
        # Second call continues the cadence: 75, 100 (no duplicate 50).
        assert seen == [25, 50, 75, 100]

    def test_per_call_reporters_override_constructor_list(self):
        constructor_seen: list[int] = []
        call_seen: list[int] = []
        executor = _make_executor(
            reporters=[CallbackReporter(lambda ctx: constructor_seen.append(ctx.step))]
        )
        executor.run(_make_f(), 10)
        assert constructor_seen == list(range(1, 11))
        # Explicit empty per-call list silences the constructor reporters
        # (but still advances the persistent step counter).
        executor.run(_make_f(), 10, reporters=[])
        assert constructor_seen == list(range(1, 11))
        # An explicit per-call list is used for that call only; the counter
        # is now at 20, so interval=5 fires at 25 and 30.
        per_call = CallbackReporter(lambda ctx: call_seen.append(ctx.step), interval=5)
        executor.run(_make_f(), 10, reporters=[per_call])
        assert call_seen == [25, 30]
        assert constructor_seen == list(range(1, 11))


# ---------------------------------------------------------------------------
# 2. Zero-reporter red line (backward compatibility)
# ---------------------------------------------------------------------------


class TestNoReporterRegression:
    def test_run_bitwise_identical_to_chained_step(self):
        f0 = _make_f()
        executor = _make_executor()
        f_run, diags_run = executor.run(f0.clone(), 25)
        # Same executor, explicit empty reporter list → identical fast path.
        f_ref = f0.clone()
        diags_ref: list[dict] = []
        for _ in range(25):
            f_ref, diag = executor.step(f_ref)
            diags_ref.append(diag)
        assert torch.equal(f_run, f_ref)
        assert diags_run == diags_ref

    def test_run_with_empty_reporters_matches_default_run(self):
        f0 = _make_f()
        executor = _make_executor()
        f_default, _ = executor.run(f0.clone(), 30)
        f_empty, _ = executor.run(f0.clone(), 30, reporters=[])
        assert torch.equal(f_default, f_empty)

    def test_run_matches_standalone_pipeline_within_1e6(self):
        """1e-6 agreement with the pre-protocol reference: the standalone
        collide→stream→boundary function pipeline used by the workers."""
        f0 = _make_f()
        tau = 0.6
        mask = _make_mask()
        f_ref = f0.clone()
        for _ in range(25):
            f_ref = collide_bgk3d(f_ref, tau)
            f_ref = stream3d(f_ref)
            f_ref = bounce_back_cells_3d(f_ref, mask)
        executor = _make_executor(tau=tau)
        f_ex, _ = executor.run(f0.clone(), 25)
        assert torch.allclose(f_ex, f_ref, atol=ATOL)

    def test_reporting_path_does_not_change_numerics(self):
        """Even with reporters active the loop must step identically."""
        f0 = _make_f()
        executor = _make_executor()
        f_plain, _ = executor.run(f0.clone(), 20)
        reporter = CallbackReporter(lambda ctx: None, interval=5)
        executor.run(f0.clone(), 1, reporters=[reporter])  # warm-up irrelevant
        f_reported, _ = executor.run(f0.clone(), 20, reporters=[reporter])
        assert torch.equal(f_plain, f_reported)


# ---------------------------------------------------------------------------
# 3. Protocol / dispatcher unit behaviour
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_duck_typed_reporter_without_base_class(self):
        class Minimal:
            def __init__(self, interval: int) -> None:
                self.interval = interval
                self.steps: list[int] = []

            def __call__(self, ctx: StepContext) -> None:
                self.steps.append(ctx.step)

        minimal = Minimal(interval=10)
        assert isinstance(minimal, Reporter)  # structural protocol
        for step in range(1, 31):
            report_dispatch(_fake_ctx(step), [minimal])
        assert minimal.steps == [10, 20, 30]

    def test_dispatch_clamps_invalid_intervals(self):
        fired: list[int] = []

        class Bad:
            interval = 0  # invalid → clamped to 1

            def __call__(self, ctx: StepContext) -> None:
                fired.append(ctx.step)

        report_dispatch(_fake_ctx(3), [Bad()])
        assert fired == [3]

    def test_base_class_validates_interval(self):
        with pytest.raises(ValueError, match="interval"):
            ThroughputReporter(interval=0)
        with pytest.raises(ValueError, match="interval"):
            CallbackReporter(lambda ctx: None, interval=True)
        with pytest.raises(TypeError, match="callable"):
            CallbackReporter(42)

    def test_ctx_macroscopic_fallback_matches_standalone(self):
        f = _make_f()
        ctx = _fake_ctx(1, f=f, num_cells=N_CELLS)
        rho, ux, uy, uz = ctx.macroscopic()
        rho_ref, ux_ref, uy_ref, uz_ref = macroscopic3d(f)
        assert torch.allclose(rho, rho_ref, atol=ATOL)
        assert torch.allclose(ux, ux_ref, atol=ATOL)
        assert torch.allclose(uy, uy_ref, atol=ATOL)
        assert torch.allclose(uz, uz_ref, atol=ATOL)

    def test_ctx_macroscopic_unknown_lattice_raises(self):
        ctx = _fake_ctx(1, f=torch.zeros(15, 2, 2, 2))
        with pytest.raises(ValueError, match="Q=15"):
            ctx.macroscopic()

    def test_ctx_state_shared_between_reporters(self):
        collector: dict[str, int] = {}

        def writer(ctx: StepContext) -> None:
            ctx.state["last"] = ctx.step

        def reader(ctx: StepContext) -> None:
            collector["seen"] = ctx.state["last"]

        for step in (1, 2):
            report_dispatch(_fake_ctx(step), [writer, reader])
        assert collector == {"seen": 2}

    def test_package_level_exports(self):
        for name in (
            "Reporter",
            "ReporterBase",
            "StepContext",
            "dispatch",
            "CallbackReporter",
            "ThroughputReporter",
            "EarlyStopReporter",
            "FieldSampleReporter",
        ):
            assert hasattr(tensorlbm, name), name


# ---------------------------------------------------------------------------
# 4. EarlyStopReporter
# ---------------------------------------------------------------------------


class TestEarlyStop:
    def test_constant_monitor_stops_after_patience(self):
        early = EarlyStopReporter(monitor=lambda ctx: 1.0, threshold=1e-12, patience=2, interval=5)
        executor = _make_executor(reporters=[early])
        f_out, diags = executor.run(_make_f(), 100)
        # Fires at 5 (baseline), 10 (streak 1), 15 (streak 2 → stop).
        assert early.stopped and early.stopped_at == 15
        assert len(diags) == 15
        assert early.reason is not None and "threshold" in early.reason

    def test_streak_resets_after_large_change(self):
        values = iter([3.0, 2.9, 1.0, 1.0, 1.0])
        early = EarlyStopReporter(
            monitor=lambda ctx: next(values), threshold=0.5, patience=2, interval=1
        )
        for step in range(1, 6):
            ctx = _fake_ctx(step)
            early(ctx)
            if ctx.stop:
                break
        # Sequence: 3.0 baseline; 2.9 Δ0.1 streak 1; 1.0 Δ1.9 streak 0;
        # 1.0 streak 1; 1.0 streak 2 → stop at the 5th fire.
        assert early.stopped and early.stopped_at == 5
        assert early.streak == 2

    def test_relative_threshold(self):
        values = iter([1.0, 1.04, 1.0816])
        early = EarlyStopReporter(
            monitor=lambda ctx: next(values),
            threshold=0.05,
            patience=1,
            relative=True,
            interval=1,
        )
        ctx = _fake_ctx(1)
        early(ctx)
        assert not ctx.stop
        ctx = _fake_ctx(2)
        early(ctx)  # Δ/prev = 4% ≤ 5% → stop
        assert ctx.stop and early.stopped_at == 2

    def test_nonfinite_monitor_stops(self):
        early = EarlyStopReporter(monitor=lambda ctx: float("nan"), threshold=1.0, interval=1)
        ctx = _fake_ctx(3)
        early(ctx)
        assert ctx.stop and early.stopped and "non-finite" in early.reason

    def test_min_step_skips_warmup(self):
        early = EarlyStopReporter(
            monitor=lambda ctx: 1.0, threshold=1e-12, patience=1, interval=2, min_step=6
        )
        for step in range(1, 9):
            ctx = _fake_ctx(step)
            report_dispatch(ctx, [early])
            if ctx.stop:
                break
        # Fires at 2 and 4 are skipped (below min_step); 6 = baseline;
        # 8 → first compared change → stop.
        assert early.values == [(6, 1.0), (8, 1.0)]
        assert early.stopped_at == 8

    def test_diag_key_monitor(self):
        early = EarlyStopReporter("max_speed", threshold=1e-12, patience=1, interval=1)
        ctx = _fake_ctx(1, diag={"max_speed": 0.5})
        early(ctx)
        ctx = _fake_ctx(2, diag={"max_speed": 0.5})
        early(ctx)
        assert ctx.stop
        with pytest.raises(KeyError, match="max_speed"):
            early(_fake_ctx(3, diag={}))

    def test_no_stop_when_quantity_keeps_changing(self):
        early = EarlyStopReporter(
            monitor=lambda ctx: float(ctx.step), threshold=0.1, patience=1, interval=1
        )
        executor = _make_executor(reporters=[early])
        _, diags = executor.run(_make_f(), 20)
        assert not early.stopped and early.stopped_at is None
        assert len(diags) == 20


# ---------------------------------------------------------------------------
# 5. ThroughputReporter
# ---------------------------------------------------------------------------


class TestThroughput:
    def test_mlups_matches_hand_computed_value(self, monkeypatch):
        import tensorlbm.reporters as reporters_module

        clock = {"now": 0.0}
        monkeypatch.setattr(reporters_module, "perf_counter", lambda: clock["now"])
        reporter = ThroughputReporter(interval=1, num_cells=1000)
        ctx1 = _fake_ctx(1)
        reporter(ctx1)  # baseline at t=0
        clock["now"] = 2.0
        ctx2 = _fake_ctx(11)
        reporter(ctx2)
        # 10 steps × 1000 cells / 1e6 / 2 s = 0.005 MLUPS.
        assert len(reporter.records) == 1
        step, elapsed, mlups = reporter.records[0]
        assert step == 11 and elapsed == 2.0
        assert mlups == pytest.approx(0.005, rel=1e-12)
        assert reporter.last_mlups == pytest.approx(0.005)
        assert reporter.mean_mlups == pytest.approx(0.005)

    def test_real_executor_run_reasonable_mlups(self):
        reporter = ThroughputReporter(interval=10)
        executor = _make_executor()
        executor.run(_make_f(), 20, reporters=[reporter])
        assert len(reporter.records) == 1
        step, elapsed, mlups = reporter.records[0]
        assert step == 20
        # Identity against the recorded elapsed time (MLUPS convention:
        # lattice-site updates / 1e6 / second).
        assert mlups == pytest.approx(10 * N_CELLS / 1e6 / elapsed, rel=1e-12)
        # CPU sanity band on a small grid.
        assert 1e-3 < mlups < 1e9

    def test_first_fire_is_baseline_only(self):
        reporter = ThroughputReporter(interval=5)
        executor = _make_executor()
        executor.run(_make_f(), 5, reporters=[reporter])
        assert reporter.records == []
        assert reporter.last_mlups is None and reporter.mean_mlups is None


# ---------------------------------------------------------------------------
# 6. FieldSampleReporter end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog(tmp_path):
    from tensorlbm.data.catalog import FieldDataCatalog

    cat = FieldDataCatalog.open(tmp_path / "catalog.db")
    yield cat
    cat.close()


class TestFieldSampleReporter:
    def _reporter(self, tmp_path, catalog, **overrides) -> FieldSampleReporter:
        kwargs: dict = {
            "path": tmp_path / "run.h5",
            "run_id": "reporter-run",
            "case": "unit-case",
            "code_sha": CODE_SHA,
            "interval": 25,
            "catalog": catalog,
            "solid_mask": _make_mask(),
            "metadata": {"collision": "BGK", "re": 100.0, "tau": 0.6},
        }
        kwargs.update(overrides)
        return FieldSampleReporter(**kwargs)

    def test_end_to_end_products_queryable_from_catalog(self, tmp_path, catalog):
        reporter = self._reporter(tmp_path, catalog)
        executor = _make_executor(reporters=[reporter])
        executor.run(_make_f(), 50)

        assert reporter.exported_steps == [25, 50]
        assert reporter.product_ids == ["reporter-run:000025", "reporter-run:000050"]

        # Catalog-side discovery (the training-side entry point).
        found = catalog.find_assets_by_metadata("case", "unit-case", kind="field_product")
        assert sorted(asset.asset_id for asset in found) == reporter.product_ids
        assert catalog.get_asset("reporter-run:000025") is not None

        # Registered products reload as verified arrays.
        from tensorlbm.data.solver_export import load_product, load_product_arrays

        product = load_product(catalog, "reporter-run:000050")
        arrays = load_product_arrays(product)
        assert arrays["velocity"].shape == (NZ, NY, NX, 3)
        assert arrays["rho"].shape == (NZ, NY, NX)
        assert arrays["solid_mask"].shape == (NZ, NY, NX)
        assert arrays["velocity"].dtype == np.float32

    def test_snapshots_written_to_hdf5_with_attrs(self, tmp_path, catalog):
        from tensorlbm.data.solver_export import read_snapshot

        reporter = self._reporter(tmp_path, catalog)
        executor = _make_executor(reporters=[reporter])
        executor.run(_make_f(), 50)

        arrays, attrs = read_snapshot(reporter.path, 25)
        assert attrs["step"] == 25
        assert attrs["run_id"] == "reporter-run"
        assert attrs["case"] == "unit-case"
        assert attrs["collision"] == "BGK"
        assert {"rho", "ux", "uy", "uz", "solid_mask"} <= set(arrays)
        # Exported fields agree with the macroscopic of that step's f
        # (recomputed through the same executor the reporter used).
        assert np.isfinite(arrays["rho"]).all()
        assert abs(float(arrays["rho"].mean()) - 1.0) < 1e-3

    def test_save_only_mode_without_catalog(self, tmp_path):
        from tensorlbm.data.solver_export import read_snapshot

        reporter = self._reporter(tmp_path, None)
        executor = _make_executor(reporters=[reporter])
        executor.run(_make_f(), 25)
        assert reporter.product_ids == []
        arrays, _ = read_snapshot(reporter.path, 25)
        assert arrays["ux"].shape == (NZ, NY, NX)

    def test_field_subset_and_validation(self, tmp_path, catalog):
        reporter = self._reporter(tmp_path, catalog, fields=("ux", "uy"))
        executor = _make_executor(reporters=[reporter])
        executor.run(_make_f(), 25)
        assert reporter.product_ids  # ux+uy is the registration minimum
        with pytest.raises(ValueError, match="subset"):
            self._reporter(tmp_path, catalog, fields=("pressure",))

    def test_extra_fields_exported_as_auxiliary(self, tmp_path, catalog):
        from tensorlbm.data.solver_export import load_product, load_product_arrays

        def extra_fields(ctx: StepContext):
            _, ux, uy, uz = ctx.macroscopic()
            return {"u_mag": torch.sqrt(ux * ux + uy * uy + uz * uz)}

        reporter = self._reporter(tmp_path, catalog, extra_fields=extra_fields)
        executor = _make_executor(reporters=[reporter])
        executor.run(_make_f(), 25)
        arrays = load_product_arrays(load_product(catalog, reporter.product_ids[0]))
        assert arrays["u_mag"].shape == (NZ, NY, NX)


# ---------------------------------------------------------------------------
# 7. Triton fused solver hook (CUDA + Triton hosts only)
# ---------------------------------------------------------------------------


class TestTritonFusedRunHook:
    def test_run_with_reporters(self):
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from tensorlbm.triton_fused import TritonFusedSolver3D, is_available

        if not is_available():
            pytest.skip("Triton fused solver unavailable")
        solver = TritonFusedSolver3D(16, 16, 16, tau=0.6, device="cuda:0")
        rho0 = torch.ones(16, 16, 16, device="cuda")
        f0 = equilibrium3d(rho0, rho0 * 0.05, rho0 * 0.01, rho0 * 0.01, device="cuda")
        seen: list[int] = []
        reporter = CallbackReporter(lambda ctx: seen.append(ctx.step), interval=2)
        f_out, diags = solver.run(f0, 4, reporters=[reporter])
        assert seen == [2, 4]
        assert f_out.shape == f0.shape
        assert len(diags) == 4

    def test_run_without_reporters_matches_step_loop(self):
        pytest.importorskip("triton")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from tensorlbm.triton_fused import TritonFusedSolver3D, is_available

        if not is_available():
            pytest.skip("Triton fused solver unavailable")
        solver = TritonFusedSolver3D(16, 16, 16, tau=0.6, device="cuda:0")
        rho0 = torch.ones(16, 16, 16, device="cuda")
        f0 = equilibrium3d(rho0, rho0 * 0.05, rho0 * 0.01, rho0 * 0.01, device="cuda")
        f_run, diags = solver.run(f0, 3)
        f_ref = f0
        for _ in range(3):
            f_ref = solver.step(f_ref)
        assert torch.equal(f_run, f_ref)
        assert len(diags) == 3
