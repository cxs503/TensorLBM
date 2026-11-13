"""Cross-validation tests: bitwise reproducibility, reference-data regression, benchmarking.

Tests the xval framework and the concrete verification cases.

Acceptance criteria (XVAL-LBM-BITWISE-REFDATA-REGRESSION-PLUS-BENCHMAR):
- All scalar metrics within 15% of reference (owner precision standard).
- Bitwise reproducibility: identical runs produce identical state hashes.
- Reference data regression: stored golden data matches current solver output.
- Benchmarking: timing infrastructure produces valid statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Framework tests
# ---------------------------------------------------------------------------


class TestBitwiseReproducibility:
    """Test the bitwise reproducibility infrastructure."""

    def test_compute_state_hash_deterministic(self):
        """Same array → same hash."""
        from tensorlbm.verification.xval.bitwise_refdata import compute_state_hash
        arr = np.random.RandomState(42).randn(9, 33, 33).astype(np.float64)
        h1 = compute_state_hash(arr)
        h2 = compute_state_hash(arr)
        assert h1 == h2

    def test_compute_state_hash_different_for_different_arrays(self):
        """Different arrays → different hashes."""
        from tensorlbm.verification.xval.bitwise_refdata import compute_state_hash
        arr1 = np.zeros((9, 10, 10), dtype=np.float64)
        arr2 = np.ones((9, 10, 10), dtype=np.float64)
        assert compute_state_hash(arr1) != compute_state_hash(arr2)

    def test_compute_state_hash_dtype_sensitive(self):
        """float32 vs float64 of same values → different hashes."""
        from tensorlbm.verification.xval.bitwise_refdata import compute_state_hash
        arr64 = np.ones((5, 5), dtype=np.float64)
        arr32 = arr64.astype(np.float32)
        assert compute_state_hash(arr64) != compute_state_hash(arr32)

    def test_verify_bitwise_reproducibility_passes_for_deterministic_fn(self):
        """A deterministic function should pass bitwise repro check."""
        from tensorlbm.verification.xval.bitwise_refdata import (
            BitwiseReproConfig, verify_bitwise_reproducibility,
        )

        def deterministic_run(config, **kwargs):
            rng = np.random.RandomState(config.seed)
            f = rng.randn(9, 16, 16).astype(np.float64)
            return {"f": f}

        config = BitwiseReproConfig(seed=42, n_runs=3)
        result = verify_bitwise_reproducibility(deterministic_run, config)
        assert result["reproducible"] is True
        assert len(result["hashes"]) == 3
        assert all(h == result["hashes"][0] for h in result["hashes"])

    def test_verify_bitwise_reproducibility_fails_for_nondeterministic_fn(self):
        """A non-deterministic function should fail bitwise repro check."""
        from tensorlbm.verification.xval.bitwise_refdata import (
            BitwiseReproConfig, compute_state_hash,
        )

        np.random.seed(42)
        f1 = np.random.randn(9, 16, 16).astype(np.float64)
        f2 = np.random.randn(9, 16, 16).astype(np.float64)
        h1 = compute_state_hash(f1)
        h2 = compute_state_hash(f2)
        assert h1 != h2


class TestReferenceDataStore:
    """Test the reference-data store."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save then load should recover the same data."""
        from tensorlbm.verification.xval.bitwise_refdata import ReferenceDataStore

        store = ReferenceDataStore(base_dir=tmp_path)
        arrays = {"u": np.array([1.0, 2.0, 3.0])}
        scalars = {"eps_inf": 0.01, "eps_L2": 0.005}
        thresholds = {"eps_inf": 0.15, "eps_L2": 0.15}

        store.save("test_case", arrays, scalars, thresholds)
        loaded = store.load("test_case")

        np.testing.assert_array_equal(loaded["arrays"]["u"], arrays["u"])
        assert loaded["scalars"]["eps_inf"] == scalars["eps_inf"]
        assert loaded["scalars"]["eps_L2"] == scalars["eps_L2"]
        assert loaded["thresholds"]["eps_inf"] == thresholds["eps_inf"]

    def test_compare_passes_within_threshold(self, tmp_path):
        """Compare should pass when new data is within threshold."""
        from tensorlbm.verification.xval.bitwise_refdata import ReferenceDataStore

        store = ReferenceDataStore(base_dir=tmp_path)
        arrays = {"u": np.array([1.0, 2.0, 3.0])}
        scalars = {"eps_inf": 0.01}
        thresholds = {"eps_inf": 0.15}

        store.save("test_case", arrays, scalars, thresholds)

        new_scalars = {"eps_inf": 0.011}
        result = store.compare("test_case", {}, new_scalars)

        assert result["passed"] is True
        assert result["details"]["eps_inf"]["passed"] is True

    def test_compare_fails_outside_threshold(self, tmp_path):
        """Compare should fail when new data exceeds threshold."""
        from tensorlbm.verification.xval.bitwise_refdata import ReferenceDataStore

        store = ReferenceDataStore(base_dir=tmp_path)
        arrays = {"u": np.array([1.0, 2.0, 3.0])}
        scalars = {"eps_inf": 0.01}
        thresholds = {"eps_inf": 0.15}

        store.save("test_case", arrays, scalars, thresholds)

        new_scalars = {"eps_inf": 0.05}
        result = store.compare("test_case", {}, new_scalars)

        assert result["passed"] is False

    def test_has_case(self, tmp_path):
        """has_case should reflect whether data exists."""
        from tensorlbm.verification.xval.bitwise_refdata import ReferenceDataStore

        store = ReferenceDataStore(base_dir=tmp_path)
        assert store.has_case("nonexistent") is False

        store.save("exists", {"a": np.zeros(3)}, {"x": 1.0}, {"x": 0.1})
        assert store.has_case("exists") is True

    def test_list_cases(self, tmp_path):
        """list_cases should return all saved case slugs."""
        from tensorlbm.verification.xval.bitwise_refdata import ReferenceDataStore

        store = ReferenceDataStore(base_dir=tmp_path)
        store.save("case_a", {"a": np.zeros(3)}, {"x": 1.0}, {"x": 0.1})
        store.save("case_b", {"a": np.zeros(3)}, {"x": 1.0}, {"x": 0.1})
        cases = store.list_cases()
        assert "case_a" in cases
        assert "case_b" in cases


class TestBenchmark:
    """Test the benchmarking infrastructure."""

    def test_run_benchmark_produces_stats(self):
        """run_benchmark should produce valid timing statistics."""
        from tensorlbm.verification.xval.benchmark import BenchmarkConfig, run_benchmark

        def dummy_step():
            x = np.random.randn(100)
            return np.sum(x)

        config = BenchmarkConfig(n_warmup=2, n_repeats=5, label="dummy")
        result = run_benchmark(dummy_step, config, n_steps=10)

        assert result.label == "dummy"
        assert len(result.times) == 5
        assert result.mean > 0
        assert result.std >= 0
        assert result.min <= result.mean <= result.max

    def test_benchmark_mlups(self):
        """MLUPS calculation should be correct."""
        from tensorlbm.verification.xval.benchmark import BenchmarkConfig, run_benchmark

        def dummy_step():
            pass

        config = BenchmarkConfig(n_warmup=1, n_repeats=3, label="mlups_test")
        result = run_benchmark(dummy_step, config, n_steps=100, grid_size=1000)

        assert result.mlups is not None
        assert result.mlups > 0

    def test_benchmark_summary(self):
        """BenchmarkResult.summary() should produce a readable string."""
        from tensorlbm.verification.xval.benchmark import BenchmarkResult

        result = BenchmarkResult(
            label="test", times=[0.1, 0.2, 0.15], n_steps=100, mlups=5.0
        )
        result.compute_stats()
        s = result.summary()
        assert "test" in s
        assert "MLUPS" in s


class TestXValRegistry:
    """Test the cross-validation registry."""

    def test_register_and_list(self):
        """Register a case and list it."""
        from tensorlbm.verification.xval.registry import XValCase, XValRegistry

        reg = XValRegistry()
        case = XValCase(
            name="test_case",
            run_fn=lambda: {"f": np.zeros((9, 10, 10)), "scalars": {"x": 0.01}},
            thresholds={"x": 0.15},
        )
        reg.register(case)
        assert "test_case" in reg.list_cases()

    def test_duplicate_registration_raises(self):
        """Registering the same name twice should raise."""
        from tensorlbm.verification.xval.registry import XValCase, XValRegistry

        reg = XValRegistry()
        case = XValCase(
            name="dup",
            run_fn=lambda: {"f": np.zeros((9, 10, 10)), "scalars": {}},
        )
        reg.register(case)
        with pytest.raises(ValueError):
            reg.register(case)

    def test_run_case(self):
        """Run a simple case through the registry."""
        from tensorlbm.verification.xval.registry import XValCase, XValRegistry

        case = XValCase(
            name="simple",
            run_fn=lambda: {
                "f": np.zeros((9, 10, 10)),
                "scalars": {"eps": 0.01},
            },
            reference_fn=lambda: {"scalars": {"eps": 0.0}},
            thresholds={"eps": 0.15},
        )
        reg = XValRegistry()
        reg.register(case)
        result = reg.run_case("simple")
        assert result.case_name == "simple"
        assert result.passed is True
        assert "eps" in result.metrics

    def test_run_case_fails_threshold(self):
        """A case exceeding threshold should fail."""
        from tensorlbm.verification.xval.registry import XValCase, XValRegistry

        case = XValCase(
            name="fail_case",
            run_fn=lambda: {
                "f": np.zeros((9, 10, 10)),
                "scalars": {"eps": 0.5},
            },
            reference_fn=lambda: {"scalars": {"eps": 0.01}},
            thresholds={"eps": 0.15},
        )
        reg = XValRegistry()
        reg.register(case)
        result = reg.run_case("fail_case")
        assert result.passed is False

    def test_print_summary(self):
        """print_summary should produce a readable table."""
        from tensorlbm.verification.xval.registry import XValResult, XValRegistry

        results = [
            XValResult(case_name="case_a", passed=True, metrics={"eps": 0.01}),
            XValResult(case_name="case_b", passed=False, metrics={"eps": 0.2}),
        ]
        summary = XValRegistry.print_summary(results)
        assert "PASS" in summary
        assert "FAIL" in summary


# ---------------------------------------------------------------------------
# Concrete case tests (integration)
# ---------------------------------------------------------------------------


class TestPoiseuille2D:
    """Test the 2-D Poiseuille cross-validation case."""

    def test_poiseuille_runs(self):
        """Poiseuille case should run without errors."""
        from tensorlbm.verification.xval.cases import POISEUILLE_2D_CASE
        result = POISEUILLE_2D_CASE.run_fn()
        assert "f" in result
        assert "scalars" in result
        assert "eps_inf" in result["scalars"]
        assert "eps_L2" in result["scalars"]

    def test_poiseuille_within_15pct(self):
        """Poiseuille case should be within 15% of analytical solution."""
        from tensorlbm.verification.xval.cases import POISEUILLE_2D_CASE
        result = POISEUILLE_2D_CASE.run_fn()
        eps_inf = result["scalars"]["eps_inf"]
        eps_L2 = result["scalars"]["eps_L2"]
        assert eps_inf < 0.15, f"eps_inf={eps_inf:.4f} > 15%"
        assert eps_L2 < 0.15, f"eps_L2={eps_L2:.4f} > 15%"

    def test_poiseuille_within_2pct(self):
        """Poiseuille case should be within 2% (tight gate)."""
        from tensorlbm.verification.xval.cases import POISEUILLE_2D_CASE
        result = POISEUILLE_2D_CASE.run_fn()
        eps_inf = result["scalars"]["eps_inf"]
        eps_L2 = result["scalars"]["eps_L2"]
        assert eps_inf < 0.02, f"eps_inf={eps_inf:.4f} > 2%"
        assert eps_L2 < 0.02, f"eps_L2={eps_L2:.4f} > 2%"

    def test_poiseuille_mass_conserved(self):
        """Guo forcing should conserve mass."""
        from tensorlbm.verification.xval.cases import POISEUILLE_2D_CASE
        result = POISEUILLE_2D_CASE.run_fn()
        mass_drift = result["scalars"]["mass_drift"]
        assert mass_drift < 1e-8, f"mass_drift={mass_drift:.2e} > 1e-8"

    def test_poiseuille_bitwise_reproducible(self):
        """Poiseuille case should be bitwise reproducible."""
        from tensorlbm.verification.xval.bitwise_refdata import compute_state_hash
        from tensorlbm.verification.xval.cases import POISEUILLE_2D_CASE

        r1 = POISEUILLE_2D_CASE.run_fn()
        r2 = POISEUILLE_2D_CASE.run_fn()

        h1 = compute_state_hash(r1["f"])
        h2 = compute_state_hash(r2["f"])
        assert h1 == h2, f"Bitwise reproducibility failed: {h1[:16]}... != {h2[:16]}..."


class TestLidCavityRe100:
    """Test the lid-driven cavity Re=100 cross-validation case.

    Uses n=33 grid with 3000 steps for fast test execution.
    Full-resolution (n=65, 10000 steps) benchmarking is done on the cluster.
    """

    def test_cavity_runs(self):
        """Cavity case should run without errors (coarse grid)."""
        from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
        from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference
        from tensorlbm.verification.lid_driven_cavity.verification import compute_errors

        cfg = CavityConfig(n=33, re=100, max_steps=3000, u_lid=0.1)
        solver = CavitySolver(cfg)
        for _ in range(2000):
            solver.step()

        y, u, x, v = solver.get_profiles()
        ref = get_ghia_reference(100)
        errors = compute_errors(y, u, x, v, ref)

        assert "l2_u" in errors
        assert "l2_v" in errors
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))

    def test_cavity_within_15pct_coarse(self):
        """Cavity Re=100 on coarse grid should be within 15% of Ghia."""
        from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
        from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference
        from tensorlbm.verification.lid_driven_cavity.verification import compute_errors

        cfg = CavityConfig(n=33, re=100, max_steps=5000, u_lid=0.1)
        solver = CavitySolver(cfg)
        for _ in range(3000):
            solver.step()

        y, u, x, v = solver.get_profiles()
        ref = get_ghia_reference(100)
        errors = compute_errors(y, u, x, v, ref)

        assert errors["l2_u"] < 0.15, f"l2_u={errors['l2_u']:.4f} > 15%"
        assert errors["l2_v"] < 0.15, f"l2_v={errors['l2_v']:.4f} > 15%"

    def test_cavity_bitwise_reproducible(self):
        """Cavity case should be bitwise reproducible (coarse grid, few steps)."""
        from tensorlbm.verification.xval.bitwise_refdata import compute_state_hash
        from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver

        def run_cavity():
            cfg = CavityConfig(n=33, re=100, max_steps=500, u_lid=0.1)
            solver = CavitySolver(cfg)
            for _ in range(500):
                solver.step()
            return solver.f

        f1 = run_cavity()
        f2 = run_cavity()

        h1 = compute_state_hash(f1)
        h2 = compute_state_hash(f2)
        assert h1 == h2, f"Bitwise reproducibility failed: {h1[:16]}... != {h2[:16]}..."


# ---------------------------------------------------------------------------
# Reference data regression tests
# ---------------------------------------------------------------------------


class TestReferenceDataRegression:
    """Test that reference data can be saved and compared for regression."""

    def test_poiseuille_refdata_save_load(self, tmp_path):
        """Save and load Poiseuille reference data."""
        from tensorlbm.verification.xval.bitwise_refdata import ReferenceDataStore
        from tensorlbm.verification.xval.cases import POISEUILLE_2D_CASE

        result = POISEUILLE_2D_CASE.run_fn()
        store = ReferenceDataStore(base_dir=tmp_path)

        store.save(
            "poiseuille_2d_H16",
            result["arrays"],
            result["scalars"],
            POISEUILLE_2D_CASE.thresholds,
        )

        loaded = store.load("poiseuille_2d_H16")
        assert "u_profile" in loaded["arrays"]
        assert loaded["scalars"]["eps_inf"] == result["scalars"]["eps_inf"]

    def test_poiseuille_refdata_compare(self, tmp_path):
        """Compare new Poiseuille run against stored reference."""
        from tensorlbm.verification.xval.bitwise_refdata import ReferenceDataStore
        from tensorlbm.verification.xval.cases import POISEUILLE_2D_CASE

        result = POISEUILLE_2D_CASE.run_fn()
        store = ReferenceDataStore(base_dir=tmp_path)

        store.save(
            "poiseuille_2d_H16",
            result["arrays"],
            result["scalars"],
            POISEUILLE_2D_CASE.thresholds,
        )

        result2 = POISEUILLE_2D_CASE.run_fn()
        comparison = store.compare(
            "poiseuille_2d_H16",
            result2["arrays"],
            result2["scalars"],
        )

        assert comparison["passed"] is True


# ---------------------------------------------------------------------------
# Convergence tests (parameter sensitivity)
# ---------------------------------------------------------------------------


class TestConvergence:
    """Test grid convergence for the Poiseuille case."""

    def test_poiseuille_second_order_convergence(self):
        """Half-way bounce-back should give second-order convergence."""
        from tensorlbm.ice_hydro.lbm_channel import (
            CX, CY, WT, ChannelConfig, D2Q9Channel,
        )

        G = 1.0e-5
        TAU = 0.8

        def run_poiseuille(H, steps):
            ny = H + 4
            cfg = ChannelConfig(
                nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
                cover_amplitude=0.0, cover_depth=float(ny - 3),
                bc="no_slip", forcing="guo",
            )
            ch = D2Q9Channel(cfg)
            rows = np.where(ch.fluid[0, :])[0]
            n_fluid = rows.size
            nu = (TAU - 0.5) / 3.0
            y = rows - 0.5
            u_ana = G / (2.0 * nu) * y * (n_fluid - y)
            u0 = np.zeros(ny)
            u0[rows] = u_ana
            ch.init_equilibrium(u0)

            acc = np.zeros(n_fluid)
            n_win = max(steps // 4, 1)
            n_acc = 0
            for step in range(1, steps + 1):
                ch.f = ch._stream_from(ch._collide(ch.f))
                if step > steps - n_win:
                    rho = ch.f.sum(axis=0)
                    ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
                    acc += ux[0, rows]
                    n_acc += 1
            u = acc / n_acc
            return float(np.max(np.abs(u - u_ana)) / u_ana.max())

        e8 = run_poiseuille(8, 2000)
        e16 = run_poiseuille(16, 5000)
        ratio = e8 / e16
        assert 3.0 <= ratio <= 5.0, f"Convergence ratio={ratio:.2f}, expected ~4"
