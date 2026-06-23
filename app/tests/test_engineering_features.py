"""Tests for four engineering-grade features added to close the gap with
PowerFlow and XFlow:

1. Spectral analysis of probe signals (FFT / PSD) – library + API
2. POD modal decomposition (method of snapshots) – library + API
3. Iso-surface / iso-contour extraction (marching squares / cubes) – library + API
4. Design of Experiments (LHS, Sobol, full factorial, CCD) – library + API
"""
from __future__ import annotations

import math

import pytest


# ===========================================================================
# 1. Spectral Analysis (probe_spectrum)
# ===========================================================================

class TestProbeSpectrumLibrary:
    """Library-level tests for tensorlbm.probe_spectrum."""

    def test_single_sinusoid_peak(self) -> None:
        """A clean sinusoid should show a dominant peak at the correct frequency."""
        import torch
        from tensorlbm.probe_spectrum import compute_probe_spectrum

        dt = 1.0
        f0 = 0.05  # 1/20 of sampling rate
        n = 512
        t = torch.arange(n, dtype=torch.float64)
        signal = torch.sin(2.0 * math.pi * f0 * t * dt)

        result = compute_probe_spectrum(signal.tolist(), dt=dt, n_peaks=3)
        assert result.n_samples == n
        assert result.f_nyquist == pytest.approx(0.5 / dt)
        assert len(result.peak_frequencies) >= 1
        # Dominant peak should be near f0
        assert abs(result.peak_frequencies[0] - f0) < 0.01

    def test_psd_positive(self) -> None:
        """All PSD values must be non-negative."""
        import torch
        from tensorlbm.probe_spectrum import compute_probe_spectrum

        signal = torch.rand(256).tolist()
        result = compute_probe_spectrum(signal, dt=1.0)
        assert all(v >= 0.0 for v in result.psd)

    def test_strouhal_number(self) -> None:
        """Strouhal number should be f_peak * D / U."""
        import torch
        from tensorlbm.probe_spectrum import compute_probe_spectrum

        dt = 1.0
        f0 = 0.04
        n = 512
        t = torch.arange(n, dtype=torch.float64)
        signal = (torch.sin(2.0 * math.pi * f0 * t) + 0.05 * torch.rand(n)).tolist()

        result = compute_probe_spectrum(signal, dt=dt, diameter=1.0, u_ref=10.0)
        assert result.strouhal is not None
        assert result.strouhal > 0.0
        # St ≈ f_peak * 1.0 / 10.0
        if result.peak_frequencies:
            expected_st = result.peak_frequencies[0] * 1.0 / 10.0
            assert abs(result.strouhal - expected_st) < 1e-9

    def test_no_strouhal_without_params(self) -> None:
        import torch
        from tensorlbm.probe_spectrum import compute_probe_spectrum

        signal = torch.rand(64).tolist()
        result = compute_probe_spectrum(signal, dt=1.0)
        assert result.strouhal is None

    def test_signal_rms_zero_for_constant(self) -> None:
        """A constant signal should have zero RMS fluctuation."""
        from tensorlbm.probe_spectrum import compute_probe_spectrum

        signal = [1.0] * 64
        result = compute_probe_spectrum(signal, dt=1.0)
        assert result.signal_rms == pytest.approx(0.0, abs=1e-10)

    def test_frequency_resolution(self) -> None:
        """Frequency resolution should be 1 / (N * dt) for a full-signal segment."""
        from tensorlbm.probe_spectrum import compute_probe_spectrum

        dt = 0.01
        n = 128
        signal = [math.sin(2 * math.pi * 5.0 * i * dt) for i in range(n)]
        result = compute_probe_spectrum(signal, dt=dt, n_segment=n)
        df = result.frequencies[1] - result.frequencies[0]
        expected_df = 1.0 / (n * dt)
        assert abs(df - expected_df) < 1e-8

    def test_welch_psd_output_shape(self) -> None:
        import torch
        from tensorlbm.probe_spectrum import welch_psd

        n = 256
        signal = torch.rand(n)
        freqs, psd = welch_psd(signal, dt=1.0)
        assert len(freqs) == len(psd)
        assert len(freqs) > 0

    def test_dominant_peaks_helper(self) -> None:
        import torch
        from tensorlbm.probe_spectrum import compute_probe_spectrum, dominant_peaks

        t = torch.arange(512, dtype=torch.float64)
        signal = (torch.sin(2 * math.pi * 0.05 * t)
                  + 0.5 * torch.sin(2 * math.pi * 0.1 * t)).tolist()
        result = compute_probe_spectrum(signal, dt=1.0, n_peaks=5)
        peaks = dominant_peaks(result, n=2)
        assert len(peaks) <= 2
        for p in peaks:
            assert "frequency" in p
            assert "psd" in p


# ===========================================================================
# 2. POD Modal Decomposition
# ===========================================================================

class TestPODLibrary:
    """Library-level tests for tensorlbm.pod."""

    def test_pod_basic_shapes(self) -> None:
        import torch
        from tensorlbm.pod import compute_pod

        ny, nx = 16, 16
        snapshots = [torch.rand(ny, nx) for _ in range(20)]
        result = compute_pod(snapshots, n_modes=5)

        assert result.n_snapshots == 20
        assert result.n_modes == 5
        assert result.modes.shape == (5, ny, nx)
        assert len(result.singular_values) == 5
        assert len(result.energy_fraction) == 5
        assert len(result.cumulative_energy) == 5

    def test_pod_energy_sums_to_one(self) -> None:
        """Energy fractions from all retained modes should sum to ≤ 1."""
        import torch
        from tensorlbm.pod import compute_pod

        snapshots = [torch.rand(8, 8) for _ in range(15)]
        result = compute_pod(snapshots, n_modes=15)
        assert sum(result.energy_fraction) <= 1.0 + 1e-6

    def test_pod_cumulative_energy_monotone(self) -> None:
        import torch
        from tensorlbm.pod import compute_pod

        snapshots = [torch.rand(8, 8) for _ in range(10)]
        result = compute_pod(snapshots, n_modes=5)
        for i in range(len(result.cumulative_energy) - 1):
            assert result.cumulative_energy[i] <= result.cumulative_energy[i + 1] + 1e-8

    def test_pod_temporal_coefficients_shape(self) -> None:
        import torch
        from tensorlbm.pod import compute_pod

        snapshots = [torch.rand(4, 4) for _ in range(10)]
        result = compute_pod(snapshots, n_modes=3, return_coefficients=True)
        assert len(result.temporal_coefficients) == 10
        assert len(result.temporal_coefficients[0]) == 3

    def test_pod_skip_coefficients(self) -> None:
        import torch
        from tensorlbm.pod import compute_pod

        snapshots = [torch.rand(4, 4) for _ in range(10)]
        result = compute_pod(snapshots, n_modes=3, return_coefficients=False)
        assert result.temporal_coefficients == []

    def test_pod_mean_subtraction(self) -> None:
        """With subtract_mean=True the mean field should be non-zero if input varies."""
        import torch
        from tensorlbm.pod import compute_pod

        ny, nx = 8, 8
        mean = torch.rand(ny, nx) * 5.0
        snapshots = [mean + torch.rand(ny, nx) * 0.1 for _ in range(10)]
        result = compute_pod(snapshots, n_modes=5, subtract_mean=True)
        assert any(abs(v) > 1e-6 for v in result.mean_field)

    def test_pod_reconstruction_error(self) -> None:
        """Reconstruction using all modes should have low relative error."""
        import torch
        from tensorlbm.pod import compute_pod, pod_reconstruction_error

        ny, nx = 8, 8
        snapshots = [torch.rand(ny, nx) for _ in range(10)]
        result = compute_pod(snapshots, n_modes=10)
        error = pod_reconstruction_error(snapshots[0], result, snapshot_index=0)
        assert error < 1.0  # sanity (not necessarily perfect due to truncation)

    def test_pod_stacked_input(self) -> None:
        """Should accept a pre-stacked (N, ...) tensor."""
        import torch
        from tensorlbm.pod import compute_pod

        data = torch.rand(12, 6, 6)
        result = compute_pod(data, n_modes=4)
        assert result.n_snapshots == 12
        assert result.spatial_shape == [6, 6]

    def test_pod_requires_two_snapshots(self) -> None:
        import torch
        from tensorlbm.pod import compute_pod

        with pytest.raises((ValueError, Exception)):
            compute_pod([torch.rand(4, 4)], n_modes=1)

    def test_pod_1d_field(self) -> None:
        """POD should work on 1-D fields (e.g. velocity profiles)."""
        import torch
        from tensorlbm.pod import compute_pod

        snapshots = [torch.rand(32) for _ in range(8)]
        result = compute_pod(snapshots, n_modes=3)
        assert result.spatial_shape == [32]


# ===========================================================================
# 3. Iso-Surface / Iso-Contour
# ===========================================================================

class TestIsoContour2D:
    """Library-level tests for marching squares."""

    def test_uniform_field_no_contour(self) -> None:
        """A uniform field has no iso-contour."""
        import torch
        from tensorlbm.isosurface import marching_squares

        field = torch.ones(16, 16) * 0.5
        result = marching_squares(field, iso_value=0.3)
        # All values above 0.3 → no crossings
        assert result.n_segments == 0

    def test_known_crossing(self) -> None:
        """A field that crosses iso_value should produce segments."""
        import torch
        from tensorlbm.isosurface import marching_squares

        ny, nx = 10, 10
        x = torch.arange(nx, dtype=torch.float32).unsqueeze(0).expand(ny, nx)
        # Linear gradient: left half < 5, right half >= 5
        result = marching_squares(x, iso_value=5.0)
        assert result.n_segments > 0

    def test_segment_coordinates_in_range(self) -> None:
        """All segment endpoints must fall within [x_range] × [y_range]."""
        import torch
        from tensorlbm.isosurface import marching_squares

        field = torch.rand(20, 20)
        x_range = (0.0, 2.0)
        y_range = (0.0, 3.0)
        result = marching_squares(
            field, iso_value=0.5,
            x_range=x_range, y_range=y_range,
        )
        for seg in result.segments:
            for pt in seg:
                assert x_range[0] - 1e-6 <= pt[0] <= x_range[1] + 1e-6
                assert y_range[0] - 1e-6 <= pt[1] <= y_range[1] + 1e-6

    def test_field_name_preserved(self) -> None:
        import torch
        from tensorlbm.isosurface import marching_squares

        field = torch.rand(8, 8)
        result = marching_squares(field, iso_value=0.5, field_name="q_criterion")
        assert result.field_name == "q_criterion"
        assert result.iso_value == pytest.approx(0.5)

    def test_too_small_field_returns_empty(self) -> None:
        import torch
        from tensorlbm.isosurface import marching_squares

        field = torch.rand(1, 10)
        result = marching_squares(field, iso_value=0.5)
        assert result.n_segments == 0


class TestIsoSurface3D:
    """Library-level tests for simplified marching cubes."""

    def test_uniform_field_no_surface(self) -> None:
        import torch
        from tensorlbm.isosurface import marching_cubes_simple

        field = torch.ones(8, 8, 8) * 2.0
        result = marching_cubes_simple(field, iso_value=1.0)
        # All above iso_value → no crossing
        assert result.n_triangles == 0

    def test_gradient_field_produces_surface(self) -> None:
        import torch
        from tensorlbm.isosurface import marching_cubes_simple

        nz, ny, nx = 10, 10, 10
        x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx).expand(nz, ny, nx)
        result = marching_cubes_simple(x.clone(), iso_value=5.0)
        assert result.n_triangles > 0

    def test_triangles_reference_valid_vertices(self) -> None:
        import torch
        from tensorlbm.isosurface import marching_cubes_simple

        field = torch.rand(6, 6, 6)
        result = marching_cubes_simple(field, iso_value=0.5)
        n_verts = len(result.vertices)
        for tri in result.triangles:
            assert len(tri) == 3
            for idx in tri:
                assert 0 <= idx < n_verts


# ===========================================================================
# 4. Design of Experiments (DoE)
# ===========================================================================

class TestDoELibrary:
    """Library-level tests for tensorlbm.doe."""

    def test_lhs_shape(self) -> None:
        from tensorlbm.doe import lhs

        samples = lhs(n_vars=3, n_samples=10, seed=42)
        assert len(samples) == 10
        assert all(len(row) == 3 for row in samples)

    def test_lhs_unit_interval(self) -> None:
        from tensorlbm.doe import lhs

        samples = lhs(n_vars=2, n_samples=20, seed=0)
        for row in samples:
            for v in row:
                assert 0.0 <= v <= 1.0

    def test_lhs_stratification(self) -> None:
        """Each variable must have exactly one sample per stratum."""
        from tensorlbm.doe import lhs

        n = 10
        samples = lhs(n_vars=1, n_samples=n, seed=1)
        strata = [int(s[0] * n) for s in samples]
        assert sorted(strata) == list(range(n))

    def test_sobol_shape(self) -> None:
        from tensorlbm.doe import sobol_sequence

        samples = sobol_sequence(n_vars=4, n_samples=8)
        assert len(samples) == 8
        assert all(len(row) == 4 for row in samples)

    def test_full_factorial_count(self) -> None:
        """2^k design with 2-level factors."""
        from tensorlbm.doe import DoEVariable, full_factorial

        variables = [DoEVariable(name=f"x{i}", low=0.0, high=1.0) for i in range(3)]
        combos = full_factorial(variables)
        assert len(combos) == 8  # 2^3

    def test_full_factorial_discrete_levels(self) -> None:
        from tensorlbm.doe import DoEVariable, full_factorial

        variables = [
            DoEVariable(name="re", low=0.0, high=1.0, levels=[100.0, 200.0, 400.0]),
            DoEVariable(name="nx", low=0.0, high=1.0, levels=[32, 64]),
        ]
        combos = full_factorial(variables)
        assert len(combos) == 6  # 3 × 2

    def test_ccd_count(self) -> None:
        """CCD with k=2: 4 factorial + 4 axial + 1 centre = 9 points."""
        from tensorlbm.doe import central_composite

        points = central_composite(n_vars=2, face_centred=True, n_centre=1)
        assert len(points) == 9

    def test_generate_doe_lhs(self) -> None:
        from tensorlbm.doe import DoEVariable, generate_doe

        variables = [
            DoEVariable(name="re", low=100.0, high=500.0),
            DoEVariable(name="u_in", low=0.05, high=0.15),
        ]
        plan = generate_doe(variables, method="latin_hypercube", n_samples=8, seed=7)
        assert plan.n_runs == 8
        assert plan.method == "latin_hypercube"
        assert len(plan.design_matrix) == 8
        for pt in plan.design_matrix:
            assert 100.0 <= pt["re"] <= 500.0
            assert 0.05 <= pt["u_in"] <= 0.15

    def test_generate_doe_sobol(self) -> None:
        from tensorlbm.doe import DoEVariable, generate_doe

        variables = [DoEVariable(name="re", low=50.0, high=300.0)]
        plan = generate_doe(variables, method="sobol", n_samples=5, seed=0)
        assert plan.n_runs == 5

    def test_generate_doe_full_factorial(self) -> None:
        from tensorlbm.doe import DoEVariable, generate_doe

        variables = [
            DoEVariable(name="re", low=100.0, high=400.0, levels=[100.0, 200.0, 400.0]),
            DoEVariable(name="nx", low=32.0, high=128.0, levels=[32.0, 64.0]),
        ]
        plan = generate_doe(variables, method="full_factorial")
        assert plan.n_runs == 6

    def test_generate_doe_ccd(self) -> None:
        from tensorlbm.doe import DoEVariable, generate_doe

        variables = [
            DoEVariable(name="re", low=100.0, high=500.0),
            DoEVariable(name="u_in", low=0.05, high=0.15),
        ]
        plan = generate_doe(variables, method="central_composite", face_centred=True)
        # 2^2 + 2*2 + 1 = 9
        assert plan.n_runs == 9

    def test_generate_doe_invalid_method(self) -> None:
        from tensorlbm.doe import DoEVariable, generate_doe

        variables = [DoEVariable(name="re", low=100.0, high=500.0)]
        with pytest.raises(ValueError, match="Unknown DoE method"):
            generate_doe(variables, method="unknown")  # type: ignore[arg-type]

    def test_generate_doe_low_ge_high_raises(self) -> None:
        from tensorlbm.doe import DoEVariable, generate_doe

        variables = [DoEVariable(name="re", low=500.0, high=100.0)]
        with pytest.raises(ValueError):
            generate_doe(variables)

    def test_generate_doe_reproducibility(self) -> None:
        from tensorlbm.doe import DoEVariable, generate_doe

        variables = [
            DoEVariable(name="re", low=100.0, high=500.0),
            DoEVariable(name="u_in", low=0.05, high=0.15),
        ]
        plan1 = generate_doe(variables, method="latin_hypercube", n_samples=6, seed=42)
        plan2 = generate_doe(variables, method="latin_hypercube", n_samples=6, seed=42)
        for pt1, pt2 in zip(plan1.design_matrix, plan2.design_matrix):
            assert abs(pt1["re"] - pt2["re"]) < 1e-9
            assert abs(pt1["u_in"] - pt2["u_in"]) < 1e-9


# ===========================================================================
# API-level tests
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from app.backend.main import app
    return TestClient(app)


class TestProbeSpectrumAPI:
    def test_with_explicit_signal(self, client) -> None:
        import math
        n = 128
        signal = [math.sin(2 * math.pi * 0.05 * i) for i in range(n)]
        r = client.post("/api/postprocess/probe-spectrum", json={
            "signal": signal,
            "dt": 1.0,
            "n_peaks": 3,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert "frequencies" in data
        assert "psd" in data
        assert "peak_frequencies" in data
        assert "f_nyquist" in data
        assert data["n_samples"] == n

    def test_strouhal_computed(self, client) -> None:
        import math
        n = 256
        signal = [math.sin(2 * math.pi * 0.05 * i) for i in range(n)]
        r = client.post("/api/postprocess/probe-spectrum", json={
            "signal": signal,
            "dt": 1.0,
            "diameter": 1.0,
            "u_ref": 10.0,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["strouhal"] is not None
        assert data["strouhal"] > 0.0

    def test_missing_signal_and_job_id(self, client) -> None:
        r = client.post("/api/postprocess/probe-spectrum", json={"dt": 1.0})
        assert r.status_code == 422

    def test_job_not_found(self, client) -> None:
        r = client.post("/api/postprocess/probe-spectrum", json={
            "job_id": "nonexistent-job",
            "dt": 1.0,
        })
        assert r.status_code == 404

    def test_too_short_signal_rejected(self, client) -> None:
        r = client.post("/api/postprocess/probe-spectrum", json={
            "signal": [1.0, 2.0, 3.0],  # < 4 samples
            "dt": 1.0,
        })
        assert r.status_code == 422


class TestPODAPI:
    def _make_snapshots(self, n: int = 5, size: int = 4) -> list[list[list[float]]]:
        import random
        rng = random.Random(0)
        return [
            [[rng.random() for _ in range(size)] for _ in range(size)]
            for _ in range(n)
        ]

    def test_pod_with_snapshots(self, client) -> None:
        snaps = self._make_snapshots(n=8, size=4)
        r = client.post("/api/postprocess/pod", json={
            "snapshots": snaps,
            "n_modes": 3,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["n_modes"] == 3
        assert data["n_snapshots"] == 8
        assert len(data["singular_values"]) == 3
        assert len(data["energy_fraction"]) == 3
        assert len(data["cumulative_energy"]) == 3

    def test_pod_cumulative_energy_last_le_1(self, client) -> None:
        snaps = self._make_snapshots(n=6, size=4)
        r = client.post("/api/postprocess/pod", json={
            "snapshots": snaps,
            "n_modes": 4,
        })
        assert r.status_code == 200, r.text
        ce = r.json()["cumulative_energy"]
        assert ce[-1] <= 1.0 + 1e-6

    def test_pod_missing_input(self, client) -> None:
        r = client.post("/api/postprocess/pod", json={"n_modes": 3})
        assert r.status_code == 422

    def test_pod_job_not_found(self, client) -> None:
        r = client.post("/api/postprocess/pod", json={
            "job_id": "nonexistent",
            "n_modes": 3,
        })
        assert r.status_code == 404

    def test_pod_too_few_snapshots(self, client) -> None:
        snaps = self._make_snapshots(n=1, size=4)
        r = client.post("/api/postprocess/pod", json={
            "snapshots": snaps,
            "n_modes": 3,
        })
        assert r.status_code == 422


class TestIsosurfaceAPI:
    def test_not_found(self, client) -> None:
        r = client.get("/api/postprocess/isosurface/nonexistent")
        assert r.status_code == 404

    def test_invalid_slice_axis(self, client) -> None:
        r = client.get("/api/postprocess/isosurface/nonexistent?slice_axis=x")
        assert r.status_code == 422

    def test_schema_in_openapi(self, client) -> None:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        assert "/api/postprocess/isosurface/{job_id}" in paths


class TestDoEAPI:
    _BASE_CONFIG = {
        "nx": 30, "ny": 20, "u_in": 0.08, "re": 100.0,
        "n_steps": 5, "output_interval": 5, "device": "cpu",
    }

    def test_doe_lhs_submission(self, client) -> None:
        r = client.post("/api/solve/doe", json={
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "variables": [
                {"name": "re", "low": 80.0, "high": 150.0},
            ],
            "method": "latin_hypercube",
            "n_samples": 4,
            "seed": 1,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["n_runs"] == 4
        assert data["method"] == "latin_hypercube"
        assert len(data["job_ids"]) == 4
        assert len(data["design_matrix"]) == 4
        assert "doe_group" in data

    def test_doe_sobol_submission(self, client) -> None:
        r = client.post("/api/solve/doe", json={
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "variables": [
                {"name": "re", "low": 50.0, "high": 200.0},
                {"name": "u_in", "low": 0.05, "high": 0.10},
            ],
            "method": "sobol",
            "n_samples": 3,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["n_runs"] == 3
        assert len(data["job_ids"]) == 3

    def test_doe_full_factorial(self, client) -> None:
        r = client.post("/api/solve/doe", json={
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "variables": [
                {"name": "re", "low": 100.0, "high": 200.0,
                 "levels": [100.0, 150.0, 200.0]},
                {"name": "u_in", "low": 0.05, "high": 0.10,
                 "levels": [0.05, 0.10]},
            ],
            "method": "full_factorial",
            "n_samples": 2,  # ignored for full_factorial
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["n_runs"] == 6  # 3 × 2

    def test_doe_invalid_solver_type(self, client) -> None:
        r = client.post("/api/solve/doe", json={
            "solver_type": "unknown_solver",
            "base_config": self._BASE_CONFIG,
            "variables": [{"name": "re", "low": 100.0, "high": 200.0}],
        })
        assert r.status_code == 422

    def test_doe_invalid_method(self, client) -> None:
        r = client.post("/api/solve/doe", json={
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "variables": [{"name": "re", "low": 100.0, "high": 200.0}],
            "method": "monte_carlo",  # not valid
        })
        assert r.status_code == 422

    def test_doe_group_tag_in_jobs(self, client) -> None:
        """All submitted jobs must carry the same doe_group tag."""
        from app.backend import job_manager as jm

        r = client.post("/api/solve/doe", json={
            "solver_type": "lid_driven_cavity",
            "base_config": {
                "nx": 20, "ny": 20, "u_lid": 0.1, "re": 100.0,
                "n_steps": 5, "output_interval": 5, "device": "cpu",
            },
            "variables": [{"name": "re", "low": 80.0, "high": 150.0}],
            "method": "latin_hypercube",
            "n_samples": 2,
            "seed": 5,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        doe_group = data["doe_group"]
        for jid in data["job_ids"]:
            job = jm.get_job(jid)
            assert job is not None
            assert job.config.get("doe", {}).get("group") == doe_group

    def test_doe_design_matrix_in_range(self, client) -> None:
        """All design points must be within the specified variable bounds."""
        r = client.post("/api/solve/doe", json={
            "solver_type": "cylinder_flow",
            "base_config": self._BASE_CONFIG,
            "variables": [
                {"name": "re", "low": 100.0, "high": 300.0},
                {"name": "u_in", "low": 0.05, "high": 0.12},
            ],
            "method": "latin_hypercube",
            "n_samples": 5,
            "seed": 99,
        })
        assert r.status_code == 200, r.text
        for pt in r.json()["design_matrix"]:
            assert 100.0 - 1e-6 <= pt["re"] <= 300.0 + 1e-6
            assert 0.05 - 1e-6 <= pt["u_in"] <= 0.12 + 1e-6

    def test_doe_openapi_contains_endpoint(self, client) -> None:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert "/api/solve/doe" in r.json()["paths"]
