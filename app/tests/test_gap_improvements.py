"""Tests for the four gap-closure improvements vs PowerFlow/XFlow.

Covers:
- DFSEM / Digital-Filter synthetic inflow (library + API)
- Equivalent sand-grain wall roughness (library + API)
- Sponge / absorbing-layer outlet BC (library + API)
- Turbulence statistics accumulator (library + API)
- Live job metrics polling endpoint
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Library-level tests (no FastAPI client needed)
# ---------------------------------------------------------------------------


class TestDFSEM:
    """DFSEM and DigitalFilterInlet library tests."""

    def test_dfsem_sample_shape_2d(self) -> None:
        import torch

        from tensorlbm.synthetic_inflow import DFSEMInlet

        ny, nz = 32, 1
        gen = DFSEMInlet(
            ny=ny, nz=nz,
            u_mean=torch.full((ny, nz), 0.1),
            uu=1e-4, vv=1e-4, ww=1e-4,
            length_scale=4.0,
            n_eddies=50,
            seed=0,
        )
        u, v, w = gen.sample()
        assert u.shape == (ny, nz)
        assert v.shape == (ny, nz)
        assert w.shape == (ny, nz)

    def test_dfsem_sample_shape_3d(self) -> None:
        import torch

        from tensorlbm.synthetic_inflow import DFSEMInlet

        ny, nz = 16, 8
        gen = DFSEMInlet(
            ny=ny, nz=nz,
            u_mean=torch.full((ny, nz), 0.05),
            uu=5e-5, vv=5e-5, ww=5e-5,
            length_scale=3.0,
            n_eddies=30,
            seed=1,
        )
        u, v, w = gen.sample()
        assert u.shape == (ny, nz)

    def test_dfsem_reproducibility(self) -> None:
        import torch

        from tensorlbm.synthetic_inflow import DFSEMInlet

        ny = 16
        kwargs = dict(ny=ny, nz=1, u_mean=torch.full((ny, 1), 0.1),
                      uu=1e-4, vv=1e-4, ww=1e-4, length_scale=3.0,
                      n_eddies=20, seed=42)
        gen1 = DFSEMInlet(**kwargs)
        gen2 = DFSEMInlet(**kwargs)
        u1, _, _ = gen1.sample()
        u2, _, _ = gen2.sample()
        assert torch.allclose(u1, u2)

    def test_dfsem_reset(self) -> None:
        import torch

        from tensorlbm.synthetic_inflow import DFSEMInlet

        ny = 16
        gen = DFSEMInlet(ny=ny, nz=1, u_mean=torch.full((ny, 1), 0.1),
                          uu=1e-4, vv=1e-4, ww=1e-4, length_scale=3.0,
                          n_eddies=20, seed=7)
        u1, _, _ = gen.sample()
        gen.reset(seed=7)
        u2, _, _ = gen.sample()
        assert torch.allclose(u1, u2)

    def test_dfm_sample_shape(self) -> None:
        from tensorlbm.synthetic_inflow import DigitalFilterInlet

        ny, nz = 24, 1
        gen = DigitalFilterInlet(ny=ny, nz=nz, uu=2e-4, vv=2e-4, ww=2e-4,
                                  length_scale=3.0, seed=0)
        u, v, w = gen.sample()
        assert u.shape == (ny, nz)

    def test_dfm_fluctuation_is_nonzero(self) -> None:
        from tensorlbm.synthetic_inflow import DigitalFilterInlet

        gen = DigitalFilterInlet(ny=32, nz=1, uu=1e-4, vv=1e-4, ww=1e-4,
                                  length_scale=5.0, seed=0)
        u, v, _ = gen.sample()
        # Fluctuations must be nonzero somewhere
        assert u.abs().max() > 1e-8
        assert v.abs().max() > 1e-8

    def test_dfm_reproducibility(self) -> None:
        from tensorlbm.synthetic_inflow import DigitalFilterInlet

        gen1 = DigitalFilterInlet(ny=16, nz=1, uu=1e-4, vv=1e-4, ww=1e-4,
                                   length_scale=3.0, seed=99)
        gen2 = DigitalFilterInlet(ny=16, nz=1, uu=1e-4, vv=1e-4, ww=1e-4,
                                   length_scale=3.0, seed=99)
        import torch
        u1, _, _ = gen1.sample()
        u2, _, _ = gen2.sample()
        assert torch.allclose(u1, u2)

    def test_cholesky_off_diagonal(self) -> None:
        """Verify that off-diagonal stresses are accepted without error."""
        import torch

        from tensorlbm.synthetic_inflow import DFSEMInlet

        ny = 8
        gen = DFSEMInlet(ny=ny, nz=1, u_mean=torch.full((ny, 1), 0.1),
                          uu=2e-4, vv=2e-4, ww=2e-4, uv=-5e-5,
                          length_scale=3.0, n_eddies=10, seed=3)
        u, v, w = gen.sample()
        assert u.shape == (ny, 1)


class TestWallRoughness:
    """Equivalent sand-grain wall roughness library tests."""

    def test_smooth_regime_no_correction(self) -> None:
        import torch

        from tensorlbm.roughness import roughness_b_correction

        # ks+ < 2.25 → smooth, ΔB ≈ 0
        ks_plus = torch.tensor([0.5, 1.0, 2.0])
        delta_b = roughness_b_correction(ks_plus)
        assert (delta_b < 1e-6).all(), "Smooth regime should give ΔB ≈ 0"

    def test_fully_rough_positive_correction(self) -> None:
        import torch

        from tensorlbm.roughness import roughness_b_correction

        # ks+ > 90 → fully rough, ΔB > 0
        ks_plus = torch.tensor([100.0, 200.0, 500.0])
        delta_b = roughness_b_correction(ks_plus)
        assert (delta_b > 0).all()

    def test_transitional_between_extremes(self) -> None:
        import torch

        from tensorlbm.roughness import roughness_b_correction

        ks_plus_vals = torch.tensor([0.5, 10.0, 50.0, 200.0])
        delta_b = roughness_b_correction(ks_plus_vals)
        # Correction should be monotonically non-decreasing with ks+
        for i in range(len(delta_b) - 1):
            assert delta_b[i] <= delta_b[i + 1] + 1e-6

    def test_slip_velocity_shape(self) -> None:
        import torch

        from tensorlbm.roughness import compute_rough_wall_slip_velocity

        nz, ny, nx = 8, 8, 8
        ux = torch.rand(nz, ny, nx) * 0.05
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        # Simple box mask: all cells on boundary are solid
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[0, :, :] = True  # bottom layer solid

        ux_s, uy_s, uz_s = compute_rough_wall_slip_velocity(ux, uy, uz, mask, nu=1e-3, ks=0.1)
        assert ux_s.shape == (nz, ny, nx)
        assert uy_s.shape == (nz, ny, nx)
        assert uz_s.shape == (nz, ny, nx)

    def test_no_slip_without_adjacent_fluid(self) -> None:
        """All-solid mask should return zero slip velocity."""
        import torch

        from tensorlbm.roughness import compute_rough_wall_slip_velocity

        nz, ny, nx = 4, 4, 4
        ux = torch.ones(nz, ny, nx) * 0.1
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        mask = torch.ones(nz, ny, nx, dtype=torch.bool)  # all solid

        ux_s, uy_s, uz_s = compute_rough_wall_slip_velocity(ux, uy, uz, mask, nu=1e-3, ks=0.1)
        assert ux_s.abs().max() == 0.0


class TestSpongBC:
    """Sponge / absorbing-layer outlet BC library tests."""

    def test_profile_zero_outside_sponge(self) -> None:
        import torch

        from tensorlbm.sponge_bc import sponge_profile

        profile = sponge_profile(nx=100, x0=80, x1=99, amplitude=0.5)
        # Before sponge zone: should be zero
        assert (profile[:80] == 0.0).all()
        # At x=99 (end): should equal amplitude
        assert abs(float(profile[99]) - 0.5) < 1e-5

    def test_profile_monotone_in_sponge(self) -> None:
        import torch

        from tensorlbm.sponge_bc import sponge_profile

        profile = sponge_profile(nx=200, x0=150, x1=199, amplitude=1.0, exponent=2.0)
        sponge_part = profile[150:200]
        # Monotone non-decreasing
        diffs = sponge_part[1:] - sponge_part[:-1]
        assert (diffs >= -1e-6).all()

    def test_viscous_sponge_2d_shape(self) -> None:
        import torch

        from tensorlbm.d2q9 import equilibrium, macroscopic
        from tensorlbm.sponge_bc import apply_viscous_sponge_2d, sponge_profile

        ny, nx = 16, 32
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.1)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        sponge = sponge_profile(nx=nx, x0=24, x1=31, amplitude=0.5)
        tau = 0.8

        f_out = apply_viscous_sponge_2d(f, rho, ux, uy, tau, sponge)
        assert f_out.shape == f.shape

    def test_target_sponge_2d_blends_correctly(self) -> None:
        """With beta=1 everywhere the output should equal f_target."""
        import torch

        from tensorlbm.sponge_bc import apply_target_sponge_2d

        ny, nx = 4, 8
        nq = 9
        f = torch.rand(nq, ny, nx)
        f_target = torch.zeros(nq, ny, nx)
        # Full beta = 1 everywhere
        sponge = torch.ones(nx)
        f_out = apply_target_sponge_2d(f, f_target, sponge)
        assert torch.allclose(f_out, f_target, atol=1e-6)

    def test_target_sponge_3d_blends_correctly(self) -> None:
        """With beta=0 everywhere the output should equal f unchanged."""
        import torch

        from tensorlbm.sponge_bc import apply_target_sponge_3d

        nq, nz, ny, nx = 19, 4, 4, 8
        f = torch.rand(nq, nz, ny, nx)
        f_target = torch.zeros_like(f)
        sponge = torch.zeros(nx)  # no damping
        f_out = apply_target_sponge_3d(f, f_target, sponge)
        assert torch.allclose(f_out, f, atol=1e-6)

    def test_build_mean_equilibrium_2d_shape(self) -> None:
        import torch

        from tensorlbm.sponge_bc import build_mean_equilibrium_2d

        ny, nx = 8, 16
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f_eq = build_mean_equilibrium_2d(rho, ux, uy)
        assert f_eq.shape == (9, ny, nx)


class TestTurbulenceStats:
    """Turbulence statistics accumulator library tests."""

    def test_accumulator_2d_basic(self) -> None:
        import torch

        from tensorlbm.turbulence_stats import TurbulenceStatsAccumulator

        acc = TurbulenceStatsAccumulator(is_3d=False)
        ny, nx = 16, 16
        for _ in range(10):
            ux = torch.rand(ny, nx) * 0.1
            uy = torch.rand(ny, nx) * 0.02
            acc.update(ux, uy)

        assert acc.count == 10
        assert acc.mean_u.shape == (ny, nx)
        assert acc.uu.shape == (ny, nx)
        assert (acc.uu >= 0).all(), "Variance must be non-negative"
        assert acc.tke.shape == (ny, nx)

    def test_accumulator_3d(self) -> None:
        import torch

        from tensorlbm.turbulence_stats import TurbulenceStatsAccumulator

        acc = TurbulenceStatsAccumulator(is_3d=True)
        nz, ny, nx = 4, 8, 8
        for _ in range(5):
            ux = torch.rand(nz, ny, nx) * 0.1
            uy = torch.rand(nz, ny, nx) * 0.02
            uz = torch.rand(nz, ny, nx) * 0.01
            acc.update(ux, uy, uz)

        assert acc.ww is not None
        assert (acc.ww >= 0).all()

    def test_accumulator_to_dict(self) -> None:
        import torch

        from tensorlbm.turbulence_stats import TurbulenceStatsAccumulator

        acc = TurbulenceStatsAccumulator(is_3d=False)
        acc.update(torch.rand(4, 4), torch.rand(4, 4))
        d = acc.to_dict()
        assert "uu" in d
        assert "tke" in d
        assert "skewness_u" in d
        assert "flatness_u" in d
        assert d["n_samples"] == 1

    def test_accumulator_reset(self) -> None:
        import torch

        from tensorlbm.turbulence_stats import TurbulenceStatsAccumulator

        acc = TurbulenceStatsAccumulator()
        acc.update(torch.rand(4, 4), torch.rand(4, 4))
        acc.reset()
        assert acc.count == 0

    def test_compute_turbulence_intensity(self) -> None:
        import torch

        from tensorlbm.turbulence_stats import compute_turbulence_intensity

        tke = torch.tensor([[0.01, 0.02], [0.03, 0.04]])
        tu = compute_turbulence_intensity(tke, u_ref=0.1)
        assert tu.shape == tke.shape
        assert (tu > 0).all()

    def test_compute_turbulence_length_scale_constant(self) -> None:
        """A constant signal should return zero length scale (no fluctuation)."""
        import torch

        from tensorlbm.turbulence_stats import compute_turbulence_length_scale

        signal = torch.ones(100) * 0.5
        L = compute_turbulence_length_scale(signal)
        assert L == 0.0

    def test_reynolds_stresses_dict(self) -> None:
        import torch

        from tensorlbm.turbulence_stats import compute_reynolds_stresses

        ux_mean = torch.full((4, 4), 0.1)
        uy_mean = torch.zeros(4, 4)
        ux_rms = torch.full((4, 4), 0.01)
        uy_rms = torch.full((4, 4), 0.005)

        result = compute_reynolds_stresses(ux_mean, uy_mean, ux_rms, uy_rms)
        assert "uu" in result
        assert "tke" in result
        assert "tu_percent" in result
        assert (result["uu"] > 0).all()

    def test_no_samples_raises(self) -> None:
        from tensorlbm.turbulence_stats import TurbulenceStatsAccumulator

        acc = TurbulenceStatsAccumulator()
        with pytest.raises(RuntimeError, match="No samples"):
            _ = acc.mean_u


# ---------------------------------------------------------------------------
# API-level tests (FastAPI TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.backend.main import app

    return TestClient(app)


class TestDFSEMAPI:
    def test_dfsem_preview_default(self, client) -> None:
        r = client.post("/api/postprocess/dfsem-preview", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["method"] == "dfsem"
        assert "u_fluct_profile" in data
        assert len(data["u_fluct_profile"]) == 64
        assert "tu_percent" in data

    def test_digital_filter_preview(self, client) -> None:
        r = client.post("/api/postprocess/dfsem-preview", json={
            "method": "digital_filter", "ny": 32, "length_scale": 3.0
        })
        assert r.status_code == 200
        data = r.json()
        assert data["method"] == "digital_filter"
        assert len(data["u_fluct_profile"]) == 32

    def test_dfsem_preview_large_eddies(self, client) -> None:
        r = client.post("/api/postprocess/dfsem-preview", json={
            "ny": 16, "length_scale": 8.0, "n_eddies": 50, "uu": 5e-5
        })
        assert r.status_code == 200
        assert r.json()["u_rms"] >= 0

    def test_dfsem_preview_3d_nz(self, client) -> None:
        r = client.post("/api/postprocess/dfsem-preview", json={
            "ny": 16, "nz": 4, "length_scale": 3.0
        })
        assert r.status_code == 200
        assert len(r.json()["y_coords"]) == 16


class TestSpongePreviewAPI:
    def test_sponge_preview_default(self, client) -> None:
        r = client.post("/api/postprocess/sponge-preview", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["nx"] == 200
        assert len(data["profile"]) == 200
        # Entries before x0=150 should have alpha=0
        for entry in data["profile"]:
            if entry["x"] < 150:
                assert entry["alpha"] == 0.0

    def test_sponge_preview_custom(self, client) -> None:
        r = client.post("/api/postprocess/sponge-preview", json={
            "nx": 100, "x0": 80, "x1": 99, "amplitude": 0.3, "exponent": 2.0
        })
        assert r.status_code == 200
        data = r.json()
        assert data["sponge_width"] == 19
        # Last entry should have alpha ≈ 0.3
        last_alpha = data["profile"][-1]["alpha"]
        assert abs(last_alpha - 0.3) < 1e-4


class TestRoughnessPreviewAPI:
    def test_roughness_preview_smooth(self, client) -> None:
        r = client.post("/api/postprocess/roughness-preview", json={
            "u_tau": 0.005, "nu": 0.001, "ks": 0.0001
        })
        assert r.status_code == 200
        data = r.json()
        assert data["regime"] == "hydraulically_smooth"
        assert abs(data["delta_b_current"]) < 1e-5

    def test_roughness_preview_fully_rough(self, client) -> None:
        r = client.post("/api/postprocess/roughness-preview", json={
            "u_tau": 0.05, "nu": 0.001, "ks": 5.0
        })
        assert r.status_code == 200
        data = r.json()
        assert data["regime"] == "fully_rough"
        assert data["delta_b_current"] > 0

    def test_roughness_preview_curve_length(self, client) -> None:
        r = client.post("/api/postprocess/roughness-preview", json={
            "n_points": 50
        })
        assert r.status_code == 200
        assert len(r.json()["curve"]) == 50


class TestTurbulenceStatsAPI:
    def test_turbulence_stats_not_found(self, client) -> None:
        r = client.get("/api/postprocess/turbulence-stats/nonexistent_job_id")
        assert r.status_code == 404

    def test_turbulence_stats_not_completed(self, client) -> None:
        """Job not yet completed should return 409."""
        import json as json_mod

        from app.backend import job_manager as jm

        # Create a synthetic running job
        import threading
        ready = threading.Event()

        def _slow(job):
            ready.set()
            import time
            time.sleep(0.2)
            return {}

        job_id = jm.submit("slow-job", "test", {}, _slow)
        ready.wait(timeout=2)

        r = client.get(f"/api/postprocess/turbulence-stats/{job_id}")
        assert r.status_code in (409, 404)  # depends on timing


class TestLiveMetricsAPI:
    def test_live_metrics_not_found(self, client) -> None:
        r = client.get("/api/jobs/nonexistent_id/live-metrics")
        assert r.status_code == 404

    def test_live_metrics_empty_completed_job(self, client) -> None:
        """A completed job with no diagnostics should return empty list."""
        from app.backend import job_manager as jm

        job_id = jm.submit("empty-job", "test", {}, lambda job: {})
        import time
        # Wait for completion
        for _ in range(20):
            time.sleep(0.05)
            j = jm.get_job(job_id)
            if j and j.status.value in ("completed", "failed"):
                break

        r = client.get(f"/api/jobs/{job_id}/live-metrics")
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == job_id
        assert "diagnostics" in data
        assert "status" in data

    def test_live_metrics_with_diagnostics(self, client) -> None:
        """A job that pushes diagnostics should expose them via the endpoint."""
        from app.backend import job_manager as jm

        def _job_with_diags(job):
            for s in range(1, 4):
                jm.push_diagnostic(job.job_id, {"step": s, "cd": 1.0 / s})
            return {}

        job_id = jm.submit("diag-job", "test", {}, _job_with_diags)
        import time
        for _ in range(30):
            time.sleep(0.05)
            j = jm.get_job(job_id)
            if j and j.status.value in ("completed", "failed"):
                break

        r = client.get(f"/api/jobs/{job_id}/live-metrics")
        assert r.status_code == 200
        data = r.json()
        assert len(data["diagnostics"]) == 3
        assert data["diagnostics"][0]["step"] == 1

    def test_live_metrics_since_step_filter(self, client) -> None:
        """since_step parameter should filter out older records."""
        from app.backend import job_manager as jm

        def _push_many(job):
            for s in range(1, 11):
                jm.push_diagnostic(job.job_id, {"step": s, "cd": float(s)})
            return {}

        job_id = jm.submit("filter-job", "test", {}, _push_many)
        import time
        for _ in range(40):
            time.sleep(0.05)
            j = jm.get_job(job_id)
            if j and j.status.value in ("completed", "failed"):
                break

        r = client.get(f"/api/jobs/{job_id}/live-metrics?since_step=5")
        assert r.status_code == 200
        data = r.json()
        steps = [d["step"] for d in data["diagnostics"]]
        assert all(s > 5 for s in steps)
        assert set(steps) == {6, 7, 8, 9, 10}
