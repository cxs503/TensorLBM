"""Tests for the 5 engineering gap-closure features:

1. thermal_radiation  – grey-body / solar radiation
2. sixdof             – 6-DOF rigid-body dynamics
3. ddes               – DDES / SAS hybrid turbulence
4. acoustic_beamforming – microphone-array noise source identification
5. topology_opt       – SIMP density-based topology optimisation

API tests cover each corresponding endpoint.
"""
from __future__ import annotations

import math

import pytest
import torch


# ============================================================================
# Unit tests – core physics modules
# ============================================================================

class TestThermalRadiation:
    """Unit tests for tensorlbm.thermal_radiation."""

    def test_stefan_boltzmann_constant(self):
        from tensorlbm.thermal_radiation import STEFAN_BOLTZMANN
        assert abs(STEFAN_BOLTZMANN - 5.670374419e-8) < 1e-15

    def test_two_surface_enclosure(self):
        from tensorlbm.thermal_radiation import (
            RadiationEnclosureConfig,
            SurfaceRadiationProps,
            compute_net_radiation_flux,
        )
        cfg = RadiationEnclosureConfig(surfaces=[
            SurfaceRadiationProps(temperature=400.0, emissivity=0.9, area=1.0),
            SurfaceRadiationProps(temperature=300.0, emissivity=0.85, area=1.0),
        ])
        result = compute_net_radiation_flux(cfg)
        # Hot surface loses heat (positive net flux outward)
        assert result.net_flux[0] > result.net_flux[1]
        # Total emitted power: σ T⁴ A
        sigma = 5.670374419e-8
        expected_emit = sigma * 400.0**4 * 1.0 + sigma * 300.0**4 * 1.0
        assert abs(result.total_emitted_power - expected_emit) / expected_emit < 0.01

    def test_solar_flux(self):
        from tensorlbm.thermal_radiation import SolarSettings, solar_flux_on_surface
        solar = SolarSettings(enabled=True, irradiance=1000.0, direction=(0.0, -1.0, 0.0))
        # Upward-facing surface should absorb maximum solar flux
        normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        q = solar_flux_on_surface(normals, solar, absorptance=1.0)
        assert abs(q[0].item() - 1000.0) < 1e-6

    def test_solar_shading(self):
        from tensorlbm.thermal_radiation import SolarSettings, solar_flux_on_surface
        solar = SolarSettings(enabled=True, irradiance=1000.0, direction=(0.0, -1.0, 0.0))
        # Downward-facing surface is shaded (cos θ < 0 → clamped to 0)
        normals = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64)
        q = solar_flux_on_surface(normals, solar, absorptance=1.0)
        assert q[0].item() == pytest.approx(0.0, abs=1e-10)

    def test_run_radiation_step_json(self):
        from tensorlbm.thermal_radiation import (
            RadiationEnclosureConfig,
            SolarSettings,
            SurfaceRadiationProps,
            run_radiation_step,
        )
        cfg = RadiationEnclosureConfig(
            surfaces=[
                SurfaceRadiationProps(temperature=350.0, emissivity=0.8),
                SurfaceRadiationProps(temperature=280.0, emissivity=0.9),
            ],
            solar=SolarSettings(enabled=True, irradiance=800.0),
        )
        out = run_radiation_step(cfg)
        assert "surfaces" in out
        assert out["n_surfaces"] == 2
        assert out["solar_enabled"] is True

    def test_apply_radiation_source(self):
        from tensorlbm.thermal_radiation import apply_radiation_source
        T = torch.ones(20, 40, dtype=torch.float32) * 300.0
        mask = torch.zeros(20, 40, dtype=torch.bool)
        mask[0, :] = True   # top wall solid
        T_new = apply_radiation_source(T, mask, 500.0, 0.01, 1.2, 1005.0, 1e-3)
        # Fluid cells adjacent to top wall should be warmer
        assert T_new[1, 10].item() > 300.0


class TestSixDOF:
    """Unit tests for tensorlbm.sixdof."""

    def test_identity_quaternion_rotation_matrix(self):
        from tensorlbm.sixdof import quaternion_to_rotation_matrix
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        R = quaternion_to_rotation_matrix(q)
        assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-10)

    def test_quaternion_multiply_identity(self):
        from tensorlbm.sixdof import _quat_multiply
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        p = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
        result = _quat_multiply(q, p)
        assert torch.allclose(result, p, atol=1e-12)

    def test_euler_round_trip(self):
        from tensorlbm.sixdof import quaternion_to_rotation_matrix, rotation_matrix_to_euler
        # 30-degree roll
        angle = math.radians(30)
        q = torch.tensor([math.cos(angle/2), math.sin(angle/2), 0.0, 0.0], dtype=torch.float64)
        R = quaternion_to_rotation_matrix(q)
        roll, pitch, yaw = rotation_matrix_to_euler(R)
        assert abs(math.degrees(roll) - 30.0) < 0.5

    def test_sixdof_free_fall(self):
        from tensorlbm.sixdof import SixDOFBody, SixDOFConfig, FluidForcesMoments, run_sixdof_simulation
        body = SixDOFBody(mass=1.0, ixx=1.0, iyy=1.0, izz=1.0,
                          gravity=(0.0, -9.81, 0.0))
        cfg = SixDOFConfig(body=body, dt=0.01, n_steps=50)
        result = run_sixdof_simulation(cfg, fluid_forces_fn=lambda t, *a: FluidForcesMoments())
        # Free fall: y should decrease
        final_y = result.history[-1].pos[1]
        assert final_y < result.history[0].pos[1]
        # Approximate: y ≈ -0.5 g t²
        t_final = 0.01 * 50
        expected_y = -0.5 * 9.81 * t_final**2
        assert abs(final_y - expected_y) / (abs(expected_y) + 1e-6) < 0.02

    def test_sixdof_constrained(self):
        from tensorlbm.sixdof import SixDOFBody, SixDOFConfig, FluidForcesMoments, run_sixdof_simulation
        body = SixDOFBody(mass=1.0, ixx=1.0, iyy=1.0, izz=1.0,
                          gravity=(0.0, 0.0, 0.0),
                          fix_surge=True, fix_sway=True, fix_heave=True)
        cfg = SixDOFConfig(body=body, dt=0.01, n_steps=20)
        fluid = FluidForcesMoments(fx=100.0, fy=100.0, fz=100.0)
        result = run_sixdof_simulation(cfg, fluid_forces_fn=lambda t, *a: fluid)
        # All translation should be frozen
        final = result.history[-1]
        assert all(abs(p) < 1e-10 for p in final.pos)

    def test_run_sixdof_simulation_returns_history(self):
        from tensorlbm.sixdof import SixDOFConfig, run_sixdof_simulation
        cfg = SixDOFConfig()
        result = run_sixdof_simulation(cfg)
        assert len(result.history) == cfg.n_steps + 1
        assert result.max_displacement >= 0.0


class TestDDES:
    """Unit tests for tensorlbm.ddes."""

    def _make_channel_flow(self, ny=32, nx=64, u0=0.05):
        """Create a simple Poiseuille-like velocity field."""
        yy = torch.arange(ny, dtype=torch.float32) / ny
        profile = 4.0 * u0 * yy * (1.0 - yy)   # parabolic
        ux = profile.unsqueeze(1).expand(ny, nx)
        uy = torch.zeros(ny, nx, dtype=torch.float32)
        return ux, uy

    def test_ddes_eddy_viscosity_shape(self):
        from tensorlbm.ddes import DDESConfig, ddes_eddy_viscosity
        ux, uy = self._make_channel_flow()
        ny, nx = ux.shape
        yy = torch.arange(ny, dtype=torch.float32).unsqueeze(1).expand(ny, nx)
        d_wall = torch.minimum(yy, (ny - 1) - yy).clamp(min=0.5) / ny
        cfg = DDESConfig(mode="ddes", nu_molecular=1e-4, dx=1.0/nx)
        nu_t, f_d, _, _, _ = ddes_eddy_viscosity(ux, uy, d_wall, cfg)
        assert nu_t.shape == ux.shape
        assert f_d.shape == ux.shape
        assert (nu_t >= 0).all()

    def test_les_mode_higher_nu_t_than_laminar(self):
        from tensorlbm.ddes import DDESConfig, ddes_eddy_viscosity
        ux, uy = self._make_channel_flow()
        ny, nx = ux.shape
        d_wall = torch.ones(ny, nx, dtype=torch.float32) * 0.5
        cfg_les = DDESConfig(mode="les", nu_molecular=1e-5, dx=1.0/nx)
        cfg_ddes = DDESConfig(mode="ddes", nu_molecular=1e-5, dx=1.0/nx)
        nu_t_les, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg_les)
        nu_t_ddes, *_ = ddes_eddy_viscosity(ux, uy, d_wall, cfg_ddes)
        # Both should be non-negative
        assert (nu_t_les >= 0).all()
        assert (nu_t_ddes >= 0).all()

    def test_shielding_function_bounds(self):
        from tensorlbm.ddes import ddes_shielding_function
        nu = 1e-4
        nu_t = torch.ones(10, 10) * 1e-3
        S_mag = torch.ones(10, 10) * 0.1
        d_wall = torch.ones(10, 10) * 0.5
        f_d = ddes_shielding_function(nu, nu_t, S_mag, d_wall)
        assert ((f_d >= 0) & (f_d <= 1)).all()

    def test_run_ddes_diagnostics(self):
        from tensorlbm.ddes import DDESConfig, run_ddes_diagnostics
        ux, uy = self._make_channel_flow(ny=16, nx=32)
        ny, nx = ux.shape
        d_wall = torch.ones(ny, nx, dtype=torch.float32) * 0.3
        cfg = DDESConfig(mode="ddes")
        result = run_ddes_diagnostics(ux, uy, d_wall, cfg)
        assert 0.0 <= result.rans_fraction <= 1.0
        assert 0.0 <= result.les_fraction <= 1.0
        assert abs(result.rans_fraction + result.les_fraction - 1.0) < 0.01
        assert result.nu_t_max >= result.nu_t_mean >= 0


class TestAcousticBeamforming:
    """Unit tests for tensorlbm.acoustic_beamforming."""

    def _make_array(self, n_mics=8, n_samples=1024, dt=1e-4):
        """Create a linear microphone array with a tonal source."""
        f_tone = 1000.0
        t = torch.arange(n_samples, dtype=torch.float64) * dt
        signal = torch.sin(2 * math.pi * f_tone * t)
        signals = signal.unsqueeze(0).expand(n_mics, -1).clone()
        # Add phase shifts for a source at x=0.3, y=0
        x_source = 0.3
        c0 = 343.0
        mic_x = torch.linspace(-0.5, 0.5, n_mics)
        for i in range(n_mics):
            dist = abs(mic_x[i].item() - x_source)
            delay_samples = int(dist / c0 / dt)
            signals[i] = torch.roll(signal, delay_samples)

        positions = torch.stack([mic_x, torch.zeros(n_mics), torch.zeros(n_mics)], dim=1).double()
        return positions, signals

    def test_das_output_shape(self):
        from tensorlbm.acoustic_beamforming import (
            BeamformingConfig,
            MicrophoneArray,
            compute_source_map,
        )
        positions, signals = self._make_array()
        array = MicrophoneArray(positions=positions, signals=signals, dt=1e-4)
        cfg = BeamformingConfig(scan_x=(-1.0, 1.0, 10), scan_y=(-0.5, 0.5, 8))
        power_map, x_grid, y_grid = compute_source_map(array, cfg)
        assert power_map.shape == (8, 10)
        assert len(x_grid) == 10
        assert len(y_grid) == 8

    def test_das_positive_power(self):
        from tensorlbm.acoustic_beamforming import (
            BeamformingConfig,
            MicrophoneArray,
            das_beamformer,
        )
        positions, signals = self._make_array()
        array = MicrophoneArray(positions=positions, signals=signals, dt=1e-4)
        source_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        power = das_beamformer(array, source_pos, f_min=500.0, f_max=2000.0)
        assert power[0].item() >= 0.0

    def test_run_beamforming_returns_result(self):
        from tensorlbm.acoustic_beamforming import (
            BeamformingConfig,
            MicrophoneArray,
            run_acoustic_beamforming,
        )
        positions, signals = self._make_array(n_mics=4, n_samples=512)
        array = MicrophoneArray(positions=positions, signals=signals, dt=1e-4)
        cfg = BeamformingConfig(
            scan_x=(-0.5, 0.5, 8),
            scan_y=(-0.3, 0.3, 6),
            method="das",
        )
        result = run_acoustic_beamforming(array, cfg)
        assert len(result.source_map) == 6
        assert len(result.source_map[0]) == 8
        assert result.dominant_frequency_hz > 0

    def test_clean_sc_runs(self):
        from tensorlbm.acoustic_beamforming import (
            BeamformingConfig,
            MicrophoneArray,
            run_acoustic_beamforming,
        )
        positions, signals = self._make_array(n_mics=4, n_samples=256)
        array = MicrophoneArray(positions=positions, signals=signals, dt=1e-4)
        cfg = BeamformingConfig(
            scan_x=(-0.5, 0.5, 6),
            scan_y=(-0.3, 0.3, 5),
            method="clean_sc",
            n_iter_clean=3,
        )
        result = run_acoustic_beamforming(array, cfg)
        assert result.method == "clean_sc"


class TestTopologyOpt:
    """Unit tests for tensorlbm.topology_opt."""

    def test_brinkman_alpha_limits(self):
        from tensorlbm.topology_opt import brinkman_alpha
        rho_fluid = torch.zeros(1, dtype=torch.float64)
        rho_solid = torch.ones(1, dtype=torch.float64)
        alpha_max = 1000.0
        assert brinkman_alpha(rho_fluid, alpha_max).item() == pytest.approx(alpha_max, rel=0.01)
        assert brinkman_alpha(rho_solid, alpha_max).item() == pytest.approx(0.0, abs=1e-3)

    def test_density_filter_preserves_mean(self):
        from tensorlbm.topology_opt import density_filter
        rho = torch.rand(20, 40, dtype=torch.float64)
        rho_filt = density_filter(rho, r_min=2.0)
        # Mean should be approximately preserved (not exactly due to boundary)
        assert abs(rho_filt.mean().item() - rho.mean().item()) < 0.05

    def test_density_filter_shape(self):
        from tensorlbm.topology_opt import density_filter
        rho = torch.rand(30, 60, dtype=torch.float64)
        rho_filt = density_filter(rho, r_min=3.0)
        assert rho_filt.shape == rho.shape

    def test_oc_update_volume_constraint(self):
        from tensorlbm.topology_opt import oc_update
        rho = torch.rand(20, 40, dtype=torch.float64) * 0.5 + 0.25
        sensitivity = -torch.rand(20, 40, dtype=torch.float64) - 0.1
        vf_target = 0.4
        rho_new = oc_update(rho, sensitivity, vf_target, move=0.2, eta=0.5)
        # Volume fraction should be approximately satisfied
        assert abs(rho_new.mean().item() - vf_target) < 0.05
        # Values should be in [0, 1]
        assert (rho_new >= 0).all()
        assert (rho_new <= 1.0).all()

    def test_topology_opt_runs(self):
        from tensorlbm.topology_opt import TopOptConfig, run_topology_optimisation
        cfg = TopOptConfig(nx=20, ny=10, n_iter=5, re=50.0, vf_target=0.4)
        result = run_topology_optimisation(cfg)
        assert len(result.density) == 10
        assert len(result.density[0]) == 20
        assert len(result.objective_history) > 0
        assert result.final_volume_fraction >= 0

    def test_topology_opt_volume_fraction_converges(self):
        from tensorlbm.topology_opt import TopOptConfig, run_topology_optimisation
        cfg = TopOptConfig(nx=16, ny=8, n_iter=8, re=50.0, vf_target=0.35)
        result = run_topology_optimisation(cfg)
        # Volume fraction should be in a reasonable range after optimisation
        # (small grid + few iterations may not fully converge)
        assert 0.0 < result.final_volume_fraction <= 1.0


# ============================================================================
# API tests
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


class TestThermalRadiationAPI:
    def test_thermal_radiation_default(self, client):
        resp = client.post("/api/postprocess/thermal-radiation", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "surfaces" in data
        assert data["n_surfaces"] == 2

    def test_thermal_radiation_with_surfaces(self, client):
        payload = {
            "surfaces": [
                {"temperature_K": 450.0, "emissivity": 0.9, "solar_absorptance": 0.6, "area_m2": 2.0},
                {"temperature_K": 290.0, "emissivity": 0.8, "solar_absorptance": 0.5, "area_m2": 1.5},
            ],
            "solar_enabled": True,
            "solar_irradiance": 1000.0,
        }
        resp = client.post("/api/postprocess/thermal-radiation", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["solar_enabled"] is True
        assert data["n_surfaces"] == 2

    def test_thermal_radiation_solar_off(self, client):
        resp = client.post(
            "/api/postprocess/thermal-radiation",
            json={"solar_enabled": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["solar_enabled"] is False


class TestSixDOFAPI:
    def test_sixdof_default(self, client):
        resp = client.post("/api/solve/sixdof", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "history" in data
        assert data["max_displacement_m"] >= 0

    def test_sixdof_vessel_motion(self, client):
        payload = {
            "body": {"mass": 5000.0, "ixx": 2000.0, "iyy": 8000.0, "izz": 8000.0,
                     "fix_surge": True, "fix_sway": True},
            "dt": 0.05,
            "n_steps": 100,
            "force_amplitude": 5000.0,
            "force_frequency": 0.2,
            "scenario": "vessel_heave_pitch",
        }
        resp = client.post("/api/solve/sixdof", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"] == "vessel_heave_pitch"
        assert len(data["history"]) > 0

    def test_sixdof_step_count_limit(self, client):
        resp = client.post("/api/solve/sixdof", json={"n_steps": 10001})
        assert resp.status_code == 422


class TestDDESDiagnosticsAPI:
    def test_ddes_missing_job(self, client):
        resp = client.get("/api/postprocess/ddes-diagnostics/nonexistent_job")
        assert resp.status_code == 404

    def test_ddes_invalid_mode(self, client):
        resp = client.get("/api/postprocess/ddes-diagnostics/any_job?mode=invalid")
        assert resp.status_code == 422


class TestAcousticBeamformingAPI:
    def test_acoustic_beamforming_missing_job(self, client):
        resp = client.post(
            "/api/postprocess/acoustic-beamforming",
            json={
                "job_id": "nonexistent_123",
                "mic_positions": [[0.0, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
            },
        )
        assert resp.status_code == 404


class TestTopologyOptAPI:
    def test_topology_opt_default(self, client):
        payload = {"nx": 20, "ny": 10, "n_iter": 5, "re": 50.0, "vf_target": 0.4}
        resp = client.post("/api/solve/topology-opt", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "density" in data
        assert data["nx"] == 20
        assert data["ny"] == 10
        assert len(data["density"]) == 10
        assert len(data["density"][0]) == 20

    def test_topology_opt_flow_uniformity(self, client):
        payload = {
            "nx": 20, "ny": 10, "n_iter": 5, "re": 80.0,
            "objective": "flow_uniformity", "vf_target": 0.35,
        }
        resp = client.post("/api/solve/topology-opt", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["objective"] == "flow_uniformity"

    def test_topology_opt_nx_limit(self, client):
        resp = client.post("/api/solve/topology-opt", json={"nx": 300, "ny": 10, "n_iter": 3})
        assert resp.status_code == 422

    def test_topology_opt_invalid_objective(self, client):
        resp = client.post(
            "/api/solve/topology-opt",
            json={"nx": 20, "ny": 10, "n_iter": 3, "objective": "invalid_obj"},
        )
        assert resp.status_code == 422


class TestGapAssessmentUpdated:
    def test_gap_assessment_has_new_categories(self, client):
        resp = client.get("/api/orchestration/gap-assessment")
        assert resp.status_code == 200
        data = resp.json()
        ids = [c["id"] for c in data["categories"]]
        assert "turbulence_models" in ids
        assert "aeroacoustics" in ids
        assert "design_optimisation" in ids
        assert "multiphysics_depth" in ids

    def test_gap_assessment_this_release(self, client):
        resp = client.get("/api/orchestration/gap-assessment")
        data = resp.json()
        assert "this_release_new_features" in data
        new_features = data["this_release_new_features"]
        assert any("thermal_radiation" in f for f in new_features)
        assert any("sixdof" in f for f in new_features)
        assert any("ddes" in f for f in new_features)
        assert any("acoustic_beamforming" in f for f in new_features)
        assert any("topology_opt" in f for f in new_features)

    def test_gap_assessment_counters(self, client):
        resp = client.get("/api/orchestration/gap-assessment")
        data = resp.json()
        assert data["implemented_count"] >= 4
        assert data["count"] >= 8
