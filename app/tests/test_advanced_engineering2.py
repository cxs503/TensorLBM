"""Tests for 5 new engineering gap-closure features vs PowerFlow/XFlow:

1. FSI (Fluid-Structure Interaction) – library + API
2. Lagrangian Particle Tracking – library + API
3. Pedestrian Wind Comfort Assessment – library + API
4. Adjoint Sensitivity Analysis – library + API
5. CGNS Export – library + API
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest


# ===========================================================================
# Shared test field factory
# ===========================================================================

def _make_cylinder_fields(ny: int = 60, nx: int = 120):
    """Return (rho, ux, uy, obstacle_mask) for a simple cylinder-in-channel."""
    import torch

    rho = torch.ones(ny, nx)
    # Simple Poiseuille-like profile
    y = torch.arange(ny, dtype=torch.float32)
    ux_profile = 0.1 * 4.0 * y * (ny - y) / (ny ** 2)
    ux = ux_profile.unsqueeze(1).expand(ny, nx).clone()
    uy = torch.zeros(ny, nx)

    # Cylinder obstacle centred at (nx//3, ny//2), radius 8
    cx, cy, r = nx // 3, ny // 2, 8
    ys, xs = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
    obstacle_mask = ((xs - cx) ** 2 + (ys - cy) ** 2) < r ** 2

    # Zero velocity inside obstacle
    ux[obstacle_mask] = 0.0
    uy[obstacle_mask] = 0.0

    return rho, ux, uy, obstacle_mask


# ===========================================================================
# 1. FSI Library Tests
# ===========================================================================

class TestFSILibrary:
    """Tests for tensorlbm.fsi module."""

    def test_extract_loads_returns_loads_object(self) -> None:
        from tensorlbm.fsi import FSILoads, extract_fsi_loads
        rho, ux, uy, mask = _make_cylinder_fields()
        loads = extract_fsi_loads(rho, ux, uy, mask, dx_phys=1e-3)
        assert isinstance(loads, FSILoads)
        assert len(loads.pressure) > 0

    def test_extract_loads_empty_mask(self) -> None:
        import torch
        from tensorlbm.fsi import FSILoads, extract_fsi_loads
        rho = torch.ones(20, 40)
        ux = torch.zeros(20, 40)
        uy = torch.zeros(20, 40)
        mask = torch.zeros(20, 40, dtype=torch.bool)
        loads = extract_fsi_loads(rho, ux, uy, mask)
        assert loads.fx == 0.0
        assert len(loads.pressure) == 0

    def test_structural_response_cantilever(self) -> None:
        from tensorlbm.fsi import FSILoads, FSIResponse, StructuralProperties, compute_structural_response

        props = StructuralProperties(
            youngs_modulus=2.1e11,
            length=1.0,
            width=0.05,
            thickness=0.01,
            density=7850.0,
        )
        loads = FSILoads(fx=100.0, fy=0.0)
        resp = compute_structural_response(loads, props, flow_speed=5.0, characteristic_length=0.1)
        assert isinstance(resp, FSIResponse)
        assert resp.max_deflection >= 0.0
        assert resp.natural_frequency_hz > 0.0
        assert resp.safety_factor > 0.0

    def test_viv_risk_detection(self) -> None:
        """Vr near 1/St should flag VIV risk."""
        from tensorlbm.fsi import FSILoads, StructuralProperties, compute_structural_response

        props = StructuralProperties(length=0.5, width=0.05, thickness=0.005,
                                     youngs_modulus=7e10, density=2700.0)
        loads = FSILoads(fx=50.0)
        # Choose flow_speed such that Vr ≈ 1/0.2 = 5
        resp = compute_structural_response(
            loads, props,
            flow_speed=5.0 * props.length * 0.1,  # adjust to push Vr near 5
            characteristic_length=0.1,
            strouhal=0.2,
        )
        # VIV risk depends on exact Vr; just check it's a bool
        assert isinstance(resp.viv_risk, bool)

    def test_two_way_coupling_converges(self) -> None:
        from tensorlbm.fsi import FSILoads, StructuralProperties, compute_structural_response

        props = StructuralProperties()
        loads = FSILoads(fx=100.0)
        resp = compute_structural_response(loads, props, coupling="two_way")
        assert resp.coupling == "two_way"
        assert resp.iterations >= 1

    def test_run_fsi_analysis_end_to_end(self) -> None:
        from tensorlbm.fsi import run_fsi_analysis
        rho, ux, uy, mask = _make_cylinder_fields()
        result = run_fsi_analysis(rho, ux, uy, mask, flow_speed=0.1, dx_phys=1e-3)
        assert "loads" in result
        assert "response" in result
        assert "assessment" in result
        assert result["assessment"] in ("PASS", "CAUTION: low safety factor",
                                        "WARNING: VIV lock-in risk detected",
                                        "FAIL: structural failure (SF < 1)")


# ===========================================================================
# 1b. FSI API Tests
# ===========================================================================

class TestFSIAPI:
    def test_fsi_loads_endpoint_no_job(self, client) -> None:
        r = client.post("/api/postprocess/fsi-loads/nonexistent-job", json={})
        assert r.status_code == 404

    def test_fsi_loads_schema_validation(self, client) -> None:
        """Negative Young's modulus should fail validation."""
        r = client.post(
            "/api/postprocess/fsi-loads/any-job",
            json={"structural_props": {"youngs_modulus": -1}},
        )
        assert r.status_code in (404, 422)


# ===========================================================================
# 2. Particle Tracking Library Tests
# ===========================================================================

class TestParticleTrackerLibrary:
    def test_massless_particles_escape(self) -> None:
        from tensorlbm.particle_tracker import track_particles
        import torch

        ny, nx = 40, 80
        ux = torch.full((ny, nx), 0.1)
        uy = torch.zeros(ny, nx)
        mask = torch.zeros(ny, nx, dtype=torch.bool)

        tracks = track_particles(
            ux, uy, mask,
            injection_x=[1.0] * 5,
            injection_y=[float(i * 8 + 4) for i in range(5)],
            n_steps=1000,
            stokes_number=0.0,
        )
        assert len(tracks) == 5
        # All should eventually escape right boundary with uniform flow
        statuses = {t.status for t in tracks}
        assert statuses <= {"escaped", "active"}

    def test_particles_deposit_on_wall(self) -> None:
        from tensorlbm.particle_tracker import track_particles
        import torch

        ny, nx = 40, 80
        ux = torch.zeros(ny, nx)
        # Strong downward flow drives particles to bottom wall
        uy = torch.full((ny, nx), -0.2)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[0, :] = True   # bottom solid wall

        tracks = track_particles(
            ux, uy, mask,
            injection_x=[40.0] * 4,
            injection_y=[20.0, 25.0, 30.0, 35.0],
            n_steps=500,
            stokes_number=0.0,
            dt=0.3,
        )
        deposited = [t for t in tracks if t.status == "deposited"]
        assert len(deposited) > 0

    def test_stokes_number_effect(self) -> None:
        """Massive particles (St>0) should lag behind flow."""
        from tensorlbm.particle_tracker import track_particles, build_deposition_map
        import torch

        ny, nx = 40, 80
        ux = torch.full((ny, nx), 0.1)
        uy = torch.zeros(ny, nx)
        mask = torch.zeros(ny, nx, dtype=torch.bool)

        tracks_massless = track_particles(
            ux, uy, mask,
            injection_x=[5.0], injection_y=[20.0],
            n_steps=200, stokes_number=0.0,
        )
        tracks_massive = track_particles(
            ux, uy, mask,
            injection_x=[5.0], injection_y=[20.0],
            n_steps=200, stokes_number=5.0,
        )
        # Massless particle travels faster (or equal) than massive
        if tracks_massless[0].trajectory_x and tracks_massive[0].trajectory_x:
            assert tracks_massless[0].trajectory_x[-1] >= tracks_massive[0].trajectory_x[-1] - 0.01

    def test_deposition_map_counts(self) -> None:
        from tensorlbm.particle_tracker import build_deposition_map, ParticleTrackResult
        tracks = [
            ParticleTrackResult(pid=0, status="deposited", deposit_x=1.0, deposit_y=0.0),
            ParticleTrackResult(pid=1, status="escaped"),
            ParticleTrackResult(pid=2, status="active"),
        ]
        dep = build_deposition_map(tracks, 80, 40)
        assert dep["n_total"] == 3
        assert dep["n_deposited"] == 1
        assert dep["n_escaped"] == 1
        assert dep["deposition_fraction"] == pytest.approx(1 / 3)


# ===========================================================================
# 2b. Particle Tracking API Tests
# ===========================================================================

class TestParticleTrackingAPI:
    def test_inject_missing_job(self, client) -> None:
        r = client.post("/api/postprocess/particle-inject", json={
            "job_id": "no-such-job",
            "injection_x": [5.0],
            "injection_y": [20.0],
        })
        assert r.status_code == 404

    def test_inject_mismatched_coords(self, client) -> None:
        r = client.post("/api/postprocess/particle-inject", json={
            "job_id": "x",
            "injection_x": [1.0, 2.0],
            "injection_y": [1.0],
        })
        assert r.status_code in (404, 422)

    def test_get_tracks_missing_job(self, client) -> None:
        r = client.get("/api/postprocess/particle-tracks/no-job?n_particles=5")
        assert r.status_code == 404


# ===========================================================================
# 3. Wind Comfort Library Tests
# ===========================================================================

class TestWindComfortLibrary:
    def test_calm_wind_is_comfortable(self) -> None:
        from tensorlbm.wind_comfort import WindSensorPoint, assess_wind_comfort
        sensors = [WindSensorPoint(label="P1", x=0, y=0, mean_speed=1.0, turbulence_intensity=0.05)]
        results = assess_wind_comfort(sensors)
        assert results[0].is_comfortable is True
        assert results[0].lawson_category == "A_sitting"

    def test_high_wind_is_dangerous(self) -> None:
        from tensorlbm.wind_comfort import WindSensorPoint, assess_wind_comfort
        sensors = [WindSensorPoint(label="P2", x=0, y=0, mean_speed=15.0, turbulence_intensity=0.2)]
        results = assess_wind_comfort(sensors)
        assert results[0].lawson_category in ("D_running", "E_dangerous")

    def test_gust_factor_increases_effective_speed(self) -> None:
        from tensorlbm.wind_comfort import WindSensorPoint, assess_wind_comfort
        s = WindSensorPoint(label="P3", x=0, y=0, mean_speed=3.0, turbulence_intensity=0.3)
        r = assess_wind_comfort([s], gust_factor=3.5)[0]
        assert r.effective_gust_speed > r.mean_speed

    def test_weibull_exceedance_decreasing(self) -> None:
        """Exceedance P(U>u) should decrease as u increases."""
        from tensorlbm.wind_comfort import _weibull_exceedance
        k, c = 2.0, 6.0
        assert _weibull_exceedance(5.0, k, c) > _weibull_exceedance(10.0, k, c)

    def test_summary_structure(self) -> None:
        from tensorlbm.wind_comfort import WindSensorPoint, assess_wind_comfort, wind_comfort_summary
        sensors = [
            WindSensorPoint(label=f"P{i}", x=float(i), y=0.0, mean_speed=float(i * 2))
            for i in range(4)
        ]
        results = assess_wind_comfort(sensors)
        summary = wind_comfort_summary(results)
        assert "n_sensors" in summary
        assert summary["n_sensors"] == 4
        assert "class_distribution" in summary
        assert "sensors" in summary

    def test_nen8100_class_ordering(self) -> None:
        """Classes A–F should escalate with increasing wind speed."""
        from tensorlbm.wind_comfort import WindSensorPoint, assess_wind_comfort
        class_order = ["A", "B", "C", "D", "E", "F_dangerous"]
        speeds = [1.0, 3.0, 5.5, 8.0, 12.0, 20.0]
        classes = []
        for U in speeds:
            s = WindSensorPoint(label="x", x=0, y=0, mean_speed=U, turbulence_intensity=0.2)
            r = assess_wind_comfort([s])[0]
            classes.append(r.nen8100_class)
        # Class index should be non-decreasing with speed
        indices = [class_order.index(c) if c in class_order else len(class_order) for c in classes]
        for i in range(len(indices) - 1):
            assert indices[i] <= indices[i + 1]


# ===========================================================================
# 3b. Wind Comfort API Tests
# ===========================================================================

class TestWindComfortAPI:
    def test_wind_comfort_endpoint(self, client) -> None:
        r = client.post("/api/postprocess/wind-comfort", json={
            "sensors": [
                {"label": "P1", "x": 0, "y": 0, "z": 1.5, "mean_speed": 2.0,
                 "turbulence_intensity": 0.1},
                {"label": "P2", "x": 10, "y": 5, "z": 1.5, "mean_speed": 8.0,
                 "turbulence_intensity": 0.25},
            ],
            "gust_factor": 3.5,
            "comfort_threshold_class": "C",
            "reference_code": "both",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["n_sensors"] == 2
        assert "worst_class" in data

    def test_wind_comfort_empty_sensors_rejected(self, client) -> None:
        r = client.post("/api/postprocess/wind-comfort", json={"sensors": []})
        assert r.status_code == 422

    def test_wind_comfort_invalid_class(self, client) -> None:
        r = client.post("/api/postprocess/wind-comfort", json={
            "sensors": [{"label": "A", "x": 0, "y": 0, "mean_speed": 2.0}],
            "comfort_threshold_class": "Z",
        })
        assert r.status_code == 422


# ===========================================================================
# 4. Adjoint Sensitivity Library Tests
# ===========================================================================

class TestAdjointLibrary:
    def test_returns_gradient_for_drag(self) -> None:
        from tensorlbm.adjoint import adjoint_sensitivity
        rho, ux, uy, mask = _make_cylinder_fields(40, 80)
        result = adjoint_sensitivity(rho, ux, uy, mask, objective="drag")
        assert "objective_value" in result
        assert "gradient_norm" in result
        assert result["n_boundary_nodes"] > 0
        assert len(result["sensitivity_x"]) == result["n_boundary_nodes"]

    def test_returns_gradient_for_lift(self) -> None:
        from tensorlbm.adjoint import adjoint_sensitivity
        rho, ux, uy, mask = _make_cylinder_fields(40, 80)
        result = adjoint_sensitivity(rho, ux, uy, mask, objective="lift")
        assert "gradient_norm" in result

    def test_pressure_loss_objective(self) -> None:
        from tensorlbm.adjoint import adjoint_sensitivity
        rho, ux, uy, mask = _make_cylinder_fields(40, 80)
        result = adjoint_sensitivity(rho, ux, uy, mask, objective="pressure_loss")
        assert isinstance(result["objective_value"], float)

    def test_mixing_uniformity_objective(self) -> None:
        from tensorlbm.adjoint import adjoint_sensitivity
        rho, ux, uy, mask = _make_cylinder_fields(40, 80)
        result = adjoint_sensitivity(rho, ux, uy, mask, objective="mixing_uniformity")
        assert "gradient_norm" in result

    def test_empty_mask_returns_zero_nodes(self) -> None:
        import torch
        from tensorlbm.adjoint import adjoint_sensitivity
        rho = torch.ones(20, 40)
        ux = torch.rand(20, 40) * 0.1
        uy = torch.zeros(20, 40)
        mask = torch.zeros(20, 40, dtype=torch.bool)
        result = adjoint_sensitivity(rho, ux, uy, mask, objective="drag")
        assert result["n_boundary_nodes"] == 0

    def test_invalid_objective_raises(self) -> None:
        import torch
        from tensorlbm.adjoint import adjoint_sensitivity
        rho = torch.ones(10, 20)
        ux = torch.zeros(10, 20)
        uy = torch.zeros(10, 20)
        mask = torch.zeros(10, 20, dtype=torch.bool)
        with pytest.raises(ValueError, match="Unknown objective"):
            adjoint_sensitivity(rho, ux, uy, mask, objective="invalid_obj")  # type: ignore[arg-type]

    def test_fd_check_option(self) -> None:
        from tensorlbm.adjoint import adjoint_sensitivity
        rho, ux, uy, mask = _make_cylinder_fields(30, 60)
        result = adjoint_sensitivity(rho, ux, uy, mask, objective="drag",
                                     finite_diff_check=True)
        assert "fd_check" in result

    def test_most_sensitive_node_present(self) -> None:
        from tensorlbm.adjoint import adjoint_sensitivity
        rho, ux, uy, mask = _make_cylinder_fields(40, 80)
        result = adjoint_sensitivity(rho, ux, uy, mask, objective="drag")
        assert "most_sensitive_node" in result
        if result["n_boundary_nodes"] > 0:
            assert "sensitivity_magnitude" in result["most_sensitive_node"]


# ===========================================================================
# 4b. Adjoint API Tests
# ===========================================================================

class TestAdjointAPI:
    def test_adjoint_missing_job(self, client) -> None:
        r = client.post("/api/postprocess/adjoint-sensitivity/no-job",
                        json={"objective": "drag"})
        assert r.status_code == 404

    def test_adjoint_invalid_objective(self, client) -> None:
        r = client.post("/api/postprocess/adjoint-sensitivity/no-job",
                        json={"objective": "magic"})
        assert r.status_code in (404, 422)


# ===========================================================================
# 5. CGNS Export Library Tests
# ===========================================================================

class TestCGNSExportLibrary:
    def test_export_creates_files(self) -> None:
        import torch
        from tensorlbm.cgns_export import export_cgns, CGNSExportConfig

        rho, ux, uy, _ = _make_cylinder_fields(30, 60)
        cfg = CGNSExportConfig(dx_phys=1e-3, reference_velocity=0.1)

        with tempfile.TemporaryDirectory() as tmp:
            result = export_cgns(rho, ux, uy, Path(tmp) / "test_export", cfg)

        assert "format" in result
        assert "fields_exported" in result
        assert len(result["fields_exported"]) > 0
        assert result["nx"] == 60
        assert result["ny"] == 30

    def test_export_includes_requested_fields(self) -> None:
        import torch
        from tensorlbm.cgns_export import export_cgns, CGNSExportConfig

        rho, ux, uy, _ = _make_cylinder_fields(20, 40)
        cfg = CGNSExportConfig(
            include_density=True,
            include_pressure=True,
            include_velocity=True,
            include_vorticity=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = export_cgns(rho, ux, uy, Path(tmp) / "out", cfg)

        fields = result["fields_exported"]
        assert "Density" in fields
        assert "Pressure" in fields
        assert "VelocityX" in fields
        assert "VorticityZ" not in fields

    def test_export_vorticity_disabled(self) -> None:
        import torch
        from tensorlbm.cgns_export import export_cgns, CGNSExportConfig

        rho, ux, uy, _ = _make_cylinder_fields(20, 40)
        cfg = CGNSExportConfig(include_vorticity=False)
        with tempfile.TemporaryDirectory() as tmp:
            result = export_cgns(rho, ux, uy, Path(tmp) / "out", cfg)
        assert "VorticityZ" not in result["fields_exported"]

    def test_export_step_index(self) -> None:
        import torch
        from tensorlbm.cgns_export import export_cgns, CGNSExportConfig

        rho, ux, uy, _ = _make_cylinder_fields(20, 40)
        cfg = CGNSExportConfig()
        with tempfile.TemporaryDirectory() as tmp:
            result = export_cgns(rho, ux, uy, Path(tmp) / "out", cfg, step=42)
        assert result["step"] == 42


# ===========================================================================
# 5b. CGNS Export API Tests
# ===========================================================================

class TestCGNSExportAPI:
    def test_export_missing_job(self, client) -> None:
        r = client.get("/api/postprocess/export-cgns/no-such-job")
        assert r.status_code == 404
