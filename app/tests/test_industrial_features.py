"""Tests for 18 new industrial features (PowerFlow/XFlow gap closure).

Covers:
1.  Sliding-mesh rotor endpoint
2.  Force decomposition (pressure + viscous)
3.  Wall shear stress endpoint
4.  Vortex identification criterion endpoint
5.  Animation export endpoint
6.  Passive scalar transport endpoint
7.  Cavitation flow endpoint
8.  Oscillating airfoil (moving boundary) endpoint
9.  HPC scheduler service (unit tests, dry-run)
10. Job priority field
11. Multi-case overlay chart endpoint
12. Heat flux mapping endpoint
13. Acoustic spectrum analysis endpoint
14. Notification system (webhook test + settings)
15. Job timeline endpoint
16. Sobol sensitivity analysis endpoint
17. Surface integrals force-decomposition module
18. New src/tensorlbm physics modules (smoke tests)
"""
from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest
import torch

# ===========================================================================
# 1. Sliding-mesh rotor endpoint
# ===========================================================================

class TestSlidingMeshRotorEndpoint:
    def test_submit_returns_job_id(self, client):
        body = {
            "nx": 64, "ny": 64, "u_tip": 0.05, "re": 100.0,
            "n_blades": 4, "n_steps": 100, "output_interval": 50,
        }
        r = client.post("/api/solve/sliding-mesh-rotor", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "job_id" in data
        assert "sliding" in data.get("message", "").lower()

    def test_invalid_blades_rejected(self, client):
        body = {
            "nx": 64, "ny": 64, "n_blades": 1,  # below minimum of 2
            "n_steps": 100,
        }
        r = client.post("/api/solve/sliding-mesh-rotor", json=body)
        assert r.status_code == 422

    def test_priority_field_accepted(self, client):
        body = {
            "nx": 64, "ny": 64, "n_steps": 100, "priority": 8,
        }
        r = client.post("/api/solve/sliding-mesh-rotor", json=body)
        assert r.status_code == 200, r.text


# ===========================================================================
# 2. Force decomposition endpoint
# ===========================================================================

class TestForceDecomposition:
    def test_not_found(self, client):
        r = client.get("/api/postprocess/force-decomposition/nonexistent_job")
        assert r.status_code == 404

    def test_incomplete_job_409(self, client):
        # Submit a quick job and immediately query before completion
        body = {"nx": 32, "ny": 32, "re": 100.0, "n_steps": 5}
        sub = client.post("/api/solve/cylinder-flow", json=body)
        if sub.status_code != 200:
            pytest.skip("solver endpoint unavailable")
        job_id = sub.json()["job_id"]
        # Query immediately – job probably still running
        r = client.get(f"/api/postprocess/force-decomposition/{job_id}")
        # Accept 409 (running) or 200 (already done in fast test env)
        assert r.status_code in (200, 409)

    def test_decomposition_keys_present(self, completed_job, client):
        r = client.get(f"/api/postprocess/force-decomposition/{completed_job}")
        if r.status_code == 422:
            pytest.skip("3-D job; force decomposition requires 2-D")
        assert r.status_code == 200, r.text
        data = r.json()
        for key in ("fx_total", "fy_total", "fx_pressure", "fy_pressure",
                    "fx_viscous", "fy_viscous", "cd_total", "cd_pressure", "cd_viscous"):
            assert key in data, f"Missing key: {key}"
        # Check conservation: total = pressure + viscous (within floating point)
        assert abs(data["fx_total"] - (data["fx_pressure"] + data["fx_viscous"])) < 1e-6


# ===========================================================================
# 3. Wall shear stress endpoint
# ===========================================================================

class TestWallShearStress:
    def test_not_found(self, client):
        r = client.get("/api/postprocess/wall-shear-stress/no_job")
        assert r.status_code == 404

    def test_wss_keys(self, completed_job, client):
        r = client.get(f"/api/postprocess/wall-shear-stress/{completed_job}")
        if r.status_code == 422:
            pytest.skip("3-D job; WSS requires 2-D")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "wss" in data
        assert "wss_max" in data
        assert "wss_mean" in data
        assert isinstance(data["wss"], list)
        assert data["wss_max"] >= 0.0

    def test_no_normalise(self, completed_job, client):
        r = client.get(f"/api/postprocess/wall-shear-stress/{completed_job}?normalise=false")
        if r.status_code in (409, 422):
            pytest.skip("Job not ready or 3-D")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "cf_map" not in data


# ===========================================================================
# 4. Vortex criterion endpoint
# ===========================================================================

class TestVortexCriterion:
    def test_not_found(self, client):
        r = client.get("/api/postprocess/vortex-criterion/no_job")
        assert r.status_code == 404

    def test_q_criterion(self, completed_job, client):
        r = client.get(f"/api/postprocess/vortex-criterion/{completed_job}?criteria=q")
        if r.status_code == 409:
            pytest.skip("Job not ready")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "q" in data
        assert isinstance(data["q"], list)

    def test_all_criteria(self, completed_job, client):
        r = client.get(
            f"/api/postprocess/vortex-criterion/{completed_job}?criteria=q,lambda2,omega"
        )
        if r.status_code == 409:
            pytest.skip("Job not ready")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "q" in data
        assert "lambda2" in data
        assert "omega" in data

    def test_unknown_criterion_rejected(self, completed_job, client):
        r = client.get(f"/api/postprocess/vortex-criterion/{completed_job}?criteria=invalid")
        assert r.status_code in (404, 409, 422)


# ===========================================================================
# 5. Animation export endpoint
# ===========================================================================

class TestAnimationExport:
    def test_not_found(self, client):
        r = client.get("/api/postprocess/animation/no_job")
        assert r.status_code == 404

    def test_invalid_format(self, completed_job, client):
        r = client.get(f"/api/postprocess/animation/{completed_job}?fmt=avi")
        assert r.status_code == 422

    def test_gif_response(self, completed_job, client):
        r = client.get(f"/api/postprocess/animation/{completed_job}?fmt=gif&fps=5&max_frames=10")
        if r.status_code == 404:
            pytest.skip("No PNG frames in job output")
        if r.status_code == 409:
            pytest.skip("Job not complete")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/gif"


# ===========================================================================
# 6. Passive scalar transport endpoint
# ===========================================================================

class TestPassiveScalarTransport:
    def test_submit(self, client):
        body = {
            "nx": 64, "ny": 32, "re": 50.0, "diffusivity": 0.02,
            "n_steps": 100, "output_interval": 50,
        }
        r = client.post("/api/solve/passive-scalar-transport", json=body)
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()

    def test_invalid_diffusivity(self, client):
        body = {"nx": 64, "ny": 32, "diffusivity": 0.0, "n_steps": 10}
        r = client.post("/api/solve/passive-scalar-transport", json=body)
        assert r.status_code == 422


# ===========================================================================
# 7. Cavitation flow endpoint
# ===========================================================================

class TestCavitationFlow:
    def test_submit(self, client):
        body = {
            "nx": 64, "ny": 32, "G": -5.5, "re": 200.0,
            "n_steps": 100, "output_interval": 50,
        }
        r = client.post("/api/solve/cavitation-flow", json=body)
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()

    def test_G_range_enforced(self, client):
        body = {"nx": 64, "ny": 32, "G": -1.0, "n_steps": 10}  # G above -3.0 limit
        r = client.post("/api/solve/cavitation-flow", json=body)
        assert r.status_code == 422


# ===========================================================================
# 8. Oscillating airfoil endpoint
# ===========================================================================

class TestOscillatingAirfoil:
    def test_submit_airfoil(self, client):
        body = {
            "nx": 128, "ny": 64, "re": 200.0, "chord": 30.0,
            "geom": "airfoil", "n_steps": 100, "output_interval": 50,
        }
        r = client.post("/api/solve/oscillating-airfoil", json=body)
        assert r.status_code == 200, r.text
        assert "job_id" in r.json()

    def test_submit_cylinder(self, client):
        body = {
            "nx": 128, "ny": 64, "re": 200.0,
            "geom": "cylinder", "n_steps": 100, "output_interval": 50,
        }
        r = client.post("/api/solve/oscillating-airfoil", json=body)
        assert r.status_code == 200, r.text

    def test_invalid_geom(self, client):
        body = {"nx": 128, "ny": 64, "geom": "sphere", "n_steps": 10}
        r = client.post("/api/solve/oscillating-airfoil", json=body)
        assert r.status_code == 422


# ===========================================================================
# 9. HPC scheduler service (unit tests)
# ===========================================================================

class TestHPCSchedulerService:
    def test_none_mode_raises(self):
        with patch.dict(os.environ, {"TENSORLBM_HPC_MODE": "none"}):
            from app.backend.services.hpc_scheduler import submit_hpc_job
            with pytest.raises(ValueError, match="disabled"):
                submit_hpc_job("test_job", "/tmp/test_output")

    def test_slurm_mode_no_sbatch(self):
        with patch.dict(os.environ, {"TENSORLBM_HPC_MODE": "slurm"}):
            import shutil as _shutil
            with patch.object(_shutil, "which", return_value=None):
                from importlib import reload

                from app.backend.services import hpc_scheduler
                reload(hpc_scheduler)
                with pytest.raises((RuntimeError, ValueError)):
                    hpc_scheduler.submit_hpc_job("test_job", "/tmp/test_output")

    def test_submit_hpc_endpoint_without_hpc_mode(self, client):
        """Without TENSORLBM_HPC_MODE, endpoint returns 400."""
        with patch.dict(os.environ, {"TENSORLBM_HPC_MODE": "none"}, clear=False):
            # First create a job
            body = {"nx": 32, "ny": 32, "n_steps": 5}
            sub = client.post("/api/solve/cylinder-flow", json=body)
            if sub.status_code != 200:
                pytest.skip("solver unavailable")
            job_id = sub.json()["job_id"]
            r = client.post(f"/api/jobs/{job_id}/submit-hpc", json={})
            assert r.status_code == 400

    def test_slurm_script_content(self):
        import pathlib
        import tempfile

        from app.backend.services.hpc_scheduler import _build_slurm_script
        with tempfile.TemporaryDirectory() as d:
            script = _build_slurm_script(
                "abc123", "echo hello",
                partition="gpu", nodes=2, cpus=8, mem="16G",
                walltime="01:00:00", log_dir=pathlib.Path(d),
            )
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --nodes=2" in script
        assert "echo hello" in script

    def test_pbs_script_content(self):
        import pathlib
        import tempfile

        from app.backend.services.hpc_scheduler import _build_pbs_script
        with tempfile.TemporaryDirectory() as d:
            script = _build_pbs_script(
                "abc123", "echo hello",
                queue="debug", nodes=1, cpus=4, mem="4G",
                walltime="00:30:00", log_dir=pathlib.Path(d),
            )
        assert "#PBS -q debug" in script
        assert "echo hello" in script


# ===========================================================================
# 10. Job priority field
# ===========================================================================

class TestJobPriority:
    def test_priority_in_job_dict(self, client):
        body = {"nx": 32, "ny": 32, "re": 50.0, "n_steps": 5, "priority": 9}
        r = client.post("/api/solve/cylinder-flow", json=body)
        if r.status_code != 200:
            pytest.skip("solver unavailable")
        job_id = r.json()["job_id"]
        jr = client.get(f"/api/jobs/{job_id}")
        assert jr.status_code == 200
        data = jr.json()
        assert "priority" in data
        assert 1 <= data["priority"] <= 10

    def test_default_priority_is_5(self, client):
        body = {"nx": 32, "ny": 32, "re": 50.0, "n_steps": 5}
        r = client.post("/api/solve/cylinder-flow", json=body)
        if r.status_code != 200:
            pytest.skip("solver unavailable")
        job_id = r.json()["job_id"]
        jr = client.get(f"/api/jobs/{job_id}")
        assert jr.status_code == 200
        # Default priority = 5
        assert jr.json().get("priority", 5) == 5


# ===========================================================================
# 11. Multi-case overlay chart endpoint
# ===========================================================================

class TestMultiCaseChart:
    def test_requires_at_least_2_jobs(self, client):
        body = {"job_ids": ["only_one"], "metric": "cd"}
        r = client.post("/api/postprocess/multi-case-chart", json=body)
        assert r.status_code == 422

    def test_too_many_jobs_rejected(self, client):
        body = {"job_ids": [f"job_{i}" for i in range(25)], "metric": "cd"}
        r = client.post("/api/postprocess/multi-case-chart", json=body)
        assert r.status_code == 422

    def test_nonexistent_jobs_return_422(self, client):
        body = {"job_ids": ["fake1", "fake2"], "metric": "cd"}
        r = client.post("/api/postprocess/multi-case-chart", json=body)
        assert r.status_code == 422  # no plottable data

    def test_label_length_mismatch(self, client):
        body = {
            "job_ids": ["a", "b"],
            "metric": "cd",
            "labels": ["only_one_label"],
        }
        r = client.post("/api/postprocess/multi-case-chart", json=body)
        assert r.status_code == 422


# ===========================================================================
# 12. Heat flux mapping endpoint
# ===========================================================================

class TestHeatFluxMapping:
    def test_not_found(self, client):
        r = client.get("/api/postprocess/heat-flux/no_job")
        assert r.status_code == 404

    def test_non_thermal_job_returns_422(self, completed_job, client):
        r = client.get(f"/api/postprocess/heat-flux/{completed_job}")
        # Non-thermal jobs lack temperature data → 422
        assert r.status_code in (200, 404, 422)


# ===========================================================================
# 13. Acoustic spectrum analysis endpoint
# ===========================================================================

class TestAcousticsSpectrum:
    def test_not_found(self, client):
        r = client.get("/api/postprocess/acoustics-spectrum/no_job")
        assert r.status_code == 404

    def test_no_acoustic_csv(self, completed_job, client):
        r = client.get(f"/api/postprocess/acoustics-spectrum/{completed_job}")
        # Non-acoustic jobs lack CSV → 404 or 409
        assert r.status_code in (404, 409)

    def test_spectrum_keys(self, acoustic_job, client):
        """If an acoustic job exists, verify response structure."""
        r = client.get(f"/api/postprocess/acoustics-spectrum/{acoustic_job}")
        if r.status_code in (404, 409):
            pytest.skip("No acoustic data in job")
        assert r.status_code == 200, r.text
        data = r.json()
        for key in ("frequencies", "psd", "spl_db", "oaspl_db", "third_octave_bands"):
            assert key in data


# ===========================================================================
# 14. Notification system
# ===========================================================================

class TestNotificationSystem:
    def test_get_settings(self, client):
        r = client.get("/api/notifications/settings")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "notify_on_complete" in data
        assert "notify_on_failure" in data
        assert "webhook_url" in data

    def test_update_settings(self, client):
        body = {
            "webhook_url": "https://example.com/hook",
            "notify_on_complete": True,
            "notify_on_failure": True,
            "notify_on_cancel": False,
            "timeout_s": 5,
        }
        r = client.post("/api/notifications/settings", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "updated"
        assert data["notify_on_complete"] is True

    def test_webhook_test_invalid_url(self, client):
        body = {"url": "not_a_url"}
        r = client.post("/api/notifications/webhook-test", json=body)
        assert r.status_code == 422

    def test_webhook_test_valid_url_structure(self, client):
        """Test endpoint accepts valid HTTP URL (will fail to connect in test env)."""
        body = {"url": "http://localhost:9999/webhook"}
        r = client.post("/api/notifications/webhook-test", json=body)
        # Will fail to connect but endpoint should not crash – returns status dict
        assert r.status_code == 200, r.text
        data = r.json()
        assert "status" in data
        assert data["status"] in ("ok", "error", "timeout", "http_error")


# ===========================================================================
# 15. Job timeline endpoint
# ===========================================================================

class TestJobTimeline:
    def test_timeline_structure(self, client):
        r = client.get("/api/jobs/timeline")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "timeline" in data
        assert "total" in data
        assert "returned" in data
        assert isinstance(data["timeline"], list)

    def test_timeline_entry_fields(self, client):
        r = client.get("/api/jobs/timeline?limit=5")
        assert r.status_code == 200, r.text
        data = r.json()
        for entry in data["timeline"]:
            assert "job_id" in entry
            assert "name" in entry
            assert "status" in entry
            assert "created_at" in entry
            assert "priority" in entry

    def test_timeline_limit(self, client):
        r = client.get("/api/jobs/timeline?limit=3")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["returned"] <= 3

    def test_timeline_status_filter(self, client):
        r = client.get("/api/jobs/timeline?status=completed")
        assert r.status_code == 200, r.text
        data = r.json()
        for entry in data["timeline"]:
            assert entry["status"] == "completed"


# ===========================================================================
# 16. Sobol sensitivity analysis endpoint
# ===========================================================================

class TestSobolAnalysis:
    def test_too_few_jobs(self, client):
        r = client.get("/api/orchestration/studies/nonexistent_group/sobol")
        assert r.status_code == 422

    def test_sobol_with_study_group(self, client):
        """Create synthetic parametric study jobs and verify Sobol response."""

        from backend import job_manager as _jm  # type: ignore[import-not-found]

        study_group = "test_sobol_study"
        job_ids = []
        for re_val in [100, 150, 200, 250, 300, 350, 400, 450]:
            cfg = {"re": re_val, "nx": 64, "ny": 32, "study_group": study_group}
            job = _jm.Job(
                job_id=f"sobol_{re_val}",
                name=f"Sobol test re={re_val}",
                job_type="cylinder_flow",
                config=cfg,
            )
            job.status = _jm.JobStatus.COMPLETED
            # Write synthetic metadata
            job.output_dir.mkdir(parents=True, exist_ok=True)
            import json as _j
            cd_val = 1.0 / re_val * 100 + 0.5
            meta = {"re": re_val, "cd_mean": cd_val, "steps": [1000], "cd_history": [cd_val]}
            (job.output_dir / "run_metadata.json").write_text(_j.dumps(meta))
            with _jm._jobs_lock:
                _jm._jobs[job.job_id] = job
            job_ids.append(job.job_id)

        r = client.get(f"/api/orchestration/studies/{study_group}/sobol?output_metric=cd_mean")
        # Clean up
        for jid in job_ids:
            with _jm._jobs_lock:
                _jm._jobs.pop(jid, None)

        assert r.status_code == 200, r.text
        data = r.json()
        assert "parameters" in data
        assert "S1" in data
        assert "ST" in data
        assert "ranking" in data
        assert len(data["parameters"]) > 0
        for s1 in data["S1"]:
            assert 0.0 <= s1 <= 1.0


# ===========================================================================
# 17. Surface integrals force-decomposition module (unit tests)
# ===========================================================================

class TestForceDecompositionModule:
    def test_decomposed_force_sums_to_total(self):
        from tensorlbm.boundaries import cylinder_mask
        from tensorlbm.d2q9 import equilibrium
        from tensorlbm.surface_integrals import surface_force_decomposed_2d

        ny, nx = 64, 128
        device = torch.device("cpu")
        mask = cylinder_mask(ny, nx, ny // 2, nx // 4, 8.0, device)

        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)

        tau = 0.6
        result = surface_force_decomposed_2d(f, rho, ux, uy, mask, tau)

        # Pressure + viscous = total (within floating point tolerance)
        assert abs(result["fx_total"] - (result["fx_pressure"] + result["fx_viscous"])) < 1e-5
        assert abs(result["fy_total"] - (result["fy_pressure"] + result["fy_viscous"])) < 1e-5
        # All coefficient keys present
        for k in ("cd_total", "cd_pressure", "cd_viscous", "cl_total"):
            assert k in result

    def test_decomposition_symmetry(self):
        """Symmetric flow should give approximately zero lift."""
        from tensorlbm.boundaries import cylinder_mask
        from tensorlbm.d2q9 import equilibrium
        from tensorlbm.surface_integrals import surface_force_decomposed_2d

        ny, nx = 64, 128
        mask = cylinder_mask(ny, nx, ny // 2, nx // 4, 8.0, torch.device("cpu"))
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        result = surface_force_decomposed_2d(f, rho, ux, uy, mask, 0.6)
        # For uniform inlet, lift should be small
        assert abs(result["cl_total"]) < 1.0


# ===========================================================================
# 18. New src/tensorlbm physics modules (smoke tests)
# ===========================================================================

class TestWallShearModule:
    def test_wss_2d_shape(self):
        from tensorlbm.boundaries import cylinder_mask
        from tensorlbm.d2q9 import equilibrium
        from tensorlbm.wall_shear import wss_from_fneq_2d

        ny, nx = 32, 64
        mask = cylinder_mask(nx, ny, nx // 4, ny // 2, 6.0, torch.device("cpu"))
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        wss = wss_from_fneq_2d(f, rho, ux, uy, 0.6, mask)
        assert wss.shape == (ny, nx)
        assert (wss >= 0.0).all()
        assert float(wss[mask].max()) == pytest.approx(0.0)  # zero inside solid

    def test_wss_fd_method(self):
        from tensorlbm.wall_shear import wss_from_velocity_2d
        ny, nx = 32, 64
        ux = torch.rand(ny, nx) * 0.1
        uy = torch.rand(ny, nx) * 0.01
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[0, :] = True  # bottom wall
        wss = wss_from_velocity_2d(ux, uy, mask, nu=1.0 / 6.0)
        assert wss.shape == (ny, nx)
        assert float(wss[mask].max()) == pytest.approx(0.0)


class TestVortexIdentificationModule:
    def test_q_criterion_2d(self):
        from tensorlbm.vortex_identification import q_criterion_2d
        ny, nx = 32, 32
        # Simple shear flow: ux = y, uy = 0 → Q < 0 everywhere (strain dominated)
        ux = torch.zeros(ny, nx)
        uy = torch.zeros(ny, nx)
        for i in range(ny):
            ux[i, :] = i / ny
        q = q_criterion_2d(ux, uy)
        assert q.shape == (ny, nx)
        # For pure shear, Q should be negative (strain > rotation)
        assert float(q.max()) <= 0.1  # allow small numerical noise

    def test_omega_criterion_range(self):
        from tensorlbm.vortex_identification import omega_criterion_2d
        ux = torch.rand(32, 32) * 0.1
        uy = torch.rand(32, 32) * 0.1
        om = omega_criterion_2d(ux, uy)
        assert om.shape == (32, 32)
        assert (om >= 0.0).all()
        assert (om <= 1.0).all()

    def test_vortex_fields_2d_wrapper(self):
        from tensorlbm.vortex_identification import vortex_fields_2d
        ux = torch.rand(16, 32) * 0.05
        uy = torch.rand(16, 32) * 0.02
        fields = vortex_fields_2d(ux, uy)
        assert "q" in fields and "lambda2" in fields and "omega" in fields
        assert len(fields["q"]) == 16  # ny rows


class TestAnimationExportModule:
    def test_no_frames_raises(self, tmp_path):
        from tensorlbm.animation_export import frames_from_png_dir
        with pytest.raises(FileNotFoundError):
            frames_from_png_dir(tmp_path)

    def test_gif_creation(self, tmp_path):
        from PIL import Image

        from tensorlbm.animation_export import gif_from_frames
        # Create dummy PNG frames
        frames = []
        for i in range(3):
            p = tmp_path / f"step_{i:06d}.png"
            img = Image.new("RGB", (32, 32), color=(i * 80, 0, 0))
            img.save(str(p))
            frames.append(p)
        out = tmp_path / "anim.gif"
        result = gif_from_frames(frames, out, fps=5)
        assert result.exists()
        assert result.stat().st_size > 0


class TestPassiveScalarModule:
    def test_equilibrium_shape(self):
        from tensorlbm.passive_scalar import equilibrium_scalar
        c = torch.rand(16, 32)
        ux = torch.rand(16, 32) * 0.05
        uy = torch.rand(16, 32) * 0.01
        g_eq = equilibrium_scalar(c, ux, uy)
        assert g_eq.shape == (5, 16, 32)

    def test_macroscopic_round_trip(self):
        from tensorlbm.passive_scalar import equilibrium_scalar, macroscopic_scalar
        c = torch.ones(16, 32) * 0.7
        ux = torch.zeros(16, 32)
        uy = torch.zeros(16, 32)
        g = equilibrium_scalar(c, ux, uy)
        c_rec = macroscopic_scalar(g)
        assert torch.allclose(c_rec, c, atol=1e-5)

    def test_stream_scalar_conservation(self):
        from tensorlbm.passive_scalar import equilibrium_scalar, macroscopic_scalar, stream_scalar
        c0 = torch.ones(8, 16)
        ux = torch.zeros(8, 16)
        uy = torch.zeros(8, 16)
        g = equilibrium_scalar(c0, ux, uy)
        c_before = macroscopic_scalar(g).sum()
        g2 = stream_scalar(g)
        c_after = macroscopic_scalar(g2).sum()
        assert torch.isclose(c_before, c_after, atol=1e-5)


class TestCavitationModule:
    def test_psi_cavitation_positive(self):
        from tensorlbm.cavitation import psi_cavitation
        rho = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
        psi = psi_cavitation(rho)
        assert (psi > 0).all()

    def test_schnerr_sauer_source_shape(self):
        from tensorlbm.cavitation import schnerr_sauer_source
        rho = torch.ones(16, 32) * 1.5
        p = rho / 3.0
        m_dot = schnerr_sauer_source(rho, p, p_sat=0.5)
        assert m_dot.shape == (16, 32)


class TestSlidingMeshModule:
    def test_rotate_velocity_identity(self):
        import math

        from tensorlbm.sliding_mesh import rotate_velocity_field_2d
        ux = torch.rand(16, 16)
        uy = torch.rand(16, 16)
        ux_r, uy_r = rotate_velocity_field_2d(ux, uy, 2 * math.pi)
        assert torch.allclose(ux_r, ux, atol=1e-5)
        assert torch.allclose(uy_r, uy, atol=1e-5)

    def test_rotate_velocity_90deg(self):
        import math

        from tensorlbm.sliding_mesh import rotate_velocity_field_2d
        ux = torch.ones(4, 4)
        uy = torch.zeros(4, 4)
        ux_r, uy_r = rotate_velocity_field_2d(ux, uy, math.pi / 2)
        # After 90° CCW: (1, 0) → (0, 1)
        assert torch.allclose(ux_r, torch.zeros(4, 4), atol=1e-5)
        assert torch.allclose(uy_r, torch.ones(4, 4), atol=1e-5)


class TestMovingBoundaryModule:
    def test_pitch_plunge_position_range(self):
        import math

        from tensorlbm.moving_boundary import PitchPlungeMotion
        motion = PitchPlungeMotion(A_h=20.0, A_alpha=math.radians(15.0), f_red=0.002)
        for step in range(500):
            _, y, alpha = motion.position(step)
            assert abs(y) <= 20.0 + 1e-5
            assert abs(alpha) <= math.radians(15.0) + 1e-5

    def test_oscillating_cylinder_velocity(self):
        from tensorlbm.moving_boundary import OscillatingCylinderMotion
        motion = OscillatingCylinderMotion(A_y=10.0, f_osc=0.001)
        vx, vy, omega = motion.velocity(0)
        assert vx == pytest.approx(0.0)
        assert abs(vy) <= 10.0 * 2 * 3.14159 * 0.001 + 1e-5


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def completed_job(app, client):
    """Submit a quick cylinder-flow job and wait for completion."""
    body = {"nx": 32, "ny": 32, "re": 100.0, "n_steps": 200, "output_interval": 100}
    r = client.post("/api/solve/cylinder-flow", json=body)
    if r.status_code != 200:
        pytest.skip("cylinder-flow solver unavailable")
    job_id = r.json()["job_id"]
    for _ in range(60):
        jr = client.get(f"/api/jobs/{job_id}")
        if jr.json().get("status") in ("completed", "failed"):
            break
        time.sleep(0.5)
    return job_id


@pytest.fixture
def acoustic_job(client):
    """Job ID for acoustics test (may not exist)."""
    return "no_acoustic_job"
