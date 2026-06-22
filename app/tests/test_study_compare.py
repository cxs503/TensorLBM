"""Tests for the study-group comparison endpoint.

``GET /api/postprocess/study-compare/{study_group}``
"""
from __future__ import annotations

import pytest


class TestStudyCompare:
    def test_no_jobs_returns_404(self, client):
        """An empty study group should return 404."""
        r = client.get("/api/postprocess/study-compare/nonexistent_group_xyz")
        assert r.status_code == 404

    def test_group_with_jobs_returns_200(self, client, job_manager):
        """After injecting a job with study metadata, the endpoint should return it."""
        import uuid
        from backend.job_manager import Job, JobStatus  # type: ignore[import-not-found]

        jm = job_manager
        group = f"test_group_{uuid.uuid4().hex[:8]}"
        job_id = str(uuid.uuid4())

        # Inject a synthetic completed job with study metadata
        job = Job(
            job_id=job_id,
            name="Test study job",
            job_type="cylinder_flow",
            config={
                "re": 100.0,
                "study": {"group": group, "design_point": {"re": 100.0}},
            },
        )
        job.status = JobStatus.COMPLETED
        job.result = {"drag_coefficient": 1.5, "strouhal": 0.21}
        with jm._jobs_lock:
            jm._jobs[job_id] = job

        try:
            r = client.get(f"/api/postprocess/study-compare/{group}")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["study_group"] == group
            assert body["n_total"] >= 1
            assert body["n_completed"] >= 1
            assert any(j["job_id"] == job_id for j in body["jobs"])
            # Metrics extracted
            first = next(j for j in body["jobs"] if j["job_id"] == job_id)
            assert "drag_coefficient" in first["metrics"]
            assert "strouhal" in first["metrics"]
        finally:
            with jm._jobs_lock:
                jm._jobs.pop(job_id, None)

    def test_multiple_jobs_metric_summary(self, client, job_manager):
        """Two jobs with numeric results produce a metric_summary with best_job_id."""
        import uuid
        from backend.job_manager import Job, JobStatus  # type: ignore[import-not-found]

        jm = job_manager
        group = f"test_group_{uuid.uuid4().hex[:8]}"
        ids = []
        cds = [1.2, 0.8]

        for i, cd in enumerate(cds):
            job_id = str(uuid.uuid4())
            ids.append(job_id)
            job = Job(
                job_id=job_id,
                name=f"Study job {i}",
                job_type="cylinder_flow",
                config={
                    "re": 50.0 * (i + 1),
                    "study": {
                        "group": group,
                        "design_point": {"re": 50.0 * (i + 1)},
                    },
                },
            )
            job.status = JobStatus.COMPLETED
            job.result = {"drag_coefficient": cd}
            with jm._jobs_lock:
                jm._jobs[job_id] = job

        try:
            r = client.get(f"/api/postprocess/study-compare/{group}")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["n_completed"] == 2
            summary = body["metric_summary"]
            assert "drag_coefficient" in summary
            assert summary["drag_coefficient"]["min"] == pytest.approx(0.8)
            assert summary["drag_coefficient"]["max"] == pytest.approx(1.2)
            # best_job_id should be the job with lowest drag (0.8)
            assert summary["drag_coefficient"]["best_job_id"] == ids[1]
        finally:
            with jm._jobs_lock:
                for jid in ids:
                    jm._jobs.pop(jid, None)

    def test_partial_completion_counts_correctly(self, client, job_manager):
        """Running and failed jobs are included in n_total but not n_completed."""
        import uuid
        from backend.job_manager import Job, JobStatus  # type: ignore[import-not-found]

        jm = job_manager
        group = f"test_group_{uuid.uuid4().hex[:8]}"
        ids = []

        statuses = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.RUNNING]
        for status in statuses:
            job_id = str(uuid.uuid4())
            ids.append(job_id)
            job = Job(
                job_id=job_id,
                name="test",
                job_type="cylinder_flow",
                config={"study": {"group": group, "design_point": {}}},
            )
            job.status = status
            with jm._jobs_lock:
                jm._jobs[job_id] = job

        try:
            r = client.get(f"/api/postprocess/study-compare/{group}")
            assert r.status_code == 200
            body = r.json()
            assert body["n_total"] == 3
            assert body["n_completed"] == 1
        finally:
            with jm._jobs_lock:
                for jid in ids:
                    jm._jobs.pop(jid, None)

    def test_time_series_extracted_as_final_and_mean(self, client, job_manager):
        """Time-series results should produce _final and _mean metric keys."""
        import uuid
        from backend.job_manager import Job, JobStatus  # type: ignore[import-not-found]

        jm = job_manager
        group = f"test_group_{uuid.uuid4().hex[:8]}"
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            name="time series job",
            job_type="cylinder_flow",
            config={"study": {"group": group, "design_point": {}}},
        )
        job.status = JobStatus.COMPLETED
        job.result = {"saturation_series": [0.1, 0.3, 0.5, 0.6]}
        with jm._jobs_lock:
            jm._jobs[job_id] = job

        try:
            r = client.get(f"/api/postprocess/study-compare/{group}")
            assert r.status_code == 200
            body = r.json()
            first = body["jobs"][0]
            assert "saturation_series_final" in first["metrics"]
            assert "saturation_series_mean" in first["metrics"]
            assert first["metrics"]["saturation_series_final"] == pytest.approx(0.6)
        finally:
            with jm._jobs_lock:
                jm._jobs.pop(job_id, None)
