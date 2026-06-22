"""Tests for the TensorLBM Reports API and Convergence endpoint."""
from __future__ import annotations


class TestReportsAPI:
    def _make_fake_job(self, client):
        """Submit a fast solver job to have a job_id to test against."""
        # Use a solver scan to quickly get a job_id without waiting for completion
        r = client.post("/api/solve/cylinder-flow", json={
            "nx": 20, "ny": 10, "re": 100, "n_steps": 1, "output_interval": 1,
            "radius": 3, "device": "cpu",
        })
        assert r.status_code == 200
        return r.json()["job_id"]

    def test_report_not_found(self, client):
        r = client.get("/api/reports/nonexistent_job_id")
        assert r.status_code == 404

    def test_report_summary_not_found(self, client):
        r = client.get("/api/reports/nonexistent_job_id/summary")
        assert r.status_code == 404

    def test_report_html_content_type(self, client):
        jid = self._make_fake_job(client)
        r = client.get(f"/api/reports/{jid}")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        html = r.text
        assert "TensorLBM" in html
        assert jid in html

    def test_report_summary_schema(self, client):
        jid = self._make_fake_job(client)
        r = client.get(f"/api/reports/{jid}/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == jid
        assert "status" in data
        assert "diagnostic_steps" in data
        assert "force_rows" in data
        assert "image_count" in data
        assert "report_url" in data
        assert "engineering_kpis" in data
        assert data["report_url"] == f"/api/reports/{jid}"

    def test_report_html_has_sections(self, client):
        jid = self._make_fake_job(client)
        html = client.get(f"/api/reports/{jid}").text
        for section in (
            "Summary",
            "Engineering KPIs",
            "Convergence",
            "Force Coefficients",
            "Result Images",
            "Configuration",
        ):
            assert section in html, f"Missing section: {section}"

    def test_compare_kpis_schema(self, client):
        jid1 = self._make_fake_job(client)
        jid2 = self._make_fake_job(client)
        r = client.get("/api/reports/compare/kpis", params=[("ids", jid1), ("ids", jid2)])
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["count"] == 2
        assert len(data["rows"]) == 2
        assert all("compare_metrics" in row for row in data["rows"])

    def test_compare_kpis_missing_jobs(self, client):
        jid = self._make_fake_job(client)
        r = client.get(
            "/api/reports/compare/kpis",
            params=[("ids", jid), ("ids", "missing-job-id")],
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["count"] == 1
        assert data["missing"] == ["missing-job-id"]


class TestConvergenceAPI:
    def test_convergence_not_found(self, client):
        r = client.get("/api/postprocess/convergence/nonexistent")
        assert r.status_code == 404

    def test_convergence_schema(self, client):
        r = client.post("/api/solve/cylinder-flow", json={
            "nx": 20, "ny": 10, "re": 100, "n_steps": 1, "output_interval": 1,
            "radius": 3, "device": "cpu",
        })
        jid = r.json()["job_id"]
        import time
        # Give the job a moment to be registered
        time.sleep(0.1)
        cr = client.get(f"/api/postprocess/convergence/{jid}")
        assert cr.status_code == 200
        data = cr.json()
        assert "job_id" in data
        assert "job_status" in data
        assert "diagnostic_count" in data
        assert "steps" in data
        assert "series" in data
        assert "forces_rows" in data
        assert "has_forces_csv" in data
        assert isinstance(data["series"], dict)
