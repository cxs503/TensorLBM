"""Tests for job management endpoints (list / get / logs / files / images / compare / delete)."""
from __future__ import annotations


def _submit_tiny_job(client) -> str:
    """Submit the cheapest possible solver job and return its job_id."""
    r = client.post(
        "/api/solve/lid-driven-cavity",
        json={"nx": 16, "u_lid": 0.1, "re": 50.0,
              "n_steps": 20, "output_interval": 10},
    )
    assert r.status_code == 200
    return r.json()["job_id"]


def test_list_jobs_returns_array(client):
    r = client.get("/api/jobs/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_unknown_job_404(client):
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_job_lifecycle_endpoints(client, waiter):
    job_id = _submit_tiny_job(client)
    final = waiter(job_id, timeout=120.0)
    assert final["status"] == "completed"

    # get_job
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["job_id"] == job_id

    # logs
    r = client.get(f"/api/jobs/{job_id}/logs")
    assert r.status_code == 200
    assert "logs" in r.json()

    # files
    r = client.get(f"/api/jobs/{job_id}/files")
    assert r.status_code == 200
    files = r.json()["files"]
    assert files, "Expected at least one output file"

    # download one file
    first_path = files[0]["path"]
    r = client.get(f"/api/jobs/{job_id}/files/{first_path}")
    assert r.status_code == 200
    assert len(r.content) > 0

    # images list
    r = client.get(f"/api/jobs/{job_id}/images")
    assert r.status_code == 200
    images = r.json()["images"]
    if images:
        r = client.get(f"/api/jobs/{job_id}/images/{images[0]}")
        assert r.status_code == 200
        assert r.json()["data"].startswith("data:image/png;base64,")

    # metadata endpoint
    r = client.get(f"/api/jobs/{job_id}/metadata")
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert isinstance(meta, dict)


def test_compare_jobs(client, waiter):
    j1 = _submit_tiny_job(client)
    j2 = _submit_tiny_job(client)
    waiter(j1)
    waiter(j2)

    r = client.get("/api/jobs/compare", params=[("ids", j1), ("ids", j2), ("ids", "missing")])
    assert r.status_code == 200
    data = r.json()
    assert len(data["jobs"]) == 2
    assert "missing" in data["missing"]


def test_compare_jobs_requires_ids(client):
    r = client.get("/api/jobs/compare", params=[("ids", "nope")] * 11)
    assert r.status_code == 400


def test_delete_job(client, waiter):
    job_id = _submit_tiny_job(client)
    waiter(job_id)
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] == job_id
    # Subsequent get returns 404
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 404


def test_cancel_running_job(client, waiter):
    r = client.post(
        "/api/solve/cylinder-flow",
        json={
            "nx": 64,
            "ny": 24,
            "u_in": 0.05,
            "re": 50.0,
            "radius": 4.0,
            "n_steps": 200,
            "output_interval": 100,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    cancel_r = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel_r.status_code in (200, 409), cancel_r.text
    final = waiter(job_id, timeout=120.0)
    assert final["status"] in {"cancelled", "completed"}


def test_cleanup_endpoint_dry_run(client, waiter):
    j1 = _submit_tiny_job(client)
    j2 = _submit_tiny_job(client)
    waiter(j1)
    waiter(j2)
    r = client.post(
        "/api/jobs/cleanup",
        json={"max_completed_jobs": 1, "dry_run": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is True
    assert "candidates" in body


def test_path_traversal_blocked(client, waiter):
    """The file download endpoint must reject ``..``-style escapes.

    Use percent-encoded path segments to bypass any client-side URL
    normalisation that ``httpx``/Starlette might apply.
    """
    job_id = _submit_tiny_job(client)
    waiter(job_id)

    # URL-encoded ``../../etc/passwd`` — the server-side guard in
    # :func:`get_file` must refuse to serve files outside the job's
    # output directory.
    bad = "%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd"
    r = client.get(f"/api/jobs/{job_id}/files/{bad}")
    # Acceptable outcomes: 403 (explicit reject), 404 (no such file inside
    # the job dir), 422 (router can't bind the path).  A 200 here would
    # mean the endpoint is leaking host files.
    assert r.status_code in (403, 404, 422), r.status_code
    # The file body must not contain real /etc/passwd content
    assert b"root:" not in r.content
