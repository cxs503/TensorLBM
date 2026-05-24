"""Unit tests for the in-process job manager (cancel, diagnostics, log routing)."""
from __future__ import annotations

import logging
import time


def _make_job(job_manager, name: str = "unit-test"):
    def _fn(job):
        return {"ran": True, "name": name}
    return job_manager.submit(name=name, job_type="unit_test", config={}, fn=_fn)


def test_submit_and_complete(job_manager, waiter):
    job_id = _make_job(job_manager)
    final = waiter(job_id, timeout=10.0)
    assert final["status"] == "completed"
    assert final["result"]["ran"] is True


def test_get_unknown(job_manager):
    assert job_manager.get_job("does-not-exist") is None


def test_delete_unknown(job_manager):
    assert job_manager.delete_job("does-not-exist") is False


def test_cancel_unknown(job_manager):
    assert job_manager.cancel_job("does-not-exist") is False


def test_failed_job_records_traceback(job_manager, waiter):
    def _boom(job):
        raise RuntimeError("boom")

    job_id = job_manager.submit(name="will-fail", job_type="unit_test", config={}, fn=_boom)
    final = waiter(job_id, timeout=10.0)
    assert final["status"] == "failed"
    assert "RuntimeError" in (final["error"] or "")


def test_cancel_queued_job(job_manager):
    """A long-running job should be cancellable; status flips to CANCELLED."""

    def _slow(job):
        # Cooperative check is not implemented in LBM kernels, but the test
        # still exercises the cancel_job() API path on a *running* job.
        for _ in range(20):
            time.sleep(0.05)
        return {"done": True}

    job_id = job_manager.submit(name="slow", job_type="unit_test", config={}, fn=_slow)
    # Give the executor a moment to start the job
    time.sleep(0.05)
    assert job_manager.cancel_job(job_id) is True

    final = job_manager.get_job(job_id)
    assert final is not None
    status = final.status.value if hasattr(final.status, "value") else final.status
    assert status == "cancelled"

    # Cancelling an already-cancelled job is a no-op (returns False)
    assert job_manager.cancel_job(job_id) is False


def test_push_diagnostic_keeps_recent_entries(job_manager):
    def _emit(job):
        for i in range(10):
            job_manager.push_diagnostic(job.job_id, {"step": i, "u_max": 0.1 * i})
        return {"ok": True}

    job_id = job_manager.submit(name="diag", job_type="unit_test", config={}, fn=_emit)
    # Wait for completion
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        j = job_manager.get_job(job_id)
        if j is not None and j.status.value == "completed":
            break
        time.sleep(0.05)

    job = job_manager.get_job(job_id)
    assert job is not None
    # to_dict() returns at most the most recent 50 diagnostics
    diag = job.to_dict()["diagnostics"]
    assert 1 <= len(diag) <= 50
    assert diag[-1]["step"] == 9


def test_push_diagnostic_unknown_job_is_safe(job_manager):
    # Should silently ignore unknown job id rather than raising
    job_manager.push_diagnostic("nope", {"x": 1})


def test_log_routing(job_manager, waiter):
    """Records emitted on the ``tensorlbm`` logger from a worker thread
    must be routed to the *active* job's log buffer."""
    msg = "hello-from-test-log-routing"

    def _log_job(job):
        tl_logger = logging.getLogger("tensorlbm")
        tl_logger.setLevel(logging.INFO)
        tl_logger.warning(msg)
        return {"ok": True}

    job_id = job_manager.submit(name="log", job_type="unit_test", config={}, fn=_log_job)
    waiter(job_id, timeout=5.0)
    job = job_manager.get_job(job_id)
    assert job is not None
    assert any(msg in line for line in job.logs), job.logs
