"""Thread-safe job lifecycle management for the TensorLBM platform.

Jobs are submitted to a thread-pool executor and their status is broadcast
to all connected WebSocket clients via an asyncio notification queue.
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import threading
import traceback
import uuid
from collections.abc import Callable  # noqa: TC003
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio


# ---------------------------------------------------------------------------
# Enums / data classes
# ---------------------------------------------------------------------------

class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Represents a single simulation or benchmark job."""

    def __init__(self, job_id: str, name: str, job_type: str, config: dict[str, Any]) -> None:
        orch = config.get("orchestration") if isinstance(config, dict) else {}
        self.job_id = job_id
        self.name = name
        self.job_type = job_type
        self.config = config
        self.status: JobStatus = JobStatus.QUEUED
        self.created_at: str = datetime.now(UTC).isoformat()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.error: str | None = None
        self.output_dir: Path = output_root() / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs: list[str] = []
        self.diagnostics: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}
        self.cancel_requested: bool = False
        self.scheduler_profile: str = _SCHEDULER_PROFILE
        self.queue_wait_seconds: float | None = None
        self.run_duration_seconds: float | None = None
        self.total_duration_seconds: float | None = None
        self.io_bytes: int = 0
        self.retry_attempt: int = 0
        self.max_retries: int = max(0, int((orch or {}).get("max_retries", 0)))
        self.resume_from: str | None = (
            str((orch or {}).get("resume_from"))
            if (orch or {}).get("resume_from") is not None
            else None
        )
        self.assigned_resource: str = str(
            (orch or {}).get("resource_label") or config.get("device") or "cpu",
        )
        self.cost_rate_per_second: float = max(
            0.0,
            float((orch or {}).get("cost_rate_per_second", 0.0)),
        )
        self.estimated_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "job_type": self.job_type,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "output_dir": str(self.output_dir),
            "logs": self.logs[-200:],
            "diagnostics": self.diagnostics[-50:],
            "result": self.result,
            "cancel_requested": self.cancel_requested,
            "scheduler_profile": self.scheduler_profile,
            "queue_wait_seconds": self.queue_wait_seconds,
            "run_duration_seconds": self.run_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "io_bytes": self.io_bytes,
            "retry_attempt": self.retry_attempt,
            "max_retries": self.max_retries,
            "resume_from": self.resume_from,
            "assigned_resource": self.assigned_resource,
            "estimated_cost": self.estimated_cost,
        }


class JobCancelledError(RuntimeError):
    """Raised by cooperative workers when a job cancellation is requested."""


# ---------------------------------------------------------------------------
# Global state (module-level singletons)
# ---------------------------------------------------------------------------

_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

# Map thread ident → job_id so the log handler knows which job to attach to
_thread_job_map: dict[int, str] = {}
_thread_job_map_lock = threading.Lock()

# Thread pool for running simulations
_MAX_WORKERS = max(1, int(os.environ.get("TENSORLBM_MAX_WORKERS", "4")))
_SCHEDULER_PROFILE = (
    os.environ.get("TENSORLBM_SCHEDULER_PROFILE", "single_node_threadpool").strip()
    or "single_node_threadpool"
)
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="tensorlbm-job")

_DEFAULT_OUTPUT_ROOT = "/tmp/tensorlbm_platform"
_OUTPUT_ROOT = Path(os.environ.get("TENSORLBM_OUTPUT_ROOT", _DEFAULT_OUTPUT_ROOT)).resolve()
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Asyncio event loop + notification queue (set by main.py on startup)
_event_loop: asyncio.AbstractEventLoop | None = None
_notify_queue: asyncio.Queue[dict[str, Any]] | None = None  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Per-job log capture
# ---------------------------------------------------------------------------

class _JobLogHandler(logging.Handler):
    """Routes tensorlbm log records to the active job's log buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        ident = threading.current_thread().ident
        with _thread_job_map_lock:
            job_id = _thread_job_map.get(ident)  # type: ignore[arg-type]
        if job_id is None:
            return
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return
        line = self.format(record)
        job.logs.append(line)
        if len(job.logs) > 500:
            job.logs = job.logs[-500:]


_log_handler = _JobLogHandler()
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))

# Attach the handler at import time so log routing works even when
# set_event_loop() has not been called (e.g. in unit tests or CLI usage).
_tl_logger = logging.getLogger("tensorlbm")
if _log_handler not in _tl_logger.handlers:
    _tl_logger.addHandler(_log_handler)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _notify(job: Job) -> None:
    """Thread-safe notification to the asyncio WebSocket broadcaster.

    Silently ignores cases where the bound event loop has already been
    closed (e.g. between TestClient invocations) so background worker
    threads never raise.
    """
    if _event_loop is None or _notify_queue is None:
        return
    try:
        if _event_loop.is_closed():
            return
        _event_loop.call_soon_threadsafe(_notify_queue.put_nowait, job.to_dict())
    except RuntimeError:
        # Loop closed concurrently – treat as best-effort.
        return


def _is_cancelled(job_id: str) -> bool:
    """Return whether a job has been marked as cancelled."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        return job is not None and (
            job.status == JobStatus.CANCELLED or job.cancel_requested
        )


def _job_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                total += p.stat().st_size
    return total


def _run_job(job: Job, fn: Callable[[Job], dict[str, Any] | None]) -> None:
    """Execute *fn* in the current thread, updating *job* status."""
    ident = threading.current_thread().ident
    with _thread_job_map_lock:
        _thread_job_map[ident] = job.job_id  # type: ignore[index]

    if _is_cancelled(job.job_id):
        job.completed_at = datetime.now(UTC).isoformat()
        _notify(job)
        with _thread_job_map_lock:
            _thread_job_map.pop(ident, None)  # type: ignore[arg-type]
        return

    created_ts = datetime.fromisoformat(job.created_at)
    started_ts = datetime.now(UTC)
    job.status = JobStatus.RUNNING
    job.started_at = started_ts.isoformat()
    job.queue_wait_seconds = max(0.0, (started_ts - created_ts).total_seconds())
    _notify(job)

    try:
        attempt = 0
        while True:
            job.retry_attempt = attempt
            try:
                result = fn(job)
                if job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.COMPLETED
                    job.result = result or {}
                break
            except JobCancelledError:
                job.status = JobStatus.CANCELLED
                job.error = "Job cancelled by user request."
                break
            except Exception:
                job.error = traceback.format_exc()
                job.logs.append(job.error)
                if attempt >= job.max_retries or job.cancel_requested:
                    job.status = JobStatus.FAILED
                    break
                attempt += 1
                job.logs.append(
                    f"Retrying job attempt={attempt}/{job.max_retries} after failure.",
                )
                _notify(job)
    finally:
        with _thread_job_map_lock:
            _thread_job_map.pop(ident, None)  # type: ignore[arg-type]
        completed_ts = datetime.now(UTC)
        job.completed_at = completed_ts.isoformat()
        runtime = max(0.0, (completed_ts - started_ts).total_seconds())
        job.run_duration_seconds = runtime
        job.total_duration_seconds = max(0.0, (completed_ts - created_ts).total_seconds())
        job.io_bytes = _job_size(job.output_dir)
        if job.cost_rate_per_second > 0:
            job.estimated_cost = round(runtime * job.cost_rate_per_second, 6)
        _notify(job)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_backends(
    *,
    jobs: dict[str, Job] | None = None,
    lock: threading.Lock | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> None:
    """Override in-process storage/scheduler backends for extension/testing.

    This is an intentionally thin abstraction seam so future distributed
    schedulers or persistent stores can be integrated without changing the
    public submit/get/list/cancel API.
    """
    global _jobs, _jobs_lock, _executor
    if jobs is not None:
        _jobs = jobs
    if lock is not None:
        _jobs_lock = lock
    if executor is not None:
        _executor = executor

def set_event_loop(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[dict[str, Any]],  # type: ignore[type-arg]
) -> None:
    """Bind the asyncio event loop and notification queue; install log handler."""
    global _event_loop, _notify_queue
    _event_loop = loop
    _notify_queue = queue
    # Attach to the tensorlbm root logger so all simulation logs are captured
    tl_logger = logging.getLogger("tensorlbm")
    if _log_handler not in tl_logger.handlers:
        tl_logger.addHandler(_log_handler)


def output_root() -> Path:
    """Return the configured root directory for platform job outputs."""
    return _OUTPUT_ROOT


def submit(
    name: str,
    job_type: str,
    config: dict[str, Any],
    fn: Callable[[Job], dict[str, Any] | None],
) -> str:
    """Create a job and schedule it on the thread pool. Returns job_id."""
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, name, job_type, config)
    with _jobs_lock:
        _jobs[job_id] = job
    _notify(job)
    _executor.submit(_run_job, job, fn)
    return job_id


def get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        return [j.to_dict() for j in sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)]  # noqa: E501


def delete_job(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.status == JobStatus.RUNNING:
            return False
        del _jobs[job_id]
    shutil.rmtree(job.output_dir, ignore_errors=True)
    return True


def cancel_job(job_id: str) -> bool:
    """Request cancellation of a queued or running job.

    Sets the job status to CANCELLED if the job is in QUEUED or RUNNING state.
    Note: this does not interrupt a running thread (LBM steps are not
    interruptible), but marks the job for cleanup and prevents queued jobs
    from starting.

    Returns:
        True if the job was found and its status changed, False otherwise.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            job.cancel_requested = True
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(UTC).isoformat()
            _notify(job)
            return True
        return False


def push_diagnostic(job_id: str, data: dict[str, Any]) -> None:
    """Push a per-step diagnostic update for a running job.

    Broadcasting is done via the existing _notify mechanism so all
    WebSocket subscribers receive live updates.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return
    job.diagnostics.append(data)
    if len(job.diagnostics) > 1000:
        job.diagnostics = job.diagnostics[-1000:]
    _notify(job)


def raise_if_cancelled(job_id: str) -> None:
    """Raise :class:`JobCancelledError` when a cancellation has been requested."""
    if _is_cancelled(job_id):
        raise JobCancelledError(f"Job {job_id} has been cancelled")


def cleanup_jobs(
    *,
    retention_seconds: int | None = None,
    max_completed: int | None = None,
    max_total_bytes: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Cleanup completed/failed/cancelled jobs by retention and storage policies."""
    now = datetime.now(UTC)

    with _jobs_lock:
        snapshot = list(_jobs.values())

    managed = [
        j for j in snapshot
        if j.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    ]
    managed.sort(key=lambda j: j.completed_at or j.created_at)
    sizes = {j.job_id: _job_size(j.output_dir) for j in managed}

    to_delete: set[str] = set()
    if retention_seconds is not None and retention_seconds >= 0:
        for j in managed:
            t = j.completed_at or j.created_at
            age = (now - datetime.fromisoformat(t)).total_seconds()
            if age > retention_seconds:
                to_delete.add(j.job_id)

    if max_completed is not None and max_completed >= 0 and len(managed) > max_completed:
        overflow = len(managed) - max_completed
        for j in managed[:overflow]:
            to_delete.add(j.job_id)

    if max_total_bytes is not None and max_total_bytes >= 0:
        total = sum(sizes.values())
        if total > max_total_bytes:
            for j in managed:
                if total <= max_total_bytes:
                    break
                if j.job_id in to_delete:
                    total -= sizes.get(j.job_id, 0)
                    continue
                to_delete.add(j.job_id)
                total -= sizes.get(j.job_id, 0)

    reclaimed = sum(sizes.get(jid, 0) for jid in to_delete)
    deleted: list[str] = []
    if not dry_run:
        for jid in sorted(to_delete):
            if delete_job(jid):
                deleted.append(jid)

    return {
        "dry_run": dry_run,
        "candidates": sorted(to_delete),
        "deleted": deleted if not dry_run else [],
        "reclaimed_bytes": reclaimed,
        "managed_jobs": len(managed),
        "output_root": str(output_root()),
    }


def max_workers() -> int:
    """Return configured worker count for the local scheduler."""
    return _MAX_WORKERS


def scheduler_profile() -> str:
    """Return configured scheduler profile string."""
    return _SCHEDULER_PROFILE


def orchestration_kpis() -> dict[str, Any]:
    """Aggregate orchestration-level KPIs from in-process jobs."""
    with _jobs_lock:
        rows = [j.to_dict() for j in _jobs.values()]
    total = len(rows)
    completed = [r for r in rows if r["status"] == JobStatus.COMPLETED]
    failed = [r for r in rows if r["status"] == JobStatus.FAILED]
    cancelled = [r for r in rows if r["status"] == JobStatus.CANCELLED]
    running = [r for r in rows if r["status"] == JobStatus.RUNNING]
    queue_waits = [
        float(r["queue_wait_seconds"])
        for r in rows
        if r["queue_wait_seconds"] is not None
    ]
    run_times = [
        float(r["run_duration_seconds"])
        for r in rows
        if r["run_duration_seconds"] is not None
    ]
    retries = [int(r.get("retry_attempt", 0)) for r in rows]
    resources: dict[str, int] = {}
    for row in rows:
        key = str(row.get("assigned_resource") or "cpu")
        resources[key] = resources.get(key, 0) + 1

    terminal = len(completed) + len(failed) + len(cancelled)
    return {
        "scheduler_profile": scheduler_profile(),
        "max_workers": max_workers(),
        "jobs_total": total,
        "jobs_running": len(running),
        "jobs_completed": len(completed),
        "jobs_failed": len(failed),
        "jobs_cancelled": len(cancelled),
        "success_rate": (len(completed) / terminal) if terminal else None,
        "avg_queue_wait_seconds": (sum(queue_waits) / len(queue_waits)) if queue_waits else None,
        "avg_run_duration_seconds": (sum(run_times) / len(run_times)) if run_times else None,
        "avg_retries": (sum(retries) / len(retries)) if retries else 0.0,
        "io_bytes_total": int(sum(int(r.get("io_bytes", 0) or 0) for r in rows)),
        "estimated_cost_total": float(
            sum(float(r.get("estimated_cost", 0.0) or 0.0) for r in rows),
        ),
        "resources": resources,
    }
