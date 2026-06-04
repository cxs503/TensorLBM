"""Thread-safe job lifecycle management for the TensorLBM platform.

Jobs are submitted to a thread-pool executor and their status is broadcast
to all connected WebSocket clients via an asyncio notification queue.
"""
from __future__ import annotations

import logging
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
        self.job_id = job_id
        self.name = name
        self.job_type = job_type
        self.config = config
        self.status: JobStatus = JobStatus.QUEUED
        self.created_at: str = datetime.now(UTC).isoformat()
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.error: str | None = None
        self.output_dir: Path = Path(f"/tmp/tensorlbm_platform/{job_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs: list[str] = []
        self.diagnostics: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {}

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
        }


# ---------------------------------------------------------------------------
# Global state (module-level singletons)
# ---------------------------------------------------------------------------

_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

# Map thread ident → job_id so the log handler knows which job to attach to
_thread_job_map: dict[int, str] = {}
_thread_job_map_lock = threading.Lock()

# Thread pool for running simulations
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tensorlbm-job")

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
        return job is not None and job.status == JobStatus.CANCELLED


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

    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC).isoformat()
    _notify(job)

    try:
        result = fn(job)
        if job.status != JobStatus.CANCELLED:
            job.status = JobStatus.COMPLETED
            job.result = result or {}
    except Exception:
        job.status = JobStatus.FAILED
        job.error = traceback.format_exc()
        job.logs.append(job.error)
    finally:
        with _thread_job_map_lock:
            _thread_job_map.pop(ident, None)  # type: ignore[arg-type]
        job.completed_at = datetime.now(UTC).isoformat()
        _notify(job)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
        if job_id in _jobs:
            del _jobs[job_id]
            return True
        return False


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
