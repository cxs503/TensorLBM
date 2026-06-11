"""Pytest fixtures for the TensorLBM Platform test-suite.

The ``platform`` directory shadows Python's stdlib :mod:`platform` module
when added directly to ``sys.path``, so we instead append the directory
containing ``backend/`` (i.e. ``platform/``) under a private alias by
prepending it to ``sys.path`` *after* ensuring stdlib ``platform`` has
already been imported.  This keeps imports such as
``from backend.main import app`` working without breaking anything else.
"""
from __future__ import annotations

import os
import platform as _stdlib_platform  # noqa: F401  (force stdlib import first)
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PLATFORM_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PLATFORM_DIR.parent
_SRC = _REPO_ROOT / "src"

for p in (str(_SRC), str(_PLATFORM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# FastAPI client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Import and return the FastAPI application instance."""
    from backend.main import app as _app  # type: ignore[import-not-found]
    return _app


@pytest.fixture()
def client(app) -> Iterator:
    """Return a Starlette TestClient bound to the platform app.

    A fresh client is created for every test so that any startup/shutdown
    hooks (e.g. the WebSocket broadcaster task) are wired correctly.
    """
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Job manager helpers (used by solver / benchmark tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def job_manager():
    """Direct access to the in-process job manager for assertions."""
    from backend import job_manager as jm  # type: ignore[import-not-found]
    return jm


def wait_for_job(job_manager_mod, job_id: str, timeout: float = 120.0) -> dict:
    """Poll the job manager until the job reaches a terminal state.

    Returns the final job dict.  Raises ``TimeoutError`` if the job does
    not complete within *timeout* seconds (defaults to 2 minutes – generous
    enough for the small test grids on a slow CI runner).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = job_manager_mod.get_job(job_id)
        assert job is not None, f"Unknown job_id {job_id!r}"
        status = job.status.value if hasattr(job.status, "value") else job.status
        if status in ("completed", "failed", "cancelled"):
            return job.to_dict()
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


@pytest.fixture()
def waiter(job_manager):
    """Return a callable that waits for a given job id to terminate."""
    def _wait(job_id: str, timeout: float = 120.0) -> dict:
        return wait_for_job(job_manager, job_id, timeout=timeout)

    return _wait


# Allow CI runners to override the default per-test wait via env-var.
DEFAULT_TIMEOUT = float(os.environ.get("PLATFORM_TEST_TIMEOUT", "120"))
