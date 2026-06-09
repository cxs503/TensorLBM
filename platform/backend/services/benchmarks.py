"""Service-layer helpers for benchmark router orchestration."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .. import job_manager


def submit_benchmark(
    *,
    name: str,
    job_type: str,
    params: BaseModel,
    runner: Callable[[job_manager.Job], dict[str, Any]],
    message: str,
) -> dict[str, str]:
    """Submit a benchmark job and return standard job payload."""
    job_id = job_manager.submit(
        name=name,
        job_type=job_type,
        config=params.model_dump(),
        fn=runner,
    )
    return {"job_id": job_id, "message": message}
