"""Service-layer helpers for benchmark router orchestration."""
from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Any

from .. import job_manager

if TYPE_CHECKING:
    from pydantic import BaseModel


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
