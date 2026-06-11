"""Extra job-manager tests for orchestration metadata and retries."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable


class JobManagerProtocol(Protocol):
    def submit(
        self,
        *,
        name: str,
        job_type: str,
        config: dict[str, object],
        fn: Callable[[object], dict[str, bool]],
    ) -> str: ...

    def orchestration_kpis(self) -> dict[str, object]: ...


def test_retry_policy_succeeds_after_retry(
    job_manager: JobManagerProtocol,
    waiter: Callable[..., dict[str, object]],
) -> None:
    state = {"count": 0}

    def _flaky(_job: object) -> dict[str, bool]:
        state["count"] += 1
        if state["count"] == 1:
            raise RuntimeError("transient")
        return {"ok": True}

    cfg = {"orchestration": {"max_retries": 1, "cost_rate_per_second": 0.1}}
    job_id = job_manager.submit(name="flaky", job_type="unit_test", config=cfg, fn=_flaky)
    final = waiter(job_id, timeout=10.0)
    assert final["status"] == "completed"
    assert final["retry_attempt"] == 1
    assert final["max_retries"] == 1
    assert final["estimated_cost"] >= 0.0


def test_orchestration_kpis(
    job_manager: JobManagerProtocol,
    waiter: Callable[..., dict[str, object]],
) -> None:
    def _ok(_job: object) -> dict[str, bool]:
        return {"ok": True}

    job_id = job_manager.submit(
        name="kpi",
        job_type="unit_test",
        config={"orchestration": {"resource_label": "cpu"}},
        fn=_ok,
    )
    waiter(job_id, timeout=10.0)
    kpi = job_manager.orchestration_kpis()
    assert kpi["max_workers"] >= 1
    assert "scheduler_profile" in kpi
    assert kpi["jobs_total"] >= 1
