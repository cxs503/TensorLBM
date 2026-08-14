"""HPC full-stack service for AI4S applications.

Wraps the data-production step of an
:class:`~tensorlbm.apps.base.AI4SApplication` as a cluster job (SLURM / PBS)
by reusing the platform's HPC scheduler (``backend.services.hpc_scheduler``).
This lets the heaviest part of the full-stack pipeline — ``produce_data`` —
run on a supercomputer, while the rest of :meth:`AI4SApplication.run`
(dataset building, training, serving, lineage) continues on the local
machine.

The scheduler module is imported lazily (and may be injected directly as the
``scheduler`` keyword argument) so the SDK stays importable and unit-testable
without a live cluster or the platform backend.

.. note::
    This module adds independent helpers next to :meth:`AI4SApplication.run`;
    the ``run()`` signature is left untouched for backward compatibility.
"""

from __future__ import annotations

import json
import shlex
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorlbm.apps.base import AI4SApplication

__all__ = [
    "HpcRunSpec",
    "submit_app_hpc",
    "query_app_hpc",
    "build_produce_data_cmd",
    "build_hpc_job_id",
    "load_hpc_scheduler",
]


class HpcScheduler(Protocol):
    """Structural interface of the platform HPC scheduler module.

    ``backend.services.hpc_scheduler`` and any test double satisfy this
    protocol structurally, so :func:`submit_app_hpc` / :func:`query_app_hpc`
    can accept either without importing the backend directly.
    """

    def submit_slurm(
        self,
        job_id: str,
        cmd: str,
        *,
        partition: str | None = None,
        nodes: int | None = None,
        cpus: int | None = None,
        mem: str | None = None,
        walltime: str | None = None,
        extra_directives: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def submit_pbs(
        self,
        job_id: str,
        cmd: str,
        *,
        queue: str | None = None,
        nodes: int | None = None,
        cpus: int | None = None,
        mem: str | None = None,
        walltime: str | None = None,
    ) -> dict[str, Any]: ...

    def query_slurm_status(self, hpc_job_id: str) -> dict[str, str]: ...


@dataclass
class HpcRunSpec:
    """Resource request + command for an HPC data-production run.

    Attributes:
        app_name: Registry name of the AI4S application whose
            ``produce_data`` step runs on the cluster.  Falls back to
            ``app.name`` when empty.
        partition: SLURM partition or PBS queue (default ``compute``).
        nodes: Number of nodes requested (``>= 1``).
        cpus: CPUs per task (``>= 1``).
        mem: Memory per node, e.g. ``8G``.
        walltime: Walltime limit, e.g. ``02:00:00``.
        script_cmd: Shell command executed on the cluster node.  When
            ``None``, a command that re-runs ``app.produce_data(produce_cfg)``
            is generated from :attr:`produce_cfg`.
        produce_cfg: Configuration mapping passed to ``produce_data`` (only
            used when ``script_cmd`` is ``None``).
        extra_directives: Additional ``#SBATCH`` directives (SLURM only).
        backend: Scheduler backend — ``slurm`` (default) or ``pbs``.
        output_dir: Directory the job writes its results into.
    """

    app_name: str = ""
    partition: str = "compute"
    nodes: int = 1
    cpus: int = 4
    mem: str = "8G"
    walltime: str = "02:00:00"
    script_cmd: str | None = None
    produce_cfg: Mapping[str, Any] = field(default_factory=dict)
    extra_directives: list[str] = field(default_factory=list)
    backend: str = "slurm"
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Scheduler loading (lazy, so the SDK stays importable without the backend)
# ---------------------------------------------------------------------------

def load_hpc_scheduler() -> HpcScheduler:
    """Import the platform HPC scheduler (SLURM/PBS wrappers).

    Tries the two import paths the platform backend can be installed under:
    ``backend.services.hpc_scheduler`` (when ``app/`` is on ``sys.path``) and
    ``app.backend.services.hpc_scheduler`` (when the repository root is).
    Prefer injecting a mock scheduler in tests instead of relying on this.
    """
    import importlib

    for module_name in (
        "backend.services.hpc_scheduler",
        "app.backend.services.hpc_scheduler",
    ):
        try:
            return cast("HpcScheduler", importlib.import_module(module_name))
        except ImportError:
            continue
    raise ImportError(
        "hpc_scheduler not importable; ensure the platform backend is on "
        "sys.path (backend.services.hpc_scheduler) or inject a scheduler."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_hpc_job_id(app_name: str) -> str:
    """Build a deterministic platform job id: ``<app>_hpc_<8 hex chars>``."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in app_name).strip("_")
    return f"{safe or 'app'}_hpc_{uuid.uuid4().hex[:8]}"


def build_produce_data_cmd(app_name: str, produce_cfg: Mapping[str, Any]) -> str:
    """Build the shell command that runs ``produce_data`` on the cluster.

    The command re-imports the application from the process-wide
    :data:`tensorlbm.apps.base.registry`, instantiates it, runs
    ``produce_data`` with *produce_cfg*, and prints the resulting
    :class:`~tensorlbm.apps.base.DataProduct` metadata as JSON for the
    scheduler logs.
    """
    cfg_json = json.dumps(dict(produce_cfg), sort_keys=True, default=str)
    code = (
        "import json;"
        "from tensorlbm.apps import registry;"
        f"_cls = registry.get({app_name!r});"
        f"_p = _cls().produce_data(json.loads({cfg_json!r}));"
        "print(json.dumps("
        "{'field_name': _p.field_name, 'shape': list(_p.shape), "
        "'dtype': _p.dtype, 'path': _p.path}, default=str))"
    )
    return f"python -c {shlex.quote(code)}"


# ---------------------------------------------------------------------------
# Submission / query
# ---------------------------------------------------------------------------

def submit_app_hpc(
    app: AI4SApplication,
    spec: HpcRunSpec | None = None,
    *,
    scheduler: HpcScheduler | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Wrap ``app.produce_data`` as an HPC job and submit it.

    Args:
        app: The AI4S application (its ``name`` is used when
            ``spec.app_name`` is empty).
        spec: Resource request + command.  A default :class:`HpcRunSpec` is
            used when omitted.
        scheduler: HPC scheduler.  Defaults to the lazily-imported platform
            scheduler; inject a mock in tests to avoid a real submission.
        job_id: Platform job id (auto-generated when omitted).

    Returns:
        Submission result dict — same shape as
        ``hpc_scheduler.submit_slurm``/``submit_pbs`` — whose ``hpc_job_id``
        holds the scheduler-assigned cluster job id (the value accepted by
        :func:`query_app_hpc`).  ``job_id``, ``app_name`` and the resolved
        ``script_cmd`` are added for observability.
    """
    spec = spec or HpcRunSpec()
    sched: HpcScheduler = scheduler or load_hpc_scheduler()

    app_name = spec.app_name or getattr(app, "name", "") or ""
    if not app_name:
        raise ValueError("HpcRunSpec.app_name (or app.name) is required")
    if spec.nodes < 1:
        raise ValueError("HpcRunSpec.nodes must be >= 1")
    if spec.cpus < 1:
        raise ValueError("HpcRunSpec.cpus must be >= 1")
    if spec.backend not in {"slurm", "pbs"}:
        raise ValueError(f"unknown HPC backend {spec.backend!r} (expected slurm|pbs)")

    job_id = job_id or build_hpc_job_id(app_name)
    cmd = (
        spec.script_cmd
        if spec.script_cmd is not None
        else build_produce_data_cmd(app_name, spec.produce_cfg)
    )

    if spec.backend == "pbs":
        result = sched.submit_pbs(
            job_id,
            cmd,
            queue=spec.partition,
            nodes=spec.nodes,
            cpus=spec.cpus,
            mem=spec.mem,
            walltime=spec.walltime,
        )
    else:
        result = sched.submit_slurm(
            job_id,
            cmd,
            partition=spec.partition,
            nodes=spec.nodes,
            cpus=spec.cpus,
            mem=spec.mem,
            walltime=spec.walltime,
            extra_directives=spec.extra_directives,
        )

    result.setdefault("job_id", job_id)
    result.setdefault("app_name", app_name)
    result.setdefault("script_cmd", cmd)
    return result


def query_app_hpc(job_id: str, *, scheduler: HpcScheduler | None = None) -> dict[str, str]:
    """Query the status of a previously submitted HPC job.

    Args:
        job_id: Scheduler-assigned cluster job id (the ``hpc_job_id`` returned
            by :func:`submit_app_hpc`).
        scheduler: HPC scheduler (defaults to the lazily-imported platform
            scheduler).

    Returns:
        Status dict with ``hpc_job_id``, ``state`` and ``elapsed`` keys (for
        SLURM); PBS backends may return a backend-specific shape.
    """
    sched: HpcScheduler = scheduler or load_hpc_scheduler()
    return sched.query_slurm_status(job_id)
