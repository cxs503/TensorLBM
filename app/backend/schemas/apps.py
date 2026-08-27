"""Pydantic schemas for the AI4S application management API.

These models back ``app/backend/routers/apps.py``: listing registered
applications, running an application's full-stack pipeline (optionally
dispatching its data-production step to HPC), and querying run status.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AppInfo(BaseModel):
    """A registered AI4S application's identifying metadata."""

    name: str
    family: str
    version: str


class AppListResponse(BaseModel):
    """Envelope for the registered-application listing."""

    apps: list[AppInfo]
    total: int


class HpcRequest(BaseModel):
    """Optional HPC dispatch parameters for a run."""

    partition: str = Field("compute", max_length=120)
    nodes: int = Field(1, ge=1)
    cpus: int = Field(4, ge=1)
    mem: str = Field("8G", max_length=40)
    walltime: str = Field("02:00:00", max_length=40)
    backend: str = Field("slurm", max_length=20)


class AppRunRequest(BaseModel):
    """Payload to run an application's full-stack pipeline."""

    produce_cfg: dict[str, Any] = Field(default_factory=dict)
    train_cfg: dict[str, Any] = Field(default_factory=dict)
    db_path: str | None = Field(None, max_length=500)
    hpc: HpcRequest | None = Field(
        None,
        description=(
            "When set, dispatch the data-production step to HPC instead of "
            "running the full-stack loop locally."
        ),
    )


class RunReportOut(BaseModel):
    """Serialized full-stack :class:`tensorlbm.apps.base.RunReport`."""

    name: str
    family: str
    data_asset_id: str
    dataset_asset_id: str
    job_id: str
    model_id: int
    metrics: dict[str, Any]
    lineage_upstream: list[str] = Field(default_factory=list)


class HpcSubmitResponse(BaseModel):
    """Response for an HPC-dispatched data-production run."""

    app_name: str
    job_id: str
    hpc_job_id: str
    status: str
    backend: str
    script_cmd: str


class RunStatusResponse(BaseModel):
    """Status of a previously submitted run."""

    job_id: str
    app_name: str
    status: str
    hpc_job_id: str | None = None
    scheduler_state: str | None = None
    elapsed: str | None = None
