"""Training-job management for the TensorLBM AI pipeline.

Clean-room implementation per
``docs/plans/model-training-cleanroom-spec.md`` — only the *functional
design* (job lifecycle state machine, metric recording, model registration,
data lineage) is borrowed from common MLOps practice; the code below is
written independently for TensorLBM, using plain :mod:`sqlite3` + dataclasses
(consistent with ``tensorlbm.ai.database`` and ``tensorlbm.data.catalog``).

A :class:`TrainingJobRegistry` lives in the same SQLite file as the
``runs``/``datasets``/``models`` tables managed by :mod:`tensorlbm.ai.database`
and the ``assets``/``lineage`` tables managed by
:mod:`tensorlbm.data.catalog`, so a finished job can register its model with
``insert_model`` and record ``training-job <- dataset <- field-product``
lineage through the catalog's lineage graph.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tensorlbm.ai.database import connect as _lbm_connect
from tensorlbm.ai.database import insert_model

# ---------------------------------------------------------------------------
# Job status model
# ---------------------------------------------------------------------------

VALID_STATUSES: tuple[str, ...] = (
    "created",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
)

_TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

_TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"queued", "running", "failed", "cancelled"}),
    "queued": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_jobs (
    job_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    model_id     INTEGER,
    dataset_id   INTEGER,
    metrics_json TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
"""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """One managed training job; an immutable in-memory view of a DB row."""

    job_id: str
    status: str
    config: dict[str, Any]
    model_id: int | None = None
    dataset_id: int | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _decode_json(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _require_job_id(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError("job_id must be a non-empty string")
    return value


def _validate_status(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("status must be a string")
    if value not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}")
    return value


def _row_to_job(row: sqlite3.Row) -> TrainingJob:
    return TrainingJob(
        job_id=row["job_id"],
        status=row["status"],
        config=_decode_json(row["config_json"], {}),
        model_id=row["model_id"],
        dataset_id=row["dataset_id"],
        metrics=_decode_json(row["metrics_json"], None),
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TrainingJobRegistry:
    """SQLite-backed registry for training-job lifecycle and lineage."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def open(cls, db_path: str | Path) -> "TrainingJobRegistry":
        """Open a registry on ``db_path``.

        The connection is created through :func:`tensorlbm.ai.database.connect`,
        so the ``runs``/``datasets``/``models`` tables coexist with the
        ``training_jobs`` table managed here (needed by ``register_model``).
        """
        conn = _lbm_connect(db_path)
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    # -- jobs ---------------------------------------------------------------

    def create_job(
        self,
        config: dict[str, Any],
        *,
        job_id: str | None = None,
        dataset_id: int | None = None,
        model_id: int | None = None,
    ) -> TrainingJob:
        """Create a training job, returning its record.

        ``job_id`` defaults to ``job_<12 hex chars>`` when not supplied.
        """
        if job_id is None:
            job_id = f"job_{uuid4().hex[:12]}"
        else:
            job_id = _require_job_id(job_id)
        if self.get_job(job_id) is not None:
            raise ValueError(f"job {job_id!r} already exists")
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")
        now = _now()
        self._conn.execute(
            "INSERT INTO training_jobs "
            "(job_id, status, config_json, model_id, dataset_id, metrics_json, "
            " error, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                "created",
                _encode_json(config),
                int(model_id) if model_id is not None else None,
                int(dataset_id) if dataset_id is not None else None,
                None,
                None,
                now,
                now,
            ),
        )
        self._conn.commit()
        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> TrainingJob | None:
        row = self._conn.execute(
            "SELECT * FROM training_jobs WHERE job_id = ?",
            (_require_job_id(job_id),),
        ).fetchone()
        return _row_to_job(row) if row else None

    def _require_job(self, job_id: str) -> TrainingJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"no training job {job_id!r}")
        return job

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TrainingJob]:
        query = "SELECT * FROM training_jobs WHERE 1=1"
        args: list[Any] = []
        if status is not None:
            query += " AND status = ?"
            args.append(_validate_status(status))
        query += " ORDER BY created_at DESC, job_id DESC LIMIT ?"
        args.append(int(limit))
        rows = self._conn.execute(query, args).fetchall()
        return [_row_to_job(r) for r in rows]

    def update_status(
        self,
        job_id: str,
        status: str,
        error: str | None = None,
    ) -> TrainingJob:
        """Transition a job's status through the validated state machine."""
        status = _validate_status(status)
        job = self._require_job(job_id)
        if job.status in _TERMINAL_STATUSES:
            raise ValueError(
                f"job {job_id!r} is in terminal state {job.status!r}; "
                "no further transitions allowed"
            )
        if status not in _TRANSITIONS[job.status]:
            raise ValueError(f"invalid transition {job.status!r} -> {status!r}")
        self._conn.execute(
            "UPDATE training_jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
            (status, error, _now(), job_id),
        )
        self._conn.commit()
        updated = self.get_job(job_id)
        assert updated is not None
        return updated

    def record_metrics(self, job_id: str, metrics: dict[str, Any]) -> TrainingJob:
        """Merge ``metrics`` into the job's existing metrics dictionary."""
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict")
        job = self._require_job(job_id)
        merged = dict(job.metrics or {})
        merged.update(metrics)
        self._conn.execute(
            "UPDATE training_jobs SET metrics_json = ?, updated_at = ? WHERE job_id = ?",
            (_encode_json(merged), _now(), job_id),
        )
        self._conn.commit()
        updated = self.get_job(job_id)
        assert updated is not None
        return updated

    # -- model registration ------------------------------------------------

    def register_model(
        self,
        job_id: str,
        *,
        name: str,
        path: str | Path,
        arch: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> int:
        """Register a finished job's model in the ``models`` table.

        The job's ``dataset_id`` (if any) is carried into the model record and
        the returned model primary key is written back to the job.
        """
        job = self._require_job(job_id)
        model_id = insert_model(
            self._conn,
            name=name,
            path=path,
            arch=arch,
            dataset_id=job.dataset_id,
            metrics=metrics,
        )
        self._conn.execute(
            "UPDATE training_jobs SET model_id = ?, updated_at = ? WHERE job_id = ?",
            (model_id, _now(), job_id),
        )
        self._conn.commit()
        return model_id

    # -- lineage ------------------------------------------------------------

    def record_lineage(
        self,
        catalog: Any,
        job_asset_id: str,
        *,
        dataset_asset_id: str | None = None,
        product_asset_id: str | None = None,
    ) -> None:
        """Record training-job lineage in a :class:`~tensorlbm.data.catalog.FieldDataCatalog`.

        Records two directed edges when the relevant asset ids are supplied:

        * ``product_asset_id -> dataset_asset_id`` (``derived_from``)
        * ``dataset_asset_id -> job_asset_id`` (``trained_on``)

        ``catalog`` is typed loosely to avoid importing the data package
        eagerly; it only needs an ``add_lineage`` method accepting a
        ``LineageRecord``-shaped object.
        """
        from tensorlbm.data.catalog import LineageRecord

        job_asset_id = _require_job_id(job_asset_id)
        if dataset_asset_id is not None:
            if product_asset_id is not None:
                catalog.add_lineage(
                    LineageRecord(
                        source_id=product_asset_id,
                        target_id=dataset_asset_id,
                        relation_type="derived_from",
                        resource_type="product",
                    )
                )
            catalog.add_lineage(
                LineageRecord(
                    source_id=dataset_asset_id,
                    target_id=job_asset_id,
                    relation_type="trained_on",
                    resource_type="dataset",
                )
            )


__all__ = [
    "VALID_STATUSES",
    "TrainingJob",
    "TrainingJobRegistry",
]
