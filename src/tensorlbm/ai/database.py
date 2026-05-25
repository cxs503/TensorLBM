"""SQLite-backed persistence layer for the AI turbulence pipeline.

Three small tables are managed:

``runs``
    A simulation run that produced data (e.g. a small turbulent-channel or
    cylinder-flow LES).  Stores the run name, type, configuration JSON
    and the output directory on disk.

``datasets``
    A training dataset extracted from one or more runs.  Stores the
    on-disk path of the ``.pt`` blob, the sample count and a JSON
    metadata payload (e.g. the Smagorinsky constant used for labels).

``models``
    A trained AI turbulence model.  Stores the on-disk path of the
    ``.pt`` checkpoint, the architecture metadata and final training
    metrics.

Only :mod:`sqlite3` from the Python standard library is used so this works
out-of-the-box with no extra dependency.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    run_type     TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    output_dir   TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    run_id        INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    path          TEXT NOT NULL,
    n_samples     INTEGER NOT NULL,
    metadata_json TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    dataset_id    INTEGER REFERENCES datasets(id) ON DELETE SET NULL,
    path          TEXT NOT NULL,
    arch_json     TEXT NOT NULL,
    metrics_json  TEXT,
    created_at    TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open (and if needed create) a SQLite database file."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_run(
    conn: sqlite3.Connection,
    name: str,
    run_type: str,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
) -> int:
    """Insert a run record, return its primary key."""
    cur = conn.execute(
        "INSERT INTO runs (name, run_type, config_json, output_dir, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            str(name),
            str(run_type),
            json.dumps(config, default=str),
            str(output_dir) if output_dir is not None else None,
            _now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def insert_dataset(
    conn: sqlite3.Connection,
    name: str,
    path: str | Path,
    n_samples: int,
    run_id: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO datasets "
        "(name, run_id, path, n_samples, metadata_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(name),
            int(run_id) if run_id is not None else None,
            str(path),
            int(n_samples),
            json.dumps(metadata or {}, default=str),
            _now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def insert_model(
    conn: sqlite3.Connection,
    name: str,
    path: str | Path,
    arch: dict[str, Any],
    dataset_id: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO models "
        "(name, dataset_id, path, arch_json, metrics_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(name),
            int(dataset_id) if dataset_id is not None else None,
            str(path),
            json.dumps(arch, default=str),
            json.dumps(metrics or {}, default=str),
            _now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        for k in ("config_json", "metadata_json", "arch_json", "metrics_json"):
            if k in d and d[k]:
                try:
                    d[k[:-5]] = json.loads(d[k])
                except (TypeError, json.JSONDecodeError):
                    d[k[:-5]] = {}
        out.append(d)
    return out


def list_runs(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (int(limit),),
    ).fetchall()
    return _rows_to_dicts(rows)


def list_datasets(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM datasets ORDER BY id DESC LIMIT ?", (int(limit),),
    ).fetchall()
    return _rows_to_dicts(rows)


def list_models(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM models ORDER BY id DESC LIMIT ?", (int(limit),),
    ).fetchall()
    return _rows_to_dicts(rows)


def get_model_record(
    conn: sqlite3.Connection, model_id: int,
) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM models WHERE id=?", (int(model_id),)).fetchone()
    if row is None:
        return None
    return _rows_to_dicts([row])[0]


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

@dataclass
class LBMDatabase:
    """Tiny object-oriented wrapper around the helpers above.

    Use this when you want to pass *one* handle around rather than a raw
    :class:`sqlite3.Connection` plus the helper module.
    """

    path: Path
    conn: sqlite3.Connection

    @classmethod
    def open(cls, db_path: str | Path) -> LBMDatabase:
        conn = connect(db_path)
        return cls(path=Path(db_path), conn=conn)

    def close(self) -> None:
        self.conn.close()

    # --- pass-throughs --------------------------------------------------
    def insert_run(
        self,
        name: str,
        run_type: str,
        config: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> int:
        return insert_run(self.conn, name, run_type, config, output_dir)

    def insert_dataset(
        self,
        name: str,
        path: str | Path,
        n_samples: int,
        run_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return insert_dataset(self.conn, name, path, n_samples, run_id, metadata)

    def insert_model(
        self,
        name: str,
        path: str | Path,
        arch: dict[str, Any],
        dataset_id: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> int:
        return insert_model(self.conn, name, path, arch, dataset_id, metrics)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return list_runs(self.conn, limit=limit)

    def list_datasets(self, limit: int = 50) -> list[dict[str, Any]]:
        return list_datasets(self.conn, limit=limit)

    def list_models(self, limit: int = 50) -> list[dict[str, Any]]:
        return list_models(self.conn, limit=limit)

    def get_model_record(self, model_id: int) -> dict[str, Any] | None:
        return get_model_record(self.conn, model_id)
