"""CFD field-data catalog: asset registry, metadata, lineage, quality.

Clean-room implementation per ``docs/plans/data-management-cleanroom-spec.md``
— only the *functional design* (asset registry, key-value metadata, lineage
graph, quality scoring) is borrowed from common data-governance practice; the
code below is written independently for TensorLBM's field-data products.

Storage is plain SQLite (consistent with ``tensorlbm.ai.database``) so the
catalog is a drop-in for the platform's existing run/dataset/model ledger.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tensorlbm.data.contracts import FieldProduct


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id         TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    description      TEXT DEFAULT '',
    kind             TEXT NOT NULL DEFAULT 'field_product',
    field_name       TEXT,
    units            TEXT,
    shape            TEXT,
    dtype            TEXT,
    tags             TEXT DEFAULT '[]',
    quality_score    INTEGER DEFAULT 0,
    sensitivity_level TEXT DEFAULT 'internal',
    source_run_id    TEXT,
    status           TEXT DEFAULT 'active',
    version          TEXT DEFAULT '1.0.0',
    created_at       REAL,
    updated_at       REAL
);
CREATE TABLE IF NOT EXISTS asset_metadata (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id    TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT,
    source      TEXT DEFAULT 'manual',
    confidence  REAL DEFAULT 1.0,
    created_at  REAL,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS lineage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    relation_type TEXT DEFAULT 'derived_from',
    transformation TEXT DEFAULT '',
    resource_type TEXT DEFAULT 'product',
    created_at    REAL
);
CREATE TABLE IF NOT EXISTS quality_reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id      TEXT NOT NULL,
    checks        TEXT DEFAULT '[]',
    overall_score INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'passed',
    created_at    REAL,
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_assets_kind   ON assets(kind);
CREATE INDEX IF NOT EXISTS idx_assets_field  ON assets(field_name);
CREATE INDEX IF NOT EXISTS idx_meta_asset    ON asset_metadata(asset_id);
CREATE INDEX IF NOT EXISTS idx_lineage_src   ON lineage(source_id);
CREATE INDEX IF NOT EXISTS idx_lineage_tgt   ON lineage(target_id);
"""

_VALID_KINDS = {"field_product", "dataset", "run", "model"}
_VALID_SENSITIVITY = {"public", "internal", "restricted"}
_VALID_STATUS = {"active", "archived"}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AssetRecord:
    asset_id: str
    name: str
    kind: str = "field_product"
    description: str = ""
    field_name: str | None = None
    units: str | None = None
    shape: str | None = None
    dtype: str | None = None
    tags: tuple[str, ...] = ()
    quality_score: int = 0
    sensitivity_level: str = "internal"
    source_run_id: str | None = None
    status: str = "active"
    version: str = "1.0.0"
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    key: str
    value: str
    source: str = "manual"
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class LineageRecord:
    source_id: str
    target_id: str
    relation_type: str = "derived_from"
    transformation: str = ""
    resource_type: str = "product"


@dataclass(frozen=True, slots=True)
class QualityCheck:
    check_name: str
    passed: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _decode_json(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _validate_asset(rec: AssetRecord) -> None:
    if not rec.asset_id or not rec.name:
        raise ValueError("asset_id and name are required")
    if rec.kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {sorted(_VALID_KINDS)}")
    if rec.sensitivity_level not in _VALID_SENSITIVITY:
        raise ValueError(f"sensitivity_level must be one of {sorted(_VALID_SENSITIVITY)}")
    if rec.status not in _VALID_STATUS:
        raise ValueError(f"status must be one of {sorted(_VALID_STATUS)}")
    if not 0 <= rec.quality_score <= 100:
        raise ValueError("quality_score must be in [0, 100]")


class FieldDataCatalog:
    """Registry for CFD field-data products, datasets and their lineage."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @classmethod
    def open(cls, db_path: str | Path) -> "FieldDataCatalog":
        conn = sqlite3.connect(str(db_path))
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    # -- assets ------------------------------------------------------------

    def register_asset(self, rec: AssetRecord) -> None:
        _validate_asset(rec)
        now = _now()
        self._conn.execute(
            """INSERT OR REPLACE INTO assets
               (asset_id, name, description, kind, field_name, units, shape,
                dtype, tags, quality_score, sensitivity_level, source_run_id,
                status, version, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.asset_id, rec.name, rec.description, rec.kind,
                rec.field_name, rec.units, rec.shape, rec.dtype,
                _encode_json(list(rec.tags)), rec.quality_score,
                rec.sensitivity_level, rec.source_run_id, rec.status,
                rec.version, now, now,
            ),
        )
        self._conn.commit()

    def register_field_product(self, product: FieldProduct, name: str) -> None:
        """Register a FieldProduct (from data/contracts.py) as an asset."""
        self.register_asset(AssetRecord(
            asset_id=product.product_id,
            name=name,
            kind="field_product",
            field_name=product.field_name,
            units=product.units,
            shape=json.dumps(list(product.shape)),
            dtype=product.dtype,
            source_run_id=product.run_manifest.run_id,
            quality_score=_quality_from_status(product.quality_status),
        ))

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        row = self._conn.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (asset_id,),
        ).fetchone()
        return _row_to_asset(row) if row else None

    def list_assets(
        self,
        kind: str | None = None,
        field_name: str | None = None,
        status: str = "active",
        limit: int = 50,
    ) -> list[AssetRecord]:
        query = "SELECT * FROM assets WHERE 1=1"
        args: list[Any] = []
        if kind is not None:
            query += " AND kind = ?"
            args.append(kind)
        if field_name is not None:
            query += " AND field_name = ?"
            args.append(field_name)
        if status is not None:
            query += " AND status = ?"
            args.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        rows = self._conn.execute(query, args).fetchall()
        return [_row_to_asset(r) for r in rows]

    def count_assets(
        self,
        kind: str | None = None,
        field_name: str | None = None,
        status: str | None = "active",
    ) -> int:
        """Return the number of assets matching the given filters."""
        query = "SELECT COUNT(*) FROM assets WHERE 1=1"
        args: list[Any] = []
        if kind is not None:
            query += " AND kind = ?"
            args.append(kind)
        if field_name is not None:
            query += " AND field_name = ?"
            args.append(field_name)
        if status is not None:
            query += " AND status = ?"
            args.append(status)
        row = self._conn.execute(query, args).fetchone()
        return int(row[0]) if row else 0

    def update_asset(
        self,
        asset_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: Sequence[str] | None = None,
        status: str | None = None,
        quality_score: int | None = None,
    ) -> None:
        cur = self.get_asset(asset_id)
        if cur is None:
            raise KeyError(f"no asset {asset_id}")
        new_name = name if name is not None else cur.name
        new_desc = description if description is not None else cur.description
        new_tags = tuple(tags) if tags is not None else cur.tags
        new_status = status if status is not None else cur.status
        new_score = quality_score if quality_score is not None else cur.quality_score
        rec = AssetRecord(
            asset_id=asset_id, name=new_name, kind=cur.kind,
            description=new_desc, field_name=cur.field_name, units=cur.units,
            shape=cur.shape, dtype=cur.dtype, tags=new_tags,
            quality_score=new_score, sensitivity_level=cur.sensitivity_level,
            source_run_id=cur.source_run_id, status=new_status,
            version=cur.version, created_at=cur.created_at,
            updated_at=cur.updated_at,
        )
        self.register_asset(rec)

    def archive_asset(self, asset_id: str) -> None:
        self._conn.execute(
            "UPDATE assets SET status='archived', updated_at=? WHERE asset_id=?",
            (_now(), asset_id),
        )
        self._conn.commit()

    # -- metadata ----------------------------------------------------------

    def add_metadata(
        self,
        asset_id: str,
        key: str,
        value: str,
        source: str = "manual",
        confidence: float = 1.0,
    ) -> None:
        self._conn.execute(
            """INSERT INTO asset_metadata (asset_id, key, value, source,
                                          confidence, created_at)
               VALUES (?,?,?,?,?,?)""",
            (asset_id, key, value, source, confidence, _now()),
        )
        self._conn.commit()

    def get_metadata(self, asset_id: str) -> list[MetadataRecord]:
        rows = self._conn.execute(
            "SELECT key, value, source, confidence FROM asset_metadata "
            "WHERE asset_id = ? ORDER BY id",
            (asset_id,),
        ).fetchall()
        return [MetadataRecord(r["key"], r["value"], r["source"], r["confidence"])
                for r in rows]

    def delete_metadata(self, asset_id: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM asset_metadata WHERE asset_id = ? AND key = ?",
            (asset_id, key),
        )
        self._conn.commit()

    # -- lineage -----------------------------------------------------------

    def add_lineage(self, rec: LineageRecord) -> None:
        self._conn.execute(
            """INSERT INTO lineage (source_id, target_id, relation_type,
                                    transformation, resource_type, created_at)
               VALUES (?,?,?,?,?,?)""",
            (rec.source_id, rec.target_id, rec.relation_type,
             rec.transformation, rec.resource_type, _now()),
        )
        self._conn.commit()

    def get_lineage(self, asset_id: str) -> list[LineageRecord]:
        rows = self._conn.execute(
            "SELECT * FROM lineage WHERE source_id = ? OR target_id = ? "
            "ORDER BY id",
            (asset_id, asset_id),
        ).fetchall()
        return [LineageRecord(
            r["source_id"], r["target_id"], r["relation_type"],
            r["transformation"], r["resource_type"],
        ) for r in rows]

    def upstream(self, asset_id: str) -> list[str]:
        """All transitive upstream assets (source side of the lineage graph)."""
        seen: set[str] = set()
        frontier = [asset_id]
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            rows = self._conn.execute(
                "SELECT source_id FROM lineage WHERE target_id = ?", (cur,),
            ).fetchall()
            frontier.extend(r["source_id"] for r in rows)
        seen.discard(asset_id)
        return sorted(seen)

    # -- quality -----------------------------------------------------------

    def record_quality(
        self,
        asset_id: str,
        checks: Sequence[QualityCheck],
        status: str | None = None,
    ) -> int:
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        score = round(100 * passed / total) if total else 0
        if status is None:
            status = "passed" if passed == total else (
                "warning" if passed > 0 else "failed"
            )
        self._conn.execute(
            """INSERT INTO quality_reports (asset_id, checks, overall_score,
                                            status, created_at)
               VALUES (?,?,?,?,?)""",
            (
                asset_id,
                _encode_json([_quality_check_to_dict(c) for c in checks]),
                score, status, _now(),
            ),
        )
        self._conn.execute(
            "UPDATE assets SET quality_score = ?, updated_at = ? WHERE asset_id = ?",
            (score, _now(), asset_id),
        )
        self._conn.commit()
        return score

    def get_quality_reports(self, asset_id: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT checks, overall_score, status, created_at FROM quality_reports "
            "WHERE asset_id = ? ORDER BY id DESC LIMIT ?",
            (asset_id, limit),
        ).fetchall()
        return [
            {
                "checks": _decode_json(r["checks"], []),
                "overall_score": r["overall_score"],
                "status": r["status"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality_from_status(status: Any) -> int:
    """Map a ValidationStatus enum to a 0-100 score."""
    name = getattr(status, "value", status)
    return {"passed": 100, "warning": 60, "failed": 0}.get(str(name), 0)


def _quality_check_to_dict(check: QualityCheck) -> dict[str, Any]:
    return {
        "check_name": check.check_name,
        "passed": check.passed,
        "detail": check.detail,
    }


def _row_to_asset(row: sqlite3.Row) -> AssetRecord:
    return AssetRecord(
        asset_id=row["asset_id"],
        name=row["name"],
        kind=row["kind"],
        description=row["description"],
        field_name=row["field_name"],
        units=row["units"],
        shape=row["shape"],
        dtype=row["dtype"],
        tags=tuple(_decode_json(row["tags"], [])),
        quality_score=row["quality_score"],
        sensitivity_level=row["sensitivity_level"],
        source_run_id=row["source_run_id"],
        status=row["status"],
        version=row["version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
