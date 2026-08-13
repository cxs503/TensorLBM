"""Field-data catalog API endpoints.

Thin FastAPI layer over ``tensorlbm.data.catalog.FieldDataCatalog`` and
``tensorlbm.data.quality``.  It exposes asset CRUD (soft delete via archive),
key-value metadata, lineage and field-quality checks under ``/api/data``.

Storage follows the same convention as ``routers/projects.py``: a SQLite
database under ``TENSORLBM_OUTPUT_ROOT`` (default ``/tmp/tensorlbm_platform``)
named ``data_catalog.db``.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..schemas.data_catalog import (
    AssetCreate,
    AssetListResponse,
    AssetOut,
    AssetUpdate,
    LineageCreate,
    LineageOut,
    MetadataCreate,
    MetadataOut,
    QualityCheckItem,
    QualityCheckRequest,
    QualityCheckResponse,
    QualityReportOut,
)
from tensorlbm.data.catalog import (
    AssetRecord,
    FieldDataCatalog,
    LineageRecord,
)
from tensorlbm.data.quality import check_field_product

router = APIRouter()

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_CATALOG_DIR = Path(os.environ.get("TENSORLBM_OUTPUT_ROOT", "/tmp/tensorlbm_platform"))
_DB_PATH = _CATALOG_DIR / "data_catalog.db"

_VALID_STATUS = {"active", "archived"}


def get_catalog() -> Iterator[FieldDataCatalog]:
    """Yield a catalog bound to the platform data catalog database.

    Override this dependency in tests to point at a temporary database.
    """
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    catalog = FieldDataCatalog(conn)
    try:
        yield catalog
    finally:
        catalog.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_asset(catalog: FieldDataCatalog, asset_id: str) -> AssetRecord:
    rec = catalog.get_asset(asset_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    return rec


def _parse_shape(raw: str | None) -> tuple[int, ...] | None:
    """Parse a stored shape string (JSON list or tuple literal) to ints."""
    if not raw:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in value
        ):
            return tuple(int(n) for n in value)
    return None


# ---------------------------------------------------------------------------
# Asset endpoints
# ---------------------------------------------------------------------------


@router.get("/assets", response_model=AssetListResponse)
async def list_assets(
    catalog: FieldDataCatalog = Depends(get_catalog),
    kind: str | None = Query(None),
    field_name: str | None = Query(None),
    status: str = Query("active"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> AssetListResponse:
    """List assets, optionally filtered by kind / field_name / status."""
    if status not in _VALID_STATUS:
        raise HTTPException(
            status_code=422,
            detail=f"status must be one of {sorted(_VALID_STATUS)}",
        )
    records = catalog.list_assets(
        kind=kind,
        field_name=field_name,
        status=status,
        limit=limit + offset,
    )
    page = records[offset : offset + limit]
    return AssetListResponse(
        assets=[AssetOut.model_validate(r) for r in page],
        total=catalog.count_assets(kind=kind, field_name=field_name, status=status),
        limit=limit,
        offset=offset,
    )


@router.post("/assets", response_model=AssetOut, status_code=201)
async def register_asset(
    body: AssetCreate,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> AssetOut:
    """Register (or replace, on id conflict) a field-data asset."""
    rec = AssetRecord(
        asset_id=body.asset_id,
        name=body.name,
        kind=body.kind,
        description=body.description,
        field_name=body.field_name,
        units=body.units,
        shape=body.shape,
        dtype=body.dtype,
        tags=tuple(body.tags),
        quality_score=body.quality_score,
        sensitivity_level=body.sensitivity_level,
        source_run_id=body.source_run_id,
        status=body.status,
        version=body.version,
    )
    try:
        catalog.register_asset(rec)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    stored = _require_asset(catalog, body.asset_id)
    return AssetOut.model_validate(stored)


@router.get("/assets/{asset_id}", response_model=AssetOut)
async def get_asset(
    asset_id: str,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> AssetOut:
    """Return a single asset by id."""
    return AssetOut.model_validate(_require_asset(catalog, asset_id))


@router.put("/assets/{asset_id}", response_model=AssetOut)
async def update_asset(
    asset_id: str,
    body: AssetUpdate,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> AssetOut:
    """Partially update an asset's editable fields."""
    _require_asset(catalog, asset_id)
    try:
        catalog.update_asset(
            asset_id,
            name=body.name,
            description=body.description,
            tags=tuple(body.tags) if body.tags is not None else None,
            status=body.status,
            quality_score=body.quality_score,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return AssetOut.model_validate(_require_asset(catalog, asset_id))


@router.delete("/assets/{asset_id}", status_code=status.HTTP_200_OK)
async def archive_asset(
    asset_id: str,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> dict[str, str]:
    """Archive an asset (soft delete).  Archived assets drop out of the
    default listing but remain queryable with ``status=archived``."""
    _require_asset(catalog, asset_id)
    catalog.archive_asset(asset_id)
    return {"asset_id": asset_id, "status": "archived"}


# ---------------------------------------------------------------------------
# Metadata endpoints
# ---------------------------------------------------------------------------


@router.post("/assets/{asset_id}/metadata", response_model=MetadataOut, status_code=201)
async def add_metadata(
    asset_id: str,
    body: MetadataCreate,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> MetadataOut:
    """Attach a key-value metadata entry to an asset."""
    _require_asset(catalog, asset_id)
    catalog.add_metadata(
        asset_id,
        key=body.key,
        value=body.value,
        source=body.source,
        confidence=body.confidence,
    )
    return MetadataOut(key=body.key, value=body.value, source=body.source, confidence=body.confidence)


@router.get("/assets/{asset_id}/metadata", response_model=list[MetadataOut])
async def get_metadata(
    asset_id: str,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> list[MetadataOut]:
    """List all metadata entries for an asset."""
    _require_asset(catalog, asset_id)
    return [MetadataOut.model_validate(m) for m in catalog.get_metadata(asset_id)]


@router.delete("/assets/{asset_id}/metadata", status_code=status.HTTP_200_OK)
async def delete_metadata(
    asset_id: str,
    key: str = Query(..., min_length=1, max_length=120),
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> dict[str, str]:
    """Delete a single metadata entry by key."""
    _require_asset(catalog, asset_id)
    catalog.delete_metadata(asset_id, key)
    return {"asset_id": asset_id, "key": key, "deleted": "true"}


# ---------------------------------------------------------------------------
# Lineage endpoints
# ---------------------------------------------------------------------------


@router.post("/assets/{asset_id}/lineage", response_model=LineageOut, status_code=201)
async def add_lineage(
    asset_id: str,
    body: LineageCreate,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> LineageOut:
    """Record a lineage edge from the path asset (source) to ``target_id``."""
    _require_asset(catalog, asset_id)
    rec = LineageRecord(
        source_id=asset_id,
        target_id=body.target_id,
        relation_type=body.relation_type,
        transformation=body.transformation,
        resource_type=body.resource_type,
    )
    catalog.add_lineage(rec)
    return LineageOut.model_validate(rec)


@router.get("/assets/{asset_id}/lineage")
async def get_lineage(
    asset_id: str,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> dict[str, Any]:
    """Return direct lineage edges involving the asset plus transitive upstream ids."""
    _require_asset(catalog, asset_id)
    edges = catalog.get_lineage(asset_id)
    return {
        "asset_id": asset_id,
        "lineage": [LineageOut.model_validate(e).model_dump() for e in edges],
        "upstream": catalog.upstream(asset_id),
    }


# ---------------------------------------------------------------------------
# Quality endpoints
# ---------------------------------------------------------------------------


@router.post("/quality/check", response_model=QualityCheckResponse)
async def run_quality_check(
    body: QualityCheckRequest,
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> QualityCheckResponse:
    """Run field-quality checks (finiteness / shape / mass) on asset data.

    The nested ``data`` array is coerced to a float array and validated
    against the asset's declared field name and shape; results are persisted
    as a quality report and the asset's ``quality_score`` is updated.
    """
    asset = _require_asset(catalog, body.asset_id)
    try:
        arr = np.asarray(body.data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"invalid field data: {error}") from error

    field_name = asset.field_name or asset.name
    expected_shape = _parse_shape(asset.shape) or tuple(arr.shape)
    proxy = SimpleNamespace(field_name=field_name, shape=expected_shape)
    result = check_field_product(
        proxy,
        arr,
        mass_field=body.mass_field,
        mass_tol=body.mass_tol,
    )
    score = catalog.record_quality(
        body.asset_id,
        list(result.checks),
        status=result.status,
    )
    return QualityCheckResponse(
        asset_id=body.asset_id,
        overall_score=score,
        status=result.status,
        checks=[
            QualityCheckItem(check_name=c.check_name, passed=c.passed, detail=c.detail)
            for c in result.checks
        ],
    )


@router.get("/quality/{asset_id}/reports", response_model=list[QualityReportOut])
async def get_quality_reports(
    asset_id: str,
    limit: int = Query(10, ge=1, le=100),
    catalog: FieldDataCatalog = Depends(get_catalog),
) -> list[QualityReportOut]:
    """Return the most recent quality reports for an asset."""
    _require_asset(catalog, asset_id)
    return [QualityReportOut.model_validate(r) for r in catalog.get_quality_reports(asset_id, limit)]
