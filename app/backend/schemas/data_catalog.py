"""Pydantic schemas for the field-data catalog API.

These models back ``app/backend/routers/data_catalog.py`` and mirror the
records exposed by ``tensorlbm.data.catalog`` (AssetRecord, MetadataRecord,
LineageRecord, QualityCheck).  They are written independently for
TensorLBM's field-data products.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


class AssetCreate(BaseModel):
    """Payload to register a new field-data asset."""

    asset_id: str = Field(..., min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    kind: str = Field("field_product", max_length=80)
    description: str = Field("", max_length=2000)
    field_name: str | None = Field(None, max_length=120)
    units: str | None = Field(None, max_length=80)
    shape: str | None = Field(None, max_length=200)
    dtype: str | None = Field(None, max_length=80)
    tags: list[str] = Field(default_factory=list)
    quality_score: int = Field(0, ge=0, le=100)
    sensitivity_level: str = Field("internal", max_length=40)
    source_run_id: str | None = Field(None, max_length=200)
    status: str = Field("active", max_length=40)
    version: str = Field("1.0.0", max_length=80)


class AssetUpdate(BaseModel):
    """Partial update payload for an existing asset."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    tags: list[str] | None = None
    status: str | None = Field(None, max_length=40)
    quality_score: int | None = Field(None, ge=0, le=100)


class AssetOut(BaseModel):
    """Serialized asset record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    name: str
    kind: str
    description: str
    field_name: str | None
    units: str | None
    shape: str | None
    dtype: str | None
    tags: list[str]
    quality_score: int
    sensitivity_level: str
    source_run_id: str | None
    status: str
    version: str
    created_at: float
    updated_at: float


class AssetListResponse(BaseModel):
    """Paginated list of assets."""

    assets: list[AssetOut]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class MetadataCreate(BaseModel):
    """Key-value metadata entry attached to an asset."""

    key: str = Field(..., min_length=1, max_length=120)
    value: str = Field(..., max_length=4000)
    source: str = Field("manual", max_length=80)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class MetadataOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    key: str
    value: str
    source: str
    confidence: float


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


class LineageCreate(BaseModel):
    """Create a lineage edge from the path asset (source) to ``target_id``."""

    target_id: str = Field(..., min_length=1, max_length=200)
    relation_type: str = Field("derived_from", max_length=80)
    transformation: str = Field("", max_length=2000)
    resource_type: str = Field("product", max_length=80)


class LineageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source_id: str
    target_id: str
    relation_type: str
    transformation: str
    resource_type: str


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


class QualityCheckRequest(BaseModel):
    """Run field-data quality checks against an asset's field values."""

    asset_id: str = Field(..., min_length=1, max_length=200)
    data: list[Any] = Field(..., description="Nested field array (rows) of numeric values")
    mass_field: bool = Field(False, description="Apply LBM mass-conservation check")
    mass_tol: float = Field(1e-6, gt=0.0, description="Density drift tolerance")


class QualityCheckItem(BaseModel):
    check_name: str
    passed: bool
    detail: str


class QualityCheckResponse(BaseModel):
    asset_id: str
    overall_score: int
    status: str
    checks: list[QualityCheckItem]


class QualityReportOut(BaseModel):
    checks: list[dict[str, Any]]
    overall_score: int
    status: str
    created_at: float
