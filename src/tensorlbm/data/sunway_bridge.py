"""SWLBM (Sunway) data bridge: native SWLBM outputs -> TensorLBM R2 products.

The Sunway port runs an independent SWLBM C codebase on the SW26010 many-core
clusters (see ``docs/sunway_data_bridge.md`` for the full format
specification).  It never executes torch; instead its *outputs* are brought
into the platform through the same cold-path R2 contracts
(:class:`~tensorlbm.data.field_r2.FieldDataProductR2` +
:class:`~tensorlbm.data.catalog.FieldDataCatalog`) used by the torch solvers,
so downstream dataset building, quality gating, and lineage queries treat
Sunway runs as first-class citizens.

This wave implements the one fully-specified converter — the SWLBM
``force_history`` CSV emitted per run:

    step,F_x,F_y,F_z,C_D,C_L,C_S,Re,C_F_ITTC,wet_nodes
    0,8.477844650e+04,0.000000000e+00,0.000000000e+00,134.929104362,...
    100,...

Field (lattice) dumps are spec'd but intentionally left as a documented
TODO stub (:func:`convert_swlbm_field_dump`).
"""
from __future__ import annotations

import csv as _csv
import json
import platform
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tensorlbm.data.catalog import (
    AssetRecord,
    FieldDataCatalog,
    LineageRecord,
)
from tensorlbm.data.contracts import FieldProduct
from tensorlbm.data.field_r2 import (
    ArrayEncoding,
    ArrayManifestR2,
    ArrayRole,
    AxisSemantic,
    AxisSpec,
    BlobRef,
    ByteOrder,
    FieldDataProductR2,
    MemoryOrder,
)
from tensorlbm.data.quality import check_field_product
from tensorlbm.data.solver_export import (
    _blob_ref,
    _evidence_payload,
    _product_to_json,
    _torch_version,
    _write_npy,
)
from tensorlbm.runtime import (
    ArtifactManifest,
    MetricEvidence,
    RunManifest,
    ValidationStatus,
)

__all__ = [
    "SWLBM_BRIDGE_SCHEMA",
    "SWLBM_FORCE_COLUMNS",
    "SwlbmForceHistory",
    "convert_swlbm_csv",
    "convert_swlbm_field_dump",
    "parse_swlbm_force_csv",
]

#: Schema token stamped into lineage and catalog metadata.
SWLBM_BRIDGE_SCHEMA = "sunway.swlbm.bridge.v1"

#: Column contract of the SWLBM ``force_history`` CSV (order as emitted).
SWLBM_FORCE_COLUMNS: tuple[str, ...] = (
    "step", "F_x", "F_y", "F_z", "C_D", "C_L", "C_S", "Re", "C_F_ITTC", "wet_nodes",
)

_FLOAT32_NPY = ArrayEncoding("NPY", "float32", MemoryOrder.C, ByteOrder.LITTLE)
_INT32_NPY = ArrayEncoding("NPY", "int32", MemoryOrder.C, ByteOrder.LITTLE)

_METRICS_ARTIFACT_ID = "swlbm:metrics"
_CSV_ARTIFACT_ID = "swlbm:source_csv"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SwlbmForceHistory:
    """Parsed SWLBM force-history time series (lattice units)."""

    steps: np.ndarray            # (N,) int64
    force_xyz: np.ndarray        # (N, 3) float64, F_x/F_y/F_z
    coefficients: np.ndarray     # (N, 3) float64, C_D/C_L/C_S
    reynolds: np.ndarray         # (N,) float64
    cf_ittc: np.ndarray          # (N,) float64
    wet_nodes: np.ndarray        # (N,) int64
    csv_path: str
    csv_sha256: str

    @property
    def rows(self) -> int:
        return int(self.steps.shape[0])


def parse_swlbm_force_csv(csv_path: str | Path) -> SwlbmForceHistory:
    """Parse and validate one SWLBM ``force_history`` CSV.

    Args:
        csv_path: Path to the CSV (header row +
            ``step,F_x,F_y,F_z,C_D,C_L,C_S,Re,C_F_ITTC,wet_nodes``).

    Returns:
        :class:`SwlbmForceHistory` with column-major numpy arrays.

    Raises:
        ValueError: If the header is missing expected columns, the file has
            no data rows, or any value is non-finite (fail closed, matching
            the solver_export gate).
    """
    path = Path(csv_path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    reader = _csv.DictReader(text.splitlines())
    header = reader.fieldnames or []
    missing = [column for column in SWLBM_FORCE_COLUMNS if column not in header]
    if missing:
        raise ValueError(
            f"SWLBM force CSV {path.name}: missing columns {missing}; "
            f"expected {list(SWLBM_FORCE_COLUMNS)} (schema {SWLBM_BRIDGE_SCHEMA})"
        )

    columns: dict[str, list[float]] = {name: [] for name in SWLBM_FORCE_COLUMNS}
    for line_number, row in enumerate(reader, start=2):
        if row.get("step") is None or str(row.get("step", "")).strip() == "":
            continue  # tolerate trailing blank lines
        for name in SWLBM_FORCE_COLUMNS:
            try:
                columns[name].append(float(row[name]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"SWLBM force CSV {path.name}:{line_number}: column "
                    f"{name!r} value {row.get(name)!r} is not numeric"
                ) from error

    rows = len(columns["step"])
    if rows == 0:
        raise ValueError(f"SWLBM force CSV {path.name} has no data rows")

    def array(name: str) -> np.ndarray:
        return np.asarray(columns[name], dtype=np.float64)

    steps = array("step")
    wet_nodes = array("wet_nodes")
    if not np.all(steps == np.floor(steps)) or steps.min() < 0:
        raise ValueError(f"SWLBM force CSV {path.name}: 'step' must be non-negative integers")
    if not np.all(wet_nodes == np.floor(wet_nodes)) or wet_nodes.min() < 0:
        raise ValueError(f"SWLBM force CSV {path.name}: 'wet_nodes' must be non-negative integers")

    floats = {name: array(name) for name in SWLBM_FORCE_COLUMNS if name not in ("step", "wet_nodes")}
    for name, values in floats.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"SWLBM force CSV {path.name}: column {name!r} has non-finite "
                f"values; refusing to bridge a non-PASS product"
            )

    return SwlbmForceHistory(
        steps=steps.astype(np.int64),
        force_xyz=np.stack([floats["F_x"], floats["F_y"], floats["F_z"]], axis=1),
        coefficients=np.stack([floats["C_D"], floats["C_L"], floats["C_S"]], axis=1),
        reynolds=floats["Re"],
        cf_ittc=floats["C_F_ITTC"],
        wet_nodes=wet_nodes.astype(np.int64),
        csv_path=str(path.resolve()),
        csv_sha256=sha256(raw).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Conversion: force-history CSV -> R2 product
# ---------------------------------------------------------------------------

def convert_swlbm_csv(
    catalog: FieldDataCatalog,
    csv_path: str | Path,
    metadata: Mapping[str, Any],
    *,
    blob_root: str | Path | None = None,
) -> str:
    """Register one SWLBM force-history CSV as a PASS-gated R2 product.

    The CSV is parsed (:func:`parse_swlbm_force_csv`), NPY blobs are
    materialised under *blob_root* (default ``<csv dir>/blobs``), the product
    is byte-verified via :meth:`FieldDataProductR2.validate_for_use`, and it
    is registered as a ``field_product`` asset with metadata, lineage, and
    quality checks — exactly like :func:`tensorlbm.data.solver_export.
    register_product` does for torch snapshots.

    Required *metadata* keys: ``run_id``, ``case``, ``code_sha`` (40-hex SWLBM
    source revision).  Optional recognised: ``lattice``, ``collision``,
    ``queue``, ``sunway_host``, ``re``, ``u_in``, ``nu``, ``tau``.

    Returns:
        The ``product_id`` (``"{run_id}:forces"``).
    """
    if not isinstance(catalog, FieldDataCatalog):
        raise TypeError("catalog must be a FieldDataCatalog")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    for key in ("run_id", "case", "code_sha"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata must contain a non-empty string {key!r}")

    history = parse_swlbm_force_csv(csv_path)
    run_id = str(metadata["run_id"])
    case = str(metadata["case"])
    product_id = f"{run_id}:forces"
    if catalog.get_asset(product_id) is not None:
        raise ValueError(
            f"product {product_id!r} is already registered; use a fresh run_id"
        )

    csv_abs = Path(history.csv_path)
    csv_bytes = csv_abs.read_bytes()
    root = Path(blob_root) if blob_root is not None else csv_abs.parent / "blobs"
    blob_dir = root / csv_abs.stem

    timestep_axis = AxisSpec("timestep", AxisSemantic.SAMPLE, history.rows)
    ordered: list[tuple[str, ArrayRole, np.ndarray, str, tuple[str, ...] | None]] = [
        (
            "force_lattice",
            ArrayRole.FEATURE,
            np.ascontiguousarray(history.force_xyz.astype("<f4")),
            "lattice_force",
            ("F_x", "F_y", "F_z"),
        ),
        (
            "force_coefficients",
            ArrayRole.TARGET,
            np.ascontiguousarray(history.coefficients.astype("<f4")),
            "dimensionless",
            ("C_D", "C_L", "C_S"),
        ),
        (
            "wet_nodes",
            ArrayRole.AUXILIARY,
            np.ascontiguousarray(history.wet_nodes.astype("<i4")),
            "1",
            None,
        ),
    ]

    manifests: list[ArrayManifestR2] = []
    payloads: dict[str, bytes] = {}
    for array_id, role, arr, units, labels in ordered:
        payload = _write_npy(blob_dir / f"{array_id}.npy", arr)
        axes = (timestep_axis,)
        if labels is not None:
            axes = axes + (AxisSpec("component", AxisSemantic.COMPONENT, arr.shape[-1]),)
        manifests.append(
            ArrayManifestR2(
                array_id=array_id,
                role=role,
                shape=tuple(arr.shape),
                axes=axes,
                units=units,
                encoding=_FLOAT32_NPY if arr.dtype == np.dtype("<f4") else _INT32_NPY,
                blob_ref=_blob_ref(product_id, array_id, blob_dir / f"{array_id}.npy", payload),
                component_labels=labels,
            )
        )
        payloads[array_id] = payload

    # --- Run manifest: metrics bound to exact-decimal JSON evidence --------
    tail = slice(max(0, history.rows - max(1, history.rows // 10)), None)
    metrics: dict[str, Any] = {
        "rows": float(history.rows),
        "step_last": float(history.steps[-1]),
        "re": float(history.reynolds[-1]),
        "cf_ittc": float(history.cf_ittc[-1]),
        "cd_last": float(history.coefficients[-1, 0]),
        "cl_last": float(history.coefficients[-1, 1]),
        "cs_last": float(history.coefficients[-1, 2]),
        "cd_mean_tail": float(history.coefficients[tail, 0].mean()),
        "f_x_last": float(history.force_xyz[-1, 0]),
        "wet_nodes_last": float(history.wet_nodes[-1]),
    }
    evidence = _evidence_payload(metrics)
    metrics_artifact = ArtifactManifest.from_bytes(
        _METRICS_ARTIFACT_ID, "application/json", evidence
    )
    csv_artifact = ArtifactManifest.from_bytes(
        _CSV_ARTIFACT_ID,
        "text/csv",
        csv_bytes,
        metadata={"schema": SWLBM_BRIDGE_SCHEMA},
    )
    run = RunManifest(
        run_id=run_id,
        model_identity={
            "solver": "SWLBM",
            "case": case,
            "lattice": str(metadata.get("lattice", "D3Q19")),
            "platform": "sunway",
        },
        config=dict(metadata),
        code_sha=str(metadata["code_sha"]),
        environment={
            "device": str(metadata.get("sunway_host", "sw")),
            "python": platform.python_version(),
            "hostname": str(metadata.get("sunway_host", platform.node())),
            "torch": _torch_version(),
        },
        artifacts=(metrics_artifact, csv_artifact),
        metrics=tuple(
            MetricEvidence(key, float(value), "lattice", _METRICS_ARTIFACT_ID, f"/{key}")
            for key, value in metrics.items()
        ),
        validation_status=ValidationStatus.PASS,
        validation_reason=(
            f"sunway bridge gate: {history.rows} force-history rows, "
            f"{len(SWLBM_FORCE_COLUMNS)} columns verified, all values finite"
        ),
    )
    product = FieldDataProductR2(
        product_id=product_id,
        run_manifest=run,
        source_artifact_id=_CSV_ARTIFACT_ID,
        arrays=tuple(manifests),
        lineage={
            "sunway_bridge": {
                "schema": SWLBM_BRIDGE_SCHEMA,
                "csv_path": history.csv_path,
                "csv_sha256": history.csv_sha256,
                "rows": history.rows,
                "columns": list(SWLBM_FORCE_COLUMNS),
                "module": "tensorlbm.data.sunway_bridge",
            }
        },
    )
    # Fail closed: verify manifests against the bytes that landed on disk.
    product.validate_for_use(payloads)

    # --- Catalog registration ---------------------------------------------
    primary = manifests[0]
    catalog.register_asset(
        AssetRecord(
            asset_id=product_id,
            name=f"{case} SWLBM force history",
            kind="field_product",
            description=(
                f"SWLBM force-history bridge of {csv_abs.name} "
                f"({history.rows} samples, {len(manifests)} arrays)"
            ),
            field_name=primary.array_id,
            units=primary.units,
            shape=json.dumps(list(primary.shape)),
            dtype=primary.encoding.dtype,
            tags=("sunway_bridge", "swlbm", f"case:{case}"),
            source_run_id=run_id,
        )
    )
    for key in sorted(metadata):
        value = metadata[key]
        if isinstance(value, (str, int, float, bool)):
            catalog.add_metadata(product_id, key, str(value), source="sunway_bridge")
    catalog.add_metadata(product_id, "csv_path", history.csv_path, source="sunway_bridge")
    catalog.add_metadata(
        product_id, "product_json", _product_to_json(product), source="sunway_bridge"
    )
    catalog.add_lineage(
        LineageRecord(
            source_id=f"run:{run_id}",
            target_id=product_id,
            relation_type="derived_from",
            transformation=f"tensorlbm.data.sunway_bridge {SWLBM_BRIDGE_SCHEMA}",
            resource_type="product",
        )
    )

    # --- Quality checks over the materialised blobs ------------------------
    from io import BytesIO

    checks = []
    loaded = {name: np.load(BytesIO(data), allow_pickle=False) for name, data in payloads.items()}
    for manifest in manifests:
        field = FieldProduct(
            product_id=f"{product_id}:{manifest.array_id}",
            run_manifest=run,
            artifact_id=_CSV_ARTIFACT_ID,
            field_name=manifest.array_id,
            shape=manifest.shape,
            dtype=manifest.encoding.dtype,
            units=manifest.units,
            quality_status=ValidationStatus.PASS,
            lineage={},
        )
        checks.extend(check_field_product(field, loaded[manifest.array_id]).checks)
    catalog.record_quality(product_id, checks)
    return product_id


# ---------------------------------------------------------------------------
# Field dumps (spec'd, TODO in this wave)
# ---------------------------------------------------------------------------

def convert_swlbm_field_dump(*_args: Any, **_kwargs: Any) -> str:
    """SWLBM lattice field-dump converter — specified, not yet implemented.

    The mapping (SWLBM binary/plt dumps -> velocity/rho/solid_mask arrays)
    is defined in ``docs/sunway_data_bridge.md``; implement against that
    spec once a real dump sample has been captured from the cluster.
    """
    raise NotImplementedError(
        "SWLBM field-dump conversion is specified in docs/sunway_data_bridge.md "
        "but not implemented in this wave; only force-history CSV "
        "(convert_swlbm_csv) is available"
    )
