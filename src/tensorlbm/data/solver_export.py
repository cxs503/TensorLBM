"""Solver→data export: HDF5 field snapshots registered as R2 catalog products.

This module closes the first break of the TensorLBM AI4S loop
(solver → data): before it, no example produced field artifacts that the
``tensorlbm.data`` contracts could ingest — ``FieldDataProductR2`` values
were only ever constructed in tests, so "run a simulation → catalog
registration → training-side discovery" was an empty path.  The two
missing pieces are added here without touching any solver hot path:

* :func:`save_fields_hdf5` — write one simulation snapshot into an HDF5
  file that follows the established :func:`tensorlbm.io.save_hdf5` group
  convention (``/step_{step:06d}``), plus per-snapshot metadata attrs.
* :func:`register_product` — read such a snapshot back, materialise the
  contract-level NPY blobs, build a PASS-gated
  :class:`~tensorlbm.data.field_r2.FieldDataProductR2`
  (RunManifest + ArrayManifestR2 values bound to those blobs), verify it
  byte-for-byte against the blobs on disk, and register it in a
  :class:`~tensorlbm.data.catalog.FieldDataCatalog`.
* :func:`load_product` / :func:`load_product_arrays` — the training-side
  view: reconstruct a registered product from the catalog and load its
  arrays as verified NumPy tensors.

HDF5 layout contract (solver ↔ training side)
---------------------------------------------
One HDF5 file per *run*; every exported snapshot is one group::

    /                        root attrs: export_schema = "tensorlbm.solver-export/v1"
    /step_{step:06d}/        one group per snapshot (tensorlbm.io.save_hdf5 naming)
        rho         float32 (nz, ny, nx)   C order, little-endian
        ux          float32 (nz, ny, nx)
        uy          float32 (nz, ny, nx)
        uz          float32 (nz, ny, nx)   present for 3-D runs
        solid_mask  int8   (nz, ny, nx)    optional; 0 = fluid, 1 = solid
        <extra>     float32 (nz, ny, nx)   optional auxiliary fields
        attrs: "step" plus every caller-supplied scalar attr
               (run_id, case, collision, re, u_in, nu, tau, ...).

2-D fields use ``(ny, nx)`` throughout.  Integer inputs other than
``solid_mask`` are stored as int32; everything else is cast to float32.

Registered R2 arrays per snapshot product (NPY blobs, little-endian
C order, referenced by absolute ``file://`` URIs under
``<h5 dir>/blobs/<h5 stem>/<step group>/<array_id>.npy``):

=================  =========  =====================  =====================
array_id           role       shape                   dtype
=================  =========  =====================  =====================
velocity           FEATURE    spatial + (2 or 3,)     float32
rho                TARGET     spatial                 float32
solid_mask         MASK       spatial                 int32
<extra dataset>    AUXILIARY  spatial                 float32/int32
=================  =========  =====================  =====================

The velocity component axis is last with labels
``("u_x", "u_y", "u_z")`` (3-D) or ``("u_x", "u_y")`` (2-D); spatial axes
are named ``z, y, x`` (3-D) or ``y, x`` (2-D).  A 2-D export therefore
produces exactly the ``velocity`` array the existing training-side
adapter :func:`tensorlbm.ml.torch_materialize.materialize_torch_velocity_snapshots`
expects.  All array values are in lattice units by default
(``units="lattice"``; override per product via the
``velocity_units``/``rho_units`` metadata keys).
"""

from __future__ import annotations

import base64
import json
import math
import platform
import re
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np

from tensorlbm.data.catalog import AssetRecord, FieldDataCatalog, LineageRecord
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
from tensorlbm.runtime import (
    ArtifactManifest,
    MetricEvidence,
    RunManifest,
    ValidationStatus,
)

__all__ = [
    "EXPORT_SCHEMA",
    "snapshot_group",
    "save_fields_hdf5",
    "read_snapshot",
    "register_product",
    "load_product",
    "load_product_arrays",
]

#: Layout/schema marker written to the HDF5 root and snapshot groups.
EXPORT_SCHEMA = "tensorlbm.solver-export/v1"

#: Snapshot group name format (identical to :mod:`tensorlbm.io`).
_GROUP_FMT = "step_{step:06d}"

#: Artifact ids inside the per-snapshot RunManifest.
_METRICS_ARTIFACT_ID = "snapshot-metrics"
_H5_ARTIFACT_ID = "hdf5-source"

_VELOCITY_ID = "velocity"
_DEFAULT_UNITS = "lattice"
_CODE_SHA = re.compile(r"[0-9a-f]{40}\Z")

_FLOAT32_NPY = ArrayEncoding.NPY_FLOAT32_C_LITTLE
_INT32_NPY = ArrayEncoding("NPY", "int32", MemoryOrder.C, ByteOrder.LITTLE)

_SPATIAL_AXIS_NAMES = {3: ("z", "y", "x"), 2: ("y", "x")}
_VELOCITY_LABELS = {3: ("u_x", "u_y", "u_z"), 2: ("u_x", "u_y")}


# ---------------------------------------------------------------------------
# HDF5 snapshot writing/reading
# ---------------------------------------------------------------------------


def snapshot_group(step: int) -> str:
    """Return the canonical HDF5 group name for *step* (``step_000123``)."""
    return _GROUP_FMT.format(step=int(step))


def _as_numpy(value: Any, name: str) -> np.ndarray:
    """Convert a torch/NumPy field to a contiguous contract dtype."""
    if hasattr(value, "detach"):  # torch.Tensor — no hard torch import needed
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype.kind == "f":
        arr = arr.astype("<f4")
    elif arr.dtype.kind in {"i", "u", "b"}:
        # The solid mask is stored compactly as int8; every other integer
        # field maps to the contract's int32.
        arr = arr.astype("i1" if name == "solid_mask" else "<i4")
    else:
        raise ValueError(f"field {name!r} has unsupported dtype {arr.dtype}")
    return np.ascontiguousarray(arr)


def _attr_value(value: Any, key: str) -> Any:
    """Validate one HDF5 attr value: JSON-safe scalars or lists of scalars."""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(f"attr {key!r} must not be an empty sequence")
        return [_attr_value(item, f"{key}[]") for item in value]
    raise ValueError(
        f"attr {key!r} must be a str/int/float/bool scalar or a list of "
        f"scalars; got {type(value).__name__}"
    )


def save_fields_hdf5(
    path: str | Path,
    arrays: Mapping[str, Any],
    attrs: Mapping[str, Any],
) -> Path:
    """Write one field snapshot to *path* under ``/step_{step:06d}``.

    Args:
        path: HDF5 file (created/appended; sibling snapshots share it).
        arrays: mapping of dataset name → field.  Canonical names are
            ``rho``, ``ux``, ``uy``, ``uz`` (float32) and ``solid_mask``
            (int8); additional float/int fields are stored verbatim.
            Values may be torch tensors or NumPy arrays; all fields must
            share one 2-D ``(ny, nx)`` or 3-D ``(nz, ny, nx)`` shape.
        attrs: per-snapshot metadata attrs written to the group.  Must
            contain the positive integer ``step``; every value must be a
            JSON-safe scalar (or list of scalars).

    Returns:
        The resolved file path (the snapshot lives at
        ``/step_{step:06d}`` inside it, overwriting any previous group
        of the same name, matching :func:`tensorlbm.io.save_hdf5`).
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment-specific
        raise ImportError("h5py is required for HDF5 export: pip install h5py") from error

    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("arrays must be a non-empty mapping of field name to field")
    if not isinstance(attrs, Mapping):
        raise TypeError("attrs must be a mapping")
    step = attrs.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("attrs must contain a non-negative integer 'step'")
    for key in arrays:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("array names must be non-empty strings")

    fields = {name: _as_numpy(value, name) for name, value in arrays.items()}
    shapes = {tuple(arr.shape) for arr in fields.values()}
    if len(shapes) != 1:
        raise ValueError(f"all fields must share one shape; got {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) not in (2, 3) or any(dim <= 0 for dim in shape):
        raise ValueError(f"fields must be 2-D or 3-D with positive dims; got {shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        handle.attrs.setdefault("export_schema", EXPORT_SCHEMA)
        group_name = snapshot_group(step)
        if group_name in handle:
            del handle[group_name]
        group = handle.create_group(group_name)
        for name, arr in fields.items():
            group.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        group.attrs["step"] = int(step)
        group.attrs["export_schema"] = EXPORT_SCHEMA
        for key, value in attrs.items():
            group.attrs[key] = _attr_value(value, key)
    return path.resolve()


def _normalise_attr(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalise_attr(item) for item in value.tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        return value.item()
    return value


def read_snapshot(h5_path: str | Path, step: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the ``/step_{step:06d}`` snapshot back as NumPy arrays + attrs.

    Raises:
        FileNotFoundError: if *h5_path* does not exist.
        ValueError: if the snapshot group is missing.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment-specific
        raise ImportError("h5py is required for HDF5 export: pip install h5py") from error

    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"no HDF5 file at {path}")
    with h5py.File(path, "r") as handle:
        group_name = snapshot_group(step)
        if group_name not in handle:
            raise ValueError(f"no snapshot group /{group_name} in {path}")
        group = handle[group_name]
        arrays = {name: np.asarray(group[name][...]) for name in group}
        attrs = {key: _normalise_attr(value) for key, value in group.attrs.items()}
    return arrays, attrs


# ---------------------------------------------------------------------------
# Metric evidence (exact-decimal JSON)
# ---------------------------------------------------------------------------


def _exact_number(value: float) -> str:
    """Decimal text that re-parses to exactly this binary float.

    ``RunManifest.verify_metric_evidence`` re-parses metric artifacts with
    ``Decimal`` and requires the parsed number to equal
    ``Decimal.from_float(metric.value)``.  ``json.dumps`` emits the
    *shortest* round-trip repr (e.g. ``"1.0003"``), which does *not*
    satisfy that check, so metric numbers are serialised as the exact
    decimal expansion of the binary float instead.
    """
    return format(Decimal.from_float(float(value)), "f")


def _evidence_payload(metrics: Mapping[str, Any]) -> bytes:
    """Serialise a flat metrics mapping as JSON with exact float text."""
    if not metrics:
        raise ValueError("metrics evidence must not be empty")
    items: list[str] = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {key!r} must be a non-boolean int or float")
        if not math.isfinite(float(value)):
            raise ValueError(f"metric {key!r} must be finite")
        if isinstance(value, int):
            if abs(value) > 2**53:
                raise ValueError(f"integer metric {key!r} exceeds float64 exact range")
            text = str(value)
        else:
            text = _exact_number(value)
        items.append(f"{json.dumps(key)}: {text}")
    return ("{" + ", ".join(items) + "}").encode("utf-8")


# ---------------------------------------------------------------------------
# Array manifest construction
# ---------------------------------------------------------------------------


def _spatial_axes(shape: tuple[int, ...]) -> tuple[AxisSpec, ...]:
    names = _SPATIAL_AXIS_NAMES[len(shape)]
    return tuple(AxisSpec(name, AxisSemantic.SPATIAL, length) for name, length in zip(names, shape))


def _write_npy(path: Path, arr: np.ndarray) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.lib.format.write_array(handle, arr, allow_pickle=False)
    return path.read_bytes()


def _blob_ref(product_id: str, array_id: str, path: Path, payload: bytes) -> BlobRef:
    return BlobRef(
        blob_id=f"{product_id}.{array_id}",
        uri=path.resolve().as_uri(),
        byte_size=len(payload),
        sha256=sha256(payload).hexdigest(),
        media_type="application/x-npy",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _require_metadata(metadata: Mapping[str, Any]) -> None:
    for key in ("run_id", "case", "code_sha"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata must contain a non-empty string {key!r}")
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("metadata must contain a non-negative integer 'step'")
    if not _CODE_SHA.fullmatch(metadata["code_sha"]):
        raise ValueError("metadata['code_sha'] must be exactly 40 lowercase hex characters")


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # pragma: no cover - torch always present in practice
        return "unavailable"


def register_product(
    catalog: FieldDataCatalog,
    h5_path: str | Path,
    metadata: Mapping[str, Any],
    *,
    blob_root: str | Path | None = None,
    mass_tol: float = 1e-4,
    extra_metrics: Mapping[str, float] | None = None,
) -> str:
    """Register one HDF5 snapshot as a PASS-gated :class:`FieldDataProductR2`.

    The snapshot at ``/step_{step:06d}`` in *h5_path* is read back, the
    R2 NPY blobs are materialised under *blob_root* (default
    ``<h5 dir>/blobs``), the product is verified byte-for-byte via
    :meth:`FieldDataProductR2.validate_for_use` against the blobs on
    disk, and the result is registered as a ``field_product`` asset with
    metadata rows, lineage, and a quality report.

    Required *metadata* keys (all values JSON-safe):

    * ``run_id``, ``case`` — identity/grouping;
    * ``step`` — non-negative int selecting the HDF5 snapshot group;
    * ``code_sha`` — 40-char lowercase hex identifying the solver code
      (e.g. the git commit of the running tree).

    Optional recognised keys: ``collision``, ``lattice``,
    ``boundary_type``, ``device``, ``re``, ``u_in``, ``nu``, ``tau``,
    ``n_steps``, ``velocity_units``, ``rho_units``.  Scalar entries are
    also written as catalog metadata rows for training-side queries.

    Args:
        catalog: the :class:`FieldDataCatalog` to register into.
        h5_path: HDF5 file written by :func:`save_fields_hdf5`.
        metadata: per-snapshot registration metadata (see above).
        blob_root: root directory for NPY blobs (default: sibling
            ``blobs/`` directory of *h5_path*).
        mass_tol: maximum tolerated ``|mean(rho) - 1|`` drift for the
            export quality gate (lattice units).
        extra_metrics: additional finite numeric metrics folded into the
            run-manifest evidence artifact.

    Returns:
        The ``product_id`` (``"{run_id}:{step:06d}"``).

    Raises:
        ValueError: if the snapshot has non-finite values, the density
            drift exceeds *mass_tol*, required metadata is missing, or a
            product with the same id is already registered.
    """
    if not isinstance(catalog, FieldDataCatalog):
        raise TypeError("catalog must be a FieldDataCatalog")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    _require_metadata(metadata)

    step = int(metadata["step"])
    group_name = snapshot_group(step)
    arrays, attrs = read_snapshot(h5_path, step)
    if "step" in attrs and int(attrs["step"]) != step:
        raise ValueError(f"metadata step {step} does not match HDF5 snapshot step {attrs['step']}")
    if "run_id" in attrs and str(attrs["run_id"]) != str(metadata["run_id"]):
        raise ValueError(
            f"metadata run_id {metadata['run_id']!r} does not match HDF5 "
            f"snapshot run_id {attrs['run_id']!r}"
        )
    if not {"ux", "uy"} <= arrays.keys():
        raise ValueError(f"snapshot /{group_name} must contain at least 'ux' and 'uy'")

    # --- Export quality gate (PASS is declared, never inferred) ---
    non_finite = {
        name: int(np.count_nonzero(~np.isfinite(arr)))
        for name, arr in arrays.items()
        if arr.dtype.kind == "f"
    }
    bad = {name: count for name, count in non_finite.items() if count}
    if bad:
        raise ValueError(
            f"snapshot /{group_name} has non-finite values ({bad}); "
            f"refusing to register a non-PASS product"
        )
    rho_drift: float | None = None
    if "rho" in arrays:
        rho_drift = abs(float(arrays["rho"].astype(np.float64).mean()) - 1.0)
        if rho_drift > mass_tol:
            raise ValueError(
                f"snapshot /{group_name} density drift |<rho>-1|={rho_drift:.3e} "
                f"exceeds mass_tol={mass_tol:.1e}"
            )

    # --- Field layout ---
    spatial = arrays["ux"].shape
    if len(spatial) not in (2, 3):
        raise ValueError(f"snapshot fields must be 2-D or 3-D; got {spatial}")
    components = tuple(name for name in ("ux", "uy", "uz") if name in arrays)
    velocity = np.ascontiguousarray(
        np.stack([arrays[name] for name in components], axis=-1).astype("<f4")
    )

    run_id = str(metadata["run_id"])
    case = str(metadata["case"])
    product_id = f"{run_id}:{step:06d}"
    if catalog.get_asset(product_id) is not None:
        raise ValueError(
            f"product {product_id!r} is already registered; use a fresh run_id for a new export"
        )

    h5_abs = Path(h5_path).resolve()
    h5_bytes = h5_abs.read_bytes()
    h5_sha = sha256(h5_bytes).hexdigest()
    root = Path(blob_root) if blob_root is not None else h5_abs.parent / "blobs"
    blob_dir = root / h5_abs.stem / group_name

    units_velocity = str(metadata.get("velocity_units", _DEFAULT_UNITS))
    units_rho = str(metadata.get("rho_units", _DEFAULT_UNITS))

    # --- Blobs + array manifests (canonical order first) ---
    ordered: list[tuple[str, ArrayRole, np.ndarray, str]] = [
        (
            _VELOCITY_ID,
            ArrayRole.FEATURE,
            velocity,
            units_velocity,
        )
    ]
    if "rho" in arrays:
        ordered.append(
            ("rho", ArrayRole.TARGET, np.ascontiguousarray(arrays["rho"].astype("<f4")), units_rho)
        )
    if "solid_mask" in arrays:
        mask_arr = np.ascontiguousarray(arrays["solid_mask"].astype("<i4"))
        ordered.append(("solid_mask", ArrayRole.MASK, mask_arr, "1"))
    for name in sorted(arrays):
        if name in {"ux", "uy", "uz", "rho", "solid_mask"}:
            continue
        if name in {_VELOCITY_ID, "rho", "solid_mask"}:
            raise ValueError(f"dataset {name!r} collides with a canonical registered array id")
        arr = arrays[name]
        if arr.dtype.kind == "f":
            arr = arr.astype("<f4")
        else:
            arr = arr.astype("<i4")
        ordered.append((name, ArrayRole.AUXILIARY, np.ascontiguousarray(arr), _DEFAULT_UNITS))

    manifests: list[ArrayManifestR2] = []
    payloads: dict[str, bytes] = {}
    for array_id, role, arr, units in ordered:
        payload = _write_npy(blob_dir / f"{array_id}.npy", arr)
        encoding = _FLOAT32_NPY if arr.dtype == np.dtype("<f4") else _INT32_NPY
        axes = _spatial_axes(arr.shape[:-1] if array_id == _VELOCITY_ID else arr.shape)
        labels = None
        if array_id == _VELOCITY_ID:
            labels = _VELOCITY_LABELS[len(components)]
            axes = axes + (AxisSpec("component", AxisSemantic.COMPONENT, arr.shape[-1]),)
        manifests.append(
            ArrayManifestR2(
                array_id=array_id,
                role=role,
                shape=tuple(arr.shape),
                axes=axes,
                units=units,
                encoding=encoding,
                blob_ref=_blob_ref(product_id, array_id, blob_dir / f"{array_id}.npy", payload),
                component_labels=labels,
            )
        )
        payloads[array_id] = payload

    # --- Run manifest: metrics bound to exact-decimal JSON evidence ---
    u_max = float(np.linalg.norm(velocity.astype(np.float64), axis=-1).max())
    metrics: dict[str, Any] = {"step": step, "u_max": u_max}
    if rho_drift is not None:
        metrics["rho_mean"] = float(arrays["rho"].astype(np.float64).mean())
    if "solid_mask" in arrays:
        metrics["solid_fraction"] = float(arrays["solid_mask"].astype(np.float64).mean())
    if extra_metrics:
        metrics.update(extra_metrics)
    evidence = _evidence_payload(metrics)
    metric_records = tuple(
        MetricEvidence(key, float(value), "lattice", _METRICS_ARTIFACT_ID, f"/{key}")
        for key, value in metrics.items()
    )
    metrics_artifact = ArtifactManifest.from_bytes(
        _METRICS_ARTIFACT_ID, "application/json", evidence
    )
    h5_artifact = ArtifactManifest.from_bytes(
        _H5_ARTIFACT_ID,
        "application/json",
        json.dumps(
            {
                "h5_path": str(h5_abs),
                "group": group_name,
                "sha256": h5_sha,
                "size_bytes": len(h5_bytes),
            },
            sort_keys=True,
        ).encode("utf-8"),
        metadata={"schema": EXPORT_SCHEMA},
    )
    run = RunManifest(
        run_id=run_id,
        model_identity={
            "solver": "TensorLBM",
            "case": case,
            "collision": str(metadata.get("collision", "unspecified")),
            "lattice": str(metadata.get("lattice", "D3Q19")),
        },
        config=dict(metadata),
        code_sha=str(metadata["code_sha"]),
        environment={
            "device": str(metadata.get("device", "unspecified")),
            "python": platform.python_version(),
            "hostname": platform.node(),
            "torch": _torch_version(),
        },
        artifacts=(metrics_artifact, h5_artifact),
        metrics=metric_records,
        validation_status=ValidationStatus.PASS,
        validation_reason=(
            f"solver export gate: {len(ordered)} arrays finite"
            + (
                f", density drift {rho_drift:.2e} <= {mass_tol:.1e}"
                if rho_drift is not None
                else ""
            )
        ),
    )
    product = FieldDataProductR2(
        product_id=product_id,
        run_manifest=run,
        source_artifact_id=_H5_ARTIFACT_ID,
        arrays=tuple(manifests),
        lineage={
            "export": {
                "schema": EXPORT_SCHEMA,
                "h5_path": str(h5_abs),
                "h5_group": group_name,
                "h5_sha256": h5_sha,
                "module": "tensorlbm.data.solver_export",
            }
        },
    )
    # Fail closed: verify the registered manifests against the blob bytes
    # that actually landed on disk before anything reaches the catalog.
    product.validate_for_use(payloads)

    # --- Catalog registration ---
    primary = manifests[0]
    catalog.register_asset(
        AssetRecord(
            asset_id=product_id,
            name=f"{case} {group_name} fields",
            kind="field_product",
            description=(
                f"Solver export of snapshot /{group_name} from {h5_abs.name} "
                f"({len(manifests)} arrays)"
            ),
            field_name=primary.array_id,
            units=primary.units,
            shape=json.dumps(list(primary.shape)),
            dtype=primary.encoding.dtype,
            tags=("solver_export", f"case:{case}"),
            source_run_id=run_id,
        )
    )
    for key in sorted(metadata):
        value = metadata[key]
        if isinstance(value, (str, int, float, bool)):
            catalog.add_metadata(product_id, key, str(value), source="solver_export")
    catalog.add_metadata(product_id, "h5_path", str(h5_abs), source="solver_export")
    catalog.add_metadata(
        product_id, "product_json", _product_to_json(product), source="solver_export"
    )
    catalog.add_lineage(
        LineageRecord(
            source_id=f"run:{run_id}",
            target_id=product_id,
            relation_type="derived_from",
            transformation=f"tensorlbm.data.solver_export {group_name}",
            resource_type="product",
        )
    )

    # --- Existing quality checks over the materialised blobs ---
    checks = []
    loaded = {name: np.load(BytesIO(data), allow_pickle=False) for name, data in payloads.items()}
    for manifest in manifests:
        field = FieldProduct(
            product_id=f"{product_id}:{manifest.array_id}",
            run_manifest=run,
            artifact_id=_H5_ARTIFACT_ID,
            field_name=manifest.array_id,
            shape=manifest.shape,
            dtype=manifest.encoding.dtype,
            units=manifest.units,
            quality_status=ValidationStatus.PASS,
            lineage={},
        )
        checks.extend(
            check_field_product(
                field,
                loaded[manifest.array_id],
                mass_field=manifest.array_id == "rho",
                mass_tol=mass_tol,
            ).checks
        )
    catalog.record_quality(product_id, checks)
    return product_id


# ---------------------------------------------------------------------------
# Training-side reconstruction
# ---------------------------------------------------------------------------


def _read_blob(blob: BlobRef) -> bytes:
    parsed = urlparse(blob.uri)
    if parsed.scheme != "file" or parsed.netloc:
        raise ValueError(f"unsupported blob uri {blob.uri!r}")
    data = Path(parsed.path).read_bytes()
    if len(data) != blob.byte_size:
        raise ValueError(
            f"blob {blob.blob_id} size {len(data)} does not match manifest "
            f"byte_size {blob.byte_size}"
        )
    return data


def load_product_arrays(product: FieldDataProductR2) -> dict[str, np.ndarray]:
    """Load and verify every array of *product* from its blob URIs.

    The full byte-level contract check
    (:meth:`FieldDataProductR2.validate_for_use`) runs before decoding, so
    a tampered or truncated blob fails here instead of silently feeding
    training.
    """
    if not isinstance(product, FieldDataProductR2):
        raise TypeError("product must be a FieldDataProductR2")
    payloads = {array.array_id: _read_blob(array.blob_ref) for array in product.arrays}
    product.validate_for_use(payloads)
    return {
        array_id: np.load(BytesIO(data), allow_pickle=False) for array_id, data in payloads.items()
    }


def load_product(catalog: FieldDataCatalog, product_id: str) -> FieldDataProductR2:
    """Reconstruct a registered :class:`FieldDataProductR2` from the catalog.

    The serialized product is stored in the ``product_json`` metadata row
    written by :func:`register_product`; reconstruction revalidates every
    contract invariant (construction of the dataclasses does that).
    """
    if catalog.get_asset(product_id) is None:
        raise KeyError(f"no product {product_id!r} in catalog")
    for record in catalog.get_metadata(product_id):
        if record.key == "product_json":
            return _product_from_json(record.value)
    raise ValueError(f"product {product_id!r} has no 'product_json' metadata row")


# ---------------------------------------------------------------------------
# Product JSON (de)serialisation
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot serialise product metadata value {type(value).__name__}")


def _product_to_json(product: FieldDataProductR2) -> str:
    run = product.run_manifest
    document = {
        "schema": EXPORT_SCHEMA,
        "product_id": product.product_id,
        "source_artifact_id": product.source_artifact_id,
        "run": {
            "run_id": run.run_id,
            "model_identity": _jsonable(run.model_identity),
            "config": _jsonable(run.config),
            "code_sha": run.code_sha,
            "environment": _jsonable(run.environment),
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "media_type": artifact.media_type,
                    "payload_b64": base64.b64encode(artifact.payload).decode("ascii"),
                    "metadata": _jsonable(artifact.metadata),
                }
                for artifact in run.artifacts
            ],
            "metrics": [
                {
                    "metric_id": metric.metric_id,
                    "value": metric.value,
                    "unit": metric.unit,
                    "artifact_id": metric.artifact_id,
                    "evidence_pointer": metric.evidence_pointer,
                }
                for metric in run.metrics
            ],
            "validation_status": run.validation_status.value,
            "validation_reason": run.validation_reason,
        },
        "arrays": [
            {
                "array_id": array.array_id,
                "role": array.role.value,
                "shape": list(array.shape),
                "axes": [
                    {"name": axis.name, "semantic": axis.semantic.value, "length": axis.length}
                    for axis in array.axes
                ],
                "units": array.units,
                "component_labels": list(array.component_labels)
                if array.component_labels is not None
                else None,
                "encoding": {
                    "format": array.encoding.format,
                    "dtype": array.encoding.dtype,
                    "order": array.encoding.order.value,
                    "byte_order": array.encoding.byte_order.value,
                },
                "blob_ref": {
                    "blob_id": array.blob_ref.blob_id,
                    "uri": array.blob_ref.uri,
                    "byte_size": array.blob_ref.byte_size,
                    "sha256": array.blob_ref.sha256,
                    "media_type": array.blob_ref.media_type,
                },
            }
            for array in product.arrays
        ],
        "lineage": _jsonable(product.lineage),
    }
    return json.dumps(document, sort_keys=True)


def _product_from_json(text: str) -> FieldDataProductR2:
    document = json.loads(text)
    run_data = document["run"]
    artifacts = tuple(
        ArtifactManifest.from_bytes(
            item["artifact_id"],
            item["media_type"],
            base64.b64decode(item["payload_b64"]),
            item.get("metadata") or {},
        )
        for item in run_data["artifacts"]
    )
    run = RunManifest(
        run_id=run_data["run_id"],
        model_identity=run_data["model_identity"],
        config=run_data["config"],
        code_sha=run_data["code_sha"],
        environment=run_data["environment"],
        artifacts=artifacts,
        metrics=tuple(
            MetricEvidence(
                item["metric_id"],
                float(item["value"]),
                item["unit"],
                item["artifact_id"],
                item["evidence_pointer"],
            )
            for item in run_data["metrics"]
        ),
        validation_status=ValidationStatus(run_data["validation_status"]),
        validation_reason=run_data["validation_reason"],
    )
    arrays = tuple(
        ArrayManifestR2(
            array_id=item["array_id"],
            role=ArrayRole(item["role"]),
            shape=tuple(item["shape"]),
            axes=tuple(
                AxisSpec(axis["name"], AxisSemantic(axis["semantic"]), axis["length"])
                for axis in item["axes"]
            ),
            units=item["units"],
            encoding=ArrayEncoding(
                item["encoding"]["format"],
                item["encoding"]["dtype"],
                MemoryOrder(item["encoding"]["order"]),
                ByteOrder(item["encoding"]["byte_order"]),
            ),
            blob_ref=BlobRef(
                item["blob_ref"]["blob_id"],
                item["blob_ref"]["uri"],
                item["blob_ref"]["byte_size"],
                item["blob_ref"]["sha256"],
                item["blob_ref"]["media_type"],
            ),
            component_labels=tuple(item["component_labels"])
            if item["component_labels"] is not None
            else None,
        )
        for item in document["arrays"]
    )
    return FieldDataProductR2(
        product_id=document["product_id"],
        run_manifest=run,
        source_artifact_id=document["source_artifact_id"],
        arrays=arrays,
        lineage=document["lineage"],
    )
