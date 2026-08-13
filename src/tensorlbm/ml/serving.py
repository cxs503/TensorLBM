"""Model serving layer: registry, inference, and ONNX export (clean-room).

This module bridges the trained AI models of :mod:`tensorlbm.ai` (the eddy
viscosity MLP and the self-supervised flow transformer) to a serving surface:

* :class:`ModelRegistry` persists and queries model records by reusing the
  ``models`` table managed by :mod:`tensorlbm.ai.database`.
* :class:`InferenceService` loads a registered model back into memory and runs
  forward inference.
* :func:`export_onnx` serialises a PyTorch model to the portable ONNX format.

Only the Python standard library, ``torch``, ``numpy`` and the ``tensorlbm.ai``
package are required.  The ``onnx`` package is an *optional* dependency: when it
is missing the export path raises a clear :class:`OnnxUnavailableError` instead
of failing obscurely or fabricating output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from tensorlbm.ai import database as _db

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "InferenceService",
    "ModelMetadata",
    "ModelNotFoundError",
    "ModelRegistry",
    "OnnxUnavailableError",
    "export_onnx",
    "infer_io_shapes",
    "onnx_available",
]

# ---------------------------------------------------------------------------
# Optional-dependency detection
# ---------------------------------------------------------------------------

_ONNX_IMPORT_ERROR: Exception | None
try:
    import onnx as _onnx  # noqa: F401  (only used to signal availability)

    _ONNX_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - depends on environment
    _onnx = None
    _ONNX_IMPORT_ERROR = _exc


def onnx_available() -> bool:
    """Return ``True`` when the ``onnx`` package is importable."""
    return _onnx is not None


class OnnxUnavailableError(RuntimeError):
    """Raised when ONNX export is requested but the ``onnx`` package is absent."""

    def __init__(self) -> None:
        super().__init__(
            "ONNX export requires the optional 'onnx' package. "
            "Install it with: pip install onnx"
        )


class ModelNotFoundError(KeyError):
    """Raised when a requested model id is not present in the registry."""


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

# Families understood by the loader inside :class:`InferenceService`.
FAMILY_EDDY_MLP = "eddy_viscosity_mlp"
FAMILY_FLOW_TRANSFORMER = "flow_transformer_ssl"


@dataclass(frozen=True)
class ModelMetadata:
    """Serving descriptor attached to every registered model.

    ``input_shapes`` / ``output_shapes`` map a tensor name to a shape list in
    which ``None`` marks a dynamic axis (typically the batch dimension).
    ``lineage`` records where the training data came from so a deployed model
    remains traceable back to its dataset.
    """

    version: str = "1"
    framework: str = "torch"
    family: str = FAMILY_EDDY_MLP
    input_shapes: Mapping[str, Sequence[int | None]] = field(
        default_factory=dict
    )
    output_shapes: Mapping[str, Sequence[int | None]] = field(
        default_factory=dict
    )
    lineage: Mapping[str, Any] = field(default_factory=dict)
    arch: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the descriptor stored inside ``models.arch_json``."""
        return {
            "family": self.family,
            "framework": self.framework,
            "version": self.version,
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "lineage": dict(self.lineage),
            "arch": dict(self.arch),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelMetadata:
        """Reconstruct from a descriptor dict (tolerates missing fields)."""
        return cls(
            version=str(data.get("version", "1")),
            framework=str(data.get("framework", "torch")),
            family=str(data.get("family", FAMILY_EDDY_MLP)),
            input_shapes=dict(data.get("input_shapes") or {}),
            output_shapes=dict(data.get("output_shapes") or {}),
            lineage=dict(data.get("lineage") or {}),
            arch=dict(data.get("arch") or {}),
        )

    @classmethod
    def from_model_record(cls, record: Mapping[str, Any]) -> ModelMetadata:
        """Build from a decoded ``models`` row (its ``arch_json`` lives under
        ``record["arch"]`` after ``database._rows_to_dicts`` has run)."""
        return cls.from_dict(record.get("arch") or {})


def infer_io_shapes(
    model: nn.Module,
    example_inputs: torch.Tensor | Sequence[torch.Tensor],
    *,
    dynamic_batch: bool = True,
) -> tuple[dict[str, list[int | None]], dict[str, list[int | None]]]:
    """Run one forward pass and derive input/output shape descriptors.

    This is a pure-PyTorch helper (no ``onnx`` dependency) used both by
    :func:`export_onnx` and by callers that want to fill :class:`ModelMetadata`
    before registering a model.
    """
    inputs = _as_tensor_tuple(example_inputs)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(*inputs)
    finally:
        if was_training:
            model.train()
    outputs = _as_tensor_tuple(outputs)

    def _shape(t: torch.Tensor) -> list[int | None]:
        dims: list[int | None] = [int(d) for d in t.shape]
        if dynamic_batch and dims:
            dims[0] = None
        return dims

    in_shapes = {f"input_{i}": _shape(t) for i, t in enumerate(inputs)}
    out_shapes = {f"output_{i}": _shape(t) for i, t in enumerate(outputs)}
    return in_shapes, out_shapes


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """SQLite-backed model registry reusing ``tensorlbm.ai.database``.

    Records live in the ``models`` table.  The serving descriptor (input/output
    shapes, lineage, version, family) is stored inside the ``arch_json`` column
    under the reserved keys documented in ``docs/plans/model-serving-cleanroom-spec.md``.
    """

    def __init__(self, db: _db.LBMDatabase) -> None:
        self._db = db

    @classmethod
    def open(cls, db_path: str | Path) -> ModelRegistry:
        """Open (and if necessary create) a registry at ``db_path``."""
        return cls(_db.LBMDatabase.open(db_path))

    def close(self) -> None:
        self._db.close()

    # -- persistence ------------------------------------------------------
    def register_model(
        self,
        name: str,
        path: str | Path,
        arch: Mapping[str, Any] | None = None,
        *,
        dataset_id: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        version: str = "1",
        framework: str = "torch",
        family: str = FAMILY_EDDY_MLP,
        input_shapes: Mapping[str, Sequence[int | None]] | None = None,
        output_shapes: Mapping[str, Sequence[int | None]] | None = None,
        lineage: Mapping[str, Any] | None = None,
    ) -> int:
        """Register a trained model and return its primary key.

        The ``arch`` mapping holds the raw architecture hyper-parameters (e.g.
        ``asdict(ModelArch(...))``); serving metadata is merged alongside it
        inside ``arch_json``.  Returns the new ``models.id``.
        """
        metadata = ModelMetadata(
            version=str(version),
            framework=str(framework),
            family=str(family),
            input_shapes=dict(input_shapes or {}),
            output_shapes=dict(output_shapes or {}),
            lineage=dict(lineage or {}),
            arch=dict(arch or {}),
        )
        return self._db.insert_model(
            name=str(name),
            path=str(path),
            arch=metadata.to_dict(),
            dataset_id=dataset_id,
            metrics=dict(metrics) if metrics else None,
        )

    def get_model(self, model_id: int) -> dict[str, Any] | None:
        """Return a decoded ``models`` row, or ``None`` when absent."""
        return self._db.get_model_record(model_id)

    def get_model_metadata(self, model_id: int) -> ModelMetadata | None:
        """Return the serving descriptor for a model, or ``None``."""
        record = self.get_model(model_id)
        if record is None:
            return None
        return ModelMetadata.from_model_record(record)

    def list_models(self, limit: int = 50) -> list[dict[str, Any]]:
        """List registered models, newest first."""
        return self._db.list_models(limit=limit)


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------

class InferenceService:
    """Load registered models and run forward inference.

    Models are cached in memory keyed by their registry id so repeated
    :meth:`predict` calls do not re-read the checkpoint from disk.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self._cache: dict[int, nn.Module] = {}

    # -- loading ----------------------------------------------------------
    def load_model(self, model_id: int) -> nn.Module:
        """Load (or return the cached) model registered under ``model_id``."""
        if model_id in self._cache:
            return self._cache[model_id]

        record = self.registry.get_model(model_id)
        if record is None:
            raise ModelNotFoundError(f"Model {model_id} not found in registry")

        metadata = ModelMetadata.from_model_record(record)
        path = record["path"]
        model = self._load_by_family(metadata.family, path)
        model.eval()
        self._cache[model_id] = model
        return model

    @staticmethod
    def _load_by_family(family: str, path: str) -> nn.Module:
        from tensorlbm.ai.model import load_model as _load_eddy_mlp
        from tensorlbm.ai.transformer import (
            load_flow_transformer_model as _load_transformer,
        )

        if family == FAMILY_FLOW_TRANSFORMER:
            model = _load_transformer(path)
        elif family in (FAMILY_EDDY_MLP, ""):
            model = _load_eddy_mlp(path)
        else:
            raise ValueError(f"Unsupported model family: {family!r}")
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Loaded model of family {family!r} is not an nn.Module"
            )
        return model

    # -- inference --------------------------------------------------------
    def predict(
        self,
        model_id: int,
        inputs: np.ndarray | torch.Tensor | Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Run inference and return the prediction as a float32 ``ndarray``."""
        model = self.load_model(model_id)
        tensors = self._as_tensors(inputs)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                output = model(*tensors)
        finally:
            if was_training:
                model.train()
        return self._to_numpy(output)

    def predict_tensor(
        self,
        model_id: int,
        inputs: np.ndarray | torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Like :meth:`predict` but return the raw ``torch.Tensor``."""
        model = self.load_model(model_id)
        tensors = self._as_tensors(inputs)
        with torch.no_grad():
            return model(*tensors)

    # -- cache management -------------------------------------------------
    def unload_model(self, model_id: int) -> None:
        """Drop a model from the in-memory cache."""
        self._cache.pop(model_id, None)

    def clear_cache(self) -> None:
        self._cache.clear()

    def list_loaded(self) -> list[int]:
        return sorted(self._cache)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _as_tensors(
        inputs: np.ndarray | torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(inputs, torch.Tensor):
            return (inputs,)
        if isinstance(inputs, np.ndarray):
            return (torch.from_numpy(inputs.astype(np.float32)),)
        return tuple(t.detach() if isinstance(t, torch.Tensor) else t for t in inputs)

    @staticmethod
    def _to_numpy(output: torch.Tensor) -> np.ndarray:
        out = output.detach().cpu().numpy()
        if out.dtype != np.float32:
            out = out.astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _as_tensor_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise TypeError(f"Expected a tensor or sequence of tensors, got {type(value).__name__}")


def export_onnx(
    model: nn.Module,
    example_inputs: torch.Tensor | Sequence[torch.Tensor],
    output_path: str | Path,
    *,
    input_names: Sequence[str] = ("input",),
    output_names: Sequence[str] = ("output",),
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    opset_version: int = 14,
    validate: bool = True,
) -> str:
    """Export a PyTorch model to ONNX.

    Args:
        model: The ``nn.Module`` to export (``.eval()`` is applied internally).
        example_inputs: One tensor or a sequence of tensors used to trace the
            graph (and, when omitted, to derive dynamic batch axes).
        output_path: Destination ``.onnx`` file (the suffix is added if absent).
        input_names: Names for the graph inputs.
        output_names: Names for the graph outputs.
        dynamic_axes: Optional explicit dynamic-axis map.  When ``None``, the
            first (batch) axis of every input/output is made dynamic.
        opset_version: ONNX opset version.
        validate: When ``True``, re-load and run ``onnx.checker`` if available.

    Returns:
        The absolute path of the written ``.onnx`` file.

    Raises:
        OnnxUnavailableError: When the optional ``onnx`` package is missing.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected an nn.Module, got {type(model).__name__}")
    if not onnx_available():
        raise OnnxUnavailableError()

    inputs = _as_tensor_tuple(example_inputs)
    if not inputs:
        raise ValueError("example_inputs must not be empty")
    inputs = tuple(t.detach() if t.requires_grad else t for t in inputs)

    model.eval()
    out_path = Path(output_path)
    if out_path.suffix != ".onnx":
        out_path = out_path.with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_names = list(input_names) or [f"input_{i}" for i in range(len(inputs))]
    out_names = list(output_names)

    # Default dynamic axes: make the leading (batch) dimension dynamic.
    if dynamic_axes is None:
        dynamic_axes = {}
        for i, t in enumerate(inputs):
            if t.ndim >= 1:
                dynamic_axes[in_names[i]] = {0: "batch_size"}
        # Output names cannot always be inferred up front; declare a best-effort
        # dynamic batch on the first output only.
        dynamic_axes[out_names[0]] = {0: "batch_size"}

    torch.onnx.export(
        model,
        inputs,
        str(out_path),
        export_params=True,
        opset_version=int(opset_version),
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    if validate and _onnx is not None:
        onnx_model = _onnx.load(str(out_path))
        _onnx.checker.check_model(onnx_model)

    return str(out_path.resolve())
