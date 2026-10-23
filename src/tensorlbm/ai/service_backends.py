"""Backend selection for the drag-surrogate / echo service (TRT slice, 2026-08-27).

:func:`make_backend` is the single construction point when the serving stack
wants to swap the inference runtime of :class:`DragSurrogateService` while
keeping everything downstream (guard, UQ, echo pipeline, HTTP shaping)
untouched:

- ``torch`` — :class:`~tensorlbm.ai.inference_service.ModelEnsembleBackend`
  itself (one forward per member; the default and the reference contract);
- ``onnx`` — :class:`~tensorlbm.ai.onnx_deploy.OnnxEnsembleBackend` behind
  the service adapter (one onnxruntime session of the fused-ensemble graph);
- ``trt`` — :class:`~tensorlbm.ai.trt_deploy.TrtEnsembleBackend` behind the
  same adapter (one deserialised TensorRT engine; plans of PR #249).

Protocol contract (what "a backend" means at the service boundary)
------------------------------------------------------------------
The #241 protocol is exactly :class:`ModelEnsembleBackend`:

- ``predict(fields (5, ny, nx), cond (N, 8)) -> (M, N)`` member matrix of
  **linear C_D** (float64), raw un-normalised inputs — the fused artifacts
  fold each member's normalisation into the graph, so the raw contract is
  the same as the torch backend's;
- ``predict_batch(fields (G, 5, ny, nx), cond (N, 8) concatenated,
  counts (G,)) -> (M, sum counts)`` — row ``n`` evaluated on the field of
  the geometry it belongs to.

``mean`` / ``std`` / ``lo`` / ``hi`` (the UQ fields of ``EchoResult``) are
**never** produced by a backend: the service recomputes them from the member
matrix via :func:`~tensorlbm.ai.inference_service.ensemble_stats` on the
host, identically for every backend.  The TRT graph does carry in-graph
statistics (``cd_mean``/``cd_std``/``cd_min``/``cd_max``, float32), which are
deliberately NOT used on the serving path (float32 accumulation differs from
the float64 host reduction); they stay reachable through the wrapped
runtime's ``predict_stats`` for parity measurements.

Engineering notes kept honest here
----------------------------------
- :class:`FusedGraphBackend` subclasses ``ModelEnsembleBackend`` WITHOUT
  building torch models: the ``isinstance`` dispatch inside
  ``DragSurrogateService.predict`` / ``GeometryEchoPipeline._run_backend``
  then routes it down the model path, so no existing file had to change
  its dispatch logic.
- TensorRT engines expose a bounded dynamic condition batch (the shipped
  plans: profile ``cond`` 1..8).  The adapter chunks cond rows into
  ``<= chunk``-row calls and concatenates along the row axis.  Row results
  can in principle depend on the chunk shape at float32 LSB level (TRT
  picks tactics per input shape); parity is therefore pinned at the served
  batch shapes by ``tests/test_service_backends.py``.
- This module needs ``torch`` (it derives from a torch-backed base class and
  shares the service types).  Running the TRT branch in the deploy venv
  (which has no torch of its own) works by borrowing the tensorlbm venv's
  site-packages on ``PYTHONPATH`` — the documented server recipe.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .inference_service import (
    CondDragCheckpoint,
    ModelEnsembleBackend,
)

__all__ = [
    "BACKEND_KINDS",
    "ENV_BACKEND_FALLBACK",
    "ENV_BACKEND_KIND",
    "ENV_BACKEND_PLAN",
    "ENV_BACKEND_PRECISION",
    "FusedGraphBackend",
    "TorchEnsembleBackend",
    "make_backend",
    "resolve_backend_kind",
]

#: Valid ``kind`` values of :func:`make_backend`.
BACKEND_KINDS = ("torch", "onnx", "trt")

#: Environment variable selecting the serving backend (default ``torch``).
ENV_BACKEND_KIND = "TENSORLBM_DRAG_BACKEND"
#: Environment variable pointing at the runtime artifact (``.plan`` / ``.onnx``).
ENV_BACKEND_PLAN = "TENSORLBM_DRAG_BACKEND_PLAN"
#: Optional label suffix distinguishing engine precisions (e.g. ``fp32_strict``).
ENV_BACKEND_PRECISION = "TENSORLBM_DRAG_BACKEND_PRECISION"
#: When set to ``torch``, an unavailable TRT/ONNX runtime degrades to the torch
#: backend with the reason recorded on the backend instead of raising.
ENV_BACKEND_FALLBACK = "TENSORLBM_DRAG_BACKEND_FALLBACK"


def resolve_backend_kind(explicit: str | None = None) -> str:
    """Serving backend kind: explicit argument > ``TENSORLBM_DRAG_BACKEND`` > torch.

    With neither the argument nor the environment variable set this returns
    ``"torch"`` — the pre-TRT service behaviour — so the default path is
    unchanged (the hard constraint of this slice).
    """
    kind = explicit if explicit is not None else os.environ.get(ENV_BACKEND_KIND)
    if kind is None or not str(kind).strip():
        return "torch"
    kind = str(kind).strip().lower()
    if kind not in BACKEND_KINDS:
        raise ValueError(
            f"unknown drag backend kind {kind!r}; valid kinds are {list(BACKEND_KINDS)} "
            f"({ENV_BACKEND_KIND} or the backend_kind argument)"
        )
    return kind


def _resolve_artifact_path(artifact_path: str | Path | None, kind: str) -> Path:
    """Artifact path from the argument or ``TENSORLBM_DRAG_BACKEND_PLAN``."""
    if artifact_path is None:
        artifact_path = os.environ.get(ENV_BACKEND_PLAN)
    if not artifact_path:
        raise ValueError(
            f"backend kind {kind!r} needs a runtime artifact: pass artifact_path= "
            f"or set {ENV_BACKEND_PLAN} to the .{'plan' if kind == 'trt' else 'onnx'} file"
        )
    return Path(artifact_path)


class TorchEnsembleBackend(ModelEnsembleBackend):
    """``ModelEnsembleBackend`` plus construction provenance.

    Built by :func:`make_backend` (including the fallback path) so callers can
    always read ``backend_kind`` / ``init_report`` regardless of which runtime
    ended up serving.  Prediction behaviour is inherited verbatim.
    """

    backend_kind: str
    init_report: dict[str, Any]

    def __init__(
        self,
        ckpts: Sequence[CondDragCheckpoint],
        *,
        device: str = "cpu",
        kind: str = "torch",
        report: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ckpts, device=device)
        self.backend_kind = str(kind)
        self.init_report = dict(report or {})


class FusedGraphBackend(ModelEnsembleBackend):
    """Service-protocol adapter over a fused-ensemble runtime (ONNX / TRT).

    Wraps one runtime exposing the raw fused contract
    ``predict(field (5, ny, nx) | (1, 5, ny, nx), cond (N, 8)) -> (M, N)``
    (:class:`~tensorlbm.ai.onnx_deploy.OnnxEnsembleBackend`,
    :class:`~tensorlbm.ai.trt_deploy.TrtEnsembleBackend`) and re-exposes it
    as the #241 ``ModelEnsembleBackend`` protocol, including the
    ``(G, 5, ny, nx)`` + counts batch form and the float64 output dtype.

    Subclassing ``ModelEnsembleBackend`` is what makes the existing
    ``isinstance`` dispatch in the service / echo pipeline route this backend
    down the model path; no torch model is materialised by ``__init__``.
    """

    backend_kind: str
    init_report: dict[str, Any]

    def __init__(
        self,
        runtime: Any,
        *,
        kind: str,
        labels: Sequence[str],
        chunk: int | None = None,
        report: dict[str, Any] | None = None,
    ) -> None:
        if len(labels) < 1:
            raise ValueError("fused backend needs at least one member label")
        self._runtime = runtime
        self._kind = str(kind)
        self._labels = [str(x) for x in labels]
        self._chunk = None if chunk is None else max(1, int(chunk))
        self.backend_kind = str(kind)
        self.init_report = dict(report or {})
        self.device = torch.device("cpu")

    @property
    def runtime(self) -> Any:
        """The wrapped raw-contract runtime (for parity probes / predict_stats)."""
        return self._runtime

    @property
    def chunk_rows(self) -> int | None:
        """Max condition rows per runtime call (None = runtime is unbounded)."""
        return self._chunk

    @property
    def n_members(self) -> int:
        return len(self._labels)

    @property
    def kind(self) -> str:
        return self._kind

    def member_labels(self) -> list[str]:
        return list(self._labels)

    def predict(self, fields: np.ndarray, cond: np.ndarray) -> np.ndarray:
        fields = np.asarray(fields, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float64)
        if fields.ndim != 3 or fields.shape[0] != 5:
            raise ValueError(f"fields must be (5, ny, nx), got {fields.shape}")
        if cond.ndim != 2 or cond.shape[1] != 8:
            raise ValueError(f"cond must be (N, 8), got {cond.shape}")
        parts = [
            np.asarray(self._runtime.predict(fields, cond[start:end]), dtype=np.float64)
            for start, end in _row_slices(cond.shape[0], self._chunk)
        ]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def predict_batch(
        self,
        fields: np.ndarray,
        cond: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        fields = np.asarray(fields, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float64)
        counts = np.asarray(counts, dtype=np.int64)
        if fields.ndim != 4 or fields.shape[1] != 5:
            raise ValueError(f"fields must be (G, 5, ny, nx), got {fields.shape}")
        if cond.ndim != 2 or cond.shape[1] != 8:
            raise ValueError(f"cond must be (N, 8), got {cond.shape}")
        if counts.ndim != 1 or counts.size != fields.shape[0] or not (counts > 0).all():
            raise ValueError(
                f"counts must be positive with one entry per field, got {counts!r} "
                f"for {fields.shape[0]} fields"
            )
        if int(counts.sum()) != cond.shape[0]:
            raise ValueError(f"counts sum {int(counts.sum())} != condition rows {cond.shape[0]}")
        parts: list[np.ndarray] = []
        offset = 0
        for g in range(fields.shape[0]):
            n = int(counts[g])
            parts.append(self.predict(fields[g], cond[offset : offset + n]))
            offset += n
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


def _row_slices(n_rows: int, chunk: int | None) -> list[tuple[int, int]]:
    """Half-open row slices of ``range(n_rows)`` in ``chunk``-sized pieces."""
    if chunk is None or chunk <= 0 or n_rows <= chunk:
        return [(0, n_rows)]
    return [(s, min(s + chunk, n_rows)) for s in range(0, n_rows, chunk)]


def _engine_cond_profile_max(runtime: Any) -> int | None:
    """Max condition batch of the engine's first optimisation profile.

    Reads the deserialised engine of ``TrtEnsembleBackend`` (same-package
    introspection of ``_engine``; the profile is a property of the plan, not
    of the constructor argument).  Returns ``None`` when the runtime does not
    expose an engine (fake runtimes in tests).
    """
    engine = getattr(runtime, "_engine", None)
    if engine is None:
        return None
    try:
        shape = engine.get_tensor_profile_shape("cond", 0)[2]
        return max(1, int(shape[0]))
    except Exception:  # noqa: BLE001 — profile query is best-effort introspection
        return None


def _build_onnx_runtime(artifact_path: Path) -> Any:
    from .onnx_deploy import OnnxEnsembleBackend

    return OnnxEnsembleBackend(artifact_path)


def _build_trt_runtime(artifact_path: Path, *, max_batch: int) -> Any:
    from .trt_deploy import TrtEnsembleBackend

    return TrtEnsembleBackend(artifact_path, max_batch=max_batch)


def _torch_fallback(
    ckpts: Sequence[CondDragCheckpoint] | None,
    device: str,
    kind: str,
    reason: str,
) -> TorchEnsembleBackend:
    if not ckpts:
        raise RuntimeError(
            f"backend kind {kind!r} is unavailable ({reason}) and no member checkpoints "
            f"were supplied to fall back on (ckpts=)"
        )
    return TorchEnsembleBackend(
        ckpts,
        device=device,
        kind="torch",
        report={"fallback_from": kind, "fallback_reason": reason},
    )


def make_backend(
    kind: str,
    *,
    ckpts: Sequence[CondDragCheckpoint] | None = None,
    device: str = "cpu",
    artifact_path: str | Path | None = None,
    precision: str | None = None,
    member_labels: Sequence[str] | None = None,
    trt_max_batch: int = 8,
    fallback: str | None = None,
) -> ModelEnsembleBackend:
    """Construct a serving backend for :class:`DragSurrogateService`.

    Parameters
    ----------
    kind:
        ``"torch"`` (default semantics), ``"onnx"`` or ``"trt"``.
    ckpts:
        Member checkpoints — required for ``torch`` and for the fallback
        path; also the source of member labels for torch.  For ``onnx`` /
        ``trt`` the labels default to the artifact's embedded provenance
        (ONNX metadata) or ``m0..m{M-1}`` (engine member axis).
    device:
        Torch device (torch kind / fallback only; the TRT runtime manages
        its own CUDA context).
    artifact_path:
        ``.plan`` (trt) / ``.onnx`` (onnx) runtime artifact; falls back to
        ``TENSORLBM_DRAG_BACKEND_PLAN``.
    precision:
        Honest label suffix only (the artifact decides the actual math):
        ``kind`` becomes ``f"{kind}_{precision}"``, e.g. ``trt_fp32_strict``.
    member_labels:
        Override the member labels reported by the backend.
    trt_max_batch:
        Cond-row cap per engine call AND device-buffer size hint.  The
        effective chunk is ``min(trt_max_batch, engine profile max)`` — the
        shipped PR #249 plans have profile max 8.
    fallback:
        ``None`` (raise when the runtime is unavailable) or ``"torch"``
        (degrade to :class:`TorchEnsembleBackend` with the reason recorded
        in ``backend.init_report["fallback_reason"]``).

    Returns
    -------
    ModelEnsembleBackend
        ``TorchEnsembleBackend`` (torch kind or fallback) or
        :class:`FusedGraphBackend` (onnx / trt).  Both carry
        ``backend_kind`` + ``init_report``; both satisfy the #241 protocol.
    """
    kind = str(kind).strip().lower()
    if kind not in BACKEND_KINDS:
        raise ValueError(
            f"unknown drag backend kind {kind!r}; valid kinds are {list(BACKEND_KINDS)}"
        )
    if precision is None:
        precision = os.environ.get(ENV_BACKEND_PRECISION)
    kind_label = (
        kind
        if precision is None or not str(precision).strip()
        else (f"{kind}_{str(precision).strip()}")
    )

    if kind == "torch":
        if not ckpts:
            raise ValueError("backend kind 'torch' needs member checkpoints (ckpts=)")
        return TorchEnsembleBackend(ckpts, device=device, kind=kind_label)

    artifact = _resolve_artifact_path(artifact_path, kind)
    try:
        if kind == "onnx":
            runtime = _build_onnx_runtime(artifact)
            labels = list(runtime.member_labels())
            chunk: int | None = None
        else:
            runtime = _build_trt_runtime(artifact, max_batch=trt_max_batch)
            profile_max = _engine_cond_profile_max(runtime)
            chunk = (
                max(1, int(trt_max_batch))
                if profile_max is None
                else (max(1, min(int(trt_max_batch), profile_max)))
            )
            labels = [f"m{i}" for i in range(runtime.n_members)]
    except Exception as exc:  # noqa: BLE001 — the reason string is the deliverable
        reason = f"{type(exc).__name__}: {exc}"
        if str(fallback).strip().lower() == "torch":
            return _torch_fallback(ckpts, device, kind, reason)
        raise RuntimeError(f"drag backend {kind_label!r} unavailable: {reason}") from exc
    if member_labels is not None:
        labels = [str(x) for x in member_labels]
    if len(labels) != runtime.n_members:
        raise ValueError(f"{len(labels)} member labels for a {runtime.n_members}-member artifact")
    return FusedGraphBackend(
        runtime,
        kind=kind_label,
        labels=labels,
        chunk=chunk,
        report={
            "artifact": str(artifact.resolve()),
            "chunk_rows": chunk,
            "n_members": runtime.n_members,
            "runtime_type": type(runtime).__name__,
        },
    )


def backend_env_help() -> str:
    """One-line documentation of the backend selection env vars (health logs)."""
    return (
        f"{ENV_BACKEND_KIND}=torch|onnx|trt (default torch), "
        f"{ENV_BACKEND_PLAN}=<.plan|.onnx>, {ENV_BACKEND_PRECISION}=<label>, "
        f"{ENV_BACKEND_FALLBACK}=torch"
    )
