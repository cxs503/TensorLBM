"""Durable, machine-readable execution status for long-running benchmarks."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


def hardware_profile_snapshot() -> dict[str, Any] | None:
    """Best-effort hardware profile for benchmark records; never raises."""
    try:
        from tensorlbm.hardware import probe as hardware_probe

        return hardware_probe().to_dict()
    except Exception as error:  # pragma: no cover - probe is guarded anyway
        return {"probe_error": f"hardware probe failed: {error}"}


def resolve_benchmark_device(requested: str) -> dict[str, Any]:
    """Resolve and validate a requested torch device before a benchmark starts."""
    device = torch.device(requested)
    metadata: dict[str, Any] = {
        "requested": requested,
        "resolved": str(device),
        "torch_version": torch.__version__,
        "host": platform.node(),
    }
    if device.type == "cpu":
        metadata["available"] = True
    elif device.type == "cuda":
        metadata["available"] = torch.cuda.is_available()
        metadata["device_count"] = torch.cuda.device_count()
    elif device.type == "sdaa":
        sdaa = getattr(torch, "sdaa", None)
        if sdaa is None:
            metadata["available"] = False
            metadata["device_count"] = 0
        else:
            metadata["available"] = sdaa.is_available()
            metadata["device_count"] = sdaa.device_count() if metadata["available"] else 0
    else:
        # Generic torch-plugin accelerator (npu/mlu/musa/...): trust the
        # tensorlbm.hardware probe rather than a hardcoded device list.
        from tensorlbm.hardware import probe as hardware_probe

        info = hardware_probe().backend(device.type)
        if info is not None and info.available:
            metadata["available"] = True
            metadata["device_count"] = info.device_count
        else:
            metadata["available"] = False
            metadata["reason"] = (
                f"unsupported or unavailable benchmark device type: {device.type}; "
                f"probed backends: {hardware_probe().available_backends or ('cpu',)}"
            )
    if not metadata["available"]:
        raise RuntimeError(f"Requested device is unavailable: {metadata}")
    # Availability alone is insufficient: an installation can report a backend
    # but fail only at its first allocation. Verify placement before a long run.
    placement = torch.empty(1, device=device)
    if placement.device != device:
        raise RuntimeError(
            f"Requested device did not honor allocation: requested={device}, "
            f"allocated={placement.device}"
        )
    metadata["allocation_device"] = str(placement.device)
    metadata["device_asserted"] = True
    # Document the full host capability snapshot alongside the resolved
    # device so benchmark records are self-describing across heterogeneous
    # (CUDA/NPU/MLU/SDAA/MUSA/CPU) hosts.
    metadata["hardware_profile"] = hardware_profile_snapshot()
    return metadata


def assert_benchmark_tensor_device(
    tensor: torch.Tensor, expected: torch.device, label: str,
) -> None:
    """Fail fast if primary simulation state is not on its requested device.

    This proves PyTorch placement and catches silent CPU-state construction.
    Backend kernel profiling is still required to prove every operation is
    accelerator-native.
    """
    if tensor.device != expected:
        raise RuntimeError(
            f"Device assertion failed for {label}: expected {expected}, "
            f"got {tensor.device}"
        )


def _json_safe(value: Any) -> Any:
    """Convert non-finite metrics to null so lifecycle reporting cannot crash."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class BenchmarkReporter:
    """Append periodic progress and atomically update benchmark lifecycle status."""

    def __init__(
        self,
        output_dir: str | Path,
        benchmark: str,
        requested_steps: int,
        device_metadata: dict[str, Any],
        *,
        hardware_profile: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.benchmark = benchmark
        self.requested_steps = requested_steps
        self.device_metadata = device_metadata
        self.status_path = self.output_dir / "run_status.json"
        self.progress_path = self.output_dir / "progress.csv"
        self.started_at = time.time()
        # Full host capability snapshot (tensorlbm.hardware.probe); probed
        # lazily once and never allowed to break status writing.
        self.hardware_profile = (
            hardware_profile
            if hardware_profile is not None
            else hardware_profile_snapshot()
        )

    def _status(self, state: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "benchmark": self.benchmark,
            "state": state,
            "requested_steps": self.requested_steps,
            "device": self.device_metadata,
            "hardware": self.hardware_profile,
            "started_unix_seconds": self.started_at,
            "updated_unix_seconds": time.time(),
        }
        payload.update(extra)
        _atomic_json(self.status_path, payload)

    def start(self, resume: bool = False, completed_steps: int = 0) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not resume or not self.progress_path.exists():
            with self.progress_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=["step", "elapsed_seconds", "tip_y", "tip_x"],
                ).writeheader()
                handle.flush()
                os.fsync(handle.fileno())
        self._status("RUNNING", completed_steps=completed_steps, resumed=resume)

    def progress(self, step: int, elapsed_seconds: float, tip_y: float, tip_x: float) -> None:
        with self.progress_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(
                handle,
                fieldnames=["step", "elapsed_seconds", "tip_y", "tip_x"],
            ).writerow({
                "step": step,
                "elapsed_seconds": elapsed_seconds,
                "tip_y": tip_y,
                "tip_x": tip_x,
            })
            handle.flush()
            os.fsync(handle.fileno())
        self._status("RUNNING", completed_steps=step, elapsed_seconds=elapsed_seconds)

    def finish(
        self,
        completed_steps: int,
        outcome: str,
        numerical_failure: str | None,
        metrics: dict[str, Any],
    ) -> None:
        self._status(
            outcome,
            completed_steps=completed_steps,
            elapsed_seconds=time.time() - self.started_at,
            numerical_failure=numerical_failure,
            metrics=metrics,
        )


class FsiCheckpoint:
    """Atomic torch-state checkpoint for resumable FSI benchmark segments.

    State is written on CPU so a checkpoint produced on SDAA can be inspected
    or resumed on another available device. Callers validate schema and move
    tensors to their selected execution device.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        os.close(descriptor)
        try:
            torch.save(state, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return torch.load(self.path, map_location="cpu", weights_only=False)
