from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DiagnosticPoint:
    step: int
    mass: float
    mass_drift: float
    max_speed: float
    mean_rho: float


def resolve_device(device_name: str) -> torch.device:
    """Resolve a device name string to a :class:`torch.device`.

    Args:
        device_name: ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        The corresponding :class:`torch.device`.

    Raises:
        RuntimeError: If CUDA or MPS is requested but not available.
        ValueError: If the device name is not recognised.
    """
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA requested but not available"
            raise RuntimeError(msg)
        return torch.device("cuda")
    if device_name == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            msg = "MPS requested but not available"
            raise RuntimeError(msg)
        return torch.device("mps")
    msg = f"Unsupported device: {device_name}"
    raise ValueError(msg)


def prepare_run_dir(output_root: Path, subdir: str, run_name: str, overwrite: bool) -> Path:
    """Create and return the run output directory.

    Args:
        output_root: Root directory for all outputs.
        subdir: Sub-directory name (e.g. ``"cylinder_flow"``).
        run_name: Unique name for this run.
        overwrite: Remove an existing directory of the same name when *True*.

    Returns:
        The newly-created run directory path.
    """
    run_dir = output_root / subdir / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


__all__ = ["DiagnosticPoint", "resolve_device", "prepare_run_dir"]
