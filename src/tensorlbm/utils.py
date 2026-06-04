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


def get_reproducibility_metadata() -> dict[str, object]:
    """Collect metadata for scientific reproducibility.

    Returns a dict with git commit hash, Python version, and key package
    versions. All fields degrade gracefully if unavailable.
    """
    import subprocess
    import sys

    meta: dict[str, object] = {
        "python_version": sys.version,
    }
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            meta["git_commit"] = result.stdout.strip()
        else:
            meta["git_commit"] = None
    except Exception:
        meta["git_commit"] = None

    pkg_versions: dict[str, str] = {}
    for pkg in ("torch", "matplotlib", "numpy"):
        try:
            import importlib.metadata as im

            pkg_versions[pkg] = im.version(pkg)
        except Exception:
            pkg_versions[pkg] = "unknown"
    meta["package_versions"] = pkg_versions
    return meta


def flow_step_image_path(run_dir: Path, step: int) -> Path:
    """Return canonical flow snapshot image path for a simulation step."""
    return run_dir / f"flow_step_{step:06d}.png"


def legacy_snapshot_image_path(run_dir: Path, step: int) -> Path:
    """Return legacy snapshot image path for backward compatibility."""
    return run_dir / f"snapshot_{step:06d}.png"


def write_legacy_snapshot_alias(run_dir: Path, step: int) -> Path:
    """Create legacy ``snapshot_*`` alias from canonical ``flow_step_*`` image.

    If the canonical file does not exist or alias already exists, this is a
    no-op. The alias keeps existing tools/scripts compatible during migration.
    """
    canonical = flow_step_image_path(run_dir, step)
    legacy = legacy_snapshot_image_path(run_dir, step)
    if canonical.exists() and not legacy.exists():
        shutil.copy2(canonical, legacy)
    return legacy


__all__ = [
    "DiagnosticPoint",
    "resolve_device",
    "prepare_run_dir",
    "get_reproducibility_metadata",
    "flow_step_image_path",
    "legacy_snapshot_image_path",
    "write_legacy_snapshot_alias",
]
