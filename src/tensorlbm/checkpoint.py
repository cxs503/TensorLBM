"""Checkpoint utilities for long-running TensorLBM simulations.

Supports saving and loading simulation state (distribution function tensor,
current step, and arbitrary metadata) so that interrupted runs can be
resumed without starting over.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch

_TENSOR_FILE = "checkpoint_f.pt"
_META_FILE = "checkpoint_meta.json"
_FORMAT_VERSION = 1


def save_checkpoint(
    f: torch.Tensor,
    step: int,
    run_dir: Path,
    extra: dict[str, object] | None = None,
) -> Path:
    """Save a checkpoint of the distribution function and step counter.

    Args:
        f: Distribution tensor (any shape).
        step: Current simulation step.
        run_dir: Directory in which to write the checkpoint files.
        extra: Optional extra metadata dict to store alongside step.

    Returns:
        Path to the checkpoint directory (same as *run_dir*).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(f.cpu(), run_dir / _TENSOR_FILE)

    meta: dict[str, object] = {
        "format_version": _FORMAT_VERSION,
        "step": step,
        "tensor_shape": list(f.shape),
        "tensor_dtype": str(f.dtype),
        "lattice_directions": int(f.shape[0]) if f.ndim >= 1 else None,
    }
    if extra:
        meta.update(extra)
    (run_dir / _META_FILE).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def load_checkpoint(
    run_dir: Path,
    device: torch.device | None = None,
    *,
    expected_shape: tuple[int, ...] | None = None,
    expected_lattice_directions: int | None = None,
) -> tuple[torch.Tensor, int, dict[str, object]]:
    """Load a previously saved checkpoint.

    Args:
        run_dir: Directory containing the checkpoint files written by
            :func:`save_checkpoint`.
        device: Target device for the distribution tensor. Defaults to CPU.

    Returns:
        Tuple ``(f, step, meta)`` where *f* is the distribution tensor,
        *step* is the saved simulation step, and *meta* is the full metadata
        dict (including ``"step"``).

    Raises:
        FileNotFoundError: If the checkpoint files do not exist.
    """
    run_dir = Path(run_dir)
    tensor_path = run_dir / _TENSOR_FILE
    meta_path = run_dir / _META_FILE

    if not tensor_path.exists():
        raise FileNotFoundError(f"Checkpoint tensor not found: {tensor_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")

    f = torch.load(
        tensor_path,
        map_location=device or torch.device("cpu"),
        weights_only=True,
    )
    try:
        loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Checkpoint metadata is not valid JSON: {meta_path}") from exc
    if not isinstance(loaded_meta, dict) or not all(
        isinstance(key, str) for key in loaded_meta
    ):
        raise ValueError(f"Checkpoint metadata must be a JSON object with string keys: {meta_path}")
    meta = cast("dict[str, object]", loaded_meta)
    format_version = meta.get("format_version", 0)
    if not isinstance(format_version, int):
        raise ValueError(f"Checkpoint metadata 'format_version' must be an integer: {meta_path}")
    if format_version > _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version={format_version}; "
            f"maximum supported is {_FORMAT_VERSION}: {meta_path}"
        )
    if "step" not in meta:
        raise ValueError(f"Checkpoint metadata missing 'step' key: {meta_path}")
    step_value = meta["step"]
    if not isinstance(step_value, int) or isinstance(step_value, bool):
        raise ValueError(f"Checkpoint metadata 'step' must be an integer: {meta_path}")
    step = step_value
    if not isinstance(f, torch.Tensor):
        raise ValueError(f"Checkpoint tensor payload is not a torch.Tensor: {tensor_path}")
    expected_shape_meta = meta.get("tensor_shape")
    if isinstance(expected_shape_meta, list) and tuple(expected_shape_meta) != tuple(f.shape):
        raise ValueError(
            "Checkpoint tensor shape does not match metadata: "
            f"tensor={tuple(f.shape)} metadata={tuple(expected_shape_meta)}"
        )
    if expected_shape is not None and tuple(f.shape) != tuple(expected_shape):
        raise ValueError(
            "Checkpoint incompatible with current run shape: "
            f"expected={tuple(expected_shape)} got={tuple(f.shape)}"
        )
    if expected_lattice_directions is not None and f.ndim >= 1:
        actual = int(f.shape[0])
        if actual != expected_lattice_directions:
            raise ValueError(
                "Checkpoint incompatible with current lattice model: "
                f"expected {expected_lattice_directions} directions, got {actual}"
            )
    return f, step, meta


__all__ = ["save_checkpoint", "load_checkpoint"]
