"""Crash-resistant checkpoint writes shared by long-running CFD benchmarks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Write a Torch checkpoint completely before atomically replacing it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = ["atomic_torch_save"]
