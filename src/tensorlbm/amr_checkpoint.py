"""Crash-resistant checkpoint/resume for nested AMR runs.

Saves the full hierarchy state (all level populations), step counter, force
history and a configuration signature, and restores them on resume so a long
run can be continued across restarts without losing statistics.

Usage pattern (see examples/amr_sphere_shell_l3_validate.py):

    state = save_amr_checkpoint(
        amr, step=current_step, force_samples=force_samples,
        configuration=config_signature, path=checkpoint_path,
    )
    # on resume:
    amr, start_step, force_samples = resume_amr_checkpoint(
        amr, configuration=config_signature, path=checkpoint_path,
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .checkpoint_io import atomic_torch_save

CHECKPOINT_SCHEMA = "tensorlbm-nested-amr-checkpoint-v1"


def save_amr_checkpoint(
    amr: Any,
    *,
    step: int,
    force_samples: Sequence[float],
    configuration: dict[str, Any],
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically persist the full nested-AMR state.

    Args:
        amr: NestedStaticBlockAMR3D (or StaticBlockAMR3D) instance.
        step: Root step count reached (the checkpoint is written AFTER this
            step completed).
        force_samples: Control-volume force history (list of floats).
        configuration: Immutable run-identity signature (geometry, Re, tau
            chain, collision, grid). Must be JSON-serializable.
        path: Destination .ckpt path.
        extra: Optional extra tensors/dicts to persist alongside.
    """
    destination = Path(path)
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "configuration": configuration,
        "step": int(step),
        "level_populations": [
            level.detach().cpu() for level in amr.level_populations
        ],
        "force_samples": [float(value) for value in force_samples],
    }
    if extra:
        payload.update(extra)
    return atomic_torch_save(payload, destination)


def resume_amr_checkpoint(
    amr: Any,
    *,
    configuration: dict[str, Any],
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[int, list[float]]:
    """Restore hierarchy populations and history onto ``amr``.

    Validates the stored configuration signature against the live run's;
    a mismatch raises ValueError (fail closed, like the rest of the platform).

    Returns ``(start_step, force_samples)``: continue the loop from
    ``start_step + 1`` and keep appending to ``force_samples``.
    """
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    state = torch.load(
        checkpoint, map_location=map_location or amr.level_populations[0].device,
        weights_only=True,
    )
    if state.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"checkpoint schema mismatch: {state.get('schema')!r}",
        )
    stored = state.get("configuration")
    if stored != configuration:
        raise ValueError(
            "checkpoint configuration does not match the current run: "
            "refusing to resume into a different problem.",
        )
    populations = state["level_populations"]
    if not isinstance(populations, (list, tuple)):
        raise ValueError("checkpoint level_populations must be a sequence")
    amr.restore_level_populations(populations)
    force_samples = [float(value) for value in state.get("force_samples", [])]
    return int(state["step"]), force_samples


def build_checkpoint_signature(
    *,
    shape: Sequence[int],
    radius: float,
    reynolds: float,
    lattice_speed: float,
    steps: int,
    warmup_steps: int,
    ramp_steps: int,
    shell_margin: int,
    wake_cells: int,
    l2_margin: int,
    ghost_interpolation: str,
    collision: str | None,
    lattice: str,
    les_model: str | None = None,
    cs_smag: float = 0.05,
    cw_wale: float = 0.5,
    tau_chain: Sequence[float],
    ratio: int,
    ghost: int,
) -> dict[str, Any]:
    """Immutable run-identity signature used to gate resume safety."""
    return {
        "schema_version": 1,
        "shape_zyx": [int(v) for v in shape],
        "radius": float(radius),
        "reynolds": float(reynolds),
        "lattice_speed": float(lattice_speed),
        "steps": int(steps),
        "warmup_steps": int(warmup_steps),
        "ramp_steps": int(ramp_steps),
        "shell_margin": int(shell_margin),
        "wake_cells": int(wake_cells),
        "l2_margin": int(l2_margin),
        "ghost_interpolation": ghost_interpolation,
        "collision": collision,
        "lattice": lattice,
        "les_model": les_model,
        "cs_smag": float(cs_smag),
        "cw_wale": float(cw_wale),
        "tau_chain": [float(v) for v in tau_chain],
        "ratio": int(ratio),
        "ghost": int(ghost),
    }


def checkpoint_signature_json(signature: dict[str, Any]) -> str:
    """Deterministic string form for equality checks / provenance."""
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CHECKPOINT_SCHEMA",
    "save_amr_checkpoint",
    "resume_amr_checkpoint",
    "build_checkpoint_signature",
    "checkpoint_signature_json",
]
