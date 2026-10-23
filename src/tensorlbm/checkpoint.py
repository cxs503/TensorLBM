"""Checkpoint utilities for long-running TensorLBM simulations.

Supports saving and loading simulation state (distribution function tensor,
current step, and arbitrary metadata) so that interrupted runs can be
resumed without starting over.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

from .checkpoint_io import atomic_torch_save

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
    (run_dir / _META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
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
    if not isinstance(loaded_meta, dict) or not all(isinstance(key, str) for key in loaded_meta):
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


__all__ = [
    "CASE_CHECKPOINT_FILENAME",
    "CheckpointError",
    "SOLVER_CHECKPOINT_FORMAT",
    "SOLVER_CHECKPOINT_VERSION",
    "SolverCheckpoint",
    "case_checkpoint_path",
    "eager_load_state_dict",
    "eager_state_dict",
    "load_case_checkpoint",
    "load_checkpoint",
    "load_solver_checkpoint",
    "save_case_checkpoint",
    "save_checkpoint",
    "save_solver_checkpoint",
    "triton_fused_load_state_dict",
    "triton_fused_state_dict",
]


# ===========================================================================
# Unified solver-state checkpoints (format v2)
# ===========================================================================
#
# The v1 API above (``save_checkpoint`` / ``load_checkpoint``) persists a
# bare distribution tensor plus a JSON sidecar for benchmark harnesses.
# The v2 API below is the *unified solver-state* convention: one
# atomically-written ``.pt`` file carrying everything needed to continue
# a run bit-exactly — populations, step counter, lattice/grid identity,
# solver parameters, optional solid mask, precision policy and RNG
# state — with a SHA-256 digest so torn or bit-rotted files fail closed
# on load.
#
# Adapters (``eager_state_dict`` / ``triton_fused_state_dict`` and their
# load counterparts) are functional: the solver classes themselves are
# never modified, and any duck-typed object exposing the usual attributes
# (``tau``, ``nz/ny/nx``, optional ``lattice``/``Q``/``mask``) works.

#: Envelope tag written into every unified solver checkpoint.
SOLVER_CHECKPOINT_FORMAT = "tensorlbm-solver-checkpoint"

#: Current unified solver-state format version.  v2 adds the envelope,
#: integrity digests, RNG state and the adapter schema on top of the v1
#: tensor+JSON pair (which keeps its own ``format_version = 1``).
SOLVER_CHECKPOINT_VERSION = 2

#: File name used by the scan_runner campaign hook inside a point directory.
CASE_CHECKPOINT_FILENAME = "case-checkpoint.pt"


class CheckpointError(ValueError):
    """Raised when a unified checkpoint is unusable (fail closed).

    Covers truncated/torn writes, payload digest mismatches, unsupported
    format versions and identity mismatches (lattice / grid / Q / tau /
    precision policy).
    """


@dataclass
class SolverCheckpoint:
    """Validated contents of a unified solver checkpoint.

    Attributes:
        state: the adapter payload (``populations`` tensor plus solver
            parameters); feed it straight to the matching
            ``*_load_state_dict`` adapter.
        step: completed simulation step saved in the file.
        lattice: lattice name (``"D3Q19"`` …) or ``None`` when unknown.
        grid: spatial grid ``(nz, ny, nx)`` (or ``(ny, nx)`` in 2D).
        q: number of lattice directions (``None`` when unknown).
        metadata: free-form caller metadata saved alongside the state.
        format_version: envelope version (currently 2).
        path: file the checkpoint was loaded from.
    """

    state: dict[str, Any]
    step: int
    lattice: str | None
    grid: tuple[int, ...] | None
    q: int | None
    metadata: dict[str, Any]
    format_version: int
    path: Path

    @property
    def populations(self) -> torch.Tensor:
        """The saved populations tensor (CPU copy; move with ``.to()``)."""
        return self.state["populations"]


def _require_state_dict(state: object) -> dict[str, Any]:
    """Validate an adapter-style state dict; raise :class:`CheckpointError`."""
    if not isinstance(state, dict):
        raise CheckpointError(f"state must be a dict, got {type(state).__name__}")
    missing = [key for key in ("populations", "step") if key not in state]
    if missing:
        raise CheckpointError(f"state dict is missing required key(s): {', '.join(missing)}")
    populations = state["populations"]
    if not isinstance(populations, torch.Tensor):
        raise CheckpointError(
            f"state['populations'] must be a torch.Tensor, got {type(populations).__name__}"
        )
    if populations.ndim < 1:
        raise CheckpointError("state['populations'] needs at least the lattice dimension")
    step = state["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise CheckpointError(f"state['step'] must be a non-negative integer, got {step!r}")
    lattice = state.get("lattice")
    if lattice is not None and not isinstance(lattice, str):
        raise CheckpointError(f"state['lattice'] must be a string or None, got {lattice!r}")
    grid = state.get("grid")
    if grid is not None and (
        not isinstance(grid, (tuple, list))
        or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in grid)
    ):
        raise CheckpointError(f"state['grid'] must be a tuple of positive ints, got {grid!r}")
    q = state.get("q")
    if q is not None and (not isinstance(q, int) or isinstance(q, bool) or q <= 0):
        raise CheckpointError(f"state['q'] must be a positive integer or None, got {q!r}")
    return state


def _tensor_digest(tensor: torch.Tensor) -> str:
    """SHA-256 over a tensor's raw bytes (CPU copy, contiguous)."""
    data = tensor.detach().to(device="cpu", copy=True).contiguous()
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()


def _capture_rng(reference: torch.Tensor) -> dict[str, Any]:
    """Snapshot the torch RNG state(s) relevant to *reference*'s device."""
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if reference.device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state(reference.device)
        rng["cuda_device"] = str(reference.device)
    return rng


def _restore_rng(rng: object) -> None:
    if not isinstance(rng, dict) or not isinstance(rng.get("cpu"), torch.Tensor):
        return
    torch.set_rng_state(rng["cpu"])
    cuda_state = rng.get("cuda")
    if not isinstance(cuda_state, torch.Tensor):
        return
    device = rng.get("cuda_device")
    index = torch.device(device).index if isinstance(device, str) else None
    if not torch.cuda.is_available() or (index is not None and index >= torch.cuda.device_count()):
        raise CheckpointError(
            f"checkpoint RNG state requires CUDA device {device or 'current'}, which is "
            "not available here; pass restore_rng=False to skip RNG restoration "
            "(continuation is then no longer bit-deterministic)."
        )
    torch.cuda.set_rng_state(cuda_state, device)


def save_solver_checkpoint(
    path: str | Path,
    state: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a unified solver-state checkpoint (format v2).

    Writes ``<name>.tmp-<pid>`` then renames, so a crash mid-save can
    never leave a half-written checkpoint behind.  Tensors are stored as
    CPU copies, making checkpoints portable across devices.

    Args:
        path: destination file (convention: ``*.pt`` / ``*.ckpt``).
        state: adapter-produced state dict.  Required keys:
            ``populations`` (tensor, leading lattice dimension) and
            ``step`` (completed step count).  Recognised optional keys:
            ``lattice``, ``grid``, ``q``, ``tau``, ``dtype``,
            ``obstacle_mask``, ``precision_policy`` plus any
            adapter-specific extras.
        metadata: free-form metadata stored alongside the state.

    Returns:
        The final checkpoint path.
    """
    checked = _require_state_dict(dict(state))
    payload = dict(checked)
    digests: dict[str, str] = {}
    for key in ("populations", "obstacle_mask"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().to(device="cpu", copy=True)
            digests[key] = _tensor_digest(payload[key])
    grid = payload.get("grid")
    envelope: dict[str, Any] = {
        "format": SOLVER_CHECKPOINT_FORMAT,
        "format_version": SOLVER_CHECKPOINT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": int(payload["step"]),
        "lattice": payload.get("lattice"),
        "grid": tuple(grid) if grid is not None else None,
        "q": payload.get("q"),
        "state": payload,
        "metadata": dict(metadata or {}),
        "rng": _capture_rng(checked["populations"]),
        "digests": digests,
    }
    return atomic_torch_save(envelope, path)


def load_solver_checkpoint(
    path: str | Path,
    *,
    expected_lattice: str | None = None,
    expected_grid: tuple[int, ...] | None = None,
    expected_q: int | None = None,
    restore_rng: bool = True,
) -> SolverCheckpoint:
    """Load and validate a unified solver checkpoint (fail closed).

    Args:
        path: checkpoint file written by :func:`save_solver_checkpoint`.
        expected_lattice: reject the checkpoint if the lattice differs.
        expected_grid: reject on grid mismatch, e.g. ``(nz, ny, nx)``.
        expected_q: reject on lattice-direction-count mismatch.
        restore_rng: restore the saved torch RNG state (default).  Pass
            ``False`` when loading on a host lacking the CUDA device the
            checkpoint was written on.

    Returns:
        A :class:`SolverCheckpoint` with the validated contents.

    Raises:
        CheckpointError: missing file, torn/corrupt payload, digest
            mismatch, unsupported format version, or an identity mismatch
            against the ``expected_*`` arguments.
    """
    target = Path(path)
    if not target.is_file():
        raise CheckpointError(f"checkpoint file not found: {target}")
    try:
        envelope = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is fail-closed
        raise CheckpointError(
            f"checkpoint {target} could not be decoded (truncated or corrupt "
            f"write?): {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(envelope, dict) or envelope.get("format") != SOLVER_CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"{target} is not a unified solver checkpoint "
            f"(missing format tag {SOLVER_CHECKPOINT_FORMAT!r})"
        )
    version = envelope.get("format_version")
    if version != SOLVER_CHECKPOINT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format_version={version!r} in {target}; "
            f"this TensorLBM supports {SOLVER_CHECKPOINT_VERSION}"
        )
    state = _require_state_dict(envelope.get("state"))
    digests = envelope.get("digests")
    if isinstance(digests, dict):
        for key, expected_digest in digests.items():
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or not isinstance(expected_digest, str):
                raise CheckpointError(
                    f"checkpoint {target} has an invalid integrity record for {key!r}"
                )
            if _tensor_digest(value) != expected_digest:
                raise CheckpointError(
                    f"checkpoint {target} failed the integrity digest for {key!r} "
                    "(payload differs from what was saved)"
                )
    lattice = state.get("lattice")
    grid = state.get("grid")
    grid_tuple = tuple(grid) if grid is not None else None
    q = state.get("q")
    if expected_lattice is not None and lattice != expected_lattice:
        raise CheckpointError(
            f"checkpoint lattice mismatch: expected {expected_lattice!r}, got {lattice!r}"
        )
    if expected_grid is not None and grid_tuple != tuple(expected_grid):
        raise CheckpointError(
            f"checkpoint grid mismatch: expected {tuple(expected_grid)}, got {grid_tuple}"
        )
    if expected_q is not None and q != expected_q:
        raise CheckpointError(f"checkpoint Q mismatch: expected {expected_q}, got {q}")
    if restore_rng:
        _restore_rng(envelope.get("rng"))
    return SolverCheckpoint(
        state=state,
        step=int(state["step"]),
        lattice=lattice if isinstance(lattice, str) else None,
        grid=grid_tuple,
        q=q if isinstance(q, int) else None,
        metadata=dict(envelope.get("metadata") or {}),
        format_version=int(version),
        path=target,
    )


# ---------------------------------------------------------------------------
# Solver adapters (functional; solver classes are not modified)
# ---------------------------------------------------------------------------


def _known_grid(solver: Any) -> tuple[int, ...] | None:
    """Grid ``(nz, ny, nx)`` when the solver exposes it, else ``None``."""
    dims: list[int] = []
    for name in ("nz", "ny", "nx"):
        value = getattr(solver, name, None)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        dims.append(int(value))
    return tuple(dims)


def _known_q(solver: Any) -> int | None:
    for name in ("Q", "_Q"):
        value = getattr(solver, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    return None


def _solver_solid_mask(solver: Any) -> torch.Tensor | None:
    for name in ("mask", "_mask", "solid_mask", "_solid_mask"):
        value = getattr(solver, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _base_state_dict(
    solver: Any,
    f: torch.Tensor,
    step: int | None,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(f, torch.Tensor):
        raise TypeError(f"populations must be a torch.Tensor, got {type(f).__name__}")
    tau = getattr(solver, "tau", None)
    policy = getattr(solver, "precision_policy", None)
    grid = _known_grid(solver)
    q = _known_q(solver)
    state: dict[str, Any] = {
        "populations": f.detach(),
        "step": int(step) if step is not None else int(getattr(solver, "_report_step", 0) or 0),
        "lattice": getattr(solver, "lattice", None),
        "grid": grid if grid is not None else tuple(int(n) for n in f.shape[1:]),
        "q": q if q is not None else int(f.shape[0]),
        "tau": float(tau) if tau is not None else None,
        "dtype": str(f.dtype),
        "obstacle_mask": _solver_solid_mask(solver),
        "precision_policy": getattr(policy, "name", None),
    }
    if extra:
        state["extra"] = dict(extra)
    return state


def _restore_populations(
    solver: Any,
    state: Mapping[str, Any],
    device: torch.device | str | None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Validate a state dict against *solver* and return its populations."""
    checked = _require_state_dict(dict(state))
    f = checked["populations"]
    lattice = checked.get("lattice")
    solver_lattice = getattr(solver, "lattice", None)
    if (
        isinstance(lattice, str)
        and isinstance(solver_lattice, str)
        and solver_lattice.upper() != lattice.upper()
    ):
        raise CheckpointError(
            f"state lattice {lattice!r} does not match solver lattice {solver_lattice!r}"
        )
    saved_q = checked.get("q")
    solver_q = _known_q(solver)
    if solver_q is not None and int(saved_q or f.shape[0]) != solver_q:
        raise CheckpointError(f"state Q {saved_q} does not match solver Q {solver_q}")
    saved_grid = checked.get("grid")
    solver_grid = _known_grid(solver)
    if solver_grid is not None and tuple(saved_grid or ()) != solver_grid:
        raise CheckpointError(
            f"state grid {tuple(saved_grid or ())} does not match solver grid {solver_grid}"
        )
    saved_tau = checked.get("tau")
    solver_tau = getattr(solver, "tau", None)
    if saved_tau is not None and solver_tau is not None and float(saved_tau) != float(solver_tau):
        raise CheckpointError(f"state tau {saved_tau} does not match solver tau {solver_tau}")
    target = device if device is not None else getattr(solver, "device", None)
    if target is not None:
        f = f.to(device=target)
    if dtype is not None:
        f = f.to(dtype=dtype)
    return f


def _restore_report_step(solver: Any, state: Mapping[str, Any]) -> None:
    if isinstance(getattr(solver, "_report_step", None), int):
        solver._report_step = int(state["step"])  # type: ignore[attr-defined]


def eager_state_dict(
    solver: Any,
    f: torch.Tensor,
    *,
    step: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a checkpointable state dict from an eager solver.

    Works with :class:`~tensorlbm.lbm_step.LBMStepExecutor`,
    :class:`~tensorlbm.perf_solver.OptimizedSolver3D` and any duck-typed
    object exposing ``tau`` plus grid attributes; the solver instance is
    only read, never modified.

    Args:
        solver: the eager solver instance (source of identity/parameters).
        f: current populations tensor ``(Q, nz, ny, nx)`` — caller-owned
            in the functional eager API.
        step: completed step count.  Defaults to the solver's reporter
            counter (``_report_step``) when it tracks one; that counter
            only advances through ``run()``/reporter dispatch, so pass
            ``step`` explicitly when driving ``step()`` by hand.
        extra: adapter-specific extras merged into the state dict.

    Returns:
        A state dict accepted by :func:`save_solver_checkpoint` and
        :func:`eager_load_state_dict`.
    """
    return _base_state_dict(solver, f, step, extra)


def eager_load_state_dict(
    solver: Any,
    state: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Restore populations from a state dict onto an eager solver.

    Validates lattice, Q, grid and tau against *solver* (fail closed with
    :class:`CheckpointError`) and re-syncs the solver's reporter step
    counter when it tracks one.  Returns the restored populations tensor
    on the solver's device.
    """
    f = _restore_populations(solver, state, device)
    _restore_report_step(solver, state)
    return f


def triton_fused_state_dict(
    solver: Any,
    f: torch.Tensor,
    *,
    step: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a checkpointable state dict from a TritonFusedSolver3D.

    Adds the storage dtype and precision-policy tier to the common eager
    fields: the fused kernel *is* the precision cast boundary, so both
    must be restored exactly to continue a trajectory.
    """
    state = _base_state_dict(solver, f, step, extra)
    if state["lattice"] is None:
        state["lattice"] = "D3Q19"
    state["storage_dtype"] = str(getattr(solver, "dtype", f.dtype))
    return state


def triton_fused_load_state_dict(
    solver: Any,
    state: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Restore populations onto a TritonFusedSolver3D (fail closed).

    Besides lattice/Q/grid/tau, validates the storage dtype and the
    precision-policy tier — an ``FP32FP16`` checkpoint loaded into an
    fp32 solver (or vice versa) would silently change the trajectory, so
    it raises :class:`CheckpointError` instead.
    """
    saved_policy = state.get("precision_policy")
    current = getattr(solver, "precision_policy", None)
    current_policy = getattr(current, "name", None)
    if saved_policy != current_policy:
        raise CheckpointError(
            f"state precision policy {saved_policy!r} does not match solver "
            f"policy {current_policy!r}"
        )
    saved_dtype = state.get("storage_dtype") or state.get("dtype")
    solver_dtype = getattr(solver, "dtype", None)
    if solver_dtype is not None and str(saved_dtype) != str(solver_dtype):
        raise CheckpointError(
            f"state storage dtype {saved_dtype} does not match solver dtype {solver_dtype}"
        )
    f = _restore_populations(solver, state, device)
    _restore_report_step(solver, state)
    return f


# ---------------------------------------------------------------------------
# Campaign hook (scan_runner per-point checkpoints)
# ---------------------------------------------------------------------------


def case_checkpoint_path(directory: str | Path) -> Path:
    """Per-point checkpoint path used by the scan_runner campaign hook."""
    return Path(directory) / CASE_CHECKPOINT_FILENAME


def save_case_checkpoint(
    directory: str | Path,
    *,
    f: torch.Tensor,
    step: int,
    lattice: str | None,
    grid: tuple[int, ...] | None,
    identity: Mapping[str, Any],
    state_extra: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a resumable per-case checkpoint inside *directory*.

    Thin campaign wrapper over :func:`save_solver_checkpoint`: the
    caller-supplied *identity* mapping (e.g. scan id, point id, plan
    digest) is stored in the metadata and re-checked on resume, so a
    checkpoint from a different sweep/plan/point never leaks into a run.
    """
    state: dict[str, Any] = {
        "populations": f,
        "step": int(step),
        "lattice": lattice,
        "grid": tuple(grid) if grid is not None else None,
        "q": int(f.shape[0]) if f.ndim >= 1 else None,
    }
    if state_extra:
        state.update(state_extra)
    meta: dict[str, Any] = {"identity": dict(identity)}
    if metadata:
        meta.update(metadata)
    return save_solver_checkpoint(case_checkpoint_path(directory), state, metadata=meta)


def load_case_checkpoint(
    directory: str | Path,
    *,
    identity: Mapping[str, Any],
) -> SolverCheckpoint | None:
    """Best-effort resume read for the campaign hook.

    Returns ``None`` — meaning "restart the point from step 0" — when the
    file is absent, corrupt, or was written by a different run (identity
    mismatch).  RNG state is deliberately not restored: campaign
    trajectories continue from the populations alone.
    """
    path = case_checkpoint_path(directory)
    if not path.is_file():
        return None
    try:
        checkpoint = load_solver_checkpoint(path, restore_rng=False)
    except CheckpointError:
        return None
    saved_identity = checkpoint.metadata.get("identity")
    if not isinstance(saved_identity, dict):
        return None
    if any(saved_identity.get(key) != value for key, value in identity.items()):
        return None
    return checkpoint
