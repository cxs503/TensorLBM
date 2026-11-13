"""Job-status + wall-clock checkpoint state machine for long unattended runs.

This module is a cross-validation (XVAL) adaptation of the
``job-status-checkpoint-machine`` pattern used by **lanl/MF-LBM**, whose behaviour
is inventoried with source locations in the platform OSS ledger
(``oss_solutions`` row ``repo='lanl/MF-LBM'``,
``feature_slug='NEW:wall-clock-checkpoint-state-machine'``):

* ``IO_multiphase.F90:579-641`` — writes/reads ``job_status.txt`` and the
  checkpoint files (primary + second backup).
* ``Main_multiphase.F90:91-95, 139-141`` — status is consulted at start-up and
  the run takes the *continue* path instead of recomputing.
* ``Main_multiphase.F90:222-303`` — checkpointing is triggered by a **wall-clock
  timer expressed in hours** (``checkpoint_save_timer`` /
  ``simulation_duration_timer``), plus ``d_vol_*`` triggers that fire output and
  monitoring by *injected pore volume* so a long case can be executed in
  unattended segments.
* ``Init_multiphase.F90:154-184`` — restart state (full PDFs and, when the
  convective outflow BC is on, the extra ``f_convec_bc``/``g_convec_bc``/
  ``phi_convec_bc`` history fields) is restored before the main loop starts.

The MF-LBM vocabulary of job states is
``new / continue / done / failed / reached_max_step``; ``double_bak_checkpoint_pdf_cmd``
selects whether the checkpoint is written twice (main + second backup).

The mapping implemented here is deliberately generic (no multiphase physics, no
domain words — this repo owns LBM numerics only):

======================  =============================================
MF-LBM                  here
======================  =============================================
``job_status.txt``      :class:`JobStatusFile` (plain-text, atomically replaced)
wall-clock hours        :class:`CheckpointPolicy.interval_hours`
primary + 2nd backup    :class:`CheckpointStore` (``dual_backup=True``)
full PDF + extra fields ``fields`` mapping handed to :meth:`JobCheckpointManager.maybe_checkpoint`
continue path           :meth:`JobCheckpointManager.begin` -> :class:`ResumeState`
``d_vol_*``             :class:`VolumeTrigger`
======================  =============================================

Example
-------
>>> from tensorlbm.job_checkpoint import JobCheckpointManager
>>> mgr = JobCheckpointManager("runs/case1", interval_hours=0.5, max_steps=10_000)
>>> state = mgr.begin()
>>> step = state.step
>>> # ... advance the solver, then:
>>> receipt = mgr.maybe_checkpoint(step, {"f": f})   # None unless the wall-clock timer fired
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .checkpoint_io import atomic_torch_save

__all__ = [
    "CHECKPOINT_SCHEMA",
    "STATUS_NEW",
    "STATUS_CONTINUE",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_REACHED_MAX_STEP",
    "JOB_STATUSES",
    "TERMINAL_STATUSES",
    "CheckpointError",
    "CheckpointPolicy",
    "CheckpointReceipt",
    "CheckpointStore",
    "JobCheckpointManager",
    "JobStatusFile",
    "JobStatusRecord",
    "ResumeState",
    "VolumeTrigger",
    "fields_digest",
    "run_with_checkpointing",
    "LEGAL_TRANSITIONS",
]

CHECKPOINT_SCHEMA = "tensorlbm-job-checkpoint-v1"

STATUS_NEW = "new"
STATUS_CONTINUE = "continue"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_REACHED_MAX_STEP = "reached_max_step"

#: Vocabulary copied from MF-LBM's ``job_status.txt`` state machine.
JOB_STATUSES = (
    STATUS_NEW,
    STATUS_CONTINUE,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_REACHED_MAX_STEP,
)

#: States from which no further compute may happen without an explicit re-arm.
TERMINAL_STATUSES = (STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP)

#: Legal status transitions.  Terminal states are absorbing (a finished case must
#: never be silently continued — MF-LBM exits when ``job_status.txt`` says
#: ``done``/``failed``); re-writing the same terminal status is idempotent.
LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_NEW: frozenset({STATUS_NEW, STATUS_CONTINUE, STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP}),
    STATUS_CONTINUE: frozenset(
        {STATUS_CONTINUE, STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP}
    ),
    STATUS_DONE: frozenset({STATUS_DONE}),
    STATUS_FAILED: frozenset({STATUS_FAILED}),
    STATUS_REACHED_MAX_STEP: frozenset({STATUS_REACHED_MAX_STEP}),
}


class CheckpointError(RuntimeError):
    """Raised when a checkpoint / job-status file cannot be trusted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fields_digest(fields: Mapping[str, Any]) -> str:
    """Order-independent sha256 over the tensor payload (dtype/shape/bytes)."""
    hasher = hashlib.sha256()
    for name in sorted(fields):
        tensor = fields[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint field {name!r} must be a torch.Tensor, got {type(tensor)!r}")
        cpu = tensor.detach().to("cpu").contiguous()
        hasher.update(name.encode())
        hasher.update(str(tuple(cpu.shape)).encode())
        hasher.update(str(cpu.dtype).encode())
        hasher.update(cpu.numpy().tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class CheckpointPolicy:
    """Checkpoint cadence, mirroring MF-LBM's hour-based timers.

    ``interval_hours`` reproduces ``checkpoint_save_timer`` (unit: hours).
    ``max_steps`` reproduces the ``reached_max_step`` exit condition.
    ``volume_interval`` reproduces the ``d_vol_*`` injected-volume trigger: when
    the caller reports a cumulative produced volume, a checkpoint is forced every
    time that volume crosses another multiple of ``volume_interval``.
    """

    interval_hours: float = 1.0
    max_steps: int | None = None
    volume_interval: float | None = None

    def __post_init__(self) -> None:
        if not (self.interval_hours > 0):  # NaN fails this too
            raise ValueError(f"interval_hours must be > 0 (got {self.interval_hours!r})")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0 when given (got {self.max_steps!r})")
        if self.volume_interval is not None and not (self.volume_interval > 0):
            raise ValueError(f"volume_interval must be > 0 when given (got {self.volume_interval!r})")

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval_hours": self.interval_hours,
            "max_steps": self.max_steps,
            "volume_interval": self.volume_interval,
        }


@dataclass(frozen=True)
class JobStatusRecord:
    """One line-addressable job-status snapshot (``job_status.txt``)."""

    status: str
    step: int = 0
    reason: str = ""
    updated_at: str = ""
    checkpoints: int = 0

    def __post_init__(self) -> None:
        if self.status not in JOB_STATUSES:
            raise CheckpointError(
                f"unknown job status {self.status!r}; expected one of {list(JOB_STATUSES)}"
            )
        if self.step < 0:
            raise CheckpointError(f"step must be >= 0 (got {self.step})")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_text(self) -> str:
        return (
            f"{self.status}\n"
            f"step={self.step}\n"
            f"reason={self.reason}\n"
            f"updated_at={self.updated_at}\n"
            f"checkpoints={self.checkpoints}\n"
        )

    @classmethod
    def from_text(cls, text: str) -> "JobStatusRecord":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise CheckpointError("empty job status file")
        status = lines[0]
        values: dict[str, str] = {}
        for line in lines[1:]:
            if "=" not in line:
                raise CheckpointError(f"malformed job status line: {line!r}")
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
        try:
            step = int(values.get("step", "0"))
            checkpoints = int(values.get("checkpoints", "0"))
        except ValueError as exc:  # pragma: no cover - defensive
            raise CheckpointError(f"non-integer job status field: {values!r}") from exc
        return cls(
            status=status,
            step=step,
            reason=values.get("reason", ""),
            updated_at=values.get("updated_at", ""),
            checkpoints=checkpoints,
        )


class JobStatusFile:
    """Crash-safe ``job_status.txt``: first token is the state, then ``k=v`` lines."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> JobStatusRecord | None:
        if not self.path.exists():
            return None
        text = self.path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return JobStatusRecord.from_text(text)

    def write(self, record: JobStatusRecord, *, previous: JobStatusRecord | None = None) -> None:
        if previous is not None and record.status not in LEGAL_TRANSITIONS[previous.status]:
            raise CheckpointError(
                f"illegal job status transition {previous.status!r} -> {record.status!r} "
                f"(legal: {sorted(LEGAL_TRANSITIONS[previous.status])})"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            tmp.write_text(record.to_text(), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            if tmp.exists():  # pragma: no cover - only if replace failed
                tmp.unlink()


@dataclass(frozen=True)
class CheckpointReceipt:
    step: int
    trigger: str
    elapsed_hours: float
    primary_bytes: int
    backup_bytes: int
    primary_path: str
    backup_path: str | None
    digest: str


class CheckpointStore:
    """Primary + second-backup checkpoint files (MF-LBM ``double_bak_checkpoint_pdf_cmd``).

    Both files are written through :func:`tensorlbm.checkpoint_io.atomic_torch_save`
    so a kill during the write leaves the previous generation intact; the backup
    is a byte copy of the freshly written primary, which is the cheapest faithful
    equivalent of MF-LBM's double write.
    """

    prim_name = "checkpoint.pt"
    backup_name = "checkpoint_bak.pt"

    def __init__(self, directory: str | Path, *, dual_backup: bool = True) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.primary_path = self.directory / self.prim_name
        self.backup_path = self.directory / self.backup_name if dual_backup else None
        self.dual_backup = dual_backup

    # -- write ---------------------------------------------------------------
    def save(self, payload: Mapping[str, Any]) -> tuple[Path, Path | None]:
        atomic_torch_save(dict(payload), self.primary_path)
        backup: Path | None = None
        if self.backup_path is not None:
            tmp = self.backup_path.with_name(f".{self.backup_path.name}.tmp-{os.getpid()}")
            try:
                shutil.copyfile(self.primary_path, tmp)
                tmp.replace(self.backup_path)
            finally:
                if tmp.exists():  # pragma: no cover - only if replace failed
                    tmp.unlink()
            backup = self.backup_path
        return self.primary_path, backup

    # -- read ----------------------------------------------------------------
    def _load_one(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise CheckpointError(f"checkpoint file missing: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # truncated / non-torch garbage
            raise CheckpointError(f"cannot deserialize checkpoint {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CheckpointError(f"checkpoint {path} is not a mapping (got {type(payload)!r})")
        schema = payload.get("schema")
        if schema != CHECKPOINT_SCHEMA:
            raise CheckpointError(
                f"checkpoint {path} schema {schema!r} != expected {CHECKPOINT_SCHEMA!r}"
            )
        fields = payload.get("fields")
        if not isinstance(fields, Mapping):
            raise CheckpointError(f"checkpoint {path} has no 'fields' mapping")
        digest = payload.get("digest")
        recomputed = fields_digest(fields)
        if digest != recomputed:
            raise CheckpointError(
                f"checkpoint {path} digest mismatch (stored {digest!r} != recomputed {recomputed!r})"
            )
        return dict(payload)

    def load(self) -> tuple[dict[str, Any], str]:
        """Load the newest valid generation; fall back to the second backup.

        Returns ``(payload, source)`` with ``source`` in ``{"primary", "backup"}``.
        Raises :class:`CheckpointError` only when *no* generation is valid.
        """
        errors: list[str] = []
        try:
            return self._load_one(self.primary_path), "primary"
        except CheckpointError as exc:
            errors.append(str(exc))
        if self.backup_path is not None:
            try:
                return self._load_one(self.backup_path), "backup"
            except CheckpointError as exc:
                errors.append(str(exc))
        else:
            errors.append("dual_backup=False: no second backup to try")
        raise CheckpointError("no usable checkpoint generation:\n  " + "\n  ".join(errors))


class VolumeTrigger:
    """``d_vol_*`` analogue: fire once per crossed multiple of a volume quantum."""

    def __init__(self, interval: float, *, initial: float = 0.0) -> None:
        if not (interval > 0):
            raise ValueError(f"volume interval must be > 0 (got {interval!r})")
        self.interval = float(interval)
        self._next = (float(initial) // self.interval + 1.0) * self.interval

    def due(self, cumulative_volume: float) -> bool:
        return cumulative_volume >= self._next

    def mark(self, cumulative_volume: float) -> None:
        while cumulative_volume >= self._next:
            self._next += self.interval

    @property
    def next_threshold(self) -> float:
        return self._next


@dataclass
class ResumeState:
    """Outcome of :meth:`JobCheckpointManager.begin`."""

    step: int
    fields: dict[str, torch.Tensor] = field(default_factory=dict)
    resumed: bool = False
    source: str = "fresh"  # "fresh" | "primary" | "backup"
    finished: bool = False
    status: JobStatusRecord | None = None
    checkpoints: int = 0


class JobCheckpointManager:
    """Drives the MF-LBM job-status/checkpoint state machine around a solver loop.

    Parameters
    ----------
    directory:
        Case directory; holds ``job_status.txt`` and the checkpoint generations.
    interval_hours:
        Wall-clock checkpoint period in hours (MF-LBM unit, no rescaling).
    max_steps:
        When the step counter reaches this, :meth:`should_stop` is true and the
        terminal status becomes ``reached_max_step``.
    dual_backup:
        Write ``checkpoint_bak.pt`` alongside ``checkpoint.pt``.
    config_signature:
        Hashable description of the case (grid, tau, BCs, ...).  Restarting with a
        different signature is refused instead of silently mixing states.
    clock:
        Monotonic clock in seconds; injectable for tests / deterministic runs.
    """

    status_filename = "job_status.txt"

    def __init__(
        self,
        directory: str | Path,
        *,
        interval_hours: float = 1.0,
        max_steps: int | None = None,
        volume_interval: float | None = None,
        dual_backup: bool = True,
        config_signature: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.policy = CheckpointPolicy(
            interval_hours=interval_hours, max_steps=max_steps, volume_interval=volume_interval
        )
        self.store = CheckpointStore(self.directory, dual_backup=dual_backup)
        self.status_file = JobStatusFile(self.directory / self.status_filename)
        self.config_signature = self._signature(config_signature)
        self._clock = clock
        self._t0 = self._clock()
        self._last_saved_step = -1
        self._volume_trigger = (
            VolumeTrigger(volume_interval) if volume_interval is not None else None
        )
        self._status: JobStatusRecord | None = None
        self.receipts: list[CheckpointReceipt] = []

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _signature(signature: Mapping[str, Any] | None) -> str | None:
        if signature is None:
            return None
        return hashlib.sha256(
            json.dumps(dict(signature), sort_keys=True, default=str).encode()
        ).hexdigest()

    @property
    def status(self) -> JobStatusRecord | None:
        return self._status

    def _write_status(self, status: str, step: int, reason: str, *, checkpoints: int | None = None) -> JobStatusRecord:
        if checkpoints is None:
            checkpoints = self._status.checkpoints if self._status is not None else 0
        record = JobStatusRecord(
            status=status,
            step=int(step),
            reason=reason,
            updated_at=_utc_now(),
            checkpoints=int(checkpoints),
        )
        self.status_file.write(record, previous=self._status)
        self._status = record
        return record

    # -- lifecycle -----------------------------------------------------------
    def begin(self, *, force_restart: bool = False) -> ResumeState:
        """Consult ``job_status.txt`` and decide fresh vs continue (MF-LBM path)."""
        self._t0 = self._clock()
        existing = None if force_restart else self.status_file.read()
        if force_restart:
            self._status = None
            existing = None

        if existing is None:
            record = self._write_status(STATUS_NEW, 0, "created", checkpoints=0)
            record = self._write_status(STATUS_CONTINUE, 0, "fresh start", checkpoints=0)
            self._last_saved_step = -1
            return ResumeState(step=0, resumed=False, source="fresh", status=record)

        self._status = existing
        if existing.terminal:
            return ResumeState(
                step=existing.step,
                resumed=False,
                source="terminal",
                finished=True,
                status=existing,
                checkpoints=existing.checkpoints,
            )

        payload, source = self.store.load()
        stored_step = int(payload.get("step", -1))
        if stored_step < 0:
            raise CheckpointError(f"checkpoint in {self.directory} has no valid 'step'")
        if self.config_signature is not None:
            stored_sig = payload.get("config_signature")
            if stored_sig != self.config_signature:
                raise CheckpointError(
                    "config signature mismatch: checkpoint was written for a different case "
                    f"({stored_sig!r} != {self.config_signature!r})"
                )
        if self.policy.max_steps is not None and stored_step > self.policy.max_steps:
            raise CheckpointError(
                f"checkpoint step {stored_step} exceeds configured max_steps "
                f"{self.policy.max_steps}"
            )
        fields = {name: tensor for name, tensor in payload["fields"].items()}
        record = self._write_status(
            STATUS_CONTINUE, stored_step, f"resumed from {source}", checkpoints=existing.checkpoints
        )
        self._last_saved_step = stored_step
        return ResumeState(
            step=stored_step,
            fields=fields,
            resumed=True,
            source=source,
            status=record,
            checkpoints=existing.checkpoints,
        )

    def should_stop(self, step: int) -> bool:
        return self.policy.max_steps is not None and step >= self.policy.max_steps

    def elapsed_hours(self) -> float:
        return (self._clock() - self._t0) / 3600.0

    def _save(self, step: int, fields: Mapping[str, torch.Tensor], *, trigger: str) -> CheckpointReceipt:
        if not fields:
            raise CheckpointError("checkpoint fields must not be empty")
        digest = fields_digest(fields)
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "step": int(step),
            "fields": {name: tensor.detach().to("cpu").clone() for name, tensor in fields.items()},
            "digest": digest,
            "config_signature": self.config_signature,
            "policy": self.policy.as_dict(),
            "written_at": _utc_now(),
            "elapsed_hours": self.elapsed_hours(),
            "trigger": trigger,
        }
        primary, backup = self.store.save(payload)
        self._last_saved_step = int(step)
        checkpoints = (self._status.checkpoints if self._status else 0) + 1
        self._write_status(STATUS_CONTINUE, step, f"checkpoint {trigger}", checkpoints=checkpoints)
        receipt = CheckpointReceipt(
            step=int(step),
            trigger=trigger,
            elapsed_hours=payload["elapsed_hours"],
            primary_bytes=primary.stat().st_size,
            backup_bytes=backup.stat().st_size if backup else 0,
            primary_path=str(primary),
            backup_path=str(backup) if backup else None,
            digest=digest,
        )
        self.receipts.append(receipt)
        return receipt

    def checkpoint_now(
        self, step: int, fields: Mapping[str, torch.Tensor], *, trigger: str = "manual"
    ) -> CheckpointReceipt:
        """Write a checkpoint immediately, ignoring the wall-clock timer.

        This is the explicit "save before you exit" entry point (MF-LBM writes a
        checkpoint before terminating a segment); the automatic path is
        :meth:`maybe_checkpoint`.
        """
        receipt = self._save(int(step), fields, trigger=trigger)
        self._t0 = self._clock()
        return receipt

    def maybe_checkpoint(
        self,
        step: int,
        fields: Mapping[str, torch.Tensor],
        *,
        cumulative_volume: float | None = None,
    ) -> CheckpointReceipt | None:
        """Write a checkpoint only when the wall-clock (or volume) trigger fires."""
        step = int(step)
        if step <= self._last_saved_step:
            return None
        elapsed = self.elapsed_hours()
        trigger: str | None = None
        if elapsed >= self.policy.interval_hours:
            trigger = "wall_clock"
        elif (
            self._volume_trigger is not None
            and cumulative_volume is not None
            and self._volume_trigger.due(cumulative_volume)
        ):
            trigger = "volume"
        if trigger is None:
            return None
        receipt = self._save(step, fields, trigger=trigger)
        self._t0 = self._clock()  # restart the wall-clock timer
        if self._volume_trigger is not None and cumulative_volume is not None:
            self._volume_trigger.mark(cumulative_volume)
        return receipt

    def finish(
        self,
        step: int,
        *,
        status: str = STATUS_DONE,
        reason: str = "",
        fields: Mapping[str, torch.Tensor] | None = None,
    ) -> CheckpointReceipt | None:
        """Write the terminal status (and a final checkpoint when fields are given)."""
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"finish() needs a terminal status, got {status!r}")
        receipt: CheckpointReceipt | None = None
        if fields:
            receipt = self._save(int(step), fields, trigger="final")
        checkpoints = self._status.checkpoints if self._status is not None else 0
        self._write_status(status, int(step), reason or f"finished: {status}", checkpoints=checkpoints)
        return receipt

    def fail(self, step: int, reason: str) -> JobStatusRecord:
        return self._write_status(STATUS_FAILED, int(step), reason)


def run_with_checkpointing(
    sim: Any,
    steps: int,
    manager: JobCheckpointManager,
) -> tuple[int, int]:
    """Advance ``sim`` by at most ``steps`` new steps under an *already begun* manager.

    ``sim`` needs ``step()``, a ``f`` tensor and an integer ``step_count``.  When
    the solver itself already drives the manager (``sim.checkpoints is manager``,
    as :class:`tensorlbm.simulation.LBMSimulation` does) the loop only advances
    and counts; otherwise it checkpoints after every step.  Returns
    ``(steps_executed, checkpoints_written)``.

    Resume happens in :meth:`JobCheckpointManager.begin` *before* this loop, so a
    resumed run never recomputes a step (MF-LBM ``continue`` path).
    """
    internal = getattr(sim, "checkpoints", None) is manager
    before = manager.status.checkpoints if manager.status is not None else 0
    executed = 0
    for _ in range(int(steps)):
        if manager.should_stop(sim.step_count):
            break
        sim.step()
        executed += 1
        if not internal:
            manager.maybe_checkpoint(sim.step_count, {"f": sim.f})
        if getattr(sim, "finished", False):
            break
    after = manager.status.checkpoints if manager.status is not None else 0
    return executed, after - before
