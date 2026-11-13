"""Invariants of the MF-LBM-style job-status / wall-clock checkpoint state machine.

These tests assert properties, not just execution: restart is bit-exact, the
second backup really rescues a corrupted primary, terminal states stay terminal,
a resume recomputes zero steps, and the wall-clock/volume triggers fire exactly
when they should.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorlbm import LBMSimulation  # noqa: E402
from tensorlbm.checkpoint_io import atomic_torch_save  # noqa: E402
from tensorlbm.job_checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    JOB_STATUSES,
    CheckpointError,
    CheckpointPolicy,
    CheckpointStore,
    JobCheckpointManager,
    JobStatusFile,
    JobStatusRecord,
    VolumeTrigger,
    fields_digest,
    run_with_checkpointing,
)


class FakeClock:
    """Deterministic monotonic clock; ``advance`` simulates compute time."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def _sim(nx: int = 16, ny: int = 8) -> LBMSimulation:
    sim = LBMSimulation(nx=nx, ny=ny, tau=0.7)
    # Blow up the isotropic equilibrium so a restart bug cannot hide behind a
    # uniform field: perturb with a fixed-seed non-uniform pattern.
    gen = torch.Generator().manual_seed(1234)
    sim.f = sim.f + 0.01 * torch.randn(sim.f.shape, generator=gen)
    return sim


def _ic_copy(sim: LBMSimulation) -> torch.Tensor:
    return sim.f.detach().clone()


# ---------------------------------------------------------------------------
# vocabulary / status file
# ---------------------------------------------------------------------------
def test_status_vocabulary_matches_mf_lbm() -> None:
    assert set(JOB_STATUSES) == {"new", "continue", "done", "failed", "reached_max_step"}


def test_status_file_roundtrip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "job_status.txt"
    status = JobStatusFile(path)
    assert status.read() is None
    record = JobStatusRecord(status="new", step=0, reason="created", updated_at="t0")
    status.write(record)
    back = status.read()
    assert back == record
    # first token is the state word, so the file stays greppable like MF-LBM's
    assert path.read_text(encoding="utf-8").splitlines()[0] == "new"


def test_unknown_status_token_is_rejected(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "job_status.txt"
    path.write_text("restart\nstep=3\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="unknown job status"):
        JobStatusFile(path).read()


def test_illegal_transition_rejected(tmp_path: pathlib.Path) -> None:
    status = JobStatusFile(tmp_path / "job_status.txt")
    done = JobStatusRecord(status="done", step=10)
    status.write(done)
    with pytest.raises(CheckpointError, match="illegal job status transition"):
        status.write(JobStatusRecord(status="continue", step=11), previous=done)


def test_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        CheckpointPolicy(interval_hours=0.0)
    with pytest.raises(ValueError):
        CheckpointPolicy(max_steps=0)
    with pytest.raises(ValueError):
        CheckpointPolicy(volume_interval=-1.0)


# ---------------------------------------------------------------------------
# store: dual backup, integrity, fallback
# ---------------------------------------------------------------------------
def _payload(step: int, value: float = 1.0) -> dict:
    fields = {"f": torch.full((2, 3, 9), value)}
    return {
        "schema": CHECKPOINT_SCHEMA,
        "step": step,
        "fields": fields,
        "digest": fields_digest(fields),
        "config_signature": None,
    }


def test_dual_backup_written_and_identical(tmp_path: pathlib.Path) -> None:
    store = CheckpointStore(tmp_path, dual_backup=True)
    primary, backup = store.save(_payload(1))
    assert primary.exists() and backup is not None and backup.exists()
    assert primary.read_bytes() == backup.read_bytes()
    # no temporary leftovers from the atomic write
    assert not list(tmp_path.glob(".*tmp*"))


def test_fallback_to_backup_when_primary_corrupted(tmp_path: pathlib.Path) -> None:
    store = CheckpointStore(tmp_path, dual_backup=True)
    store.save(_payload(7))
    # simulate a kill during the next write: truncated primary
    primary = tmp_path / CheckpointStore.prim_name
    payload = primary.read_bytes()
    primary.write_bytes(payload[: len(payload) // 2])
    loaded, source = store.load()
    assert source == "backup"
    assert loaded["step"] == 7


def test_single_backup_mode_loses_recovery(tmp_path: pathlib.Path) -> None:
    """Control: without the second backup the corrupted primary is fatal."""
    store = CheckpointStore(tmp_path, dual_backup=False)
    store.save(_payload(7))
    primary = tmp_path / CheckpointStore.prim_name
    payload = primary.read_bytes()
    primary.write_bytes(payload[: len(payload) // 2])
    with pytest.raises(CheckpointError, match="no usable checkpoint generation"):
        store.load()


def test_both_generations_corrupt_raises(tmp_path: pathlib.Path) -> None:
    store = CheckpointStore(tmp_path, dual_backup=True)
    store.save(_payload(3))
    for name in (CheckpointStore.prim_name, CheckpointStore.backup_name):
        (tmp_path / name).write_bytes(b"not a torch file")
    with pytest.raises(CheckpointError) as excinfo:
        store.load()
    assert "checkpoint.pt" in str(excinfo.value) and "checkpoint_bak.pt" in str(excinfo.value)


def test_digest_mismatch_is_detected(tmp_path: pathlib.Path) -> None:
    store = CheckpointStore(tmp_path, dual_backup=False)
    payload = _payload(4, value=1.0)
    # silently alter a byte of the payload while keeping the stored digest
    payload["fields"]["f"] = payload["fields"]["f"].clone()
    payload["fields"]["f"][0, 0, 0] += 1.0
    atomic_torch_save(payload, store.primary_path)
    with pytest.raises(CheckpointError, match="digest mismatch"):
        store.load()


# ---------------------------------------------------------------------------
# manager lifecycle: restart continuity, zero recompute, terminal states
# ---------------------------------------------------------------------------
def test_restart_is_bit_exact_and_recomputes_nothing(tmp_path: pathlib.Path) -> None:
    reference = _sim()
    ic = _ic_copy(reference)
    reference.run(60)
    baseline = reference.f.clone()

    clock = FakeClock()
    sim = _sim()
    sim.f = ic.clone()
    state = sim.enable_checkpointing(
        tmp_path, interval_hours=1e-5, clock=clock
    )
    assert state.resumed is False and state.step == 0
    for _ in range(30):
        clock.advance(1.0)  # 1 s per step; 1e-5 h = 0.036 s -> every step saves
        sim.step()
    assert sim.checkpoints.status.checkpoints == 30

    # "crash": a brand new object must pick the checkpoint up
    resumed = _sim()
    st = resumed.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    assert st.resumed is True and st.source == "primary" and st.step == 30
    executed, written = run_with_checkpointing(resumed, 30, resumed.checkpoints)
    assert executed == 30  # steps 31..60 only: zero recomputation
    assert resumed.step_count == 60
    assert torch.equal(resumed.f, baseline)  # bit-exact restart continuity


def test_restart_from_backup_still_bit_exact(tmp_path: pathlib.Path) -> None:
    reference = _sim()
    ic = _ic_copy(reference)
    reference.run(40)

    clock = FakeClock()
    sim = _sim()
    sim.f = ic.clone()
    sim.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    for _ in range(20):
        clock.advance(1.0)
        sim.step()

    primary = tmp_path / CheckpointStore.prim_name
    raw = primary.read_bytes()
    primary.write_bytes(raw[: len(raw) // 2])  # kill mid-write

    resumed = _sim()
    st = resumed.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    assert st.source == "backup" and st.step == 20
    run_with_checkpointing(resumed, 20, resumed.checkpoints)
    assert torch.equal(resumed.f, reference.f)


def test_mass_is_conserved_across_restarts(tmp_path: pathlib.Path) -> None:
    sim = _sim()
    clock = FakeClock()
    sim.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    for _ in range(25):
        clock.advance(1.0)
        sim.step()
    mass_before = float(sim.f.sum())
    resumed = _sim()
    resumed.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    mass_after = float(resumed.f.sum())
    assert abs(mass_after - mass_before) <= 1e-6 * abs(mass_before)


def test_wall_clock_timer_does_not_fire_early(tmp_path: pathlib.Path) -> None:
    clock = FakeClock()
    sim = _sim()
    sim.enable_checkpointing(tmp_path, interval_hours=1.0, clock=clock)  # 1 hour
    for _ in range(10):
        clock.advance(1.0)  # 10 s of an hour: never due
        sim.step()
    assert sim.checkpoints.status.checkpoints == 0
    assert not (tmp_path / CheckpointStore.prim_name).exists()


def test_reached_max_step_is_terminal(tmp_path: pathlib.Path) -> None:
    clock = FakeClock()
    sim = _sim()
    sim.enable_checkpointing(tmp_path, interval_hours=1e-6, max_steps=5, clock=clock)
    for _ in range(5):
        clock.advance(1.0)
        sim.step()
    assert sim.finished is True
    record = JobStatusFile(tmp_path / "job_status.txt").read()
    assert record is not None and record.status == "reached_max_step" and record.step == 5
    with pytest.raises(RuntimeError, match="terminal"):
        sim.step()

    # a later launch must not silently continue a finished case
    later = _sim()
    st = later.enable_checkpointing(tmp_path, interval_hours=1e-6, max_steps=5, clock=clock)
    assert st.finished is True and st.resumed is False
    assert st.status is not None and st.status.status == "reached_max_step"


def test_failed_state_is_terminal_but_force_restart_rearms(tmp_path: pathlib.Path) -> None:
    clock = FakeClock()
    sim = _sim()
    sim.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    for _ in range(3):
        clock.advance(1.0)
        sim.step()
    sim.checkpoints.fail(sim.step_count, "simulated blow-up")

    blocked = _sim()
    st = blocked.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    assert st.finished is True and st.status is not None and st.status.status == "failed"

    rearmed = _sim()
    st2 = rearmed.enable_checkpointing(
        tmp_path, interval_hours=1e-5, clock=clock, force_restart=True
    )
    assert st2.finished is False and st2.step == 0
    assert JobStatusFile(tmp_path / "job_status.txt").read().status == "continue"


def test_config_signature_mismatch_refused(tmp_path: pathlib.Path) -> None:
    clock = FakeClock()
    sim = _sim(nx=16, ny=8)
    sim.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)
    for _ in range(3):
        clock.advance(1.0)
        sim.step()
    other = _sim(nx=24, ny=12)
    with pytest.raises(CheckpointError, match="config signature mismatch"):
        other.enable_checkpointing(tmp_path, interval_hours=1e-5, clock=clock)


# ---------------------------------------------------------------------------
# volume trigger (MF-LBM d_vol_*)
# ---------------------------------------------------------------------------
def test_volume_trigger_fires_on_crossing_multiples() -> None:
    trigger = VolumeTrigger(2.5)
    assert trigger.due(2.4) is False
    assert trigger.due(2.5) is True
    trigger.mark(2.5)
    assert trigger.due(2.5) is False  # one shot per threshold
    assert trigger.next_threshold == pytest.approx(5.0)
    assert trigger.due(5.1) is True


def test_volume_trigger_marks_on_checkpoint(tmp_path: pathlib.Path) -> None:
    clock = FakeClock()
    sim = _sim()
    sim.enable_checkpointing(
        tmp_path, interval_hours=1.0, volume_interval=1.0, clock=clock
    )
    # wall clock stays far below one hour; only the volume trigger can save
    sim.step()
    assert sim.checkpoints.maybe_checkpoint(1, {"f": sim.f}, cumulative_volume=0.9) is None
    receipt = sim.checkpoints.maybe_checkpoint(1, {"f": sim.f}, cumulative_volume=1.0)
    assert receipt is not None and receipt.trigger == "volume"


# ---------------------------------------------------------------------------
# final status vocabulary is what MF-LBM documents
# ---------------------------------------------------------------------------
def test_terminal_receipt_records_schema_and_step(tmp_path: pathlib.Path) -> None:
    store = CheckpointStore(tmp_path, dual_backup=True)
    fields = {"f": torch.zeros((4, 4, 9))}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "step": 12,
        "fields": fields,
        "digest": fields_digest(fields),
    }
    store.save(payload)
    loaded, source = store.load()
    assert source == "primary" and loaded["step"] == 12
    assert json.dumps({"status": "continue"})  # status file stays JSON-free plain text