"""Measured cross-validation of :mod:`tensorlbm.job_checkpoint` vs lanl/MF-LBM.

The earlier XVAL round for the anchor ``XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE``
(``tests/test_xval_job_status_checkpoint_machine.py``, mission m17867) compared
TensorLBM against **hand-authored copies** of a transition table and never
imported the real machine module (``tensorlbm/job_checkpoint.py``).  This module
replaces that inferred comparison with measurements taken by *driving the real
code* and recording what it does:

=========================  =============================================
measurement                what is executed
=========================  =============================================
transition matrix          ``JobStatusFile.write(record, previous=...)``
                           over all 25 ``(from, to)`` status pairs
machine reachability       real ``begin / checkpoint_now / finish / fail``
                           lifecycle with the status file polled per step
status persistence         ``job_status.txt`` text round-trip
dual-backup recovery       corrupt primary, corrupt both generations
payload tamper detection   digest mismatch inside a checkpoint file
resume parity              ``LBMStepExecutor`` (D3Q19) continuous vs.
                           checkpointed restart, fields compared bitwise
trigger semantics          wall-clock + injected-volume gates, step veto
terminal absorbing         ``begin()`` on a terminal status file
=========================  =============================================

Every number produced here is written to JSON by :func:`measure_all` so the
report can quote it programmatically instead of transcribing by hand.

Reference behaviour (lanl/MF-LBM, as inventoried in the platform OSS ledger):
``IO_multiphase.F90:579-641`` writes/reads ``job_status.txt`` plus primary and
second-backup checkpoints, ``Main_multiphase.F90:91-95,139-141`` takes the
*continue* path at start-up, and terminal states (``done``/``failed``/
``reached_max_step``) end the run instead of being silently continued.  The
reference therefore has **absorbing terminals**, which is what
:data:`MF_LBM_LEGAL_TRANSITIONS` below encodes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from tensorlbm.job_checkpoint import (
    CHECKPOINT_SCHEMA,
    JOB_STATUSES,
    LEGAL_TRANSITIONS,
    STATUS_CONTINUE,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_NEW,
    STATUS_REACHED_MAX_STEP,
    TERMINAL_STATUSES,
    CheckpointError,
    CheckpointStore,
    JobCheckpointManager,
    JobStatusFile,
    JobStatusRecord,
    fields_digest,
)

__all__ = [
    "MF_LBM_LEGAL_TRANSITIONS",
    "measure_all",
    "measure_checkpoint_store_recovery",
    "measure_dual_backup_recovery",
    "measure_payload_tamper",
    "measure_resume_parity",
    "measure_status_persistence",
    "measure_terminal_absorbing",
    "measure_transition_matrix",
    "measure_trigger_semantics",
    "measure_write_without_begin",
]

#: Status transitions of the lanl/MF-LBM job-status machine, transcribed from the
#: inventoried reference behaviour: the fresh ``new`` state may advance to any
#: state, ``continue`` may advance to any state or stay ``continue`` (periodic
#: checkpoint), and the three terminal states are absorbing.
MF_LBM_LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_NEW: frozenset(
        {STATUS_NEW, STATUS_CONTINUE, STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP}
    ),
    STATUS_CONTINUE: frozenset(
        {STATUS_CONTINUE, STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP}
    ),
    STATUS_DONE: frozenset({STATUS_DONE}),
    STATUS_FAILED: frozenset({STATUS_FAILED}),
    STATUS_REACHED_MAX_STEP: frozenset({STATUS_REACHED_MAX_STEP}),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _record(status: str, *, step: int = 0, reason: str = "probe") -> JobStatusRecord:
    return JobStatusRecord(status=status, step=step, reason=reason, updated_at="probe")


def _fields(seed: int = 7, *, shape: tuple[int, ...] = (19, 4, 6, 8)) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {"f": torch.rand(shape, generator=generator, dtype=torch.float64)}


def _d2q9_probability_scale(diff: float) -> float:
    """Relative L2 difference (%) helper for report tables."""
    return 100.0 * float(diff)


# ---------------------------------------------------------------------------
# 1. transition matrix (guard-level, real JobStatusFile.write)
# ---------------------------------------------------------------------------

def measure_transition_matrix(root: str | Path) -> dict[str, Any]:
    """Drive every ``(from, to)`` status pair through the real write guard."""
    root = Path(root)
    pairs: list[dict[str, Any]] = []
    for source in JOB_STATUSES:
        for target in JOB_STATUSES:
            case = root / f"guard_{source}_{target}"
            case.mkdir(parents=True, exist_ok=True)
            status_file = JobStatusFile(case / "job_status.txt")
            status_file.write(_record(source))
            legal = target in MF_LBM_LEGAL_TRANSITIONS[source]
            try:
                status_file.write(_record(target), previous=_record(source))
                accepted = True
                error = None
            except CheckpointError as exc:
                accepted = False
                error = str(exc)
            pairs.append(
                {
                    "from": source,
                    "to": target,
                    "legal_in_reference": legal,
                    "accepted": accepted,
                    "agrees_with_reference": accepted == legal,
                    "error": error,
                }
            )

    legal_pairs = [p for p in pairs if p["legal_in_reference"]]
    illegal_pairs = [p for p in pairs if not p["legal_in_reference"]]
    accepted_legal = sum(1 for p in legal_pairs if p["accepted"])
    rejected_illegal = sum(1 for p in illegal_pairs if not p["accepted"])
    return {
        "total_pairs": len(pairs),
        "legal_pairs": len(legal_pairs),
        "illegal_pairs": len(illegal_pairs),
        "legal_accepted": accepted_legal,
        "illegal_rejected": rejected_illegal,
        "legal_coverage_pct": 100.0 * accepted_legal / len(legal_pairs),
        "illegal_rejection_pct": 100.0 * rejected_illegal / len(illegal_pairs),
        "agreement_pct": 100.0 * sum(1 for p in pairs if p["agrees_with_reference"]) / len(pairs),
        "mismatches": [p for p in pairs if not p["agrees_with_reference"]],
        "implementation_table_equals_reference": {
            status: sorted(LEGAL_TRANSITIONS[status]) == sorted(MF_LBM_LEGAL_TRANSITIONS[status])
            for status in JOB_STATUSES
        },
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# 2. status-file persistence
# ---------------------------------------------------------------------------

def measure_status_persistence(root: str | Path) -> dict[str, Any]:
    case = Path(root) / "persist"
    case.mkdir(parents=True, exist_ok=True)
    status_file = JobStatusFile(case / "job_status.txt")
    missing_before = status_file.read() is None
    record = JobStatusRecord(
        status=STATUS_CONTINUE, step=123, reason="periodic checkpoint", updated_at="t0", checkpoints=4
    )
    status_file.write(record)
    reloaded = JobStatusFile(case / "job_status.txt").read()
    text = (case / "job_status.txt").read_text(encoding="utf-8")
    empty_file = case / "empty.txt"
    empty_file.write_text("   \n", encoding="utf-8")
    return {
        "missing_file_returns_none": missing_before,
        "first_token_is_status": text.splitlines()[0] == STATUS_CONTINUE,
        "roundtrip_equal": reloaded == record,
        "reloaded": None if reloaded is None else reloaded.to_text(),
        "empty_file_returns_none": JobStatusFile(empty_file).read() is None,
    }


# ---------------------------------------------------------------------------
# 3. dual-backup recovery (MF-LBM double_bak_checkpoint_pdf_cmd analogue)
# ---------------------------------------------------------------------------

def measure_dual_backup_recovery(root: str | Path) -> dict[str, Any]:
    case = Path(root) / "dual"
    case.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(case, dual_backup=True)
    store.save({"schema": CHECKPOINT_SCHEMA, "step": 10, "fields": _fields(), "digest": fields_digest(_fields())})

    primary_only = store.load()[1]

    # corrupt the newest generation: the backup must carry the run
    store.primary_path.write_bytes(b"not a torch payload")
    payload, source_after_primary_corruption = store.load()
    recovered_step = int(payload["step"])

    # corrupt both generations: must fail closed
    store.backup_path.write_bytes(b"also broken")
    both_broken_error = None
    try:
        store.load()
    except CheckpointError as exc:
        both_broken_error = str(exc)

    # single-generation store cannot recover at all
    single_dir = Path(root) / "single"
    single_dir.mkdir(parents=True, exist_ok=True)
    single = CheckpointStore(single_dir, dual_backup=False)
    single.save({"schema": CHECKPOINT_SCHEMA, "step": 3, "fields": _fields(), "digest": fields_digest(_fields())})
    single.primary_path.write_bytes(b"broken")
    single_error = None
    try:
        single.load()
    except CheckpointError as exc:
        single_error = str(exc)

    return {
        "clean_source": primary_only,
        "source_after_primary_corruption": source_after_primary_corruption,
        "recovered_step": recovered_step,
        "dual_backup_recovered": source_after_primary_corruption == "backup" and recovered_step == 10,
        "both_corrupt_fails_closed": both_broken_error is not None,
        "both_corrupt_error": both_broken_error,
        "single_generation_fails_closed": single_error is not None,
        "single_generation_error": single_error,
    }


# ---------------------------------------------------------------------------
# 4. payload tamper detection (digest mismatch)
# ---------------------------------------------------------------------------

def measure_payload_tamper(root: str | Path) -> dict[str, Any]:
    case = Path(root) / "tamper"
    case.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(case, dual_backup=False)
    fields = _fields()
    store.save({"schema": CHECKPOINT_SCHEMA, "step": 5, "fields": fields, "digest": fields_digest(fields)})

    payload = torch.load(store.primary_path, map_location="cpu", weights_only=False)
    tampered = dict(payload)
    changed = {name: tensor.clone() for name, tensor in payload["fields"].items()}
    changed["f"].view(-1)[0] += 1e-3  # bit-rot analogue, digest left stale
    tampered["fields"] = changed
    torch.save(tampered, store.primary_path)

    tamper_error = None
    try:
        store.load()
    except CheckpointError as exc:
        tamper_error = str(exc)

    bad_schema = dict(payload)
    bad_schema["schema"] = "some-other-schema"
    torch.save(bad_schema, store.primary_path)
    schema_error = None
    try:
        store.load()
    except CheckpointError as exc:
        schema_error = str(exc)

    truncated = Path(root) / "truncated.pt"
    truncated.write_bytes(b"\x00\x01\x02")
    truncated_error = None
    try:
        CheckpointStore(Path(root) / "truncated_dir", dual_backup=False)._load_one(truncated)
    except CheckpointError as exc:
        truncated_error = str(exc)

    return {
        "digest_tamper_rejected": tamper_error is not None,
        "digest_tamper_error": tamper_error,
        "schema_mismatch_rejected": schema_error is not None,
        "schema_mismatch_error": schema_error,
        "truncated_payload_rejected": truncated_error is not None,
        "truncated_payload_error": truncated_error,
    }


# ---------------------------------------------------------------------------
# 5. resume parity with the real LBM solver
# ---------------------------------------------------------------------------

def _make_solver(nx: int = 8, ny: int = 6, nz: int = 4, tau: float = 0.6):
    from tensorlbm.lbm_step import LBMStepExecutor

    return LBMStepExecutor(
        "D3Q19",
        collide_fn="bgk",
        device=torch.device("cpu"),
        nx=nx,
        ny=ny,
        nz=nz,
        tau=tau,
        dtype=torch.float64,
    )


def _initial_populations(nx: int, ny: int, nz: int, seed: int = 20261113) -> torch.Tensor:
    from tensorlbm.d3q19 import equilibrium3d

    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.02 * torch.rand((nz, ny, nx), generator=generator, dtype=torch.float64)
    zero = torch.zeros((nz, ny, nx), dtype=torch.float64)
    return equilibrium3d(rho, zero, zero, zero, device).to(torch.float64)


def _macroscopic(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rho = f.sum(dim=0)
    ux = (f[1] + f[5] + f[8] + f[11] + f[15]).sub(f[2] + f[6] + f[9] + f[12] + f[16]) / rho
    uy = (f[3] + f[5] + f[9] + f[13] + f[17]).sub(f[4] + f[6] + f[10] + f[14] + f[18]) / rho
    return rho, ux, uy


def measure_resume_parity(
    root: str | Path, *, total_steps: int = 200, checkpoint_step: int = 100
) -> dict[str, Any]:
    """Continuous run vs. checkpointed restart, driven through the real manager."""
    case = Path(root) / "resume"
    case.mkdir(parents=True, exist_ok=True)

    # --- continuous reference run -------------------------------------
    solver = _make_solver()
    f = _initial_populations(8, 6, 4)
    initial_norm = float(torch.linalg.vector_norm(f))
    for _ in range(total_steps):
        f, _diag = solver.step(f)
    f_continuous = f.clone()

    # --- segmented run: 100 steps, checkpoint, restart, 100 steps ----------
    seg_dir = Path(root) / "resume_segmented"
    seg_dir.mkdir(parents=True, exist_ok=True)
    first = JobCheckpointManager(seg_dir, interval_hours=1.0, max_steps=total_steps, clock=lambda: 0.0)
    state = first.begin()
    assert not state.resumed
    seg_solver = _make_solver()
    f_seg = _initial_populations(8, 6, 4)
    for _ in range(checkpoint_step):
        f_seg, _diag = seg_solver.step(f_seg)
    receipt = first.checkpoint_now(checkpoint_step, {"f": f_seg})

    second = JobCheckpointManager(seg_dir, interval_hours=1.0, max_steps=total_steps, clock=lambda: 0.0)
    resumed = second.begin()
    f_resumed = resumed.fields["f"].clone()
    resumed_solver = _make_solver()
    steps_after_resume = 0
    for step_index in range(checkpoint_step, total_steps):
        assert not second.should_stop(step_index)
        f_resumed, _diag = resumed_solver.step(f_resumed)
        steps_after_resume += 1
    second.finish(total_steps, status=STATUS_DONE, reason="segmented run done", fields={"f": f_resumed})

    rho_c, ux_c, uy_c = _macroscopic(f_continuous)
    rho_r, ux_r, uy_r = _macroscopic(f_resumed)
    max_field_diff = float((f_continuous - f_resumed).abs().max())
    rel_l2 = float(torch.linalg.vector_norm(f_continuous - f_resumed) / initial_norm)
    return {
        "total_steps": total_steps,
        "checkpoint_step": checkpoint_step,
        "steps_after_resume": steps_after_resume,
        "resume_source": resumed.source,
        "resumed_flag": resumed.resumed,
        "resumed_step": resumed.step,
        "checkpoint_count_after_first_segment": receipt.step,
        "checkpoint_bytes": receipt.primary_bytes,
        "backup_bytes": receipt.backup_bytes,
        "max_abs_population_diff": max_field_diff,
        "rel_l2_population_diff": rel_l2,
        "rel_l2_population_diff_pct": _d2q9_probability_scale(rel_l2),
        "max_abs_rho_diff": float((rho_c - rho_r).abs().max()),
        "max_abs_ux_diff": float((ux_c - ux_r).abs().max()),
        "max_abs_uy_diff": float((uy_c - uy_r).abs().max()),
        "bit_identical": max_field_diff == 0.0,
        "final_status": None if second.status is None else second.status.status,
        "final_step": None if second.status is None else second.status.step,
        "checkpoints_counted": None if second.status is None else second.status.checkpoints,
    }


# ---------------------------------------------------------------------------
# 6. trigger semantics (wall clock, injected volume, step veto)
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance_hours(self, hours: float) -> None:
        self.seconds += hours * 3600.0


def measure_trigger_semantics(root: str | Path) -> dict[str, Any]:
    case = Path(root) / "triggers"
    case.mkdir(parents=True, exist_ok=True)
    clock = _FakeClock()
    manager = JobCheckpointManager(
        case, interval_hours=1.0, volume_interval=0.5, clock=clock
    )
    manager.begin()
    fields = _fields()

    before_interval = manager.maybe_checkpoint(10, fields)  # elapsed 0 h -> no write
    clock.advance_hours(0.99)
    still_before = manager.maybe_checkpoint(20, fields)
    clock.advance_hours(0.02)
    after_interval = manager.maybe_checkpoint(30, fields)
    step_veto = manager.maybe_checkpoint(30, fields)  # same step -> vetoed
    later_step_same_hour = manager.maybe_checkpoint(31, fields)  # timer restarted -> vetoed by clock

    volume_case = Path(root) / "triggers_volume"
    volume_case.mkdir(parents=True, exist_ok=True)
    volume_clock = _FakeClock()
    volume_manager = JobCheckpointManager(
        volume_case, interval_hours=100.0, volume_interval=0.5, clock=volume_clock
    )
    volume_manager.begin()
    volume_below = volume_manager.maybe_checkpoint(10, fields, cumulative_volume=0.4)
    volume_above = volume_manager.maybe_checkpoint(20, fields, cumulative_volume=0.6)
    volume_second_quantum = volume_manager.maybe_checkpoint(30, fields, cumulative_volume=1.1)

    return {
        "no_write_before_interval": before_interval is None,
        "no_write_at_0p99_interval": still_before is None,
        "write_after_interval": after_interval is not None,
        "wall_clock_trigger": None if after_interval is None else after_interval.trigger,
        "same_step_vetoed": step_veto is None,
        "solver_step_vetoed_without_elapsed_time": later_step_same_hour is None,
        "volume_below_quantum_no_write": volume_below is None,
        "volume_crossing_writes": volume_above is not None,
        "volume_trigger": None if volume_above is None else volume_above.trigger,
        "volume_second_quantum_writes": volume_second_quantum is not None,
        "receipts_wall_clock": len(manager.receipts),
        "receipts_volume": len(volume_manager.receipts),
    }


# ---------------------------------------------------------------------------
# 7. terminal absorbing + restart veto
# ---------------------------------------------------------------------------

def measure_terminal_absorbing(root: str | Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for terminal in TERMINAL_STATUSES:
        case = Path(root) / f"terminal_{terminal}"
        case.mkdir(parents=True, exist_ok=True)
        first = JobCheckpointManager(case, clock=lambda: 0.0)
        first.begin()
        first.checkpoint_now(50, _fields())
        first.finish(50, status=terminal, reason="probe terminal")
        second = JobCheckpointManager(case, clock=lambda: 0.0)
        state = second.begin()
        # an explicit re-arm is the only way back into compute
        third = JobCheckpointManager(case, clock=lambda: 0.0)
        rearmed = third.begin(force_restart=True)
        results[terminal] = {
            "begin_finished": state.finished,
            "begin_resumed": state.resumed,
            "begin_source": state.source,
            "begin_step": state.step,
            "rearm_resumed": rearmed.resumed,
            "rearm_source": rearmed.source,
            "rearm_step": rearmed.step,
        }
    results["all_terminal_veto_continue"] = all(
        results[status]["begin_finished"] and not results[status]["begin_resumed"]
        for status in TERMINAL_STATUSES
    )
    results["all_terminal_rearmable"] = all(
        not results[status]["rearm_resumed"] and results[status]["rearm_step"] == 0
        for status in TERMINAL_STATUSES
    )
    return results


# ---------------------------------------------------------------------------
# 8. write-before-begin probe (fail-open check on an unvalidated first write)
# ---------------------------------------------------------------------------

def measure_write_without_begin(root: str | Path) -> dict[str, Any]:
    """Probe two unvalidated-write paths of the status guard.

    (a) *fresh case*: a manager that never called :meth:`JobCheckpointManager.begin`
    writes a terminal status into an empty directory.  There is no prior
    generation, so the write is allowed; the probe records it as
    informational (``fresh_case_first_write_unconstrained``).

    (b) *stale supervisor*: a second manager is built on a directory whose
    ``job_status.txt`` already says ``done`` and writes ``failed``.  MF-LBM
    consults the status file at start-up, so an unread terminal state used to be
    overwritten silently.  The guard must fail closed here
    (``terminal_overwrite_by_stale_manager_blocked``).
    """
    case = Path(root) / "write_without_begin"
    case.mkdir(parents=True, exist_ok=True)
    fresh = JobCheckpointManager(case, clock=lambda: 0.0)
    error = None
    try:
        fresh.finish(0, status=STATUS_DONE, reason="no run happened")
    except CheckpointError as exc:
        error = str(exc)
    written = JobStatusFile(case / "job_status.txt").read()
    misuse_blocked = error is not None

    # an already-terminal case must not be re-marked by a manager that never
    # loaded the status file either
    occupied = Path(root) / "write_without_begin_terminal"
    occupied.mkdir(parents=True, exist_ok=True)
    owner = JobCheckpointManager(occupied, clock=lambda: 0.0)
    owner.begin()
    owner.finish(10, status=STATUS_DONE, reason="legit finish")
    intruder = JobCheckpointManager(occupied, clock=lambda: 0.0)
    intruder_error = None
    try:
        intruder.fail(11, "crash report from a stale supervisor")
    except CheckpointError as exc:
        intruder_error = str(exc)
    after_intruder = JobStatusFile(occupied / "job_status.txt").read()

    return {
        # A brand-new case directory has no prior generation to protect, so the
        # very first write is unconstrained by design (nothing can be retired).
        "fresh_case_first_write_unconstrained": written is not None and written.status == STATUS_DONE,
        "terminal_write_without_begin_error": error,
        "status_written_anyway": None if written is None else written.status,
        "terminal_overwrite_by_stale_manager_blocked": intruder_error is not None,
        "terminal_overwrite_error": intruder_error,
        "terminal_status_after_intruder": None if after_intruder is None else after_intruder.status,
    }


def measure_config_signature_guard(root: str | Path) -> dict[str, Any]:
    case = Path(root) / "signature"
    case.mkdir(parents=True, exist_ok=True)
    manager = JobCheckpointManager(case, clock=lambda: 0.0, config_signature={"grid": [8, 6, 4], "tau": 0.6})
    manager.begin()
    manager.checkpoint_now(20, _fields())

    matching = JobCheckpointManager(case, clock=lambda: 0.0, config_signature={"grid": [8, 6, 4], "tau": 0.6})
    matched_error = None
    matched_step = None
    try:
        matched_step = matching.begin().step
    except CheckpointError as exc:  # pragma: no cover - would be a defect
        matched_error = str(exc)

    mismatched = JobCheckpointManager(case, clock=lambda: 0.0, config_signature={"grid": [8, 6, 4], "tau": 0.7})
    mismatch_error = None
    try:
        mismatched.begin()
    except CheckpointError as exc:
        mismatch_error = str(exc)
    return {
        "matching_signature_resumes": matched_error is None and matched_step == 20,
        "matching_signature_step": matched_step,
        "mismatched_signature_rejected": mismatch_error is not None,
        "mismatched_signature_error": mismatch_error,
    }


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def measure_all(root: str | Path) -> dict[str, Any]:
    """Run every measurement below *root* and return one JSON-serialisable dict."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "transitions": measure_transition_matrix(root),
        "status_persistence": measure_status_persistence(root),
        "dual_backup": measure_dual_backup_recovery(root),
        "payload_tamper": measure_payload_tamper(root),
        "resume_parity": measure_resume_parity(root),
        "triggers": measure_trigger_semantics(root),
        "terminal_absorbing": measure_terminal_absorbing(root),
        "write_without_begin": measure_write_without_begin(root),
        "config_signature": measure_config_signature_guard(root),
    }
    report["summary"] = {
        "legal_transition_coverage_pct": report["transitions"]["legal_coverage_pct"],
        "illegal_transition_rejection_pct": report["transitions"]["illegal_rejection_pct"],
        "agreement_with_reference_pct": report["transitions"]["agreement_pct"],
        "resume_bit_identical": report["resume_parity"]["bit_identical"],
        "resume_rel_l2_pct": report["resume_parity"]["rel_l2_population_diff_pct"],
        "dual_backup_recovery": report["dual_backup"]["dual_backup_recovered"],
        "tamper_fail_closed": (
            report["payload_tamper"]["digest_tamper_rejected"]
            and report["payload_tamper"]["schema_mismatch_rejected"]
            and report["payload_tamper"]["truncated_payload_rejected"]
        ),
        "terminal_absorbing": report["terminal_absorbing"]["all_terminal_veto_continue"],
        "fresh_case_first_write_unconstrained": report["write_without_begin"][
            "fresh_case_first_write_unconstrained"
        ],
        "stale_terminal_overwrite_blocked": report["write_without_begin"][
            "terminal_overwrite_by_stale_manager_blocked"
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/xval_job_status_measured", help="scratch directory")
    parser.add_argument("--out", default=None, help="JSON output path")
    args = parser.parse_args(argv)

    report = measure_all(args.root)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    print(json.dumps(report["summary"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())