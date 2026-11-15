"""Measured cross-validation of the job-status/checkpoint machine.

These tests assert *invariants* of :mod:`tensorlbm.job_checkpoint` measured by
driving the real module (see
:mod:`tensorlbm.verification.job_status_checkpoint_measured`), not just that the
code runs:

* the transition guard agrees with the lanl/MF-LBM reference table on all 25
  ``(from, to)`` status pairs (absorbing terminals included);
* resume through a checkpoint is bit-identical to an uninterrupted run;
* the second (backup) generation carries a run whose primary generation rotted;
* corrupt / truncated / tampered payloads are rejected instead of mis-loaded;
* wall-clock and injected-volume triggers gate checkpoints (and the step veto
  stops duplicate writes);
* a manager that never read ``job_status.txt`` cannot overwrite a terminal
  generation (regression test for the fail-closed status guard).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorlbm.job_checkpoint import (
    STATUS_CONTINUE,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_REACHED_MAX_STEP,
    CheckpointError,
    JobCheckpointManager,
    JobStatusFile,
    JobStatusRecord,
)
from tensorlbm.verification.job_status_checkpoint_measured import (
    MF_LBM_LEGAL_TRANSITIONS,
    measure_all,
)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("xval_job_status_measured")
    return measure_all(root)


class TestTransitionMatrix:
    def test_guard_agrees_with_reference_on_every_pair(self, evidence: dict) -> None:
        transitions = evidence["transitions"]
        assert transitions["total_pairs"] == 25
        assert transitions["mismatches"] == []
        assert transitions["agreement_pct"] == 100.0

    def test_legal_transitions_are_all_accepted(self, evidence: dict) -> None:
        transitions = evidence["transitions"]
        assert transitions["legal_pairs"] == 12
        assert transitions["legal_coverage_pct"] == 100.0

    def test_illegal_transitions_are_all_rejected(self, evidence: dict) -> None:
        transitions = evidence["transitions"]
        assert transitions["illegal_pairs"] == 13
        assert transitions["illegal_rejection_pct"] == 100.0

    def test_terminal_states_are_absorbing_in_reference_and_impl(self, evidence: dict) -> None:
        for status in (STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP):
            assert MF_LBM_LEGAL_TRANSITIONS[status] == frozenset({status})
            assert evidence["transitions"]["implementation_table_equals_reference"][status] is True

    def test_rejected_pair_raises_with_actionable_message(self, tmp_path: Path) -> None:
        status_file = JobStatusFile(tmp_path / "job_status.txt")
        status_file.write(JobStatusRecord(status=STATUS_DONE, step=9))
        with pytest.raises(CheckpointError, match="illegal job status transition"):
            status_file.write(
                JobStatusRecord(status=STATUS_FAILED, step=10),
                previous=JobStatusRecord(status=STATUS_DONE, step=9),
            )


class TestStatusFile:
    def test_roundtrip_preserves_all_fields(self, evidence: dict) -> None:
        persistence = evidence["status_persistence"]
        assert persistence["first_token_is_status"] is True
        assert persistence["roundtrip_equal"] is True
        assert persistence["missing_file_returns_none"] is True
        assert persistence["empty_file_returns_none"] is True


class TestCheckpointIntegrity:
    def test_backup_generation_carries_corrupt_primary(self, evidence: dict) -> None:
        backup = evidence["dual_backup"]
        assert backup["clean_source"] == "primary"
        assert backup["source_after_primary_corruption"] == "backup"
        assert backup["recovered_step"] == 10
        assert backup["dual_backup_recovered"] is True

    def test_both_generations_corrupt_fails_closed(self, evidence: dict) -> None:
        backup = evidence["dual_backup"]
        assert backup["both_corrupt_fails_closed"] is True
        assert "no usable checkpoint generation" in backup["both_corrupt_error"]
        assert backup["single_generation_fails_closed"] is True

    def test_payload_tamper_and_schema_and_truncation_rejected(self, evidence: dict) -> None:
        tamper = evidence["payload_tamper"]
        assert tamper["digest_tamper_rejected"] is True
        assert "digest mismatch" in tamper["digest_tamper_error"]
        assert tamper["schema_mismatch_rejected"] is True
        assert tamper["truncated_payload_rejected"] is True


class TestResumeParity:
    def test_resume_is_bit_identical_to_continuous_run(self, evidence: dict) -> None:
        parity = evidence["resume_parity"]
        assert parity["resumed_flag"] is True
        assert parity["resumed_step"] == parity["checkpoint_step"]
        assert parity["steps_after_resume"] == parity["total_steps"] - parity["checkpoint_step"]
        assert parity["bit_identical"] is True
        assert parity["max_abs_population_diff"] == 0.0
        assert parity["max_abs_rho_diff"] == 0.0
        assert parity["max_abs_ux_diff"] == 0.0

    def test_resume_error_stays_far_below_owner_bar(self, evidence: dict) -> None:
        parity = evidence["resume_parity"]
        # owner precision bar is 15%; the honest target of this suite is 3%
        assert parity["rel_l2_population_diff_pct"] < 3.0
        assert parity["rel_l2_population_diff_pct"] == 0.0

    def test_terminal_status_records_step_and_checkpoint_count(self, evidence: dict) -> None:
        parity = evidence["resume_parity"]
        assert parity["final_status"] == STATUS_DONE
        assert parity["final_step"] == 200
        assert parity["checkpoints_counted"] == 2


class TestTriggers:
    def test_wall_clock_gate(self, evidence: dict) -> None:
        triggers = evidence["triggers"]
        assert triggers["no_write_before_interval"] is True
        assert triggers["no_write_at_0p99_interval"] is True
        assert triggers["write_after_interval"] is True
        assert triggers["wall_clock_trigger"] == "wall_clock"

    def test_step_and_elapsed_time_veto_duplicate_writes(self, evidence: dict) -> None:
        triggers = evidence["triggers"]
        assert triggers["same_step_vetoed"] is True
        assert triggers["solver_step_vetoed_without_elapsed_time"] is True
        assert triggers["receipts_wall_clock"] == 1

    def test_injected_volume_trigger(self, evidence: dict) -> None:
        triggers = evidence["triggers"]
        assert triggers["volume_below_quantum_no_write"] is True
        assert triggers["volume_crossing_writes"] is True
        assert triggers["volume_trigger"] == "volume"
        assert triggers["volume_second_quantum_writes"] is True


class TestTerminalStates:
    def test_every_terminal_state_vetoes_continue(self, evidence: dict) -> None:
        terminal = evidence["terminal_absorbing"]
        assert terminal["all_terminal_veto_continue"] is True
        for status in (STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP):
            assert terminal[status]["begin_finished"] is True
            assert terminal[status]["begin_resumed"] is False
            assert terminal[status]["begin_source"] == "terminal"

    def test_force_restart_rearms_terminal_case(self, evidence: dict) -> None:
        assert evidence["terminal_absorbing"]["all_terminal_rearmable"] is True

    def test_stale_manager_cannot_overwrite_terminal_status(self, evidence: dict) -> None:
        probe = evidence["write_without_begin"]
        assert probe["terminal_overwrite_by_stale_manager_blocked"] is True
        assert "illegal job status transition" in probe["terminal_overwrite_error"]
        assert probe["terminal_status_after_intruder"] == STATUS_DONE
        # a brand-new directory has no generation to protect
        assert probe["fresh_case_first_write_unconstrained"] is True


class TestStatusGuardRegression:
    """Direct regression tests for the fail-closed ``_write_status`` guard."""

    def test_unread_terminal_generation_is_not_overwritten(self, tmp_path: Path) -> None:
        owner = JobCheckpointManager(tmp_path, clock=lambda: 0.0)
        owner.begin()
        owner.finish(10, status=STATUS_DONE, reason="legit finish")

        stale = JobCheckpointManager(tmp_path, clock=lambda: 0.0)
        with pytest.raises(CheckpointError, match="done"):
            stale.fail(11, "crash report from a stale supervisor")
        record = JobStatusFile(tmp_path / "job_status.txt").read()
        assert record is not None
        assert record.status == STATUS_DONE
        assert record.step == 10

    def test_unread_continue_generation_is_appendable(self, tmp_path: Path) -> None:
        owner = JobCheckpointManager(tmp_path, clock=lambda: 0.0)
        owner.begin()
        owner.checkpoint_now(5, {"f": __import__("torch").ones((19, 2, 2, 2))})

        resumed = JobCheckpointManager(tmp_path, clock=lambda: 0.0)
        state = resumed.begin()
        assert state.resumed is True and state.step == 5
        assert resumed.status is not None and resumed.status.status == STATUS_CONTINUE

    def test_force_restart_still_bypasses_terminal_guard(self, tmp_path: Path) -> None:
        first = JobCheckpointManager(tmp_path, clock=lambda: 0.0)
        first.begin()
        first.finish(3, status=STATUS_REACHED_MAX_STEP, reason="max steps")

        rearmed = JobCheckpointManager(tmp_path, clock=lambda: 0.0)
        state = rearmed.begin(force_restart=True)
        assert state.resumed is False and state.step == 0
        assert rearmed.status is not None and rearmed.status.status == STATUS_CONTINUE
        assert JobStatusFile(tmp_path / "job_status.txt").read().status == STATUS_CONTINUE

    def test_begin_still_works_on_empty_directory(self, tmp_path: Path) -> None:
        manager = JobCheckpointManager(tmp_path, clock=lambda: 0.0)
        state = manager.begin()
        assert state.resumed is False and state.source == "fresh"
        assert JobStatusFile(tmp_path / "job_status.txt").read().status == STATUS_CONTINUE


class TestConfigSignature:
    def test_matching_signature_resumes_and_mismatch_is_refused(self, evidence: dict) -> None:
        signature = evidence["config_signature"]
        assert signature["matching_signature_resumes"] is True
        assert signature["matching_signature_step"] == 20
        assert signature["mismatched_signature_rejected"] is True
        assert "config signature mismatch" in signature["mismatched_signature_error"]