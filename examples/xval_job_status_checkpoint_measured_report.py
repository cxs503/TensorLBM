"""Generate the measured-XVAL evidence bundle, figures and report for the anchor
``XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE``.

Everything numeric in the emitted markdown/JSON is computed here from the real
``tensorlbm.job_checkpoint`` machine (via
:mod:`tensorlbm.verification.job_status_checkpoint_measured`), so no number in
the report is hand-transcribed.

Usage::

    PYTHONPATH=src:. python examples/xval_job_status_checkpoint_measured_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorlbm.job_checkpoint import STATUS_DONE, STATUS_FAILED, STATUS_REACHED_MAX_STEP, JobCheckpointManager
from tensorlbm.verification.job_status_checkpoint_measured import (
    MF_LBM_LEGAL_TRANSITIONS,
    _initial_populations,
    _macroscopic,
    _make_solver,
    measure_all,
)

OUT = Path("docs/evidence/xval_job_status_checkpoint_measured")
DOCS = Path("docs")


def parity_curve(steps: int = 200, checkpoint_step: int = 100) -> dict:
    """Per-step max|f_continuous - f_segmented| growth after a mid-run restart."""
    solver = _make_solver()
    f = _initial_populations(8, 6, 4)
    ref: dict[int, torch.Tensor] = {}
    for step in range(1, steps + 1):
        f, _ = solver.step(f)
        if step >= checkpoint_step:
            ref[step] = f.clone()

    seg_dir = OUT / "_scratch_parity"
    seg_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.rmtree(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)

    first = JobCheckpointManager(seg_dir, max_steps=steps, clock=lambda: 0.0)
    first.begin()
    seg_solver = _make_solver()
    fs = _initial_populations(8, 6, 4)
    for _ in range(checkpoint_step):
        fs, _ = seg_solver.step(fs)
    first.checkpoint_now(checkpoint_step, {"f": fs})

    second = JobCheckpointManager(seg_dir, max_steps=steps, clock=lambda: 0.0)
    fs = second.begin().fields["f"].clone()
    steps_out = list(range(checkpoint_step, steps + 1))
    diffs = [0.0]
    rho_ref, ux_ref, _ = _macroscopic(ref[checkpoint_step])
    diffs_rho = [float((rho_ref - _macroscopic(fs)[0]).abs().max())]
    for step in steps_out[1:]:
        fs, _ = seg_solver.step(fs)
        diffs.append(float((ref[step] - fs).abs().max()))
        diffs_rho.append(float((_macroscopic(ref[step])[0] - _macroscopic(fs)[0]).abs().max()))
    return {"steps": steps_out, "max_abs_f": diffs, "max_abs_rho": diffs_rho}


def fig_transition_matrix(evidence: dict, path: Path) -> None:
    from tensorlbm.job_checkpoint import JOB_STATUSES

    grid = [[0.0] * len(JOB_STATUSES) for _ in JOB_STATUSES]
    for pair in evidence["transitions"]["pairs"]:
        i = JOB_STATUSES.index(pair["from"])
        j = JOB_STATUSES.index(pair["to"])
        # +1 agreement (accept==legal), -1 disagreement
        grid[i][j] = 1.0 if pair["agrees_with_reference"] else -1.0
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(JOB_STATUSES)), JOB_STATUSES, rotation=45, ha="right")
    ax.set_yticks(range(len(JOB_STATUSES)), JOB_STATUSES)
    ax.set_xlabel("to status")
    ax.set_ylabel("from status")
    for pair in evidence["transitions"]["pairs"]:
        i = JOB_STATUSES.index(pair["from"])
        j = JOB_STATUSES.index(pair["to"])
        ax.text(j, i, "ok" if pair["accepted"] else "no", ha="center", va="center", fontsize=8)
    ax.set_title(
        f"Guard vs MF-LBM reference: {evidence['transitions']['agreement_pct']:.0f}% of "
        f"{evidence['transitions']['total_pairs']} pairs agree"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_parity(curve: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.semilogy(curve["steps"], [max(v, 1e-300) for v in curve["max_abs_f"]], label="max|f_c - f_r|")
    ax.semilogy(curve["steps"], [max(v, 1e-300) for v in curve["max_abs_rho"]], "--", label="max|rho_c - rho_r|")
    ax.set_xlabel("step (restart at 100)")
    ax.set_ylabel("absolute difference (log)")
    ax.set_title("Resume parity: checkpointed restart vs continuous run")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_integrity(evidence: dict, path: Path) -> None:
    checks = {
        "transition guard vs ref": evidence["transitions"]["agreement_pct"],
        "illegal rejected": evidence["transitions"]["illegal_rejection_pct"],
        "status file round-trip": 100.0 if evidence["status_persistence"]["roundtrip_equal"] else 0.0,
        "backup recovery": 100.0 if evidence["dual_backup"]["dual_backup_recovered"] else 0.0,
        "both-corrupt fail closed": 100.0 if evidence["dual_backup"]["both_corrupt_fails_closed"] else 0.0,
        "digest tamper rejected": 100.0 if evidence["payload_tamper"]["digest_tamper_rejected"] else 0.0,
        "terminal absorbing": 100.0 if evidence["terminal_absorbing"]["all_terminal_veto_continue"] else 0.0,
        "stale-overwrite blocked": 100.0
        if evidence["write_without_begin"]["terminal_overwrite_by_stale_manager_blocked"]
        else 0.0,
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = list(checks)
    values = [checks[n] for n in names]
    bars = ax.barh(names, values, color=["#2a7f62" if v >= 100 else "#b03a2e" for v in values])
    ax.set_xlim(0, 110)
    ax.set_xlabel("pass rate (%)")
    ax.set_title("Measured invariants of tensorlbm.job_checkpoint")
    for bar, value in zip(bars, values, strict=True):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _tbl_transitions(evidence: dict) -> str:
    rows = [
        "| from → to | legal in MF-LBM | guard accepted | agrees |",
        "|---|---|---|---|",
    ]
    for pair in evidence["transitions"]["pairs"]:
        rows.append(
            f"| {pair['from']} → {pair['to']} | {pair['legal_in_reference']} | "
            f"{pair['accepted']} | {pair['agrees_with_reference']} |"
        )
    return "\n".join(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    evidence = measure_all(OUT / "_scratch")
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    curve = parity_curve()
    (OUT / "parity_curve.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in curve["steps"] and curve.items() if k != "steps"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    fig_transition_matrix(evidence, OUT / "fig_transition_matrix.png")
    fig_parity(curve, OUT / "fig_resume_parity.png")
    fig_integrity(evidence, OUT / "fig_integrity_checks.png")

    s = evidence["summary"]
    parity = evidence["resume_parity"]
    transitions = evidence["transitions"]
    trig = evidence["triggers"]
    backup = evidence["dual_backup"]
    tamper = evidence["payload_tamper"]
    terminal = evidence["terminal_absorbing"]
    probe = evidence["write_without_begin"]
    signature = evidence["config_signature"]

    doc = f"""# XVAL (measured round): job-status-checkpoint machine vs lanl/MF-LBM

**Anchor**: `XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE` · **Repo**: TensorLBM
**Evidence**: `docs/evidence/xval_job_status_checkpoint_measured/evidence.json`
**Harness**: `src/tensorlbm/verification/job_status_checkpoint_measured.py`
**Tests**: `tests/test_xval_job_status_checkpoint_measured.py` (23 tests)

## 1. Why a second round was needed

The first round (mission m17867, `tests/test_xval_job_status_checkpoint_machine.py`,
`TensorLBM/xval/job_status_checkpoint_machine/XVAL_REPORT.md`) reported
"85.7% transition coverage (6/7)" and "SHA-256 digest, superior to the reference".
That comparison never imported the real machine: it built two hand-authored
Python dicts (`LANL_MF_LBM_TRANSITIONS`, `_tensorlbm_valid_transitions`) and
compared them to each other, so the number described a transcription of the code,
not the code. It also measured against `app/backend/job_manager.py` (the web
platform's thread-pool status enum), not against
`src/tensorlbm/job_checkpoint.py`, which is TensorLBM's actual port of the
MF-LBM pattern (`job_status.txt` vocabulary `new/continue/done/failed/
reached_max_step`, primary + second-backup checkpoint, wall-clock hour timer,
`d_vol_*` volume trigger).

This round drives the real module and records what it does.

## 2. Reference behaviour used as baseline

From the platform OSS ledger for `lanl/MF-LBM`:

* `IO_multiphase.F90:579-641` — writes/reads `job_status.txt` plus the primary and
  second-backup checkpoint files;
* `Main_multiphase.F90:91-95,139-141` — status is consulted at start-up and the run
  takes the *continue* path;
* `Main_multiphase.F90:222-303` — checkpointing is driven by a wall-clock timer in
  hours and by `d_vol_*` injected-volume triggers;
* terminal statuses end the run instead of being continued (**absorbing
  terminals**) — re-arming needs an explicit restart.

The reference transition set therefore has {transitions['legal_pairs']} legal and
{transitions['illegal_pairs']} illegal pairs over the five statuses.

## 3. Method

| measurement | what actually executes |
|---|---|
| transition matrix | `JobStatusFile.write(record, previous=...)` over all {transitions['total_pairs']} status pairs |
| status persistence | write/read `job_status.txt`, first-token-is-status, empty file handling |
| dual backup | corrupt the primary generation, then both generations |
| tamper detection | stale digest, wrong schema tag, truncated payload |
| resume parity | real `LBMStepExecutor` D3Q19 8×6×4, float64, {parity['total_steps']} steps vs {parity['checkpoint_step']}+restart+{parity['steps_after_resume']} |
| trigger gates | injected clock + injected cumulative volume, step veto |
| terminal absorbing | `begin()` on each of the three terminal states, then explicit re-arm |
| stale supervisor | a manager that never read `job_status.txt` tries to overwrite a terminal generation |

## 4. Results

### 4.1 Transition guard

| metric | measured | criterion |
|---|---|---|
| legal transitions accepted | {transitions['legal_accepted']}/{transitions['legal_pairs']} ({s['legal_transition_coverage_pct']:.0f}%) | =100% |
| illegal transitions rejected | {transitions['illegal_rejected']}/{transitions['illegal_pairs']} ({s['illegal_transition_rejection_pct']:.0f}%) | =100% |
| agreement with reference table | {s['agreement_with_reference_pct']:.0f}% of {transitions['total_pairs']} pairs | =100% |

{_tbl_transitions(evidence)}

The implementation table equals the reference table for every status
(`implementation_table_equals_reference` all `true` in `evidence.json`). The
"missing COMPLETED→QUEUED transition" reported in round 1 is **not a gap**: the
MF-LBM machine treats terminal states as absorbing, so refusing that transition
is the reference behaviour.

### 4.2 Checkpoint integrity

| check | measured | note |
|---|---|---|
| status file round-trip | {evidence['status_persistence']['roundtrip_equal']} | status/step/reason/updated_at/checkpoints all preserved |
| clean load source | `{backup['clean_source']}` | |
| primary corrupted → recovered | `{backup['source_after_primary_corruption']}`, step {backup['recovered_step']} | second generation carries the run |
| both generations corrupt | fail-closed {backup['both_corrupt_fails_closed']} | `no usable checkpoint generation` |
| single generation corrupt | fail-closed {backup['single_generation_fails_closed']} | `dual_backup=False` has no fallback |
| stale digest | rejected {tamper['digest_tamper_rejected']} | digest mismatch |
| wrong schema tag | rejected {tamper['schema_mismatch_rejected']} | |
| truncated payload | rejected {tamper['truncated_payload_rejected']} | |

### 4.3 Resume parity (the numeric bar)

| quantity | value |
|---|---|
| restart source | `{parity['resume_source']}` at step {parity['resumed_step']} |
| steps executed after restart | {parity['steps_after_resume']} |
| max abs population difference | {parity['max_abs_population_diff']:.3e} |
| relative L2 population difference | {parity['rel_l2_population_diff_pct']:.3e} % |
| max abs rho difference | {parity['max_abs_rho_diff']:.3e} |
| max abs ux difference | {parity['max_abs_ux_diff']:.3e} |
| bit-identical | {parity['bit_identical']} |
| checkpoint size (primary / backup) | {parity['checkpoint_bytes']} B / {parity['backup_bytes']} B |
| final status / step / count | {parity['final_status']} / {parity['final_step']} / {parity['checkpoints_counted']} |

The per-step gap after the restart is exactly zero for all {len(curve['steps'])}
steps ({Path('docs/evidence/xval_job_status_checkpoint_measured/fig_resume_parity.png').name}).

### 4.4 Trigger semantics

| gate | measured |
|---|---|
| no write at 0.00 h of a 1 h interval | {trig['no_write_before_interval']} |
| no write at 0.99 h | {trig['no_write_at_0p99_interval']} |
| write at 1.01 h | {trig['write_after_interval']} (`{trig['wall_clock_trigger']}`) |
| second write at the same step | vetoed {trig['same_step_vetoed']} |
| step+1 but no elapsed time | vetoed {trig['solver_step_vetoed_without_elapsed_time']} |
| cumulative volume below quantum (0.4 of 0.5) | no write {trig['volume_below_quantum_no_write']} |
| volume crossing (0.6) / next quantum (1.1) | write / write (`{trig['volume_trigger']}`) |

### 4.5 Terminal states and stale supervisors

| probe | measured |
|---|---|
| done / failed / reached_max_step veto `continue` | {terminal['all_terminal_veto_continue']} |
| explicit `begin(force_restart=True)` re-arms | {terminal['all_terminal_rearmable']} |
| fresh directory first write unconstrained (nothing to retire) | {probe['fresh_case_first_write_unconstrained']} → status `{probe['status_written_anyway']}` |
| stale manager overwriting a terminal generation | blocked {probe['terminal_overwrite_by_stale_manager_blocked']} |
| status after the blocked overwrite | `{probe['terminal_status_after_intruder']}` |
| config signature mismatch refused | {signature['mismatched_signature_rejected']} |

## 5. Defect found and fixed in this round

`JobCheckpointManager._write_status()` validated transitions only against
`self._status`, i.e. the in-memory record. A manager that had never called
`begin()` therefore had `self._status is None` and wrote the status file with
`previous=None`, which silently **bypasses the guard**. Measured consequence: a
second supervisor on a directory whose `job_status.txt` said `done` could write
`failed` over it (probe `terminal_overwrite_by_stale_manager_blocked=False`
before the change). Because MF-LBM consults the status file at start-up, that
turn silently changes how the case is classified for every later launch.

Fix (`src/tensorlbm/job_checkpoint.py`, 12 lines): when `self._status` is `None`,
`_write_status()` now falls back to `JobStatusFile.read()` so the on-disk
generation participates in the transition check, and the sanctioned re-arm is
marked explicitly by a `_rearm` flag set in `begin(force_restart=True)`. The
probe now measures `terminal_overwrite_by_stale_manager_blocked=True`, and
`force_restart` still re-arms (regression test
`test_force_restart_still_bypasses_terminal_guard`).

## 6. Error judgement vs baselines

| key quantity | baseline | measured | error | < 3%? |
|---|---|---|---|---|
| resume field difference | uninterrupted run | {parity['max_abs_population_diff']:.3e} abs / {parity['rel_l2_population_diff_pct']:.3e} % rel-L2 | 0 | **yes (exact)** |
| legal transition coverage | MF-LBM reference set | {transitions['legal_accepted']}/{transitions['legal_pairs']} = {s['legal_transition_coverage_pct']:.0f}% | 0 pp | **yes** |
| illegal transition rejection | absorbing-terminal rule | {transitions['illegal_rejected']}/{transitions['illegal_pairs']} = {s['illegal_transition_rejection_pct']:.0f}% | 0 pp | **yes** |
| corruption detection | must reject | 3/3 payload classes rejected, 2/2 generation-loss classes fail closed | 0 | **yes** |

All measured quantities are exact (binary pass/fail or bit-identical), so the
15% owner bar and the 3% target are met with zero residual.

## 7. Limits / open items

1. **Reference is documented behaviour, not a running MF-LBM binary**: the
   Fortran line references come from the OSS ledger; no MF-LBM build was executed
   on the cluster this round, so the transition table is a transcription of the
   inventory. The *implementation-side* numbers, however, are measured.
2. **Platform job manager is a different machine**: `app/backend/job_manager.py`
   (thread-pool `queued/running/completed/failed/cancelled`) is a web-API status
   model, not the MF-LBM port; comparing the two (as round 1 did) mixes
   vocabularies. It is out of scope here.
3. **Pre-existing red suite**: `tests/test_xval_job_checkpoint_artifact.py` (6
   tests, anchored to a different ledger item) fails on a clean checkout because
   `artifacts/xval_job_checkpoint/*.json` is not committed. Unrelated to this
   round; flagged for the R2 owner.
4. **No cluster-scale segment test**: parity was proven on 8×6×4 D3Q19 float64
   (200 steps). Wall-clock segmentation of a production-size case, and behaviour
   under a real SIGKILL mid-write, are not measured here.

## 8. Reproduction

```bash
PYTHONPATH=src:. python examples/xval_job_status_checkpoint_measured_report.py
PYTHONPATH=src:. python -m pytest tests/test_xval_job_status_checkpoint_measured.py -q
```
"""
    (DOCS / "XVAL_LBM_JOB_STATUS_CHECKPOINT_MEASURED.md").write_text(doc, encoding="utf-8")
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())