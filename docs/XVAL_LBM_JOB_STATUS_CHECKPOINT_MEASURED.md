# XVAL (measured round): job-status-checkpoint machine vs lanl/MF-LBM

**Anchor**: `XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE` · **Repo**: TensorLBM
**Evidence**: `outputs/xval_job_status_checkpoint_measured/evidence.json`
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

The reference transition set therefore has 12 legal and
13 illegal pairs over the five statuses.

## 3. Method

| measurement | what actually executes |
|---|---|
| transition matrix | `JobStatusFile.write(record, previous=...)` over all 25 status pairs |
| status persistence | write/read `job_status.txt`, first-token-is-status, empty file handling |
| dual backup | corrupt the primary generation, then both generations |
| tamper detection | stale digest, wrong schema tag, truncated payload |
| resume parity | real `LBMStepExecutor` D3Q19 8×6×4, float64, 200 steps vs 100+restart+100 |
| trigger gates | injected clock + injected cumulative volume, step veto |
| terminal absorbing | `begin()` on each of the three terminal states, then explicit re-arm |
| stale supervisor | a manager that never read `job_status.txt` tries to overwrite a terminal generation |

## 4. Results

### 4.1 Transition guard

| metric | measured | criterion |
|---|---|---|
| legal transitions accepted | 12/12 (100%) | =100% |
| illegal transitions rejected | 13/13 (100%) | =100% |
| agreement with reference table | 100% of 25 pairs | =100% |

| from → to | legal in MF-LBM | guard accepted | agrees |
|---|---|---|---|
| new → new | True | True | True |
| new → continue | True | True | True |
| new → done | True | True | True |
| new → failed | True | True | True |
| new → reached_max_step | True | True | True |
| continue → new | False | False | True |
| continue → continue | True | True | True |
| continue → done | True | True | True |
| continue → failed | True | True | True |
| continue → reached_max_step | True | True | True |
| done → new | False | False | True |
| done → continue | False | False | True |
| done → done | True | True | True |
| done → failed | False | False | True |
| done → reached_max_step | False | False | True |
| failed → new | False | False | True |
| failed → continue | False | False | True |
| failed → done | False | False | True |
| failed → failed | True | True | True |
| failed → reached_max_step | False | False | True |
| reached_max_step → new | False | False | True |
| reached_max_step → continue | False | False | True |
| reached_max_step → done | False | False | True |
| reached_max_step → failed | False | False | True |
| reached_max_step → reached_max_step | True | True | True |

The implementation table equals the reference table for every status
(`implementation_table_equals_reference` all `true` in `evidence.json`). The
"missing COMPLETED→QUEUED transition" reported in round 1 is **not a gap**: the
MF-LBM machine treats terminal states as absorbing, so refusing that transition
is the reference behaviour.

### 4.2 Checkpoint integrity

| check | measured | note |
|---|---|---|
| status file round-trip | True | status/step/reason/updated_at/checkpoints all preserved |
| clean load source | `primary` | |
| primary corrupted → recovered | `backup`, step 10 | second generation carries the run |
| both generations corrupt | fail-closed True | `no usable checkpoint generation` |
| single generation corrupt | fail-closed True | `dual_backup=False` has no fallback |
| stale digest | rejected True | digest mismatch |
| wrong schema tag | rejected True | |
| truncated payload | rejected True | |

### 4.3 Resume parity (the numeric bar)

| quantity | value |
|---|---|
| restart source | `primary` at step 100 |
| steps executed after restart | 100 |
| max abs population difference | 0.000e+00 |
| relative L2 population difference | 0.000e+00 % |
| max abs rho difference | 0.000e+00 |
| max abs ux difference | 0.000e+00 |
| bit-identical | True |
| checkpoint size (primary / backup) | 31194 B / 31194 B |
| final status / step / count | done / 200 / 2 |

The per-step gap after the restart is exactly zero for all 101
steps (fig_resume_parity.png).

### 4.4 Trigger semantics

| gate | measured |
|---|---|
| no write at 0.00 h of a 1 h interval | True |
| no write at 0.99 h | True |
| write at 1.01 h | True (`wall_clock`) |
| second write at the same step | vetoed True |
| step+1 but no elapsed time | vetoed True |
| cumulative volume below quantum (0.4 of 0.5) | no write True |
| volume crossing (0.6) / next quantum (1.1) | write / write (`volume`) |

### 4.5 Terminal states and stale supervisors

| probe | measured |
|---|---|
| done / failed / reached_max_step veto `continue` | True |
| explicit `begin(force_restart=True)` re-arms | True |
| fresh directory first write unconstrained (nothing to retire) | True → status `done` |
| stale manager overwriting a terminal generation | blocked True |
| status after the blocked overwrite | `done` |
| config signature mismatch refused | True |

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
| resume field difference | uninterrupted run | 0.000e+00 abs / 0.000e+00 % rel-L2 | 0 | **yes (exact)** |
| legal transition coverage | MF-LBM reference set | 12/12 = 100% | 0 pp | **yes** |
| illegal transition rejection | absorbing-terminal rule | 13/13 = 100% | 0 pp | **yes** |
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
