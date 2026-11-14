"""Round-2 XVAL tests for the job-status/checkpoint machine (mission m20947).

Two things are asserted that round 1 could not assert:

* a *genuine* numeric reference (analytic Taylor-Green decay ``lambda = 4*nu*k^2``)
  and a second, independently written implementation, both free of code shared with
  ``tensorlbm`` — so the error is measured rather than manufactured;
* restart transparency as an invariant of the state machine: a segmented
  checkpoint+resume run is bit-identical to the continuous run, therefore its
  measured decay exponent carries **zero** checkpoint contribution.

Fast enough for the default suite (32x32, 200 steps, < 5 s per arm).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_job_checkpoint"))

from run_xval_r2 import (  # noqa: E402
    STATUS_WORD_MAP,
    fit_lambda,
    kinetic_energy,
    lambda_analytic,
    rel_err_pct,
    tg_numpy,
    tg_sim,
)

from tensorlbm.job_checkpoint import (  # noqa: E402
    JOB_STATUSES,
    LEGAL_TRANSITIONS,
    STATUS_DONE,
    STATUS_REACHED_MAX_STEP,
    CheckpointError,
    JobStatusFile,
    JobStatusRecord,
)

N = 32
TAU = 0.55
U0 = 0.05
STEPS = 200
FIT = (20, 200)
TOL = 15.0


def _trace(n=N, tau=TAU, u0=U0, steps=STEPS, sample=10):
    sim = tg_sim(n, tau, u0)
    ts, kes = [0], [kinetic_energy(sim)]
    for s in range(1, steps + 1):
        sim.step()
        if s % sample == 0:
            ts.append(s)
            kes.append(kinetic_energy(sim))
    return np.array(ts, float), np.array(kes, float)


def test_analytic_decay_anchor_within_owner_bar():
    t, ke = _trace()
    lam = fit_lambda(t, ke, *FIT)
    err = rel_err_pct(lam, lambda_analytic(N, TAU))
    assert abs(err) < TOL, f"lambda={lam:.6e} err={err:+.3f}%"


def test_second_implementation_agrees():
    t, ke = _trace()
    lam_torch = fit_lambda(t, ke, *FIT)
    res = tg_numpy(N, TAU, U0, STEPS)
    lam_np = fit_lambda(res["t"], res["ke"], *FIT)
    err = rel_err_pct(lam_np, lam_torch)
    assert abs(err) < TOL, f"numpy={lam_np:.6e} torch={lam_torch:.6e} err={err:+.3f}%"
    assert abs(rel_err_pct(lam_np, lambda_analytic(N, TAU))) < TOL


def test_restart_is_bit_exact_and_shifts_no_exponent(tmp_path):
    base = tg_sim(N, TAU, U0)
    trace = {0: kinetic_energy(base)}
    for s in range(1, STEPS + 1):
        base.step()
        if s % 10 == 0:
            trace[s] = kinetic_energy(base)
    t_ref = np.array(sorted(trace), float)
    lam_ref = fit_lambda(t_ref, np.array([trace[int(s)] for s in t_ref]), *FIT)

    case = tmp_path / "restart3"
    nseg, per = 3, STEPS // 3
    seg_trace: dict[int, float] = {}
    sim = tg_sim(N, TAU, U0)
    sim.enable_checkpointing(case, interval_hours=1e9, dual_backup=True)
    seg_trace[0] = kinetic_energy(sim)
    executed = 0
    for seg in range(nseg):
        if seg > 0:
            sim = tg_sim(N, TAU, U0)
            state = sim.enable_checkpointing(case, interval_hours=1e9, dual_backup=True)
            assert state.resumed and not state.finished
            if sim.step_count % 10 == 0:
                seg_trace[sim.step_count] = kinetic_energy(sim)
        n_run = per if seg < nseg - 1 else STEPS - per * (nseg - 1)
        for _ in range(n_run):
            sim.step()
            executed += 1
            if sim.step_count % 10 == 0:
                seg_trace[sim.step_count] = kinetic_energy(sim)
        sim.checkpoints.checkpoint_now(sim.step_count, {"f": sim.f})
    sim.checkpoints.finish(sim.step_count, status=STATUS_DONE, reason="test")

    assert executed == STEPS  # zero recomputation
    assert torch.equal(sim.f, base.f), "restart must reproduce the continuous populations bit-exactly"
    assert float((sim.f - base.f).abs().max()) == 0.0

    t_seg = np.array(sorted(seg_trace), float)
    lam_seg = fit_lambda(t_seg, np.array([seg_trace[int(s)] for s in t_seg]), *FIT)
    assert rel_err_pct(lam_seg, lam_ref) == 0.0, "checkpoint must contribute 0% to the decay exponent"
    assert JobStatusFile(case / "job_status.txt").read().status == STATUS_DONE


def test_status_vocabulary_map_is_declared_not_hidden():
    assert len(STATUS_WORD_MAP) == 4
    assert all(ours in JOB_STATUSES for ours in STATUS_WORD_MAP.values())
    # declared drift: MF-LBM writes long words, this repo writes short ones
    assert sum(1 for up in STATUS_WORD_MAP if up in JOB_STATUSES) == 0
    assert "failed" in JOB_STATUSES  # extra state, declared in the report


def test_terminal_states_absorb_and_veto_restart(tmp_path):
    status = JobStatusFile(tmp_path / "job_status.txt")
    status.write(JobStatusRecord(status="new", step=0, reason="created"))
    status.write(JobStatusRecord(status="continue", step=5, reason="seg"), previous=status.read())
    status.write(JobStatusRecord(status=STATUS_DONE, step=10, reason="done"), previous=status.read())
    with pytest.raises(CheckpointError):
        status.write(JobStatusRecord(status="continue", step=11, reason="illegal"), previous=status.read())
    for terminal in (STATUS_DONE, "failed", STATUS_REACHED_MAX_STEP):
        assert LEGAL_TRANSITIONS[terminal] == frozenset({terminal})


def test_reached_max_step_blocks_further_steps(tmp_path):
    sim = tg_sim(N, TAU, U0)
    state = sim.enable_checkpointing(tmp_path / "cap", interval_hours=1e9, max_steps=40, force_restart=True)
    assert not state.finished
    executed = sim.run_with_checkpointing(100)[0]
    assert executed == 40
    status = JobStatusFile(tmp_path / "cap" / "job_status.txt").read()
    assert status.status == STATUS_REACHED_MAX_STEP
    # a fresh run on the same directory must not silently continue
    again = tg_sim(N, TAU, U0)
    st2 = again.enable_checkpointing(tmp_path / "cap", interval_hours=1e9, max_steps=40)
    assert st2.finished
    assert again.run_with_checkpointing(50)[0] == 0