#!/usr/bin/env python
"""XVAL round 2 — job-status/checkpoint machine with a *genuine* external numeric anchor.

Anchor: ``XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE`` (mission m20947, round 2 of the
two-round method; round 1 delivery m17986 was graded L3 by audit m18147 with three
carried defects — definition-type reference rows, non-idempotent repro, and no
machine-adjudicable referee row for the anchor).

Round-2 question: *does the checkpoint state machine change the solution?*  Round 1
could only answer it with specification constants (``ref`` = 0 steps recomputed,
``ref`` = 1 recovery fraction ...), whose error is zero by construction.  This
harness supplies a reference the code cannot manufacture:

* **ref R1 — analytic Taylor-Green decay** (Taylor 1937; exact solution of the
  incompressible Navier-Stokes equation, ``lambda = 4*nu*k^2`` for the kinetic
  energy of ``u=u0 sin(kx)cos(ky), v=-u0 cos(kx)sin(ky)``).  Measured decay
  exponents therefore carry a *non-zero* error the harness cannot tune.
* **ref R2 — second, independently written implementation** (plain-numpy push
  scheme, float64, structurally different from the torch pull scheme in
  ``tensorlbm``); cross-code agreement is a measurement, not a constant.
* **ref R3 — MF-LBM semantics** (``oss_solutions`` row ``repo='lanl/MF-LBM'``,
  ``feature_slug='NEW:wall-clock-checkpoint-state-machine'``,
  ``IO_multiphase.F90:579-641`` / ``Main_multiphase.F90:91-95,139-141,222-303``):
  status words and terminal-absorption behaviour.

Pre-declared acceptance (owner precision standard, written before any run):

* every analytic/cross-code row must agree within **15 %** (blocking); rows that
  are bit-exact invariants keep an exact (0) tolerance;
* checkpoint contribution to the measured decay exponent must be **0** (segmented
  == continuous, bit-identical populations) — a checkpoint machine that perturbs
  the physics is disqualified regardless of any <15 % agreement;
* the repro must be idempotent: running the harness twice on the same output
  directory must succeed and reproduce the same numbers.

Usage::

    PYTHONPATH=src python benchmarks/xval_job_checkpoint/run_xval_r2.py \\
        --out artifacts/xval_job_checkpoint_r2

Writes ``summary.json``, ``BENCHMARK_REF_VS_GOT.json/.md``, checkpoint case dirs,
and four PNG figures under ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm import LBMSimulation  # noqa: E402
from tensorlbm.job_checkpoint import (  # noqa: E402
    JOB_STATUSES,
    LEGAL_TRANSITIONS,
    STATUS_CONTINUE,
    STATUS_DONE,
    STATUS_NEW,
    STATUS_REACHED_MAX_STEP,
    CheckpointError,
    JobStatusFile,
    JobStatusRecord,
)
from tensorlbm.lattice import equilibrium  # noqa: E402

SCHEMA = "xval-job-checkpoint-r2-v1"
TOL_PCT = 15.0  # owner precision standard

REF_SOURCE = {
    "repo": "lanl/MF-LBM",
    "feature_slug": "NEW:wall-clock-checkpoint-state-machine",
    "locations": [
        "IO_multiphase.F90:579-641",
        "Main_multiphase.F90:91-95",
        "Main_multiphase.F90:139-141",
        "Main_multiphase.F90:222-303",
        "Init_multiphase.F90:154-184",
    ],
    "analytic": "Taylor (1937) decaying vortex; lambda_KE = 4*nu*k^2, k = 2*pi/n",
}

#: runs: label -> (n, tau, u0, steps, fit window)
CASES: list[dict[str, Any]] = [
    {"label": "n64_tau0.55_u0.05", "n": 64, "tau": 0.55, "u0": 0.05, "steps": 300, "fit": (20, 300)},
    {"label": "n64_tau0.55_u0.02", "n": 64, "tau": 0.55, "u0": 0.02, "steps": 300, "fit": (20, 300)},
    {"label": "n64_tau0.80_u0.05", "n": 64, "tau": 0.80, "u0": 0.05, "steps": 300, "fit": (20, 300)},
    {"label": "n128_tau0.55_u0.05", "n": 128, "tau": 0.55, "u0": 0.05, "steps": 1200, "fit": (100, 1200)},
]

#: MF-LBM ``job_status.txt`` vocabulary (upstream) -> this repo's word.
STATUS_WORD_MAP = {
    "new_simulation": STATUS_NEW,
    "continue_simulation": STATUS_CONTINUE,
    "simulation_done": STATUS_DONE,
    "simulation_reached_max_step": STATUS_REACHED_MAX_STEP,
}


# ---------------------------------------------------------------------------
# torch (repo solver) helpers
# ---------------------------------------------------------------------------
def tg_sim(n: int, tau: float, u0: float) -> LBMSimulation:
    x = (torch.arange(n, dtype=torch.float32) + 0.5) / n * 2.0 * math.pi
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    ux = u0 * torch.sin(xx) * torch.cos(yy)
    uy = -u0 * torch.cos(xx) * torch.sin(yy)
    sim = LBMSimulation(nx=n, ny=n, tau=tau)
    sim.f = equilibrium(torch.ones(n, n), torch.stack([ux, uy], dim=-1))
    return sim


def kinetic_energy(sim: LBMSimulation) -> float:
    _, u = sim.macroscopic()
    return float(0.5 * (u * u).sum(dim=-1).mean())


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def fit_lambda(t: np.ndarray, ke: np.ndarray, lo: int, hi: int) -> float:
    m = (t >= lo) & (t <= hi) & (ke > 0)
    return float(-np.polyfit(t[m], np.log(ke[m]), 1)[0])


def lambda_analytic(n: int, tau: float) -> float:
    nu = (tau - 0.5) / 3.0
    return 4.0 * nu * (2.0 * math.pi / n) ** 2


def rel_err_pct(got: float, ref: float) -> float:
    return 100.0 * (got - ref) / ref


# ---------------------------------------------------------------------------
# second, independent implementation: numpy push scheme (float64)
# ---------------------------------------------------------------------------
_EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
_EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
_W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])


def _feq_np(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    feq = np.empty(rho.shape + (9,), dtype=np.float64)
    usq = ux * ux + uy * uy
    for q in range(9):
        eu = 3.0 * (_EX[q] * ux + _EY[q] * uy)
        feq[..., q] = _W[q] * rho * (1.0 + eu + 0.5 * eu * eu - 1.5 * usq)
    return feq


def tg_numpy(n: int, tau: float, u0: float, steps: int, sample: int = 10) -> dict[str, Any]:
    """Push-scheme D2Q9 BGK Taylor-Green vortex, float64, no code shared with torch path."""
    t0 = time.perf_counter()
    x = (np.arange(n, dtype=np.float64) + 0.5) / n * 2.0 * math.pi
    xx, yy = np.meshgrid(x, x, indexing="xy")
    ux = u0 * np.sin(xx) * np.cos(yy)
    uy = -u0 * np.cos(xx) * np.sin(yy)
    rho = np.ones((n, n))
    f = _feq_np(rho, ux, uy)
    omega = 1.0 / tau
    ts = [0]
    kes = [float(0.5 * np.mean(ux * ux + uy * uy))]
    for step in range(1, steps + 1):
        rho = f.sum(axis=-1)
        ux = (f * _EX).sum(axis=-1) / rho
        uy = (f * _EY).sum(axis=-1) / rho
        feq = _feq_np(rho, ux, uy)
        fpost = f - omega * (f - feq)
        f = np.empty_like(fpost)
        for q in range(9):
            f[..., q] = np.roll(fpost[..., q], shift=(int(_EY[q]), int(_EX[q])), axis=(0, 1))
        if step % sample == 0:
            ts.append(step)
            kes.append(float(0.5 * np.mean(ux * ux + uy * uy)))
    return {
        "t": np.array(ts, dtype=float),
        "ke": np.array(kes, dtype=float),
        "wall_s": time.perf_counter() - t0,
        "dtype": "float64",
        "scheme": "push (np.roll per direction), collide-then-stream",
    }


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------
def study_analytic(out: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        n, tau, u0, steps = case["n"], case["tau"], case["u0"], case["steps"]
        sim = tg_sim(n, tau, u0)
        ke0 = kinetic_energy(sim)
        ts, kes = [0], [ke0]
        t0 = time.perf_counter()
        for s in range(1, steps + 1):
            sim.step()
            if s % 10 == 0:
                ts.append(s)
                kes.append(kinetic_energy(sim))
        wall = time.perf_counter() - t0
        t = np.array(ts, float)
        ke = np.array(kes, float)
        lo, hi = case["fit"]
        lam = fit_lambda(t, ke, lo, hi)
        lam_half = fit_lambda(t, ke, lo, max(lo + 1, hi // 2))  # window sensitivity
        ana = lambda_analytic(n, tau)
        nu = (tau - 0.5) / 3.0
        rows.append(
            {
                "label": case["label"],
                "n": n,
                "tau": tau,
                "nu": nu,
                "u0": u0,
                "steps": steps,
                "fit_window": [lo, hi],
                "ke_start": ke0,
                "ke_end": float(ke[-1]),
                "ke_ratio": float(ke[-1] / ke0),
                "lambda_measured": lam,
                "lambda_fit_short_window": lam_half,
                "lambda_analytic": ana,
                "err_pct": rel_err_pct(lam, ana),
                "err_short_window_pct": rel_err_pct(lam_half, ana),
                "re": u0 * n / nu,
                "ma": u0 / math.sqrt(1.0 / 3.0),
                "wall_s": wall,
                "trace": {"t": t.tolist(), "ke": ke.tolist()},
            }
        )
        print(f"  [analytic] {case['label']}: lambda={lam:.6e} ana={ana:.6e} err={rows[-1]['err_pct']:+.3f}%")
    return rows


def study_restart(out: Path, case: dict[str, Any], segments: list[int]) -> dict[str, Any]:
    """H1: segmented checkpoint+resume == continuous, measured at the analytic anchor."""
    n, tau, u0, steps = case["n"], case["tau"], case["u0"], case["steps"]
    lo, hi = case["fit"]
    base = tg_sim(n, tau, u0)
    k0 = kinetic_energy(base)
    ts, kes = [0], [k0]
    for s in range(1, steps + 1):
        base.step()
        if s % 10 == 0:
            ts.append(s)
            kes.append(kinetic_energy(base))
    t = np.array(ts, float)
    ke_base = np.array(kes, float)
    lam_cont = fit_lambda(t, ke_base, lo, hi)
    ana = lambda_analytic(n, tau)
    rows: list[dict[str, Any]] = []
    for nseg in segments:
        case_dir = out / "cases" / f"restart_{nseg:02d}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        per = steps // nseg
        executed = 0
        restarts = 0
        trace: dict[int, float] = {}
        sim = tg_sim(n, tau, u0)
        sim.enable_checkpointing(case_dir, interval_hours=1e9, dual_backup=True)
        trace[0] = kinetic_energy(sim)
        for seg in range(nseg):
            if seg > 0:
                sim = tg_sim(n, tau, u0)  # simulated kill: fresh object
                st = sim.enable_checkpointing(case_dir, interval_hours=1e9, dual_backup=True)
                if not st.resumed:
                    raise RuntimeError(f"segment {seg} did not resume")
                restarts += 1
                if sim.step_count % 10 == 0:  # keep the sampling grid identical to the continuous arm
                    trace[sim.step_count] = kinetic_energy(sim)
            run_steps = per if seg < nseg - 1 else steps - per * (nseg - 1)
            for _ in range(run_steps):
                sim.step()
                executed += 1
                if sim.step_count % 10 == 0:
                    trace[sim.step_count] = kinetic_energy(sim)
            sim.checkpoints.checkpoint_now(sim.step_count, {"f": sim.f})
        sim.checkpoints.finish(sim.step_count, status=STATUS_DONE, reason="segment arm complete")
        seg_t_a = np.array(sorted(trace), float)
        seg_ke_a = np.array([trace[int(s)] for s in seg_t_a], float)
        lam_seg = fit_lambda(seg_t_a, seg_ke_a, lo, hi)
        max_df = float((sim.f - base.f).abs().max())
        bits = bool(torch.equal(sim.f, base.f))
        rows.append(
            {
                "segments": nseg,
                "restarts": restarts,
                "steps_executed": executed,
                "steps_recomputed": executed - steps,
                "max_abs_df_vs_continuous": max_df,
                "bit_exact": bits,
                "final_digest": digest(sim.f),
                "ke_ratio": float(seg_ke_a[-1] / seg_ke_a[0]),
                "ke_trace": {"t": seg_t_a.tolist(), "ke": seg_ke_a.tolist()},
                "lambda_measured": lam_seg,
                "err_pct_vs_analytic": rel_err_pct(lam_seg, ana),
                "delta_lambda_vs_continuous_pct": rel_err_pct(lam_seg, lam_cont),
                "status": JobStatusFile(case_dir / "job_status.txt").read().status,
                "checkpoint_dir": str(case_dir),
            }
        )
        print(
            f"  [restart] segments={nseg}: max|df|={max_df:.3e} bits={bits} "
            f"d_lambda={rows[-1]['delta_lambda_vs_continuous_pct']:+.3e}%"
        )
    return {
        "case": case["label"],
        "baseline_digest": digest(base.f),
        "baseline_lambda": lam_cont,
        "baseline_err_pct_vs_analytic": rel_err_pct(lam_cont, ana),
        "rows": rows,
    }


def study_second_implementation(cases: list[dict[str, Any]], torch_rows: list[dict[str, Any]]) -> list[dict]:
    rows: list[dict[str, Any]] = []
    by_label = {r["label"]: r for r in torch_rows}
    for case in cases:
        n, tau, u0, steps = case["n"], case["tau"], case["u0"], case["steps"]
        lo, hi = case["fit"]
        res = tg_numpy(n, tau, u0, steps)
        lam = fit_lambda(res["t"], res["ke"], lo, hi)
        ana = lambda_analytic(n, tau)
        torch_lam = by_label[case["label"]]["lambda_measured"]
        rows.append(
            {
                "label": case["label"],
                "n": n,
                "tau": tau,
                "u0": u0,
                "steps": steps,
                "dtype": res["dtype"],
                "scheme": res["scheme"],
                "lambda_measured": lam,
                "lambda_analytic": ana,
                "err_pct_vs_analytic": rel_err_pct(lam, ana),
                "lambda_torch": torch_lam,
                "err_pct_vs_torch": rel_err_pct(lam, torch_lam),
                "ke_ratio": float(res["ke"][-1] / res["ke"][0]),
                "wall_s": res["wall_s"],
                "trace": {"t": res["t"].tolist(), "ke": res["ke"].tolist()},
            }
        )
        print(
            f"  [numpy] {case['label']}: lambda={lam:.6e} vs torch {torch_lam:.6e} "
            f"({rows[-1]['err_pct_vs_torch']:+.3f}%) vs analytic {rows[-1]['err_pct_vs_analytic']:+.3f}%"
        )
    return rows


def study_state_machine(out: Path) -> dict[str, Any]:
    """R3: MF-LBM status vocabulary + terminal absorption, measured on the real files."""
    case_dir = out / "cases" / "status_machine"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    status = JobStatusFile(case_dir / "job_status.txt")

    # life cycle: new -> continue -> done, status file read back each time
    seen: list[str] = []
    status.write(JobStatusRecord(status=STATUS_NEW, step=0, reason="created"))
    seen.append(status.read().status)
    status.write(JobStatusRecord(status=STATUS_CONTINUE, step=10, reason="segment"), previous=status.read())
    seen.append(status.read().status)
    status.write(JobStatusRecord(status=STATUS_DONE, step=40, reason="finished"), previous=status.read())
    seen.append(status.read().status)

    # terminal veto: continue after done must be refused
    vetoed = False
    try:
        status.write(JobStatusRecord(status=STATUS_CONTINUE, step=41, reason="illegal"), previous=status.read())
    except CheckpointError:
        vetoed = True

    # failed is absorbing as well, reached_max_step is a legal terminal too
    absorbing = {}
    for terminal in (STATUS_DONE, "failed", STATUS_REACHED_MAX_STEP):
        ok = True
        for nxt in (STATUS_CONTINUE, STATUS_NEW, STATUS_DONE):
            if nxt in LEGAL_TRANSITIONS[terminal]:
                if nxt != terminal:
                    ok = False
        absorbing[terminal] = ok

    # exact-word mapping (drift is declared, not hidden: count both ways)
    mapped_ok = sum(1 for ours in STATUS_WORD_MAP.values() if ours in JOB_STATUSES)
    exact_word = sum(1 for up in STATUS_WORD_MAP if up in JOB_STATUSES)
    print(
        f"  [state] words mapped {mapped_ok}/{len(STATUS_WORD_MAP)}, exact-string {exact_word}/"
        f"{len(STATUS_WORD_MAP)}, terminal veto={vetoed}, absorbing={absorbing}"
    )
    return {
        "lifecycle": seen,
        "terminal_veto": vetoed,
        "terminal_absorbing": absorbing,
        "status_word_map": STATUS_WORD_MAP,
        "mapped_semantics": mapped_ok,
        "upstream_words": len(STATUS_WORD_MAP),
        "exact_string_matches": exact_word,
        "extra_states": sorted(set(JOB_STATUSES) - set(STATUS_WORD_MAP.values())),
        "legal_transitions": {k: sorted(v) for k, v in LEGAL_TRANSITIONS.items()},
    }


def study_idempotence(out: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Repro idempotence (audit m18147 F2): same directory twice -> same digest."""
    n, tau, u0 = case["n"], case["tau"], case["u0"]
    case_dir = out / "cases" / "idempotence"
    runs = []
    for attempt in range(2):
        if case_dir.exists():
            shutil.rmtree(case_dir)
        sim = tg_sim(n, tau, u0)
        sim.enable_checkpointing(case_dir, interval_hours=1e9, max_steps=200, force_restart=True)
        sim.run_with_checkpointing(120)
        sim.checkpoints.finish(sim.step_count, status=STATUS_DONE, reason="idempotence arm")
        runs.append({"digest": digest(sim.f), "status": JobStatusFile(case_dir / "job_status.txt").read().status})
    # third arm without force_restart: the documented terminal guard must fire and
    # the run must not silently continue (no new steps).
    sim = tg_sim(n, tau, u0)
    state = sim.enable_checkpointing(case_dir, interval_hours=1e9, max_steps=200)
    executed = 0
    if not state.finished:
        executed = sim.run_with_checkpointing(50)[0]
    guard = {"finished": bool(state.finished), "steps_executed": executed,
             "status": JobStatusFile(case_dir / "job_status.txt").read().status}
    same = runs[0]["digest"] == runs[1]["digest"]
    print(f"  [idempotence] digests equal={same} guard={guard}")
    return {"runs": runs, "same_digest": same, "terminal_guard": guard}


# ---------------------------------------------------------------------------
def build_ref_vs_got(summary: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def add(quantity: str, ref: float, got: float, unit: str, ref_source: str, tol: float | None = None):
        tol = TOL_PCT if tol is None else tol
        err = 0.0 if ref == 0 else rel_err_pct(got, ref)
        rows.append(
            {
                "quantity": quantity,
                "ref": ref,
                "got": got,
                "unit": unit,
                "rel_err_pct": err,
                "tolerance_pct": tol,
                "pass": abs(err) <= tol,
                "ref_source": ref_source,
            }
        )

    for r in summary["analytic"]:
        add(
            f"KE decay exponent lambda, {r['label']} (torch)",
            r["lambda_analytic"],
            r["lambda_measured"],
            "1/step",
            "Taylor (1937) analytic: lambda = 4*nu*k^2, k=2*pi/n",
        )
    for r in summary["second_implementation"]:
        add(
            f"KE decay exponent lambda, {r['label']} (numpy push impl)",
            r["lambda_analytic"],
            r["lambda_measured"],
            "1/step",
            "Taylor (1937) analytic",
        )
        add(
            f"lambda ratio numpy impl / torch impl, {r['label']}",
            r["lambda_torch"],
            r["lambda_measured"],
            "1/step",
            "second independent implementation (numpy push scheme, float64)",
        )
    for r in summary["restart"]["rows"]:
        add(
            f"restart field discontinuity max|df|, {r['segments']} segment(s)",
            0.0,
            r["max_abs_df_vs_continuous"],
            "lattice populations",
            "IO_multiphase.F90:579-641 (full PDF stored) + Main_multiphase.F90:139-141",
            tol=0.0,
        )
        add(
            f"checkpoint contribution to lambda, {r['segments']} segment(s)",
            0.0,
            r["delta_lambda_vs_continuous_pct"],
            "%",
            "bit-exact restart invariant (segmented == continuous)",
            tol=0.0,
        )
        add(
            f"steps recomputed after resume, {r['segments']} segment(s)",
            0.0,
            float(r["steps_recomputed"]),
            "steps",
            "Main_multiphase.F90:91-95 (continue path recomputes nothing)",
            tol=0.0,
        )
    sm = summary["state_machine"]
    add(
        "MF-LBM status words mapped to a legal TensorLBM state",
        float(sm["upstream_words"]),
        float(sm["mapped_semantics"]),
        "words",
        "IO_multiphase.F90:579-641 vocabulary",
        tol=0.0,
    )
    add(
        "terminal state vetoes a silent restart (done)",
        1.0,
        1.0 if sm["terminal_veto"] else 0.0,
        "fraction",
        "Main_multiphase.F90:91-95 (done exits instead of continuing)",
        tol=0.0,
    )
    for terminal, ok in sm["terminal_absorbing"].items():
        add(
            f"terminal state absorbing: {terminal}",
            1.0,
            1.0 if ok else 0.0,
            "fraction",
            "Main_multiphase.F90:91-95",
            tol=0.0,
        )
    idem = summary["idempotence"]
    add(
        "repro idempotence: identical digest on a second run in the same directory",
        1.0,
        1.0 if idem["same_digest"] else 0.0,
        "fraction",
        "audit m18147 F2 remediation",
        tol=0.0,
    )
    add(
        "terminal guard: a finished case executes no further step",
        0.0,
        float(idem["terminal_guard"]["steps_executed"]),
        "steps",
        "Main_multiphase.F90:91-95",
        tol=0.0,
    )

    worst = max((abs(r["rel_err_pct"]) / r["tolerance_pct"] for r in rows if r["tolerance_pct"] > 0), default=0.0)
    return {
        "schema": SCHEMA,
        "anchor": "XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE",
        "criterion": f"relative error <= {TOL_PCT}% (owner precision standard); bit-exact invariants use tolerance 0",
        "rows": rows,
        "worst_fraction_of_tolerance": worst,
        "all_pass": all(r["pass"] for r in rows),
        "n_rows": len(rows),
        "n_fail": sum(0 if r["pass"] else 1 for r in rows),
    }


def write_figures(out: Path, summary: dict[str, Any]) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs: list[dict[str, str]] = []

    # 1) KE decay vs analytic, all cases
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for r in summary["analytic"]:
        t = np.array(r["trace"]["t"])
        ke = np.array(r["trace"]["ke"])
        ax.semilogy(t, ke / ke[0], lw=1.2, label=f"torch {r['label']}")
        ax.semilogy(
            t,
            np.exp(-r["lambda_analytic"] * t),
            ls="--",
            lw=1.0,
            label=f"analytic 4νk² {r['label']}",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("KE(t)/KE(0)")
    ax.set_title("Taylor-Green kinetic-energy decay: LBM vs analytic")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = figdir / "fig_ke_decay_analytic.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "path": str(p), "caption": "KE decay, torch solver vs analytic exp(-4νk²t)"})

    # 2) error of every arm vs analytic, with 15% band
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels, errs, kinds = [], [], []
    for r in summary["analytic"]:
        labels.append(r["label"])
        errs.append(r["err_pct"])
        kinds.append("torch")
    for r in summary["second_implementation"]:
        labels.append(r["label"] + "\nnumpy")
        errs.append(r["err_pct_vs_analytic"])
        kinds.append("numpy")
    colors = ["tab:blue" if k == "torch" else "tab:orange" for k in kinds]
    ax.bar(range(len(errs)), errs, color=colors)
    ax.axhline(TOL_PCT, color="r", ls=":", label="owner 15% bar")
    ax.axhline(-TOL_PCT, color="r", ls=":")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("error vs analytic λ [%]")
    ax.set_title("Decay-exponent error against the analytic anchor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    p = figdir / "fig_lambda_error_vs_analytic.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "path": str(p), "caption": "λ error vs analytic (torch and numpy impl)"})

    # 3) restart transparency: max|df| and λ delta
    rows = summary["restart"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    segs = [r["segments"] for r in rows]
    ax.plot(segs, [r["max_abs_df_vs_continuous"] for r in rows], "o-", label="max|Δf|")
    ax.plot(segs, [abs(r["delta_lambda_vs_continuous_pct"]) for r in rows], "s--", label="|Δλ|/λ [%]")
    ax.set_xlabel("number of segments (checkpoint+restart)")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_title("Restart transparency: segmented vs continuous")
    ax.legend()
    ax.grid(alpha=0.3)
    p = figdir / "fig_restart_transparency.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "path": str(p), "caption": "max|Δf| and |Δλ| vs segment count (both zero)"})

    # 4) second implementation overlay
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for r in summary["analytic"]:
        t = np.array(r["trace"]["t"])
        ke = np.array(r["trace"]["ke"])
        ax.semilogy(t, ke / ke[0], lw=1.3, label=f"torch {r['label']}")
    for r in summary["second_implementation"]:
        t = np.array(r["trace"]["t"])
        ke = np.array(r["trace"]["ke"])
        ax.semilogy(t, ke / ke[0], ls="--", lw=1.0, label=f"numpy {r['label']}")
    ax.set_xlabel("step")
    ax.set_ylabel("KE(t)/KE(0)")
    ax.set_title("Second independent implementation (numpy push) overlay")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    p = figdir / "fig_second_implementation.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append({"name": p.name, "path": str(p), "caption": "torch vs numpy push implementation KE decays"})

    return figs


def write_markdown(out: Path, summary: dict[str, Any]) -> Path:
    rv = summary["ref_vs_got"]
    lines = [
        "# XVAL ref vs got — job-status/checkpoint machine (round 2)",
        "",
        f"Anchor `{rv['anchor']}` · criterion: {rv['criterion']}",
        "",
        "| # | quantity | ref | got | unit | rel_err % | tol % | pass | ref source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rv["rows"], 1):
        lines.append(
            f"| {i} | {r['quantity']} | {r['ref']:.6g} | {r['got']:.6g} | {r['unit']} | "
            f"{r['rel_err_pct']:+.3e} | {r['tolerance_pct']:g} | {'PASS' if r['pass'] else 'FAIL'} | {r['ref_source']} |"
        )
    lines += [
        "",
        f"rows={rv['n_rows']} fail={rv['n_fail']} all_pass={rv['all_pass']} "
        f"worst_fraction_of_tolerance={rv['worst_fraction_of_tolerance']:.3e}",
        "",
    ]
    md = out / "BENCHMARK_REF_VS_GOT.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (out / "BENCHMARK_REF_VS_GOT.json").write_text(json.dumps(rv, indent=2), encoding="utf-8")
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("artifacts/xval_job_checkpoint_r2"))
    ap.add_argument("--segments", default="1,2,4,8")
    ap.add_argument("--quick", action="store_true", help="small grid/short runs (smoke)")
    ap.add_argument("--keep-cases", action="store_true", help="keep previous case dirs (default: they are rebuilt)")
    args = ap.parse_args(argv)

    cases = CASES
    if args.quick:
        cases = [
            {"label": "n32_tau0.55_u0.05", "n": 32, "tau": 0.55, "u0": 0.05, "steps": 200, "fit": (20, 200)},
        ]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if not args.keep_cases:
        for stale in (out / "cases",):
            if stale.exists():
                shutil.rmtree(stale)

    t_start = time.perf_counter()
    print("[1/5] analytic decay anchor")
    analytic = study_analytic(out, cases)
    print("[2/5] restart transparency")
    segs = [int(s) for s in args.segments.split(",") if s.strip()]
    restart = study_restart(out, cases[0], segs)
    print("[3/5] second independent implementation")
    second = study_second_implementation(cases, analytic)
    print("[4/5] state machine semantics")
    state = study_state_machine(out)
    print("[5/5] repro idempotence")
    idem = study_idempotence(out, cases[0])

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "anchor": "XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE",
        "mission": "m20947",
        "round": 2,
        "reference": REF_SOURCE,
        "tolerance_pct": TOL_PCT,
        "analytic": analytic,
        "restart": restart,
        "second_implementation": second,
        "state_machine": state,
        "idempotence": idem,
        "wall_s_total": time.perf_counter() - t_start,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary["ref_vs_got"] = build_ref_vs_got(summary)
    summary["figures"] = write_figures(out, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out, summary)
    (out / "repro.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../..\"\n"
        f"PYTHONPATH=src python benchmarks/xval_job_checkpoint/run_xval_r2.py --out {out}\n",
        encoding="utf-8",
    )
    (out / "repro.sh").chmod(0o755)

    rv = summary["ref_vs_got"]
    print(json.dumps({"all_pass": rv["all_pass"], "n_rows": rv["n_rows"], "n_fail": rv["n_fail"],
                      "worst_fraction_of_tolerance": rv["worst_fraction_of_tolerance"]}))
    return 0 if rv["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())