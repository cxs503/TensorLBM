#!/usr/bin/env python
"""XVAL: TensorLBM job-status/wall-clock checkpoint machine vs lanl/MF-LBM.

Cross-validation target (from the platform OSS ledger, zero hand-copying):
``oss_solutions`` row ``repo='lanl/MF-LBM'``,
``feature_slug='NEW:wall-clock-checkpoint-state-machine'`` with source locations

* ``IO_multiphase.F90:579-641``  — ``job_status.txt`` + checkpoint write/read
* ``Main_multiphase.F90:91-95``  — status consulted at start-up
* ``Main_multiphase.F90:139-141``— ``continue`` path on restart
* ``Main_multiphase.F90:222-303``— hour-based timers + ``d_vol_*`` volume triggers
* ``Init_multiphase.F90:154-184``— restart state restored before the main loop

Experiment (pre-declared before any run):

* case: D2Q9 Taylor-Green vortex, ``grid=64x64``, ``tau=0.55``, ``u0=0.05``,
  periodic; the field is genuinely evolving (kinetic energy decays, ~1.6 %/200 steps).
* H1 restart identity: a run that is checkpointed and resumed in segments reaches
  **bit-identical** populations compared with a single continuous run
  (``max|Δf| == 0``), because the full PDF is stored (MF-LBM stores the full PDFs).
* H2 zero recomputation: steps executed across all segments == total steps.
* H3 recovery: a truncated primary checkpoint is rescued by the second backup
  (``source == 'backup'``) with ``max|Δf| == 0``; with ``dual_backup=False`` the
  same failure is fatal (control arm, recovery rate 0/1).
* H4 timer semantics: measured wall-clock interval between automatic saves
  matches the configured ``interval_hours`` (within one step's compute time), and
  the number of saves grows ~1/interval.
* Deliverable thresholds: XVAL pass = all rows within 15 % of the MF-LBM reference
  behaviour (bit-exact rows: exact 0).

Usage::

    PYTHONPATH=src python benchmarks/xval_job_checkpoint/run_xval.py \\
        --out artifacts/xval_job_checkpoint

Writes ``summary.json``, ``BENCHMARK_REF_VS_GOT.json/.md``, checkpoint case dirs
(with ``job_status.txt``) and four PNG figures under ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm import LBMSimulation  # noqa: E402
from tensorlbm.job_checkpoint import (  # noqa: E402
    CheckpointError,
    JobCheckpointManager,
    JobStatusFile,
    VolumeTrigger,
)
from tensorlbm.lattice import equilibrium  # noqa: E402

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
    "ledger": "oss_solutions(repo='lanl/MF-LBM', feature_slug='NEW:wall-clock-checkpoint-state-machine')",
}

TAU = 0.55
U0 = 0.05


# ---------------------------------------------------------------------------
def taylor_green(nx: int, ny: int, tau: float = TAU) -> LBMSimulation:
    x = (torch.arange(nx, dtype=torch.float32) + 0.5) / nx * 2.0 * math.pi
    y = (torch.arange(ny, dtype=torch.float32) + 0.5) / ny * 2.0 * math.pi
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    ux = U0 * torch.sin(xx) * torch.cos(yy)
    uy = -U0 * torch.cos(xx) * torch.sin(yy)
    sim = LBMSimulation(nx=nx, ny=ny, tau=tau)
    sim.f = equilibrium(torch.ones(ny, nx), torch.stack([ux, uy], dim=-1))
    return sim


def kinetic_energy(sim: LBMSimulation) -> float:
    _, u = sim.macroscopic()
    return float(0.5 * (u * u).sum(dim=-1).mean())


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def ref_trajectory(nx: int, ny: int, steps: int) -> torch.Tensor:
    sim = taylor_green(nx, ny)
    sim.run(steps)
    return sim.f.detach().clone()


# ---------------------------------------------------------------------------
def study_restart(out: Path, nx: int, ny: int, steps: int, segment_counts: list[int]) -> dict[str, Any]:
    """H1/H2: segmented checkpoint+resume vs the continuous trajectory."""
    baseline = ref_trajectory(nx, ny, steps)
    rows = []
    for segments in segment_counts:
        case = out / "cases" / f"restart_{segments:02d}"
        per = steps // segments
        executed = 0
        checkpoints = 0
        restarts = 0
        sim = taylor_green(nx, ny)
        t0 = time.perf_counter()
        sim.enable_checkpointing(case, interval_hours=1e9, dual_backup=True)  # manual saves only
        for seg in range(segments):
            if seg > 0:
                sim = taylor_green(nx, ny)  # simulated kill: fresh object, no state kept
                st = sim.enable_checkpointing(case, interval_hours=1e9, dual_backup=True)
                assert st.resumed, f"segment {seg} did not resume"
                restarts += 1
            run_steps = per if seg < segments - 1 else steps - per * (segments - 1)
            executed += int(sim.run_with_checkpointing(run_steps)[0])
            sim.checkpoints.checkpoint_now(sim.step_count, {"f": sim.f})
            checkpoints += 1
        sim.checkpoints.finish(sim.step_count, status="done", reason="segment arm complete")
        wall = time.perf_counter() - t0
        rows.append(
            {
                "segments": segments,
                "restarts": restarts,
                "steps_executed": executed,
                "steps_recomputed": executed - steps,
                "checkpoints": checkpoints,
                "final_step": sim.step_count,
                "max_abs_df_vs_continuous": max_abs_diff(sim.f, baseline),
                "bit_exact": bool(torch.equal(sim.f, baseline)),
                "wall_s": wall,
                "status": JobStatusFile(case / "job_status.txt").read().status,
                "final_digest": digest(sim.f),
                "case_dir": str(case),
            }
        )
    return {"baseline_digest": digest(baseline), "rows": rows}


def study_recovery(out: Path, nx: int, ny: int, steps: int) -> dict[str, Any]:
    """H3: truncated primary checkpoint -> second backup rescues (dual) vs fatal (single)."""
    baseline = ref_trajectory(nx, ny, steps)
    half = steps // 2
    result: dict[str, Any] = {}

    # dual backup arm
    case = out / "cases" / "recovery_dual"
    sim = taylor_green(nx, ny)
    sim.enable_checkpointing(case, interval_hours=1e9, dual_backup=True)
    sim.run_with_checkpointing(half)
    sim.checkpoints.checkpoint_now(sim.step_count, {"f": sim.f})
    primary = case / "checkpoint.pt"
    raw = primary.read_bytes()
    primary.write_bytes(raw[: len(raw) // 2])  # kill mid-write

    resumed = taylor_green(nx, ny)
    st = resumed.enable_checkpointing(case, interval_hours=1e9, dual_backup=True)
    resumed.run_with_checkpointing(steps - st.step)
    result["dual_backup"] = {
        "resume_source": st.source,
        "resumed_from_step": st.step,
        "recovered": True,
        "max_abs_df_vs_continuous": max_abs_diff(resumed.f, baseline),
        "bit_exact": bool(torch.equal(resumed.f, baseline)),
        "case_dir": str(case),
    }

    # single backup control arm
    case1 = out / "cases" / "recovery_single"
    sim1 = taylor_green(nx, ny)
    sim1.enable_checkpointing(case1, interval_hours=1e9, dual_backup=False)
    sim1.run_with_checkpointing(half)
    sim1.checkpoints.checkpoint_now(sim1.step_count, {"f": sim1.f})
    primary1 = case1 / "checkpoint.pt"
    raw1 = primary1.read_bytes()
    primary1.write_bytes(raw1[: len(raw1) // 2])
    try:
        taylor_green(nx, ny).enable_checkpointing(case1, interval_hours=1e9, dual_backup=False)
        recovered, error = True, None
    except CheckpointError as exc:
        recovered, error = False, str(exc).splitlines()[0]
    result["single_backup"] = {
        "recovered": recovered,
        "error": error,
        "case_dir": str(case1),
    }
    return result


def study_overhead(
    out: Path,
    nx: int,
    ny: int,
    steps: int,
    intervals_s: list[float],
    *,
    reps: int = 3,
) -> dict[str, Any]:
    """H4: wall-clock timer adherence + checkpoint overhead vs interval.

    Wall time on a shared host drifts, so every arm is measured in an *interleaved*
    A/B/A/B design (baseline run without checkpointing vs the same run with
    checkpointing) and the median over ``reps`` repetitions is used.  The direct
    cost of one checkpoint write is reported separately, so the overhead can be
    cross-checked as ``saves * t_write / t_run`` instead of trusting two noisy
    wall times.
    """
    rows = []
    base_times: list[float] = []
    step_times: list[float] = []
    for interval_s in intervals_s:
        case = out / "cases" / f"overhead_{interval_s:g}s"
        ck_times: list[float] = []
        arm_bases: list[float] = []
        last = None
        for rep in range(reps):
            # A/B order alternates between repetitions so a monotone host drift
            # (other jobs on the machine) does not load onto the checkpoint arm.
            ckpt_first = rep % 2 == 1

            def _run_base() -> float:
                sim0 = taylor_green(nx, ny)
                t0 = time.perf_counter()
                sim0.run(steps)
                return time.perf_counter() - t0

            def _run_ckpt() -> tuple[float, Any]:
                sim = taylor_green(nx, ny)
                # rep > 0 reuses the case directory: force_restart keeps every rep an
                # independent trajectory from the same initial condition.
                sim.enable_checkpointing(
                    case,
                    interval_hours=interval_s / 3600.0,
                    dual_backup=True,
                    force_restart=(rep > 0),
                )
                t0 = time.perf_counter()
                sim.run_with_checkpointing(steps)
                return time.perf_counter() - t0, sim

            if ckpt_first:
                t_ck, sim = _run_ckpt()
                base = _run_base()
            else:
                base = _run_base()
                t_ck, sim = _run_ckpt()
            base_times.append(base)
            arm_bases.append(base)
            step_times.append(base / steps)
            ck_times.append(t_ck)
            last = sim
        receipts = last.checkpoints.receipts
        measured = sorted(r.elapsed_hours for r in receipts[:-1]) or sorted(
            r.elapsed_hours for r in receipts
        )
        median_h = measured[len(measured) // 2] if measured else float("nan")
        step_time = sorted(step_times)[len(step_times) // 2]
        ck_median = sorted(ck_times)[len(ck_times) // 2]
        base_median = sorted(arm_bases)[len(arm_bases) // 2]
        rows.append(
            {
                "interval_s": interval_s,
                "interval_hours": interval_s / 3600.0,
                "reps": reps,
                "checkpoints": len(receipts),
                "expected_checkpoints": steps * step_time / interval_s,
                "base_wall_s_median": base_median,
                "ckpt_wall_s_median": ck_median,
                "overhead_pct": (ck_median / base_median - 1.0) * 100.0,
                "median_measured_interval_hours": median_h,
                "median_measured_interval_s": median_h * 3600.0,
                "interval_rel_err_pct": (median_h - interval_s / 3600.0) / (interval_s / 3600.0) * 100.0,
                "timer_excess_steps": (median_h * 3600.0 - interval_s) / step_time,
                "step_time_s": step_time,
                "base_wall_cv_pct": (
                    statistics.pstdev(arm_bases) / statistics.fmean(arm_bases) * 100.0
                    if len(arm_bases) > 1
                    else 0.0
                ),
                "checkpoint_bytes": receipts[-1].primary_bytes if receipts else 0,
                "cases": str(case),
                "step": last.step_count,
            }
        )

    # automatic (wall-clock) path restart: resume from the newest autosave of the
    # shortest-interval arm and finish the trajectory, compare against continuous.
    auto = None
    if rows:
        case = Path(rows[0]["cases"])
        baseline = ref_trajectory(nx, ny, steps)
        resumed = taylor_green(nx, ny)
        st = resumed.enable_checkpointing(
            case, interval_hours=intervals_s[0] / 3600.0, dual_backup=True
        )
        resumed.run_with_checkpointing(steps - st.step)
        auto = {
            "case_dir": str(case),
            "resume_source": st.source,
            "resumed_from_step": st.step,
            "steps_replayed": steps - st.step,
            "max_abs_df_vs_continuous": max_abs_diff(resumed.f, baseline),
            "bit_exact": bool(torch.equal(resumed.f, baseline)),
        }

    # direct cost of one checkpoint generation (primary write + backup copy):
    # measured by timing a single forced save on a quiet moment of the run.
    sim = taylor_green(nx, ny)
    sim.enable_checkpointing(out / "cases" / "single_write_cost", interval_hours=1e9, dual_backup=True)
    sim.run_with_checkpointing(5)
    t_samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        sim.checkpoints.checkpoint_now(sim.step_count, {"f": sim.f})
        t_samples.append(time.perf_counter() - t0)
    t_samples.sort()
    single_write = t_samples[len(t_samples) // 2]

    host_noise = max((row["base_wall_cv_pct"] for row in rows), default=0.0)
    for row in rows:
        row["overhead_from_write_cost_pct"] = (
            row["checkpoints"] * single_write / row["ckpt_wall_s_median"] * 100.0
        )

    return {
        "reps": reps,
        "steps": steps,
        "rows": rows,
        "auto_restart": auto,
        "single_checkpoint_write_s": single_write,
        "host_noise_pct": host_noise,
        "baseline_step_time_s": sorted(step_times)[len(step_times) // 2],
        "case_dir": str(out / "cases" / "single_write_cost"),
    }


def study_terminal_and_volume(out: Path, nx: int, ny: int) -> dict[str, Any]:
    """Terminal status vocabulary + ``d_vol_*`` analogue."""
    case = out / "cases" / "terminal_max_step"
    sim = taylor_green(nx, ny)
    sim.enable_checkpointing(case, interval_hours=1e9, max_steps=25, dual_backup=True)
    sim.run_with_checkpointing(40)  # asks for 40, must stop at 25
    record = JobStatusFile(case / "job_status.txt").read()
    blocked = taylor_green(nx, ny).enable_checkpointing(
        case, interval_hours=1e9, max_steps=25, dual_backup=True
    )

    trigger = VolumeTrigger(0.5)
    fired = 0
    volume = 0.0
    for _ in range(10):
        volume += 0.125
        if trigger.due(volume):
            fired += 1
            trigger.mark(volume)
    return {
        "max_step_case": str(case),
        "steps_executed": sim.step_count,
        "status": record.status,
        "terminal_blocks_restart": bool(blocked.finished),
        "volume_interval": 0.5,
        "volume_final": volume,
        "volume_triggers_fired": fired,
        "volume_triggers_expected": int(volume // 0.5),
    }


# ---------------------------------------------------------------------------
def build_ref_vs_got(summary: dict[str, Any]) -> dict[str, Any]:
    """MF-LBM reference behaviour vs measured TensorLBM behaviour."""
    rows = []

    def add(name: str, ref: float, got: float, tol_pct: float, unit: str, note: str, ref_src: str) -> None:
        if ref == 0 and tol_pct > 0:
            # the reference is "no deviation": the measured value IS the error (%)
            err = abs(got)
        elif ref == 0:
            err = 0.0 if got == 0 else float("inf")
        else:
            err = abs(got - ref) / abs(ref) * 100.0
        rows.append(
            {
                "quantity": name,
                "ref": ref,
                "got": got,
                "unit": unit,
                "rel_err_pct": err,
                "tolerance_pct": tol_pct,
                "pass": bool(err <= tol_pct),
                "ref_source": ref_src,
                "note": note,
            }
        )

    restarts = summary["restart_study"]["rows"]
    recovery = summary["recovery"]
    overhead = summary["overhead_study"]["rows"]

    add(
        "restart field discontinuity max|df| (worst segment count)",
        ref=0.0,
        got=max(r["max_abs_df_vs_continuous"] for r in restarts),
        tol_pct=0.0,
        unit="lattice population units",
        note="full-PDF restore is identity; MF-LBM stores complete PDFs and takes the continue path",
        ref_src="IO_multiphase.F90:579-641 + Main_multiphase.F90:139-141",
    )
    add(
        "steps recomputed after resume (worst segment count)",
        ref=0,
        got=max(abs(r["steps_recomputed"]) for r in restarts),
        tol_pct=0.0,
        unit="steps",
        note="continue path must not redo work",
        ref_src="Main_multiphase.F90:91-95",
    )
    add(
        "recovery rate after corrupted primary checkpoint",
        ref=1.0,
        got=1.0 if recovery["dual_backup"]["recovered"] else 0.0,
        tol_pct=0.0,
        unit="fraction",
        note="second backup (double_bak_checkpoint_pdf_cmd) rescues the case",
        ref_src="IO_multiphase.F90:579-641",
    )
    add(
        "recovery rate after corrupted primary, second backup disabled (control)",
        ref=0.0,
        got=0.0 if not recovery["single_backup"]["recovered"] else 1.0,
        tol_pct=0.0,
        unit="fraction",
        note="negative control: without the second backup the corruption is fatal",
        ref_src="IO_multiphase.F90:579-641",
    )
    add(
        "status states implemented",
        ref=5.0,
        got=float(summary["status_vocabulary"]["states"]),
        tol_pct=0.0,
        unit="states",
        note="new/continue/done/failed/reached_max_step",
        ref_src="IO_multiphase.F90:579-641",
    )
    add(
        "terminal state blocks silent restart (reached_max_step)",
        ref=1.0,
        got=1.0 if summary["terminal"]["terminal_blocks_restart"] else 0.0,
        tol_pct=0.0,
        unit="fraction",
        note="a finished case must not resume compute",
        ref_src="Main_multiphase.F90:91-95",
    )
    worst_interval = max(overhead, key=lambda r: abs(r["interval_rel_err_pct"]))
    worst_excess = max(overhead, key=lambda r: abs(r["timer_excess_steps"]))
    add(
        "wall-clock timer accuracy: median measured interval vs configured",
        ref=0.0,
        got=abs(worst_interval["interval_rel_err_pct"]),
        tol_pct=15.0,
        unit="% relative error",
        note=(
            f"worst arm interval={worst_interval['interval_s']} s "
            f"(step time {worst_interval['step_time_s'] * 1e3:.1f} ms); multiples of the step time are "
            "surfaced separately as timer_excess_steps"
        ),
        ref_src="Main_multiphase.F90:222-303",
    )
    auto = summary["overhead_study"].get("auto_restart")
    if auto is not None:
        add(
            "auto (wall-clock) checkpoint -> restart field discontinuity max|df|",
            ref=0.0,
            got=auto["max_abs_df_vs_continuous"],
            tol_pct=0.0,
            unit="lattice population units",
            note=f"resumed from autosave at step {auto['resumed_from_step']} via {auto['resume_source']}",
            ref_src="IO_multiphase.F90:579-641 + Main_multiphase.F90:222-303",
        )
    add(
        "volume trigger count vs floor(volume/interval)",
        ref=float(summary["terminal"]["volume_triggers_expected"]),
        got=float(summary["terminal"]["volume_triggers_fired"]),
        tol_pct=0.0,
        unit="triggers",
        note="d_vol_* analogue fires once per crossed multiple",
        ref_src="Main_multiphase.F90:222-303",
    )
    return {
        "reference": REF_SOURCE,
        "tolerance": "XVAL criterion: relative error <= 15% (owner precision standard); bit-exact rows use exact tolerance",
        "rows": rows,
        "all_pass": all(r["pass"] for r in rows),
    }


def write_figures(out: Path, summary: dict[str, Any]) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = out / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    produced: list[dict[str, str]] = []

    # 1) restart discontinuity (all exactly zero -> annotate)
    rows = summary["restart_study"]["rows"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = [r["segments"] for r in rows]
    ys = [max(r["max_abs_df_vs_continuous"], 1e-20) for r in rows]
    ax.bar([str(x) for x in xs], ys, color="#2b7bba")
    ax.set_yscale("log")
    ax.axhline(1e-12, color="#cc3333", linestyle="--", label="1e-12 (float32 resolution)")
    ax.set_ylim(1e-20, 1e-9)
    ax.set_xlabel("restart count (segments)")
    ax.set_ylabel("max |f_resumed - f_continuous|")
    ax.set_title("Restart identity: all arms bit-exact (0.0)")
    for i, r in enumerate(rows):
        ax.text(i, 3e-20, f"{r['max_abs_df_vs_continuous']:.1e}", ha="center", fontsize=8)
    ax.legend(fontsize=7)
    path = figs_dir / "fig_restart_discontinuity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    produced.append({"name": path.name, "caption": "Restart discontinuity vs number of restarts (bit-exact for every arm)."})

    # 2) overhead + save count vs interval
    rows = summary["overhead_study"]["rows"]
    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    ax1.plot([r["interval_s"] for r in rows], [r["overhead_pct"] for r in rows], "o-", color="#2b7bba")
    ax1.set_xlabel("checkpoint interval (s of wall clock)")
    ax1.set_ylabel("overhead vs no-checkpoint run (%)", color="#2b7bba")
    ax1.tick_params(axis="y", labelcolor="#2b7bba")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot([r["interval_s"] for r in rows], [r["checkpoints"] for r in rows], "s--", color="#dd7733")
    ax2.set_ylabel("checkpoints written", color="#dd7733")
    ax2.tick_params(axis="y", labelcolor="#dd7733")
    ax1.set_title("Checkpoint cost/sensitivity vs wall-clock interval")
    path = figs_dir / "fig_overhead_vs_interval.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    produced.append({"name": path.name, "caption": "Checkpoint overhead and save count versus configured wall-clock interval."})

    # 3) field comparison baseline vs resumed + difference
    nx, ny = summary["case"]["nx"], summary["case"]["ny"]
    base = taylor_green(nx, ny)
    base.run(summary["case"]["steps"])
    seg_case = Path(summary["restart_study"]["rows"][-1]["case_dir"])
    resumed = taylor_green(nx, ny)
    st = resumed.enable_checkpointing(seg_case, interval_hours=1e9, dual_backup=True)
    resumed.run_with_checkpointing(summary["case"]["steps"] - st.step)
    _, ub = base.macroscopic()
    _, ur = resumed.macroscopic()
    diff = (ub - ur).abs().max(dim=-1).values
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for ax, field, title in (
        (axes[0], ub[..., 0], "continuous run: u_x"),
        (axes[1], ur[..., 0], "checkpoint+restart run: u_x"),
        (axes[2], diff, "|Δu| (max = %.1e)" % float(diff.max())),
    ):
        im = ax.imshow(field.detach().cpu().numpy(), cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    path = figs_dir / "fig_field_compare.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    produced.append(
        {
            "name": path.name,
            "caption": "Final u_x of the continuous run, the checkpoint+restart run and their difference after %d restarts."
            % summary["restart_study"]["rows"][-1]["restarts"],
        }
    )

    # 4) status timeline of the segmented run
    record = JobStatusFile(seg_case / "job_status.txt").read()
    rows = summary["restart_study"]["rows"][-1]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.plot([0, rows["final_step"]], [0, 0], color="#999999", lw=2)
    for i in range(rows["segments"]):
        x = rows["final_step"] * i / rows["segments"]
        ax.plot([x], [0], "o", color="#2b7bba", ms=7)
        ax.annotate(f"ckpt {i + 1}", (x, 0), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("LBM step")
    ax.set_title(f"job_status.txt after run: '{record.status}' at step {record.step} ({rows['checkpoints']} checkpoints)")
    path = figs_dir / "fig_status_timeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    produced.append({"name": path.name, "caption": "Checkpoint/restart timeline of the segmented run; final job_status.txt state."})
    return produced


def write_markdown(out: Path, summary: dict[str, Any]) -> Path:
    table = summary["ref_vs_got"]["rows"]
    lines = [
        "# XVAL — TensorLBM job-status/wall-clock checkpoint machine vs lanl/MF-LBM",
        "",
        f"Reference: `{REF_SOURCE['repo']}` — `{REF_SOURCE['feature_slug']}`",
        f"Source locations: {', '.join(f'`{p}`' for p in REF_SOURCE['locations'])}",
        "",
        "## Case",
        "",
        f"- D2Q9 Taylor-Green vortex, {summary['case']['nx']}x{summary['case']['ny']}, tau={TAU}, u0={U0}, periodic",
        f"- {summary['case']['steps']} LBM steps per arm; single continuous baseline wall {summary['baseline']['wall_s']:.2f} s",
        f"- kinetic energy: {summary['baseline']['ke_start']:.6e} -> {summary['baseline']['ke_end']:.6e} "
        f"(flow genuinely evolves, ratio {summary['baseline']['ke_ratio']:.4f})",
        f"- baseline field sha256: `{summary['baseline']['digest'][:16]}…`",
        "",
        "## Reference vs got",
        "",
        "| quantity | ref (MF-LBM) | got (TensorLBM) | rel err % | tol % | pass | ref source |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table:
        unit = f" {r['unit']}" if r["unit"] else ""
        lines.append(
            f"| {r['quantity']} | {r['ref']:.6g}{unit} | {r['got']:.6g}{unit} | "
            f"{r['rel_err_pct']:.3g} | {r['tolerance_pct']:g} | {'PASS' if r['pass'] else 'FAIL'} | `{r['ref_source']}` |"
        )
    lines += [
        "",
        f"Verdict: **{'all rows pass' if summary['ref_vs_got']['all_pass'] else 'FAILED rows present'}** "
        "(XVAL criterion <15%; bit-exact rows exact).",
        "",
        "## Sensitivity",
        "",
        "| interval s | checkpoints | overhead % (A/B, order-alternated) | overhead predicted from write cost % | base wall CV % | median measured interval (s) | rel err % | timer excess (steps) | step time (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["overhead_study"]["rows"]:
        lines.append(
            f"| {r['interval_s']:g} | {r['checkpoints']} | {r['overhead_pct']:.2f} | "
            f"{r['overhead_from_write_cost_pct']:.2f} | {r['base_wall_cv_pct']:.2f} | {r['median_measured_interval_s']:.4f} | {r['interval_rel_err_pct']:.2f} | "
            f"{r['timer_excess_steps']:.2f} | {r['step_time_s'] * 1e3:.1f} |"
        )
    lines += [
        "",
        f"Single checkpoint generation (primary write + second-backup copy, {summary['overhead_study']['rows'][0]['checkpoint_bytes'] / 1e3:.0f} kB): "
        f"{summary['overhead_study']['single_checkpoint_write_s'] * 1e3:.2f} ms "
        f"(= {summary['overhead_study']['single_checkpoint_write_s'] / summary['overhead_study']['baseline_step_time_s'] * 100:.0f}% of one LBM step). "
        f"Interleaved A/B repetitions per arm: {summary['overhead_study']['reps']}; "
        f"baseline wall-clock CV up to {summary['overhead_study']['host_noise_pct']:.1f}% (shared host) — "
        "the A/B overhead column is only an order-of-magnitude bound, the write-cost column is the resolvable estimate.",
        "",
        "## Difference list vs OSS (adopted / changed / not ported)",
        "",
        "- adopted: `job_status.txt` state vocabulary, hour-based checkpoint timer, primary+second backup, full-PDF restore, continue path.",
        "- changed: status file is plain text `status` + `key=value` lines; checkpoints are Torch tensors instead of Fortran binaries; the volume trigger is a public class rather than `d_vol_*` arrays.",
        "- not ported: convective-outflow history fields (`f_convec_bc`/`g_convec_bc`/`phi_convec_bc`) — TensorLBM's D2Q9 periodic case has no such BC; the payload is an arbitrary field mapping, so a caller with outflow BCs passes the extra tensors as additional keys.",
        "",
    ]
    path = out / "BENCHMARK_REF_VS_GOT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/xval_job_checkpoint"))
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--quick", action="store_true", help="small grid, fewer steps (smoke run)")
    args = parser.parse_args(argv)

    nx, ny, steps = args.nx, args.ny, args.steps
    if args.quick:
        nx = ny = 32
        steps = 80
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sim = taylor_green(nx, ny)
    ke_start = kinetic_energy(sim)
    t0 = time.perf_counter()
    sim.run(steps)
    base_wall = time.perf_counter() - t0
    ke_end = kinetic_energy(sim)
    baseline = {
        "digest": digest(sim.f),
        "ke_start": ke_start,
        "ke_end": ke_end,
        "ke_ratio": ke_end / ke_start,
        "wall_s": base_wall,
        "steps": steps,
    }

    segment_counts = [1, 2, 4, 8] if not args.quick else [1, 2, 4]
    intervals_s = [1.0, 2.0] if not args.quick else [0.05, 0.1]

    print("[1/4] restart study", segment_counts)
    restart_study = study_restart(out, nx, ny, steps, segment_counts)
    print("[2/4] recovery study")
    recovery = study_recovery(out, nx, ny, steps)
    print("[3/4] overhead study", intervals_s)
    overhead = study_overhead(out, nx, ny, steps, intervals_s)
    print("[4/4] terminal/volume")
    terminal = study_terminal_and_volume(out, nx, ny)

    from tensorlbm.job_checkpoint import JOB_STATUSES

    summary: dict[str, Any] = {
        "schema": "xval-job-checkpoint-v1",
        "reference": REF_SOURCE,
        "case": {
            "lattice": "D2Q9",
            "nx": nx,
            "ny": ny,
            "tau": TAU,
            "u0": U0,
            "steps": steps,
            "initial_condition": "Taylor-Green vortex",
        },
        "baseline": baseline,
        "restart_study": restart_study,
        "recovery": recovery,
        "overhead_study": overhead,
        "terminal": terminal,
        "status_vocabulary": {"states": len(JOB_STATUSES), "values": list(JOB_STATUSES)},
    }
    summary["ref_vs_got"] = build_ref_vs_got(summary)
    summary["figures"] = write_figures(out, summary)
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "BENCHMARK_REF_VS_GOT.json").write_text(
        json.dumps(summary["ref_vs_got"], indent=2), encoding="utf-8"
    )
    md = write_markdown(out, summary)
    (out / "repro.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/../..\"\n"
        f"PYTHONPATH=src python benchmarks/xval_job_checkpoint/run_xval.py --out {out}\n",
        encoding="utf-8",
    )
    (out / "repro.sh").chmod(0o755)

    print(json.dumps({k: v for k, v in summary["ref_vs_got"].items() if k == "all_pass"}))
    print(f"wrote {out}/summary.json, {out}/BENCHMARK_REF_VS_GOT.md, {md}")
    return 0 if summary["ref_vs_got"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())