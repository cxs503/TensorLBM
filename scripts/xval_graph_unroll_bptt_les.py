#!/usr/bin/env python3
"""XVAL: graph-unroll BPTT calibration of an LES parameter (TensorLBM vs OSS).

Cross-validation of ``XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER`` — the upstream
feature of ``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
recorded in the platform OSS inventory (``oss_solutions`` ids 37 & 39,
report 9555, source lines ``LatFlow/Domain.py:90-96, 121-128, 134-140,
192-196, 244-253``).  Nothing is transcribed from that repository: the method
is re-derived in PyTorch autograd
(:mod:`tensorlbm.les_graph_unroll`) and then cross-validated against routes
that share no code with it.

Predeclared protocol (fixed before any result was inspected)
------------------------------------------------------------
Workload
    2D decaying turbulence, periodic ``2 pi x 2 pi`` box, ``U0 = 0.1``,
    ``Ma = 0.17``, physical ``Re = U0 L / nu = 500`` (``L = 1``).
    Two initial conditions, both *analytic* so that grid refinement changes
    only the discretisation and no random seed enters the run:
    the 2D Taylor-Green vortex (``tgv``) and a six-mode synthetic solenoidal
    field (``fourier``).
    Coarse LES grid ``nx_c = 32``; reference grid ``nx_r = 128`` whose
    solution is 4x4 box-filtered onto the coarse grid (the explicit LES
    filter); unroll window ``T = 200`` steps; loss snapshots 50/100/150/200.

Competing alternatives for the model constant ``Cs``
    B0 ``Cs = 0``      — no model (baseline control)
    B1 ``Cs = 0.17``   — upstream default (``Domain.py:57``)
    B2 BPTT            — autograd through the unrolled graph (the method
                         under test; Adam on ``log10 Cs``, 20 iterations,
                         ``lr = 0.1``, fixed, never retuned)
    B3 golden-section + 13-point dense sweep of the *same* loss — the
                         derivative-free optimum (shares no autograd code)

Two stages, because they answer different questions
    Stage 1 (*identical twin*, method validation): the reference is a coarse
        run of the very same kernel at a known ``Cs_truth`` in {0.17, 0.30}.
        The recoverable optimum is known exactly, so the calibration itself is
        falsifiable.
    Stage 2 (*physical*, accuracy vs a DNS proxy): the reference is the
        filtered fine-grid solution with no sub-grid model.  This is where the
        owner's 15 % accuracy bar applies to the predicted fields.

Predeclared thresholds (``pass`` = ``< 15 %``, owner's bar)
    X1 gradient      autograd ``dJ/dCs`` vs central FD (``h = 1e-3``) at
                     ``Cs`` in {0.05, 0.17, 0.30}.
    X2 optimum       ``|Cs_BPTT - Cs_golden| / Cs_golden`` on the twin problem.
    X3 recovery      ``|Cs_BPTT - Cs_truth| / Cs_truth`` (twin).
    X4 closure       realised eddy viscosity vs analytic ``Cs^2 |S|``
                     (linear shear probe).
    X5 physical      terminal relative-L2 velocity error of each arm vs the
                     filtered fine-grid reference; kinetic-energy decay error.
    Exploratory (reported, *not* gated, no threshold moved after the fact):
        unroll-horizon sensitivity (``T = 100`` vs ``200``), reference-grid
        convergence (128 vs 256), BPTT start-point sensitivity, loss-curve
        identifiability.

Stopping rule: fixed iteration budgets as above; the run stops when the
predeclared stage list is exhausted.  Every number quoted in
``docs/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER.md`` is emitted by this script
into ``artifacts/xval_graph_unroll_bptt_les/results.json`` (zero transcription).
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.les_graph_unroll import reference2d as ref
from tensorlbm.les_graph_unroll.calibrate import (
    CalibProblem,
    autograd_gradient,
    bptt_calibrate,
    fd_gradient,
    field_loss,
    golden_minimize,
    measure_eddy_viscosity,
    snapshot_errors,
    sweep_loss,
)
from tensorlbm.les_graph_unroll.solver2d import macroscopic, unroll

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "xval_graph_unroll_bptt_les"

CASE = dict(u0=0.1, re=500.0, length=1.0, nx_coarse=32, nx_ref=128, nx_conv=256)
T_STEPS = 200
LOSS_SNAPSHOTS = tuple(range(50, T_STEPS + 1, 50))
KE_SNAPSHOTS = tuple(range(25, T_STEPS + 1, 25))
CS_UPSTREAM = 0.17
CS_TRUTHS = (0.17, 0.30)
CS_INIT = 0.02
SWEEP = tuple(round(0.025 * i, 3) for i in range(0, 25))  # 0.000 .. 0.600
BPTT_ITERS = 20
BPTT_LR = 0.1
GOLDEN_ITERS = 25


def _t(ux, uy):
    return (torch.as_tensor(ux, dtype=torch.float64),
            torch.as_tensor(uy, dtype=torch.float64))


def initial_field(kind: str, nx: int):
    if kind == "tgv":
        return ref.tgv_ic(nx, u0=CASE["u0"])
    if kind == "fourier":
        return ref.fourier_ic(nx, u0=CASE["u0"])
    raise ValueError(kind)


def tau_coarse() -> float:
    return ref.tau_for_re(CASE["nx_coarse"], CASE["u0"], CASE["re"], CASE["length"])


# --------------------------------------------------------------------------- #
# Stage 1 reference: identical twin (same kernel, known Cs)
# --------------------------------------------------------------------------- #
def twin_reference(kind: str, cs_truth: float) -> dict:
    f0, _, _ = initial_field(kind, CASE["nx_coarse"])
    f = torch.as_tensor(f0, dtype=torch.float64)
    snaps = {}
    with torch.no_grad():
        _, states = unroll(f, T_STEPS, tau_coarse(), cs_truth, "smagorinsky",
                           snapshots=LOSS_SNAPSHOTS)
        for k, fs in states.items():
            _, ux, uy = macroscopic(fs)
            snaps[int(k)] = (ux.clone(), uy.clone())
    return snaps


def twin_problem(kind: str, cs_truth: float) -> CalibProblem:
    f0, _, _ = initial_field(kind, CASE["nx_coarse"])
    return CalibProblem(f0=torch.as_tensor(f0, dtype=torch.float64),
                        ref_snapshots=twin_reference(kind, cs_truth),
                        tau0=tau_coarse(), n_steps=T_STEPS)


# --------------------------------------------------------------------------- #
# Stage 2 reference: fine-grid DNS proxy, box-filtered onto the coarse grid
# --------------------------------------------------------------------------- #
def dns_reference(kind: str, nx_ref: int, steps: int, snapshots) -> dict:
    tau_ref = ref.tau_for_re(nx_ref, CASE["u0"], CASE["re"], CASE["length"])
    f0, _, _ = initial_field(kind, nx_ref)
    t0 = time.time()
    _, snaps = ref.run_reference(f0, steps, tau_ref, snapshot_steps=snapshots)
    return dict(tau=tau_ref, snaps=snaps, seconds=time.time() - t0, nx=nx_ref)


def dns_problem(kind: str, run: dict, nx_coarse: int, steps: int, snaps) -> CalibProblem:
    ratio = run["nx"] // nx_coarse
    f0, _, _ = initial_field(kind, nx_coarse)
    refs = {}
    for k in snaps:
        rx, ry = run["snaps"][k]
        refs[int(k)] = _t(ref.box_filter(rx, ratio), ref.box_filter(ry, ratio))
    return CalibProblem(f0=torch.as_tensor(f0, dtype=torch.float64),
                        ref_snapshots=refs, tau0=tau_coarse(),
                        n_steps=int(steps))


def ke_curve(prob: CalibProblem, cs: float, snaps) -> dict:
    _, states = unroll(prob.f0, prob.n_steps, prob.tau0, cs, prob.closure,
                       snapshots=tuple(snaps))
    out = {}
    for k, f in states.items():
        _, ux, uy = macroscopic(f)
        out[int(k)] = 0.5 * float((ux * ux + uy * uy).mean())
    return out


def ref_ke(run: dict, snaps) -> dict:
    return {int(k): ref.kinetic_energy(*run["snaps"][k]) for k in snaps}


def err_pct(got: float, want: float) -> float:
    return float(abs(got - want) / abs(want) * 100.0) if want else float("nan")


def starmap(prob: CalibProblem, snaps=KE_SNAPSHOTS):
    """Full solve bookkeeping for one problem: optimum by all three routes."""
    bptt = bptt_calibrate(prob, CS_INIT, iters=BPTT_ITERS, lr=BPTT_LR)
    golden = golden_minimize(prob, 0.005, 0.6, iters=GOLDEN_ITERS)
    sweep = sweep_loss(prob, SWEEP)
    return bptt, golden, sweep


# --------------------------------------------------------------------------- #
# Stage registry: every stage is re-runnable on its own and merges into
# ``results.json``.  ``all`` reproduces the full run in one command.
# --------------------------------------------------------------------------- #
STAGE_SPECS = (
    ("closure",), ("twin", "tgv", "0.17"), ("twin", "tgv", "0.3"),
    ("twin", "fourier", "0.17"), ("phys", "tgv"), ("phys", "fourier"),
    ("horizon", "tgv"), ("horizon", "fourier"),
    ("convergence", "tgv"), ("convergence", "fourier"),
    ("twin_start",), ("oss_diff",),
)


def _load() -> dict:
    path = OUT / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "mission": "17843",
        "anchor": "XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER",
        "protocol": dict(CASE=CASE, T_STEPS=T_STEPS,
                         loss_snapshots=list(LOSS_SNAPSHOTS),
                         ke_snapshots=list(KE_SNAPSHOTS),
                         cs_upstream=CS_UPSTREAM, cs_truths=list(CS_TRUTHS),
                         cs_init=CS_INIT, sweep_grid=list(SWEEP),
                         bptt_iters=BPTT_ITERS, bptt_lr=BPTT_LR,
                         golden_iters=GOLDEN_ITERS),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
            "command": "python3 scripts/xval_graph_unroll_bptt_les.py <stage ...>",
        },
        "closure_check": {},
        "stage1_twin": {},
        "stage2_physical": {},
        "sensitivity": {},
        "oss_closure_difference": {},
    }


def _save(results: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))


def stage_closure(results: dict) -> None:
    checks = {f"literature_cs_{cs:g}": measure_eddy_viscosity(cs)
              for cs in (0.1, 0.17, 0.3, 0.5)}
    checks["oss_proxy_cs_0.17"] = measure_eddy_viscosity(0.17, closure="oss_proxy")
    for row in checks.values():
        row["ratio_measured_over_analytic"] = (
            row["nu_t_measured"] / row["nu_t_predicted"] if row["nu_t_predicted"] else None)
        row["err_pct"] = err_pct(row["nu_t_measured"], row["nu_t_predicted"])
    results["closure_check"] = checks


def stage_twin(results: dict, kind: str, cs_truth: float) -> str:
    prob = twin_problem(kind, cs_truth)
    grad_rows = []
    # Probe points are kept strictly away from the twin optimum (at the optimum
    # J = 0 exactly, so a relative gradient error there would be 0/0).
    for cs in sorted({0.05, round(0.5 * cs_truth, 3), round(0.8 * cs_truth, 3),
                      round(1.3 * cs_truth, 3)}):
        g_ad = autograd_gradient(prob, cs)
        g_fd = fd_gradient(prob, cs, h=1e-3)
        grad_rows.append({"cs": cs, "autograd": g_ad, "finite_difference": g_fd,
                          "abs_diff": abs(g_ad - g_fd),
                          "rel_err_pct": err_pct(g_ad, g_fd)})
    bptt, golden, sweep = starmap(prob)
    results["stage1_twin"].setdefault(kind, {})[f"cs_truth_{cs_truth:g}"] = {
        "x1_gradient": grad_rows,
        "bptt": {k: v for k, v in bptt.items() if k != "history"},
        "bptt_history": bptt["history"],
        "golden": golden,
        "sweep": sweep,
        "x2_rel_diff_bptt_vs_golden_pct": err_pct(bptt["cs"], golden["cs"]),
        "x3_recovery_err_pct": err_pct(bptt["cs"], cs_truth),
        "x3_golden_recovery_err_pct": err_pct(golden["cs"], cs_truth),
        "sweep_argmin": min(sweep, key=lambda r: r[1])[0],
    }
    return (f"twin {kind} truth={cs_truth}: bptt={bptt['cs']:.4f} golden={golden['cs']:.4f} "
            f"recovery_err={err_pct(bptt['cs'], cs_truth):.2f}% "
            f"grad_err={[round(r['rel_err_pct'], 4) for r in grad_rows]}")


def stage_phys(results: dict, kind: str) -> str:
    run = dns_reference(kind, CASE["nx_ref"], T_STEPS, KE_SNAPSHOTS)
    prob = dns_problem(kind, run, CASE["nx_coarse"], T_STEPS, LOSS_SNAPSHOTS)
    bptt, golden, sweep = starmap(prob)
    arms = {"no_model": 0.0, "upstream_0.17": CS_UPSTREAM,
            "bptt": bptt["cs"], "golden": golden["cs"]}
    acc, ke_arms = {}, {}
    for name, cs in arms.items():
        acc[name] = {str(k): v for k, v in snapshot_errors(prob, cs=cs).items()}
        ke_arms[name] = {str(k): v for k, v in ke_curve(prob, cs, KE_SNAPSHOTS).items()}
    ke_ref = {str(k): v for k, v in ref_ke(run, KE_SNAPSHOTS).items()}
    term = str(max(LOSS_SNAPSHOTS))
    f0_ref, ux0, uy0 = initial_field(kind, CASE["nx_ref"])
    ke0 = 0.5 * float(np.mean(ux0 ** 2 + uy0 ** 2))
    terminal_err = {n: acc[n][term] * 100.0 for n in arms}
    results["stage2_physical"][kind] = {
        "reference": {"nx": run["nx"], "tau": run["tau"], "seconds": run["seconds"]},
        "coarse_tau0": tau_coarse(),
        "field_err_pct": {n: {k: v * 100.0 for k, v in acc[n].items()} for n in arms},
        "terminal_field_err_pct": terminal_err,
        "terminal_reduction_vs_no_model_pct": 100.0 * (
            1.0 - acc["bptt"][term] / acc["no_model"][term]),
        "ke_ref": ke_ref,
        "ke_arms": ke_arms,
        "ke0": ke0,
        "ke_terminal_err_pct": {n: err_pct(ke_arms[n][term], ke_ref[term]) for n in arms},
        "ke_decay_err_pct": {n: err_pct(ke0 - ke_arms[n][term], ke0 - ke_ref[term])
                             for n in arms},
        "bptt": {k: v for k, v in bptt.items() if k != "history"},
        "bptt_history": bptt["history"],
        "golden": golden,
        "sweep": sweep,
        "x2_rel_diff_bptt_vs_golden_pct": err_pct(bptt["cs"], golden["cs"]),
        "sweep_range": {"min": min(v for _, v in sweep), "max": max(v for _, v in sweep),
                        "loss_at_cs0": dict(sweep)[0.0],
                        "argmin_cs": min(sweep, key=lambda r: r[1])[0]},
    }
    return (f"phys {kind}: bptt={bptt['cs']:.4f} golden={golden['cs']:.4f} "
            f"term_err_pct={ {k: round(v, 2) for k, v in terminal_err.items()} }")


def stage_horizon(results: dict, kind: str) -> str:
    run = dns_reference(kind, CASE["nx_ref"], T_STEPS, KE_SNAPSHOTS)
    half = tuple(s for s in LOSS_SNAPSHOTS if s <= T_STEPS // 2)
    prob_half = dns_problem(kind, run, CASE["nx_coarse"], T_STEPS // 2, half)
    bptt_half = bptt_calibrate(prob_half, CS_INIT, iters=BPTT_ITERS, lr=BPTT_LR)
    full = results["stage2_physical"].get(kind, {}).get("bptt")
    payload = {"T_100": {"cs": bptt_half["cs"], "loss": bptt_half["loss"]}}
    if full:
        payload["T_200"] = {"cs": full["cs"], "loss": full["loss"]}
        payload["rel_diff_pct"] = err_pct(bptt_half["cs"], full["cs"])
    results["sensitivity"].setdefault(kind, {})["horizon"] = payload
    return f"horizon {kind}: T=100 Cs={bptt_half['cs']:.4f}"


def stage_convergence(results: dict, kind: str) -> str:
    conv = dns_reference(kind, CASE["nx_conv"], T_STEPS, KE_SNAPSHOTS)
    run = dns_reference(kind, CASE["nx_ref"], T_STEPS, KE_SNAPSHOTS)
    k128, k256 = ref_ke(run, KE_SNAPSHOTS), ref_ke(conv, KE_SNAPSHOTS)
    e = err_pct(k128[max(KE_SNAPSHOTS)], k256[max(KE_SNAPSHOTS)])
    results["sensitivity"].setdefault(kind, {})["reference_convergence"] = {
        "ke_128": {str(k): v for k, v in k128.items()},
        "ke_256": {str(k): v for k, v in k256.items()},
        "ke_rel_err_pct_at_T": e, "tau_256": conv["tau"],
        "seconds_256": conv["seconds"],
    }
    return f"convergence {kind}: 128 vs 256 KE rel err {e:.2f}%"


def stage_twin_start(results: dict) -> str:
    prob = twin_problem("tgv", CS_UPSTREAM)
    alt = bptt_calibrate(prob, 0.45, iters=BPTT_ITERS, lr=BPTT_LR)
    base = results["stage1_twin"].get("tgv", {}).get("cs_truth_0.17", {}).get("bptt")
    payload = {"start_0.45_cs": alt["cs"], "cs_truth": CS_UPSTREAM}
    if base:
        payload["start_0.02_cs"] = base["cs"]
        payload["rel_diff_pct"] = err_pct(alt["cs"], base["cs"])
    results["sensitivity"]["twin_start_point"] = payload
    return f"twin_start: from 0.45 -> Cs={alt['cs']:.4f}"


def stage_oss_diff(results: dict) -> str:
    run = dns_reference("tgv", CASE["nx_ref"], T_STEPS, LOSS_SNAPSHOTS)
    p_lit = dns_problem("tgv", run, CASE["nx_coarse"], T_STEPS, LOSS_SNAPSHOTS)
    p_tmp = dns_problem("tgv", run, CASE["nx_coarse"], T_STEPS, LOSS_SNAPSHOTS)
    p_oss = CalibProblem(f0=p_tmp.f0, ref_snapshots=p_tmp.ref_snapshots,
                         tau0=p_tmp.tau0, n_steps=p_tmp.n_steps, closure="oss_proxy")
    lit = measure_eddy_viscosity(0.17)
    oss_cl = measure_eddy_viscosity(0.17, closure="oss_proxy")
    results["oss_closure_difference"] = {
        "loss_literature_at_cs_0.17": float(field_loss(p_lit, CS_UPSTREAM)),
        "loss_oss_proxy_at_cs_0.17": float(field_loss(p_oss, CS_UPSTREAM)),
        "golden_literature": golden_minimize(p_lit, 0.005, 0.6, iters=GOLDEN_ITERS),
        "golden_oss_proxy": golden_minimize(p_oss, 0.005, 0.6, iters=GOLDEN_ITERS),
        "eddy_viscosity_ratio_oss_over_literature_at_cs_0.17":
            oss_cl["nu_t_measured"] / lit["nu_t_measured"],
    }
    g_lit = results["oss_closure_difference"]["golden_literature"]["cs"]
    g_oss = results["oss_closure_difference"]["golden_oss_proxy"]["cs"]
    return f"oss_diff: golden literature Cs={g_lit:.4f} vs oss_proxy Cs={g_oss:.4f}"


def run_stage(results: dict, stage: tuple) -> str:
    name = stage[0]
    if name == "closure":
        stage_closure(results)
        return "closure: done"
    if name == "twin":
        return stage_twin(results, stage[1], float(stage[2]))
    if name == "phys":
        return stage_phys(results, stage[1])
    if name == "horizon":
        return stage_horizon(results, stage[1])
    if name == "convergence":
        return stage_convergence(results, stage[1])
    if name == "twin_start":
        return stage_twin_start(results)
    if name == "oss_diff":
        return stage_oss_diff(results)
    raise SystemExit(f"unknown stage {stage}")


def main(argv) -> int:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    results = _load()
    stages = list(STAGE_SPECS) if not argv or argv[0] == "all" else [tuple(argv)]
    t0 = time.time()
    for stage in stages:
        msg = run_stage(results, stage)
        _save(results)
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
    results["runtime_seconds"] = results.get("runtime_seconds", 0.0) + (time.time() - t0)
    results["stages_done"] = sorted({st[0] + ("_" + "_".join(st[1:]) if len(st) > 1 else "")
                                     for st in stages})
    _save(results)
    print(f"wrote {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
