#!/usr/bin/env python3
"""GAP-DIFFLBM-3D-COST acceptance entry point.

Runs the three acceptance cases of the 3-D differentiable-LBM cost envelope:

  gradient   AD (checkpointed, disk-checkpointed reverse mode) vs
             (a) central/Richardson finite difference — independent reference,
             (b) continuum ABC-decay anchor — external anchor,
             (c) full-storage BPTT — internal consistency.
             Gate: |g_AD - g_FD| / |g_FD| <= 1e-6.
  envelope   measured peak RSS + wall time vs grid size, compared with the
             analytic state-count model, extrapolated to 256^3 and judged
             against the anchor gates (peak memory <= 8 GB, inversion <= 2 h
             on 16 cores).
  inversion  real tau recovery from a terminal population snapshot, driven by
             the memory-bounded gradient (solver-side entry:
             :func:`tensorlbm.solver3d.inversion_gradient_3d`).

Usage
-----
    PYTHONPATH=src python3 benchmarks/gap_difflbm_3d_cost/run.py --case all --out artifacts/gap_difflbm_3d_cost

Every number in the emitted JSON is produced by this run; nothing is typed in
by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm import diff3d  # noqa: E402
from tensorlbm.solver3d import inversion_gradient_3d  # noqa: E402

GATE_GRAD_RTOL = 1e-6
GATE_MEM_BYTES = 8 * 1024 ** 3
GATE_TIME_S = 2.0 * 3600.0
GATE_CORES = 16


def _probe(args_list):
    """Run memprobe.py in a fresh process and return its JSON record."""
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "memprobe.py")] + args_list
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def _step_timing(grids, reps=None, threads=4):
    """Measured per-step wall time and effective bandwidth of the forward step."""
    torch.set_num_threads(threads)
    c, w = diff3d.constants()
    rows = []
    for n in grids:
        r = reps or max(2, int(round(3e6 / n ** 3)))
        f = diff3d.abc_initial(n, 0.05)
        with torch.no_grad():
            f = diff3d.step3d(f, 0.8, c, w)          # warm-up
            t0 = time.perf_counter()
            for _ in range(r):
                f = diff3d.step3d(f, 0.8, c, w)
            dt = (time.perf_counter() - t0) / r
        sb = diff3d.state_bytes(n)
        rows.append({"grid": n, "cells": n ** 3, "reps": r,
                     "step_seconds": dt, "ns_per_cell": dt / n ** 3 * 1e9,
                     "state_bytes": sb, "assumed_traffic_states": 6,
                     "effective_bandwidth_gbs": 6 * sb / dt / 1e9})
        print(f"[envelope] forward step N={n:>3}: {dt*1e3:8.1f} ms  "
              f"{dt/n**3*1e9:7.1f} ns/cell  eff.BW={6*sb/dt/1e9:5.2f} GB/s")
    return rows


def _overhead_factor(n, steps, segment, threads=4):
    """Measured reverse-mode cost per reverse step, in forward-step units."""
    torch.set_num_threads(threads)
    f0 = diff3d.abc_initial(n, 0.05)
    c, w = diff3d.constants()
    f = f0.clone()
    with torch.no_grad():
        f = diff3d.step3d(f, 0.8, c, w)
        t0 = time.perf_counter()
        for _ in range(3):
            f = diff3d.step3d(f, 0.8, c, w)
        t_fwd = (time.perf_counter() - t0) / 3
    res = diff3d.checkpointed_gradient(f0, 0.8, steps, segment=segment)
    # cost per reverse step: total / steps, expressed in forward-step units
    per_rev = res.seconds / steps
    return {"grid": n, "steps": steps, "segment": segment,
            "forward_step_seconds": t_fwd, "reverse_seconds": res.seconds,
            "seconds_per_reverse_step": per_rev,
            "overhead_factor_vs_forward": per_rev / t_fwd,
            "n_forward_steps": res.n_forward_steps}


def case_gradient(args, grids, steps):
    """Gradient correctness: checkpointed AD vs FD / analytic / full BPTT."""
    rows = []
    for n in grids:
        c, w = diff3d.constants()
        f0 = diff3d.abc_initial(n, amp=args.amp, dtype=torch.float64)
        ad = diff3d.checkpointed_gradient(f0, args.tau, steps, segment=args.segment)
        fd = diff3d.finite_difference_gradient(f0, args.tau, steps,
                                               h=args.fd_h, richardson=True)
        bptt = diff3d.full_bptt_gradient(f0, args.tau, steps)
        ana = diff3d.analytic_decay_gradient(n, args.tau, steps, amp=args.amp,
                                            j_value=ad.value)
        rel_fd = abs(ad.grad["tau"] - fd.grad["tau"]) / abs(fd.grad["tau"])
        rel_bptt = abs(ad.grad["tau"] - bptt.grad["tau"]) / abs(bptt.grad["tau"])
        rel_ana = abs(ad.grad["tau"] - ana["dJ_dtau"]) / abs(ana["dJ_dtau"])
        # convergence of the *value* against the continuum decay
        j_cont = 1.5 * (n ** 3) * args.amp ** 2
        j0_meas = None
        rows.append({
            "grid": n, "steps": steps, "tau": args.tau, "amp": args.amp,
            "segment": args.segment,
            "J_ad": ad.value, "J_fd": fd.value,
            "grad_ad": ad.grad["tau"], "grad_fd_richardson": fd.grad["tau"],
            "grad_full_bptt": bptt.grad["tau"],
            "grad_analytic_anchor": ana["dJ_dtau"],
            "rel_err_ad_vs_fd": rel_fd, "rel_err_ad_vs_bptt": rel_bptt,
            "rel_err_ad_vs_analytic": rel_ana,
            "gate_1e-6": bool(rel_fd <= GATE_GRAD_RTOL),
            "fd_mode": fd.mode, "fd_h": args.fd_h,
            "seconds_ad": ad.seconds, "seconds_fd": fd.seconds,
            "n_forward_steps_ad": ad.n_forward_steps,
            "checkpoints": ad.checkpoints,
        })
        print(f"[gradient] N={n:>3} T={steps} tau={args.tau}: "
              f"g_AD={ad.grad['tau']:+.8e} g_FD={fd.grad['tau']:+.8e} "
              f"g_BPTT={bptt.grad['tau']:+.8e} g_ana={ana['dJ_dtau']:+.8e} | "
              f"rel(FD)={rel_fd:.3e} rel(BPTT)={rel_bptt:.3e} "
              f"rel(analytic)={rel_ana:.3e} {'PASS' if rel_fd <= GATE_GRAD_RTOL else 'FAIL'}")
    return rows


def case_two_tau(args, n, steps):
    """Parameter sensitivity: same gate at a second, lower relaxation time."""
    rows = []
    for tau in [float(x) for x in args.tau_points.split(",")]:
        f0 = diff3d.abc_initial(n, amp=args.amp, dtype=torch.float64)
        ad = diff3d.checkpointed_gradient(f0, tau, steps, segment=args.segment)
        fd = diff3d.finite_difference_gradient(f0, tau, steps, h=args.fd_h,
                                               richardson=True)
        rel = abs(ad.grad["tau"] - fd.grad["tau"]) / abs(fd.grad["tau"])
        rows.append({"grid": n, "steps": steps, "tau": tau,
                     "grad_ad": ad.grad["tau"], "grad_fd": fd.grad["tau"],
                     "rel_err_ad_vs_fd": rel, "J_ad": ad.value,
                     "gate_1e-6": bool(rel <= GATE_GRAD_RTOL)})
        print(f"[two-tau] N={n} T={steps} tau={tau}: rel(FD)={rel:.3e} "
              f"{'PASS' if rel <= GATE_GRAD_RTOL else 'FAIL'}")
    return rows


def case_envelope(args):
    """Memory / time envelope: measured vs model, extrapolated to 256^3."""
    grids = [int(x) for x in args.env_grids.split(",")]
    timing = _step_timing(grids, threads=args.threads)
    mem_rows = []
    for n in grids:
        common = ["--grid", str(n), "--dtype",
                  str(args.dtype).replace("torch.", ""), "--threads",
                  str(args.threads)]
        short = _probe(common + ["--steps", str(args.env_steps), "--mode", "grad",
                                 "--segment", str(args.segment)])
        short["baseline_bytes"] = _probe(common + ["--mode", "baseline"])["peak_bytes"]
        short["states_equivalent"] = ((short["peak_bytes"] - short["baseline_bytes"])
                                      / short["state_bytes"])
        mem_rows.append({"grid": n, "steps": args.env_steps,
                         "state_bytes": short["state_bytes"],
                         "peak_bytes": short["peak_bytes"],
                         "baseline_bytes": short["baseline_bytes"],
                         "states_equivalent_rss": short["states_equivalent"],
                         "model_states": args.segment + 6,
                         "bytes_model": diff3d.memory_model_bytes(
                             n, args.segment, args.dtype)})
        print(f"[envelope] mem N={n:>3} T={args.env_steps}: "
              f"peakRSS={(short['peak_bytes']-short['baseline_bytes'])/2**20:8.1f} MiB over baseline "
              f"({short['states_equivalent']:5.1f} states, model {args.segment+6})")

    # boundedness: memory must not grow with the number of steps
    n_probe = grids[-1]
    common = ["--grid", str(n_probe), "--dtype",
              str(args.dtype).replace("torch.", ""), "--threads", str(args.threads)]
    base = _probe(common + ["--mode", "baseline"])["peak_bytes"]
    t_rows = []
    for T in args.mem_steps:
        rec = _probe(common + ["--steps", str(T), "--mode", "grad",
                               "--segment", str(args.segment)])
        t_rows.append({"grid": n_probe, "steps": T,
                       "state_bytes": rec["state_bytes"],
                       "peak_bytes_over_baseline": rec["peak_bytes"] - base,
                       "states_equivalent_rss": (rec["peak_bytes"] - base)
                       / rec["state_bytes"],
                       "seconds": rec.get("seconds")})
        print(f"[envelope] boundedness N={n_probe} T={T:>4}: "
              f"{(rec['peak_bytes']-base)/2**20:8.1f} MiB over baseline "
              f"({(rec['peak_bytes']-base)/rec['state_bytes']:5.1f} states)")
    over = _overhead_factor(args.over_grid, args.over_steps, args.segment,
                            threads=args.threads)
    print(f"[envelope] reverse overhead N={over['grid']}: "
          f"{over['overhead_factor_vs_forward']:.2f} x forward step")

    # extrapolation to 256^3: per-cell step cost is size independent (measured
    # flat over the N range above), so scale by cells x overhead x threads
    cells_256 = 256 ** 3
    ns_cell = sum(r["ns_per_cell"] for r in timing) / len(timing)
    t_step_4t = ns_cell * 1e-9 * cells_256
    ext = []
    for dtype in (torch.float64, torch.float32):
        for threads, scale in ((4, 1.0), (16, 4.0)):
            t_rev = t_step_4t * over["overhead_factor_vs_forward"] / scale
            n_steps = args.final_T
            ext.append({
                "grid": 256, "dtype": str(dtype).replace("torch.", ""),
                "threads": threads, "thread_scaling_assumption": scale,
                "assumed_T": n_steps, "segment": args.segment,
                "projected_forward_step_seconds": t_step_4t / scale,
                "projected_reverse_step_seconds": t_rev,
                "projected_inversion_seconds": t_rev * n_steps,
                "projected_peak_bytes_model": diff3d.memory_model_bytes(
                    256, args.segment, dtype),
            })
            e = ext[-1]
            e["gate_memory_8GB"] = bool(e["projected_peak_bytes_model"] <= GATE_MEM_BYTES)
            e["gate_time_2h"] = bool(e["projected_inversion_seconds"] <= GATE_TIME_S)
            e["max_T_for_2h"] = GATE_TIME_S / t_rev
            print(f"[envelope] 256^3 {e['dtype']:>7} {threads:>2} cores: "
                  f"fwd step={e['projected_forward_step_seconds']:6.2f}s "
                  f"rev step={t_rev:6.2f}s  T={n_steps} -> "
                  f"{e['projected_inversion_seconds']/3600:6.2f} h "
                  f"(T_max(2h)={e['max_T_for_2h']:.0f})  peak="
                  f"{e['projected_peak_bytes_model']/2**30:5.2f} GiB "
                  f"mem{'OK' if e['gate_memory_8GB'] else 'OVER'} "
                  f"time{'OK' if e['gate_time_2h'] else 'OVER'}")
    return {"step_timing": timing, "memory": mem_rows, "boundedness": t_rows,
            "reverse_overhead": over, "extrapolation_256": ext,
            "ns_per_cell_mean": ns_cell}


def case_inversion(args):
    """Real tau inversion through the solver-side entry point."""
    c, w = diff3d.constants()
    n = args.inv_grid
    f0 = diff3d.abc_initial(n, amp=args.amp, dtype=torch.float64)
    f_obs = f0.clone()
    with torch.no_grad():
        for _ in range(args.inv_steps):
            f_obs = diff3d.step3d(f_obs, args.tau_true, c, w)
    t0 = time.perf_counter()
    tau_hat, history = inversion_gradient_3d(
        f0, f_obs, args.inv_steps, tau0=args.tau0, lr=args.lr,
        iters=args.iters, segment=args.segment)
    seconds = time.perf_counter() - t0
    err = abs(tau_hat - args.tau_true) / args.tau_true
    print(f"[inversion] N={n} T={args.inv_steps} tau_true={args.tau_true} "
          f"tau0={args.tau0} -> tau_hat={tau_hat:.8f} "
          f"rel_err={err:.3e} ({args.iters} iters, {seconds:.1f}s) "
          f"{'PASS(<=3%)' if err <= 0.03 else 'FAIL'}")
    return {"grid": n, "steps": args.inv_steps, "tau_true": args.tau_true,
            "tau0": args.tau0, "tau_hat": tau_hat, "rel_err": err,
            "iters": args.iters, "lr": args.lr, "seconds": seconds,
            "history": history,
            "gate_3pct": bool(err <= 0.03)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all",
                    choices=["gradient", "two_tau", "envelope", "inversion", "all"])
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts",
                                                  "gap_difflbm_3d_cost"))
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--amp", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--segment", type=int, default=4)
    ap.add_argument("--fd-h", type=float, default=1e-4)
    ap.add_argument("--grids", default="16,24,32")
    ap.add_argument("--env-grids", default="16,24,32,48")
    ap.add_argument("--env-steps", type=int, default=32)
    ap.add_argument("--mem-steps", default="32,128,256")
    # memory-boundedness sweep step counts
    ap.add_argument("--over-grid", type=int, default=32)
    ap.add_argument("--over-steps", type=int, default=8)
    ap.add_argument("--dtype", default="float64",
                    choices=["float64", "float32"])
    ap.add_argument("--tau-points", default="0.6,0.8,1.1")
    ap.add_argument("--final-T", type=int, default=800,
                    help="assumed step count of the 256^3 inversion")
    ap.add_argument("--inv-grid", type=int, default=16)
    ap.add_argument("--inv-steps", type=int, default=64)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--tau0", type=float, default=0.62)
    ap.add_argument("--tau-true", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.dtype = getattr(torch, args.dtype)
    args.mem_steps = [int(x) for x in args.mem_steps.split(",")]

    os.makedirs(args.out, exist_ok=True)
    grids = [int(x) for x in args.grids.split(",")]
    payload = {"anchor": "GAP-DIFFLBM-3D-COST", "torch": torch.__version__,
               "threads": args.threads,
               "gates": {"grad_rtol": GATE_GRAD_RTOL,
                         "peak_mem_bytes": GATE_MEM_BYTES,
                         "inversion_seconds": GATE_TIME_S,
                         "cores": GATE_CORES},
               "cuda_available": torch.cuda.is_available()}

    if args.case in ("gradient", "all"):
        payload["gradient"] = case_gradient(args, grids, args.steps)
    if args.case in ("two_tau", "all"):
        payload["two_tau"] = case_two_tau(args, grids[-1], args.steps)
    if args.case in ("envelope", "all"):
        payload["envelope"] = case_envelope(args)
    if args.case in ("inversion", "all"):
        payload["inversion"] = case_inversion(args)

    name = f"run_{args.case}.json"
    path = os.path.join(args.out, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())