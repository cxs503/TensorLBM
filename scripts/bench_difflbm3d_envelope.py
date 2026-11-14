#!/usr/bin/env python3
"""GAP-DIFFLBM-3D-COST: cost + gradient envelope of 3D D3Q19 expanded-form AD.

One entry point, four measured stages (each stage is normally run in a *fresh
subprocess* because ``VmHWM`` is a high-water mark that cannot be reset):

``grad``
    Builds a deterministic 256^3-class problem: 3D divergence-free random IC
    (seed 0), ``J(tau) = 1/2 ||u(T) - u_target||^2`` against the terminal
    velocity field of a reference rollout at ``tau_true``.  Computes
    ``dJ/dtau`` and ``dJ/du0`` with the segmented (disk-offload) unrolled
    autograd of :mod:`tensorlbm.diff3d_envelope`, cross-checks against the
    full-graph ``tensorlbm.autograd_path.rollout`` backward, and against central
    finite differences of the forward solver (the acceptance criterion).

``envelope``
    Peak RSS (VmHWM) + wall time of one forward+backward, for three engines
    (plain full graph / per-step checkpoint / segmented disk replay) so the
    memory-vs-resolution scaling can be measured, not assumed.

``physics``
    Physical fidelity of the same 3D operator against analytic solutions
    (2D Taylor-Green embedded in 3D: ``E(t) = E0 exp(-4 nu t)``; z shear wave:
    ``A(t) = A0 exp(-nu k^2 t)``) at two resolutions -- the "did we build the
    right operator" anchor.

``inversion``
    Actual tau-recovery inversion (gradient descent from the segmented AD
    gradient) with wall clock, i.e. the "one inversion" cost of the criterion.

Usage (heavy configs go through the cluster queue, never the 99 control plane)::

    python scripts/bench_difflbm3d_envelope.py --stage all --n 32 --steps 20 \\
        --out artifacts/difflbm3d_envelope/n32.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.autograd_path import differentiable_step, rollout  # noqa: E402
from tensorlbm.diff3d_envelope import (  # noqa: E402
    SegmentStore,
    divergence_free_ic,
    peak_rss_gib,
    segmented_rollout,
    shear_wave_amplitude,
    shear_wave_ic,
    taylor_green_2d_ic,
    taylor_green_energy,
)
from tensorlbm.solver3d import equilibrium3d, macroscopic3d  # noqa: E402

TORCH_DTYPE = {"float32": torch.float32, "float64": torch.float64}


# --------------------------------------------------------------------------- #
# problem definition
# --------------------------------------------------------------------------- #
def step_fn(f: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """One periodic D3Q19 BGK step (the differentiable operator under test)."""
    return differentiable_step(f, tau, None)


def ic_distribution(u: torch.Tensor, *, dtype=torch.float32) -> torch.Tensor:
    """Local-equilibrium populations of a velocity field (rho = 1)."""
    rho = torch.ones_like(u[0])
    return equilibrium3d(rho, u[0], u[1], u[2]).to(dtype)


def velocity_loss(f: torch.Tensor, u_target: torch.Tensor) -> torch.Tensor:
    """``J = 1/2 ||u(f) - u_target||^2 / Ncells`` accumulated in float64.

    The fp32 residual is upcast before the reduction: the sum has ~10^7 terms,
    a float64 accumulator keeps the *loss* rounding at the 1e-11 relative level
    so a central-difference probe is not swamped by summation noise.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    du = torch.stack([ux, uy, uz]).to(torch.float64) - u_target.to(torch.float64)
    return 0.5 * du.square().sum() / du[0].numel()


def forward_u(n: int, u0: torch.Tensor, tau: float, n_steps: int, dtype) -> torch.Tensor:
    """Plain (no-grad) forward rollout returning the terminal velocity field."""
    f = ic_distribution(u0, dtype=dtype)
    t = torch.tensor(tau, dtype=dtype)
    with torch.no_grad():
        for _ in range(n_steps):
            f = step_fn(f.detach(), t)
    _, ux, uy, uz = macroscopic3d(f)
    return torch.stack([ux, uy, uz])


# --------------------------------------------------------------------------- #
# stage: gradient correctness (AD vs FD) + AD vs full-graph
# --------------------------------------------------------------------------- #
def stage_grad(args) -> dict:
    """AD (segmented, disk replay) vs central FD -- the gradient criterion."""
    dtype = TORCH_DTYPE[args.dtype]
    torch.set_num_threads(args.threads)
    n, steps, h = args.n, args.steps, args.fd_step
    u0 = divergence_free_ic(n, seed=args.seed, amplitude=args.amplitude, dtype=dtype)

    t0 = time.perf_counter()
    u_target = forward_u(n, u0, args.tau_true, steps, dtype)
    t_target = time.perf_counter() - t0
    du = u_target.to(torch.float64)
    n_cells = du[0].numel()
    loss_scale = 2 * n_cells

    def loss_of(taux: float, amp: float = 1.0) -> float:
        """Forward-only J (no graph): the finite-difference oracle."""
        u = forward_u(n, (u0 * amp).to(dtype), taux, steps, dtype)
        return float((u.to(torch.float64) - du).square().sum() / loss_scale)

    # ---- segmented AD (disk replay, bounded memory) -----------------------
    tau = torch.tensor(args.tau, dtype=dtype, requires_grad=True)
    u0g = u0.clone().requires_grad_(True)
    f0 = ic_distribution(u0g, dtype=dtype)  # graph-connected through u0
    store = SegmentStore(args.store_dir, -(-steps // args.seg_len), f0.shape, dtype)
    t0 = time.perf_counter()
    f_T = segmented_rollout(f0, tau, steps, step_fn, seg_len=args.seg_len, store=store)
    J = velocity_loss(f_T, u_target)
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_tau, g_u0 = torch.autograd.grad(J, [tau, u0g])
    t_bwd = time.perf_counter() - t0
    peak = peak_rss_gib()
    store.close(remove=True)

    # ---- full-graph AD (identical quantity, unbounded memory) -------------
    tau_full = torch.tensor(args.tau, dtype=dtype, requires_grad=True)
    f0_full = ic_distribution(u0, dtype=dtype)
    f_full = rollout(f0_full, steps, tau_full, None, checkpoint=False)
    J_full = velocity_loss(f_full, u_target)
    g_full = torch.autograd.grad(J_full, [tau_full])
    g_tau_full = float(g_full[0])
    J_full_v = float(J_full)
    del f_full, J_full

    # ---- finite differences ----------------------------------------------
    # 2-point central differences at +-h and +-2h, reused by both the
    # 4th-order (5-point) stencil and the FD-step scan
    Jp1, Jm1 = loss_of(args.tau + h), loss_of(args.tau - h)
    Jp2, Jm2 = loss_of(args.tau + 2 * h), loss_of(args.tau - 2 * h)
    g_tau_fd_2pt = (Jp1 - Jm1) / (2 * h)
    g_tau_fd_4pt = (-Jp2 + 8 * Jp1 - 8 * Jm1 + Jm2) / (12 * h)
    fd_scan = {"2pt_h": g_tau_fd_2pt, "4pt_h": g_tau_fd_4pt}
    if 2 * h in [float(x) for x in args.fd_scan]:
        fd_scan["2pt_2h"] = (Jp2 - Jm2) / (4 * h)
    # amplitude (initial-condition) direction: FD of J((1+d) u0) measures the
    # directional derivative <g_u0, u0>, not the plain sum of g_u0
    Lp1, Lm1 = loss_of(args.tau, 1 + h), loss_of(args.tau, 1 - h)
    Lp2, Lm2 = loss_of(args.tau, 1 + 2 * h), loss_of(args.tau, 1 - 2 * h)
    g_u0_fd = (-Lp2 + 8 * Lp1 - 8 * Lm1 + Lm2) / (12 * h)
    g_u0_ad = float((g_u0.double() * u0.double()).sum())

    def relerr(a, b):
        return abs(a - b) / max(abs(b), 1e-300)

    res = {
        "stage": "grad",
        "n": n,
        "cells": n**3,
        "steps": steps,
        "seg_len": args.seg_len,
        "dtype": args.dtype,
        "threads": args.threads,
        "tau": args.tau,
        "tau_true": args.tau_true,
        "fd_step": h,
        "J": float(J),
        "J_full_graph": J_full_v,
        "dJ_dtau_ad_segmented": float(g_tau),
        "dJ_dtau_ad_full_graph": g_tau_full,
        "dJ_dtau_fd_2pt": g_tau_fd_2pt,
        "dJ_dtau_fd_4pt": g_tau_fd_4pt,
        "dJ_du0_directional_ad": g_u0_ad,
        "dJ_du0_directional_fd": g_u0_fd,
        "rel_err_tau_ad_vs_fd_2pt": relerr(float(g_tau), g_tau_fd_2pt),
        "rel_err_tau_ad_vs_fd_4pt": relerr(float(g_tau), g_tau_fd_4pt),
        "rel_err_tau_ad_vs_fd": relerr(float(g_tau), g_tau_fd_4pt),
        "rel_err_u0_ad_vs_fd": relerr(g_u0_ad, g_u0_fd),
        "rel_err_tau_segmented_vs_full": relerr(float(g_tau), g_tau_full),
        "fd_scan": fd_scan,
        "t_target_forward_s": t_target,
        "t_ad_forward_s": t_fwd,
        "t_ad_backward_s": t_bwd,
        "peak_rss_gib": peak,
    }
    res["criterion_grad_1e-6"] = bool(
        res["rel_err_tau_ad_vs_fd_4pt"] <= 1e-6 and res["rel_err_u0_ad_vs_fd"] <= 1e-6
    )
    return res


# --------------------------------------------------------------------------- #
# stage: memory / time envelope of the three engines
# --------------------------------------------------------------------------- #
def stage_envelope(args) -> dict:
    dtype = TORCH_DTYPE[args.dtype]
    torch.set_num_threads(args.threads)
    n, steps = args.n, args.steps
    u0 = divergence_free_ic(n, seed=args.seed, amplitude=args.amplitude, dtype=dtype)
    u_target = forward_u(n, u0, args.tau_true, 4, dtype)  # cheap fixed target
    f0 = ic_distribution(u0, dtype=dtype)
    field_gib = f0.numel() * f0.element_size() / 2**30
    rows = []

    def run(engine: str):
        tau = torch.tensor(args.tau, dtype=dtype, requires_grad=True)
        t0 = time.perf_counter()
        if engine == "plain":
            f = rollout(f0, steps, tau, None, checkpoint=False)
        elif engine == "step_ckpt":
            f = rollout(f0, steps, tau, None, checkpoint=True)
        else:
            f = segmented_rollout(f0, tau, steps, step_fn, seg_len=args.seg_len)
        J = velocity_loss(f, u_target)
        t_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        J.backward()
        t_bwd = time.perf_counter() - t0
        row = {
            "engine": engine,
            "n": n,
            "steps": steps,
            "dtype": args.dtype,
            "seg_len": args.seg_len if engine == "segmented" else None,
            "field_gib": field_gib,
            "peak_rss_gib": peak_rss_gib(),
            "peak_rss_fields": peak_rss_gib() / field_gib,
            "t_forward_s": t_fwd,
            "t_backward_s": t_bwd,
            "t_total_s": t_fwd + t_bwd,
            "s_per_step_fwd": t_fwd / steps,
            "dJ_dtau": float(tau.grad) if tau.grad is not None else None,
        }
        del f, J
        return row

    for engine in args.engines:
        rows.append(run(engine))
    return {"stage": "envelope", "rows": rows, "field_gib": field_gib}


# --------------------------------------------------------------------------- #
# stage: physical fidelity against analytic solutions
# --------------------------------------------------------------------------- #
def stage_physics(args) -> dict:
    dtype = TORCH_DTYPE[args.dtype]
    torch.set_num_threads(args.threads)
    out = {"stage": "physics", "dtype": args.dtype, "cases": []}
    for n in args.resolutions:
        tau = args.tau
        nu = (tau - 0.5) / 3.0
        # --- 2D Taylor-Green embedded in 3D -----------------------------
        k_tg = 2 * 3.141592653589793 / n
        u0 = taylor_green_2d_ic(n, amplitude=args.amplitude, dtype=dtype)
        f = ic_distribution(u0, dtype=dtype)
        t = torch.tensor(tau, dtype=dtype)
        series = []
        with torch.no_grad():
            for k in range(args.steps + 1):
                _, ux, uy, uz = macroscopic3d(f)
                e = float(0.5 * (ux.double() ** 2 + uy.double() ** 2 + uz.double() ** 2).mean())
                series.append({"t": k, "E": e, "E_analytic": taylor_green_energy(k, amplitude=args.amplitude, nu=nu, k=k_tg)})
                if k < args.steps:
                    f = step_fn(f, t)
        e_end, e_ref = series[-1]["E"], series[-1]["E_analytic"]
        out["cases"].append(
            {
                "case": "taylor_green_2d_in_3d",
                "n": n,
                "tau": tau,
                "nu_lattice": nu,
                "steps": args.steps,
                "E_measured": e_end,
                "E_analytic": e_ref,
                "rel_err": abs(e_end - e_ref) / e_ref,
                "series": series,
            }
        )
        # --- z shear wave ------------------------------------------------
        u0 = shear_wave_ic(n, amplitude=args.amplitude, wavelength=1, dtype=dtype)
        f = ic_distribution(u0, dtype=dtype)
        k_wave = 2 * 3.141592653589793 / n
        series = []
        with torch.no_grad():
            for k in range(args.steps + 1):
                _, ux, _, _ = macroscopic3d(f)
                a = float(ux.double().max())
                series.append(
                    {
                        "t": k,
                        "A": a,
                        "A_analytic": shear_wave_amplitude(k, amplitude=args.amplitude, nu=nu, k=k_wave),
                    }
                )
                if k < args.steps:
                    f = step_fn(f, t)
        a_end, a_ref = series[-1]["A"], series[-1]["A_analytic"]
        out["cases"].append(
            {
                "case": "shear_wave_z",
                "n": n,
                "tau": tau,
                "nu_lattice": nu,
                "steps": args.steps,
                "A_measured": a_end,
                "A_analytic": a_ref,
                "rel_err": abs(a_end - a_ref) / a_ref,
                "series": series,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# stage: one real inversion (tau recovery) with wall clock
# --------------------------------------------------------------------------- #
def stage_inversion(args) -> dict:
    dtype = TORCH_DTYPE[args.dtype]
    torch.set_num_threads(args.threads)
    n, steps = args.n, args.steps
    u0 = divergence_free_ic(n, seed=args.seed, amplitude=args.amplitude, dtype=dtype)
    u_target = forward_u(n, u0, args.tau_true, steps, dtype)

    tau = torch.tensor(args.tau_init, dtype=dtype, requires_grad=True)
    f0 = ic_distribution(u0, dtype=dtype)
    lr = 1e-3
    best = None
    hist = []
    n_iter = args.inv_iters
    t_start = time.perf_counter()
    for it in range(n_iter):
        tau.grad = None
        f_T = segmented_rollout(f0, tau, steps, step_fn, seg_len=args.seg_len)
        J = velocity_loss(f_T, u_target)
        J.backward()
        g = float(tau.grad)
        step = lr if best is None else lr * max(1.0, 0.0)  # fixed step (deterministic)
        with torch.no_grad():
            tau -= step * tau.grad
        hist.append({"iter": it, "tau": float(tau), "J": float(J), "dJ_dtau": g})
        del f_T, J
    t_total = time.perf_counter() - t_start
    return {
        "stage": "inversion",
        "n": n,
        "steps": steps,
        "dtype": args.dtype,
        "threads": args.threads,
        "seg_len": args.seg_len,
        "tau_init": args.tau_init,
        "tau_true": args.tau_true,
        "tau_final": float(tau),
        "tau_abs_err": abs(float(tau) - args.tau_true),
        "iters": n_iter,
        "wall_s": t_total,
        "wall_per_iter_s": t_total / n_iter,
        "history": hist,
        "peak_rss_gib": peak_rss_gib(),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def provenance() -> dict:
    import torch as _t

    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        pass
    return {
        "commit": commit,
        "torch": _t.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cuda": _t.cuda.is_available(),
        "threads": _t.get_num_threads(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=["grad", "envelope", "physics", "inversion", "all"])
    ap.add_argument("--n", type=int, default=32, help="lattice resolution per axis")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--resolutions", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--dtype", default="float32", choices=list(TORCH_DTYPE))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.55, help="relaxation time where gradients are taken")
    ap.add_argument("--tau-true", type=float, default=0.60, help="reference tau generating the target field")
    ap.add_argument("--tau-init", type=float, default=0.52)
    ap.add_argument("--amplitude", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seg-len", type=int, default=2, help="steps per stored boundary state")
    ap.add_argument("--fd-step", type=float, default=1e-3)
    ap.add_argument("--fd-scan", type=float, nargs="+", default=[1e-3, 2e-3])
    ap.add_argument("--inv-iters", type=int, default=5)
    ap.add_argument("--threads-list", type=int, nargs="+", default=None)
    ap.add_argument("--engines", nargs="+", default=["plain", "step_ckpt", "segmented"])
    ap.add_argument("--store-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.store_dir is None:
        import tempfile

        args.store_dir = tempfile.mkdtemp(prefix="lbm3d_envelope_store_")

    result: dict = {"provenance": provenance(), "params": vars(args).copy()}
    stages = ["grad", "envelope", "physics", "inversion"] if args.stage == "all" else [args.stage]
    t0 = time.perf_counter()
    for st in stages:
        fn = {
            "grad": stage_grad,
            "envelope": stage_envelope,
            "physics": stage_physics,
            "inversion": stage_inversion,
        }[st]
        result[st] = fn(args)
        print(f"[{st}] done in {time.perf_counter() - t0:.1f}s", flush=True)
    result["wall_s"] = time.perf_counter() - t0
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"-> {out}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("provenance", "params")}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())