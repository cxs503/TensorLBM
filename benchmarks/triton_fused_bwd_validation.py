"""Acceptance harness for GAP-PP-1-triton-bwd (differentiable fused Triton step).

Runs the three registered criteria of the gap ledger and writes one JSON
artifact plus one figure:

1. **gradient parity** — ``d(loss)/d(tau)`` of a 100-step fused-Triton rollout
   versus the same 100-step rollout through eager torch autograd
   (:func:`tensorlbm.triton_fused_diff.eager_stream_collide_step`), relative
   difference < 1e-5;
2. **finite-difference cross-check** — central difference of the *same* loss
   in fp64 (eager reference, ``h = 1e-4``), relative difference < 1e-4;
3. **throughput** — fused Triton step vs eager step.  Only meaningful on a
   CUDA device: Triton's ``TRITON_INTERPRET=1`` NumPy interpreter executes the
   identical kernel source on CPU at interpreter (not compiled) speed, so the
   harness records the gate as ``not_measurable`` there instead of inventing a
   speed-up.

Usage
-----
::

    # CPU (interpreter; correctness gates only)
    TRITON_INTERPRET=1 PYTHONPATH=src python benchmarks/triton_fused_bwd_validation.py

    # GPU (compiled kernels; all three gates)
    PYTHONPATH=src python benchmarks/triton_fused_bwd_validation.py --device cuda

The JSON artifact lands in ``runs/m15355/gap_pp1_triton_bwd/verify.json`` by
default (``--out`` overrides), the figure next to it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.triton_fused_diff import (
    eager_stream_collide_step,
    interpreter_mode,
    macroscopic,
    triton_fused_step,
)


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


def initial_field(n: int, u_amp: float = 0.05, seed: int = 0) -> torch.Tensor:
    """D3Q19 equilibrium on a periodic ``n^3`` box with a small random velocity.

    ``rho = 1``, ``|u| ~ u_amp`` (incompressible LBM limit, Mach ~ 0.09), which
    keeps the 100-step rollout in the stable BGK regime for ``tau >= 0.55``.
    """
    torch.manual_seed(seed)
    rho = torch.ones((n, n, n), dtype=torch.float32)
    u = u_amp * torch.randn((3, n, n, n), dtype=torch.float32)
    return equilibrium3d(rho, u[0], u[1], u[2]).to(torch.float32)


def step_loss(f: torch.Tensor) -> torch.Tensor:
    """Per-step observable: scaled mean square of the streamwise velocity."""
    _, u = macroscopic(f)
    return (u[0] ** 2).mean() * 100.0


def trajectory_loss(states) -> torch.Tensor:
    """Trajectory-accumulated objective ``(1/N) sum_t L(f_t)``.

    This is the objective differentiable-simulation papers actually optimise.
    A *terminal* loss is useless here: unforced periodic BGK decays to the
    uniform state, so by step ~50 the terminal ``dL/dtau`` is O(1e-8) - i.e.
    below fp32 roundoff of the loss (|L| ~ 1e-7), and the relative-difference
    gate would be measuring noise rather than the adjoint.  Accumulating over
    the trajectory keeps the early-time sensitivity (which does not decay) in
    the gradient, as in Um et al. 2020 / XLB's differentiable rollouts.
    """
    total = None
    for f in states:
        term = step_loss(f)
        total = term if total is None else total + term
    return total / len(states)


def rollout_eager(f0: torch.Tensor, tau, steps: int, dtype=None):
    """Eager rollout returning the list of states (for the trajectory loss)."""
    f = f0 if dtype is None else f0.to(dtype)
    states = []
    for _ in range(int(steps)):
        f = eager_stream_collide_step(f, tau)
        states.append(f)
    return states


def fd_gradient(f0: torch.Tensor, tau: float, steps: int, h: float = 1e-4) -> float:
    """Central difference of the trajectory loss in fp64 (eager reference)."""
    f64 = f0.to(torch.float64)

    def L(t: float) -> float:
        return float(trajectory_loss(rollout_eager(f64, t, steps)))

    return (L(tau + h) - L(tau - h)) / (2.0 * h)


# ---------------------------------------------------------------------------
# Measurement blocks
# ---------------------------------------------------------------------------


@dataclass
class Case:
    n: int
    tau: float
    steps: int
    block: int
    rel_grad: float = 0.0
    grad_triton: float = 0.0
    grad_eager: float = 0.0
    rel_fd: float = 0.0
    fd: float = 0.0
    loss: float = 0.0
    fwd_rel_1step: float = 0.0
    t_triton_fwd_s: float = 0.0
    t_triton_bwd_s: float = 0.0
    t_eager_fwd_s: float = 0.0
    t_eager_bwd_s: float = 0.0


def run_case(n: int, tau: float, steps: int, block: int, device: str) -> Case:
    c = Case(n=n, tau=tau, steps=steps, block=block)
    f0 = initial_field(n).to(device)

    # --- forward parity, single step (kernel vs eager primitive chain)
    ft = triton_fused_step(f0, tau, block=block)
    fe = eager_stream_collide_step(f0, tau)
    c.fwd_rel_1step = float((ft - fe).abs().max() / fe.abs().max())

    # --- Triton rollout + backward (time both phases)
    tau_t = torch.tensor(tau, requires_grad=True)
    t0 = time.perf_counter()
    f = f0
    states = []
    for _ in range(steps):
        f = triton_fused_step(f, tau_t, block=block)
        states.append(f)
    t1 = time.perf_counter()
    L = trajectory_loss(states)
    L.backward()
    t2 = time.perf_counter()
    c.t_triton_fwd_s = t1 - t0
    c.t_triton_bwd_s = t2 - t1
    c.grad_triton = float(tau_t.grad)
    c.loss = float(L)

    # --- eager rollout + backward (reference)
    tau_e = torch.tensor(tau, requires_grad=True)
    t0 = time.perf_counter()
    states_ref = rollout_eager(f0, tau_e, steps)
    t1 = time.perf_counter()
    L_ref = trajectory_loss(states_ref)
    L_ref.backward()
    t2 = time.perf_counter()
    c.t_eager_fwd_s = t1 - t0
    c.t_eager_bwd_s = t2 - t1
    c.grad_eager = float(tau_e.grad)
    c.rel_grad = abs(c.grad_triton - c.grad_eager) / abs(c.grad_eager)

    # --- finite-difference cross-check (fp64)
    c.fd = fd_gradient(f0, tau, steps)
    c.rel_fd = abs(c.grad_triton - c.fd) / abs(c.fd)
    return c


def step_count_sweep(n: int, tau: float, counts, block: int, device: str):
    """Gradient parity vs rollout length (convergence / sensitivity table)."""
    rows = []
    for steps in counts:
        rows.append(run_case(n, tau, steps, block, device))
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(cases, sweep, path: Path, throughput_ok: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    steps = [c.steps for c in sweep]
    ax.semilogy(steps, [max(c.rel_grad, 1e-16) for c in sweep], "o-", label="|dL/dtau Triton - eager| / eager")
    ax.semilogy(steps, [max(c.rel_fd, 1e-16) for c in sweep], "s--", label="|dL/dtau Triton - FD(fp64)| / FD")
    ax.axhline(1e-5, color="C3", lw=1, ls=":", label="gate 1e-5 (eager)")
    ax.axhline(1e-4, color="C4", lw=1, ls=":", label="gate 1e-4 (FD)")
    ax.set_xlabel("rollout length (fused Triton steps)")
    ax.set_ylabel("relative difference")
    ax.set_title(f"adjoint accuracy, n={sweep[0].n}^3, tau={sweep[0].tau}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = [f"n={c.n}\ntau={c.tau}\n{c.steps} steps" for c in cases]
    x = range(len(cases))
    tw = [c.t_triton_fwd_s + c.t_triton_bwd_s for c in cases]
    ew = [c.t_eager_fwd_s + c.t_eager_bwd_s for c in cases]
    ax.bar([i - 0.2 for i in x], tw, width=0.4, label="Triton fwd+bwd")
    ax.bar([i + 0.2 for i in x], ew, width=0.4, label="eager fwd+bwd")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("wall time [s]")
    mode = "compiled GPU kernels" if throughput_ok else "TRITON_INTERPRET=1 (CPU, NOT a throughput claim)"
    ax.set_title(f"per-rollout wall time - {mode}", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--block", type=int, default=0, help="tile size; 0 -> n")
    ap.add_argument("--taus", default="0.8,0.6")
    ap.add_argument("--sweep", default="2,5,10,20,50")
    ap.add_argument("--out", default="runs/m15355/gap_pp1_triton_bwd/verify.json")
    args = ap.parse_args(argv)

    block = args.block or args.n
    taus = [float(t) for t in args.taus.split(",")]
    sweep_counts = [int(s) for s in args.sweep.split(",") if int(s) < args.steps]

    cases = []
    # headline: 100-step criterion at the two tau parameter points, plus a
    # second resolution at a shorter rollout (cost control under the
    # interpreter).
    for tau in taus:
        cases.append(run_case(args.n, tau, args.steps, block, args.device))
    if args.n < 8:
        cases.append(run_case(8, taus[0], 20, 8, args.device))
    sweep = step_count_sweep(args.n, taus[0], sweep_counts, block, args.device)

    throughput_ok = bool(torch.cuda.is_available()) and not interpreter_mode()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "gap": "GAP-PP-1-triton-bwd",
        "mission": 15355,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "triton_interpret": interpreter_mode(),
            "device_used": args.device,
            "kernel_execution_mode": (
                "compiled Triton kernels on CUDA"
                if throughput_ok
                else "TRITON_INTERPRET=1 NumPy interpreter on CPU (same kernel source, not compiled)"
            ),
        },
        "criteria": {
            "grad_rel_vs_eager_1e-5": {
                "gate": 1e-5,
                "worst": max(c.rel_grad for c in cases if c.steps >= 2),
                "pass": bool(max(c.rel_grad for c in cases if c.steps >= 2) < 1e-5),
            },
            "fd_rel_1e-4": {
                "gate": 1e-4,
                "worst": max(c.rel_fd for c in cases),
                "pass": bool(max(c.rel_fd for c in cases) < 1e-4),
            },
            "throughput_ge_2x_eager": {
                "gate": 2.0,
                "measured": throughput_ok,
                "status": (
                    "measured" if throughput_ok else "not_measurable_on_cpu_interpreter"
                ),
                "reason": None
                if throughput_ok
                else (
                    "no CUDA device on this host; TRITON_INTERPRET=1 runs the same "
                    "kernel source through Triton's NumPy interpreter at "
                    "interpreter speed, which says nothing about compiled "
                    "throughput. Re-run with --device cuda on a GPU host."
                ),
            },
        },
        "cases": [asdict(c) for c in cases],
        "step_sweep": [asdict(c) for c in sweep],
    }
    fig_path = out.parent / "gap_pp1_triton_bwd.png"
    make_figure(cases, sweep, fig_path, throughput_ok)
    payload["figure"] = str(fig_path)
    out.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload["criteria"], indent=2))
    print("artifact:", out)
    print("figure:", fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())