"""Differentiable LBM: gradient-based optimisation through the eager solver.

Demonstrates the *differentiable reference path* of TensorLBM: the plain
eager 2-D solver (``tensorlbm.solver.stream`` + ``collide_bgk``, gather-based
streaming, tensorised BGK collision) is differentiable end to end, so a
scalar loss computed after N time steps can be optimised directly against
solver parameters or the initial distribution — no adjoint hand-derivation,
no frozen-field surrogate (see ``docs/differentiable_path.md`` and
``tests/test_autograd.py``).

Two modes
---------
``--mode tau`` (default) — viscosity inverse problem. A decaying shear wave
u = U0*sin(ky)*exp(-nu*k^2*t) is rolled out for ``--steps`` steps with a
hidden tau* = 0.9; the observed final velocity field is the target. SGD
starting from tau = 0.55 recovers tau* through the solver.

``--mode initial-field`` — initial-condition shaping (the XLB demo). The
initial distribution f_0 is optimised so that, after ``--steps`` steps, the
normalised density matches a target pattern (circle); f_0 is clipped to the
physical range [0.01*w_i, 10*w_i] exactly like the original example.

Activation memory vs steps (the practical cost of backprop-through-time)
-------------------------------------------------------------------------
Measured on one RTX 5090, 256x256 D2Q9 fp32 grid, full forward+backward of
the tau-mode loss, standard autograd:

    steps N | plain autograd | per-step checkpoint(use_reentrant=False)
    --------+----------------+----------------------------------------
       10   |   138.4 MiB    |   54.4 MiB   (2.5x less)
       50   |   648.4 MiB    |  144.4 MiB   (4.5x less)
      200   |  2560.9 MiB    |  481.9 MiB   (5.3x less)

Plain autograd grows linearly (~12.8 MiB/step = the saved collide/stream
intermediates); per-step checkpointing keeps a near-constant floor plus a
small per-segment term and produces bit-identical gradients (relative
difference 0.0 in this measurement). Re-run with ``--measure-memory``.

Attribution
-----------
Adapted from Autodesk XLB ``examples/cfd/differentiable_lbm.py``
(Apache License 2.0, Copyright 2023 Autodesk Inc.;
https://github.com/Autodesk/XLB). Changes for TensorLBM:

* ported from JAX/XLB to the TensorLBM eager PyTorch solver
  (``collide_bgk``/``stream`` instead of ``IncompressibleNavierStokesStepper``);
* added the ``tau`` mode (relaxation-time recovery, a viscosity inverse
  problem) alongside the original initial-condition shaping mode;
* ``torch.func.grad_and_value`` replaces ``jax.value_and_grad`` (when
  gradient checkpointing is enabled, standard ``loss.backward()`` is used
  instead: ``torch.func`` transforms do not support checkpoint's saved-tensor
  hooks);
* added activation-memory measurement (``--measure-memory``).

Usage
-----
    python examples/differentiable_lbm.py                      # tau recovery, 50 steps
    python examples/differentiable_lbm.py --mode initial-field # XLB-style shaping
    python examples/differentiable_lbm.py --measure-memory     # memory table only
    python examples/differentiable_lbm.py --checkpoint --iters 60
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.d2q9 import W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Forward model: the plain eager solver, optionally gradient-checkpointed
# ---------------------------------------------------------------------------


def shear_wave_f0(
    n: int, amplitude: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """D2Q9 equilibrium initialisation of u = a*sin(ky), v = 0 (periodic)."""
    y, _x = torch.meshgrid(
        torch.arange(n, dtype=dtype, device=device),
        torch.arange(n, dtype=dtype, device=device),
        indexing="ij",
    )
    ux = amplitude * torch.sin(TWO_PI / n * y)
    return equilibrium(torch.ones_like(ux), ux, torch.zeros_like(ux))


def rollout(
    f0: torch.Tensor,
    tau: torch.Tensor,
    steps: int,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """N-step stream->collide rollout on the differentiable reference path.

    ``use_checkpoint`` wraps every step in ``torch.utils.checkpoint`` so the
    backward pass recomputes instead of storing all per-step activations
    (trade-off quantified in the module docstring).
    """

    def one_step(f: torch.Tensor) -> torch.Tensor:
        return collide_bgk(stream(f), tau)

    f = f0
    for _ in range(steps):
        if use_checkpoint:
            f = torch.utils.checkpoint.checkpoint(one_step, f, use_reentrant=False)
        else:
            f = one_step(f)
    return f


# ---------------------------------------------------------------------------
# Mode 1: tau recovery (viscosity inverse problem)
# ---------------------------------------------------------------------------


class TauRecovery:
    """Recover the relaxation time of a decaying shear wave by SGD."""

    def __init__(self, grid: int, steps: int, tau_star: float, device, dtype):
        self.steps = steps
        self.tau_star = tau_star
        self.f0 = shear_wave_f0(grid, 0.05, device, dtype)
        with torch.no_grad():
            self.target = macroscopic(rollout(self.f0, torch.tensor(tau_star, device=device, dtype=dtype), steps))[1]

    def loss(self, tau: torch.Tensor) -> torch.Tensor:
        ux = macroscopic(rollout(self.f0, tau, self.steps))[1]
        return ((ux - self.target) ** 2).mean()

    def run(
        self,
        tau_init: float,
        lr: float,
        iters: int,
        device,
        dtype,
        use_checkpoint: bool = False,
        log_every: int = 10,
    ) -> list[float]:
        tau = torch.tensor(tau_init, device=device, dtype=dtype)
        print(
            f"tau recovery: grid={self.f0.shape[-1]} steps={self.steps} "
            f"tau*={self.tau_star} tau0={tau_init} lr={lr} iters={iters} "
            f"dtype={dtype} checkpoint={use_checkpoint}"
        )
        losses: list[float] = []
        t0 = time.time()
        for it in range(iters):
            if use_checkpoint:
                # torch.func transforms don't support checkpoint's saved-tensor
                # hooks -> standard autograd on a fresh leaf parameter
                tau_leaf = tau.detach().requires_grad_(True)
                loss = self.loss(tau_leaf)
                grad = torch.autograd.grad(loss, tau_leaf)[0]
            else:
                from torch.func import grad_and_value

                grad, loss = grad_and_value(self.loss)(tau)
            with torch.no_grad():
                tau = tau - lr * grad
                tau.clamp_(0.5005, 5.0)  # physical range: nu = (tau - 0.5)/3 > 0
            losses.append(float(loss.detach()))
            if it % log_every == 0 or it == iters - 1:
                print(
                    f"  iter {it:4d}  loss={losses[-1]:.6e}  tau={float(tau):.5f}",
                    flush=True,
                )
        print(
            f"converged: tau={float(tau):.5f} (target {self.tau_star}, "
            f"error {abs(float(tau) - self.tau_star) / self.tau_star:.2%}), "
            f"loss {losses[0]:.3e} -> {losses[-1]:.3e} in {time.time() - t0:.1f}s"
        )
        return losses


# ---------------------------------------------------------------------------
# Mode 2: initial-condition shaping (port of the XLB demo)
# ---------------------------------------------------------------------------


class InitialFieldShaping:
    """Optimise f_0 so the evolved normalised density matches a target pattern."""

    def __init__(self, grid: int, steps: int, device, dtype, coverage: float = 0.05):
        self.steps = steps
        self.rho_bg, self.rho_var = 1.0, 0.2
        n = grid
        y, x = torch.meshgrid(
            torch.arange(n, dtype=dtype, device=device),
            torch.arange(n, dtype=dtype, device=device),
            indexing="ij",
        )
        radius2 = coverage * n * n / math.pi
        self.target = (((x - n / 2) ** 2 + (y - n / 2) ** 2) < radius2).to(dtype)
        self.w = W.to(device=device, dtype=dtype).view(9, 1, 1)
        self.tau = torch.tensor(1.0, device=device, dtype=dtype)
        self.f_init = equilibrium(
            torch.full((n, n), self.rho_bg - self.rho_var, dtype=dtype, device=device),
            torch.zeros(n, n, dtype=dtype, device=device),
            torch.zeros(n, n, dtype=dtype, device=device),
        ).detach()

    def _normalise(self, rho: torch.Tensor) -> torch.Tensor:
        return ((rho - (self.rho_bg - self.rho_var)) / (2 * self.rho_var)).clamp(0.0, 1.0)

    def loss(self, f0: torch.Tensor) -> torch.Tensor:
        rho, _ux, _uy = macroscopic(rollout(f0, self.tau, self.steps))
        return ((self._normalise(rho) - self.target) ** 2).mean()

    def run(
        self, lr: float, iters: int, device, dtype, log_every: int = 20
    ) -> list[float]:
        f0 = self.f_init.clone()
        print(
            f"initial-field shaping: grid={f0.shape[-1]} steps={self.steps} "
            f"lr={lr} iters={iters} dtype={dtype}"
        )
        losses: list[float] = []
        t0 = time.time()
        for it in range(iters):
            f_leaf = f0.detach().requires_grad_(True)
            loss = self.loss(f_leaf)
            grad = torch.autograd.grad(loss, f_leaf)[0]
            with torch.no_grad():
                # plain gradient descent + physical-range clip (as in the XLB demo)
                f0 = f0 - lr * grad
                f0.clamp_(0.01 * self.w, 10.0 * self.w)
            losses.append(float(loss.detach()))
            if it % log_every == 0 or it == iters - 1:
                print(f"  iter {it:4d}  loss={losses[-1]:.6e}", flush=True)
        print(f"final loss {losses[-1]:.6e} in {time.time() - t0:.1f}s")
        return losses


# ---------------------------------------------------------------------------
# Activation-memory measurement
# ---------------------------------------------------------------------------


def measure_memory(grid: int, step_counts: list[int], device) -> list[dict]:
    """Peak CUDA memory of one forward+backward of the tau-mode loss."""
    dtype = torch.float32
    rows = []
    for n_steps in step_counts:
        f0 = shear_wave_f0(grid, 0.05, device, dtype)
        with torch.no_grad():
            target = macroscopic(rollout(f0, torch.tensor(0.9, device=device), n_steps))[1]

        for use_ckpt in (False, True):
            tau = torch.tensor(0.6, device=device, requires_grad=True)

            def loss_of(t: torch.Tensor) -> torch.Tensor:
                return ((macroscopic(rollout(f0, t, n_steps, use_ckpt))[1] - target) ** 2).mean()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loss_of(tau).backward()
            peak_mib = torch.cuda.max_memory_allocated() / 2**20
            grad = tau.grad.item()
            rows.append(
                {
                    "grid": grid,
                    "steps": n_steps,
                    "checkpoint": use_ckpt,
                    "peak_MiB": round(peak_mib, 1),
                    "dL/dtau": grad,
                }
            )
            del tau
    print(f"activation memory, {grid}x{grid} D2Q9 fp32 (forward+backward):")
    print(f"  {'steps':>5} | {'plain':>10} | {'checkpoint':>10} | plain/ckpt")
    for i in range(0, len(rows), 2):
        plain, ckpt = rows[i], rows[i + 1]
        assert plain["steps"] == ckpt["steps"]
        print(
            f"  {plain['steps']:5d} | {plain['peak_MiB']:9.1f} | {ckpt['peak_MiB']:9.1f} | "
            f"{plain['peak_MiB'] / ckpt['peak_MiB']:.2f}x"
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["tau", "initial-field"], default="tau")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50, help="forward steps per loss eval")
    ap.add_argument("--iters", type=int, default=100, help="optimisation iterations")
    ap.add_argument("--lr", type=float, default=None, help="default: 2000 (tau) / 1.0 (field)")
    ap.add_argument("--tau-star", type=float, default=0.9)
    ap.add_argument("--tau-init", type=float, default=0.55)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", action="store_true", help="per-step gradient checkpointing")
    ap.add_argument("--measure-memory", action="store_true", help="print memory table and exit")
    ap.add_argument("--mem-grid", type=int, default=256)
    ap.add_argument("--out", default=None, help="output dir for convergence CSV / memory JSON")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.measure_memory:
        if device.type != "cuda":
            raise SystemExit("--measure-memory requires CUDA (peak allocator stats)")
        rows = measure_memory(args.mem_grid, [10, 50, 200], device)
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "memory_measurement.json").write_text(json.dumps(rows, indent=2))
            print(f"-> {out / 'memory_measurement.json'}")
        return

    if args.mode == "tau":
        lr = args.lr if args.lr is not None else 2000.0
        problem = TauRecovery(args.grid, args.steps, args.tau_star, device, dtype)
        losses = problem.run(
            args.tau_init, lr, args.iters, device, dtype, use_checkpoint=args.checkpoint
        )
    else:
        lr = args.lr if args.lr is not None else 1.0
        problem = InitialFieldShaping(args.grid, args.steps, device, dtype)
        losses = problem.run(lr, args.iters, device, dtype)

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"convergence_{args.mode}.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["iter", "loss"])
            writer.writerows(enumerate(losses))
        print(f"-> {out / f'convergence_{args.mode}.csv'}")


if __name__ == "__main__":
    main()
