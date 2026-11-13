#!/usr/bin/env python
"""Lightweight production of the opt-loop convergence JSON artifact.

Runs the twin16 track at n=8 (fast) with 3 seeds to demonstrate the
end-to-end pipeline and produce a machine-readable convergence JSON.
This is the acceptance artifact for GAP-PP-4-opt-loop.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

# Ensure src is on the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorlbm.inverse import (
    ConvergenceLog,
    default_relative_mse,
    jittered_init,
    optimise_loop,
    seed_everything,
)
from tensorlbm.opt_loop_tau_recovery import (
    TaylorGreenCase,
    twin_observations,
    simulate,
    TAU_MIN,
    TAU_MAX,
)

DTYPE = torch.float64
SEEDS = (0, 1, 2)
TAU_TRUE = 0.55
ITERS = 20
LR = 0.05
POLISH = 10


def run_single(seed: int) -> dict:
    """Run one seed of the twin experiment."""
    seed_everything(seed)
    case = TaylorGreenCase(n=8, u0=0.05, nz=1, dtype=DTYPE)
    nu_true = case.nu_from_tau(TAU_TRUE)
    times = case.times(nu_true, decay=0.6, n_obs=3)
    obs = twin_observations(case, TAU_TRUE, times, seed=seed)

    # Jitter nu directly (not tau) to avoid negative nu from tau < 0.5
    nu_init = jittered_init(nu_true, seed=seed, relative_range=0.25)
    log_nu = torch.nn.Parameter(
        torch.tensor(math.log(max(nu_init, 1e-6)), dtype=DTYPE, requires_grad=True)
    )
    amplitude = torch.nn.Parameter(torch.tensor(1.0, dtype=DTYPE, requires_grad=True))

    log = ConvergenceLog(problem="tg_tau_recovery", seed=seed)

    def forward_fn():
        tau = 0.5 + 3.0 * torch.exp(log_nu)
        sim = simulate(case, tau, obs.times, amplitude=amplitude)
        return sim, obs

    lo = math.log((TAU_MIN - 0.5) / 3.0)
    hi = math.log((TAU_MAX - 0.5) / 3.0)

    result = optimise_loop(
        params=[log_nu, amplitude],
        param_names=["log_nu", "amplitude"],
        loss_fn=default_relative_mse(),
        forward_fn=forward_fn,
        iters=ITERS,
        lr=LR,
        polish=POLISH,
        box_constraints={"log_nu": (lo, hi)},
        convergence_log=log,
        problem_name="tg_tau_recovery",
        seed=seed,
    )

    nu_hat = math.exp(float(log_nu.detach()))
    tau_hat = 0.5 + 3.0 * nu_hat
    re_hat = case.reynolds(nu_hat)
    rel_err = 100.0 * (nu_hat - nu_true) / nu_true

    # Write per-seed convergence JSON
    log_path = Path("artifacts") / f"opt_loop_seed{seed}.json"
    log.write(log_path)

    return {
        "seed": seed,
        "tau_true": TAU_TRUE,
        "nu_true": nu_true,
        "re_true": case.reynolds(nu_true),
        "tau_hat": tau_hat,
        "nu_hat": nu_hat,
        "re_hat": re_hat,
        "amplitude_hat": float(amplitude.detach()),
        "rel_err_pct": rel_err,
        "converged": result.converged,
        "final_loss": result.final_loss,
        "n_iterations": result.n_iterations,
        "wall_s": round(result.wall_s, 3),
        "convergence_json": str(log_path),
    }


def main():
    t0 = time.time()
    results = []
    for seed in SEEDS:
        r = run_single(seed)
        results.append(r)
        print(f"seed={seed}: tau_hat={r['tau_hat']:.6f} Re_hat={r['re_hat']:.4f} "
              f"err={r['rel_err_pct']:+.4f}% converged={r['converged']}")

    all_pass = all(abs(r["rel_err_pct"]) < 3.0 for r in results)
    payload = {
        "schema": "tensorlbm/opt-loop-tau-recovery/v1",
        "created_unix": time.time(),
        "torch_version": torch.__version__,
        "seeds": list(SEEDS),
        "config": {
            "n": 8,
            "u0": 0.05,
            "tau_true": TAU_TRUE,
            "iters": ITERS,
            "lr": LR,
            "polish": POLISH,
            "dtype": "float64",
        },
        "runs": results,
        "summary": {
            "all_under_3pct": all_pass,
            "max_abs_rel_err_pct": max(abs(r["rel_err_pct"]) for r in results),
            "tau_hat_mean": sum(r["tau_hat"] for r in results) / len(results),
            "re_hat_mean": sum(r["re_hat"] for r in results) / len(results),
            "wall_s_total": round(time.time() - t0, 3),
            "criterion": "TG decay tau/Re recovery < 3%; 3 seeds reproducible; machine-readable convergence JSON",
        },
    }

    out = Path("artifacts") / "opt_loop_tau_recovery.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out}")
    print(f"All under 3%: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
