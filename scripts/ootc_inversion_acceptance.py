#!/usr/bin/env python3
"""GAP-PP-3-adjoint-ootc: close the inversion half of the anchor.

The first pass met the gradient gate (OOTC == BPTT bit-for-bit) but left
tau inversion UNFINISHED at |d tau|/tau = 20 % with a hand-tuned
steepest-descent driver.  This script produces the replacement evidence:

  A. unbiased objective   u_ref = u_T(tau_true)      -> recovered tau vs tau_true
  B. first-pass objective u_ref = 0.9 * u_T(tau_true) -> quantifies the
     objective-specification bias (the minimiser of J is *not* tau_true)
  C. forward-only central difference of dJ/dtau vs the OOTC adjoint
  D. peak memory of one OOTC gradient at two horizons (O(k), not O(T))
  E. J(tau) landscape + the inversion trajectory (for the figures)

Writes JSON to artifacts/adjoint_ootc_acceptance/ and PNGs to
artifacts/adjoint_ootc_acceptance/figures/.

Usage: python scripts/ootc_inversion_acceptance.py [--N 64] [--T 1000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm.adjoint_ootc import (  # noqa: E402
    fd_gradient,
    initial_pulse,
    invert_tau,
    ootc_gradient,
    reference_from_tau,
    terminal_objective,
)

OUT = os.path.join(REPO, "artifacts", "adjoint_ootc_acceptance")
TAU_TRUE, TAU0 = 0.8, 1.2


def peak_mem(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1024**3, dt


def landscape(N, T, tau_true, scale, npts=13):
    f0 = initial_pulse(N, N, amp=0.05, sigma=N / 8.0)
    ux_t, uy_t = reference_from_tau(tau_true, T, f0)
    ux_ref, uy_ref = scale * ux_t, scale * uy_t
    taus = np.linspace(0.6, 1.4, npts)
    Js = [terminal_objective(t, T, f0, ux_ref, uy_ref) for t in taus]
    return f0, ux_ref, uy_ref, taus.tolist(), Js


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    N, T, K = a.N, a.T, a.k
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    res = {"N": N, "T": T, "k": K, "tau_true": TAU_TRUE, "tau_start": TAU0}

    f0 = initial_pulse(N, N, amp=0.05, sigma=N / 8.0)
    ux_t, uy_t = reference_from_tau(TAU_TRUE, T, f0)

    # ---- C. adjoint vs forward-only FD ----
    dJ_adj, J0 = ootc_gradient(TAU0, T, f0, ux_t, uy_t, k=K)[:2]
    g_fd = fd_gradient(TAU0, T, f0, ux_t, uy_t, h=1e-4)
    res["gradient_check"] = {
        "tau": TAU0,
        "ootc_dJ_dtau": dJ_adj,
        "fd_dJ_dtau": g_fd,
        "J": J0,
        "rel_diff": abs(dJ_adj - g_fd) / abs(dJ_adj),
    }
    print(
        f"[C] dJ/dtau ootc={dJ_adj:.10e} fd={g_fd:.10e} "
        f"rel={res['gradient_check']['rel_diff']:.3e}",
        flush=True,
    )

    # ---- A. unbiased objective ----
    t0 = time.perf_counter()
    invA = invert_tau(TAU0, T, f0, ux_t, uy_t, k=K, verbose=True)
    invA["wall_s"] = time.perf_counter() - t0
    invA["rel_err_tau"] = abs(invA["tau"] - TAU_TRUE) / TAU_TRUE
    res["inversion_unbiased"] = {k: v for k, v in invA.items()}
    print(
        f"[A] tau*={invA['tau']:.12f} conv={invA['converged']} "
        f"rel={invA['rel_err_tau']:.3e} nfev={invA['nfev']} "
        f"{invA['wall_s']:.0f}s",
        flush=True,
    )

    # ---- B. first-pass objective (amplitude-scaled reference) ----
    ux_s, uy_s = 0.9 * ux_t, 0.9 * uy_t
    t0 = time.perf_counter()
    invB = invert_tau(TAU0, T, f0, ux_s, uy_s, k=K)
    invB["wall_s"] = time.perf_counter() - t0
    invB["rel_err_vs_tau_true"] = abs(invB["tau"] - TAU_TRUE) / TAU_TRUE
    # the minimiser of the biased objective (what a perfect solver would hit)
    grid = np.linspace(0.6, 1.2, 11)
    Jg = [terminal_objective(x, T, f0, ux_s, uy_s) for x in grid]
    i = int(np.argmin(Jg))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 11)
    Jf = [terminal_objective(x, T, f0, ux_s, uy_s) for x in fine]
    tau_min = float(fine[int(np.argmin(Jf))])
    invB["landscape_minimiser"] = tau_min
    invB["rel_err_vs_landscape_min"] = abs(invB["tau"] - tau_min) / tau_min
    res["inversion_scaled_reference"] = {k: v for k, v in invB.items()}
    print(
        f"[B] tau={invB['tau']:.12f} conv={invB['converged']} "
        f"tau_min={tau_min:.6f} err_vs_min={invB['rel_err_vs_landscape_min']:.3e} "
        f"err_vs_true={invB['rel_err_vs_tau_true']:.3e}",
        flush=True,
    )

    # ---- E. landscape of the unbiased objective ----
    taus = np.linspace(0.6, 1.4, 9)
    Jl = [terminal_objective(x, T, f0, ux_t, uy_t) for x in taus]
    gl = [fd_gradient(x, T, f0, ux_t, uy_t, h=1e-4) for x in taus]
    res["landscape_unbiased"] = {"tau": taus.tolist(), "J": Jl, "dJ_dtau": gl}
    print(f"[E] landscape {len(taus)} points done", flush=True)

    # ---- D. peak memory vs horizon ----
    mem = []
    for Th in (T // 2, T):
        _, peak, dt = peak_mem(lambda Th=Th: ootc_gradient(TAU0, Th, f0, ux_t, uy_t, k=K))
        mem.append({"T": Th, "peak_gib": peak, "wall_s": dt})
        print(f"[D] T={Th} peak={peak:.4f} GiB wall={dt:.1f}s", flush=True)
    res["memory"] = mem

    with open(os.path.join(OUT, "inversion_convergence.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[done] -> {os.path.join(OUT, 'inversion_convergence.json')}")


if __name__ == "__main__":
    main()
