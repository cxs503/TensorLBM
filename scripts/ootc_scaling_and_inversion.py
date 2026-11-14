"""Scaling + inversion evidence for the OOTC adjoint (GAP-PP-3-adjoint-ootc).

Produces, at a local-affordable scale (single core, minutes):

  (a) memory scaling with T: peak traced memory of the full-storage BPTT
      vs the OOTC (k=100) variant, plus the analytic O(T*N*Q) line --
      shows the OOTC peak is flat in T;
  (b) checkpoint-interval sweep: |dJ_ootc(k) - dJ_bptt| / |dJ_bptt| --
      shows the k-independence of the gradient (the OOTC is exact, not
      approximate, because the recomputation is bit-identical);
  (c) a tau inversion driven by the OOTC gradient: J(tau) landscape,
      gradient-descent trajectory, and the recovered tau.

All numbers are written to JSON (no hand-copied values in the report).

Usage:
  python3 scripts/ootc_scaling_and_inversion.py \
      --out artifacts/adjoint_ootc_acceptance/scaling_inversion.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "TensorLBM", "src"))

from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro  # noqa: E402
from tensorlbm.adjoint.bptt import bptt_gradient  # noqa: E402
from tensorlbm.adjoint.ootc import ootc_gradient  # noqa: E402

N = 64
TAU = 1.0
AMP = 0.05
K = 100
TAU_TRUE = 0.8
TAU_START = 1.2
T_INV = 1000


def _problem(T: int, tau: float = TAU, tau_ref: float | None = None):
    f0 = initial_pulse(N, N, amp=AMP, sigma=N / 8.0)
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau_ref if tau_ref is not None else tau)
    _, ux_t, uy_t = macro(f)
    return f0, ux_t * 0.9, uy_t * 0.9


def _peak(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1024 ** 3, dt


def memory_scaling(Ts):
    rows = []
    for T in Ts:
        f0, ux_ref, uy_ref = _problem(T)
        _, peak_bptt, t_bptt = _peak(
            lambda: bptt_gradient(TAU, T, f0, ux_ref, uy_ref))
        _, peak_ootc, t_ootc = _peak(
            lambda: ootc_gradient(TAU, T, f0, ux_ref, uy_ref, k=K))
        rows.append({
            "T": T, "N": N,
            "bptt_peak_gib": peak_bptt, "ootc_peak_gib": peak_ootc,
            "analytic_bptt_gib": T * f0.size * 8 / 1024 ** 3,
            "analytic_ootc_gib": (K + 3) * f0.size * 8 / 1024 ** 3,
            "bptt_time_s": t_bptt, "ootc_time_s": t_ootc,
        })
        print(f"[mem] T={T:5d} bptt={peak_bptt:.3f} GiB ootc={peak_ootc:.3f} GiB "
              f"t_bptt={t_bptt:.1f}s t_ootc={t_ootc:.1f}s", flush=True)
    return rows


def k_sweep(T=1000, ks=(10, 25, 50, 100, 250, 500, 1000)):
    f0, ux_ref, uy_ref = _problem(T)
    dJ_ref, J_ref, _, _ = bptt_gradient(TAU, T, f0, ux_ref, uy_ref)
    rows = []
    for k in ks:
        dJ, J, _, n_ckpt, ckpt = ootc_gradient(TAU, T, f0, ux_ref, uy_ref, k=k)
        rows.append({"k": k, "dJ": dJ, "J": J,
                     "rel_err": abs(dJ - dJ_ref) / abs(dJ_ref),
                     "ckpt_gib": os.path.getsize(ckpt) / 1024 ** 3})
        os.unlink(ckpt)
        print(f"[k] k={k:5d} rel={rows[-1]['rel_err']:.3e}", flush=True)
    return {"T": T, "dJ_ref": dJ_ref, "J_ref": J_ref, "rows": rows}


def inversion(n_iter=12):
    """Gradient descent on tau using the OOTC gradient (J = 1/2|u_T-u_ref|^2)."""
    f0, ux_ref, uy_ref = _problem(T_INV, tau=TAU_TRUE)  # reference from tau_true
    # objective landscape
    taus = np.linspace(0.6, 1.4, 17)
    landscape = []
    for tau in taus:
        f = f0.copy()
        for _ in range(T_INV):
            f = step(f, tau)
        _, ux, uy = macro(f)
        landscape.append(0.5 * float(np.sum((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2)))
        print(f"[landscape] tau={tau:.3f} J={landscape[-1]:.6e}", flush=True)

    def J_of(tau):
        f = f0.copy()
        for _ in range(T_INV):
            f = step(f, tau)
        _, ux, uy = macro(f)
        return 0.5 * float(np.sum((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2))

    tau = TAU_START
    hist = []
    for it in range(n_iter):
        dJ, J, _, _, _ = ootc_gradient(tau, T_INV, f0, ux_ref, uy_ref, k=K)
        step_dir = -1.0 if dJ > 0 else 1.0
        alpha = 2e-2
        ok = False
        for _ in range(25):
            cand = tau + step_dir * alpha
            if 0.55 < cand < 1.45 and J_of(cand) < J:
                ok = True
                break
            alpha *= 0.5
        hist.append({"iter": it, "tau": tau, "J": J, "dJ_dtau": dJ,
                     "alpha": alpha, "step_taken": bool(ok)})
        print(f"[inv] it={it:2d} tau={tau:.6f} J={J:.6e} dJ={dJ:+.6e} "
              f"alpha={alpha:.2e}", flush=True)
        if not ok:
            break
        tau = tau + step_dir * alpha
    return {"tau_true": TAU_TRUE, "tau_start": TAU_START, "T": T_INV, "k": K,
            "landscape": {"tau": taus.tolist(), "J": landscape},
            "history": hist,
            "tau_recovered": tau,
            "abs_err_tau": abs(tau - TAU_TRUE)}


def main() -> None:
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(
        REPO, "artifacts", "adjoint_ootc_acceptance",
        "scaling_inversion.json"))
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    N = a.N
    skip = {s for s in a.skip.split(",") if s}
    res = {"N": N, "tau": TAU, "k": K, "amp": AMP}
    if "mem" not in skip:
        res["memory_scaling"] = memory_scaling([250, 500, 1000, 2000])
    if "k" not in skip:
        res["k_sweep"] = k_sweep()
    if "inv" not in skip:
        res["inversion"] = inversion()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()