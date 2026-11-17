#!/usr/bin/env python3
"""Full-scale tau inversion for GAP-PP-3-adjoint-ootc (N=128, T=10^4).

The acceptance wording of the anchor is "10^4 步 128^2 τ 反演峰值内存 < 8 GiB".
Prior evidence covered:

  * gradient + memory gates at full scale (mission 20040, cj2481) -- but the
    inversion *driver* was only run at N=64/T=1000 (not converged, first-pass
    driver) and later N=32/T=600 with the converging driver (mission 22781);
  * the converging bracketed-Newton inversion, small scale only.

This script closes the remaining half: the converging ``invert_tau`` driver
(unbiased objective u_ref = u_T(tau_true)) run at the acceptance scale, with

  A. gradient cross-check  OOTC adjoint vs forward central difference
  B. the full-scale inversion itself (bracketed Newton / Illinois)
  C. lean J(tau) landscape (forward only) around tau_true for the figure
  D. terminal-field + costate-field snapshot (return_field=True) for the
     report's field plots
  E. peak-memory evidence: VmHWM of this single process, recorded after
     each stage (the OOTC checkpointed gradient is the only memory-heavy
     component; the gate is the final VmHWM < 8 GiB)

Everything numeric is written to artifacts/adjoint_ootc_acceptance/
(in-repo: a job's /tmp is not part of the fetched artifact -- the cj1625
lesson).  Heavy scratch (f0, reference fields) goes to --workdir.

Usage:
  python3 scripts/ootc_fullscale_inversion.py            # full scale
  python3 scripts/ootc_fullscale_inversion.py --smoke    # N=32 T=200
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm.adjoint_ootc import (  # noqa: E402
    fd_gradient,
    initial_pulse,
    invert_tau,
    macro,
    ootc_gradient,
    step,
)

OUT = os.path.join(REPO, "artifacts", "adjoint_ootc_acceptance")
TAU_TRUE = 0.8
TAU0 = 1.2
AMP = 0.05


def _hwm_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _forward_terminal(f0: np.ndarray, tau: float, T: int):
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    return macro(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--tau-true", type=float, default=TAU_TRUE)
    ap.add_argument("--tau0", type=float, default=TAU0)
    ap.add_argument("--max-iter", type=int, default=24)
    ap.add_argument("--landscape", default="0.70,0.75,0.80,0.85,0.90",
                    help="comma-separated tau values for the lean J landscape")
    ap.add_argument("--workdir", default="/tmp/ootc_inv_full")
    ap.add_argument("--smoke", action="store_true",
                    help="N=32, T=200 smoke test for local validation")
    a = ap.parse_args()

    N, T, K = a.N, a.T, a.k
    tau_true, tau0 = a.tau_true, a.tau0
    if a.smoke:
        N, T, K = 32, 200, 50
    sigma = N / 8.0

    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    os.makedirs(a.workdir, exist_ok=True)
    res = {
        "N": N, "T": T, "k": K, "tau_true": tau_true, "tau_start": tau0,
        "amp": AMP, "sigma": sigma, "max_iter": a.max_iter,
        "git_sha": _git_sha(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "stages": {},
    }

    # -- setup: unbiased target u_ref = u_T(tau_true) ----------------------
    t0 = time.perf_counter()
    f0 = initial_pulse(N, N, amp=AMP, sigma=sigma)
    _, ux_t, uy_t = _forward_terminal(f0, tau_true, T)
    ux_ref, uy_ref = ux_t, uy_t          # unbiased: no amplitude scaling
    np.save(os.path.join(a.workdir, "f0.npy"), f0)
    dt_setup = time.perf_counter() - t0
    res["stages"]["setup"] = {"wall_s": dt_setup, "hwm_gib": _hwm_gib()}
    print(f"[setup] forward {T} steps in {dt_setup:.1f}s "
          f"HWM={_hwm_gib():.3f} GiB", flush=True)

    # -- A. gradient cross-check at tau0 -----------------------------------
    t0 = time.perf_counter()
    dJ_adj, J0 = ootc_gradient(tau0, T, f0, ux_ref, uy_ref, k=K)[:2]
    g_fd = fd_gradient(tau0, T, f0, ux_ref, uy_ref, h=1e-4)
    rel = abs(dJ_adj - g_fd) / abs(dJ_adj)
    res["gradient_check"] = {
        "tau": tau0, "ootc_dJ_dtau": dJ_adj, "fd_dJ_dtau": g_fd, "J": J0,
        "rel_diff": rel, "gate": 1e-5, "pass": bool(rel < 1e-5),
        "wall_s": time.perf_counter() - t0, "hwm_gib": _hwm_gib(),
    }
    print(f"[A] ootc={dJ_adj:.10e} fd={g_fd:.10e} rel={rel:.3e} "
          f"HWM={_hwm_gib():.3f} GiB", flush=True)

    # -- B. the full-scale inversion ---------------------------------------
    t0 = time.perf_counter()
    inv = invert_tau(tau0, T, f0, ux_ref, uy_ref, k=K,
                     max_iter=a.max_iter, verbose=True)
    inv["wall_s"] = time.perf_counter() - t0
    inv["rel_err_tau"] = abs(inv["tau"] - tau_true) / tau_true
    inv["hwm_gib"] = _hwm_gib()
    res["inversion"] = inv
    print(f"[B] tau*={inv['tau']:.12f} conv={inv['converged']} "
          f"({inv['reason']}) rel={inv['rel_err_tau']:.3e} "
          f"nfev={inv['nfev']} {inv['wall_s']:.0f}s "
          f"HWM={inv['hwm_gib']:.3f} GiB", flush=True)

    # -- C. lean landscape (forward only) ----------------------------------
    taus = [float(x) for x in a.landscape.split(",") if x.strip()]
    t0 = time.perf_counter()
    Js = []
    for tv in taus:
        _, ux_i, uy_i = _forward_terminal(f0, tv, T)
        Js.append(0.5 * float(np.sum((ux_i - ux_ref) ** 2
                                     + (uy_i - uy_ref) ** 2)))
    res["landscape"] = {"tau": taus, "J": Js,
                        "wall_s": time.perf_counter() - t0}
    print(f"[C] landscape {len(taus)} pts in "
          f"{res['landscape']['wall_s']:.0f}s", flush=True)

    # -- D. field snapshot at the recovered tau ----------------------------
    dJf, Jrec, (ux_r, uy_r), _, _, gfield = ootc_gradient(
        inv["tau"], T, f0, ux_ref, uy_ref, k=K, return_field=True)
    np.savez(os.path.join(OUT, "field_snapshot_fullscale.npz"),
             ux_terminal=ux_r, uy_terminal=uy_r, costate=gfield,
             tau_recovered=inv["tau"], N=N, T=T, k=K)
    res["field_snapshot"] = {
        "tau": inv["tau"], "J": Jrec, "dJ_dtau": dJf,
        "costate_absmax": float(np.abs(gfield).max()),
        "u_terminal_max": float(np.hypot(ux_r, uy_r).max()),
    }
    print(f"[D] snapshot at tau={inv['tau']:.6f} "
          f"|costate|max={res['field_snapshot']['costate_absmax']:.3e}",
          flush=True)

    # -- E. memory evidence -------------------------------------------------
    res["memory_gate"] = {
        "metric": "VmHWM of this single-process run (ru_maxrss)",
        "hwm_gib_final": _hwm_gib(),
        "gate_gib": 8.0,
        "pass": bool(_hwm_gib() < 8.0),
        "note": "OOTC checkpoint interval k=%d; checkpoint memory O(k*N*Q), "
                "independent of T; BPTT full storage at this scale is "
                "~11.2 GiB (mission 20040, cj2481)" % K,
    }
    res["acceptance_pass"] = bool(
        res["gradient_check"]["pass"] and inv["converged"]
        and inv["rel_err_tau"] < 0.03 and res["memory_gate"]["pass"])

    out_json = os.path.join(OUT, "inversion_fullscale.json")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[done] acceptance_pass={res['acceptance_pass']} -> {out_json}",
          flush=True)


if __name__ == "__main__":
    main()
