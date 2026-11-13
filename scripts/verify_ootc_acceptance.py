"""Full-scale acceptance verification for the OOTC adjoint (GAP-PP-3-adjoint-ootc).

Acceptance (feature_ledger, slug GAP-PP-3-adjoint-ootc):
  * 10^4 steps on a 128^2 grid, tau inversion
  * OOTC peak memory < 8 GiB
  * |dJ_ootc - dJ_bptt| / |dJ_bptt| < 1e-5

Independent cross-check: central finite difference of J(tau)
  dJ/dtau ~ (J(tau+h) - J(tau-h)) / (2h),  h = 1e-4,
which validates the adjoint derivation itself (BPTT and OOTC share the
same formulas; FD is the independent reference).

Each variant runs in its own subprocess so that ru_maxrss (VmHWM) is a
clean per-variant peak.  The driver runs the variants in parallel
(4 cores) and merges the per-variant JSON into one results file
(zero hand-copied numbers: the report reads this JSON).

Usage:
  python3 scripts/verify_ootc_acceptance.py            # driver (parallel)
  python3 scripts/verify_ootc_acceptance.py --mode setup|bptt|ootc|fd
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TensorLBM", "src"))

from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro  # noqa: E402
from tensorlbm.adjoint.bptt import bptt_gradient  # noqa: E402
from tensorlbm.adjoint.ootc import ootc_gradient  # noqa: E402

N = 128          # grid (128^2)
T = 10000        # 10^4 steps
TAU = 1.0        # relaxation time
AMP = 0.05
SIGMA = 16.0     # N/8, same convention as the unit tests
H_FD = 1e-4      # finite-difference half-step


def _hwm_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _setup(workdir: str) -> None:
    """Forward pass once; store f0 and the reference terminal velocity."""
    os.makedirs(workdir, exist_ok=True)
    f0 = initial_pulse(N, N, amp=AMP, sigma=SIGMA)
    np.save(os.path.join(workdir, "f0.npy"), f0)
    f = f0.copy()
    t0 = time.perf_counter()
    for _ in range(T):
        f = step(f, TAU)
    dt = time.perf_counter() - t0
    rho_t, ux_t, uy_t = macro(f)
    np.save(os.path.join(workdir, "ux_ref.npy"), ux_t * 0.9)
    np.save(os.path.join(workdir, "uy_ref.npy"), uy_t * 0.9)
    np.save(os.path.join(workdir, "ux_t.npy"), ux_t)
    np.save(os.path.join(workdir, "uy_t.npy"), uy_t)
    print(f"[setup] forward {T} steps in {dt:.1f}s "
          f"({T / dt:.1f} steps/s), HWM={_hwm_gib():.2f} GiB", flush=True)


def _load(workdir: str):
    f0 = np.load(os.path.join(workdir, "f0.npy"))
    ux_ref = np.load(os.path.join(workdir, "ux_ref.npy"))
    uy_ref = np.load(os.path.join(workdir, "uy_ref.npy"))
    return f0, ux_ref, uy_ref


def _run_bptt(workdir: str, out: str) -> None:
    f0, ux_ref, uy_ref = _load(workdir)
    t0 = time.perf_counter()
    dJ, J, (ux_t, uy_t), states = bptt_gradient(TAU, T, f0, ux_ref, uy_ref)
    dt = time.perf_counter() - t0
    res = {"mode": "bptt", "dJ": dJ, "J": J, "time_s": dt,
           "hwm_gib": _hwm_gib(), "n_states": len(states)}
    np.save(os.path.join(workdir, "ux_t_bptt.npy"), ux_t)
    np.save(os.path.join(workdir, "uy_t_bptt.npy"), uy_t)
    json.dump(res, open(out, "w"), indent=1)
    print(f"[bptt] dJ={dJ:.10e} J={J:.10e} t={dt:.1f}s HWM={res['hwm_gib']:.2f} GiB",
          flush=True)


def _run_ootc(workdir: str, out: str, k: int) -> None:
    f0, ux_ref, uy_ref = _load(workdir)
    t0 = time.perf_counter()
    dJ, J, (ux_t, uy_t), n_ckpt, ckpt_path = ootc_gradient(
        TAU, T, f0, ux_ref, uy_ref, k=k)
    dt = time.perf_counter() - t0
    res = {"mode": f"ootc_k{k}", "k": k, "dJ": dJ, "J": J, "time_s": dt,
           "hwm_gib": _hwm_gib(), "n_ckpt": n_ckpt,
           "ckpt_bytes": os.path.getsize(ckpt_path)}
    json.dump(res, open(out, "w"), indent=1)
    print(f"[ootc k={k}] dJ={dJ:.10e} J={J:.10e} t={dt:.1f}s "
          f"HWM={res['hwm_gib']:.2f} GiB ckpt={res['ckpt_bytes'] / 1e6:.0f} MB",
          flush=True)


def _run_fd(workdir: str, out: str) -> None:
    f0, ux_ref, uy_ref = _load(workdir)

    def forward_J(tau: float) -> float:
        f = f0.copy()
        for _ in range(T):
            f = step(f, tau)
        rho_t, ux_t, uy_t = macro(f)
        return 0.5 * float(np.sum((ux_t - ux_ref) ** 2 + (uy_t - uy_ref) ** 2))

    t0 = time.perf_counter()
    J_plus = forward_J(TAU + H_FD)
    J_minus = forward_J(TAU - H_FD)
    dt = time.perf_counter() - t0
    dJ_fd = (J_plus - J_minus) / (2.0 * H_FD)
    res = {"mode": "fd", "h": H_FD, "J_plus": J_plus, "J_minus": J_minus,
           "dJ": dJ_fd, "time_s": dt, "hwm_gib": _hwm_gib()}
    json.dump(res, open(out, "w"), indent=1)
    print(f"[fd] dJ={dJ_fd:.10e} t={dt:.1f}s HWM={res['hwm_gib']:.2f} GiB",
          flush=True)


def _driver(workdir: str, out: str) -> None:
    os.makedirs(workdir, exist_ok=True)
    _setup(workdir)
    here = os.path.abspath(__file__)
    jobs = {
        "bptt": ["--mode", "bptt", "--workdir", workdir,
                 "--out", os.path.join(workdir, "r_bptt.json")],
        "ootc_k100": ["--mode", "ootc", "--k", "100", "--workdir", workdir,
                      "--out", os.path.join(workdir, "r_ootc_k100.json")],
        "ootc_k1000": ["--mode", "ootc", "--k", "1000", "--workdir", workdir,
                       "--out", os.path.join(workdir, "r_ootc_k1000.json")],
        "fd": ["--mode", "fd", "--workdir", workdir,
                "--out", os.path.join(workdir, "r_fd.json")],
    }
    procs = {name: subprocess.Popen(
        [sys.executable, here, *args],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    ) for name, args in jobs.items()}
    for name, p in procs.items():
        rc = p.wait()
        print(f"[driver] {name} rc={rc}", flush=True)
        if rc != 0:
            raise SystemExit(f"variant {name} failed rc={rc}")
    merged = {"N": N, "T": T, "tau": TAU, "amp": AMP, "sigma": SIGMA,
              "h_fd": H_FD, "variants": {}}
    for name in jobs:
        merged["variants"][name] = json.load(
            open(os.path.join(workdir, f"r_{name}.json")))
    v = merged["variants"]
    rel_ootc = abs(v["ootc_k100"]["dJ"] - v["bptt"]["dJ"]) / abs(v["bptt"]["dJ"])
    rel_fd = abs(v["bptt"]["dJ"] - v["fd"]["dJ"]) / abs(v["bptt"]["dJ"])
    merged["gates"] = {
        "ootc_vs_bptt_rel": rel_ootc,
        "ootc_vs_bptt_pass": bool(rel_ootc < 1e-5),
        "bptt_vs_fd_rel": rel_fd,
        "ootc_hwm_gib": v["ootc_k100"]["hwm_gib"],
        "mem_gate_pass": bool(v["ootc_k100"]["hwm_gib"] < 8.0),
    }
    json.dump(merged, open(out, "w"), indent=1)
    print(json.dumps(merged["gates"], indent=1))
    print(f"[driver] results -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="driver",
                    choices=["driver", "setup", "bptt", "ootc", "fd"])
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--workdir", default="/tmp/ootc_acc")
    ap.add_argument("--out", default="/tmp/ootc_acc/results.json")
    a = ap.parse_args()
    if a.mode == "driver":
        _driver(a.workdir, a.out)
    elif a.mode == "setup":
        _setup(a.workdir)
    elif a.mode == "bptt":
        _run_bptt(a.workdir, a.out)
    elif a.mode == "ootc":
        _run_ootc(a.workdir, a.out, a.k)
    elif a.mode == "fd":
        _run_fd(a.workdir, a.out)


if __name__ == "__main__":
    main()
