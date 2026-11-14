"""Full-scale acceptance verification for the OOTC adjoint (GAP-PP-3-adjoint-ootc).

Acceptance (feature_ledger, slug GAP-PP-3-adjoint-ootc):
  * 10^4 steps on a 128^2 grid, tau inversion
  * OOTC peak memory < 8 GiB
  * |dJ_ootc - dJ_bptt| / |dJ_bptt| < 1e-5

Independent cross-checks (the module-internal OOTC-vs-BPTT identity is
NOT sufficient: both share the same formulas, so a wrong formula would
pass -- see docs/ADJOINT_OOTC_VERIFICATION.md, defects D1/D2):

  fd     : central difference of J(tau), h = 1e-4          -> dJ/dtau
  field  : central difference of J(f0 +/- h*v) vs the adjoint directional
           derivative <dJ/df0, v>  (validates the full costate field at
           n = 0, not just the scalar tau derivative)

Every variant runs in its own subprocess so ru_maxrss (VmHWM) is a clean
per-variant peak; the driver runs them in parallel and merges the
per-variant JSON.  The merged results file is written INSIDE the repo
(default ``artifacts/adjoint_ootc_acceptance/results.json``) because a
compute job's /tmp is not part of the fetched artifact: job cj1625
(m17532) succeeded but its numbers were written to /tmp on the node and
were unrecoverable, which is why the ledger anchor stayed open.  Heavy
arrays (f0, reference fields) stay in ``--workdir`` (default: /tmp).

Usage:
  python3 scripts/verify_ootc_acceptance.py                 # full scale
  python3 scripts/verify_ootc_acceptance.py --quick         # smoke test
  python3 scripts/verify_ootc_acceptance.py --mode ootc --k 100
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
sys.path.insert(0, os.path.join(REPO, "TensorLBM", "src"))

from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro  # noqa: E402
from tensorlbm.adjoint.bptt import bptt_gradient  # noqa: E402
from tensorlbm.adjoint.ootc import ootc_gradient  # noqa: E402

N = 128          # grid (128^2)
T = 10000        # 10^4 steps
TAU = 1.0        # relaxation time
AMP = 0.05
SIGMA = 16.0     # N/8, same convention as the unit tests
H_FD = 1e-4      # finite-difference half-step for dJ/dtau
H_DIR = 1e-5     # step for the f0-directional derivative
MEM_GATE_GIB = 8.0
GRAD_GATE = 1e-5
DIR_GATE = 1e-5
SEED = 20261114


def _hwm_gib() -> float:
    """Peak resident set size of this process (Linux: VmHWM via ru_maxrss kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _meta(mode: str) -> dict:
    return {
        "mode": mode,
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }


def _setup(workdir: str) -> None:
    """Forward pass once; store f0, the reference terminal velocity, and v."""
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
    # deterministic random direction in f0-space for the field check,
    # scaled to |f0| so that h * |v| is a h-relative perturbation.  The
    # SAME saved v is used by both the adjoint and the FD variants (a
    # scale mismatch here silently corrupts the comparison).
    rng = np.random.default_rng(SEED)
    v = rng.standard_normal(f0.shape)
    v = v * (np.sqrt((f0 * f0).sum()) / np.sqrt((v * v).sum()))
    np.save(os.path.join(workdir, "v.npy"), v)
    print(f"[setup] N={N} T={T} forward {T} steps in {dt:.1f}s "
          f"({T / dt:.1f} steps/s), HWM={_hwm_gib():.2f} GiB, "
          f"|u_T|_max={np.hypot(ux_t, uy_t).max():.3e}", flush=True)


def _load(workdir: str):
    f0 = np.load(os.path.join(workdir, "f0.npy"))
    ux_ref = np.load(os.path.join(workdir, "ux_ref.npy"))
    uy_ref = np.load(os.path.join(workdir, "uy_ref.npy"))
    return f0, ux_ref, uy_ref


def _run_bptt(workdir: str, out: str) -> None:
    f0, ux_ref, uy_ref = _load(workdir)
    t0 = time.perf_counter()
    dJ, J, (ux_t, uy_t), states, g = bptt_gradient(
        TAU, T, f0, ux_ref, uy_ref, return_field=True)
    dt = time.perf_counter() - t0
    v = np.load(os.path.join(workdir, "v.npy"))
    res = {**_meta("bptt"), "dJ": dJ, "J": J, "time_s": dt,
           "hwm_gib": _hwm_gib(), "n_states": len(states),
           "dir_adjoint": float((g * v).sum()),
           "state_bytes": 8 * f0.size * len(states)}
    del states, g
    json.dump(res, open(out, "w"), indent=1)
    print(f"[bptt] dJ={dJ:.10e} J={J:.10e} t={dt:.1f}s HWM={res['hwm_gib']:.2f} GiB",
          flush=True)


def _run_ootc(workdir: str, out: str, k: int) -> None:
    f0, ux_ref, uy_ref = _load(workdir)
    ckpt = os.path.join(workdir, f"ckpt_k{k}.npy")
    t0 = time.perf_counter()
    dJ, J, (ux_t, uy_t), n_ckpt, ckpt_path, g = ootc_gradient(
        TAU, T, f0, ux_ref, uy_ref, k=k, ckpt_path=ckpt, return_field=True)
    dt = time.perf_counter() - t0
    v = np.load(os.path.join(workdir, "v.npy"))
    ckpt_bytes = os.path.getsize(ckpt_path)
    res = {**_meta(f"ootc_k{k}"), "k": k, "dJ": dJ, "J": J, "time_s": dt,
           "hwm_gib": _hwm_gib(), "n_ckpt": n_ckpt,
           "ckpt_bytes": ckpt_bytes, "dir_adjoint": float((g * v).sum())}
    del g
    os.unlink(ckpt_path)  # checkpoint file is scratch, keep the evidence JSON
    json.dump(res, open(out, "w"), indent=1)
    print(f"[ootc k={k}] dJ={dJ:.10e} J={J:.10e} t={dt:.1f}s "
          f"HWM={res['hwm_gib']:.2f} GiB ckpt={ckpt_bytes / 1e6:.0f} MB",
          flush=True)


def _forward_J(f0: np.ndarray, tau: float, ux_ref: np.ndarray,
               uy_ref: np.ndarray) -> float:
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    _, ux_t, uy_t = macro(f)
    return 0.5 * float(np.sum((ux_t - ux_ref) ** 2 + (uy_t - uy_ref) ** 2))


def _run_fd(workdir: str, out: str) -> None:
    f0, ux_ref, uy_ref = _load(workdir)
    t0 = time.perf_counter()
    J_plus = _forward_J(f0, TAU + H_FD, ux_ref, uy_ref)
    J_minus = _forward_J(f0, TAU - H_FD, ux_ref, uy_ref)
    dt = time.perf_counter() - t0
    dJ_fd = (J_plus - J_minus) / (2.0 * H_FD)
    res = {**_meta("fd"), "h": H_FD, "J_plus": J_plus, "J_minus": J_minus,
           "dJ": dJ_fd, "time_s": dt, "hwm_gib": _hwm_gib()}
    json.dump(res, open(out, "w"), indent=1)
    print(f"[fd] dJ={dJ_fd:.10e} t={dt:.1f}s HWM={res['hwm_gib']:.2f} GiB",
          flush=True)


def _run_field(workdir: str, out: str) -> None:
    """Directional central difference of J w.r.t. f0 along v."""
    f0, ux_ref, uy_ref = _load(workdir)
    v = np.load(os.path.join(workdir, "v.npy"))
    t0 = time.perf_counter()
    J_plus = _forward_J(f0 + H_DIR * v, TAU, ux_ref, uy_ref)
    J_minus = _forward_J(f0 - H_DIR * v, TAU, ux_ref, uy_ref)
    dt = time.perf_counter() - t0
    dir_fd = (J_plus - J_minus) / (2.0 * H_DIR)
    res = {**_meta("field_fd"), "h": H_DIR, "J_plus": J_plus,
           "J_minus": J_minus, "dir_fd": dir_fd, "v_norm": float(np.sqrt((v * v).sum())),
           "time_s": dt, "hwm_gib": _hwm_gib()}
    json.dump(res, open(out, "w"), indent=1)
    print(f"[field_fd] <dJ/df0,v>={dir_fd:.10e} t={dt:.1f}s "
          f"HWM={res['hwm_gib']:.2f} GiB", flush=True)


def _driver(workdir: str, out: str, ks) -> None:
    os.makedirs(workdir, exist_ok=True)
    _setup(workdir)
    here = os.path.abspath(__file__)
    variants = {"bptt": ["--mode", "bptt"], "fd": ["--mode", "fd"],
                "field_fd": ["--mode", "field"]}
    for k in ks:
        variants[f"ootc_k{k}"] = ["--mode", "ootc", "--k", str(k)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    procs = {}
    for name, args in variants.items():
        base = ["--N", str(N), "--T", str(T), "--workdir", workdir,
                "--out", os.path.join(workdir, f"r_{name}.json")]
        procs[name] = subprocess.Popen([sys.executable, here, *args, *base], env=env)
        print(f"[driver] launched {name}", flush=True)
    failed = []
    for name, p in procs.items():
        rc = p.wait()
        print(f"[driver] {name} rc={rc}", flush=True)
        if rc != 0:
            failed.append(name)
    if failed:
        raise SystemExit(f"variants failed: {failed}")

    merged = {"N": N, "T": T, "tau": TAU, "amp": AMP, "sigma": SIGMA,
              "h_fd_tau": H_FD, "h_fd_field": H_DIR, "seed": SEED,
              "git_sha": _git_sha(),
              "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "host": socket.gethostname(), "variants": {}}
    for name in variants:
        merged["variants"][name] = json.load(
            open(os.path.join(workdir, f"r_{name}.json")))

    v = merged["variants"]
    bptt = v["bptt"]
    k_first = ks[0]
    ootc = v[f"ootc_k{k_first}"]
    rel_ootc = abs(ootc["dJ"] - bptt["dJ"]) / abs(bptt["dJ"])
    rel_fd = abs(bptt["dJ"] - v["fd"]["dJ"]) / abs(bptt["dJ"])
    dir_adj = ootc["dir_adjoint"]
    dir_fd = v["field_fd"]["dir_fd"]
    rel_dir = abs(dir_adj - dir_fd) / abs(dir_fd)
    merged["gates"] = {
        # acceptance gate 1: gradient agrees with the BPTT reference
        "ootc_vs_bptt_rel": rel_ootc,
        "ootc_vs_bptt_pass": bool(rel_ootc < GRAD_GATE),
        # acceptance gate 2: peak memory of the long-horizon inversion
        "ootc_hwm_gib": ootc["hwm_gib"],
        "mem_gate_gib": MEM_GATE_GIB,
        "mem_gate_pass": bool(ootc["hwm_gib"] < MEM_GATE_GIB),
        # context: the full-storage reference exceeds the gate (why OOTC)
        "bptt_hwm_gib": bptt["hwm_gib"],
        "bptt_exceeds_mem_gate": bool(bptt["hwm_gib"] > MEM_GATE_GIB),
        "mem_ratio_bptt_over_ootc": bptt["hwm_gib"] / ootc["hwm_gib"],
        # independent checks of the adjoint derivation itself
        "bptt_vs_fd_rel": rel_fd,
        "bptt_vs_fd_pass": bool(rel_fd < 1e-5),
        "field_dir_rel": rel_dir,
        "field_dir_pass": bool(rel_dir < DIR_GATE),
        "dir_adjoint": dir_adj,
        "dir_fd": dir_fd,
        # cost
        "time_bptt_s": bptt["time_s"],
        "time_ootc_s": ootc["time_s"],
        "ckpt_bytes": ootc["ckpt_bytes"],
    }
    merged["acceptance_pass"] = bool(
        merged["gates"]["ootc_vs_bptt_pass"] and merged["gates"]["mem_gate_pass"])
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    json.dump(merged, open(out, "w"), indent=1)
    print(json.dumps(merged["gates"], indent=1))
    print(f"[driver] acceptance_pass={merged['acceptance_pass']} -> {out}")


def main() -> None:
    global N, T, SIGMA
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="driver",
                    choices=["driver", "setup", "bptt", "ootc", "fd", "field"])
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--ks", default="100,1000",
                    help="comma separated checkpoint intervals for the driver")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test (N=64, T=500) for local <10 min runs")
    ap.add_argument("--workdir", default="/tmp/ootc_acc")
    ap.add_argument("--out", default=os.path.join(
        REPO, "artifacts", "adjoint_ootc_acceptance", "results.json"))
    a = ap.parse_args()
    N, T = a.N, a.T
    SIGMA = N / 8.0
    if a.quick:
        N, T, SIGMA = 64, 500, 8.0
    if a.mode == "driver":
        ks = [int(x) for x in a.ks.split(",") if x.strip()]
        _driver(a.workdir, a.out, ks)
    elif a.mode == "setup":
        _setup(a.workdir)
    elif a.mode == "bptt":
        _run_bptt(a.workdir, a.out)
    elif a.mode == "ootc":
        _run_ootc(a.workdir, a.out, a.k)
    elif a.mode == "fd":
        _run_fd(a.workdir, a.out)
    elif a.mode == "field":
        _run_field(a.workdir, a.out)


if __name__ == "__main__":
    main()