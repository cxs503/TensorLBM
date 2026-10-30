#!/usr/bin/env python
"""SDAA sweep for the verified/acoustics benchmark.

Runs the D2Q9 plane acoustic wave convergence cases (N=16,32,64, tau=1.0,
eps=1e-3) on the SDAA device, computes the pre-registered verdict criteria,
and writes artifacts/sdaa_acoustics.json.

No business code is modified.  The script imports the library kernels
(tensorlbm.solver.collide_bgk / stream, tensorlbm.d2q9.equilibrium /
macroscopic) and re-implements the measurement logic from
benchmarks/verified/acoustics/run.py verbatim.

If triton is needed by any import path, TRITON_INTERPRET=1 is set and
recorded in the output.  (The acoustics kernels themselves do not use
triton.)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# TRITON_INTERPRET workaround (record if set)
# ---------------------------------------------------------------------------
_triton_interpret = os.environ.get("TRITON_INTERPRET", "0")
_triton_interpret_set = False
# Check if triton is imported by any of our dependencies
try:
    import triton  # noqa: F401
    if _triton_interpret == "0":
        os.environ["TRITON_INTERPRET"] = "1"
        _triton_interpret_set = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
DEVICE = torch.device("sdaa:0")
DTYPE = torch.float64
CS = 1.0 / math.sqrt(3.0)  # lattice sound speed, LU/step
SQ3 = math.sqrt(3.0)

# ---------------------------------------------------------------------------
# Import library kernels (repo path)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d2q9 import OPPOSITE, C, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

# ---------------------------------------------------------------------------
# Theory references (exact discrete eigenvalue of the linear update)
# ---------------------------------------------------------------------------
_W9 = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
_CX9 = C[:, 0].numpy().astype(float)


def eig_acoustic(k: float, tau: float) -> complex:
    """Right-going acoustic eigenvalue s = log(lambda) of the linear
    collide-then-stream update for wavenumber k (axis-aligned).
    """
    r_ = 1.0 / tau
    I9 = np.eye(9)
    A = np.zeros((9, 9), dtype=complex)
    for q in range(9):
        A[q, :] = np.exp(1j * k * _CX9[q]) * (
            (1.0 - r_) * I9[q] + r_ * _W9[q] * (np.ones(9) + 3.0 * _CX9[q] * _CX9)
        )
    lam = np.linalg.eigvals(A)
    j = int(np.argmin(np.abs(np.log(lam) - 1j * k * CS)))
    return complex(np.log(lam[j]))


def refs(k: float, tau: float) -> dict:
    """Continuum + discrete-theory references for one (k, tau)."""
    s_star = eig_acoustic(k, tau)
    nu = CS * CS * (tau - 0.5)
    return {
        "k": k,
        "tau": tau,
        "nu": nu,
        "c_cont": CS,
        "delta_cont": nu * k * k,
        "s_discrete": [s_star.real, s_star.imag],
        "c_discrete": s_star.imag / k,
        "delta_discrete": -s_star.real,
    }


def fit_decay_robust(
    zp: np.ndarray, n_arr: np.ndarray, i0: int, i1: int, omega_hat: float
) -> tuple[float, float, float, float]:
    """Attenuation from |zp|^2 = A e^{-2 delta n} (1 + eta cos(2 w n + phi))."""
    n = n_arr[i0 : i1 + 1]
    y = np.abs(zp[i0 : i1 + 1]) ** 2
    cosb = np.cos(2.0 * omega_hat * n)
    sinb = np.sin(2.0 * omega_hat * n)
    M = np.stack([np.ones_like(n), cosb, sinb], axis=1)

    def sse(delta: float) -> tuple[float, np.ndarray]:
        w = y * np.exp(2.0 * delta * n)
        c, *_ = np.linalg.lstsq(M, w, rcond=None)
        r = w - M @ c
        return float(r @ r), c

    d0 = -float(np.polyfit(n, 0.5 * np.log(y), 1)[0])
    grid = np.geomspace(max(0.2 * d0, 1e-12), 5.0 * d0, 240)
    vals = [sse(d)[0] for d in grid]
    j = int(np.argmin(vals))
    lo, hi = grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]
    for _ in range(80):
        m1 = lo + 0.38197 * (hi - lo)
        m2 = hi - 0.38197 * (hi - lo)
        if sse(m1)[0] < sse(m2)[0]:
            hi = m2
        else:
            lo = m1
    d_star = 0.5 * (lo + hi)
    _, c = sse(d_star)
    eta = float(math.hypot(c[1], c[2]) / c[0])
    pred = np.exp(-2.0 * d_star * n) * (M @ c)
    ssr = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ssr / sst
    return d_star, eta, r2, d0


# ---------------------------------------------------------------------------
# Plane-wave simulation (library kernels, periodic, fp64, SDAA device)
# ---------------------------------------------------------------------------
def plane_wave_run(N: int, tau: float, eps: float, ny: int = 8, n_periods: float = 16.0) -> dict:
    """Run one plane-wave case and measure the fundamental mode."""
    nx = N
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, device=DEVICE, dtype=DTYPE)
    phase = k * x
    rho0 = 1.0 + eps * torch.sin(phase)
    ux0 = CS * eps * torch.sin(phase)
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(
        rho0.unsqueeze(0).expand(ny, nx).contiguous(),
        ux0.unsqueeze(0).expand(ny, nx).contiguous(),
        uy0.unsqueeze(0).expand(ny, nx).contiguous(),
    )

    period = SQ3 * nx
    nsteps = int(math.ceil(n_periods * period))
    e_minus = np.exp(-1j * k * np.arange(nx))
    e_plus = np.conj(e_minus)

    zp = np.empty(nsteps + 1, dtype=complex)
    zm = np.empty(nsteps + 1, dtype=complex)
    t0 = time.time()
    for n in range(nsteps + 1):
        rho = macroscopic(f)[0]
        rp = (rho[0].cpu().numpy()).astype(np.float64) - 1.0
        zp[n] = rp @ e_minus / nx
        zm[n] = rp @ e_plus / nx
        if n == nsteps:
            break
        f = collide_bgk(f, tau)
        f = stream(f)
    wall = time.time() - t0

    i0 = int(math.ceil(0.5 * period))
    i1 = min(int(math.floor(15.5 * period)), nsteps)
    n_arr = np.arange(nsteps + 1, dtype=np.float64)
    sl = slice(i0, i1 + 1)

    ph = np.unwrap(np.angle(zp[sl]))
    slope_p, icept_p = np.polyfit(n_arr[sl], ph, 1)
    predp = np.polyval([slope_p, icept_p], n_arr[sl])
    ss_resp = float(np.sum((ph - predp) ** 2))
    ss_totp = float(np.sum((ph - ph.mean()) ** 2))
    r2_ph = 1.0 - ss_resp / ss_totp
    omega_meas = -slope_p

    delta_meas, eta, r2_amp, delta_naive = fit_decay_robust(zp, n_arr, i0, i1, omega_meas)
    c_meas = omega_meas / k

    rho_end = macroscopic(f)[0]
    out = {
        "N": N,
        "ny": ny,
        "k": k,
        "tau": tau,
        "eps": eps,
        "nsteps": nsteps,
        "period_steps": period,
        "fit_window": [i0, i1],
        "wall_s": wall,
        "delta_meas": delta_meas,
        "delta_meas_naive": delta_naive,
        "omega_meas": omega_meas,
        "c_meas": c_meas,
        "r2_amp": r2_amp,
        "r2_phase": r2_ph,
        "counterwave_ratio_eta_over_2": eta / 2.0,
        "max_abs_rho_minus_1_end": float((rho_end - 1).abs().max().cpu().item()),
        "rho_min_end": float(rho_end.min().cpu().item()),
        "rho_max_end": float(rho_end.max().cpu().item()),
        "finite": bool(torch.isfinite(f).all().cpu().item()),
    }
    out.update(refs(k, tau))
    out["c_err_cont_pct"] = 100.0 * (c_meas / CS - 1.0)
    out["delta_err_cont_pct"] = 100.0 * (delta_meas / out["delta_cont"] - 1.0)
    out["c_err_disc_pct"] = 100.0 * (c_meas / out["c_discrete"] - 1.0)
    out["delta_err_disc_pct"] = 100.0 * (delta_meas / out["delta_discrete"] - 1.0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    wall_start = time.time()

    # Device evidence
    device_name = "N/A"
    try:
        device_name = torch.sdaa.get_device_name(0)
    except Exception as e:
        device_name = f"error: {e}"

    print(f"[sdaa_acoustics] device: {device_name}")
    print(f"[sdaa_acoustics] TRITON_INTERPRET workaround: {_triton_interpret_set}")

    # Run convergence cases
    cases = [16, 32, 64]
    tau = 1.0
    eps = 1e-3
    results = []
    for N in cases:
        print(f"[sdaa_acoustics] Running N={N} ...")
        r = plane_wave_run(N, tau, eps)
        results.append(r)
        print(
            f"  c_meas={r['c_meas']:.10f}  c_err_cont={r['c_err_cont_pct']:+.4f}%  "
            f"delta_err_cont={r['delta_err_cont_pct']:+.4f}%"
        )

    wall_total = time.time() - wall_start

    # Verdict (pre-registered criteria from README §1.6)
    c_errs = [abs(r["c_err_cont_pct"]) for r in results]
    delta_errs = [abs(r["delta_err_cont_pct"]) for r in results]
    c_err_max = max(c_errs)
    delta_err_max = max(delta_errs)

    # Monotonicity check: errors should decrease with N
    c_monotone = all(c_errs[i] >= c_errs[i + 1] for i in range(len(c_errs) - 1))
    delta_monotone = all(delta_errs[i] >= delta_errs[i + 1] for i in range(len(delta_errs) - 1))

    # Criteria: ≤3% and monotone
    pass_c = c_err_max <= 3.0 and c_monotone
    pass_delta = delta_err_max <= 3.0 and delta_monotone

    verdict = {
        "c_err_pct_max": c_err_max,
        "delta_err_pct_max": delta_err_max,
        "c_monotone": c_monotone,
        "delta_monotone": delta_monotone,
        "pass_c": pass_c,
        "pass_delta": pass_delta,
        "pass_overall": pass_c and pass_delta,
    }

    # Build output
    output = {
        "meta": {
            "task": "SDAA sweep: D2Q9 plane acoustic wave dispersion & attenuation",
            "device": "sdaa:0",
            "device_name": device_name,
            "dtype": "float64",
            "triton_interpret_workaround": _triton_interpret_set,
            "wall_s_total": wall_total,
        },
        "convergence": results,
        "verdict_main": verdict,
        "criteria": [
            {
                "name": "判据 A: |c_meas/c_s - 1| ≤ 3% 且单调",
                "threshold_pct": 3.0,
                "max_err_pct": c_err_max,
                "monotone": c_monotone,
                "pass": pass_c,
            },
            {
                "name": "判据 B: |δ_meas/(νk²) - 1| ≤ 3% 且单调",
                "threshold_pct": 3.0,
                "max_err_pct": delta_err_max,
                "monotone": delta_monotone,
                "pass": pass_delta,
            },
        ],
    }

    # Write artifact
    artifact_path = REPO_ROOT / "artifacts" / "sdaa_acoustics.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\n[sdaa_acoustics] Artifact written to {artifact_path}")
    print(f"[sdaa_acoustics] Wall clock: {wall_total:.1f}s")
    print(f"[sdaa_acoustics] Verdict: c_err_max={c_err_max:.4f}% ({'PASS' if pass_c else 'FAIL'}), "
          f"delta_err_max={delta_err_max:.4f}% ({'PASS' if pass_delta else 'FAIL'})")
    print(f"[sdaa_acoustics] Overall: {'PASS' if verdict['pass_overall'] else 'FAIL'}")


if __name__ == "__main__":
    main()
