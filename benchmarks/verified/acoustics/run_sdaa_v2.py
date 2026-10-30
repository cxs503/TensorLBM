"""SDAA v2 runner for D2Q9 plane acoustic wave benchmark.

Uses the EXACT same algorithm, criteria, thresholds, and reference values
as run.py — only differences:
  1. DEVICE = sdaa:0 (instead of cpu)
  2. .cpu() added before .numpy() for measurement extraction (device transfer)

No criteria, thresholds, or reference values are changed.

Output: artifacts/sdaa_acoustics_v2.json
"""

from __future__ import annotations

import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── device setup ────────────────────────────────────────────────────────────
assert torch.sdaa.is_available(), "SDAA device not available"
DEVICE = torch.device("sdaa:0")
DEV_NAME = torch.sdaa.get_device_name(0)
TCAI_CARDS = sorted(glob.glob("/dev/tcaicard*"))

# ── import library kernels + theory from run.py ─────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))  # <repo>/benchmarks for tensorlbm

from tensorlbm.d2q9 import C, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

import run as bm  # noqa: E402  — for theory functions (eig_acoustic, refs, fit_decay_robust)

DTYPE = torch.float64
CS = 1.0 / math.sqrt(3.0)
TAU_MAIN = 1.0
EPS_MAIN = 1.0e-3
SQ3 = math.sqrt(3.0)


# ── plane-wave simulation on SDAA (same algorithm as run.py::plane_wave_run) ─
def plane_wave_run_sdaa(N: int, tau: float, eps: float, ny: int = 8,
                        n_periods: float = 16.0) -> dict:
    """Run one plane-wave case and measure the fundamental mode.

    Identical algorithm to run.py::plane_wave_run.  Only difference:
    DEVICE=sdaa:0 and .cpu() before .numpy() for measurement.
    """
    nx = N
    k = 2.0 * math.pi / nx
    x = torch.arange(nx, device=DEVICE, dtype=DTYPE)
    phase = k * x
    rho0 = 1.0 + eps * torch.sin(phase)
    ux0 = CS * eps * torch.sin(phase)  # isothermal u' = c_s rho'/rho0
    uy0 = torch.zeros_like(rho0)
    f = equilibrium(
        rho0.unsqueeze(0).expand(ny, nx).contiguous(),
        ux0.unsqueeze(0).expand(ny, nx).contiguous(),
        uy0.unsqueeze(0).expand(ny, nx).contiguous(),
    )

    period = SQ3 * nx  # acoustic period in steps
    nsteps = int(math.ceil(n_periods * period))
    e_minus = np.exp(-1j * k * np.arange(nx))
    e_plus = np.conj(e_minus)

    zp = np.empty(nsteps + 1, dtype=complex)
    zm = np.empty(nsteps + 1, dtype=complex)
    t0 = time.time()
    for n in range(nsteps + 1):
        rho = macroscopic(f)[0]
        # .cpu() transfer for measurement (only difference from run.py)
        rp = (rho[0].cpu().numpy()).astype(np.float64) - 1.0  # y-invariant
        zp[n] = rp @ e_minus / nx
        zm[n] = rp @ e_plus / nx
        if n == nsteps:
            break
        f = collide_bgk(f, tau)
        f = stream(f)
    torch.sdaa.synchronize()
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
    omega_meas = -slope_p  # arg zp rotates at -omega

    delta_meas, eta, r2_amp, delta_naive = bm.fit_decay_robust(
        zp, n_arr, i0, i1, omega_meas)
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
        "max_abs_rho_minus_1_end": float((rho_end - 1).abs().max().cpu()),
        "rho_min_end": float(rho_end.min().cpu()),
        "rho_max_end": float(rho_end.max().cpu()),
        "finite": bool(torch.isfinite(f).all().cpu()),
    }
    out.update(bm.refs(k, tau))
    out["c_err_cont_pct"] = 100.0 * (c_meas / CS - 1.0)
    out["delta_err_cont_pct"] = 100.0 * (delta_meas / out["delta_cont"] - 1.0)
    out["c_err_disc_pct"] = 100.0 * (c_meas / out["c_discrete"] - 1.0)
    out["delta_err_disc_pct"] = 100.0 * (delta_meas / out["delta_discrete"] - 1.0)
    return out


# ── run the main convergence cases (frozen criteria) ────────────────────────
print(f"[SDAA v2] device={DEV_NAME}  cards={TCAI_CARDS}")
print(f"[SDAA v2] dtype={DTYPE}  DEVICE={DEVICE}")
print(f"[SDAA v2] Frozen criteria: Gate A |c/c_s-1|<=3% monotone; "
      f"Gate B |delta/(nu*k^2)-1|<=3% monotone")
print(f"[SDAA v2] N in {{16,32,64}}, tau=1.0, eps=1e-3, threshold=3%")
print()

t_start = time.time()

convergence = []
for N in [16, 32, 64]:
    t0 = time.time()
    res = plane_wave_run_sdaa(N, tau=TAU_MAIN, eps=EPS_MAIN)
    elapsed = time.time() - t0
    res["wall_s_sdaa"] = elapsed
    convergence.append(res)
    print(f"  N={N:3d}  c_meas={res['c_meas']:.9f}  "
          f"c_err_cont={res['c_err_cont_pct']:+.4f}%  "
          f"delta_err_cont={res['delta_err_cont_pct']:+.4f}%  "
          f"wall={elapsed:.3f}s")

# Verdict (same logic as run.py)
c_errs = [r["c_err_cont_pct"] for r in convergence]
d_errs = [r["delta_err_cont_pct"] for r in convergence]
c_mono = all(abs(c_errs[i]) > abs(c_errs[i + 1]) for i in range(len(c_errs) - 1))
d_mono = all(abs(d_errs[i]) > abs(d_errs[i + 1]) for i in range(len(d_errs) - 1))
pass_c = max(abs(e) for e in c_errs) <= 3.0 and c_mono
pass_d = max(abs(e) for e in d_errs) <= 3.0 and d_mono

verdict = {
    "c_err_pct_max": max(abs(e) for e in c_errs),
    "delta_err_pct_max": max(abs(e) for e in d_errs),
    "c_monotone": c_mono,
    "delta_monotone": d_mono,
    "pass_c": pass_c,
    "pass_delta": pass_d,
    "threshold_pct": 3.0,
}

t_total = time.time() - t_start

# ── assemble output ─────────────────────────────────────────────────────────
out = {
    "meta": {
        "task": "SDAA v2 D2Q9 plane acoustic wave dispersion & attenuation",
        "runner": "run_sdaa_v2.py (same algorithm as run.py, DEVICE=sdaa:0, "
                  ".cpu() before .numpy() for measurement; no criteria changes)",
        "repo": "TensorLBM",
        "branch": "agent/m213",
        "library": [
            "tensorlbm.solver.collide_bgk",
            "tensorlbm.solver.stream",
            "tensorlbm.d2q9.equilibrium",
            "tensorlbm.d2q9.macroscopic",
        ],
        "dtype": "float64",
        "device_str": str(DEVICE),
        "device_name": DEV_NAME,
        "tcaicard_devices": TCAI_CARDS,
        "torch_version": torch.__version__,
    },
    "frozen_criteria": {
        "gate_A": "|c_meas/c_s - 1| <= 3% and monotone decreasing with N",
        "gate_B": "|delta_meas/(nu*k^2) - 1| <= 3% and monotone",
        "threshold_pct": 3.0,
        "N_values": [16, 32, 64],
        "tau": 1.0,
        "eps": 1e-3,
        "reference": "continuum linear acoustics: c_s=1/sqrt(3), "
                     "delta=nu*k^2, nu=c_s^2*(tau-1/2) — zero adjustable params",
        "source": "benchmarks/verified/acoustics/README.md section 1.6",
    },
    "convergence": convergence,
    "verdict_main": verdict,
    "wall_clock_total_s": t_total,
}

# ── write artifact ──────────────────────────────────────────────────────────
artifacts_dir = HERE.parents[2] / "artifacts"
artifacts_dir.mkdir(exist_ok=True)
out_path = artifacts_dir / "sdaa_acoustics_v2.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\n[SDAA v2] Artifact written to {out_path}")
print(f"[SDAA v2] Total wall clock: {t_total:.3f}s")
print(f"[SDAA v2] Verdict: pass_c={pass_c}  pass_delta={pass_d}")
print(f"[SDAA v2]   c_err_max={verdict['c_err_pct_max']:.4f}%  "
      f"delta_err_max={verdict['delta_err_pct_max']:.4f}%")
