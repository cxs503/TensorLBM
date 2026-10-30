#!/usr/bin/env python
"""SDAA sweep for verified/stokes_second_problem (oscillating plate, D2Q9).

Runs a reduced-grid scan on the SDAA device, records device evidence,
and evaluates all acceptance criteria from the case README.

Grid choice: H = 20, 30, 40 (feasible smaller档 of the recommended 50/100/200).
The physics and criteria are identical; only the domain height is scaled.
kH = 6, 8 (same as the verified case).

Criteria (from README, fixed a priori):
  1. amplitude:  max |A_num/A_ana - 1| <= 3%  (rows A_ana >= 10% U)
  2. phase:      max |phi_num - phi_ana| / phi_ana <= 3%  (rows phi_ana >= 1 rad AND A_ana >= 10% U)
  3. complex amplitude: max |U_num/U_ana - 1| <= 3%  (same mask as amplitude)
  4. grid convergence: error decreases monotonically across the grid chain
     per kH line (metric: complex_rel_max_pct)

Device: torch.sdaa (Hygon DCU)
Compile mode: eager (SDAA teco_inductor cannot compile the full LBM step chain;
  compile_route.py has an automatic eager fallback, so we use eager directly).

Wall-clock target: <= 20 minutes.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Bootstrap tensorlbm import
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_BENCH_DIR = _REPO_ROOT / "benchmarks"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import numpy as np
import torch

# ---- Device setup ----
DEVICE = torch.device("sdaa")
torch.sdaa.set_device(0)

# ---- Import case modules ----
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.lid_driven_cavity import zou_he_moving_lid
from tensorlbm.solver import collide_bgk, stream

# ---- Constants from the case ----
SPECULAR = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)
AMP_MASK_FRAC = 0.10
PHASE_MASK_RAD = 1.0
MEASURE_PERIODS = 4
TOL_PCT = 3.0

# ---- Reduced grid scan (feasible smaller档 of 50/100/200) ----
H_LIST = [20, 30, 40]
KH_LIST = [6.0, 8.0]
TAU = 0.8
U = 0.05
NX = 8


def specular_replacement(f_pre: torch.Tensor) -> torch.Tensor:
    return f_pre[SPECULAR.to(f_pre.device)]


def run_case_sdaa(
    H: int,
    kH: float,
    tau: float = TAU,
    U: float = U,
    nx: int = NX,
    seed: int = 0,
) -> dict:
    """Run one Stokes-2 configuration on SDAA (eager mode)."""
    torch.manual_seed(seed)
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    k = kH / H
    omega = 2.0 * k * k * nu
    T = 2.0 * math.pi / omega
    T_int = int(round(T))
    tau_diff = H * H / (math.pi**2 * nu)
    skip_periods = max(10, math.ceil(8.0 * tau_diff / T))
    skip_steps = skip_periods * T_int
    n_meas = MEASURE_PERIODS * T_int
    total_steps = skip_steps + n_meas

    wall_bottom = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall_bottom[0, :] = True

    rho0 = torch.ones((ny, nx), device=DEVICE)
    u0 = torch.zeros((ny, nx), device=DEVICE)
    f = equilibrium(rho0, u0, u0)
    initial_mass = float(f.sum().item())

    Sc = torch.zeros(ny, device=DEVICE)
    Ss = torch.zeros(ny, device=DEVICE)
    snap_steps = [skip_steps + 1 + (m * T_int) // 8 for m in range(8)]
    snaps: dict[int, torch.Tensor] = {}

    t0 = time.time()
    for step in range(1, total_steps + 1):
        u_lid_t = torch.tensor(U * math.cos(omega * step), device=DEVICE)
        f_pre = f
        f = collide_bgk(f, tau)
        f = torch.where(wall_bottom.unsqueeze(0), specular_replacement(f_pre), f)
        f = stream(f)
        f = zou_he_moving_lid(f, u_lid_t)

        if step > skip_steps:
            _, ux, _ = macroscopic(f)
            col_mean = ux.mean(dim=1)
            Sc += col_mean * math.cos(omega * step)
            Ss += col_mean * math.sin(omega * step)
            if step in snap_steps:
                snaps[step] = col_mean.detach().clone()

        if step % max(1, total_steps // 10) == 0:
            print(f"  H={H} kH={kH}: step {step}/{total_steps}", flush=True)

    elapsed = time.time() - t0

    # ---- analysis (float64) ----
    ur = (2.0 / n_meas) * Sc.cpu().numpy().astype(np.float64)
    ui = (-(2.0 / n_meas)) * Ss.cpu().numpy().astype(np.float64)
    A_num = np.hypot(ur, ui)
    phi_num = -np.arctan2(ui, ur)

    rows = np.arange(ny)
    y = (ny - 1) - rows.astype(np.float64)
    A_ana = U * np.exp(-k * y)
    phi_ana = k * y
    fluid = (rows >= 1) & (rows <= ny - 2)

    amp_mask = fluid & (A_ana >= AMP_MASK_FRAC * U)
    phase_mask = amp_mask & (phi_ana >= PHASE_MASK_RAD)
    amp_rel = np.abs(A_num / A_ana - 1.0)[amp_mask] * 100.0
    dphi = np.angle(np.exp(1j * (phi_num - phi_ana)))
    phase_rel = (np.abs(dphi) / phi_ana)[phase_mask] * 100.0
    z_num = ur + 1j * ui
    z_ana = A_ana * np.exp(-1j * phi_ana)
    complex_rel = (np.abs(z_num / z_ana - 1.0))[amp_mask] * 100.0

    snap_rows = []
    for sstep in snap_steps:
        u_snap = snaps[sstep].cpu().numpy().astype(np.float64)
        u_ref = U * np.exp(-k * y) * np.cos(omega * sstep - k * y)
        m = fluid & (np.abs(u_ref) >= AMP_MASK_FRAC * U)
        l2 = float(np.linalg.norm(u_snap[fluid] - u_ref[fluid]) / np.linalg.norm(u_ref[fluid]))
        snap_rows.append({
            "step": sstep,
            "l2_rel": l2,
            "max_abs_over_U_pct": float(np.max(np.abs(u_snap[m] - u_ref[m])) / U * 100.0),
        })

    return {
        "H": H,
        "ny": ny,
        "nx": nx,
        "kH": kH,
        "k": k,
        "omega": omega,
        "T_int": T_int,
        "tau": tau,
        "nu_lb": nu,
        "U": U,
        "skip_periods": skip_periods,
        "skip_steps": skip_steps,
        "n_meas_steps": n_meas,
        "total_steps": total_steps,
        "amp_rel_max_pct": float(np.max(amp_rel)),
        "amp_rel_mean_pct": float(np.mean(amp_rel)),
        "phase_rel_max_pct": float(np.max(phase_rel)),
        "phase_err_max_deg": float(np.max(np.abs(dphi[phase_mask])) * 180.0 / math.pi),
        "complex_rel_max_pct": float(np.max(complex_rel)),
        "n_rows_amp_mask": int(amp_mask.sum()),
        "n_rows_phase_mask": int(phase_mask.sum()),
        "snapshot_check": snap_rows,
        "lid_row_amp_ratio": float(A_num[ny - 1] / U),
        "far_row_amp_over_U": float(A_num[1] / U),
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    print("=" * 70)
    print("SDAA Sweep: stokes_second_problem (oscillating plate, D2Q9)")
    print("=" * 70)

    # Device evidence
    device_name = torch.sdaa.get_device_name(0)
    device_count = torch.sdaa.device_count()
    print(f"Device: sdaa:0 -> {device_name}")
    print(f"Device count: {device_count}")
    print()

    # Run all configurations
    cases = []
    t_start = time.time()
    for kH in KH_LIST:
        for H in H_LIST:
            print(f"\n--- Running H={H}, kH={kH} ---", flush=True)
            r = run_case_sdaa(H, kH)
            cases.append(r)
            print(
                f"  H={H:4d} kH={kH:g}: amp={r['amp_rel_max_pct']:.3f}% "
                f"phase={r['phase_rel_max_pct']:.3f}% ({r['phase_err_max_deg']:.2f} deg) "
                f"complex={r['complex_rel_max_pct']:.3f}% steps={r['total_steps']} "
                f"elapsed={r['elapsed_s']}s",
                flush=True,
            )

    wall_clock = round(time.time() - t_start, 1)
    print(f"\nTotal wall clock: {wall_clock}s")

    # ---- Evaluate criteria ----
    amp_pass = all(r["amp_rel_max_pct"] <= TOL_PCT for r in cases)
    phase_pass = all(r["phase_rel_max_pct"] <= TOL_PCT for r in cases)
    complex_pass = all(r["complex_rel_max_pct"] <= TOL_PCT for r in cases)

    convergence_pass = True
    per_kH = []
    for kH in KH_LIST:
        rs = sorted([r for r in cases if r["kH"] == kH], key=lambda r: r["H"])
        errs = [r["complex_rel_max_pct"] for r in rs]
        decreased = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
        per_kH.append({
            "kH": kH,
            "H_list": [r["H"] for r in rs],
            "complex_rel_max_pct": errs,
            "monotonic_decrease": decreased,
        })
        if not decreased:
            convergence_pass = False

    all_pass = amp_pass and phase_pass and complex_pass and convergence_pass

    # ---- Build output JSON ----
    output = {
        "case": "stokes_second_problem",
        "device_evidence": {
            "device": "sdaa:0",
            "device_name": device_name,
            "device_count": device_count,
        },
        "config": {
            "lattice": "D2Q9",
            "collision": "bgk",
            "tau": TAU,
            "U": U,
            "nx": NX,
            "H_list": H_LIST,
            "kH_list": KH_LIST,
            "compile_mode": "eager",
            "grid_note": "Reduced grid H=20/30/40 (feasible smaller档 of recommended 50/100/200; SDAA kernel launch overhead dominates for small grids, making full grid chain exceed 20 min wall clock)",
        },
        "wall_clock_s": wall_clock,
        "triton_interpret": False,
        "triton_interpret_note": "No triton dependency in this case (pure D2Q9 BGK with library functions).",
        "criteria": {
            "amplitude_max_rel_pct_threshold": TOL_PCT,
            "phase_max_rel_pct_threshold": TOL_PCT,
            "complex_amplitude_max_rel_pct_threshold": TOL_PCT,
            "grid_convergence": "monotonic decrease of complex_rel_max_pct across H=20/30/40 per kH line",
        },
        "per_config": cases,
        "convergence": {
            "per_kH": per_kH,
        },
        "verdict": {
            "amplitude_pass": amp_pass,
            "phase_pass": phase_pass,
            "complex_amplitude_pass": complex_pass,
            "convergence_pass": convergence_pass,
            "all_pass": all_pass,
            "status": "PASS" if all_pass else "FAIL",
        },
    }

    # Write artifact
    artifact_path = _REPO_ROOT / "artifacts" / "sdaa_stokes_second_problem.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(output, indent=2))
    print(f"\nArtifact written to: {artifact_path}")

    # Print verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Amplitude <= {TOL_PCT}%:  {'PASS' if amp_pass else 'FAIL'}")
    for r in cases:
        print(f"    H={r['H']:4d} kH={r['kH']:g}: {r['amp_rel_max_pct']:.3f}%")
    print(f"  Phase <= {TOL_PCT}%:      {'PASS' if phase_pass else 'FAIL'}")
    for r in cases:
        print(f"    H={r['H']:4d} kH={r['kH']:g}: {r['phase_rel_max_pct']:.3f}% ({r['phase_err_max_deg']:.2f} deg)")
    print(f"  Complex <= {TOL_PCT}%:    {'PASS' if complex_pass else 'FAIL'}")
    for r in cases:
        print(f"    H={r['H']:4d} kH={r['kH']:g}: {r['complex_rel_max_pct']:.3f}%")
    print(f"  Convergence:          {'PASS' if convergence_pass else 'FAIL'}")
    for pk in per_kH:
        print(f"    kH={pk['kH']:g}: {pk['complex_rel_max_pct']} -> monotonic={pk['monotonic_decrease']}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Wall clock: {wall_clock}s")
    print(f"  Device: {device_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
