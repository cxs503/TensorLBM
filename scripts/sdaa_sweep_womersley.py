#!/usr/bin/env python3
"""SDAA sweep: Womersley oscillating channel flow (D2Q9) — verified benchmark.

Independent run script for the TensorLBM womersley benchmark on SDAA device.
Does NOT modify any business code; only orchestrates the run and records results.

Criteria (from benchmarks/verified/womersley/README.md, fixed a priori):
  1. Profile L2 error (fixed-scale) <= 3% at each of 8 phase instants
  2. Max pointwise error / u_peak <= 3% (8 phases)
  3. Per-row complex-amplitude |U_num/U_ana - 1| <= 3% (|U_ana| >= 10% u_peak)
  4. Error decreases from H to ~2H for both alpha in {4, 8}

Grid: H=29 and H=59 (2x ratio, smaller than the original H=59/119 to fit
wall-clock budget).  All other parameters follow the case documentation.

If torch.compile fails on SDAA (expected: teco_inductor incomplete), the
compile_route auto-fallback to eager is used and recorded.  TRITON_INTERPRET=1
is set as a precaution but does not bypass the inductor backend gap.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# ── TRITON_INTERPRET bypass (precautionary; does not fix teco inductor gap) ──
os.environ["TRITON_INTERPRET"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import numpy as np
import torch

from compile_route import compile_status_of, route_step

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream
from tensorlbm.turbulent_channel import _apply_body_force_2d

# ── Constants from the verified benchmark ──
MEASURE_PERIODS = 4
N_PHASES = 8
TAU = 0.8
U_PEAK_TARGET = 0.03
NX = 16
TOL_PCT = 3.0
H_LIST = [29, 59]
ALPHA_LIST = [4.0, 8.0]


def womersley_phasor(
    y_phys: np.ndarray, H: float, alpha: float, a0: float, omega: float
) -> np.ndarray:
    """Complex amplitude U(y) of the Womersley analytic solution."""
    lam = np.sqrt(1j) * alpha
    yp = (y_phys - H / 2.0) / (H / 2.0)
    return (a0 / (1j * omega)) * (1.0 - np.cosh(lam * yp) / np.cosh(lam))


def run_single(
    H: int,
    alpha: float,
    device: torch.device,
    compile_mode: str | None = "default",
) -> dict:
    """Run one (H, alpha) configuration and return the result dict."""
    torch.manual_seed(0)
    ny = H + 2
    nu = (TAU - 0.5) / 3.0
    H_eff = float(ny - 2)
    omega = 4.0 * alpha * alpha * nu / (H_eff * H_eff)
    T = 2.0 * math.pi / omega
    T_int = int(round(T))

    rows = np.arange(1, ny - 1, dtype=np.float64)
    y_phys = rows - 0.5
    lam = np.sqrt(1j) * alpha
    yp = (y_phys - H_eff / 2.0) / (H_eff / 2.0)
    shape_max = float(np.max(np.abs(1.0 - np.cosh(lam * yp) / np.cosh(lam))))
    a0 = U_PEAK_TARGET * omega / shape_max
    u_peak_ana = a0 * shape_max / omega

    tau_diff = H_eff * H_eff / (math.pi**2 * nu)
    skip_periods = max(5, math.ceil(6.5 * tau_diff / T))
    skip_steps = skip_periods * T_int
    n_meas = MEASURE_PERIODS * T_int
    total_steps = skip_steps + n_meas
    Ma = U_PEAK_TARGET / math.sqrt(1.0 / 3.0)

    wall = torch.zeros((ny, NX), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    rho0 = torch.ones((ny, NX), device=device)
    u0 = torch.zeros((ny, NX), device=device)
    f = equilibrium(rho0, u0, u0)
    initial_mass = float(f.sum().item())

    opposite = OPPOSITE.to(device)

    def _step(f: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        f_pre = f
        f = collide_bgk(f, TAU)
        f = torch.where(wall.unsqueeze(0), f_pre[opposite], f)
        f = _apply_body_force_2d(f, a_t)
        return stream(f)

    step_fn = route_step(_step, compile_mode, name=f"womersley[H{H}_a{alpha:g}]")

    Sc = torch.zeros(ny, device=device)
    Ss = torch.zeros(ny, device=device)
    snap_steps = [skip_steps + 1 + (m * T_int) // N_PHASES for m in range(N_PHASES)]
    snaps: dict[int, torch.Tensor] = {}

    t0 = time.time()
    for step in range(1, total_steps + 1):
        a_t = torch.tensor(a0 * math.cos(omega * step), device=device)
        f = step_fn(f, a_t)
        if step > skip_steps:
            _, ux, _ = macroscopic(f)
            col_mean = ux.mean(dim=1)
            Sc += col_mean * math.cos(omega * step)
            Ss += col_mean * math.sin(omega * step)
            if step in snap_steps:
                snaps[step] = col_mean.detach().clone()
    elapsed = time.time() - t0

    compile_info = compile_status_of(step_fn)

    # ── Analysis (float64) ──
    ur = (2.0 / n_meas) * Sc.cpu().numpy().astype(np.float64)
    ui = (-(2.0 / n_meas)) * Ss.cpu().numpy().astype(np.float64)
    z_num = (ur + 1j * ui)[1 : ny - 1]
    z_ana = womersley_phasor(y_phys, H_eff, float(alpha), a0, omega)

    row_mask = np.abs(z_ana) >= 0.10 * u_peak_ana
    complex_rel = np.abs(z_num / z_ana - 1.0)[row_mask] * 100.0

    # Fixed-scale L2 normalisation
    _a, _b = np.real(z_ana), np.imag(z_ana)
    _na, _nb, _ab = float(_a @ _a), float(_b @ _b), float(_a @ _b)
    _tr, _det = _na + _nb, _na * _nb - _ab * _ab
    ref_norm_peak = math.sqrt(0.5 * (_tr + math.sqrt(max(_tr * _tr - 4.0 * _det, 0.0))))

    phase_rows = []
    l2_list, l2p_list, ptp_list = [], [], []
    for sstep in snap_steps:
        u_snap = snaps[sstep].cpu().numpy().astype(np.float64)[1 : ny - 1]
        u_ref = np.real(z_ana * np.exp(1j * omega * sstep))
        du = u_snap - u_ref
        l2 = float(np.linalg.norm(du) / np.linalg.norm(u_ref))
        l2p = float(np.linalg.norm(du) / ref_norm_peak)
        ptp = float(np.max(np.abs(du)) / u_peak_ana)
        l2_list.append(l2)
        l2p_list.append(l2p)
        ptp_list.append(ptp)
        phase_rows.append({
            "step": sstep,
            "phase_deg": round(math.degrees((omega * sstep) % (2 * math.pi)), 2),
            "l2_rel_instant": l2,
            "l2_rel_peaknorm": l2p,
            "u_ref_norm": float(np.linalg.norm(u_ref)),
            "max_abs_over_upeak": ptp,
        })

    worst_l2 = float(np.max(l2_list))
    worst_l2p = float(np.max(l2p_list))
    worst_ptp = float(np.max(ptp_list))

    center_row = ny // 2
    zc_num = (ur + 1j * ui)[center_row]
    zc_ana = womersley_phasor(np.array([H_eff / 2.0]), H_eff, float(alpha), a0, omega)[0]

    final_mass = float(f.sum().item())
    mass_drift = abs(final_mass - initial_mass) / initial_mass * 100.0
    all_finite = bool(torch.isfinite(f).all().item())

    return {
        "H": H,
        "ny": ny,
        "nx": NX,
        "H_eff": H_eff,
        "alpha": alpha,
        "omega": omega,
        "T_int": T_int,
        "tau": TAU,
        "nu_lb": nu,
        "a0": a0,
        "u_peak_ana": u_peak_ana,
        "Ma": Ma,
        "skip_periods": skip_periods,
        "skip_steps": skip_steps,
        "measure_periods": MEASURE_PERIODS,
        "n_meas_steps": n_meas,
        "total_steps": total_steps,
        "profile_l2_instant_max_pct": worst_l2 * 100.0,
        "profile_l2_max_pct": worst_l2p * 100.0,
        "profile_pointwise_max_over_upeak_pct": worst_ptp * 100.0,
        "complex_rel_max_pct": float(np.max(complex_rel)),
        "center_complex_rel_pct": float(abs(zc_num / zc_ana - 1.0) * 100.0),
        "mass_drift_pct": mass_drift,
        "finite": all_finite,
        "elapsed_s": round(elapsed, 2),
        "compile_info": compile_info,
        "phase_table": phase_rows,
    }


def main() -> None:
    device = torch.device("sdaa")

    # ── Device evidence ──
    device_name = "UNKNOWN"
    try:
        device_name = torch.sdaa.get_device_name(0)
    except Exception as e:
        device_name = f"ERROR: {e}"

    print(f"=== SDAA Womersley Sweep ===", flush=True)
    print(f"Device: {device_name}", flush=True)
    print(f"Grids: H={H_LIST}, alpha={ALPHA_LIST}", flush=True)
    print(f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET', 'not set')}", flush=True)

    cases = []
    for alpha in ALPHA_LIST:
        for H in H_LIST:
            print(f"\n--- H={H} alpha={alpha} ---", flush=True)
            r = run_single(H, alpha, device, compile_mode="default")
            cases.append(r)
            print(
                f"  l2(fixed)={r['profile_l2_max_pct']:.3f}% "
                f"ptp={r['profile_pointwise_max_over_upeak_pct']:.3f}% "
                f"complex={r['complex_rel_max_pct']:.3f}% "
                f"steps={r['total_steps']} "
                f"elapsed={r['elapsed_s']}s "
                f"compile={r['compile_info']['compile_status']}",
                flush=True,
            )

    # ── Convergence check: error decreases from H to ~2H ──
    convergence = []
    for alpha in ALPHA_LIST:
        rs = sorted([r for r in cases if r["alpha"] == alpha], key=lambda r: r["H"])
        e_lo, e_hi = rs[0]["profile_l2_max_pct"], rs[-1]["profile_l2_max_pct"]
        conv = bool(e_hi < e_lo)
        convergence.append({
            "alpha": alpha,
            "H_list": [r["H"] for r in rs],
            "l2_max_pct": [r["profile_l2_max_pct"] for r in rs],
            "complex_max_pct": [r["complex_rel_max_pct"] for r in rs],
            "err_decreased": conv,
        })
        print(f"  alpha={alpha}: l2 {rs[0]['profile_l2_max_pct']:.3f}% -> {rs[-1]['profile_l2_max_pct']:.3f}% "
              f"converged={conv}", flush=True)

    # ── Per-case PASS/FAIL against the 4 criteria ──
    verdicts = []
    for r in cases:
        c1 = r["profile_l2_max_pct"] <= TOL_PCT
        c2 = r["profile_pointwise_max_over_upeak_pct"] <= TOL_PCT
        c3 = r["complex_rel_max_pct"] <= TOL_PCT
        verdicts.append({
            "H": r["H"],
            "alpha": r["alpha"],
            "c1_l2_fixed_scale": {"value_pct": r["profile_l2_max_pct"], "threshold_pct": TOL_PCT, "pass": c1},
            "c2_pointwise_over_upeak": {"value_pct": r["profile_pointwise_max_over_upeak_pct"], "threshold_pct": TOL_PCT, "pass": c2},
            "c3_complex_amplitude": {"value_pct": r["complex_rel_max_pct"], "threshold_pct": TOL_PCT, "pass": c3},
        })

    # Criterion 4: convergence per alpha
    c4_pass = all(c["err_decreased"] for c in convergence)
    for c in convergence:
        print(f"  alpha={c['alpha']} convergence: {c['err_decreased']}", flush=True)

    all_tol_pass = all(v["c1_l2_fixed_scale"]["pass"] and v["c2_pointwise_over_upeak"]["pass"] and v["c3_complex_amplitude"]["pass"] for v in verdicts)
    overall_pass = all_tol_pass and c4_pass

    # ── Build output JSON ──
    output = {
        "case": "womersley",
        "device": "sdaa",
        "device_name": device_name,
        "triton_interpret": os.environ.get("TRITON_INTERPRET", "not set"),
        "compile_bypass_note": (
            "torch.compile fails on SDAA (teco_inductor incomplete); "
            "compile_route auto-fallback to eager is used. "
            "TRITON_INTERPRET=1 set as precaution but does not fix the inductor backend gap."
        ),
        "grid_H_list": H_LIST,
        "alpha_list": ALPHA_LIST,
        "tau": TAU,
        "u_peak_target": U_PEAK_TARGET,
        "nx": NX,
        "tol_pct": TOL_PCT,
        "per_grid": cases,
        "convergence": convergence,
        "c4_err_decreased": c4_pass,
        "verdicts": verdicts,
        "all_tol_pass": all_tol_pass,
        "overall_pass": overall_pass,
        "status": "PASS" if overall_pass else "FAIL",
    }

    out_path = REPO_ROOT / "artifacts" / "sdaa_womersley.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}", flush=True)
    print(f"Overall: {output['status']}", flush=True)


if __name__ == "__main__":
    main()
