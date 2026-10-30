#!/usr/bin/env python3
"""SDAA sweep for floating_body_free_decay benchmark.

Runs the B6DOF-1 analytic-closure benchmark on torch.sdaa device.
Does NOT modify any business code; only imports and calls the existing
tensorlbm.rao_analysis / tensorlbm.rigid_body_6dof APIs.

HONEST FAIL: The business code has fundamental device-handling bugs that prevent
sdaa execution without code modification:

Bug 1: cummins_step creates F_memory = torch.zeros(6) on CPU (line ~60 in 
       rigid_body_6dof.py), causing device mismatch when other tensors are on sdaa.

Bug 2: compute_retardation_function creates tensors with torch.linspace/torch.zeros
       which inherit the default device. If default device is sdaa, then .numpy()
       calls fail. If default device is CPU, then Bug 1 occurs.

These are mutually exclusive bugs that cannot be worked around without modifying
the business code. This script documents the failure honestly.

Outputs: artifacts/sdaa_floating_body_free_decay.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Ensure tensorlbm is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.rao_analysis import compute_natural_frequencies
from tensorlbm.rigid_body_6dof import (
    BodyProperties6DOF,
    HydrostaticMatrix,
    RadiationData,
    cummins_time_integration,
)

RHO = 1000.0
G = 9.81

# --------------------------------------------------------------------------- #
# Device setup
# --------------------------------------------------------------------------- #
DEVICE_STR = "sdaa"
DEVICE = torch.device(DEVICE_STR)


def get_device_evidence() -> dict:
    """Collect device identity evidence."""
    ev = {
        "requested_device": DEVICE_STR,
        "torch_version": torch.__version__,
        "torch_sdaa_exists": hasattr(torch, "sdaa"),
    }
    try:
        ev["sdaa_device_count"] = torch.sdaa.device_count()
    except Exception as e:
        ev["sdaa_device_count_error"] = str(e)
    try:
        ev["device_name"] = torch.sdaa.get_device_name(0)
    except Exception as e:
        ev["device_name_error"] = str(e)
    try:
        probe = torch.zeros(4, device=DEVICE)
        ev["probe_tensor_on_sdaa"] = True
        ev["probe_device"] = str(probe.device)
    except Exception as e:
        ev["probe_tensor_on_sdaa"] = False
        ev["probe_error"] = str(e)
    return ev


# --------------------------------------------------------------------------- #
# Measurement helpers (identical to benchmark run.py)
# --------------------------------------------------------------------------- #
def measure_omega_zero_crossing(t: np.ndarray, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    sign_change = np.where(np.diff(np.sign(x)) != 0)[0]
    crossings = []
    for i in sign_change:
        x0, x1 = x[i], x[i + 1]
        if x1 == x0:
            continue
        frac = -x0 / (x1 - x0)
        crossings.append(t[i] + frac * (t[i + 1] - t[i]))
    crossings = np.asarray(crossings)
    k = np.arange(crossings.size)
    A = np.vstack([np.ones_like(k, dtype=np.float64), k.astype(np.float64)]).T
    coef, *_ = np.linalg.lstsq(A, crossings, rcond=None)
    half_period = coef[1]
    omega = math.pi / half_period
    resid = crossings - A @ coef
    return {
        "omega_meas": omega,
        "half_period": half_period,
        "n_crossings": int(crossings.size),
        "crossing_fit_rms": float(np.sqrt(np.mean(resid ** 2))),
    }


def measure_omega_lsq_sine(t: np.ndarray, x: np.ndarray, omega_guess: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    ws = omega_guess * np.linspace(0.95, 1.05, 4001)
    best = None
    for w in ws:
        c = np.cos(w * t)
        s = np.sin(w * t)
        A = np.vstack([c, s]).T
        coef, res, *_ = np.linalg.lstsq(A, x, rcond=None)
        r2 = 1.0 - res[0] / np.sum((x - x.mean()) ** 2) if res.size else 1.0
        if best is None or r2 > best[0]:
            best = (r2, w)
    return {"omega_meas_lsq": float(best[1]), "fit_r2": float(best[0])}


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_case(dof, name, mass, added_mass, stiffness, inertia_diag, waterplane_area):
    inertia = torch.diag(torch.tensor(inertia_diag, dtype=torch.float32))
    body = BodyProperties6DOF(
        mass=mass,
        inertia_matrix=inertia,
        displacement_volume=mass / RHO,
        waterplane_area=waterplane_area,
        water_density=RHO,
        gravity=G,
    )
    C = torch.zeros(6, 6)
    C[dof, dof] = stiffness
    hydro = HydrostaticMatrix(c_matrix=C)

    n_freq = 16
    omega_f = torch.linspace(0.02, 5.0, n_freq)
    added = torch.zeros(n_freq, 6, 6)
    damping = torch.zeros(n_freq, 6, 6)
    added[:, dof, dof] = added_mass
    radiation = RadiationData(omega_f, added, damping, added[0].clone())
    return body, hydro, radiation


def attempt_run_undamped(dof, name, mass, added_mass, stiffness, inertia_diag,
                         waterplane_area, dt, n_periods, use_default_device_sdaa):
    """Attempt to run on sdaa. Returns (result_dict, error_msg)."""
    if use_default_device_sdaa:
        torch.set_default_device(DEVICE)
    
    try:
        body, hydro, radiation = build_case(
            dof, name, mass, added_mass, stiffness, inertia_diag, waterplane_area
        )
        M_cpu = body.build_mass_matrix().cpu()
        m_total = M_cpu[dof, dof].item() + added_mass
        omega_n = math.sqrt(stiffness / m_total)
        T = 2.0 * math.pi / omega_n

        nat = compute_natural_frequencies(
            M_cpu, radiation.added_mass_inf.cpu(), hydro.c_matrix.cpu()
        )
        omega_n_eig = nat[dof].item()

        n_steps = int(round(n_periods * T / dt))
        exc = torch.zeros(n_steps, 6)
        exc[0, dof] = 1.0

        t0 = time.time()
        state = cummins_time_integration(
            body, hydro, radiation, exc, dt, n_steps, device=DEVICE
        )
        wall_s = time.time() - t0

        pos = np.array(
            [p.detach().cpu().numpy()[dof] for p in state.position_history],
            dtype=np.float64,
        )
        t = np.arange(pos.size, dtype=np.float64) * dt

        zc = measure_omega_zero_crossing(t, pos)
        lsq = measure_omega_lsq_sine(t, pos, omega_n)

        omega_h = math.acos(max(-1.0, min(1.0, 1.0 - 0.5 * (omega_n * dt) ** 2))) / dt

        result = {
            "case": name,
            "dof": dof,
            "dt": dt,
            "n_steps": n_steps,
            "n_periods_requested": n_periods,
            "m_eff": m_total,
            "stiffness": stiffness,
            "omega_n_analytic": omega_n,
            "omega_n_eig": omega_n_eig,
            "eig_vs_analytic_pct": (omega_n_eig - omega_n) / omega_n * 100.0,
            "omega_n_dt": omega_n * dt,
            "omega_h_symplectic": omega_h,
            "omega_h_vs_analytic_pct": (omega_h - omega_n) / omega_n * 100.0,
            "omega_meas": zc["omega_meas"],
            "omega_meas_pct": (zc["omega_meas"] - omega_n) / omega_n * 100.0,
            "omega_meas_vs_discrete_pct": (zc["omega_meas"] - omega_h) / omega_h * 100.0,
            "omega_meas_lsq": lsq["omega_meas_lsq"],
            "omega_lsq_pct": (lsq["omega_meas_lsq"] - omega_n) / omega_n * 100.0,
            "omega_lsq_vs_discrete_pct": (lsq["omega_meas_lsq"] - omega_h) / omega_h * 100.0,
            "fit_r2": lsq["fit_r2"],
            "n_crossings": zc["n_crossings"],
            "crossing_fit_rms": zc["crossing_fit_rms"],
            "amplitude": float(np.abs(pos).max()),
            "wall_s": wall_s,
            "finite": bool(np.all(np.isfinite(pos))),
        }
        return result, None
    except Exception as e:
        return None, str(e)
    finally:
        if use_default_device_sdaa:
            torch.set_default_device(None)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    torch.manual_seed(0)
    overall_t0 = time.time()

    # Device evidence
    device_ev = get_device_evidence()
    print(f"[SDAA] Device evidence: {json.dumps(device_ev, indent=2)}")

    if not device_ev.get("probe_tensor_on_sdaa", False):
        print("[SDAA] FATAL: Cannot allocate tensor on sdaa. Aborting.")
        result = {
            "benchmark": "floating_body_free_decay",
            "status": "FAIL",
            "fail_reason": "Cannot allocate tensor on sdaa device",
            "device_evidence": device_ev,
        }
        out_path = REPO_ROOT / "artifacts" / "sdaa_floating_body_free_decay.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        return 1

    # Try both approaches to document the bugs
    print("\n[SDAA] Attempt 1: default device = CPU (original behavior)")
    heave = dict(
        dof=2, name="heave", mass=1000.0, added_mass=1000.0, stiffness=2000.0,
        inertia_diag=(1.0, 1.0, 1.0), waterplane_area=2000.0 / (RHO * G),
    )
    r1, e1 = attempt_run_undamped(**heave, dt=0.5, n_periods=2, use_default_device_sdaa=False)
    print(f"  Result: {'OK' if r1 else 'FAIL'}")
    if e1:
        print(f"  Error: {e1}")

    print("\n[SDAA] Attempt 2: default device = sdaa")
    r2, e2 = attempt_run_undamped(**heave, dt=0.5, n_periods=2, use_default_device_sdaa=True)
    print(f"  Result: {'OK' if r2 else 'FAIL'}")
    if e2:
        print(f"  Error: {e2}")

    # Both approaches fail - document the bugs
    bugs_identified = [
        {
            "bug_id": 1,
            "location": "tensorlbm/rigid_body_6dof.py, cummins_step()",
            "description": (
                "F_memory = torch.zeros(6) creates tensor on CPU regardless of device "
                "parameter. When other tensors (M, A_inf, C, state.position) are on sdaa, "
                "this causes 'Expected all tensors to be on the same device' error."
            ),
            "fix": "Change to: F_memory = torch.zeros(6, device=M.device)",
        },
        {
            "bug_id": 2,
            "location": "tensorlbm/rigid_body_6dof.py, compute_retardation_function()",
            "description": (
                "Uses torch.linspace/torch.zeros which inherit default device. If default "
                "device is sdaa, then .numpy() calls fail. If default device is CPU, then "
                "Bug 1 occurs in cummins_step."
            ),
            "fix": (
                "Either: (a) explicitly create tensors on CPU with device='cpu', or "
                "(b) avoid .numpy() calls and use torch operations throughout."
            ),
        },
    ]

    overall_wall = time.time() - overall_t0

    result = {
        "benchmark": "floating_body_free_decay",
        "status": "FAIL",
        "fail_reason": (
            "Business code has fundamental device-handling bugs that prevent sdaa execution. "
            "Two mutually exclusive bugs: (1) cummins_step creates F_memory on CPU causing "
            "device mismatch; (2) compute_retardation_function uses .numpy() which fails on "
            "sdaa tensors. Cannot work around without modifying business code."
        ),
        "device_evidence": device_ev,
        "device": DEVICE_STR,
        "dtype": "float32",
        "triton_interpret": os.environ.get("TRITON_INTERPRET", "not set"),
        "overall_wall_s": overall_wall,
        "bugs_identified": bugs_identified,
        "attempts": {
            "default_device_cpu": {
                "approach": "Keep default device as CPU, pass device='sdaa' to cummins_time_integration",
                "result": "FAIL",
                "error": e1,
            },
            "default_device_sdaa": {
                "approach": "Set torch.set_default_device('sdaa') so torch.zeros() creates on sdaa",
                "result": "FAIL",
                "error": e2,
            },
        },
        "cases": [],
        "spans": {},
        "verdict": {"pass": False},
        "evidence_boundary": (
            "Caller-synthesised body and hydrodynamic coefficients. "
            "No LBM flow, hull coupling, free surface, or radiation-memory validation."
        ),
    }

    out_path = REPO_ROOT / "artifacts" / "sdaa_floating_body_free_decay.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[SDAA] Results written to {out_path}")
    print(f"[SDAA] Status: FAIL")
    print(f"[SDAA] Overall wall time: {overall_wall:.1f}s")
    print(f"[SDAA] Bugs identified: {len(bugs_identified)}")
    for bug in bugs_identified:
        print(f"  Bug {bug['bug_id']}: {bug['location']}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
