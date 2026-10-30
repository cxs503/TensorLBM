#!/usr/bin/env python3
"""SDAA sweep — maneuvering_steady_turn benchmark on SDAA device.

This script runs the verified ``maneuvering_steady_turn`` benchmark
(``benchmarks/verified/maneuvering_steady_turn``) with all tensor computation
on the **sdaa** device, records device evidence, and writes the results
(including per-criterion PASS/FAIL) to
``artifacts/sdaa_maneuvering_steady_turn.json``.

Acceptance criteria (from the case README / result.json, original口径):
  C1  Each (rudder, dt, integrator) level: |R_err| <= 3 %
  C2  Each level: |beta_err| <= 3 %
  C3  Two-level span (coarse vs fine, per rudder): R_span <= 3 % and beta_span <= 3 %
  C4  Independent algebraic routes agree < 1e-6 %  (NumPy solve & Nomoto closed form)
  C5  Euler transient r(t) error at fine level <= 3 %
  C6  Transient refinement ratio (coarse/fine error) > 3
  C7  RK4 transient r(t) error at fine level <= 3 %
  C8  Directionally stable (all eigenvalues of M^-1 A have negative real part)

Device strategy:
  * ``torch.set_default_device('sdaa')`` + ``torch.set_default_dtype(float64)``
    so every ``torch.tensor(...)`` / ``torch.linalg.*`` call inside
    ``tensorlbm.maneuvering`` executes on sdaa.
  * The library's ``analytic_transient`` and the benchmark's cross-route
    verification call ``.numpy()`` on tensors.  On sdaa this raises, so we
    monkey-patch ``torch.Tensor.numpy`` to transparently ``.cpu().numpy()``.
    The torch linalg operations (solve, eigvals, matmul) still execute on sdaa;
    only the final host-side numpy extraction is routed through CPU.
    This workaround is recorded in the output JSON.

No triton dependency exists in this benchmark (pure torch + numpy), so
``TRITON_INTERPRET`` is not needed.

Configuration note:
  The verified benchmark uses T_END=600 s.  On sdaa, each time step incurs
  kernel-launch overhead for 2×2 tensor ops, making the full configuration
  exceed the 120 s shell timeout.  We reduce T_END to 300 s (still >7× the
  slowest modal time constant τ=42 s), which keeps all eight criteria
  testable while fitting within the wall-clock budget.  The steady-state
  measurement window (final 50 % of the run, t∈[150,300] s) captures
  >3.5 e-folding times of the slowest mode.

Run:
    python scripts/sdaa_sweep_maneuvering_steady_turn.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# 0.  Device setup — must happen before importing tensorlbm
# --------------------------------------------------------------------------- #
import torch

_DEVICE = "sdaa"
torch.set_default_dtype(torch.float64)

# Monkey-patch .numpy() so sdaa tensors are transparently moved to CPU first.
# The torch linalg ops (solve, eigvals, matmul) still run on sdaa; only the
# final host extraction goes through CPU.  Recorded as a workaround.
_orig_numpy = torch.Tensor.numpy


def _sdaa_safe_numpy(self, *args, **kwargs):
    if self.device.type != "cpu":
        return _orig_numpy(self.cpu(), *args, **kwargs)
    return _orig_numpy(self, *args, **kwargs)


torch.Tensor.numpy = _sdaa_safe_numpy

torch.set_default_device(_DEVICE)
# Force a dummy op to confirm the device is live
_probe = torch.tensor([1.0, 2.0])
assert _probe.device.type == "sdaa", f"expected sdaa, got {_probe.device}"

# Device evidence
_device_name = "UNKNOWN"
try:
    _device_name = torch.sdaa.get_device_name(0)
except Exception as exc:
    _device_name = f"ERROR: {exc}"

# --------------------------------------------------------------------------- #
# 1.  Import the benchmark library (same as run.py)
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[0]  # repo root
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from tensorlbm.maneuvering import (  # noqa: E402
    NOMINAL_TANKER_PRIME,
    ManeuveringModel,
    analytic_transient,
    prime_to_dimensional,
    simulate_turn,
)

# --------------------------------------------------------------------------- #
# 2.  Ship parameters (identical to the verified benchmark)
# --------------------------------------------------------------------------- #
L = 320.0       # length between perpendiculars [m]
B = 58.0        # beam [m]
D = 20.8        # draft [m]
CB = 0.81       # block coefficient
RHO = 1000.0    # water density [kg/m^3]
U = 7.97        # forward speed [m/s] (~15.5 kn)
MASS = RHO * CB * L * B * D
IZ = MASS * (0.25 * L) ** 2

RUDDERS_DEG = (10.0, 20.0)
DT_LEVELS = {"coarse": 1.0, "fine": 0.25}
# Reduced from 600 to 300 s for sdaa wall-clock feasibility (see docstring).
# Still >7× the slowest time constant (τ_max = 42.1 s).
T_END = 300.0
TOL = 3.0  # percent — same as the verified benchmark


def build_model() -> ManeuveringModel:
    dim = prime_to_dimensional(NOMINAL_TANKER_PRIME, L, D, U, RHO)
    return ManeuveringModel(
        L=L, d=D, U=U, mass=MASS, inertia_z=IZ,
        x_G=0.0, rho=RHO, name="kvlcc2-class tanker", **dim,
    )


# --------------------------------------------------------------------------- #
# 3.  Helper functions (mirrors run.py, adapted for sdaa tensors)
# --------------------------------------------------------------------------- #
def circle_fit_radius(x, y) -> float:
    """Kasa algebraic circle fit; returns the fitted radius."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx ** 2 + cy ** 2
    return float(math.sqrt(max(r2, 0.0)))


def steady_measure(sim: dict, t_frac: float = 0.5) -> dict:
    n = sim["n_steps"]
    i0 = int(t_frac * n)
    v = np.asarray(sim["v"][i0:], dtype=float)
    r = np.asarray(sim["r"][i0:], dtype=float)
    x = np.asarray(sim["x"][i0:], dtype=float)
    y = np.asarray(sim["y"][i0:], dtype=float)
    return {
        "v": float(v.mean()),
        "r": float(r.mean()),
        "v_std": float(v.std()),
        "r_std": float(r.std()),
        "R_circle": circle_fit_radius(x, y),
    }


def run_case(model, delta_deg, level, dt, integrator) -> dict:
    delta = math.radians(delta_deg)
    n_steps = int(round(T_END / dt))

    ana = model.steady_turn(delta)
    R_geo_ana = ana.speed / abs(ana.r)

    t0 = time.time()
    sim = simulate_turn(model, delta, dt, n_steps, integrator=integrator)
    wall = time.time() - t0

    m = steady_measure(sim)
    R_geo_meas = math.hypot(U, m["v"]) / abs(m["r"])
    beta_meas = math.degrees(math.atan2(-m["v"], U))

    # transient: analytic modal solution vs numerical r(t), v(t)
    t = np.asarray(sim["t"], dtype=float)
    v_a, r_a = analytic_transient(model, delta, t)
    v_n = np.asarray(sim["v"], dtype=float)
    r_n = np.asarray(sim["r"], dtype=float)
    r_scale = abs(ana.r)
    trans_r_err = float(np.max(np.abs(r_n - r_a)) / r_scale) * 100.0
    trans_v_err = float(np.max(np.abs(v_n - v_a)) / abs(ana.v)) * 100.0

    return {
        "level": level,
        "delta_deg": delta_deg,
        "integrator": integrator,
        "dt": dt,
        "n_steps": n_steps,
        "wall_s": wall,
        "v_ana": ana.v,
        "r_ana": ana.r,
        "R_ana": R_geo_ana,
        "R_over_L_ana": R_geo_ana / L,
        "beta_ana_deg": ana.beta_deg,
        "K_ana": ana.K,
        "v_meas": m["v"],
        "r_meas": m["r"],
        "R_meas": R_geo_meas,
        "R_over_L_meas": R_geo_meas / L,
        "beta_meas_deg": beta_meas,
        "R_circle_fit": m["R_circle"],
        "R_err_pct": (R_geo_meas - R_geo_ana) / R_geo_ana * 100.0,
        "beta_err_pct": (beta_meas - ana.beta_deg) / ana.beta_deg * 100.0,
        "R_circle_err_pct": (m["R_circle"] - R_geo_ana) / R_geo_ana * 100.0,
        "r_err_pct": (m["r"] - ana.r) / ana.r * 100.0,
        "steady_r_std": m["r_std"],
        "transient_r_err_pct": trans_r_err,
        "transient_v_err_pct": trans_v_err,
    }


# --------------------------------------------------------------------------- #
# 4.  Main
# --------------------------------------------------------------------------- #
def main() -> int:
    out_dir = _REPO / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sdaa_maneuvering_steady_turn.json"

    t_start = time.time()

    model = build_model()

    # ---- exact analytic fixed point + independent cross-routes ------------- #
    delta_ref = math.radians(20.0)
    A_np = model.damping_matrix().numpy()   # sdaa tensor -> cpu -> numpy
    b_np = model.input_vector(delta_ref).numpy()
    q_lib = np.array([model.steady_turn(delta_ref).v,
                      model.steady_turn(delta_ref).r])
    q_np = np.linalg.solve(A_np, -b_np)     # independent numpy route
    K_r = model.nomoto_K() * delta_ref      # independent Nomoto closed form
    route_agree_pct = {
        "numpy_vs_library_r": abs(q_np[1] - q_lib[1]) / abs(q_lib[1]) * 100.0,
        "nomoto_vs_library_r": abs(K_r - q_lib[1]) / abs(q_lib[1]) * 100.0,
    }

    modes = model.modes()
    eigenvalues = [float(e) for e in modes["eigenvalues"]]
    directionally_stable = all(e < 0 for e in eigenvalues)

    # ---- cases: 2 rudders x 2 dt levels x 2 integrators ------------------- #
    cases = []
    for ddeg in RUDDERS_DEG:
        for level, dt in DT_LEVELS.items():
            cases.append(run_case(model, ddeg, level, dt, "semi_implicit_euler"))
    for ddeg in RUDDERS_DEG:
        cases.append(run_case(model, ddeg, "fine_rk4", DT_LEVELS["fine"], "rk4"))

    # ---- per-rudder two-level span ---------------------------------------- #
    spans = {}
    for ddeg in RUDDERS_DEG:
        rs = [c for c in cases
              if c["delta_deg"] == ddeg and c["integrator"] == "semi_implicit_euler"]
        Rs = [c["R_meas"] for c in rs]
        betas = [c["beta_meas_deg"] for c in rs]
        te = {c["level"]: c["transient_r_err_pct"] for c in rs}
        ratio = (te["coarse"] / te["fine"]) if te.get("fine", 0) else float("nan")
        spans[f"{ddeg:g}deg"] = {
            "R_span_pct": (max(Rs) - min(Rs)) / np.mean(Rs) * 100.0,
            "beta_span_pct": (max(betas) - min(betas)) / abs(np.mean(betas)) * 100.0,
            "max_R_err_pct": max(abs(c["R_err_pct"]) for c in rs),
            "max_beta_err_pct": max(abs(c["beta_err_pct"]) for c in rs),
            "transient_r_err_coarse_pct": te.get("coarse"),
            "transient_r_err_fine_pct": te.get("fine"),
            "transient_refinement_ratio": ratio,
        }

    # ---- verdict (per-criterion PASS/FAIL) -------------------------------- #
    euler_fine = [c for c in cases
                  if c["integrator"] == "semi_implicit_euler" and c["level"] == "fine"]
    rk4_fine = [c for c in cases if c["integrator"] == "rk4"]

    c1_pass = all(abs(c["R_err_pct"]) <= TOL for c in cases)
    c2_pass = all(abs(c["beta_err_pct"]) <= TOL for c in cases)
    c3_pass = all(v["R_span_pct"] <= TOL and v["beta_span_pct"] <= TOL
                  for v in spans.values())
    c4_pass = all(v < 1e-6 for v in route_agree_pct.values())
    c5_pass = all(c["transient_r_err_pct"] <= TOL for c in euler_fine)
    c6_pass = all(v["transient_refinement_ratio"] > 3.0 for v in spans.values())
    c7_pass = all(c["transient_r_err_pct"] <= TOL for c in rk4_fine)
    c8_pass = directionally_stable

    criteria = [
        {"id": "C1", "name": "Each level |R_err| <= 3%",
         "pass": c1_pass,
         "max_value": max(abs(c["R_err_pct"]) for c in cases)},
        {"id": "C2", "name": "Each level |beta_err| <= 3%",
         "pass": c2_pass,
         "max_value": max(abs(c["beta_err_pct"]) for c in cases)},
        {"id": "C3", "name": "Two-level span <= 3% (R and beta)",
         "pass": c3_pass,
         "max_value": max(max(v["R_span_pct"], v["beta_span_pct"]) for v in spans.values())},
        {"id": "C4", "name": "Independent algebraic routes agree < 1e-6%",
         "pass": c4_pass,
         "max_value": max(route_agree_pct.values())},
        {"id": "C5", "name": "Euler transient r(t) err at fine <= 3%",
         "pass": c5_pass,
         "max_value": max(c["transient_r_err_pct"] for c in euler_fine)},
        {"id": "C6", "name": "Transient refinement ratio > 3",
         "pass": c6_pass,
         "max_value": min(v["transient_refinement_ratio"] for v in spans.values())},
        {"id": "C7", "name": "RK4 transient r(t) err at fine <= 3%",
         "pass": c7_pass,
         "max_value": max(c["transient_r_err_pct"] for c in rk4_fine)},
        {"id": "C8", "name": "Directionally stable",
         "pass": c8_pass,
         "max_value": max(eigenvalues) if eigenvalues else None},
    ]

    all_pass = all(c["pass"] for c in criteria)
    wall_total = time.time() - t_start

    # ---- assemble output --------------------------------------------------- #
    output = {
        "benchmark": "maneuvering_steady_turn",
        "sdaa_sweep": True,
        "device_evidence": {
            "device_type": _DEVICE,
            "device_name": _device_name,
            "torch_version": torch.__version__,
            "default_dtype": str(torch.get_default_dtype()),
            "probe_tensor_device": str(_probe.device),
            "get_device_name_call": "torch.sdaa.get_device_name(0)",
        },
        "workarounds": [
            {
                "what": "torch.Tensor.numpy monkey-patched to .cpu().numpy() for sdaa tensors",
                "why": "analytic_transient and cross-route verification call .numpy() on "
                       "tensors created on sdaa; torch does not support .numpy() on "
                       "non-CPU tensors",
                "impact": "All torch.linalg operations (solve, eigvals, matmul) execute "
                          "on sdaa. Only the final host-side numpy extraction is routed "
                          "through CPU. No numerical results are affected.",
            },
            {
                "what": "TRITON_INTERPRET not needed",
                "why": "This benchmark uses pure torch + numpy; no triton kernels.",
                "impact": "None",
            },
            {
                "what": "T_END reduced from 600 s to 300 s",
                "why": "sdaa kernel-launch overhead for 2x2 tensor ops makes the full "
                       "configuration exceed the 120 s shell timeout",
                "impact": "Still >7x the slowest modal time constant (tau=42.1 s). "
                          "Steady-state window captures >3.5 e-foldings. All eight "
                          "criteria remain testable.",
            },
        ],
        "ship": {
            "class": "KVLCC2-class large tanker (full scale)",
            "L_m": L, "B_m": B, "draft_m": D, "Cb": CB,
            "U_m_s": U, "mass_kg": MASS, "Iz_kg_m2": IZ,
        },
        "config": {
            "rudders_deg": list(RUDDERS_DEG),
            "dt_levels": DT_LEVELS,
            "T_end_s": T_END,
            "T_end_note": "reduced from 600 s (verified benchmark) to 300 s for sdaa "
                          "wall-clock feasibility; still >7x slowest time constant",
            "tolerance_pct": TOL,
            "integrators": ["semi_implicit_euler", "rk4"],
        },
        "derivatives": {
            "prime_input": NOMINAL_TANKER_PRIME,
            "dimensional": prime_to_dimensional(NOMINAL_TANKER_PRIME, L, D, U, RHO),
            "note": "representative KVLCC2-class linear derivative set (caller-supplied, "
                    "NOT extracted from an LBM flow or PMM experiment)",
        },
        "modes": {
            "eigenvalues": eigenvalues,
            "time_constants": modes["time_constants"],
            "directionally_stable": directionally_stable,
        },
        "analytic_route_agreement_pct": route_agree_pct,
        "cases": cases,
        "spans": spans,
        "criteria": criteria,
        "verdict": {
            "pass": all_pass,
            "all_criteria_pass": all_pass,
            "tolerance_pct": TOL,
        },
        "wall_clock_total_s": wall_total,
        "evidence_boundary": "Caller-supplied representative KVLCC2-class LINEAR "
                             "derivatives. No LBM flow, no free surface, no rudder/"
                             "propeller slipstream, no nonlinear cross-flow/higher-"
                             "order Abkowitz terms, no LBM end-to-end hull-in-flow "
                             "maneuvering. End-to-end fluid-maneuvering coupling is a "
                             "structural blocker (free surface).",
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ---- console summary --------------------------------------------------- #
    print(f"=== SDAA sweep: maneuvering_steady_turn ===")
    print(f"Device: {_device_name}  (torch {torch.__version__})")
    print(f"Wall clock: {wall_total:.2f} s")
    print(f"Criteria:")
    for c in criteria:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  {c['id']}  {status}  {c['name']}  (max/val={c['max_value']})")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"Output: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
