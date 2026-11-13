#!/usr/bin/env python3
"""XVAL driver: does the OSS (sailfish) pressure-tensor-equilibrium recipe, once
transplanted into TensorLBM, reproduce the free-energy mechanical pressure?

Arms
----
``oss``  the transplant (:mod:`tensorlbm.phasefield.pressure_tensor_equilibrium`):
         the Korteweg stress lives in the equilibrium distribution, so the
         mechanical pressure tensor is ``sum_i f_i c_ia c_ib - rho u_a u_b``.
``tl``   the TensorLBM force-based route (standard BGK equilibrium + explicit
         Korteweg force ``mu grad(phi)``, Guo forcing): the second moment of the
         populations is the *ideal-gas* moment, so the mechanical pressure is not
         in the populations at all.

Everything the report quotes is written to
``artifacts/xval_pressure_tensor_equilibrium/results.json``.

Run: ``python xval/XVAL-LBM-PRESSURE-TENSOR-EQUILIBRIUM-TRANSPLANT/run_xval.py``
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorlbm.phasefield.pressure_tensor_equilibrium import (  # noqa: E402
    C,
    WI_OSS,
    W_STD,
    analytic_sigma,
    bgk_equilibrium,
    chemical_potential,
    droplet_profile,
    flat_interface_profile,
    fluid_equilibrium,
    grad_lap,
    interface_width,
    moments,
    order_param_equilibrium,
    pressure_from_populations,
    pressure_tensor,
    quiescent_droplet,
    quiescent_flat_interface,
    run_quiescent_steps,
    sigma_from_flat_dip,
    sigma_from_radial_dip,
)

ART = os.path.join(ROOT, "artifacts", "xval_pressure_tensor_equilibrium")
PREDECL = json.load(open(os.path.join(HERE, "predeclared.json")))
OSS_COMMIT = PREDECL["oss_reference"]["commit"]


# ---------------------------------------------------------------------------
# stage 0: provenance of the vendored OSS sources
# ---------------------------------------------------------------------------
def stage_provenance() -> dict:
    ref_dir = os.path.join(HERE, "ref", "sailfish_src")
    files = {}
    for root, _dirs, names in os.walk(ref_dir):
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, HERE)
            files[rel] = {
                "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                "bytes": os.path.getsize(p),
            }
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "derive_oss_pressure_tensor.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    derived = json.load(open(os.path.join(HERE, "ref", "oss_discrete_pressure_tensor.json")))
    return {
        "oss_repo": PREDECL["oss_reference"]["repo"],
        "oss_commit": OSS_COMMIT,
        "vendored_files": files,
        "symbolic_residuals": derived,
        "derivation_stdout_tail": out.stdout.strip().splitlines()[-6:],
    }


# ---------------------------------------------------------------------------
# stage A: OSS transplant (equilibrium carries the pressure tensor)
# ---------------------------------------------------------------------------
def stage_oss_flat(xi_list, A=0.1) -> list[dict]:
    rows = []
    for xi_t in xi_list:
        kappa = A * xi_t**2 / 2.0
        r = quiescent_flat_interface(A=A, kappa=kappa, ny=96, nx=8)
        rows.append(
            {
                "xi_target": xi_t,
                "A": A,
                "kappa": kappa,
                "sigma_analytic": r.sigma_analytic,
                "sigma_measured": r.sigma_measured,
                "rel_error": r.rel_error,
                "dpnn_residual": r.dpnn_residual,
                "closure_residual": r.closure_residual,
                "max_speed": r.max_speed,
            }
        )
    return rows


def stage_oss_droplet(xi_list, radii, A=0.1) -> list[dict]:
    rows = []
    for xi_t in xi_list:
        kappa = A * xi_t**2 / 2.0
        for R in radii:
            r = quiescent_droplet(A=A, kappa=kappa, radius=R)
            rows.append(
                {
                    "xi": xi_t,
                    "kappa": kappa,
                    "radius": R,
                    "sigma_analytic": r.sigma_analytic,
                    "sigma_measured": r.sigma_measured,
                    "rel_error": r.rel_error,
                    "dp_bulk": r.dp_bulk,
                    "dp_curvature_bound": r.dp_curvature_bound,
                    "max_speed": r.max_speed,
                }
            )
    return rows


def stage_order_param_closure(A=0.1, kappa=None, Gamma=1.0) -> dict:
    kappa = A * 2.0**2 / 2.0 if kappa is None else kappa
    n = 64
    phi = droplet_profile(n, n, A=A, kappa=kappa, radius=16.0)
    rng = torch.Generator().manual_seed(18195)
    ux = 0.01 * torch.rand(n, n, generator=rng, dtype=torch.float64)
    uy = -0.01 * torch.rand(n, n, generator=rng, dtype=torch.float64)
    mu = chemical_potential(phi, A=A, kappa=kappa)
    g = order_param_equilibrium(phi, ux, uy, mu, Gamma=Gamma)
    shape = phi.shape
    cxp = torch.tensor([c[0] for c in C], dtype=torch.float64).view(-1, *([1] * len(shape)))
    cyp = torch.tensor([c[1] for c in C], dtype=torch.float64).view(-1, *([1] * len(shape)))
    return {
        "sum_g_minus_phi": float((g.sum(dim=0) - phi).abs().max()),
        "sum_g_cx_minus_phi_ux": float((g * cxp).sum(dim=0).sub(phi * ux).abs().max()),
        "sum_g_cy_minus_phi_uy": float((g * cyp).sum(dim=0).sub(phi * uy).abs().max()),
    }


def stage_tau_sweep(tau_list, A=0.1, xi_t=2.0, radius=16.0, steps=300) -> list[dict]:
    kappa = A * xi_t**2 / 2.0
    rows = []
    for tau in tau_list:
        try:
            r = run_quiescent_steps(A=A, kappa=kappa, radius=radius, steps=steps, tau=tau)
            r["nan"] = False
        except RuntimeError as exc:  # pragma: no cover - divergence guard
            r = {"steps": steps, "tau": tau, "nan": True, "error": str(exc)[:120]}
        rows.append(r)
    return rows


def stage_step_series(A=0.1, xi_t=2.0, radius=16.0, tau=0.8) -> list[dict]:
    """Spurious-current vs time at a finite relaxation time.

    ``tau = 1`` is degenerate (collision sets ``f = f^eq`` in one sweep, so the
    state is the fixed point ``S(f^eq)`` from step 1); ``tau = 0.8`` is used here
    so the series actually probes the relaxation.
    """
    kappa = A * xi_t**2 / 2.0
    return [
        run_quiescent_steps(A=A, kappa=kappa, radius=radius, steps=n, tau=tau)
        for n in (0, 1, 50, 200, 800)
    ]


def stage_A_sweep(A_list, xi_t=2.0) -> list[dict]:
    rows = []
    for A in A_list:
        kappa = A * xi_t**2 / 2.0
        r = quiescent_flat_interface(A=A, kappa=kappa, ny=96, nx=8)
        rows.append({"A": A, "kappa": kappa, "sigma_analytic": r.sigma_analytic,
                     "sigma_measured": r.sigma_measured, "rel_error": r.rel_error})
    return rows


# ---------------------------------------------------------------------------
# stage B: TensorLBM force-based control arm (same fields, same measurement)
# ---------------------------------------------------------------------------
def stage_tl_force_arm(A=0.1, xi_t=2.0, radius=16.0, tau=0.9) -> dict:
    """Standard BGK equilibrium + explicit Korteweg force ``F = mu grad(phi)``.

    The Guo forcing term is added exactly as the repository's force-based chain
    does (``phasefield/free_energy.py::force_mu_grad_phi`` gives the force field;
    here it is applied with the standard ``(1 - omega/2)`` Guo weight).  The
    question of this arm is not accuracy but *capability*: can the mechanical
    pressure be read off the populations?
    """
    kappa = A * xi_t**2 / 2.0
    xi = interface_width(A, kappa)
    n = int(2 * (radius + 5 * xi))
    phi = droplet_profile(n, n, A=A, kappa=kappa, radius=radius)
    gx, gy, _ = grad_lap(phi)
    mu = chemical_potential(phi, A=A, kappa=kappa)
    fx, fy = mu * gx, mu * gy  # Korteweg body force (repo: force_mu_grad_phi)
    rho = torch.ones_like(phi)
    ux = torch.zeros_like(phi)
    uy = torch.zeros_like(phi)
    feq = bgk_equilibrium(rho, ux, uy)
    shape = phi.shape
    cxp = torch.tensor([c[0] for c in C], dtype=torch.float64).view(-1, *([1] * len(shape)))
    cyp = torch.tensor([c[1] for c in C], dtype=torch.float64).view(-1, *([1] * len(shape)))
    w = W_STD.view(-1, *([1] * len(shape)))
    force_i = (1.0 - 0.5 / tau) * 3.0 * w * (cxp * fx.unsqueeze(0) + cyp * fy.unsqueeze(0))
    f = feq + 0.5 * force_i  # f = feq + dt/2 F_i (u = 0 start)
    pxx, pyy, _ = pressure_from_populations(f)
    sigma_meas = sigma_from_flat_dip(pxx, pyy, axis="y")
    p = 0.5 * (pxx + pyy)
    yy = torch.arange(n, dtype=torch.float64).view(-1, 1)
    xx = torch.arange(n, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((yy - (n - 1) / 2.0) ** 2 + (xx - (n - 1) / 2.0) ** 2)
    inside, outside = r < radius - 3 * xi, r > radius + 3 * xi
    sig = analytic_sigma(A, kappa)
    return {
        "kappa": kappa,
        "radius": radius,
        "tau": tau,
        "sigma_analytic": sig,
        # pressure from the populations of the force-based scheme:
        "sigma_measured_from_populations": sigma_meas,
        "rel_error_from_populations": (sigma_meas - sig) / sig,
        "dp_bulk_from_populations": float(p[inside].mean() - p[outside].mean()),
        # what the force-based route can offer instead: the Korteweg stress as a
        # post-hoc reconstruction (NOT a measurement from the populations)
        "sigma_analytic_same_formula": sig,
        # how far the population pressure is from the ideal-gas moment inside the
        # interface: the force term only shifts it by the half-force increment
        "population_pressure_deviation_from_ideal_gas": float((pxx - rho / 3.0).abs().max()),
    }


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
def verdicts(res: dict) -> dict:
    th = PREDECL["predeclared_thresholds"]
    flat = res["oss_flat"]
    resolved = [r for r in flat if r["xi_target"] >= 2.0]
    g1 = all(r["closure_residual"] < th["closure_residual_max"] for r in flat)
    g2 = all(abs(r["rel_error"]) < th["sigma_rel_error_max"] for r in resolved)
    g2_target = all(abs(r["rel_error"]) < th["sigma_rel_error_target"] for r in resolved)
    g3 = (
        res["order_param_closure"]["sum_g_minus_phi"] < th["order_param_closure_max"]
        and res["order_param_closure"]["sum_g_cx_minus_phi_ux"] < th["order_param_closure_max"]
        and res["order_param_closure"]["sum_g_cy_minus_phi_uy"] < th["order_param_closure_max"]
    )
    g4 = max(abs(r["rel_error"]) for r in res["oss_droplet"]) < th["sigma_rel_error_max"]
    return {
        "G1_closure_identity": bool(g1),
        "G2_sigma_flat_under_15pct": bool(g2),
        "G2b_sigma_flat_under_3pct_target": bool(g2_target),
        "G3_order_param_closure": bool(g3),
        "G4_sigma_droplet_under_15pct": bool(g4),
        "verdict": "PASS" if (g1 and g2 and g3 and g4) else "FAIL",
        "worst_resolved_sigma_error": max(abs(r["rel_error"]) for r in flat if r["xi_target"] >= 2.0),
        "xi1_under_resolution_error": min(r["rel_error"] for r in flat),
    }


def main() -> int:
    t0 = time.time()
    os.makedirs(ART, exist_ok=True)
    res: dict = {
        "anchor": PREDECL["anchor"],
        "mission": PREDECL["mission"],
        "predeclared": PREDECL,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
        },
        "stages": {},
    }

    res["provenance"] = stage_provenance()

    res["oss_flat"] = stage_oss_flat([1.0, 1.5, 2.0, 3.0, 4.0])
    res["oss_droplet"] = stage_oss_droplet([2.0, 3.0], [10.0, 12.0, 16.0, 24.0])
    res["order_param_closure"] = stage_order_param_closure()
    res["step_series"] = stage_step_series()
    res["tau_sweep"] = stage_tau_sweep([0.6, 0.8, 1.0, 1.5])
    res["A_sweep"] = stage_A_sweep([0.05, 0.1, 0.2])
    res["tl_force_arm"] = stage_tl_force_arm()
    res["verdicts"] = verdicts(res)
    res["wall_seconds"] = round(time.time() - t0, 1)

    out = os.path.join(ART, "results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(json.dumps(res["verdicts"], indent=2))
    print("wrote", out, f"({res['wall_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())