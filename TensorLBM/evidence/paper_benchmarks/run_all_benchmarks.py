#!/usr/bin/env python3
"""Unified benchmark runner for GAP-DIFFLBM-PAPER-EVIDENCE.

Runs ≥6 independent benchmarks with analytical/literature references,
computes error tables, and produces paper-level evidence artifacts.

Benchmarks:
1. Plane Poiseuille (2D) — analytical parabolic profile
2. Lid-driven cavity Re=100 — Ghia et al. (1982)
3. Womersley pulsatile flow α=5 — Womersley (1955) analytical
4. Circular Poiseuille — analytical parabolic profile
5. Square duct Poiseuille — Shah & London (1978)
6. Layered Poiseuille (2-layer) — analytical piecewise solution
7. ASBL suction boundary layer — analytical exponential profile
8. CDR diffusion-reaction Da=1 — analytical closed-form

Usage:
    python run_all_benchmarks.py --out artifacts/paper_evidence
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure we use the worktree's src
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_poiseuille_2d() -> dict:
    """Benchmark 1: 2D plane Poiseuille flow (body-force driven)."""
    from tensorlbm.ice_hydro.lbm_channel import ChannelConfig, D2Q9Channel

    G = 1.0e-5
    TAU = 0.8
    H = 16
    STEPS = 5000

    ny = H + 4
    cfg = ChannelConfig(
        nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
        cover_amplitude=0.0, cover_depth=float(ny - 3),
        bc="no_slip", forcing="guo",
    )
    ch = D2Q9Channel(cfg)
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size
    nu = (TAU - 0.5) / 3.0
    y = rows - 0.5
    u_ana = G / (2.0 * nu) * y * (n_fluid - y)
    u0 = np.zeros(ny)
    u0[rows] = u_ana
    ch.init_equilibrium(u0)

    acc = np.zeros(n_fluid)
    rho_acc = 0.0
    n_win = max(STEPS // 4, 1)
    n_acc = 0
    for step in range(1, STEPS + 1):
        ch.f = ch._stream_from(ch._collide(ch.f))
        if step > STEPS - n_win:
            rho = ch.f.sum(axis=0)
            ux = (ch.f * ch.CX[:, None, None]).sum(axis=0) / rho
            acc += ux[0, rows]
            rho_acc += float(rho[ch.fluid].mean())
            n_acc += 1

    u = acc / n_acc
    eps_inf = float(np.max(np.abs(u - u_ana)) / u_ana.max())
    eps_L2 = float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana))

    return {
        "name": "Plane Poiseuille 2D",
        "category": "analytical",
        "external_anchor": "Schlichting & Gersten (2017), Boundary-Layer Theory, §III.2.1",
        "params": {"H": H, "tau": TAU, "force": G, "steps": STEPS, "nu": nu},
        "ref_peak": float(u_ana.max()),
        "got_peak": float(u.max()),
        "eps_inf": eps_inf,
        "eps_L2": eps_L2,
        "passed": eps_inf < 0.03,
    }


def run_lid_cavity_re100() -> dict:
    """Benchmark 2: Lid-driven cavity Re=100 vs Ghia et al. (1982)."""
    import torch
    from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig, CavitySolver
    from tensorlbm.verification.lid_driven_cavity.reference_data import get_ghia_reference
    from tensorlbm.verification.lid_driven_cavity.verification import compute_errors

    cfg = CavityConfig(n=65, re=100, max_steps=10000, u_lid=0.1)
    solver = CavitySolver(cfg)
    for _ in range(10000):
        solver.step()

    y, u, x, v = solver.get_profiles()
    ref = get_ghia_reference(100)
    errors = compute_errors(y, u, x, v, ref)

    return {
        "name": "Lid-driven cavity Re=100",
        "category": "numerical_benchmark",
        "external_anchor": "Ghia, U., Ghia, K.N., & Shin, C.T. (1982). J. Comp. Phys. 48(3), 387-411",
        "params": {"Re": 100, "n": 65, "u_lid": 0.1},
        "ref_u_min": float(ref.u.min()),
        "got_u_min": float(u.min()),
        "l2_u": errors["l2_u"],
        "l2_v": errors["l2_v"],
        "max_u": errors["max_u"],
        "max_v": errors["max_v"],
        "passed": errors["l2_u"] < 0.15 and errors["l2_v"] < 0.15,
    }


def run_womersley_alpha5() -> dict:
    """Benchmark 3: Womersley pulsatile flow α=5."""
    from tensorlbm.verification.womersley_pulsatile.channel2d import run_case as run_channel

    r = run_channel(H=15, alpha=5.0, nx=8, skip_periods=8)
    m = r["metrics"]

    return {
        "name": "Womersley pulsatile flow α=5",
        "category": "analytical",
        "external_anchor": "Womersley, J.R. (1955). J. Physiol. 127:553-563",
        "params": {"alpha": 5.0, "H": 15, "measure_periods": 4},
        "amplitude_rel_pct": m["amplitude_rel_pct"],
        "phase_rel_pct": m["phase_rel_pct"],
        "profile_l2_pct": m["profile_l2_pct"],
        "passed": m["amplitude_rel_pct"] < 5.0 and m["phase_rel_pct"] < 5.0,
    }


def run_circular_poiseuille() -> dict:
    """Benchmark 4: Circular Poiseuille flow (Inamuro wet-node BC)."""
    import torch
    from tensorlbm.verification.circular_poiseuille.solver import CircularPoiseuilleConfig, CircularPoiseuilleSolver
    from tensorlbm.verification.circular_poiseuille.analytical import poiseuille_velocity_profile

    cfg = CircularPoiseuilleConfig(
        n=65, radius=25.0, body_force=1e-5, tau=1.0,
        max_steps=25000, use_inamuro=True
    )
    solver = CircularPoiseuilleSolver(cfg)
    solver.run(resid_tol=1e-7, monitor_interval=2000)

    r, u = solver.get_radial_profile()
    nu = cfg.nu
    u_max_ana = cfg.body_force * cfg.radius**2 / (4.0 * nu)
    u_ana = poiseuille_velocity_profile(r, cfg.radius, u_max_ana)

    mask = r <= cfg.radius
    l2 = float(np.linalg.norm(u[mask] - u_ana[mask]) / np.linalg.norm(u_ana[mask]))
    linf = float(np.max(np.abs(u[mask] - u_ana[mask])) / u_max_ana)

    return {
        "name": "Circular Poiseuille (Inamuro BC)",
        "category": "analytical",
        "external_anchor": "White, F.M. (2011). Fluid Mechanics, 7th ed., McGraw-Hill",
        "params": {"n": 65, "radius": 25.0, "tau": 1.0, "body_force": 1e-5},
        "ref_u_max": u_max_ana,
        "got_u_max": solver.compute_max_velocity(),
        "l2_relative": l2,
        "linf_relative": linf,
        "passed": linf < 0.10,
    }


def run_square_duct_poiseuille() -> dict:
    """Benchmark 5: Square duct Poiseuille flow (D3Q19)."""
    from tensorlbm.verification.square_duct_poiseuille.analytical import (
        shah_london_fRe, square_duct_velocity_profile, square_duct_mean_velocity
    )
    from tensorlbm.verification.square_duct_poiseuille.solver import SquareDuctConfig, SquareDuctSolver

    cfg = SquareDuctConfig(ny=32, nz=32, nx=8, tau=1.0, force=1e-5, steps=20000, warmup=5000)
    solver = SquareDuctSolver(cfg)
    solver.run()

    # Get velocity profile at mid-plane
    u_field = solver.get_velocity_field()
    u_mid = u_field[:, cfg.nz//2, cfg.ny//2]  # centerline velocity along x

    # Analytical f·Re
    fRe_ref = shah_london_fRe(1.0)

    # Compute mean velocity from simulation
    u_mean_sim = float(np.mean(u_field[solver.fluid]))

    # Analytical mean velocity
    a = float(cfg.ny - 2)
    dpdx = -cfg.force
    mu = cfg.nu  # rho=1
    u_mean_ana = square_duct_mean_velocity(a, dpdx, mu)

    rel_error = abs(u_mean_sim - u_mean_ana) / u_mean_ana

    return {
        "name": "Square duct Poiseuille (D3Q19)",
        "category": "analytical",
        "external_anchor": "Shah, R.K. & London, A.L. (1978). Laminar Flow Forced Convection in Ducts, Academic Press",
        "params": {"ny": 32, "nz": 32, "tau": 1.0, "force": 1e-5},
        "ref_fRe": fRe_ref,
        "ref_u_mean": u_mean_ana,
        "got_u_mean": u_mean_sim,
        "rel_error_mean": rel_error,
        "passed": rel_error < 0.15,
    }


def run_layered_poiseuille() -> dict:
    """Benchmark 6: Layered (2-layer) Poiseuille flow."""
    from tensorlbm.verification.layered_poiseuille import analytic as A
    from tensorlbm.verification.layered_poiseuille.solver import ChannelConfig, LayeredChannel
    from tensorlbm.verification.layered_poiseuille.metrics import profile_errors

    H = 32
    nus = (0.15, 0.05)
    fracs = (0.5,)
    steps = 8000

    cfg = ChannelConfig(height=H, nx=8, nus=nus, interface_fracs=fracs, u_peak=0.05,
                        x_boundary="periodic", init_mode="analytic", steps=steps,
                        sample_every=200, tol_steady=1e-12)
    unit = A.solve_layered(float(H), [f * H for f in fracs], nus, 1.0)
    ch = LayeredChannel(cfg, unit_profile=lambda y: unit.u(y))
    ch.run()

    ref = A.solve_layered(float(H), [f * H for f in fracs], nus, ch.g)
    u_sim = ch.profile()
    u_ana = ref.u(ch.y_wall)

    errs = profile_errors(u_sim, u_ana)

    return {
        "name": "Layered Poiseuille (2-layer, ν ratio=3)",
        "category": "analytical",
        "external_anchor": "Piecewise Poiseuille: d/dy(ν du/dy) = -g, continuity of u and ν·du/dy at interface",
        "params": {"H": H, "nus": nus, "interface_frac": 0.5, "steps": steps},
        "ref_u_peak": ref.u_peak(),
        "got_u_peak": float(np.max(u_sim)),
        "eps_inf": errs["eps_inf"],
        "eps_L2_rel": errs["eps_l2_rel"],
        "passed": errs["eps_inf"] < 0.03,
    }


def run_two_component_sc_layered_poiseuille() -> dict:
    """Benchmark: two-component Shan-Chen layered Poiseuille (MCMP, EDM forcing).

    Shares its scoring code with the anchor cross-validation entry point via
    :mod:`tensorlbm.verification.layered_poiseuille.benchmark`, so the numbers
    quoted here are produced by exactly the same comparison against the exact
    piecewise-quadratic layered solution.
    """
    from tensorlbm.verification.layered_poiseuille.benchmark import benchmark_single_case

    out = benchmark_single_case("layered_r167_light")
    rec = out["record"]
    m = rec["metrics"]
    return {
        "name": "Two-component Shan-Chen layered Poiseuille (nu ratio 1.67, EDM)",
        "category": "analytical",
        "external_anchor": "Shan & Chen (1993) PRE 47:1815; Martys & Chen (1996) PRE 53:743; "
                           "piecewise Poiseuille: d/dy(nu du/dy) = -g with u and nu*du/dy continuous",
        "params": dict(rec["config"]),
        "ref_u_peak": m["u_peak_ref"],
        "got_u_peak": m["u_peak_num"],
        "u_peak_err": m["u_peak_err"],
        "eps_inf": m["eps_inf"],
        "eps_L2_rel": m["eps_l2_rel"],
        "Q_err": m["Q_err"],
        "nu_fit_lower": m.get("nu_fit_lower"),
        "nu_fit_upper": m.get("nu_fit_upper"),
        "ref_nu": [rec["nu_a"], rec["nu_b"]],
        "y_iface_ref": m["y_iface_ref"],
        "y_iface": m["y_iface"],
        "gates": rec["gates"],
        "passed": bool(rec["gate_pass"]),
    }


def run_asbl() -> dict:
    """Benchmark 7: Asymptotic Suction Boundary Layer."""
    import torch
    from tensorlbm.verification.asbl.analytical import asbl_exact_velocity, asbl_skin_friction
    from tensorlbm.verification.asbl.simulation import run_asbl_case

    U_inf = 0.1
    V_suction = 0.001
    nu = 0.01
    nx, ny = 128, 64

    result = run_asbl_case(nx, ny, U_inf, V_suction, nu, max_steps=10000)

    # Compare velocity profile at mid-domain
    y = torch.linspace(0, ny - 1, ny, dtype=torch.float64)
    u_ana = asbl_exact_velocity(y, U_inf, V_suction, nu)
    u_sim = result["u_profile"]

    # Only compare in boundary layer region
    delta = nu / V_suction
    mask = y <= 5 * delta
    if mask.sum() > 0:
        l2 = float(torch.sqrt(torch.mean((u_sim[mask] - u_ana[mask])**2)) / torch.max(u_ana[mask]))
    else:
        l2 = 0.0

    cf_ref = asbl_skin_friction(U_inf, V_suction)
    cf_sim = result.get("cf_measured", 0.0)
    cf_error = abs(cf_sim - cf_ref) / cf_ref if cf_ref > 0 else 0.0

    return {
        "name": "ASBL (Asymptotic Suction Boundary Layer)",
        "category": "analytical",
        "external_anchor": "Schlichting, H. (1979). Boundary-Layer Theory, 7th ed., McGraw-Hill, §14.2",
        "params": {"U_inf": U_inf, "V_suction": V_suction, "nu": nu, "nx": nx, "ny": ny},
        "ref_Cf": cf_ref,
        "got_Cf": cf_sim,
        "Cf_error_pct": cf_error * 100,
        "profile_L2": l2,
        "passed": cf_error < 0.05 and l2 < 0.05,
    }


def run_cdr_da1() -> dict:
    """Benchmark 8: CDR diffusion-reaction Da=1."""
    from tensorlbm.verification.cdr_pnp.cdr_solver import (
        CDRConfig, analytical_cdr_1d, run_cdr_1d
    )

    Da = 1.0
    D = 0.01
    k = Da * D  # Da = k*L²/D, L=1
    cfg = CDRConfig(nx=256, L=1.0, D=D, u=0.0, k=k, n=1, c0=1.0, steps=10000, dt=0.001)

    result = run_cdr_1d(cfg)
    x = np.linspace(0, cfg.L, cfg.nx)
    c_ana = analytical_cdr_1d(x, D=D, k=k, c0=1.0, L=cfg.L)
    c_sim = result["c_final"]

    l2 = float(np.linalg.norm(c_sim - c_ana) / np.linalg.norm(c_ana))
    linf = float(np.max(np.abs(c_sim - c_ana)) / np.max(np.abs(c_ana)))

    return {
        "name": "CDR diffusion-reaction (Da=1)",
        "category": "analytical",
        "external_anchor": "Crank, J. (1975). The Mathematics of Diffusion, 2nd ed., Oxford University Press",
        "params": {"Da": Da, "D": D, "k": k, "nx": 256, "steps": 10000},
        "ref_c_max": float(c_ana.max()),
        "got_c_max": float(c_sim.max()),
        "l2_relative": l2,
        "linf_relative": linf,
        "passed": l2 < 0.03 and linf < 0.05,
    }


#: benchmark registry (name -> runner); ``main`` runs a (possibly sharded) subset.
BENCHMARKS = [
    ("poiseuille_2d", run_poiseuille_2d),
    ("lid_cavity_re100", run_lid_cavity_re100),
    ("womersley_alpha5", run_womersley_alpha5),
    ("circular_poiseuille", run_circular_poiseuille),
    ("square_duct_poiseuille", run_square_duct_poiseuille),
    ("layered_poiseuille", run_layered_poiseuille),
    ("two_component_sc_layered_poiseuille", run_two_component_sc_layered_poiseuille),
    ("asbl", run_asbl),
    ("cdr_da1", run_cdr_da1),
]


def main():
    parser = argparse.ArgumentParser(description="Run all paper-level benchmarks")
    parser.add_argument("--out", default="artifacts/paper_evidence", help="Output directory")
    parser.add_argument("--shard", default=None, help="Shard spec k/n (e.g. 0/3)")
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    benchmarks = list(BENCHMARKS)

    # Handle sharding
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        benchmarks = [b for i, b in enumerate(benchmarks) if i % n == k]

    results = {}
    summary = {
        "mission": "GAP-DIFFLBM-PAPER-EVIDENCE",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_benchmarks": len(benchmarks),
        "passed": 0,
        "failed": 0,
        "results": {},
    }

    for key, fn in benchmarks:
        print(f"\n{'='*60}")
        print(f"Running: {key}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            result["wall_time_s"] = round(elapsed, 2)
            results[key] = result
            status = "PASS" if result["passed"] else "FAIL"
            summary["results"][key] = {
                "status": status,
                "name": result["name"],
                "external_anchor": result["external_anchor"],
                "key_metrics": {k: v for k, v in result.items() if k not in ("name", "category", "external_anchor", "params", "wall_time_s")},
            }
            if result["passed"]:
                summary["passed"] += 1
            else:
                summary["failed"] += 1
            print(f"  Status: {status} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            summary["results"][key] = {"status": "ERROR", "error": str(e)}
            summary["failed"] += 1

    # Save results
    with open(outdir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Benchmark':<40} {'Status':<8} {'Key Metric':<25}")
    print(f"{'-'*60}")
    for key, res in summary["results"].items():
        name = res.get("name", key)[:38]
        status = res["status"]
        metrics = res.get("key_metrics", {})
        # Pick first numeric metric
        metric_str = ""
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_str = f"{k}={v:.4e}"[:23]
                break
        print(f"{name:<40} {status:<8} {metric_str}")
    print(f"{'-'*60}")
    print(f"Total: {summary['passed']} PASS, {summary['failed']} FAIL out of {summary['total_benchmarks']}")

    return summary


if __name__ == "__main__":
    main()
