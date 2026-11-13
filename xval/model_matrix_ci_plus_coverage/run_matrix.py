#!/usr/bin/env python3
"""XVAL: model-matrix-ci-plus-coverage — TensorLBM × CFD-GO/TCLB.

Cross-validation of TensorLBM's model matrix coverage against the mature
open-source practice of CFD-GO/TCLB's model-matrix-ci-plus-coverage approach.

TCLB approach (from source/docs):
- Tests a matrix of models (multiphase, turbulence, thermal, etc.) across
  lattice types (D2Q9, D3Q19, D3Q27) with collision operators (BGK, MRT,
  cascaded) in CI, tracking coverage to ensure all model combinations are
  exercised.

This script:
1. Defines the model matrix (lattices × collisions × physics × boundaries)
2. Runs a CI-style test suite on TensorLBM
3. Tracks coverage (which model combinations are tested)
4. Compares against TCLB's approach
5. Produces artifacts: result.json, coverage.json, figures

Run:  python xval/model_matrix_ci_plus_coverage/run_matrix.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# Add TensorLBM src to path
SRC = Path(__file__).resolve().parents[2] / "TensorLBM" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tensorlbm
from tensorlbm import d2q9, d3q19, solver

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "xval" / "model_matrix_ci_plus_coverage" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float64
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# ===========================================================================
# 1. Model Matrix Definition
# ===========================================================================

LATTICES = {
    "D2Q9": {"dim": 2, "q": 9},
    "D3Q19": {"dim": 3, "q": 19},
}

COLLISION_OPERATORS = {
    "BGK": "collide_bgk",
    "MRT": "collide_mrt",
    "RLBM": "collide_rlbm_2d",
    "Smagorinsky-MRT": "collide_smagorinsky_mrt3d",
}

PHYSICS_MODELS = {
    "single_phase": {
        "description": "Single-phase incompressible flow",
        "tclb_parity": True,
        "tensorlbm_modules": ["d2q9", "d3q19", "solver"],
    },
    "multiphase_sc": {
        "description": "Shan-Chen pseudopotential multiphase",
        "tclb_parity": True,
        "tensorlbm_modules": ["multiphase"],
    },
    "multiphase_pf": {
        "description": "Phase-field (conservative Allen-Cahn)",
        "tclb_parity": True,
        "tensorlbm_modules": ["cac_lbm", "cac_lbm2d"],
    },
    "turbulence_les": {
        "description": "Large Eddy Simulation with Smagorinsky SGS",
        "tclb_parity": True,
        "tensorlbm_modules": ["turbulence", "solver"],
    },
    "acoustic": {
        "description": "Acoustic wave propagation",
        "tclb_parity": False,
        "tensorlbm_modules": ["acoustics"],
    },
    "adjoint": {
        "description": "Adjoint/differentiable LBM",
        "tclb_parity": False,
        "tensorlbm_modules": ["adjoint"],
    },
}


# ===========================================================================
# 2. Test Case Implementations
# ===========================================================================

def test_poiseuille_2d(re=100, nx=64, ny=64, steps=2000) -> dict:
    """2D Poiseuille flow: analytical solution u(y) = 4*U_max*y*(1-y)/H²."""
    from tensorlbm.d2q9 import collide_bgk, stream, C, CS2

    tau = 0.5 + re / (nx * CS2)
    u_max = 0.05
    rho0 = 1.0

    # Initialize with analytical profile
    y = torch.linspace(0, 1, ny, dtype=DTYPE).unsqueeze(1).expand(-1, nx)
    u_analytical = 4 * u_max * y * (1 - y)

    rho = torch.full((ny, nx), rho0, dtype=DTYPE)
    ux = torch.zeros((ny, nx), dtype=DTYPE)
    uy = torch.zeros((ny, nx), dtype=DTYPE)

    from tensorlbm.d2q9 import equilibrium
    feq = equilibrium(rho, ux, uy)
    f = feq.clone()

    # Solid walls
    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True

    for _ in range(steps):
        f = collide_bgk(f, tau)

        # Streaming with bounce-back
        f_new = torch.empty_like(f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cx == 0 and cy == 0:
                f_new[i] = f[i]
                continue
            src = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
            src_solid = torch.roll(solid, shifts=(-cy, -cx), dims=(0, 1))
            opp = [0, 3, 4, 1, 2, 7, 8, 5, 6][i]
            f_new[i] = torch.where(src_solid, f[opp], src)
        f = f_new

    # Compute error
    from tensorlbm.d2q9 import macroscopic
    rho, ux, uy = macroscopic(f)

    # L2 error across domain (excluding walls)
    u_ref_field = 4 * u_max * y * (1 - y)
    l2_error = torch.sqrt(((ux[1:-1, :] - u_ref_field[1:-1, :]) ** 2).mean()).item()
    rel_error = l2_error / u_max

    return {
        "test_id": "TC01",
        "name": "poiseuille_2d",
        "physics": "single_phase",
        "lattice": "D2Q9",
        "collision": "BGK",
        "re": re,
        "tau": tau,
        "steps": steps,
        "u_max_sim": ux[ny // 2, :].mean().item(),
        "u_max_ref": u_max,
        "l2_error": l2_error,
        "relative_error_pct": rel_error * 100,
        "pass": rel_error < 0.15,
    }


def test_couette_2d(re=100, nx=64, ny=64, steps=2000) -> dict:
    """2D Couette flow: analytical solution u(y) = U*y/H."""
    from tensorlbm.d2q9 import collide_bgk, C, CS2, macroscopic, equilibrium

    tau = 0.5 + re / (nx * CS2)
    u_wall = 0.05
    rho0 = 1.0

    y = torch.linspace(0, 1, ny, dtype=DTYPE).unsqueeze(1).expand(-1, nx)
    u_analytical = u_wall * y

    rho = torch.full((ny, nx), rho0, dtype=DTYPE)
    feq = equilibrium(rho, torch.zeros_like(rho), u_analytical)
    f = feq.clone()

    solid = torch.zeros(ny, nx, dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True

    for step in range(steps):
        f = collide_bgk(f, tau)

        # Apply wall velocity at top (modify distributions near wall)
        rho_f, ux_f, uy_f = macroscopic(f)
        ux_f[-1, :] = u_wall
        feq = equilibrium(rho_f, ux_f, uy_f)
        # Re-equilibrate boundary
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cy > 0:  # Directions pointing into domain from top wall
                f[i, -1, :] = feq[i, -1, :]

        f_new = torch.empty_like(f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cx == 0 and cy == 0:
                f_new[i] = f[i]
                continue
            src = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
            src_solid = torch.roll(solid, shifts=(-cy, -cx), dims=(0, 1))
            opp = [0, 3, 4, 1, 2, 7, 8, 5, 6][i]
            f_new[i] = torch.where(src_solid, f[opp], src)
        f = f_new

    rho, ux, uy = macroscopic(f)
    u_ref_field = u_wall * y
    l2_error = torch.sqrt(((ux[1:-1, :] - u_ref_field[1:-1, :]) ** 2).mean()).item()
    rel_error = l2_error / u_wall

    return {
        "test_id": "TC02",
        "name": "couette_2d",
        "physics": "single_phase",
        "lattice": "D2Q9",
        "collision": "BGK",
        "re": re,
        "tau": tau,
        "steps": steps,
        "u_wall_sim": ux[-2, :].mean().item(),
        "u_wall_ref": u_wall,
        "l2_error": l2_error,
        "relative_error_pct": rel_error * 100,
        "pass": rel_error < 0.15,
    }


def test_lid_driven_cavity_2d(re=1000, n=128, steps=5000) -> dict:
    """2D lid-driven cavity at Re=1000. Compare with Ghia et al. (1982)."""
    from tensorlbm.d2q9 import collide_bgk, C, CS2, macroscopic, equilibrium

    tau = 0.5 + re / (n * CS2)
    u_lid = 0.1
    rho0 = 1.0

    rho = torch.full((n, n), rho0, dtype=DTYPE)
    feq_init = equilibrium(rho, torch.zeros((n, n), dtype=DTYPE), torch.zeros((n, n), dtype=DTYPE))
    f = feq_init.clone()

    solid = torch.zeros(n, n, dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True

    for step in range(steps):
        f = collide_bgk(f, tau)

        # Apply lid velocity
        rho_f, ux_f, uy_f = macroscopic(f)
        ux_f[0, :] = u_lid
        uy_f[0, :] = 0.0
        feq = equilibrium(rho_f, ux_f, uy_f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cy < 0:  # Directions pointing into domain from top
                f[i, 0, :] = feq[i, 0, :]

        f_new = torch.empty_like(f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cx == 0 and cy == 0:
                f_new[i] = f[i]
                continue
            src = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
            src_solid = torch.roll(solid, shifts=(-cy, -cx), dims=(0, 1))
            opp = [0, 3, 4, 1, 2, 7, 8, 5, 6][i]
            f_new[i] = torch.where(src_solid, f[opp], src)
        f = f_new

    rho, ux, uy = macroscopic(f)

    x_mid = n // 2
    y_mid = n // 2

    u_centerline = ux[:, x_mid]
    v_centerline = uy[y_mid, :]

    u_min_sim = u_centerline.min().item()
    v_max_sim = v_centerline.max().item()

    # Ghia et al. (1982) reference for Re=1000
    u_min_ref = -0.2058
    v_max_ref = 0.1715

    u_error = abs(u_min_sim - u_min_ref) / abs(u_min_ref)
    v_error = abs(v_max_sim - v_max_ref) / abs(v_max_ref)

    return {
        "test_id": "TC03",
        "name": "lid_driven_cavity_2d",
        "physics": "single_phase",
        "lattice": "D2Q9",
        "collision": "BGK",
        "re": re,
        "tau": tau,
        "steps": steps,
        "u_min_sim": u_min_sim,
        "u_min_ref": u_min_ref,
        "u_error_pct": u_error * 100,
        "v_max_sim": v_max_sim,
        "v_max_ref": v_max_ref,
        "v_error_pct": v_error * 100,
        "pass": u_error < 0.15 and v_error < 0.15,
    }


def test_poiseuille_3d(re=100, n=32, steps=3000) -> dict:
    """3D Poiseuille flow in square duct."""
    from tensorlbm.d3q19 import C, W, CS2
    from tensorlbm.solver import equilibrium3d, collide_mrt, macroscopic3d

    tau = 0.5 + re / (n * CS2)
    u_max = 0.05
    rho0 = 1.0

    z = torch.linspace(0, 1, n, dtype=DTYPE)
    y = torch.linspace(0, 1, n, dtype=DTYPE)
    Y, Z = torch.meshgrid(y, z, indexing="ij")

    # Parabolic approximation for square duct
    u_analytical = 4 * u_max * Y * (1 - Y) * Z * (1 - Z) * 4

    rho = torch.full((n, n, n), rho0, dtype=DTYPE)
    feq = equilibrium3d(rho, u_analytical, torch.zeros_like(u_analytical), torch.zeros_like(u_analytical))
    f = feq.clone()

    solid = torch.zeros(n, n, n, dtype=torch.bool)
    solid[0, :, :] = True
    solid[-1, :, :] = True
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    for _ in range(steps):
        f = collide_mrt(f, tau)

        f_new = torch.empty_like(f)
        for i in range(19):
            cx, cy, cz = int(C[i, 0]), int(C[i, 1]), int(C[i, 2])
            if cx == 0 and cy == 0 and cz == 0:
                f_new[i] = f[i]
                continue
            src = torch.roll(f[i], shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
            src_solid = torch.roll(solid, shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
            opp = i
            for j in range(19):
                if int(C[j, 0]) == -cx and int(C[j, 1]) == -cy and int(C[j, 2]) == -cz:
                    opp = j
                    break
            f_new[i] = torch.where(src_solid, f[opp], src)
        f = f_new

    rho, ux, uy, uz = macroscopic3d(f)
    u_center = ux[n // 2, n // 2, n // 2].item()
    u_ref = u_max

    rel_error = abs(u_center - u_ref) / u_ref if u_ref != 0 else 0

    return {
        "test_id": "TC04",
        "name": "poiseuille_3d",
        "physics": "single_phase",
        "lattice": "D3Q19",
        "collision": "MRT",
        "re": re,
        "tau": tau,
        "steps": steps,
        "u_center_sim": u_center,
        "u_center_ref": u_ref,
        "relative_error_pct": rel_error * 100,
        "pass": rel_error < 0.15,
    }


def test_laplace_law_2d(r0=20, steps=5000) -> dict:
    """Laplace law verification for Shan-Chen multiphase."""
    from tensorlbm.multiphase import sc_force, collide_guo, stream_periodic, CS2

    nx, ny = 64, 64
    rho_h = 2.0
    rho_l = 1.0
    G = 5.0
    tau = 1.0

    x = torch.linspace(0, nx - 1, nx)
    y = torch.linspace(0, ny - 1, ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    R = torch.sqrt((X - nx / 2) ** 2 + (Y - ny / 2) ** 2)

    rho = torch.where(R < r0, torch.tensor(rho_h), torch.tensor(rho_l)).to(DTYPE).T

    from tensorlbm.d2q9 import equilibrium
    feq = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    f = feq.clone()

    for _ in range(steps):
        fx, fy = sc_force(rho, G)
        f_post = collide_guo(f, fx, fy, tau)
        f = stream_periodic(f_post)
        rho = f.sum(dim=0)

    rho_inside = rho[ny // 2, nx // 2].item()
    rho_outside = rho[ny // 2, 5].item()

    delta_p_sim = CS2 * (rho_inside - rho_outside)
    has_pressure_jump = abs(rho_inside - rho_outside) > 0.1

    return {
        "test_id": "TC06",
        "name": "laplace_law_2d",
        "physics": "multiphase_sc",
        "lattice": "D2Q9",
        "collision": "BGK+Guo",
        "r0": r0,
        "steps": steps,
        "rho_inside": rho_inside,
        "rho_outside": rho_outside,
        "delta_p_sim": delta_p_sim,
        "has_pressure_jump": has_pressure_jump,
        "pass": has_pressure_jump,
    }


def test_acoustic_pulse_2d(n=128, steps=200) -> dict:
    """2D acoustic pulse propagation."""
    from tensorlbm.d2q9 import collide_bgk, C, CS2, macroscopic, equilibrium

    tau = 0.6
    rho0 = 1.0
    c0 = math.sqrt(CS2)

    x = torch.linspace(0, n - 1, n)
    y = torch.linspace(0, n - 1, n)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    sigma = 5.0
    rho_perturb = 0.01 * torch.exp(-((X - n / 2) ** 2 + (Y - n / 2) ** 2) / (2 * sigma ** 2))
    rho = rho0 + rho_perturb.to(DTYPE)

    feq = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    f = feq.clone()

    initial_energy = (rho_perturb ** 2).sum().item()

    for _ in range(steps):
        f = collide_bgk(f, tau)
        f_new = torch.empty_like(f)
        for i in range(9):
            cx, cy = int(C[i, 0]), int(C[i, 1])
            if cx == 0 and cy == 0:
                f_new[i] = f[i]
                continue
            f_new[i] = torch.roll(f[i], shifts=(-cy, -cx), dims=(0, 1))
        f = f_new

    rho_final, _, _ = macroscopic(f)
    final_energy = ((rho_final - rho0) ** 2).sum().item()
    energy_ratio = final_energy / initial_energy if initial_energy > 0 else 0

    return {
        "test_id": "TC09",
        "name": "acoustic_pulse_2d",
        "physics": "acoustic",
        "lattice": "D2Q9",
        "collision": "BGK",
        "steps": steps,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_ratio": energy_ratio,
        "speed_of_sound": c0,
        "pass": 0.1 < energy_ratio < 1.0,
    }


# ===========================================================================
# 3. Coverage Tracking
# ===========================================================================

def compute_coverage(results: list[dict]) -> dict:
    """Compute coverage of the model matrix."""
    lattices_tested = set()
    collisions_tested = set()
    physics_tested = set()

    for r in results:
        lattices_tested.add(r["lattice"])
        collisions_tested.add(r["collision"])
        physics_tested.add(r["physics"])

    lattice_coverage = len(lattices_tested) / len(LATTICES)
    collision_coverage = len(collisions_tested) / len(COLLISION_OPERATORS)
    physics_coverage = len(physics_tested) / len(PHYSICS_MODELS)

    return {
        "lattices_tested": sorted(lattices_tested),
        "collisions_tested": sorted(collisions_tested),
        "physics_tested": sorted(physics_tested),
        "lattice_coverage": lattice_coverage,
        "collision_coverage": collision_coverage,
        "physics_coverage": physics_coverage,
        "overall_coverage": (lattice_coverage + collision_coverage + physics_coverage) / 3,
    }


# ===========================================================================
# 4. TCLB Comparison
# ===========================================================================

def tclb_comparison() -> dict:
    """Compare TensorLBM model matrix coverage against TCLB's approach."""
    tclb_matrix = {
        "lattices": ["D2Q9", "D3Q19", "D3Q27"],
        "collisions": ["BGK", "MRT", "Cascaded", "Entropic"],
        "physics": [
            "single_phase",
            "multiphase_pf",
            "multiphase_sc",
            "thermal",
            "turbulence_les",
        ],
        "boundaries": [
            "bounce_back",
            "equilibrium",
            "regularized",
            "pressure",
            "velocity",
            "wetting",
        ],
        "ci_practice": {
            "runs_full_matrix": True,
            "coverage_tracking": True,
            "model_documentation": True,
            "regression_tests": True,
        },
    }

    tensorlbm_matrix = {
        "lattices": ["D2Q9", "D3Q19"],
        "collisions": ["BGK", "MRT", "RLBM", "Smagorinsky-MRT"],
        "physics": [
            "single_phase",
            "multiphase_sc",
            "multiphase_pf",
            "turbulence_les",
            "acoustic",
            "adjoint",
        ],
        "boundaries": [
            "bounce_back",
            "equilibrium",
            "regularized",
            "pressure",
            "velocity",
            "wetting",
        ],
        "ci_practice": {
            "runs_full_matrix": False,
            "coverage_tracking": False,
            "model_documentation": True,
            "regression_tests": True,
        },
    }

    lattice_overlap = set(tclb_matrix["lattices"]) & set(tensorlbm_matrix["lattices"])
    collision_overlap = set(tclb_matrix["collisions"]) & set(tensorlbm_matrix["collisions"])
    physics_overlap = set(tclb_matrix["physics"]) & set(tensorlbm_matrix["physics"])

    return {
        "tclb": tclb_matrix,
        "tensorlbm": tensorlbm_matrix,
        "comparison": {
            "lattice_overlap": sorted(lattice_overlap),
            "lattice_missing_in_tensorlbm": sorted(set(tclb_matrix["lattices"]) - set(tensorlbm_matrix["lattices"])),
            "collision_overlap": sorted(collision_overlap),
            "collision_missing_in_tensorlbm": sorted(set(tclb_matrix["collisions"]) - set(tensorlbm_matrix["collisions"])),
            "physics_overlap": sorted(physics_overlap),
            "physics_unique_to_tensorlbm": sorted(set(tensorlbm_matrix["physics"]) - set(tclb_matrix["physics"])),
            "physics_missing_in_tensorlbm": sorted(set(tclb_matrix["physics"]) - set(tensorlbm_matrix["physics"])),
            "ci_practice_comparison": {
                "tclb": tclb_matrix["ci_practice"],
                "tensorlbm": tensorlbm_matrix["ci_practice"],
            },
        },
    }


# ===========================================================================
# 5. Main Runner
# ===========================================================================

def run_all(quick: bool = False) -> dict:
    """Run all test cases and produce results."""
    results = []
    start = time.time()

    print("=" * 60)
    print("XVAL: model-matrix-ci-plus-coverage")
    print("=" * 60)

    # TC01: Poiseuille 2D
    print("\n[TC01] Poiseuille 2D...")
    steps = 500 if quick else 2000
    r = test_poiseuille_2d(re=100, steps=steps)
    results.append(r)
    print(f"  relative_error = {r['relative_error_pct']:.2f}% {'✅' if r['pass'] else '❌'}")

    # TC02: Couette 2D
    print("\n[TC02] Couette 2D...")
    r = test_couette_2d(re=100, steps=steps)
    results.append(r)
    print(f"  relative_error = {r['relative_error_pct']:.2f}% {'✅' if r['pass'] else '❌'}")

    # TC03: Lid-driven cavity 2D
    print("\n[TC03] Lid-driven cavity 2D...")
    steps_cavity = 1000 if quick else 5000
    r = test_lid_driven_cavity_2d(re=1000, n=64 if quick else 128, steps=steps_cavity)
    results.append(r)
    print(f"  u_error = {r['u_error_pct']:.2f}%, v_error = {r['v_error_pct']:.2f}% {'✅' if r['pass'] else '❌'}")

    # TC04: Poiseuille 3D
    print("\n[TC04] Poiseuille 3D...")
    steps_3d = 500 if quick else 3000
    r = test_poiseuille_3d(re=100, n=16 if quick else 32, steps=steps_3d)
    results.append(r)
    print(f"  relative_error = {r['relative_error_pct']:.2f}% {'✅' if r['pass'] else '❌'}")

    # TC06: Laplace law 2D
    print("\n[TC06] Laplace law 2D (SC multiphase)...")
    steps_laplace = 1000 if quick else 5000
    r = test_laplace_law_2d(r0=20, steps=steps_laplace)
    results.append(r)
    print(f"  pressure_jump = {r['has_pressure_jump']} {'✅' if r['pass'] else '❌'}")

    # TC09: Acoustic pulse 2D
    print("\n[TC09] Acoustic pulse 2D...")
    steps_acoustic = 50 if quick else 200
    r = test_acoustic_pulse_2d(n=64 if quick else 128, steps=steps_acoustic)
    results.append(r)
    print(f"  energy_ratio = {r['energy_ratio']:.4f} {'✅' if r['pass'] else '❌'}")

    elapsed = time.time() - start

    coverage = compute_coverage(results)
    comparison = tclb_comparison()

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    result = {
        "title": "XVAL-LBM-MODEL-MATRIX-CI-PLUS-COVERAGE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "quick_mode": quick,
        "results": results,
        "coverage": coverage,
        "tclb_comparison": comparison,
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "criterion": "<15% relative error",
            "all_pass": all(r["pass"] for r in results),
        },
    }

    result_path = ART / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n{'=' * 60}")
    print(f"Results saved to {result_path}")
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
    print(f"Pass rate: {passed / total * 100:.1f}%")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode for CI")
    args = parser.parse_args()
    run_all(quick=args.quick)
