#!/usr/bin/env python3
"""XVAL-LBM-VOXEL-IMGPROC-POROSITY-REPORT runner.

Cross-validates the voxelized image-processing porosity pipeline in TensorLBM
against analytical benchmarks and self-consistency checks.

Arms
----
A1: Random 2-D mask porosity — requested vs actual (random_porosity_mask_2d)
A2: Random 3-D mask porosity — requested vs actual (random_porosity_mask_3d)
A3: Analytical geometry porosity — sphere, cylinder, ellipsoid voxelized masks
A4: Convergence study — porosity error vs resolution for analytical geometries
A5: Sigma sensitivity — effect of Gaussian correlation length on porosity accuracy

Acceptance criterion (owner): relative error < 15% on all arms.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tensorlbm.preprocess_geo import random_porosity_mask_2d, random_porosity_mask_3d

# torch.set_default_dtype(torch.float64)  # not needed for porosity

RESULTS = {
    "meta": {"anchor": "XVAL-LBM-VOXEL-IMGPROC-POROSITY-REPORT", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
    "A1_random_2d": {},
    "A2_random_3d": {},
    "A3_analytical": {},
    "A4_convergence": {},
    "A5_sigma_sensitivity": {},
    "verdict": {},
}


# =========================================================================== A1
def arm_A1_random_2d() -> dict:
    """Random 2-D mask: does actual porosity match requested porosity?

    For each target φ ∈ {0.2, 0.4, 0.6, 0.8}, generate a mask and measure
    the actual porosity.  The relative error must be < 15%.
    """
    out = {"cases": []}
    ny, nx = 128, 128
    device = torch.device("cpu")

    for phi_target in [0.2, 0.4, 0.6, 0.8]:
        for seed in [0, 1, 2]:
            mask = random_porosity_mask_2d(ny, nx, porosity=phi_target, device=device, seed=seed, sigma=0.0)
            # mask is True where solid; porosity = fraction of fluid (non-solid)
            phi_actual = float((~mask).sum()) / float(mask.numel())
            rel_err = abs(phi_actual - phi_target) / phi_target
            out["cases"].append({
                "phi_target": phi_target,
                "seed": seed,
                "phi_actual": round(phi_actual, 6),
                "rel_error": round(rel_err, 6),
                "pass_15pct": rel_err < 0.15,
            })

    # Also test with sigma > 0 (correlated masks)
    for phi_target in [0.4, 0.6]:
        for sigma in [1.0, 3.0]:
            for seed in [0, 1]:
                mask = random_porosity_mask_2d(ny, nx, porosity=phi_target, device=device, seed=seed, sigma=sigma)
                phi_actual = float((~mask).sum()) / float(mask.numel())
                rel_err = abs(phi_actual - phi_target) / phi_target
                out["cases"].append({
                    "phi_target": phi_target,
                    "seed": seed,
                    "sigma": sigma,
                    "phi_actual": round(phi_actual, 6),
                    "rel_error": round(rel_err, 6),
                    "pass_15pct": rel_err < 0.15,
                })

    all_pass = all(c["pass_15pct"] for c in out["cases"])
    out["all_pass_15pct"] = all_pass
    out["max_rel_error"] = max(c["rel_error"] for c in out["cases"])
    out["mean_rel_error"] = sum(c["rel_error"] for c in out["cases"]) / len(out["cases"])
    return out


# =========================================================================== A2
def arm_A2_random_3d() -> dict:
    """Random 3-D mask: does actual porosity match requested porosity?

    Same as A1 but in 3-D.  Uses smaller grids due to memory.
    """
    out = {"cases": []}
    nz, ny, nx = 64, 64, 64
    device = torch.device("cpu")

    for phi_target in [0.2, 0.4, 0.6, 0.8]:
        for seed in [0, 1]:
            mask = random_porosity_mask_3d(nz, ny, nx, porosity=phi_target, device=device, seed=seed, sigma=0.0)
            phi_actual = float((~mask).sum()) / float(mask.numel())
            rel_err = abs(phi_actual - phi_target) / phi_target
            out["cases"].append({
                "phi_target": phi_target,
                "seed": seed,
                "phi_actual": round(phi_actual, 6),
                "rel_error": round(rel_err, 6),
                "pass_15pct": rel_err < 0.15,
            })

    # With sigma
    for phi_target in [0.4, 0.6]:
        for sigma in [1.0, 2.0]:
            mask = random_porosity_mask_3d(nz, ny, nx, porosity=phi_target, device=device, seed=0, sigma=sigma)
            phi_actual = float((~mask).sum()) / float(mask.numel())
            rel_err = abs(phi_actual - phi_target) / phi_target
            out["cases"].append({
                "phi_target": phi_target,
                "sigma": sigma,
                "phi_actual": round(phi_actual, 6),
                "rel_error": round(rel_err, 6),
                "pass_15pct": rel_err < 0.15,
            })

    all_pass = all(c["pass_15pct"] for c in out["cases"])
    out["all_pass_15pct"] = all_pass
    out["max_rel_error"] = max(c["rel_error"] for c in out["cases"])
    out["mean_rel_error"] = sum(c["rel_error"] for c in out["cases"]) / len(out["cases"])
    return out


# =========================================================================== A3
def _sphere_mask_3d(n: int, radius_frac: float = 0.4) -> tuple[np.ndarray, float]:
    """Create a voxelized sphere mask and return (mask, analytical_porosity).

    The sphere is centred in the domain.  radius_frac is the radius as a
    fraction of n/2.
    """
    radius = radius_frac * n / 2.0
    z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
    r2 = (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2
    solid = r2 < radius**2
    # Analytical porosity = 1 - V_sphere / V_domain
    v_sphere = (4.0 / 3.0) * math.pi * radius**3
    v_domain = float(n)**3
    phi_analytical = 1.0 - v_sphere / v_domain
    return solid, phi_analytical


def _cylinder_mask_3d(n: int, radius_frac: float = 0.3, height_frac: float = 0.6) -> tuple[np.ndarray, float]:
    """Create a voxelized cylinder mask (axis along z) and return (mask, analytical_porosity)."""
    radius = radius_frac * n / 2.0
    height = height_frac * n
    z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
    r2 = (x + 0.5)**2 + (y + 0.5)**2
    z_phys = (z + 0.5)
    solid = (r2 < radius**2) & (np.abs(z_phys) < height / 2.0)
    v_cyl = math.pi * radius**2 * height
    v_domain = float(n)**3
    phi_analytical = 1.0 - v_cyl / v_domain
    return solid, phi_analytical


def _ellipsoid_mask_3d(n: int, a_frac: float = 0.4, b_frac: float = 0.25, c_frac: float = 0.15) -> tuple[np.ndarray, float]:
    """Create a voxelized ellipsoid mask and return (mask, analytical_porosity)."""
    a = a_frac * n / 2.0
    b = b_frac * n / 2.0
    c = c_frac * n / 2.0
    z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
    r2 = ((x + 0.5) / a)**2 + ((y + 0.5) / b)**2 + ((z + 0.5) / c)**2
    solid = r2 < 1.0
    v_ell = (4.0 / 3.0) * math.pi * a * b * c
    v_domain = float(n)**3
    phi_analytical = 1.0 - v_ell / v_domain
    return solid, phi_analytical


def arm_A3_analytical() -> dict:
    """Analytical geometry porosity: sphere, cylinder, ellipsoid.

    Voxelize known geometries and compare porosity against analytical values.
    """
    out = {"cases": []}
    n = 80  # resolution

    for geom_name, mask_fn in [
        ("sphere", lambda: _sphere_mask_3d(n)),
        ("cylinder", lambda: _cylinder_mask_3d(n)),
        ("ellipsoid", lambda: _ellipsoid_mask_3d(n)),
    ]:
        solid, phi_analytical = mask_fn()
        phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
        rel_err = abs(phi_computed - phi_analytical) / phi_analytical if phi_analytical > 0 else float("inf")
        out["cases"].append({
            "geometry": geom_name,
            "n": n,
            "phi_analytical": round(phi_analytical, 6),
            "phi_computed": round(phi_computed, 6),
            "rel_error": round(rel_err, 6),
            "pass_15pct": rel_err < 0.15,
        })

    # Also test sphere at different radii
    for radius_frac in [0.2, 0.3, 0.45]:
        solid, phi_analytical = _sphere_mask_3d(n, radius_frac=radius_frac)
        phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
        rel_err = abs(phi_computed - phi_analytical) / phi_analytical if phi_analytical > 0 else float("inf")
        out["cases"].append({
            "geometry": f"sphere_r{radius_frac}",
            "n": n,
            "phi_analytical": round(phi_analytical, 6),
            "phi_computed": round(phi_computed, 6),
            "rel_error": round(rel_err, 6),
            "pass_15pct": rel_err < 0.15,
        })

    all_pass = all(c["pass_15pct"] for c in out["cases"])
    out["all_pass_15pct"] = all_pass
    out["max_rel_error"] = max(c["rel_error"] for c in out["cases"])
    return out


# =========================================================================== A4
def arm_A4_convergence() -> dict:
    """Convergence study: porosity error vs resolution for analytical geometries.

    Test sphere porosity at n = 20, 40, 80, 160 and fit convergence order.
    """
    out = {"cases": [], "convergence": {}}
    n_range = [20, 40, 80, 160]
    radius_frac = 0.4

    for n in n_range:
        solid, phi_analytical = _sphere_mask_3d(n, radius_frac=radius_frac)
        phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
        rel_err = abs(phi_computed - phi_analytical) / phi_analytical
        out["cases"].append({
            "n": n,
            "h": 1.0 / n,
            "phi_analytical": round(phi_analytical, 6),
            "phi_computed": round(phi_computed, 6),
            "rel_error": round(rel_err, 6),
        })

    # Fit convergence order: error ~ h^p => log(error) = p * log(h) + const
    if len(out["cases"]) >= 3:
        log_h = np.log([c["h"] for c in out["cases"]])
        log_err = np.log([c["rel_error"] for c in out["cases"]])
        coeffs = np.polyfit(log_h, log_err, 1)
        p = float(coeffs[0])
        # R²
        pred = np.polyval(coeffs, log_h)
        ss_res = np.sum((log_err - pred)**2)
        ss_tot = np.sum((log_err - np.mean(log_err))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out["convergence"] = {
            "order_p": round(p, 4),
            "r_squared": round(r2, 4),
            "pass_p_ge_1": p >= 1.0,
        }

    # Also test cylinder convergence
    cyl_cases = []
    for n in n_range:
        solid, phi_analytical = _cylinder_mask_3d(n)
        phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
        rel_err = abs(phi_computed - phi_analytical) / phi_analytical
        cyl_cases.append({
            "n": n,
            "h": 1.0 / n,
            "phi_analytical": round(phi_analytical, 6),
            "phi_computed": round(phi_computed, 6),
            "rel_error": round(rel_err, 6),
        })

    if len(cyl_cases) >= 3:
        log_h = np.log([c["h"] for c in cyl_cases])
        log_err = np.log([c["rel_error"] for c in cyl_cases])
        coeffs = np.polyfit(log_h, log_err, 1)
        p = float(coeffs[0])
        pred = np.polyval(coeffs, log_h)
        ss_res = np.sum((log_err - pred)**2)
        ss_tot = np.sum((log_err - np.mean(log_err))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out["convergence_cylinder"] = {
            "order_p": round(p, 4),
            "r_squared": round(r2, 4),
            "pass_p_ge_1": p >= 1.0,
        }
    out["cylinder_cases"] = cyl_cases

    return out


# =========================================================================== A5
def arm_A5_sigma_sensitivity() -> dict:
    """Sigma sensitivity: effect of Gaussian correlation length on porosity accuracy.

    For a fixed target porosity, vary sigma and check that actual porosity
    remains within 15% of target.
    """
    out = {"cases": []}
    ny, nx = 128, 128
    device = torch.device("cpu")
    phi_target = 0.5

    for sigma in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        for seed in [0, 1, 2]:
            mask = random_porosity_mask_2d(ny, nx, porosity=phi_target, device=device, seed=seed, sigma=sigma)
            phi_actual = float((~mask).sum()) / float(mask.numel())
            rel_err = abs(phi_actual - phi_target) / phi_target
            out["cases"].append({
                "sigma": sigma,
                "seed": seed,
                "phi_actual": round(phi_actual, 6),
                "rel_error": round(rel_err, 6),
                "pass_15pct": rel_err < 0.15,
            })

    all_pass = all(c["pass_15pct"] for c in out["cases"])
    out["all_pass_15pct"] = all_pass
    out["max_rel_error"] = max(c["rel_error"] for c in out["cases"])
    return out


# =========================================================================== main
def main():
    t0 = time.time()

    print("=" * 70)
    print("XVAL-LBM-VOXEL-IMGPROC-POROSITY-REPORT")
    print("=" * 70)

    print("\n--- A1: Random 2-D mask porosity ---")
    RESULTS["A1_random_2d"] = arm_A1_random_2d()
    print(f"  All pass <15%: {RESULTS['A1_random_2d']['all_pass_15pct']}")
    print(f"  Max rel error: {RESULTS['A1_random_2d']['max_rel_error']:.6f}")
    print(f"  Mean rel error: {RESULTS['A1_random_2d']['mean_rel_error']:.6f}")

    print("\n--- A2: Random 3-D mask porosity ---")
    RESULTS["A2_random_3d"] = arm_A2_random_3d()
    print(f"  All pass <15%: {RESULTS['A2_random_3d']['all_pass_15pct']}")
    print(f"  Max rel error: {RESULTS['A2_random_3d']['max_rel_error']:.6f}")
    print(f"  Mean rel error: {RESULTS['A2_random_3d']['mean_rel_error']:.6f}")

    print("\n--- A3: Analytical geometry porosity ---")
    RESULTS["A3_analytical"] = arm_A3_analytical()
    print(f"  All pass <15%: {RESULTS['A3_analytical']['all_pass_15pct']}")
    for c in RESULTS["A3_analytical"]["cases"]:
        print(f"    {c['geometry']:20s}  φ_ref={c['phi_analytical']:.4f}  φ_got={c['phi_computed']:.4f}  err={c['rel_error']:.4f}  {'✓' if c['pass_15pct'] else '✗'}")

    print("\n--- A4: Convergence study ---")
    RESULTS["A4_convergence"] = arm_A4_convergence()
    for c in RESULTS["A4_convergence"]["cases"]:
        print(f"    n={c['n']:4d}  φ_ref={c['phi_analytical']:.4f}  φ_got={c['phi_computed']:.4f}  err={c['rel_error']:.6f}")
    conv = RESULTS["A4_convergence"].get("convergence", {})
    print(f"  Sphere convergence order p={conv.get('order_p', 'N/A')}, R²={conv.get('r_squared', 'N/A')}, pass p≥1: {conv.get('pass_p_ge_1', 'N/A')}")
    conv_cyl = RESULTS["A4_convergence"].get("convergence_cylinder", {})
    print(f"  Cylinder convergence order p={conv_cyl.get('order_p', 'N/A')}, R²={conv_cyl.get('r_squared', 'N/A')}, pass p≥1: {conv_cyl.get('pass_p_ge_1', 'N/A')}")

    print("\n--- A5: Sigma sensitivity ---")
    RESULTS["A5_sigma_sensitivity"] = arm_A5_sigma_sensitivity()
    print(f"  All pass <15%: {RESULTS['A5_sigma_sensitivity']['all_pass_15pct']}")
    print(f"  Max rel error: {RESULTS['A5_sigma_sensitivity']['max_rel_error']:.6f}")

    # Verdict
    all_arms_pass = (
        RESULTS["A1_random_2d"]["all_pass_15pct"]
        and RESULTS["A2_random_3d"]["all_pass_15pct"]
        and RESULTS["A3_analytical"]["all_pass_15pct"]
        and RESULTS["A4_convergence"].get("convergence", {}).get("pass_p_ge_1", False)
        and RESULTS["A5_sigma_sensitivity"]["all_pass_15pct"]
    )
    RESULTS["verdict"] = {
        "all_arms_pass": all_arms_pass,
        "threshold": 0.15,
        "A1_pass": RESULTS["A1_random_2d"]["all_pass_15pct"],
        "A2_pass": RESULTS["A2_random_3d"]["all_pass_15pct"],
        "A3_pass": RESULTS["A3_analytical"]["all_pass_15pct"],
        "A4_pass": RESULTS["A4_convergence"].get("convergence", {}).get("pass_p_ge_1", False),
        "A5_pass": RESULTS["A5_sigma_sensitivity"]["all_pass_15pct"],
        "wall_seconds": round(time.time() - t0, 2),
    }

    print("\n" + "=" * 70)
    print(f"VERDICT: {'PASS' if all_arms_pass else 'PARTIAL/FAIL'}")
    for arm in ["A1_pass", "A2_pass", "A3_pass", "A4_pass", "A5_pass"]:
        print(f"  {arm}: {RESULTS['verdict'][arm]}")
    print(f"  Wall time: {RESULTS['verdict']['wall_seconds']:.1f}s")
    print("=" * 70)

    # Save results
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return RESULTS


if __name__ == "__main__":
    main()
