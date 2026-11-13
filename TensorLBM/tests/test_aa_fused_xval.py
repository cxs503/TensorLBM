"""Cross-validation: AA-pattern fused stream-collide vs separate stream+collide.

Compares the AA-pattern fused approach (lanl/MF-LBM style) against the
traditional separate stream+collide approach in TensorLBM.

Decision criteria:
- Numerical equivalence: max absolute difference < 1e-12 (should be bit-identical for BGK)
- Velocity profile error vs Ghia reference: < 15% (owner precision standard)
- Performance: MLUPS comparison

Test case: 2D lid-driven cavity, Re=100, n=65
"""

from __future__ import annotations

import json
import time
import torch
import numpy as np
from pathlib import Path

from tensorlbm.aa_fused_stream_collide import (
    aa_fused_step,
    separate_step,
    _macroscopic,
)
from tensorlbm.verification.lid_driven_cavity.solver import CavityConfig
from tensorlbm.verification.lid_driven_cavity.reference_data import (
    get_ghia_reference,
    interpolate_ghia,
)


def run_cavity_aa(n=65, re=100, u_lid=0.1, max_steps=10000, tau=None):
    """Run lid-driven cavity with AA-pattern fused stream-collide."""
    cfg = CavityConfig(n=n, re=re, u_lid=u_lid, max_steps=max_steps)
    if tau is not None:
        cfg.tau = tau

    device = torch.device("cpu")
    dtype = torch.float64

    # Initialize at equilibrium (fluid at rest)
    rho = torch.ones((n, n), dtype=dtype, device=device)
    ux = torch.zeros((n, n), dtype=dtype, device=device)
    uy = torch.zeros((n, n), dtype=dtype, device=device)

    # D2Q9 equilibrium
    C = torch.tensor([
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1],
    ], dtype=torch.int64, device=device)
    W = torch.tensor([
        4./9., 1./9., 1./9., 1./9., 1./9.,
        1./36., 1./36., 1./36., 1./36.,
    ], dtype=dtype, device=device)

    uc = C[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + C[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
    u2 = ux**2 + uy**2
    f = W.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )

    # Solid mask
    solid = torch.zeros((n, n), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True

    # Run
    start = time.perf_counter()
    for _ in range(max_steps):
        f = aa_fused_step(f, cfg.tau, solid, u_lid=u_lid, lid_row=-1)
    elapsed = time.perf_counter() - start

    # Extract profiles
    rho_f, ux_f, uy_f = _macroscopic(f)
    mid = n // 2
    u_center = (ux_f[:, mid] / u_lid).numpy()
    v_center = (uy_f[mid, :] / u_lid).numpy()
    y_norm = np.linspace(0, 1, n)
    x_norm = np.linspace(0, 1, n)

    mlups = (n * n * max_steps) / (elapsed * 1e6)

    return {
        "y": y_norm, "u": u_center,
        "x": x_norm, "v": v_center,
        "f": f, "elapsed": elapsed, "mlups": mlups,
        "mass": f.sum().item(),
    }


def run_cavity_separate(n=65, re=100, u_lid=0.1, max_steps=10000, tau=None):
    """Run lid-driven cavity with separate stream+collide."""
    cfg = CavityConfig(n=n, re=re, u_lid=u_lid, max_steps=max_steps)
    if tau is not None:
        cfg.tau = tau

    device = torch.device("cpu")
    dtype = torch.float64

    # Initialize at equilibrium (fluid at rest)
    rho = torch.ones((n, n), dtype=dtype, device=device)
    ux = torch.zeros((n, n), dtype=dtype, device=device)
    uy = torch.zeros((n, n), dtype=dtype, device=device)

    C = torch.tensor([
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1],
    ], dtype=torch.int64, device=device)
    W = torch.tensor([
        4./9., 1./9., 1./9., 1./9., 1./9.,
        1./36., 1./36., 1./36., 1./36.,
    ], dtype=dtype, device=device)

    uc = C[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + C[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
    u2 = ux**2 + uy**2
    f = W.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )

    solid = torch.zeros((n, n), dtype=torch.bool, device=device)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True

    start = time.perf_counter()
    for _ in range(max_steps):
        f = separate_step(f, cfg.tau, solid, u_lid=u_lid, lid_row=-1)
    elapsed = time.perf_counter() - start

    rho_f, ux_f, uy_f = _macroscopic(f)
    mid = n // 2
    u_center = (ux_f[:, mid] / u_lid).numpy()
    v_center = (uy_f[mid, :] / u_lid).numpy()
    y_norm = np.linspace(0, 1, n)
    x_norm = np.linspace(0, 1, n)

    mlups = (n * n * max_steps) / (elapsed * 1e6)

    return {
        "y": y_norm, "u": u_center,
        "x": x_norm, "v": v_center,
        "f": f, "elapsed": elapsed, "mlups": mlups,
        "mass": f.sum().item(),
    }


def compute_errors(y_sim, u_sim, x_sim, v_sim, ref):
    """Compute L2 and max errors against Ghia reference."""
    u_interp = np.interp(ref.y, y_sim, u_sim)
    v_interp = np.interp(ref.x, x_sim, v_sim)

    l2_u = np.sqrt(np.mean((u_interp - ref.u)**2))
    l2_v = np.sqrt(np.mean((v_interp - ref.v)**2))
    max_u = np.max(np.abs(u_interp - ref.u))
    max_v = np.max(np.abs(v_interp - ref.v))

    return {"l2_u": l2_u, "l2_v": l2_v, "max_u": max_u, "max_v": max_v}


def main():
    """Run cross-validation experiment."""
    print("=" * 70)
    print("XVAL: AA-pattern fused stream-collide vs separate stream+collide")
    print("=" * 70)

    # Parameters
    n = 65
    re = 100
    u_lid = 0.1
    max_steps = 10000

    print(f"\nCase: Lid-driven cavity, Re={re}, n={n}, u_lid={u_lid}")
    print(f"Steps: {max_steps}")

    # Run both approaches
    print("\n--- Running AA-pattern fused stream-collide ---")
    result_aa = run_cavity_aa(n=n, re=re, u_lid=u_lid, max_steps=max_steps)
    print(f"  Time: {result_aa['elapsed']:.3f}s, MLUPS: {result_aa['mlups']:.1f}")

    print("\n--- Running separate stream+collide ---")
    result_sep = run_cavity_separate(n=n, re=re, u_lid=u_lid, max_steps=max_steps)
    print(f"  Time: {result_sep['elapsed']:.3f}s, MLUPS: {result_sep['mlups']:.1f}")

    # Numerical equivalence check
    print("\n--- Numerical Equivalence Check ---")
    f_diff = torch.abs(result_aa["f"] - result_sep["f"])
    max_diff = f_diff.max().item()
    mean_diff = f_diff.mean().item()
    print(f"  Max |f_aa - f_sep|: {max_diff:.2e}")
    print(f"  Mean |f_aa - f_sep|: {mean_diff:.2e}")

    # Mass conservation check
    mass_aa = result_aa["mass"]
    mass_sep = result_sep["mass"]
    mass_diff = abs(mass_aa - mass_sep) / mass_aa
    print(f"  Mass AA: {mass_aa:.6f}, Mass Sep: {mass_sep:.6f}, rel diff: {mass_diff:.2e}")

    # Compare against Ghia reference
    print("\n--- Ghia Reference Comparison (Re=100) ---")
    ref = get_ghia_reference(100)

    errors_aa = compute_errors(result_aa["y"], result_aa["u"],
                               result_aa["x"], result_aa["v"], ref)
    errors_sep = compute_errors(result_sep["y"], result_sep["u"],
                                result_sep["x"], result_sep["v"], ref)

    print(f"  AA-pattern:  L2_u={errors_aa['l2_u']:.6f}, L2_v={errors_aa['l2_v']:.6f}, "
          f"max_u={errors_aa['max_u']:.6f}, max_v={errors_aa['max_v']:.6f}")
    print(f"  Separate:    L2_u={errors_sep['l2_u']:.6f}, L2_v={errors_sep['l2_v']:.6f}, "
          f"max_u={errors_sep['max_u']:.6f}, max_v={errors_sep['max_v']:.6f}")

    # Performance comparison
    print("\n--- Performance Comparison ---")
    speedup = result_sep["elapsed"] / result_aa["elapsed"]
    print(f"  AA-pattern: {result_aa['mlups']:.1f} MLUPS")
    print(f"  Separate:   {result_sep['mlups']:.1f} MLUPS")
    print(f"  Speedup (AA/Sep): {speedup:.3f}x")

    # Verdict
    print("\n--- Verdict ---")
    equiv_pass = max_diff < 1e-10
    print(f"  Numerical equivalence (max diff < 1e-10): {'PASS' if equiv_pass else 'FAIL'} ({max_diff:.2e})")

    aa_pass = errors_aa['l2_u'] < 0.15 and errors_aa['l2_v'] < 0.15
    print(f"  AA-pattern vs Ghia < 15%: {'PASS' if aa_pass else 'FAIL'} "
          f"(L2_u={errors_aa['l2_u']:.4f}, L2_v={errors_aa['l2_v']:.4f})")

    sep_pass = errors_sep['l2_u'] < 0.15 and errors_sep['l2_v'] < 0.15
    print(f"  Separate vs Ghia < 15%: {'PASS' if sep_pass else 'FAIL'} "
          f"(L2_u={errors_sep['l2_u']:.4f}, L2_v={errors_sep['l2_v']:.4f})")

    # Build result JSON
    result = {
        "experiment": "aa_pattern_fused_stream_collide_xval",
        "case": {"n": n, "re": re, "u_lid": u_lid, "max_steps": max_steps},
        "numerical_equivalence": {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "mass_rel_diff": mass_diff,
            "pass": equiv_pass,
        },
        "aa_pattern": {
            "elapsed_s": result_aa["elapsed"],
            "mlups": result_aa["mlups"],
            "errors_vs_ghia": errors_aa,
            "pass_vs_ghia_15pct": aa_pass,
        },
        "separate": {
            "elapsed_s": result_sep["elapsed"],
            "mlups": result_sep["mlups"],
            "errors_vs_ghia": errors_sep,
            "pass_vs_ghia_15pct": sep_pass,
        },
        "performance": {
            "speedup_aa_over_sep": speedup,
        },
        "verdict": {
            "numerical_equivalence": "PASS" if equiv_pass else "FAIL",
            "aa_pattern_accuracy": "PASS" if aa_pass else "FAIL",
            "separate_accuracy": "PASS" if sep_pass else "FAIL",
        },
    }

    # Save results
    out_dir = Path(__file__).parent.parent.parent / "artifacts" / "xval_aa_pattern"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xval_aa_pattern_fused_stream_collide.json"
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"\nResults saved to: {out_path}")

    return result


if __name__ == "__main__":
    main()
