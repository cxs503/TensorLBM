"""Cross-validation: AA-pattern fused stream-collide vs separate stream+collide.

Compares the AA-pattern fused approach (lanl/MF-LBM style) against the
traditional separate stream+collide approach in TensorLBM.

Key finding: The AA-pattern (pull scheme) and separate (push scheme) are
mathematically equivalent for periodic BCs but differ when solid boundaries
are present due to operation ordering. Both converge to the same physical
solution (verified against Ghia reference).

Decision criteria:
- Numerical equivalence (periodic BCs): max diff < 1e-14 (machine epsilon)
- Velocity profile error vs Ghia reference: < 15% (owner precision standard)
- Both approaches should produce physically correct results
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
from tensorlbm.verification.lid_driven_cavity.reference_data import (
    get_ghia_reference,
)


def run_cavity_quick(n=33, re=100, u_lid=0.1, max_steps=2000, tau=None):
    """Run lid-driven cavity with both approaches on small grid."""
    if tau is None:
        L = n - 1
        nu = u_lid * L / re
        tau = nu / (1.0/3.0) + 0.5

    device = torch.device("cpu")
    dtype = torch.float64

    # Initialize at equilibrium
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

    # Run AA-pattern
    f_aa = f.clone()
    start = time.perf_counter()
    for _ in range(max_steps):
        f_aa = aa_fused_step(f_aa, tau, solid, u_lid=u_lid, lid_row=-1)
    elapsed_aa = time.perf_counter() - start

    # Run separate
    f_sep = f.clone()
    start = time.perf_counter()
    for _ in range(max_steps):
        f_sep = separate_step(f_sep, tau, solid, u_lid=u_lid, lid_row=-1)
    elapsed_sep = time.perf_counter() - start

    # Extract profiles
    rho_aa, ux_aa, uy_aa = _macroscopic(f_aa)
    rho_sep, ux_sep, uy_sep = _macroscopic(f_sep)

    mid = n // 2
    u_aa = (ux_aa[:, mid] / u_lid).numpy()
    v_aa = (uy_aa[mid, :] / u_lid).numpy()
    u_sep = (ux_sep[:, mid] / u_lid).numpy()
    v_sep = (uy_sep[mid, :] / u_lid).numpy()

    y_norm = np.linspace(0, 1, n)
    x_norm = np.linspace(0, 1, n)

    mlups_aa = (n * n * max_steps) / (elapsed_aa * 1e6)
    mlups_sep = (n * n * max_steps) / (elapsed_sep * 1e6)

    return {
        "y": y_norm, "u_aa": u_aa, "v_aa": v_aa,
        "x": x_norm, "u_sep": u_sep, "v_sep": v_sep,
        "f_aa": f_aa, "f_sep": f_sep,
        "elapsed_aa": elapsed_aa, "elapsed_sep": elapsed_sep,
        "mlups_aa": mlups_aa, "mlups_sep": mlups_sep,
        "mass_aa": f_aa.sum().item(), "mass_sep": f_sep.sum().item(),
        "tau": tau,
    }


def compute_errors(y_sim, u_sim, x_sim, v_sim, ref):
    """Compute L2 and max errors against Ghia reference."""
    u_interp = np.interp(ref.y, y_sim, u_sim)
    v_interp = np.interp(ref.x, x_sim, v_sim)

    l2_u = float(np.sqrt(np.mean((u_interp - ref.u)**2)))
    l2_v = float(np.sqrt(np.mean((v_interp - ref.v)**2)))
    max_u = float(np.max(np.abs(u_interp - ref.u)))
    max_v = float(np.max(np.abs(v_interp - ref.v)))

    return {"l2_u": l2_u, "l2_v": l2_v, "max_u": max_u, "max_v": max_v}


def test_periodic_equivalence():
    """Test that AA and separate are equivalent for periodic BCs."""
    n = 33
    f = torch.ones((9, n, n), dtype=torch.float64) * 0.1
    f[0] = 0.5
    tau = 1.0

    from tensorlbm.aa_fused_stream_collide import aa_fused_stream_collide, separate_stream_collide
    f_aa = aa_fused_stream_collide(f, tau)
    f_sep = separate_stream_collide(f, tau)
    diff = float(torch.abs(f_aa - f_sep).max().item())

    return {"max_diff": diff, "pass": bool(diff < 1e-14)}


def main():
    print("=" * 70)
    print("XVAL: AA-pattern fused vs separate stream-collide")
    print("=" * 70)

    # Test 1: Periodic BC equivalence
    print("\n--- Test 1: Periodic BC Equivalence ---")
    periodic_result = test_periodic_equivalence()
    print(f"  Max diff (periodic): {periodic_result['max_diff']:.2e}")
    print(f"  Pass (< 1e-14): {'PASS' if periodic_result['pass'] else 'FAIL'}")

    # Test 2: Lid-driven cavity with BCs
    n = 33
    re = 100
    u_lid = 0.1
    max_steps = 2000

    print(f"\n--- Test 2: Lid-driven cavity (Re={re}, n={n}, steps={max_steps}) ---")
    result = run_cavity_quick(n=n, re=re, u_lid=u_lid, max_steps=max_steps)

    # Numerical equivalence (with BCs, expect differences due to operation ordering)
    print("\n--- Numerical Comparison (with BCs) ---")
    f_diff = torch.abs(result["f_aa"] - result["f_sep"])
    max_diff = float(f_diff.max().item())
    mean_diff = float(f_diff.mean().item())
    print(f"  Max |f_aa - f_sep|: {max_diff:.2e}")
    print(f"  Mean |f_aa - f_sep|: {mean_diff:.2e}")
    print(f"  Note: Differences expected due to BC operation ordering")

    mass_diff = float(abs(result["mass_aa"] - result["mass_sep"]) / result["mass_aa"])
    print(f"  Mass rel diff: {mass_diff:.2e}")

    # Compare against Ghia reference
    print("\n--- Ghia Reference Comparison (Re=100) ---")
    ref = get_ghia_reference(100)

    errors_aa = compute_errors(result["y"], result["u_aa"],
                               result["x"], result["v_aa"], ref)
    errors_sep = compute_errors(result["y"], result["u_sep"],
                                result["x"], result["v_sep"], ref)

    print(f"  AA-pattern:  L2_u={errors_aa['l2_u']:.6f}, L2_v={errors_aa['l2_v']:.6f}")
    print(f"  Separate:    L2_u={errors_sep['l2_u']:.6f}, L2_v={errors_sep['l2_v']:.6f}")

    # Performance
    print("\n--- Performance ---")
    print(f"  AA-pattern: {result['mlups_aa']:.1f} MLUPS ({result['elapsed_aa']:.3f}s)")
    print(f"  Separate:   {result['mlups_sep']:.1f} MLUPS ({result['elapsed_sep']:.3f}s)")
    speedup = result["elapsed_sep"] / result["elapsed_aa"]
    print(f"  Speedup (AA/Sep): {speedup:.3f}x")

    # Verdict
    print("\n--- Verdict ---")
    periodic_pass = periodic_result['pass']
    print(f"  Periodic BC equivalence: {'PASS' if periodic_pass else 'FAIL'}")

    aa_pass = bool(errors_aa['l2_u'] < 0.15 and errors_aa['l2_v'] < 0.15)
    print(f"  AA-pattern vs Ghia < 15%: {'PASS' if aa_pass else 'FAIL'} "
          f"(L2_u={errors_aa['l2_u']:.4f}, L2_v={errors_aa['l2_v']:.4f})")

    sep_pass = bool(errors_sep['l2_u'] < 0.15 and errors_sep['l2_v'] < 0.15)
    print(f"  Separate vs Ghia < 15%: {'PASS' if sep_pass else 'FAIL'} "
          f"(L2_u={errors_sep['l2_u']:.4f}, L2_v={errors_sep['l2_v']:.4f})")

    # Save results
    out_dir = Path(__file__).parent.parent.parent / "artifacts" / "xval_aa_pattern"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xval_aa_pattern_fused_stream_collide.json"

    result_json = {
        "experiment": "aa_pattern_fused_stream_collide_xval",
        "case": {"n": n, "re": re, "u_lid": u_lid, "max_steps": max_steps, "tau": float(result["tau"])},
        "periodic_equivalence": periodic_result,
        "numerical_comparison_with_bcs": {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "mass_rel_diff": mass_diff,
            "note": "Differences expected due to BC operation ordering",
        },
        "aa_pattern": {
            "elapsed_s": float(result["elapsed_aa"]),
            "mlups": float(result["mlups_aa"]),
            "errors_vs_ghia": errors_aa,
            "pass_vs_ghia_15pct": aa_pass,
        },
        "separate": {
            "elapsed_s": float(result["elapsed_sep"]),
            "mlups": float(result["mlups_sep"]),
            "errors_vs_ghia": errors_sep,
            "pass_vs_ghia_15pct": sep_pass,
        },
        "performance": {
            "speedup_aa_over_sep": float(speedup),
        },
        "verdict": {
            "periodic_equivalence": "PASS" if periodic_pass else "FAIL",
            "aa_pattern_accuracy": "PASS" if aa_pass else "FAIL",
            "separate_accuracy": "PASS" if sep_pass else "FAIL",
        },
    }

    with open(out_path, "w") as fp:
        json.dump(result_json, fp, indent=2)
    print(f"\nResults saved to: {out_path}")

    return result_json


if __name__ == "__main__":
    main()
