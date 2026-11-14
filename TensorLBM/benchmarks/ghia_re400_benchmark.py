#!/usr/bin/env python3
"""Ghia Re=400 benchmark for TensorLBM D2Q9 lid-driven cavity.

Runs the cavity solver at Re=400 with multiple resolutions, compares against
the BOUND Ghia (1982) reference data, and produces:
- artifacts/ghia_re400_comparison.json  (numerical results)
- artifacts/ghia_re400_profiles.png     (comparison plots)
- artifacts/ghia_re400_report.md        (markdown report)

Usage:
    python benchmarks/ghia_re400_benchmark.py [--n 129] [--max-steps 50000]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.verification.lid_driven_cavity import (
    CavityConfig,
    CavitySolver,
    get_ghia_reference,
    interpolate_ghia,
)

ARTIFACTS = REPO / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def run_cavity_re400(n: int = 129, max_steps: int = 50000, u_lid: float = 0.1,
                     use_rlbm: bool = True, ramp_steps: int = 2000) -> dict:
    """Run a single Re=400 cavity simulation and return profiles + errors."""
    cfg = CavityConfig(
        n=n, re=400.0, u_lid=u_lid, max_steps=max_steps,
        use_rlbm=use_rlbm, ramp_steps=ramp_steps,
    )
    solver = CavitySolver(cfg)

    t0 = time.time()
    solver.run(resid_tol=1e-7, monitor_interval=2000)
    elapsed = time.time() - t0

    y_norm, u_norm, x_norm, v_norm = solver.get_profiles()

    ref = get_ghia_reference(400.0)

    # Interpolate reference at simulation points
    u_ref = interpolate_ghia(ref, y_norm, axis='u')
    v_ref = interpolate_ghia(ref, x_norm, axis='v')

    # Error metrics
    l2_u = float(np.sqrt(np.mean((u_norm - u_ref)**2)))
    l2_v = float(np.sqrt(np.mean((v_norm - v_ref)**2)))
    max_u = float(np.max(np.abs(u_norm - u_ref)))
    max_v = float(np.max(np.abs(v_norm - v_ref)))

    # Point-wise relative errors (avoid division by zero)
    rel_u = np.where(np.abs(u_ref) > 1e-6,
                     np.abs(u_norm - u_ref) / np.abs(u_ref),
                     np.abs(u_norm - u_ref))
    rel_v = np.where(np.abs(v_ref) > 1e-6,
                     np.abs(v_norm - v_ref) / np.abs(v_ref),
                     np.abs(v_norm - v_ref))
    max_rel_u = float(np.max(rel_u))
    max_rel_v = float(np.max(rel_v))

    # Extrema
    u_min_idx = np.argmin(u_norm)
    v_max_idx = np.argmax(np.abs(v_norm))

    # Reference extrema
    ref_u_min = float(np.min(ref.u))
    ref_v_min = float(np.min(ref.v))

    return {
        "n": n,
        "re": 400.0,
        "u_lid": u_lid,
        "use_rlbm": use_rlbm,
        "ramp_steps": ramp_steps,
        "elapsed_seconds": round(elapsed, 1),
        "profiles": {
            "y": y_norm.tolist(),
            "u_sim": u_norm.tolist(),
            "u_ref": u_ref.tolist(),
            "x": x_norm.tolist(),
            "v_sim": v_norm.tolist(),
            "v_ref": v_ref.tolist(),
        },
        "errors": {
            "l2_u": l2_u,
            "l2_v": l2_v,
            "max_u": max_u,
            "max_v": max_v,
            "max_rel_u": max_rel_u,
            "max_rel_v": max_rel_v,
        },
        "extrema": {
            "u_min_sim": float(u_norm[u_min_idx]),
            "u_min_y": float(y_norm[u_min_idx]),
            "u_min_ref": ref_u_min,
            "u_min_rel_err": abs(u_norm[u_min_idx] - ref_u_min) / abs(ref_u_min) * 100,
            "v_min_sim": float(v_norm[v_max_idx]),
            "v_min_x": float(x_norm[v_max_idx]),
            "v_min_ref": ref_v_min,
            "v_min_rel_err": abs(v_norm[v_max_idx] - ref_v_min) / abs(ref_v_min) * 100,
        },
    }


def plot_comparison(results: list[dict], output_path: Path):
    """Generate comparison plots for all resolutions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {129: "tab:blue", 257: "tab:orange", 65: "tab:green"}

    # u profile
    ax = axes[0]
    for r in results:
        c = colors.get(r["n"], "tab:gray")
        ax.plot(r["profiles"]["u_ref"], r["profiles"]["y"],
                "k--", lw=1, alpha=0.3, label="_nolegend_" if r["n"] != 129 else None)
        ax.plot(r["profiles"]["u_sim"], r["profiles"]["y"],
                "o-", ms=3, c=c, label=f"n={r['n']}")
    ax.set_xlabel("u / U_lid")
    ax.set_ylabel("y / H")
    ax.set_title("Vertical centreline u-velocity (Re=400)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # v profile
    ax = axes[1]
    for r in results:
        c = colors.get(r["n"], "tab:gray")
        ax.plot(r["profiles"]["v_ref"], r["profiles"]["x"],
                "k--", lw=1, alpha=0.3, label="_nolegend_" if r["n"] != 129 else None)
        ax.plot(r["profiles"]["v_sim"], r["profiles"]["x"],
                "s-", ms=3, c=c, label=f"n={r['n']}")
    ax.set_xlabel("v / U_lid")
    ax.set_ylabel("x / H")
    ax.set_title("Horizontal centreline v-velocity (Re=400)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Ghia et al. (1982) Re=400: TensorLBM D2Q9 RLBM vs reference", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_report(results: list[dict]) -> str:
    """Generate markdown report."""
    lines = [
        "# Ghia et al. (1982) Re=400 Lid-Driven Cavity Benchmark",
        "",
        "## Problem Setting",
        "",
        "- **Geometry**: Square cavity (1×1), moving lid at u=U_lid, no-slip walls",
        "- **Reynolds number**: Re = 400 (based on lid velocity and cavity width)",
        "- **Reference**: Ghia, U., Ghia, K.N., & Shin, C.T. (1982). JCP 48:387-411, Tables I & II",
        "- **Provenance**: BOUND from OCR witness (sha256: 3637ef9e...)",
        "- **Method**: D2Q9 LBM with Regularized BGK (RLBM), half-way bounce-back",
        "",
        "## Reference Data (Re=400, BOUND)",
        "",
        "| Quantity | Value | Source |",
        "|----------|-------|--------|",
        "| u_min | -0.32726 | Table I, y=0.2813 |",
        "| u(y=0.5) | -0.11477 | Table I |",
        "| v_min | -0.44993 | Table II, x=0.8594 |",
        "| v(x=0.5) | 0.05186 | Table II |",
        "",
        "## Results",
        "",
    ]

    for r in results:
        lines.append(f"### Resolution n={r['n']}")
        lines.append("")
        lines.append(f"- **Runtime**: {r['elapsed_seconds']:.1f}s")
        lines.append(f"- **L2 error (u)**: {r['errors']['l2_u']:.6f}")
        lines.append(f"- **L2 error (v)**: {r['errors']['l2_v']:.6f}")
        lines.append(f"- **Max error (u)**: {r['errors']['max_u']:.6f}")
        lines.append(f"- **Max error (v)**: {r['errors']['max_v']:.6f}")
        lines.append(f"- **Max relative error (u)**: {r['errors']['max_rel_u']*100:.2f}%")
        lines.append(f"- **Max relative error (v)**: {r['errors']['max_rel_v']*100:.2f}%")
        lines.append("")
        lines.append("| Quantity | Simulation | Reference | Rel. Error |")
        lines.append("|----------|-----------|-----------|------------|")
        lines.append(f"| u_min | {r['extrema']['u_min_sim']:.5f} | {r['extrema']['u_min_ref']:.5f} | {r['extrema']['u_min_rel_err']:.2f}% |")
        lines.append(f"| u_min location (y) | {r['extrema']['u_min_y']:.4f} | 0.2813 | — |")
        lines.append(f"| v_min | {r['extrema']['v_min_sim']:.5f} | {r['extrema']['v_min_ref']:.5f} | {r['extrema']['v_min_rel_err']:.2f}% |")
        lines.append(f"| v_min location (x) | {r['extrema']['v_min_x']:.4f} | 0.8594 | — |")
        lines.append("")

        # Pass/fail
        passed = r['errors']['max_rel_u'] < 0.05 and r['errors']['max_rel_v'] < 0.05
        lines.append(f"**Max-point relative error < 5%**: {'✓ PASS' if passed else '✗ FAIL'}")
        lines.append("")

    lines.append("## Suspect Entries Removed")
    lines.append("")
    lines.append("The following fabricated entries from the old `reference_data.py` have been")
    lines.append("replaced with BOUND transcriptions from the OCR witness:")
    lines.append("")
    lines.append("| Re | Issue | Action |")
    lines.append("|----|-------|--------|")
    lines.append("| 400 | Values do not match OCR at any coordinate; max rel error >400% | REPLACED |")
    lines.append("| 100 | UNVERIFIED_OFFLINE; centre value 24% off | REPLACED |")
    lines.append("| 1000 | UNVERIFIED_OFFLINE; wrong coordinate grid | REPLACED |")
    lines.append("")
    lines.append("See `benchmarks/anchors/ghia1982_provenance.json` for full details.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    n_values = [129, 257]  # Primary + convergence check
    results = []

    for n in n_values:
        print(f"Running Re=400, n={n}...")
        r = run_cavity_re400(n=n, max_steps=50000)
        results.append(r)
        print(f"  L2_u={r['errors']['l2_u']:.6f}, L2_v={r['errors']['l2_v']:.6f}")
        print(f"  max_rel_u={r['errors']['max_rel_u']*100:.2f}%, max_rel_v={r['errors']['max_rel_v']*100:.2f}%")

    # Save JSON
    json_path = ARTIFACTS / "ghia_re400_comparison.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {json_path}")

    # Plot
    png_path = ARTIFACTS / "ghia_re400_profiles.png"
    plot_comparison(results, png_path)
    print(f"Saved: {png_path}")

    # Report
    report = generate_report(results)
    md_path = ARTIFACTS / "ghia_re400_report.md"
    md_path.write_text(report)
    print(f"Saved: {md_path}")

    # Final verdict
    primary = results[0]  # n=129
    passed = primary['errors']['max_rel_u'] < 0.05 and primary['errors']['max_rel_v'] < 0.05
    print(f"\n=== Verdict (n={primary['n']}) ===")
    print(f"Max-point relative error (u): {primary['errors']['max_rel_u']*100:.2f}% {'< 5% ✓' if primary['errors']['max_rel_u'] < 0.05 else '≥ 5% ✗'}")
    print(f"Max-point relative error (v): {primary['errors']['max_rel_v']*100:.2f}% {'< 5% ✓' if primary['errors']['max_rel_v'] < 0.05 else '≥ 5% ✗'}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
