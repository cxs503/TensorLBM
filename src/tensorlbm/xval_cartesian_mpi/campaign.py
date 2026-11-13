"""Campaign driver for XVAL-LBM-CARTESIAN-MPI.

Runs the Poiseuille benchmark with both monolithic and domain-decomposed
TensorLBM, compares against the analytical solution and between the two
approaches, and writes the mission deliverables.

Deliverables:
    <out>/case_spec.json           the single spec consumed by all runners
    <out>/k_history_monolithic.csv  raw monolithic output
    <out>/k_history_decomposed.csv  raw decomposed output
    <out>/field_monolithic_final.csv  final velocity field (monolithic)
    <out>/field_decomposed_final.csv  final velocity field (decomposed)
    <out>/run_info_monolithic.json   run metadata
    <out>/run_info_decomposed.json   run metadata
    <out>/result.json                xval_table + criteria + status
    <out>/xval_cartesian_mpi.png     comparison figure
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .case_spec import PoiseuilleSpec
from .metrics import (
    compute_metrics,
    compare_monolithic_vs_decomposed,
    field_comparison,
)
from .runners import run_monolithic, run_decomposed


def run_campaign(
    spec: PoiseuilleSpec,
    out_dir: str | Path,
    *,
    dtype: str = "float64",
    threads: int = 8,
) -> dict[str, Any]:
    """Run the full cartesian-MPI cross-validation campaign.

    Parameters
    ----------
    spec : PoiseuilleSpec
        Case specification.
    out_dir : str | Path
        Output directory for artefacts.
    dtype : str
        Floating-point precision ("float64" or "float32").
    threads : int
        Number of CPU threads for torch.

    Returns
    -------
    dict
        Campaign result with xval_table, criteria, and status.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save case spec
    with open(out / "case_spec.json", "w") as fh:
        json.dump(spec.to_dict(), fh, indent=2)

    # Run monolithic
    print("[xval_cartesian_mpi] Running monolithic solver...")
    mono_result = run_monolithic(spec, out / "monolithic", dtype=dtype, threads=threads)

    # Run decomposed
    print("[xval_cartesian_mpi] Running decomposed solver (pylbm cartesian-MPI emulation)...")
    decomp_result = run_decomposed(spec, out / "decomposed", dtype=dtype, threads=threads)

    # Compute metrics
    mono_metrics = compute_metrics(mono_result, spec)
    decomp_metrics = compute_metrics(decomp_result, spec)
    decomp_comparison = compare_monolithic_vs_decomposed(mono_result, decomp_result, spec)

    # Field comparison
    field_comp = field_comparison(
        str(out / "monolithic" / "field_monolithic_final.csv"),
        str(out / "decomposed" / "field_decomposed_final.csv"),
    )

    # Build xval table
    xval_table = [
        {
            "tool": "tensorlbm_monolithic",
            **mono_metrics,
        },
        {
            "tool": "tensorlbm_decomposed",
            **decomp_metrics,
            "decomposition": f"{spec.npx}x{spec.npy}",
            "nprocs": spec.npx * spec.npy,
        },
    ]

    # Acceptance criteria (owner standard: <15% error vs analytical)
    accept_pct = 15.0
    criteria = {
        "mono_umax_error_pct": mono_metrics["umax_centreline_error_pct"],
        "decomp_umax_error_pct": decomp_metrics["umax_centreline_error_pct"],
        "mono_l2_error_pct": mono_metrics["l2_error_pct"],
        "decomp_l2_error_pct": decomp_metrics["l2_error_pct"],
        "decomposition_consistency": decomp_comparison["decomposition_consistent"],
        "max_centreline_diff": decomp_comparison["max_centreline_diff"],
        "fields_identical": field_comp["fields_identical"],
        "accept_threshold_pct": accept_pct,
        "mono_accepted": mono_metrics["umax_centreline_error_pct"] < accept_pct,
        "decomp_accepted": decomp_metrics["umax_centreline_error_pct"] < accept_pct,
    }

    # Bug suspects
    bug_suspects = []
    if not criteria["mono_accepted"]:
        bug_suspects.append({
            "tool": "tensorlbm_monolithic",
            "issue": f"umax error {mono_metrics['umax_centreline_error_pct']:.2f}% >= {accept_pct}%",
            "severity": "high",
        })
    if not criteria["decomp_accepted"]:
        bug_suspects.append({
            "tool": "tensorlbm_decomposed",
            "issue": f"umax error {decomp_metrics['umax_centreline_error_pct']:.2f}% >= {accept_pct}%",
            "severity": "high",
        })
    if not decomp_comparison["decomposition_consistent"]:
        bug_suspects.append({
            "tool": "decomposition",
            "issue": f"max centreline diff = {decomp_comparison['max_centreline_diff']:.2e}",
            "severity": "medium",
        })
    if not field_comp["fields_identical"]:
        bug_suspects.append({
            "tool": "field_comparison",
            "issue": f"max ux diff = {field_comp['max_ux_diff']:.2e}",
            "severity": "medium",
        })

    # Frictions
    frictions = []
    try:
        import mpi4py
    except ImportError:
        frictions.append({
            "item": "mpi4py",
            "impact": "Cannot run actual pylbm with MPI; using single-process emulation of cartesian-MPI decomposition",
            "severity": "medium",
        })

    # Overall status
    all_accepted = criteria["mono_accepted"] and criteria["decomp_accepted"]
    status = "pass" if all_accepted else "fail"

    result = {
        "anchor": "XVAL-LBM-CARTESIAN-MPI",
        "case_spec": spec.to_dict(),
        "xval_table": xval_table,
        "criteria": criteria,
        "bug_suspects": bug_suspects,
        "frictions": frictions,
        "status": status,
        "pylbm_comparison": {
            "approach": "cartesian-MPI domain decomposition (emulated)",
            "pylbm_source": "~/oss-cache/pylbm (v0.12.0)",
            "key_elements_copied": [
                "compute_dims-style load balancing",
                "cartesian topology with neighbour identification",
                "halo exchange after collision, before streaming",
                "region indices matching pylbm's get_region_indices",
            ],
            "differences_from_pylbm": [
                "Single-process emulation (no mpi4py available)",
                "TensorLBM D2Q9 BGK kernel (pylbm uses generated Cython/loopy code)",
                "Guo forcing (pylbm uses source term in moments)",
                "Half-way bounce-back (pylbm uses BFLBounceBack)",
            ],
        },
    }

    with open(out / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    # Write flat CSV
    with open(out / "xval_table.csv", "w") as fh:
        fh.write("tool,umax_centreline,umax_computed,umax_analytical,umax_centreline_error_pct,umax_computed_error_pct,l2_error_pct,n_samples,final_step\n")
        for row in xval_table:
            fh.write(f"{row['tool']},{row['umax_centreline']},{row['umax_computed']},{row['umax_analytical']},{row['umax_centreline_error_pct']},{row['umax_computed_error_pct']},{row['l2_error_pct']},{row['n_samples']},{row['final_step']}\n")

    # Generate figure
    try:
        _make_figure(out, spec, mono_result, decomp_result)
    except Exception as e:
        frictions.append({
            "item": "figure_generation",
            "impact": f"Failed to generate comparison figure: {e}",
            "severity": "low",
        })

    print(f"[xval_cartesian_mpi] Campaign complete: status={status}")
    print(f"  Monolithic umax error: {mono_metrics['umax_centreline_error_pct']:.4f}%")
    print(f"  Decomposed umax error: {decomp_metrics['umax_centreline_error_pct']:.4f}%")
    print(f"  Decomposition consistent: {decomp_comparison['decomposition_consistent']}")
    print(f"  Fields identical: {field_comp['fields_identical']}")

    return result


def _make_figure(
    out: Path,
    spec: PoiseuilleSpec,
    mono_result: dict,
    decomp_result: dict,
) -> None:
    """Generate comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mono_hist = mono_result["history"]
    decomp_hist = decomp_result["history"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Centreline velocity convergence
    ax = axes[0]
    ax.plot(mono_hist["step"], mono_hist["umax_centreline"], "b-o", markersize=3, label="Monolithic")
    ax.plot(decomp_hist["step"], decomp_hist["umax_centreline"], "r-s", markersize=3, label="Decomposed")
    ax.axhline(spec.u_max, color="k", linestyle="--", label=f"Analytical ({spec.u_max})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Centreline velocity")
    ax.set_title("Centreline velocity vs step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: L2 error convergence
    ax = axes[1]
    ax.plot(mono_hist["step"], mono_hist["l2_error_pct"], "b-o", markersize=3, label="Monolithic")
    ax.plot(decomp_hist["step"], decomp_hist["l2_error_pct"], "r-s", markersize=3, label="Decomposed")
    ax.set_xlabel("Step")
    ax.set_ylabel("L2 error (%)")
    ax.set_title("L2 error vs analytical")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Final velocity profile
    ax = axes[2]
    # Load final fields
    mono_ux, _ = _load_field(out / "monolithic" / "field_monolithic_final.csv")
    decomp_ux, _ = _load_field(out / "decomposed" / "field_decomposed_final.csv")

    # Average over x
    mono_profile = mono_ux.mean(axis=1)
    decomp_profile = decomp_ux.mean(axis=1)
    y = np.arange(spec.ny, dtype=float) + 0.5
    analytical = np.array([spec.analytical_velocity(yi) for yi in y])

    ax.plot(mono_profile, y, "b-", label="Monolithic")
    ax.plot(decomp_profile, y, "r--", label="Decomposed")
    ax.plot(analytical, y, "k:", label="Analytical")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.set_title("Final velocity profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "xval_cartesian_mpi.png", dpi=150)
    plt.close(fig)


def _load_field(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load field CSV."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    n_x = int(raw[:, 0].max()) + 1
    n_y = int(raw[:, 1].max()) + 1
    ux = np.zeros((n_y, n_x))
    uy = np.zeros((n_y, n_x))
    idx_i = raw[:, 0].astype(int)
    idx_j = raw[:, 1].astype(int)
    ux[idx_j, idx_i] = raw[:, 2]
    uy[idx_j, idx_i] = raw[:, 3]
    return ux, uy
