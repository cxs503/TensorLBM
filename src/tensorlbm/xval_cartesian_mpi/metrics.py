"""Metrics for XVAL-LBM-CARTESIAN-MPI: comparison between monolithic, decomposed, and analytical.

Metrics:
- ``umax_centreline``: centreline velocity vs analytical U_max
- ``umax_computed``: maximum velocity in domain vs analytical U_max
- ``l2_error_pct``: L2 norm of velocity error vs analytical profile
- ``profile_error_pct``: max pointwise error in velocity profile
- ``decomposition_consistency``: max difference between monolithic and decomposed fields
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .case_spec import PoiseuilleSpec


def analytical_profile(spec: PoiseuilleSpec, ny: int) -> np.ndarray:
    """Analytical Poiseuille velocity profile at ny points."""
    y = np.arange(ny, dtype=np.float64) + 0.5
    return 4.0 * spec.u_max / (spec.H ** 2) * y * (spec.H - y)


def compute_metrics(
    tool_result: dict[str, Any],
    spec: PoiseuilleSpec,
) -> dict[str, float]:
    """Reduce one tool's raw artefacts to comparison metrics."""
    hist = tool_result["history"]
    steps = np.asarray(hist["step"], dtype=float)
    u_centre = np.asarray(hist["umax_centreline"], dtype=float)
    u_max = np.asarray(hist["umax_computed"], dtype=float)
    l2_err = np.asarray(hist["l2_error_pct"], dtype=float)

    # Use the last sample (most developed flow)
    idx = -1
    return {
        "umax_centreline": float(u_centre[idx]),
        "umax_computed": float(u_max[idx]),
        "umax_analytical": spec.u_max,
        "umax_centreline_error_pct": float(abs(u_centre[idx] - spec.u_max) / spec.u_max * 100.0),
        "umax_computed_error_pct": float(abs(u_max[idx] - spec.u_max) / spec.u_max * 100.0),
        "l2_error_pct": float(l2_err[idx]),
        "n_samples": float(len(steps)),
        "final_step": float(steps[idx]),
    }


def compare_monolithic_vs_decomposed(
    mono_result: dict[str, Any],
    decomp_result: dict[str, Any],
    spec: PoiseuilleSpec,
) -> dict[str, float]:
    """Compare monolithic and decomposed results for consistency."""
    mono_hist = mono_result["history"]
    decomp_hist = decomp_result["history"]

    mono_u = np.asarray(mono_hist["umax_centreline"], dtype=float)
    decomp_u = np.asarray(decomp_hist["umax_centreline"], dtype=float)

    mono_l2 = np.asarray(mono_hist["l2_error_pct"], dtype=float)
    decomp_l2 = np.asarray(decomp_hist["l2_error_pct"], dtype=float)

    # Max difference in centreline velocity
    u_diff = np.abs(mono_u - decomp_u)
    max_u_diff = float(np.max(u_diff))
    mean_u_diff = float(np.mean(u_diff))

    # Difference in L2 error
    l2_diff = np.abs(mono_l2 - decomp_l2)
    max_l2_diff = float(np.max(l2_diff))

    return {
        "max_centreline_diff": max_u_diff,
        "mean_centreline_diff": mean_u_diff,
        "max_l2_error_diff": max_l2_diff,
        "decomposition_consistent": max_u_diff < 1e-10,  # should be bit-identical
    }


def load_field_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a field CSV (i,j,ux,uy) into 2D arrays."""
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


def field_comparison(
    mono_field_path: str,
    decomp_field_path: str,
) -> dict[str, float]:
    """Compare final velocity fields between monolithic and decomposed runs."""
    ux_mono, uy_mono = load_field_csv(mono_field_path)
    ux_decomp, uy_decomp = load_field_csv(decomp_field_path)

    diff_ux = np.abs(ux_mono - ux_decomp)
    diff_uy = np.abs(uy_mono - uy_decomp)

    return {
        "max_ux_diff": float(np.max(diff_ux)),
        "mean_ux_diff": float(np.mean(diff_ux)),
        "max_uy_diff": float(np.max(diff_uy)),
        "mean_uy_diff": float(np.mean(diff_uy)),
        "fields_identical": bool(np.allclose(ux_mono, ux_decomp, atol=1e-14) and
                                  np.allclose(uy_mono, uy_decomp, atol=1e-14)),
    }
