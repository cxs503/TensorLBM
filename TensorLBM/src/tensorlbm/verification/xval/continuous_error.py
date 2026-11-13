"""Continuous-error-print-vs-DOI-refs: analytical reference comparison framework.

This module implements the "continuous error" cross-validation approach:
for each LBM simulation case, compute the error between the discrete LBM
result and the continuous (analytical) reference solution cited by DOI.

The approach follows the pattern established in mature OSS implementations
(e.g., haimingz/vivsim continuous-error-print-vs-doi-refs):
1. Define an analytical solution with DOI citation
2. Run the LBM simulation to steady state or specified time
3. Sample the LBM field at the same locations as the analytical solution
4. Compute error norms (L2, L∞, relative)
5. Print/report the continuous error with DOI reference

Design principles
-----------------
* Every analytical reference must have a DOI or textbook citation.
* Error norms are computed consistently: L2 (RMS), L∞ (max), relative.
* Results are machine-readable (JSON) and human-readable (formatted table).
* Acceptance thresholds are declared per case, not hard-coded.

References
----------
* Poiseuille flow: Schlichting & Gersten, "Boundary-Layer Theory", 8th ed.,
  Springer (2017), §III.2.1 — u(y) = G/(2ν) · y(H−y)
* Couette flow: same reference, §III.2.2 — u(y) = U·y/H
* Womersley flow: Womersley, J.R. (1955), "Method for the calculation of
  velocity…", J. Physiol. 127:553–563. doi:10.1113/jphysiol.1955.sp005276
* Lid-driven cavity: Ghia et al. (1982), J. Comp. Phys. 48(3):387–411.
  doi:10.1016/0021-9991(82)90058-4
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class DOIReference:
    """A DOI-cited analytical or experimental reference.

    Attributes
    ----------
    doi : str
        Digital Object Identifier (e.g., "10.1016/0021-9991(82)90058-4").
    citation : str
        Full bibliographic citation.
    solution_type : str
        Type of reference: "analytical", "experimental", "numerical_benchmark".
    """
    doi: str
    citation: str
    solution_type: str = "analytical"


@dataclasses.dataclass
class ContinuousErrorResult:
    """Result of a continuous-error comparison.

    Attributes
    ----------
    case_name : str
        Name of the test case.
    doi_ref : DOIReference
        The DOI-cited reference used for comparison.
    l2_relative : float
        Relative L2 (RMS) error: ||u_num - u_ref||_2 / ||u_ref||_2.
    linf_relative : float
        Relative L∞ (max) error: max|u_num - u_ref| / max|u_ref|.
    l2_absolute : float
        Absolute L2 error: ||u_num - u_ref||_2.
    linf_absolute : float
        Absolute L∞ error: max|u_num - u_ref|.
    n_points : int
        Number of comparison points.
    passed : bool
        Whether the error is within the acceptance threshold.
    threshold : float
        Acceptance threshold (relative error).
    wall_time_s : float
        Wall-clock time for the simulation.
    params : dict
        Simulation parameters.
    """
    case_name: str
    doi_ref: DOIReference
    l2_relative: float = 0.0
    linf_relative: float = 0.0
    l2_absolute: float = 0.0
    linf_absolute: float = 0.0
    n_points: int = 0
    passed: bool = False
    threshold: float = 0.15
    wall_time_s: float = 0.0
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.case_name}: "
            f"L2_rel={self.l2_relative:.6f}  L∞_rel={self.linf_relative:.6f}  "
            f"n={self.n_points}  (threshold={self.threshold:.4f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "doi": self.doi_ref.doi,
            "citation": self.doi_ref.citation,
            "solution_type": self.doi_ref.solution_type,
            "l2_relative": self.l2_relative,
            "linf_relative": self.linf_relative,
            "l2_absolute": self.l2_absolute,
            "linf_absolute": self.linf_absolute,
            "n_points": self.n_points,
            "passed": self.passed,
            "threshold": self.threshold,
            "wall_time_s": round(self.wall_time_s, 3),
            "params": self.params,
        }


def compute_continuous_errors(
    u_numerical: np.ndarray,
    u_reference: np.ndarray,
    threshold: float = 0.15,
) -> Tuple[float, float, float, float, int, bool]:
    """Compute continuous error norms between numerical and reference solutions.

    Parameters
    ----------
    u_numerical : np.ndarray
        Numerical (LBM) solution values at comparison points.
    u_reference : np.ndarray
        Analytical/reference solution values at the same points.
    threshold : float
        Acceptance threshold for relative error (default 0.15 = 15%).

    Returns
    -------
    l2_rel, linf_rel, l2_abs, linf_abs, n_points, passed
    """
    assert u_numerical.shape == u_reference.shape, (
        f"Shape mismatch: {u_numerical.shape} vs {u_reference.shape}"
    )

    diff = u_numerical - u_reference
    n_points = u_numerical.size

    # Absolute errors
    l2_abs = float(np.linalg.norm(diff) / np.sqrt(n_points))  # RMS
    linf_abs = float(np.max(np.abs(diff)))

    # Relative errors
    ref_norm = np.linalg.norm(u_reference)
    ref_max = np.max(np.abs(u_reference))

    if ref_norm > 1e-15:
        l2_rel = float(np.linalg.norm(diff) / ref_norm)
    else:
        l2_rel = float(np.linalg.norm(diff))

    if ref_max > 1e-15:
        linf_rel = float(linf_abs / ref_max)
    else:
        linf_rel = float(linf_abs)

    passed = l2_rel <= threshold and linf_rel <= threshold

    return l2_rel, linf_rel, l2_abs, linf_abs, n_points, passed


def print_error_report(
    results: List[ContinuousErrorResult],
    title: str = "Continuous Error Report (vs DOI References)",
) -> str:
    """Print a formatted error report.

    Parameters
    ----------
    results : list of ContinuousErrorResult
        Results from one or more cases.
    title : str
        Report title.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)
    lines.append("")

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    lines.append(f"Total: {len(results)}  PASS: {n_pass}  FAIL: {n_fail}")
    lines.append("")

    # Table header
    lines.append(f"{'Case':<30} {'L2_rel':>10} {'L∞_rel':>10} {'n_pts':>8} {'DOI':>12} {'Status':>8}")
    lines.append("-" * 80)

    for r in results:
        doi_short = r.doi_ref.doi[:12] + "…" if len(r.doi_ref.doi) > 12 else r.doi_ref.doi
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"{r.case_name:<30} {r.l2_relative:>10.6f} {r.linf_relative:>10.6f} "
            f"{r.n_points:>8} {doi_short:>12} {status:>8}"
        )

    lines.append("-" * 80)

    # Detailed citations
    lines.append("")
    lines.append("DOI References:")
    seen_dois = set()
    for r in results:
        if r.doi_ref.doi not in seen_dois:
            lines.append(f"  {r.doi_ref.doi}: {r.doi_ref.citation}")
            seen_dois.add(r.doi_ref.doi)

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def save_artifacts(
    results: List[ContinuousErrorResult],
    output_dir: Path,
    label: str = "continuous_error_xval",
) -> Dict[str, str]:
    """Save cross-validation artifacts to disk.

    Parameters
    ----------
    results : list of ContinuousErrorResult
        Results to save.
    output_dir : Path
        Directory for output artifacts.
    label : str
        Label for the artifact files.

    Returns
    -------
    dict
        Paths to saved artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    results_json = {
        "schema": "tensorlbm/continuous_error_xval/1",
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_cases": len(results),
        "n_pass": sum(1 for r in results if r.passed),
        "n_fail": sum(1 for r in results if not r.passed),
        "results": [r.to_dict() for r in results],
    }

    json_path = output_dir / f"{label}.json"
    json_path.write_text(json.dumps(results_json, indent=2) + "\n")

    # CSV summary
    csv_path = output_dir / f"{label}.csv"
    csv_lines = ["case_name,l2_relative,linf_relative,n_points,passed,threshold,doi"]
    for r in results:
        csv_lines.append(
            f"{r.case_name},{r.l2_relative:.8f},{r.linf_relative:.8f},"
            f"{r.n_points},{r.passed},{r.threshold},{r.doi_ref.doi}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n")

    # SHA-256 of JSON for provenance
    sha = hashlib.sha256(json_path.read_bytes()).hexdigest()
    sha_path = output_dir / f"{label}.sha256"
    sha_path.write_text(f"{sha}  {json_path.name}\n")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "sha256": str(sha_path),
    }
