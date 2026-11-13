"""Registry of cross-validation cases.

Each case declares: name, solver factory, reference data source, acceptance
thresholds, and parameter table.  The registry runs all registered cases and
produces a summary table of pass/fail with error metrics.

Design follows the "test registry" pattern (cf. pytest parametrize, but
explicit and self-documenting for V&V reports).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclasses.dataclass
class XValResult:
    """Result of a single cross-validation case.

    Attributes
    ----------
    case_name : str
        Name of the case.
    passed : bool
        Whether all acceptance criteria were met.
    metrics : dict
        Named error metrics (e.g. ``{'eps_inf': 0.012, 'eps_L2': 0.008}``).
    thresholds : dict
        Named acceptance thresholds.
    details : dict
        Additional details (e.g. reference values, computed values).
    state_hash : str or None
        SHA-256 hash of the final distribution (for bitwise repro check).
    benchmark : dict or None
        Benchmark timing data (if benchmarking was enabled).
    """

    case_name: str
    passed: bool
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    thresholds: Dict[str, float] = dataclasses.field(default_factory=dict)
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    state_hash: Optional[str] = None
    benchmark: Optional[Dict[str, float]] = None

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"[{status}] {self.case_name}: {metrics_str}"


@dataclasses.dataclass
class XValCase:
    """A cross-validation case specification.

    Attributes
    ----------
    name : str
        Unique case name (e.g. ``poiseuille_2d_H16``).
    run_fn : callable
        Function that runs the solver and returns a dict with:
        - ``'f'``: final distribution array (for hashing)
        - ``'scalars'``: dict of scalar metrics
        - ``'arrays'``: dict of named arrays (e.g. velocity profiles)
    reference_fn : callable or None
        Function that returns reference data dict with same keys as
        ``run_fn`` output.  If None, the case uses stored reference data.
    thresholds : dict
        Acceptance thresholds for each scalar metric.
    params : dict
        Parameter table for the case (documented in reports).
    category : str
        Category for grouping (e.g. 'analytical', 'experimental', 'benchmark').
    external_anchor : str
        Citation or DOI for the reference data.
    """

    name: str
    run_fn: Callable[[], Dict[str, Any]]
    reference_fn: Optional[Callable[[], Dict[str, Any]]] = None
    thresholds: Dict[str, float] = dataclasses.field(default_factory=dict)
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    category: str = "analytical"
    external_anchor: str = ""


class XValRegistry:
    """Registry and runner for cross-validation cases.

    Usage::

        reg = XValRegistry()
        reg.register(my_case)
        results = reg.run_all()
        reg.print_summary(results)
    """

    def __init__(self):
        self._cases: Dict[str, XValCase] = {}

    def register(self, case: XValCase) -> None:
        """Register a cross-validation case."""
        if case.name in self._cases:
            raise ValueError(f"Case '{case.name}' already registered")
        self._cases[case.name] = case

    def get(self, name: str) -> XValCase:
        """Get a registered case by name."""
        return self._cases[name]

    def list_cases(self) -> List[str]:
        """List all registered case names."""
        return list(self._cases.keys())

    def run_case(self, name: str, compute_hash: bool = True) -> XValResult:
        """Run a single cross-validation case.

        Parameters
        ----------
        name : str
            Case name.
        compute_hash : bool
            Whether to compute the state hash for bitwise reproducibility.

        Returns
        -------
        XValResult
        """
        case = self._cases[name]

        # Run solver
        result = case.run_fn()

        # Compute state hash
        state_hash = None
        if compute_hash and "f" in result:
            from .bitwise_refdata import compute_state_hash

            f_arr = result["f"]
            if not isinstance(f_arr, np.ndarray):
                f_arr = f_arr.detach().cpu().numpy()
            state_hash = compute_state_hash(f_arr)

        # Get reference data
        if case.reference_fn is not None:
            ref = case.reference_fn()
        else:
            ref = result.get("reference", {})

        # Compute metrics
        metrics = {}
        details = {}
        new_scalars = result.get("scalars", {})
        ref_scalars = ref.get("scalars", {})

        for k, new_val in new_scalars.items():
            if k in ref_scalars:
                ref_val = ref_scalars[k]
                if abs(ref_val) > 1e-15:
                    rel_err = abs(new_val - ref_val) / abs(ref_val)
                else:
                    rel_err = abs(new_val - ref_val)
                metrics[k] = rel_err
                details[k] = {"ref": ref_val, "new": new_val, "rel_err": rel_err}

        # Check thresholds.  Fail closed on non-finite metrics: ``nan > t`` is
        # False in IEEE-754, so the plain comparison below used to certify a
        # diverged run (all-NaN fields) as PASS.  A NaN/inf metric can never
        # satisfy an acceptance criterion (owner standard: error > 15%
        # unacceptable), so it is an explicit failure with a recorded reason.
        passed = True
        for k, val in metrics.items():
            if not np.isfinite(val):
                passed = False
                details[k] = {
                    "value": val,
                    "threshold": case.thresholds.get(k),
                    "nonfinite": True,
                    "reason": "metric is not finite (solver diverged or reference missing)",
                }
        for k, threshold in case.thresholds.items():
            if k in metrics and np.isfinite(metrics[k]):
                if metrics[k] > threshold:
                    passed = False

        # Also check absolute error for metrics that are already errors
        for k, val in new_scalars.items():
            if k.startswith("eps_") or k.startswith("err_"):
                if k in case.thresholds and (not np.isfinite(val) or val > case.thresholds[k]):
                    passed = False
                    metrics[k] = val
                    details[k] = {"value": val, "threshold": case.thresholds[k]}
                    if not np.isfinite(val):
                        details[k]["nonfinite"] = True
                        details[k]["reason"] = (
                            "metric is not finite (solver diverged or reference missing)"
                        )

        return XValResult(
            case_name=name,
            passed=passed,
            metrics=metrics,
            thresholds=case.thresholds,
            details=details,
            state_hash=state_hash,
            benchmark=result.get("benchmark"),
        )

    def run_all(self, compute_hash: bool = True) -> List[XValResult]:
        """Run all registered cases.

        Returns
        -------
        list[XValResult]
        """
        results = []
        for name in self._cases:
            results.append(self.run_case(name, compute_hash=compute_hash))
        return results

    @staticmethod
    def print_summary(results: List[XValResult]) -> str:
        """Print a summary table of all results.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = []
        lines.append("=" * 72)
        lines.append("Cross-Validation Summary")
        lines.append("=" * 72)
        n_pass = sum(1 for r in results if r.passed)
        n_fail = len(results) - n_pass
        lines.append(f"Total: {len(results)}  PASS: {n_pass}  FAIL: {n_fail}")
        lines.append("-" * 72)
        for r in results:
            lines.append(r.summary_line())
        lines.append("=" * 72)
        return "\n".join(lines)
