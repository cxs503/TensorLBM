"""Cross-validation framework: bitwise reproducibility, reference-data regression,
continuous-error-vs-DOI-refs, and benchmarking.

This module provides infrastructure for:

1. **Bitwise reproducibility**: deterministic seeding, float-precision control,
   and hash-based verification that identical inputs produce identical outputs.
2. **Reference-data regression**: store golden reference snapshots (numpy arrays
   + scalar metrics) and detect regressions by comparing new runs against them.
3. **Continuous-error-vs-DOI-refs**: compare LBM results against analytical
   solutions cited by DOI, computing L2/L∞ error norms.
4. **Benchmarking**: wall-clock timing with warm-up, repetition, and
   statistical reporting (mean, std, min, max).

Design principles
-----------------
* Zero external dependencies beyond numpy/torch (already required by TensorLBM).
* Reference data stored as compressed ``.npz`` files alongside the source code
  so that regression tests are self-contained and version-controlled.
* Acceptance thresholds are *declared* per case, not hard-coded in the framework.
* All hash computations use SHA-256 for collision resistance.

Method selection rationale (external_anchor)
--------------------------------------------
* Bitwise reproducibility: Krüger et al. (2017) "The Lattice Boltzmann Method",
  §4.4 (deterministic collision/streaming); Latt (2007) "Choice of units in LB",
  for seed control.
* Reference-data regression: standard practice in CFD codes (OpenFOAM functionObjects,
  SU2 regression tests); we use numpy .npz for portability.
* Continuous-error-vs-DOI-refs: follows the pattern from haimingz/vivsim
  continuous-error-print-vs-doi-refs — compute analytical reference at DOI-cited
  conditions, compare LBM output, report error norms with citation.
* Benchmarking: ASME V&V 20-2009 "Standard for Verification and Validation in
  Computational Fluid Dynamics and Heat Transfer" for convergence metrics.
"""

from .bitwise_refdata import (
    BitwiseReproConfig,
    ReferenceDataStore,
    compute_state_hash,
    verify_bitwise_reproducibility,
)
from .benchmark import BenchmarkConfig, BenchmarkResult, run_benchmark
from .registry import XValCase, XValRegistry, XValResult
from .continuous_error import (
    DOIReference,
    ContinuousErrorResult,
    compute_continuous_errors,
    print_error_report,
    save_artifacts,
)

__all__ = [
    "BitwiseReproConfig",
    "ReferenceDataStore",
    "compute_state_hash",
    "verify_bitwise_reproducibility",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "XValCase",
    "XValRegistry",
    "XValResult",
    "DOIReference",
    "ContinuousErrorResult",
    "compute_continuous_errors",
    "print_error_report",
    "save_artifacts",
]
