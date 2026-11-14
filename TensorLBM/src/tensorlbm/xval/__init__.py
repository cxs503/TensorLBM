"""Cross-validation (xval) module for TensorLBM.

Provides infrastructure for verification and validation of LBM solvers,
including bitwise reproducibility, reference-data regression, continuous
error analysis, and the YAML-driven XML template pipeline.
"""

from . import pipeline

__all__ = ["pipeline"]
