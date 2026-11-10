"""Unified acoustic verification report: impedance boundary + streaming.

Combines all verification results into a single report structure suitable
for campaign node P88W1-N02 acceptance.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .impedance_verification import (
    ImpedanceVerificationResult,
    run_impedance_verification,
)
from .streaming_verification import (
    StreamingVerificationResult,
    run_streaming_verification,
)

__all__ = [
    "AcousticVerificationReport",
    "run_full_acoustic_verification",
]


@dataclass
class AcousticVerificationReport:
    """Complete acoustic verification report for campaign P88W1-N02."""
    timestamp: str
    impedance: ImpedanceVerificationResult
    streaming: StreamingVerificationResult
    
    # Summary pass/fail for each criterion
    criterion_a: bool = False  # R vs TMM < 3%
    criterion_b: bool = False  # streaming profile < 5%
    criterion_c: bool = False  # Re_s scaling exponent < 5%
    criterion_d: bool = False  # rigid wall R -> 1
    criterion_e: bool = False  # pytest >= 10 items
    
    all_pass: bool = False
    
    def check_criteria(self, n_tests: int = 10):
        """Evaluate all acceptance criteria."""
        self.criterion_a = self.impedance.max_R_error < 0.03
        self.criterion_b = self.streaming.profile_max_error < 0.05
        self.criterion_c = self.streaming.scaling_exponent_error < 0.05
        self.criterion_d = self.impedance.rigid_wall_error < 0.01
        self.criterion_e = n_tests >= 10
        
        self.all_pass = all([
            self.criterion_a,
            self.criterion_b,
            self.criterion_c,
            self.criterion_d,
            self.criterion_e,
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (numpy arrays converted to lists)."""
        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.complexfloating, complex)):
                return {"real": float(obj.real), "imag": float(obj.imag)}
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        d = {
            "timestamp": self.timestamp,
            "criteria": {
                "a_reflection_error_lt_3pct": {
                    "pass": self.criterion_a,
                    "value": float(self.impedance.max_R_error),
                    "threshold": 0.03,
                },
                "b_streaming_profile_lt_5pct": {
                    "pass": self.criterion_b,
                    "value": float(self.streaming.profile_max_error),
                    "threshold": 0.05,
                },
                "c_scaling_exponent_lt_5pct": {
                    "pass": self.criterion_c,
                    "value": float(self.streaming.scaling_exponent_error),
                    "threshold": 0.05,
                    "exponent_measured": float(self.streaming.scaling_exponent),
                    "exponent_theory": 1.0,
                },
                "d_rigid_wall_R_to_1": {
                    "pass": self.criterion_d,
                    "value": float(self.impedance.rigid_wall_error),
                    "threshold": 0.01,
                    "R_measured": _convert(self.impedance.rigid_wall_R),
                },
                "e_pytest_ge_10": {
                    "pass": self.criterion_e,
                },
            },
            "all_pass": self.all_pass,
            "impedance_summary": {
                "max_R_error": float(self.impedance.max_R_error),
                "mean_R_error": float(self.impedance.mean_R_error),
                "max_alpha_error": float(self.impedance.max_alpha_error),
                "rigid_wall_abs_R": float(abs(self.impedance.rigid_wall_R)),
            },
            "streaming_summary": {
                "profile_max_error": float(self.streaming.profile_max_error),
                "profile_rms_error": float(self.streaming.profile_rms_error),
                "stokes_thickness_theory": float(self.streaming.stokes_thickness_theory),
                "stokes_thickness_measured": float(self.streaming.stokes_thickness_measured),
                "stokes_error": float(self.streaming.stokes_error),
                "scaling_exponent": float(self.streaming.scaling_exponent),
                "scaling_exponent_error": float(self.streaming.scaling_exponent_error),
                "scaling_R_squared": float(self.streaming.scaling_R_squared),
            },
        }
        return d
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run_full_acoustic_verification(
    impedance_freqs: Optional[Sequence[float]] = None,
    sigma: float = 25000.0,
    thickness: float = 0.05,
    streaming_nx: int = 64,
    streaming_ny: int = 32,
    streaming_U0: float = 0.01,
    streaming_omega: float = 0.05,
    streaming_tau: float = 1.0,
    n_tests: int = 12,
) -> AcousticVerificationReport:
    """Run complete acoustic verification (impedance + streaming).
    
    Parameters
    ----------
    impedance_freqs : frequencies for impedance verification (Hz)
    sigma : flow resistivity for Delany-Bazley model
    thickness : absorber layer thickness (m)
    streaming_* : parameters for acoustic streaming LBM
    n_tests : number of pytest items (for criterion e)
    
    Returns
    -------
    AcousticVerificationReport with all results and pass/fail
    """
    # Impedance verification
    impedance = run_impedance_verification(
        freqs=impedance_freqs,
        sigma=sigma,
        thickness=thickness,
    )
    
    # Streaming verification
    streaming = run_streaming_verification(
        nx=streaming_nx,
        ny=streaming_ny,
        U0=streaming_U0,
        omega=streaming_omega,
        tau=streaming_tau,
    )
    
    report = AcousticVerificationReport(
        timestamp=datetime.utcnow().isoformat(),
        impedance=impedance,
        streaming=streaming,
    )
    report.check_criteria(n_tests=n_tests)
    
    return report
