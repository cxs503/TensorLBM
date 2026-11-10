"""Acoustic impedance boundary + streaming unified verification surface.

Campaign node P88W1-N02: combines Delany-Bazley porous absorber TMM
reflection-coefficient anchors with Rayleigh/Schlichting second-order
acoustic streaming verification.
"""
from .impedance_verification import (
    ImpedanceVerificationResult,
    run_impedance_verification,
)
from .streaming_verification import (
    StreamingVerificationResult,
    run_streaming_verification,
    run_scaling_law_verification,
)
from .unified import (
    AcousticVerificationReport,
    run_full_acoustic_verification,
)

__all__ = [
    "ImpedanceVerificationResult",
    "run_impedance_verification",
    "StreamingVerificationResult",
    "run_streaming_verification",
    "run_scaling_law_verification",
    "AcousticVerificationReport",
    "run_full_acoustic_verification",
]
