"""Gas-layer / microbubble drag-reduction closure for TensorLBM.

This package owns the solver-local physics of *gas-layer and microbubble drag
reduction under a ship bottom*, and the energy ledger that decides whether the
net power saving is positive:

``reference``      analytical plane-channel references (Cf, u+, Re_tau mapping)
``single_phase``   D2Q9 channel calibration gate (Cf and profile vs exact)
``laplace``        static-bubble Young-Laplace gate at density ratio >= 100
``two_phase``      Shan-Chen two-phase channel with a hull gas layer, mass-
                   conserving wall gas source, open outlet and sponge zone
``economics``      compressor-work / net-power ledger with compressor efficiency
                   sensitivity and the positive-net-benefit (U, Q_g) boundary
``literature``     sha256-anchored published drag-reduction reference envelope
``campaign``       campaign driver: gates + DR-theta-Q_g closure + artifacts
"""

from __future__ import annotations

from .api import design_point, load_closure, profit_boundary
from .economics import (
    compressor_specific_work,
    net_power_density,
    positive_net_benefit_boundary,
    required_efficiency,
)
from .laplace import LaplaceResult, run_laplace_gate
from .literature import (
    REFERENCE_TABLE,
    REFERENCE_TABLE_SHA256,
    literature_envelope_check,
)
from .single_phase import SinglePhaseChannel2D, run_single_phase_gate
from .two_phase import GasLayerChannel2D, TwoPhaseConfig

__all__ = [
    "REFERENCE_TABLE",
    "REFERENCE_TABLE_SHA256",
    "GasLayerChannel2D",
    "LaplaceResult",
    "SinglePhaseChannel2D",
    "design_point",
    "load_closure",
    "profit_boundary",
    "TwoPhaseConfig",
    "compressor_specific_work",
    "literature_envelope_check",
    "net_power_density",
    "positive_net_benefit_boundary",
    "required_efficiency",
    "run_laplace_gate",
    "run_single_phase_gate",
]