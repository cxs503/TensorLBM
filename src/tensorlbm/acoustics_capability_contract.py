"""Fail-closed public capability contract for aeroacoustic post-processing.

This module is an audit boundary, not an acoustics dispatcher.  It reports the
post-processing functions actually present in the repository and refuses to
certify any acoustic capability as physics-validated.  In particular,
successful shape / finiteness / causality contract tests do not establish FWH
acoustic accuracy, spectral correctness, or experimental validation.

Audit scope (source files read, not docstring assertions):
    - ``tensorlbm/acoustics.py`` – FWH far-field, SPL spectrum, surface-pressure
      extraction, OASPL, and the ``compute_fwh_result`` convenience wrapper.
    - No ``acoustic_beamforming.py`` exists in the repository.
    - tests: ``test_acoustics.py`` (shape, finite, causality, mean-removal,
      zero-pressure floor, known-tone OASPL, n_fft padding).

Key architectural property:
    The entire acoustics module is **post-processing**.  It operates on saved
    density / pressure history and never enters the timestep hot path.  It can
    be composed with any collision / turbulence / boundary configuration as a
    post-processing step without modifying solver internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

AcousticsFunction = Literal[
    "fwh_far_field",
    "spl_spectrum",
    "surface_pressure_extraction",
    "oaspl",
    "fwh_result_wrapper",
]
LatticeName = Literal["D2Q9", "D3Q19", "D3Q27", "N/A"]

# ---------------------------------------------------------------------------
# Machine-readable withheld codes (fail-closed)
# ---------------------------------------------------------------------------

WITHHELD_NO_PHYSICS_VALIDATION = "WITHHELD_NO_PHYSICS_VALIDATION"
"""Implementation exists with contract tests, but no physics validation evidence."""

WITHHELD_NO_IMPLEMENTATION = "WITHHELD_NO_IMPLEMENTATION"
"""No implementation found for this function/lattice combination."""

WITHHELD_UNKNOWN_FUNCTION = "WITHHELD_UNKNOWN_FUNCTION"
WITHHELD_UNKNOWN_LATTICE = "WITHHELD_UNKNOWN_LATTICE"

# ---------------------------------------------------------------------------
# Verification levels
# ---------------------------------------------------------------------------

VERIFICATION_CONTRACT_TESTED = "CONTRACT_TESTED"
"""Shape / finiteness / causality / identity unit tests exist.

These verify operator well-formedness, NOT acoustic physics accuracy, spectral
accuracy, or experimental validation.
"""

VERIFICATION_NO_IMPLEMENTATION = "NO_IMPLEMENTATION"
"""No implementation found for this combination."""

# ---------------------------------------------------------------------------
# Implementation statuses
# ---------------------------------------------------------------------------

IMPLEMENTED = "IMPLEMENTED"
NO_IMPLEMENTATION = "NO_IMPLEMENTATION"

# ---------------------------------------------------------------------------
# Audited dimensions
# ---------------------------------------------------------------------------

_AUDITED_FUNCTIONS: tuple[str, ...] = (
    "fwh_far_field",
    "spl_spectrum",
    "surface_pressure_extraction",
    "oaspl",
    "fwh_result_wrapper",
)
_AUDITED_LATTICES: tuple[str, ...] = ("D2Q9", "D3Q19", "D3Q27", "N/A")

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class AcousticsWithheldError(NotImplementedError):
    """Raised when an acoustics capability request lacks physics validation."""


# ---------------------------------------------------------------------------
# Capability dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcousticsCapability:
    """Audited state of one acoustics function/lattice combination.

    ``verification_level`` records what evidence exists in the repository.
    ``status`` is the fail-closed frontend claim: it is always a
    ``WITHHELD_*`` code because no current combination has physics validation
    evidence.

    ``hot_path_note`` is always ``None`` for implemented acoustics functions
    because the entire module is post-processing and does not enter the
    timestep hot path.
    """

    function: str
    lattice: str
    implementation_status: str
    verification_level: str
    status: str
    entrypoint: str | None
    test_evidence: str | None
    hot_path_note: str | None
    note: str

    @property
    def available(self) -> bool:
        """Whether a frontend may claim this combination is physics-validated."""
        return self.status == "AVAILABLE"


# ---------------------------------------------------------------------------
# Post-processing audit entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PostProcessingAudit:
    """Observation that a function is post-processing (not in the timestep hot path)."""

    function: str
    entrypoint: str
    path_type: str  # always "POST_PROCESSING"
    note: str


# ---------------------------------------------------------------------------
# Audit registry
#
# Each entry maps (function, lattice) to a tuple of:
#   (implementation_status, verification_level, entrypoint,
#    test_evidence, hot_path_note, note)
#
# Lattice "N/A" is used for lattice-agnostic functions that operate on the
# FWHSurface / pressure-tensor data structure rather than directly on a lattice
# grid.  Lattice-specific entries for those functions are NO_IMPLEMENTATION.
# ---------------------------------------------------------------------------

_RegistryEntry = tuple[str, str, str | None, str | None, str | None, str]

_REGISTRY: dict[str, dict[str, _RegistryEntry]] = {
    # -----------------------------------------------------------------------
    # FWH far-field — lattice-agnostic (operates on FWHSurface data structure)
    # -----------------------------------------------------------------------
    "fwh_far_field": {
        "N/A": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.acoustics.compute_fwh_far_field",
            "test_acoustics.py: shape, finite, zero-pressure→zero-output, "
            "causality (no signal before propagation delay), multi-observer, "
            "invalid-shape raises",
            None,
            "FWH compact-source far-field approximation (1/r + 1/r² terms). "
            "Post-processing: operates on saved pressure history, not in timestep "
            "hot path. Composable with any collision/turbulence/boundary as a "
            "post-processing step.",
        ),
        "D2Q9": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "FWH far-field is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q19": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "FWH far-field is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q27": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "FWH far-field is lattice-agnostic; use lattice='N/A'.",
        ),
    },

    # -----------------------------------------------------------------------
    # SPL spectrum — lattice-agnostic (operates on pressure time series)
    # -----------------------------------------------------------------------
    "spl_spectrum": {
        "N/A": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.acoustics.compute_spl_spectrum",
            "test_acoustics.py: shape, finite, frequency-axis correctness, "
            "zero-pressure eps-floor, n_fft padding",
            None,
            "Welch-method SPL spectrum via Hann-windowed rFFT. "
            "Post-processing: operates on pressure time series, not in timestep "
            "hot path. Composable with any collision/turbulence/boundary as a "
            "post-processing step.",
        ),
        "D2Q9": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "SPL spectrum is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q19": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "SPL spectrum is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q27": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "SPL spectrum is lattice-agnostic; use lattice='N/A'.",
        ),
    },

    # -----------------------------------------------------------------------
    # Surface pressure extraction — lattice-specific (reads rho_history grid)
    # -----------------------------------------------------------------------
    "surface_pressure_extraction": {
        "D2Q9": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.acoustics.extract_surface_pressure",
            "test_acoustics.py: 2D shape, mean-removal, pressure ∝ density "
            "fluctuation (c_s²=1/3)",
            None,
            "Extracts pressure fluctuations from 2-D density history (ρ-ρ̄)·c_s². "
            "Post-processing: operates on saved density history, not in timestep "
            "hot path. Composable with any collision/turbulence/boundary as a "
            "post-processing step.",
        ),
        "D3Q19": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.acoustics.extract_surface_pressure",
            "test_acoustics.py: 3D shape, mean-removal",
            None,
            "Extracts pressure fluctuations from 3-D density history (ρ-ρ̄)·c_s². "
            "Post-processing: operates on saved density history, not in timestep "
            "hot path. Composable with any collision/turbulence/boundary as a "
            "post-processing step.",
        ),
        "D3Q27": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.acoustics.extract_surface_pressure",
            "test_acoustics.py: 3D shape (same code path as D3Q19)",
            None,
            "Extracts pressure fluctuations from 3-D density history (ρ-ρ̄)·c_s². "
            "Post-processing: operates on saved density history, not in timestep "
            "hot path. Composable with any collision/turbulence/boundary as a "
            "post-processing step.",
        ),
        "N/A": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "Surface pressure extraction is lattice-specific; specify D2Q9, D3Q19, or D3Q27.",
        ),
    },

    # -----------------------------------------------------------------------
    # OASPL — lattice-agnostic (operates on pressure time series)
    # -----------------------------------------------------------------------
    "oaspl": {
        "N/A": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.acoustics.oaspl",
            "test_acoustics.py: output length, finite, zero-pressure eps-floor, "
            "known-tone RMS",
            None,
            "Overall Sound Pressure Level: 20·log10(p_rms / p_ref). "
            "Post-processing: operates on pressure time series, not in timestep "
            "hot path. Composable with any collision/turbulence/boundary as a "
            "post-processing step.",
        ),
        "D2Q9": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "OASPL is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q19": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "OASPL is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q27": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "OASPL is lattice-agnostic; use lattice='N/A'.",
        ),
    },

    # -----------------------------------------------------------------------
    # FWH result wrapper — lattice-agnostic (convenience wrapper)
    # -----------------------------------------------------------------------
    "fwh_result_wrapper": {
        "N/A": (
            IMPLEMENTED, VERIFICATION_CONTRACT_TESTED,
            "tensorlbm.acoustics.compute_fwh_result",
            "test_acoustics.py: result has all fields (time, p_prime, frequencies, "
            "spl, oaspl, observers), finite output",
            None,
            "Convenience wrapper: runs FWH integration + spectral analysis + OASPL "
            "in one call. Post-processing: not in timestep hot path. Composable "
            "with any collision/turbulence/boundary as a post-processing step.",
        ),
        "D2Q9": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "FWH result wrapper is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q19": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "FWH result wrapper is lattice-agnostic; use lattice='N/A'.",
        ),
        "D3Q27": (
            NO_IMPLEMENTATION, VERIFICATION_NO_IMPLEMENTATION,
            None, None, None,
            "FWH result wrapper is lattice-agnostic; use lattice='N/A'.",
        ),
    },
}


# ---------------------------------------------------------------------------
# Post-processing audit registry
#
# Every implemented acoustics function is post-processing: it operates on saved
# fields and does not enter the timestep hot path.  This is a cold-path audit
# of the module's architectural role; it does not modify any numerical kernel.
# ---------------------------------------------------------------------------

_POST_PROCESSING_AUDIT: tuple[PostProcessingAudit, ...] = (
    PostProcessingAudit(
        function="fwh_far_field",
        entrypoint="tensorlbm.acoustics.compute_fwh_far_field",
        path_type="POST_PROCESSING",
        note="Operates on saved FWHSurface pressure history; not in timestep hot path.",
    ),
    PostProcessingAudit(
        function="spl_spectrum",
        entrypoint="tensorlbm.acoustics.compute_spl_spectrum",
        path_type="POST_PROCESSING",
        note="Operates on saved pressure time series; not in timestep hot path.",
    ),
    PostProcessingAudit(
        function="surface_pressure_extraction",
        entrypoint="tensorlbm.acoustics.extract_surface_pressure",
        path_type="POST_PROCESSING",
        note="Operates on saved density history; not in timestep hot path.",
    ),
    PostProcessingAudit(
        function="oaspl",
        entrypoint="tensorlbm.acoustics.oaspl",
        path_type="POST_PROCESSING",
        note="Operates on saved pressure time series; not in timestep hot path.",
    ),
    PostProcessingAudit(
        function="fwh_result_wrapper",
        entrypoint="tensorlbm.acoustics.compute_fwh_result",
        path_type="POST_PROCESSING",
        note="Convenience wrapper chaining FWH + SPL + OASPL; not in timestep hot path.",
    ),
)


# ---------------------------------------------------------------------------
# Internal: determine fail-closed status from verification level
# ---------------------------------------------------------------------------

def _status_for(verification_level: str) -> str:
    if verification_level == VERIFICATION_NO_IMPLEMENTATION:
        return WITHHELD_NO_IMPLEMENTATION
    # CONTRACT_TESTED: implementation exists with contract tests, but no
    # physics validation evidence.
    return WITHHELD_NO_PHYSICS_VALIDATION


# ---------------------------------------------------------------------------
# Internal: look up one capability
# ---------------------------------------------------------------------------

def _capability_for(function: str, lattice: str) -> AcousticsCapability:
    func_map = _REGISTRY.get(function, {})
    entry = func_map.get(lattice)

    if entry is None:
        return AcousticsCapability(
            function=function,
            lattice=lattice,
            implementation_status=NO_IMPLEMENTATION,
            verification_level=VERIFICATION_NO_IMPLEMENTATION,
            status=WITHHELD_NO_IMPLEMENTATION,
            entrypoint=None,
            test_evidence=None,
            hot_path_note=None,
            note=f"No implementation found for {function}/{lattice}.",
        )

    impl_status, verif_level, entrypoint, test_ev, hot_note, note = entry
    return AcousticsCapability(
        function=function,
        lattice=lattice,
        implementation_status=impl_status,
        verification_level=verif_level,
        status=_status_for(verif_level),
        entrypoint=entrypoint,
        test_evidence=test_ev,
        hot_path_note=hot_note,
        note=note,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def acoustics_capability_matrix() -> dict[str, dict[str, AcousticsCapability]]:
    """Return the complete audited acoustics function/lattice capability matrix.

    Every entry is fail-closed: no combination has physics validation evidence.
    Contract tests (shape/finiteness/causality) verify operator well-formedness
    only, not acoustic accuracy.

    The entire acoustics module is post-processing: it does not enter the
    timestep hot path and can be composed with any collision / turbulence /
    boundary configuration.
    """
    return {
        function: {
            lattice: _capability_for(function, lattice)
            for lattice in _AUDITED_LATTICES
        }
        for function in _AUDITED_FUNCTIONS
    }


def require_acoustics_capability(
    function: AcousticsFunction,
    lattice: LatticeName,
) -> AcousticsCapability:
    """Return only a physics-validated capability; otherwise fail closed.

    No current acoustics function combination has physics validation evidence.
    This function always raises :class:`AcousticsWithheldError`.

    Request identities are validated before matrix lookup, so an unsupported
    public input is always rejected with a stable, machine-readable withholding
    code rather than leaking a ``KeyError``.
    """
    if function not in _AUDITED_FUNCTIONS:
        raise AcousticsWithheldError(
            f"{WITHHELD_UNKNOWN_FUNCTION}: {function!r} is not an audited acoustics function."
        )
    if lattice not in _AUDITED_LATTICES:
        raise AcousticsWithheldError(
            f"{WITHHELD_UNKNOWN_LATTICE}: {lattice!r} is not an audited lattice."
        )
    capability = acoustics_capability_matrix()[function][lattice]
    if not capability.available:
        raise AcousticsWithheldError(f"{capability.status}: {capability.note}")
    return capability


def acoustics_post_processing_audit() -> tuple[PostProcessingAudit, ...]:
    """Return observations documenting that all acoustics functions are post-processing.

    This is a cold-path audit of the module's architectural role; it does not
    modify any numerical kernel.  Every implemented acoustics function operates
    on saved fields and does not enter the timestep hot path.
    """
    return _POST_PROCESSING_AUDIT


__all__ = [
    "AcousticsCapability",
    "AcousticsWithheldError",
    "PostProcessingAudit",
    "WITHHELD_NO_PHYSICS_VALIDATION",
    "WITHHELD_NO_IMPLEMENTATION",
    "WITHHELD_UNKNOWN_FUNCTION",
    "WITHHELD_UNKNOWN_LATTICE",
    "VERIFICATION_CONTRACT_TESTED",
    "VERIFICATION_NO_IMPLEMENTATION",
    "IMPLEMENTED",
    "NO_IMPLEMENTATION",
    "acoustics_capability_matrix",
    "require_acoustics_capability",
    "acoustics_post_processing_audit",
]
