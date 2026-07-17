"""Fail-closed contract tests for audited local-refinement paths."""
from __future__ import annotations

import pytest

from tensorlbm.amr_capability_contract import (
    LocalRefinementWithheldError,
    REQUIRED_FRONTEND_METADATA,
    WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT,
    WITHHELD_NO_D3Q27_LOCAL_REFINEMENT,
    WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE,
    WITHHELD_REQUIRED_METADATA_NOT_EMITTED,
    WITHHELD_UNKNOWN_LATTICE,
    WITHHELD_UNKNOWN_PATH,
    WITHHELD_UNKNOWN_PHYSICS,
    local_refinement_capability_matrix,
    require_local_refinement_capability,
)


def test_matrix_distinguishes_mechanics_from_frontend_ready_physics() -> None:
    matrix = local_refinement_capability_matrix()

    assert matrix["adaptive_dynamic"]["D2Q9"]["single_phase"].mechanics_status == "AVAILABLE_MECHANICS_ONLY"
    exchange_scheme = matrix["adaptive_dynamic"]["D3Q19"]["single_phase"].exchange_scheme
    assert exchange_scheme is not None
    assert exchange_scheme.startswith("FH helper (specific adaptive path")
    multigrid_exchange = matrix["multigrid_static"]["D3Q19"]["single_phase"].exchange_scheme
    assert multigrid_exchange is not None
    assert multigrid_exchange.startswith("plain trilinear interpolation/block-average")
    assert matrix["surface_shell"]["D3Q19"]["single_phase"].mechanics_status == "AVAILABLE_MECHANICS_ONLY"

    for physics in ("turbulence", "multiphase", "ibm", "curved_wall"):
        assert matrix["adaptive_dynamic"]["D3Q19"][physics].status.startswith("WITHHELD_")

    assert matrix["adaptive_dynamic"]["D3Q27"]["single_phase"].status == "WITHHELD_NO_D3Q27_LOCAL_REFINEMENT"
    assert all(
        not capability.available
        for lattices in matrix.values()
        for physicses in lattices.values()
        for capability in physicses.values()
    )


def test_required_frontend_metadata_is_explicit_and_current_paths_fail_closed() -> None:
    assert REQUIRED_FRONTEND_METADATA == (
        "subcycling",
        "ratio",
        "exchange_scheme",
        "geometry_remesh_provenance",
        "flux_inventory_ledger",
        "refinement_decision_evidence",
    )

    with pytest.raises(LocalRefinementWithheldError, match="WITHHELD_REQUIRED_METADATA_NOT_EMITTED"):
        require_local_refinement_capability(
            "adaptive_dynamic",
            "D3Q19",
            "single_phase",
            metadata={key: object() for key in REQUIRED_FRONTEND_METADATA},
        )


def test_contract_rejects_unknown_or_under_evidenced_combinations() -> None:
    with pytest.raises(LocalRefinementWithheldError, match="WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT"):
        require_local_refinement_capability("adaptive_dynamic", "D3Q19", "multiphase")

    with pytest.raises(ValueError, match="metadata missing required keys"):
        require_local_refinement_capability(
            "adaptive_dynamic", "D3Q19", "single_phase", metadata={"ratio": 2}
        )


@pytest.mark.parametrize(
    ("path", "lattice", "physics", "withheld_code"),
    (
        ("unknown_path", "D3Q19", "single_phase", WITHHELD_UNKNOWN_PATH),
        ("adaptive_dynamic", "D9Q99", "single_phase", WITHHELD_UNKNOWN_LATTICE),
        ("adaptive_dynamic", "D3Q19", "unknown_physics", WITHHELD_UNKNOWN_PHYSICS),
    ),
)
def test_public_contract_fail_closes_unknown_inputs_before_matrix_lookup(
    path: str, lattice: str, physics: str, withheld_code: str,
) -> None:
    with pytest.raises(LocalRefinementWithheldError, match=withheld_code):
        require_local_refinement_capability(
            path, lattice, physics, metadata={"ratio": 2},  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Common module path (amr_common)
# ---------------------------------------------------------------------------

def test_common_module_path_has_d3q19_and_d3q27_mechanics() -> None:
    """The common_module path provides AVAILABLE_MECHANICS_ONLY for D3Q19 and D3Q27."""
    matrix = local_refinement_capability_matrix()
    for lattice in ("D3Q19", "D3Q27"):
        cap = matrix["common_module"][lattice]["single_phase"]
        assert cap.mechanics_status == "AVAILABLE_MECHANICS_ONLY"
        assert cap.entrypoint == "tensorlbm.amr_common"
        assert cap.exchange_scheme is not None
        assert "FH second-order" in cap.exchange_scheme
        assert "solver-agnostic" in cap.exchange_scheme
        # Frontend status is still withheld (no metadata emitted)
        assert cap.status == WITHHELD_REQUIRED_METADATA_NOT_EMITTED
        assert not cap.available


def test_common_module_path_withholds_non_single_phase_physics() -> None:
    matrix = local_refinement_capability_matrix()
    for physics in ("turbulence", "multiphase", "ibm", "curved_wall"):
        cap = matrix["common_module"]["D3Q19"][physics]
        assert cap.mechanics_status == "AVAILABLE_MECHANICS_ONLY"
        assert cap.status == WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT


def test_common_module_path_rejects_d2q9() -> None:
    matrix = local_refinement_capability_matrix()
    cap = matrix["common_module"]["D2Q9"]["single_phase"]
    assert cap.mechanics_status == "NO_IMPLEMENTATION"
    assert cap.status == WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE


def test_common_module_path_withholds_without_complete_metadata() -> None:
    with pytest.raises(LocalRefinementWithheldError, match="WITHHELD_REQUIRED_METADATA_NOT_EMITTED"):
        require_local_refinement_capability(
            "common_module",
            "D3Q27",
            "single_phase",
            metadata={key: object() for key in REQUIRED_FRONTEND_METADATA},
        )


def test_legacy_paths_still_withhold_d3q27() -> None:
    """Non-common-module paths still withhold D3Q27."""
    matrix = local_refinement_capability_matrix()
    for path in ("adaptive_dynamic", "multigrid_static", "surface_shell", "multipatch_static"):
        cap = matrix[path]["D3Q27"]["single_phase"]
        assert cap.status == WITHHELD_NO_D3Q27_LOCAL_REFINEMENT
        assert "common_module" in cap.note
