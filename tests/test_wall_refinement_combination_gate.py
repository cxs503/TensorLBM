"""TDD specification for the fail-closed wall/refinement combination gate."""
from typing import Any
from tensorlbm.wall_refinement_combination_gate import (
    CollisionFamily,
    CombinationEvidence,
    GateStatus,
    GeometryKind,
    GeometryOwnership,
    Lattice,
    PhysicsModel,
    RefinementType,
    WallRefinementCombination,
    WallTreatment,
    WITHHELD_D3Q27_WALL_FUNCTION,
    WITHHELD_REFINEMENT_IBM,
    WITHHELD_REFINEMENT_MULTIPHASE,
    WITHHELD_UNSUPPORTED_COLLISION,
    WITHHELD_WALL_FUNCTION_WITH_REFINEMENT,
    assess_wall_refinement_combination,
)


def baseline(**changes: Any) -> WallRefinementCombination:
    values: dict[str, Any] = dict(
        lattice=Lattice.D3Q19,
        collision=CollisionFamily.MRT,
        wall_treatment=WallTreatment.STANDARD_STATIC,
        refinement=RefinementType.NONE,
        geometry_ownership=GeometryOwnership.SINGLE_LEVEL,
    )
    values.update(changes)
    return WallRefinementCombination(**values)


def test_only_audited_unrefined_single_level_baselines_are_allowed():
    assert assess_wall_refinement_combination(baseline()).status is GateStatus.ALLOWED
    assert assess_wall_refinement_combination(baseline(wall_treatment=WallTreatment.NONE)).status is GateStatus.ALLOWED


def test_static_local_refinement_is_not_mistaken_for_a_validated_wall_combination():
    decision = assess_wall_refinement_combination(
        baseline(refinement=RefinementType.STATIC_LOCAL, geometry_ownership=GeometryOwnership.FINE_LEVEL)
    )
    assert decision.status is GateStatus.WITHHELD


def test_wall_function_amr_is_withheld_with_cross_level_evidence_requirements():
    decision = assess_wall_refinement_combination(
        baseline(
            wall_treatment=WallTreatment.WALL_FUNCTION,
            refinement=RefinementType.DYNAMIC_AMR,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
        )
    )
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_WALL_FUNCTION_WITH_REFINEMENT in decision.reasons
    assert set(decision.missing_required_evidence) == {
        "wall_distance_dy", "y_plus", "level_link_owner", "wall_geometry_owner", "interface_transfer_proof"
    }


def test_providing_evidence_does_not_silently_promote_an_unvalidated_combination():
    decision = assess_wall_refinement_combination(
        baseline(
            wall_treatment=WallTreatment.WALL_FUNCTION,
            refinement=RefinementType.STATIC_LOCAL,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
            evidence=CombinationEvidence(1.0, 50.0, "fine", "fine", "FH proof"),
        )
    )
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_WALL_FUNCTION_WITH_REFINEMENT in decision.reasons
    assert decision.missing_required_evidence == ()


def test_d3q27_wall_function_is_explicitly_withheld():
    decision = assess_wall_refinement_combination(
        baseline(lattice=Lattice.D3Q27, wall_treatment=WallTreatment.WALL_FUNCTION)
    )
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_D3Q27_WALL_FUNCTION in decision.reasons


def test_refinement_with_multiphase_or_ibm_is_explicitly_withheld():
    multiphase = assess_wall_refinement_combination(
        baseline(refinement=RefinementType.DYNAMIC_AMR, physics=PhysicsModel.MULTIPHASE)
    )
    ibm = assess_wall_refinement_combination(
        baseline(refinement=RefinementType.STATIC_LOCAL, geometry_kind=GeometryKind.IBM)
    )
    assert WITHHELD_REFINEMENT_MULTIPHASE in multiphase.reasons
    assert WITHHELD_REFINEMENT_IBM in ibm.reasons


def test_unavailable_collision_contract_cannot_be_promoted_by_baseline_shape():
    decision = assess_wall_refinement_combination(baseline(collision=CollisionFamily.CM))
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_UNSUPPORTED_COLLISION in decision.reasons


# ---------------------------------------------------------------------------
# Common wall-function × refinement combination path
# ---------------------------------------------------------------------------

def test_common_wall_function_without_refinement_is_allowed():
    """Common wall function on a single-level planar grid is a baseline."""
    decision = assess_wall_refinement_combination(
        baseline(wall_treatment=WallTreatment.COMMON_WALL_FUNCTION)
    )
    assert decision.status is GateStatus.ALLOWED


def test_common_wall_function_with_amr_is_withheld_without_evidence():
    """Without cross-level evidence, common_wf + AMR is fail-closed."""
    decision = assess_wall_refinement_combination(
        baseline(
            wall_treatment=WallTreatment.COMMON_WALL_FUNCTION,
            refinement=RefinementType.DYNAMIC_AMR,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
        )
    )
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_WALL_FUNCTION_WITH_REFINEMENT in decision.reasons
    assert set(decision.missing_required_evidence) == {
        "wall_distance_dy", "y_plus", "level_link_owner",
        "wall_geometry_owner", "interface_transfer_proof",
    }


def test_common_wall_function_with_amr_is_allowed_with_complete_evidence():
    """With all cross-level evidence, common_wf + AMR has a clear admission path."""
    decision = assess_wall_refinement_combination(
        baseline(
            wall_treatment=WallTreatment.COMMON_WALL_FUNCTION,
            refinement=RefinementType.DYNAMIC_AMR,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
            evidence=CombinationEvidence(
                wall_distance_dy=0.5,
                y_plus=50.0,
                level_link_owner="fine",
                wall_geometry_owner="fine",
                interface_transfer_proof="FH proof",
            ),
        )
    )
    assert decision.status is GateStatus.ALLOWED
    assert decision.missing_required_evidence == ()


def test_common_wall_function_d3q27_with_amr_is_allowed_with_evidence():
    """D3Q27 common wall function + AMR is admissible (unlike legacy WALL_FUNCTION)."""
    decision = assess_wall_refinement_combination(
        baseline(
            lattice=Lattice.D3Q27,
            wall_treatment=WallTreatment.COMMON_WALL_FUNCTION,
            refinement=RefinementType.DYNAMIC_AMR,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
            evidence=CombinationEvidence(
                wall_distance_dy=0.5,
                y_plus=50.0,
                level_link_owner="fine",
                wall_geometry_owner="fine",
                interface_transfer_proof="FH proof",
            ),
        )
    )
    assert decision.status is GateStatus.ALLOWED


def test_legacy_wall_function_with_amr_still_withheld_even_with_evidence():
    """The legacy WALL_FUNCTION + AMR has no admission path (always withheld)."""
    decision = assess_wall_refinement_combination(
        baseline(
            wall_treatment=WallTreatment.WALL_FUNCTION,
            refinement=RefinementType.DYNAMIC_AMR,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
            evidence=CombinationEvidence(1.0, 50.0, "fine", "fine", "FH proof"),
        )
    )
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_WALL_FUNCTION_WITH_REFINEMENT in decision.reasons


def test_common_wall_function_with_amr_withholds_for_multiphase():
    """Common_wf + AMR + multiphase is withheld regardless of evidence."""
    decision = assess_wall_refinement_combination(
        baseline(
            wall_treatment=WallTreatment.COMMON_WALL_FUNCTION,
            refinement=RefinementType.DYNAMIC_AMR,
            physics=PhysicsModel.MULTIPHASE,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
            evidence=CombinationEvidence(0.5, 50.0, "fine", "fine", "FH proof"),
        )
    )
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_REFINEMENT_MULTIPHASE in decision.reasons
