"""TDD specification for the fail-closed wall/refinement combination gate."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

from tensorlbm.advanced_collision_contract import collision_capability_matrix
from tensorlbm.wall_refinement_combination_gate import (
    WITHHELD_D3Q27_WALL_FUNCTION,
    WITHHELD_NON_BASELINE_REFINEMENT,
    WITHHELD_UNKNOWN_COMBINATION,
    WITHHELD_WALL_FUNCTION_CURVED_WALL,
    WITHHELD_WALL_FUNCTION_IBM,
    WITHHELD_REFINEMENT_IBM,
    WITHHELD_REFINEMENT_MULTIPHASE,
    WITHHELD_UNSUPPORTED_COLLISION,
    WITHHELD_WALL_FUNCTION_WITH_REFINEMENT,
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
    assess_wall_refinement_combination,
)


def _gate_impl_module():
    """The module that *defines* the gate (the alias module is the same object)."""
    from tensorlbm.boundary import wall_refinement_combination_gate

    return wall_refinement_combination_gate


def _capability_matrix_copy():
    """Deep copy so an injected withdrawn row cannot leak between tests."""
    return deepcopy(_gate_impl_module().collision_capability_matrix())


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
    assert (
        assess_wall_refinement_combination(baseline(wall_treatment=WallTreatment.NONE)).status
        is GateStatus.ALLOWED
    )


def test_static_local_refinement_is_not_mistaken_for_a_validated_wall_combination():
    decision = assess_wall_refinement_combination(
        baseline(
            refinement=RefinementType.STATIC_LOCAL, geometry_ownership=GeometryOwnership.FINE_LEVEL
        )
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
        "wall_distance_dy",
        "y_plus",
        "level_link_owner",
        "wall_geometry_owner",
        "interface_transfer_proof",
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


def test_unavailable_collision_contract_cannot_be_promoted_by_baseline_shape(monkeypatch):
    """A family withdrawn from the capability matrix must not pass on shape alone.

    History: this test used to assert that ``CollisionFamily.CM`` is unavailable.
    That expectation drifted: ``collision_capability_matrix()`` now registers all
    seven families (BGK/TRT/RLBM/MRT/CM/CUMULANT/KBC) as AVAILABLE for both
    D3Q19 and D3Q27, so *no* family expressible by this gate's enum can reach the
    fail-closed ``WITHHELD_UNSUPPORTED_COLLISION`` branch.  The guard was
    therefore dead and untestable (it sat in tests/ci_quarantine.txt, so CI never
    even reported it).  The guard is still needed - the matrix can lose a family
    again - so it is exercised here by injecting a matrix with CM withdrawn, which
    is the only way to reach the branch through the public API.
    See docs/AUDIT_WALL_REFINEMENT_GATE_FALSIFIABILITY.md.
    """
    gate_module = _gate_impl_module()
    matrix = _capability_matrix_copy()
    matrix["D3Q19"]["CM"] = replace(
        matrix["D3Q19"]["CM"], available=False, status="WITHHELD_NO_D3Q19_CM_KERNEL"
    )
    monkeypatch.setattr(gate_module, "collision_capability_matrix", lambda: matrix)

    decision = assess_wall_refinement_combination(baseline(collision=CollisionFamily.CM))
    assert decision.status is GateStatus.WITHHELD
    assert WITHHELD_UNSUPPORTED_COLLISION in decision.reasons
    # The withheld matrix row must not leak into the other lattice.
    assert (
        assess_wall_refinement_combination(
            baseline(lattice=Lattice.D3Q27, collision=CollisionFamily.CM)
        ).status
        is GateStatus.ALLOWED
    )


def test_every_withheld_reason_code_is_reachable_by_some_input(monkeypatch):
    """Invariant: no fail-closed reason code may become dead (unreachable) code.

    A guard that no input can trigger cannot fail, so it cannot be evidence.  This
    sweep drives the gate over explicit witness combinations (plus one injected
    withdrawn-capability matrix) and requires that *every* ``WITHHELD_*`` constant
    defined by the module appears in at least one decision.
    """
    gate_module = _gate_impl_module()
    defined = {
        name for name in dir(gate_module) if name.startswith("WITHHELD_") and name.isupper()
    }
    assert len(defined) == 9

    witnesses = {
        WITHHELD_NON_BASELINE_REFINEMENT: baseline(
            refinement=RefinementType.STATIC_LOCAL,
            geometry_kind=GeometryKind.CURVED_STATIC,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
        ),
        WITHHELD_WALL_FUNCTION_WITH_REFINEMENT: baseline(
            wall_treatment=WallTreatment.WALL_FUNCTION,
            refinement=RefinementType.DYNAMIC_AMR,
            geometry_ownership=GeometryOwnership.FINE_LEVEL,
        ),
        WITHHELD_D3Q27_WALL_FUNCTION: baseline(
            lattice=Lattice.D3Q27, wall_treatment=WallTreatment.WALL_FUNCTION
        ),
        WITHHELD_WALL_FUNCTION_CURVED_WALL: baseline(
            wall_treatment=WallTreatment.WALL_FUNCTION,
            geometry_kind=GeometryKind.CURVED_STATIC,
        ),
        WITHHELD_WALL_FUNCTION_IBM: baseline(
            wall_treatment=WallTreatment.WALL_FUNCTION, geometry_kind=GeometryKind.IBM
        ),
        WITHHELD_REFINEMENT_MULTIPHASE: baseline(
            refinement=RefinementType.DYNAMIC_AMR, physics=PhysicsModel.MULTIPHASE
        ),
        WITHHELD_REFINEMENT_IBM: baseline(
            refinement=RefinementType.STATIC_LOCAL, geometry_kind=GeometryKind.IBM
        ),
        WITHHELD_UNKNOWN_COMBINATION: baseline(
            geometry_ownership=GeometryOwnership.COARSE_LEVEL
        ),
        WITHHELD_UNSUPPORTED_COLLISION: baseline(collision=CollisionFamily.CM),
    }
    assert set(witnesses) == defined

    seen = {}
    for reason, combination in witnesses.items():
        decision = assess_wall_refinement_combination(combination)
        seen[reason] = decision.reasons

    # WITHHELD_UNSUPPORTED_COLLISION has no public-input witness today; inject it.
    matrix = _capability_matrix_copy()
    for lattice in ("D3Q19", "D3Q27"):
        matrix[lattice]["CM"] = replace(matrix[lattice]["CM"], available=False)
    monkeypatch.setattr(gate_module, "collision_capability_matrix", lambda: matrix)
    seen[WITHHELD_UNSUPPORTED_COLLISION] = assess_wall_refinement_combination(
        witnesses[WITHHELD_UNSUPPORTED_COLLISION]
    ).reasons

    assert {
        reason for reason, fired in seen.items() if reason in fired
    } == defined, seen


def test_each_admitted_baseline_tracks_the_collision_capability_matrix():
    """The gate and the collision contract must not diverge silently.

    For the plain unrefined single-level planar single-phase shape the gate is
    ALLOWED exactly when the matrix says the collision family is AVAILABLE, for
    every (lattice, family) pair the gate can express.
    """
    matrix = collision_capability_matrix()
    for lattice in Lattice:
        for family in CollisionFamily:
            decision = assess_wall_refinement_combination(
                baseline(lattice=lattice, collision=family)
            )
            available = matrix[lattice.value][family.value].available
            assert decision.allowed is bool(available), (lattice, family, decision)
            assert (WITHHELD_UNSUPPORTED_COLLISION in decision.reasons) is (not available)


def test_gate_family_enum_stays_a_documented_subset_of_the_contract_matrix():
    """Coverage gap, asserted so it cannot widen or narrow unnoticed.

    The gate can express only MRT/CM/KBC.  BGK, TRT, RLBM and CUMULANT are
    AVAILABLE kernels in the matrix but cannot be requested here, i.e. those
    combos are refused with TypeError (fail-closed) rather than by a decision.
    Widening the enum changes the admission surface and must come with a review
    of docs/AUDIT_WALL_REFINEMENT_GATE_FALSIFIABILITY.md.
    """
    matrix = collision_capability_matrix()
    expressible = {family.value for family in CollisionFamily}
    contract_families = set(matrix["D3Q19"]) & set(matrix["D3Q27"])
    assert expressible <= contract_families
    assert contract_families - expressible == {"BGK", "TRT", "RLBM", "CUMULANT"}


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
        "wall_distance_dy",
        "y_plus",
        "level_link_owner",
        "wall_geometry_owner",
        "interface_transfer_proof",
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
