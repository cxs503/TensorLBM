"""R1 contracts remain solver-independent and fail closed on link ownership."""
from __future__ import annotations

import pytest


def test_suboff_case_definition_is_frozen_hashable_and_withholds_source_by_default() -> None:
    from tensorlbm.suboff_case_definition import SuboffCaseDefinition

    case = SuboffCaseDefinition(configuration="with_sail")

    assert case.configuration == "with_sail"
    assert case.reference.source_status == "withheld"
    assert case.reference.sha256
    assert case.reference.units["length"] == "m"
    with pytest.raises(TypeError):
        case.reference.units["length"] = "ft"  # type: ignore[index]
    assert hash(case.reference)
    assert hash(case)
    with pytest.raises((AttributeError, TypeError)):
        case.configuration = "full"  # type: ignore[misc]


@pytest.mark.parametrize("configuration", ["bare_hull", "with_sail", "full"])
def test_case_accepts_all_suboff_configurations(configuration: str) -> None:
    from tensorlbm.suboff_case_definition import SuboffCaseDefinition

    assert SuboffCaseDefinition(configuration=configuration).configuration == configuration


def test_common_case_contract_can_label_ch_and_korner_hulls_without_external_reference() -> None:
    from tensorlbm.suboff_case_definition import SuboffCaseDefinition

    assert SuboffCaseDefinition(application="ch_hull").reference.source_status == "withheld"
    assert SuboffCaseDefinition(application="korner_hull").reference.source_status == "withheld"


def test_missing_link_ownership_is_diagnostic_only_and_never_constructs_validated_ct() -> None:
    from tensorlbm.marine_resistance_contract import build_resistance_force_contract

    contract = build_resistance_force_contract(
        reference_area=2.0,
        length=4.0,
        rho=1000.0,
        U=3.0,
        direction=(1.0, 0.0, 0.0),
        method="momentum_exchange",
        sample_phase="post_boundary",
        link_ownership=None,
        force=(10.0, 0.0, 0.0),
    )

    assert contract.status == "diagnostic_only"
    assert contract.Ct is None
    assert contract.validated is False
    assert any(item.startswith("link_ownership") for item in contract.diagnostics)


def test_complete_owned_force_builds_measured_candidate_not_physical_validation() -> None:
    from tensorlbm.marine_resistance_contract import build_resistance_force_contract

    contract = build_resistance_force_contract(
        reference_area=2.0,
        length=4.0,
        rho=1000.0,
        U=3.0,
        direction=(1.0, 0.0, 0.0),
        method="linkwise_momentum_exchange",
        sample_phase="post_boundary",
        link_ownership={"status": "complete", "owner": "solid_surface", "owned_links": 12},
        force=(10.0, 0.0, 0.0),
    )

    assert contract.status == "measured_candidate"
    assert contract.Ct == pytest.approx(10.0 / (0.5 * 1000.0 * 3.0**2 * 2.0))
    assert contract.validated is False
    assert "physical_validation: withheld" in contract.diagnostics


def test_null_inputs_remain_diagnostic_only_without_throwing() -> None:
    from tensorlbm.marine_resistance_contract import build_resistance_force_contract

    contract = build_resistance_force_contract(
        reference_area=None, length=None, rho=None, U=None, direction=None,
        method=None, sample_phase=None, link_ownership=None, force=None,
    )

    assert contract.status == "diagnostic_only"
    assert contract.Ct is None
    assert contract.force is None
