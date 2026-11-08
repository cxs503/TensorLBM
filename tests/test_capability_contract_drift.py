"""Gate for the capability contracts' own evidence surface.

The contracts in ``tensorlbm/*_capability_contract.py`` are the machine-readable
evidence surface of this package: each cell names the callable that implements
the mechanism and the test file that backs its verification level.  Both are
plain strings that nothing resolved against the repository, so a rename could
leave a contract certifying a symbol or a test file that no longer exists.

``tensorlbm.capability_contract_drift`` resolves them; these tests keep that
auditor honest in three ways:

1. **Repository state** — no drift today, in every audited contract, and every
   combination stays fail-closed (no ``PHYSICS_VALIDATED``-style status).
2. **Detection power (mutation)** — each check must *fire* on an injected
   defect built from a real claim.  An auditor that returns "ok" because it
   never looks at anything would pass check 1 alone; these tests make that
   impossible.  The stale-evidence case is the historical one actually found in
   this repository (the sliding-mesh contract cited ``test_sliding_mesh_common
   .py``, which does not exist), so it doubles as the regression pin.
3. **Fail-closed public API** — the ``require_*`` entry points still refuse
   unknown *and* known-but-unvalidated combinations.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json

import pytest

from tensorlbm import (
    fsi_capability_contract,
    ibm_capability_contract,
    sixdof_capability_contract,
    sliding_mesh_capability_contract,
)
from tensorlbm.capability_contract_drift import (
    CONTRACT_POLICIES,
    CONTRACTS,
    Claim,
    ContractPolicy,
    audit_claims,
    audit_contracts,
    collect_claims,
    main,
    resolve_dotted,
    resolve_test_evidence,
)
from tensorlbm.capability_contract_drift import tests_root as _tests_root

EXPECTED_CONTRACTS = {
    "ibm",
    "sliding_mesh",
    "fsi",
    "sixdof",
    "acoustics",
    "physics",
    "amr",
    "advanced_collision",
}

#: Cell counts at the time of writing (2026-11-08).  Kept per contract rather
#: than as one total so that a contract silently dropping out is visible.
EXPECTED_CELLS = {
    "ibm": 4,
    "sliding_mesh": 6,
    "fsi": 4,
    "sixdof": 2,
    "acoustics": 20,
    "physics": 45,
    "amr": 75,
    "advanced_collision": 14,
}


@pytest.fixture(scope="module")
def claims() -> list[Claim]:
    return collect_claims()


@pytest.fixture(scope="module")
def report():
    return audit_contracts()


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


# --------------------------------------------------------------------------- #
# 1. Repository state
# --------------------------------------------------------------------------- #


def test_every_audited_contract_contributes_claims(claims: list[Claim]) -> None:
    """No contract may be silently skipped (a skipped contract is unchecked)."""
    assert {claim.contract for claim in claims} == EXPECTED_CONTRACTS
    counts = {name: 0 for name in EXPECTED_CONTRACTS}
    for claim in claims:
        counts[claim.contract] += 1
    assert counts == EXPECTED_CELLS


def test_repository_capability_contracts_show_no_drift(report) -> None:
    assert report.tests_root is not None, "tests/ directory not discoverable"
    assert report.ok, "capability-contract drift:\n" + "\n".join(
        f"  [{i.severity}] {i.code} {i.contract}[{'/'.join(i.path)}]: {i.detail}"
        for i in report.issues
    )


def test_all_reported_statuses_are_withheld() -> None:
    """Fail-closed invariant, minus the contracts that opt out in writing.

    ``advanced_collision`` is an availability matrix (``AVAILABLE`` statuses,
    no evidence claim); its exemption must be *declared* in
    ``CONTRACT_POLICIES`` with a reason, never assumed.
    """
    claims = collect_claims()
    assert claims
    exempt = {
        name
        for name, policy in CONTRACT_POLICIES.items()
        if not policy.fail_closed
    }
    for name in exempt:
        assert CONTRACT_POLICIES[name].reason, f"{name} exempt without a reason"
    for claim in claims:
        assert claim.status, f"{claim.label} has no status"
        if claim.contract in exempt:
            continue
        assert claim.status.startswith("WITHHELD"), (
            f"{claim.label} reports non-withheld status {claim.status!r}"
        )


def test_every_claim_entrypoint_resolves() -> None:
    from types import ModuleType

    for claim in collect_claims():
        for dotted in _dotted_paths(claim.entrypoint):
            target = resolve_dotted(dotted)
            assert callable(target) or isinstance(target, ModuleType), (
                claim.label,
                dotted,
            )


def _dotted_paths(text: str | None) -> list[str]:
    from tensorlbm.capability_contract_drift import _DOTTED_RE

    return _DOTTED_RE.findall(text or "")


# --------------------------------------------------------------------------- #
# 2. Detection power — the auditor must fire on injected defects
# --------------------------------------------------------------------------- #


def _real_claim(contract: str, path: tuple[str, ...]) -> Claim:
    for claim in collect_claims():
        if claim.contract == contract and claim.path == path:
            return claim
    raise AssertionError(f"no such claim {contract}{path}")


def test_detects_stale_test_evidence_file(monkeypatch) -> None:
    """Historic defect: evidence names a test file that does not exist."""
    claim = _real_claim("sliding_mesh", ("sliding_mesh_3d", "D3Q19"))
    stale = dataclasses.replace(
        claim,
        test_evidence="test_sliding_mesh_common.py: shape, finite",
    )
    local = audit_claims([stale], root=_tests_root())
    assert "EVIDENCE_UNRESOLVED" in _codes(local.errors)
    assert "test_sliding_mesh_common.py" in local.errors[0].detail


def test_detects_missing_entrypoint_symbol(monkeypatch) -> None:
    claim = _real_claim("fsi", ("D3Q19", "one_way_explicit"))
    broken = dataclasses.replace(
        claim, entrypoint="tensorlbm.fsi_common.fsi_step_removed_in_refactor"
    )
    local = audit_claims([broken], root=_tests_root())
    assert "ENTRYPOINT_UNRESOLVED" in _codes(local.errors)


def test_detects_entrypoint_that_is_not_callable() -> None:
    claim = _real_claim("ibm", ("D3Q19", "hat"))
    broken = dataclasses.replace(claim, entrypoint="tensorlbm.ibm_common.DEFAULT_TAU")
    local = audit_claims([broken], root=_tests_root())
    codes = _codes(local.errors)
    assert "ENTRYPOINT_NOT_CALLABLE" in codes or "ENTRYPOINT_UNRESOLVED" in codes


def test_detects_non_fail_closed_status() -> None:
    claim = _real_claim("fsi", ("D3Q19", "one_way_explicit"))
    leaked = dataclasses.replace(claim, status="AVAILABLE")
    local = audit_claims([leaked], root=_tests_root())
    assert "NOT_FAIL_CLOSED" in _codes(local.errors)


def test_detects_evidence_on_implemented_only_entry() -> None:
    claim = _real_claim("sixdof", ("cummins",))
    assert claim.verification_level == "IMPLEMENTED_ONLY"
    mismatched = dataclasses.replace(
        claim, test_evidence="test_sixdof_common.py: shape"
    )
    local = audit_claims([mismatched], root=_tests_root())
    assert "EVIDENCE_LEVEL_MISMATCH" in _codes(local.errors)


def test_detects_tested_level_without_evidence() -> None:
    claim = _real_claim("ibm", ("D3Q27", "4pt"))
    bare = dataclasses.replace(claim, test_evidence=None)
    local = audit_claims([bare], root=_tests_root())
    assert "EVIDENCE_MISSING" in _codes(local.errors)


def test_detects_contract_matrix_that_yields_nothing(monkeypatch) -> None:
    """A contract whose records stop being recognised must fail, not skip."""
    import sys
    import types

    probe = types.ModuleType("tensorlbm._empty_contract_probe")
    # A matrix whose leaves carry no capability attributes at all.
    probe.empty_matrix = lambda: {  # noqa: E731 - trivial probe factory
        "lattice": {"mode": {"note": "no capability record here"}}
    }
    monkeypatch.setitem(sys.modules, "tensorlbm._empty_contract_probe", probe)
    ghost_report = audit_contracts(
        {"ghost": ("tensorlbm._empty_contract_probe", "empty_matrix")}
    )
    assert "CONTRACT_EMPTY" in _codes(ghost_report.errors)
    assert ghost_report.claims == []


def test_evidence_resolution_is_relative_to_tests_root(tmp_path) -> None:
    refs, missing = resolve_test_evidence("test_ibm_common.py: shape", tmp_path)
    assert refs == []
    assert missing == ["test_ibm_common.py"]
    refs, missing = resolve_test_evidence(
        "test_ibm_common.py: shape", _tests_root()
    )
    assert missing == []
    assert refs and refs[0].file.is_file()


# --------------------------------------------------------------------------- #
# 3. Public API stays fail-closed
# --------------------------------------------------------------------------- #


def test_require_functions_reject_unknown_and_validated_requests() -> None:
    with pytest.raises(ibm_capability_contract.IBMWithheldError):
        ibm_capability_contract.require_ibm_capability("D3Q99", "hat")  # type: ignore[arg-type]
    with pytest.raises(ibm_capability_contract.IBMWithheldError):
        ibm_capability_contract.require_ibm_capability("D3Q19", "hat")

    with pytest.raises(fsi_capability_contract.FSIWithheldError):
        fsi_capability_contract.require_fsi_capability("D3Q99", "one_way_explicit")  # type: ignore[arg-type]
    with pytest.raises(fsi_capability_contract.FSIWithheldError):
        fsi_capability_contract.require_fsi_capability("D3Q19", "one_way_explicit")

    with pytest.raises(sixdof_capability_contract.SixDOFWithheldError):
        sixdof_capability_contract.require_sixdof_capability("runge_kutta_4")  # type: ignore[arg-type]
    with pytest.raises(sixdof_capability_contract.SixDOFWithheldError):
        sixdof_capability_contract.require_sixdof_capability("symplectic_euler")

    kwargs = dict(
        kind="sliding_mesh_3d",
        lattice="D3Q19",
        collision="bgk",
        physics="single_phase",
        backend="torch_cpu",
    )
    with pytest.raises(sliding_mesh_capability_contract.SlidingMeshWithheldError):
        sliding_mesh_capability_contract.require_sliding_mesh_capability(
            **{**kwargs, "kind": "sliding_mesh_9d"}  # type: ignore[arg-type]
        )
    with pytest.raises(sliding_mesh_capability_contract.SlidingMeshWithheldError):
        sliding_mesh_capability_contract.require_sliding_mesh_capability(**kwargs)


# --------------------------------------------------------------------------- #
# 4. CLI contract (CI reuse)
# --------------------------------------------------------------------------- #


def test_cli_emits_json_and_exit_code(capsys) -> None:
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["claims"] == len(collect_claims())
    assert set(payload["contracts"]) == EXPECTED_CONTRACTS
    assert payload["errors"] == []


def test_cli_contract_table_covers_all_audited_modules() -> None:
    for name, (module, factory) in CONTRACTS.items():
        assert module.startswith("tensorlbm."), name
        assert factory.endswith("_matrix"), name

# --------------------------------------------------------------------------- #
# 5. Coverage of the 2026-11-08 extension (sequence matrices, tiers, policy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Record:
    """Synthetic capability record (never a domain record — gate test only)."""

    entrypoint: str | None = None
    verification_level: str | None = None
    status: str | None = "WITHHELD_X"
    test_evidence: str | None = None


def test_sequence_valued_matrix_is_traversed() -> None:
    """A list of records is a matrix too (``physics_capability_matrix``).

    Before 2026-11-08 only mappings were traversed, so a list-returning
    contract contributed zero claims and went unaudited.
    """
    from tensorlbm.capability_contract_drift import _iter_capabilities

    found = list(_iter_capabilities([_Record(), _Record()]))
    assert [path for path, _ in found] == [("0",), ("1",)]


def test_entrypoint_tier_does_not_demand_evidence() -> None:
    """A contract with no evidence field must not be failed for having none."""
    claim = Claim("synthetic", ("a",), "tensorlbm.ibm_common", None, None, "WITHHELD_X",
                  evidence_field=False)
    local = audit_claims([claim], root=_tests_root())
    assert local.ok, [i.code for i in local.errors]
    assert claim.tier == "entrypoint"


def test_entrypoint_tier_still_requires_a_resolvable_target() -> None:
    claim = Claim("synthetic", ("a",), "tensorlbm.ibm_common.gone_forever", None, None,
                  "WITHHELD_X", evidence_field=False)
    local = audit_claims([claim], root=_tests_root())
    assert "ENTRYPOINT_UNRESOLVED" in _codes(local.errors)


def test_module_entrypoint_is_accepted() -> None:
    """``amr`` points at a module, not at one callable — that is legitimate."""
    claim = Claim("synthetic", ("a",), "tensorlbm.amr_common", None, None, "WITHHELD_X",
                  evidence_field=False)
    assert audit_claims([claim], root=_tests_root()).ok


def test_analytic_verified_needs_evidence() -> None:
    """``ANALYTIC_VERIFIED`` is accepted *because* it still cites a test file."""
    without = Claim("synthetic", ("a",), "tensorlbm.ibm_common",
                    None, "ANALYTIC_VERIFIED", "WITHHELD_X")
    assert "EVIDENCE_MISSING" in _codes(
        audit_claims([without], root=_tests_root()).errors
    )
    with_evidence = dataclasses.replace(
        without, test_evidence="test_ibm_common.py: analytic benchmark"
    )
    assert audit_claims([with_evidence], root=_tests_root()).ok


def test_declared_policy_exemption_is_reported() -> None:
    """An exemption from the fail-closed invariant must be visible in the report."""
    claim = Claim(
        "synthetic", ("a",), "tensorlbm.ibm_common", None, None, "AVAILABLE",
        evidence_field=False,
    )
    strict = audit_claims([claim], root=_tests_root())
    assert "NOT_FAIL_CLOSED" in _codes(strict.errors)
    lenient = audit_claims(
        [claim],
        root=_tests_root(),
        policies={"synthetic": ContractPolicy(fail_closed=False, reason="availability")},
    )
    assert lenient.ok


def test_cli_reports_contract_policies(capsys) -> None:
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["policies"]["advanced_collision"]["fail_closed"] is False
    assert payload["policies"]["advanced_collision"]["reason"]
