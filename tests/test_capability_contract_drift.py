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
import json

import pytest

from tensorlbm import (
    fsi_capability_contract,
    ibm_capability_contract,
    sixdof_capability_contract,
    sliding_mesh_capability_contract,
)
from tensorlbm.capability_contract_drift import (
    CONTRACTS,
    Claim,
    audit_claims,
    audit_contracts,
    collect_claims,
    main,
    resolve_dotted,
    resolve_test_evidence,
)
from tensorlbm.capability_contract_drift import tests_root as _tests_root

EXPECTED_CONTRACTS = {"ibm", "sliding_mesh", "fsi", "sixdof"}


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
    # 4 IBM + 6 sliding-mesh + 4 FSI + 2 6-DOF cells at the time of writing.
    assert len(claims) == 16


def test_repository_capability_contracts_show_no_drift(report) -> None:
    assert report.tests_root is not None, "tests/ directory not discoverable"
    assert report.ok, "capability-contract drift:\n" + "\n".join(
        f"  [{i.severity}] {i.code} {i.contract}[{'/'.join(i.path)}]: {i.detail}"
        for i in report.issues
    )


def test_all_reported_statuses_are_withheld() -> None:
    """Fail-closed invariant: no contract cell claims availability/validation."""
    claims = collect_claims()
    assert claims
    for claim in claims:
        assert claim.status, f"{claim.label} has no status"
        assert claim.status.startswith("WITHHELD"), (
            f"{claim.label} reports non-withheld status {claim.status!r}"
        )


def test_every_claim_entrypoint_resolves() -> None:
    for claim in collect_claims():
        for dotted in _dotted_paths(claim.entrypoint):
            assert callable(resolve_dotted(dotted)), (claim.label, dotted)


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