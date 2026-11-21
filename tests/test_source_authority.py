from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_source_authority", ROOT / "tools/check_source_authority.py"
)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_repository_matches_source_authority_policy() -> None:
    policy = CHECKER.load_policy()
    assert CHECKER.validate(policy) == []


def test_checker_does_not_register_its_own_policy_literal() -> None:
    policy = CHECKER.load_policy()
    assert "tools/check_source_authority.py" not in CHECKER.collect_active_references(policy)


def test_new_active_nested_consumer_is_rejected() -> None:
    policy = CHECKER.load_policy()
    policy = copy.deepcopy(policy)
    policy["allowed_active_references"].pop(
        next(iter(policy["allowed_active_references"]))
    )
    assert any("new active nested-source reference" in item for item in CHECKER.validate(policy))


def test_reference_line_drift_is_rejected() -> None:
    policy = CHECKER.load_policy()
    policy = copy.deepcopy(policy)
    first = next(iter(policy["allowed_active_references"].values()))
    first["matching_lines_sha256"] = "0" * 64
    assert any("reference lines changed" in item for item in CHECKER.validate(policy))
