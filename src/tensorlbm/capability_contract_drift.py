"""Repository-versus-contract drift audit for the fail-closed capability contracts.

The capability contracts (``ibm_capability_contract``,
``sliding_mesh_capability_contract``, ``fsi_capability_contract``,
``sixdof_capability_contract``, and siblings) are the machine-readable evidence
surface of this package: every entry names

* an ``entrypoint`` — the dotted path of the callable that implements the
  mechanism, and
* ``test_evidence`` — the test file (and the mechanics it asserts) that backs
  the ``verification_level``.

Both are *strings*.  Nothing in the package resolved them against the
repository, so a rename of a solver function or of a test module left the
contract asserting a capability, and an evidence level, that no longer exist.
That failure mode is worse than having no contract at all: ``WITHHELD_*`` codes
make the contract look conservative and audited while the pointer underneath it
has rotted (a concrete instance was found in this repository: the sliding-mesh
contract cited ``test_sliding_mesh_common.py``, a file that does not exist —
the tests live in ``test_sliding_mesh_regression.py``).

This module is the mechanical gate that closes that hole.  It walks the public
capability matrices, resolves every claimed entrypoint and every claimed test
file, and reports:

Scope (2026-11-08 extension): eight contracts are audited, in two tiers.

* *evidence tier* — records carry an ``entrypoint`` **and** an evidence field
  and a verification level (``ibm``, ``sliding_mesh``, ``fsi``, ``sixdof``,
  ``acoustics``, ``physics``): entrypoint resolution, fail-closed status,
  level vocabulary and cited-test resolution are all enforced;
* *entrypoint tier* — records carry an ``entrypoint`` but no evidence field at
  all (``amr``, ``advanced_collision``): only entrypoint resolution is
  enforced, because such a contract makes no evidence claim to audit.  Silently
  demanding evidence of it, or silently skipping it, would both be wrong.


``error`` (drift — the contract claims something the repository cannot back)
    * ``NOT_FAIL_CLOSED`` — status is not a ``WITHHELD_*`` code;
    * ``ENTRYPOINT_UNRESOLVED`` / ``ENTRYPOINT_NOT_CALLABLE`` — the dotted path
      does not resolve, or resolves to a non-callable;
    * ``ENTRYPOINT_UNPARSEABLE`` — an implemented entry carries no resolvable
      dotted path at all;
    * ``ENTRYPOINT_FOR_UNIMPLEMENTED`` — ``NO_IMPLEMENTATION`` entry that still
      names an entrypoint;
    * ``EVIDENCE_MISSING`` / ``EVIDENCE_UNRESOLVED`` / ``EVIDENCE_UNPARSEABLE``
      / ``EVIDENCE_EMPTY`` / ``EVIDENCE_NODE_MISSING`` — a tested entry whose
      evidence text names no test file, a test file that is absent, a file with
      no ``test_*`` functions, or a named test function that is gone;
    * ``EVIDENCE_LEVEL_MISMATCH`` — ``IMPLEMENTED_ONLY`` (or other untested
      level) entry that nonetheless carries evidence, which would let the
      "no contract tests" withholding code masquerade as tested;
    * ``UNKNOWN_VERIFICATION_LEVEL`` — level outside the audited vocabulary.

``warning`` (honest disclosure, not a hard failure)
    * ``ENTRYPOINT_SEGMENT_UNRESOLVED`` — a prose segment of a multi-entry
      ``entrypoint`` string could not be resolved against the claim's own
      module (the strings are free text, so this is reported, not enforced);
    * ``LATTICE_NOT_IN_EVIDENCE`` — the claim is keyed by a lattice name that
      never appears in the cited evidence file (the evidence may not exercise
      the dimension the contract certifies).

The ``evidence tier`` also accepts ``ANALYTIC_VERIFIED`` — "the mechanism
reproduces a closed-form solution" is strictly stronger than "a mechanics test
exists" — but only because it still has to name a resolvable test file, so the
level cannot be asserted without machine-checkable backing.

Per-contract policy (``CONTRACT_POLICIES``) records the one convention that is
*not* universal: ``fail_closed``.  ``advanced_collision`` is an availability
matrix whose statuses are ``AVAILABLE``/``WITHHELD_*``; forcing it into the
``WITHHELD_*`` invariant would be a false positive, so its policy declares
``fail_closed=False`` and the report lists that exemption explicitly instead of
hiding it.

Deliberately *not* checked: whether the mechanisms are physically accurate.
Operator-algebra contract tests cannot certify physics; the contracts already
withhold on that axis and this auditor keeps them there — it only enforces that
the withholding is honest about what exists.

Usage::

    python -m tensorlbm.capability_contract_drift          # human readable
    python -m tensorlbm.capability_contract_drift --json   # machine readable
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from types import ModuleType
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "CONTRACT_POLICIES",
    "CONTRACTS",
    "ContractPolicy",
    "Claim",
    "DriftReport",
    "EntrypointNotFound",
    "EvidenceRef",
    "Issue",
    "audit_claims",
    "audit_contracts",
    "collect_claims",
    "main",
    "resolve_dotted",
    "resolve_test_evidence",
    "tests_root",
]

#: ``contract name -> (module, matrix factory)`` for every audited contract.
CONTRACTS: dict[str, tuple[str, str]] = {
    "ibm": ("tensorlbm.ibm_capability_contract", "ibm_capability_matrix"),
    "sliding_mesh": (
        "tensorlbm.sliding_mesh_capability_contract",
        "sliding_mesh_capability_matrix",
    ),
    "fsi": ("tensorlbm.fsi_capability_contract", "fsi_capability_matrix"),
    "sixdof": ("tensorlbm.sixdof_capability_contract", "sixdof_capability_matrix"),
    "acoustics": (
        "tensorlbm.acoustics_capability_contract",
        "acoustics_capability_matrix",
    ),
    "physics": (
        "tensorlbm.physics_capability_contract",
        "physics_capability_matrix",
    ),
    "amr": (
        "tensorlbm.amr_capability_contract",
        "local_refinement_capability_matrix",
    ),
    "advanced_collision": (
        "tensorlbm.advanced_collision_contract",
        "collision_capability_matrix",
    ),
}


@dataclass(frozen=True)
class ContractPolicy:
    """Conventions a contract may opt out of, stated out loud.

    ``fail_closed``
        ``True`` (default): every record's ``status`` must be a ``WITHHELD_*``
        code, i.e. the contract may never assert a capability as available.
        ``False``: the contract is an availability matrix (it legitimately
        reports ``AVAILABLE``) and only its entrypoints are enforced.
    ``reason``
        Free text copied into the report so that an exemption is always
        accompanied by its justification rather than silently assumed.
    """

    fail_closed: bool = True
    reason: str = ""


#: ``contract name -> policy``; a contract absent from this table gets the
#: strict default (fail-closed, evidence tier).
CONTRACT_POLICIES: dict[str, ContractPolicy] = {
    "advanced_collision": ContractPolicy(
        fail_closed=False,
        reason=(
            "availability matrix (status AVAILABLE/WITHHELD_*) that makes no "
            "verification-level or test-evidence claim; only entrypoint "
            "resolution is enforced"
        ),
    ),
}

#: Verification levels that assert "tests exist and pass" (or better: an
#: analytic solution is reproduced, which is *also* backed by a test file).
TESTED_LEVELS = frozenset({"CONTRACT_TESTED", "MECHANICS_TESTED", "ANALYTIC_VERIFIED"})
#: Verification levels that assert "nothing is implemented".
UNIMPLEMENTED_LEVELS = frozenset({"NO_IMPLEMENTATION"})
#: Verification level that asserts "implemented, deliberately untested".
IMPLEMENTED_LEVELS = frozenset({"IMPLEMENTED_ONLY"})
#: Every status a fail-closed contract may report starts with this prefix.
FAIL_CLOSED_PREFIX = "WITHHELD"

_LATTICE_RE = re.compile(r"\bD[123]Q\d{2}\b")
_DOTTED_RE = re.compile(r"tensorlbm(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_TESTFILE_RE = re.compile(r"\b(test_[A-Za-z0-9_]+\.py)(?:::([A-Za-z0-9_]+))?")
_LEADING_IDENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)")


class EntrypointNotFound(LookupError):
    """A dotted entrypoint claimed by a contract does not resolve."""


@dataclass(frozen=True)
class Claim:
    """One cell of a capability matrix, flattened out of its nested mapping."""

    contract: str
    path: tuple[str, ...]
    entrypoint: str | None
    test_evidence: str | None
    verification_level: str | None
    status: str | None
    evidence_field: bool = True
    """Whether the record's schema has an evidence field at all.

    ``False`` marks the *entrypoint tier*: the contract never claims test
    evidence, so ``EVIDENCE_*`` must not fire for it.  ``True`` with an empty
    string is a different thing — a schema that has the field and left it
    blank — and that is still an error for a tested level.
    """

    @property
    def label(self) -> str:
        return f"{self.contract}[{'/'.join(self.path)}]"

    @property
    def tier(self) -> str:
        return "evidence" if self.evidence_field else "entrypoint"


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning"
    code: str
    contract: str
    path: tuple[str, ...]
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "contract": self.contract,
            "path": list(self.path),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EvidenceRef:
    """A test file (optionally a single test node) cited by a contract."""

    name: str
    file: Path
    node: str | None = None


@dataclass
class DriftReport:
    claims: list[Claim] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    tests_root: str | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    #: ``contract -> policy`` for the contracts audited in this report, so that
    #: an exemption from the fail-closed invariant is never implicit.
    policies: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tests_root": self.tests_root,
            "policies": self.policies,
            "claims": len(self.claims),
            "contracts": sorted({c.contract for c in self.claims}),
            "errors": [i.as_dict() for i in self.errors],
            "warnings": [i.as_dict() for i in self.warnings],
        }


# --------------------------------------------------------------------------- #
# Contract -> claims
# --------------------------------------------------------------------------- #


#: Attribute spellings used by the capability records for the same two facts.
_LEVEL_ATTRS = ("verification_level", "implementation_status")
_EVIDENCE_ATTRS = ("test_evidence", "verification_evidence")


def _first_attr(node: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(node, name):
            return getattr(node, name)
    return None


def _iter_capabilities(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield ``(matrix path, capability object)`` pairs from a nested matrix.

    Capability records are duck-typed: the contracts spell the verification
    level and the evidence pointer differently (``verification_level`` /
    ``implementation_status``, ``test_evidence`` / ``verification_evidence``),
    so a record is recognised by carrying ``entrypoint`` plus an evidence field.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from _iter_capabilities(value, path + (str(key),))
        return
    # Some matrices are flat *sequences* of capability records rather than
    # nested mappings (``physics_capability_matrix`` returns a ``list``).
    # Before 2026-11-08 this branch did not exist and every record of such a
    # contract was invisible: ``CONTRACT_EMPTY`` caught the vacuity, but the
    # contract still went unaudited.  Only list/tuple are traversed — sets would
    # make the report ordering non-deterministic.
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _iter_capabilities(value, path + (str(index),))
        return
    if hasattr(node, "entrypoint"):
        yield path, node


def collect_claims(
    contracts: Mapping[str, tuple[str, str]] | None = None,
) -> list[Claim]:
    """Flatten every audited capability matrix into a list of :class:`Claim`."""
    claims: list[Claim] = []
    for name, (module_name, factory) in (contracts or CONTRACTS).items():
        module = importlib.import_module(module_name)
        matrix = getattr(module, factory)()
        for path, capability in _iter_capabilities(matrix):
            claims.append(
                Claim(
                    contract=name,
                    path=path,
                    entrypoint=getattr(capability, "entrypoint", None),
                    test_evidence=_first_attr(capability, _EVIDENCE_ATTRS),
                    verification_level=_first_attr(capability, _LEVEL_ATTRS),
                    status=getattr(capability, "status", None),
                    evidence_field=any(
                        hasattr(capability, n) for n in _EVIDENCE_ATTRS
                    ),
                )
            )
    return claims


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def resolve_dotted(dotted: str) -> Any:
    """Resolve ``tensorlbm.a.b.func`` to the object, importing as needed.

    Raises :class:`EntrypointNotFound` when the module cannot be imported or an
    attribute along the chain is missing.
    """
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise EntrypointNotFound(f"{dotted!r} is empty")
    last_error: Exception | None = None
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:  # try a shorter module prefix
            last_error = exc
            continue
        for attr in parts[split:]:
            if not hasattr(obj, attr):
                raise EntrypointNotFound(
                    f"{dotted}: {type(obj).__name__} has no attribute {attr!r}"
                )
            obj = getattr(obj, attr)
        return obj
    raise EntrypointNotFound(f"{dotted}: no importable module prefix ({last_error})")


def tests_root(start: Path | None = None) -> Path | None:
    """Locate the repository ``tests/`` directory, or ``None`` if absent."""
    override = os.environ.get("TENSORLBM_TESTS_DIR")
    if override:
        candidate = Path(override)
        if candidate.is_dir():
            return candidate.resolve()
    start = (start or Path(__file__)).resolve()
    for parent in start.parents:
        candidate = parent / "tests"
        if candidate.is_dir() and (parent / "src").is_dir():
            return candidate.resolve()
    return None


def resolve_test_evidence(
    text: str | None, root: Path | None
) -> tuple[list[EvidenceRef], list[str]]:
    """Split an evidence string into resolved test files and missing names."""
    if not text:
        return [], []
    refs: list[EvidenceRef] = []
    missing: list[str] = []
    seen: set[tuple[str, str | None]] = set()
    for match in _TESTFILE_RE.finditer(text):
        name, node = match.group(1), match.group(2)
        if (name, node) in seen:
            continue
        seen.add((name, node))
        found: Path | None = None
        if root is not None:
            direct = root / name
            if direct.is_file():
                found = direct
            else:
                hits = sorted(root.rglob(name))
                found = hits[0] if hits else None
        if found is None:
            missing.append(name)
            continue
        refs.append(EvidenceRef(name=name, file=found, node=node))
    return refs, missing


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def _has_attribute(module_name: str, attribute: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - a non-importable context module is a miss
        return False
    return hasattr(module, attribute)


def _count_tests(path: Path) -> int:
    body = path.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"^\s*def test_[A-Za-z0-9_]*\s*\(", body, re.MULTILINE))


def audit_claims(
    claims: Sequence[Claim],
    root: Path | None = None,
    policies: Mapping[str, ContractPolicy] | None = None,
) -> DriftReport:
    """Check every claim against the repository it describes.

    ``policies`` defaults to :data:`CONTRACT_POLICIES`; a contract without an
    entry gets the strict default (fail-closed, evidence-tier expectations).
    """
    resolved_root = root if root is not None else tests_root()
    report = DriftReport(
        claims=list(claims),
        tests_root=str(resolved_root) if resolved_root is not None else None,
    )
    issues: list[Issue] = []
    policy_table = policies if policies is not None else CONTRACT_POLICIES

    def policy_for(claim: Claim) -> ContractPolicy:
        return policy_table.get(claim.contract, ContractPolicy())

    def error(code: str, claim: Claim, detail: str) -> None:
        issues.append(Issue("error", code, claim.contract, claim.path, detail))

    def warn(code: str, claim: Claim, detail: str) -> None:
        issues.append(Issue("warning", code, claim.contract, claim.path, detail))

    for claim in claims:
        level = (claim.verification_level or "").strip()
        status = (claim.status or "").strip()
        known_level = level in (
            TESTED_LEVELS | UNIMPLEMENTED_LEVELS | IMPLEMENTED_LEVELS
        )
        entrypoint_tier = not claim.evidence_field

        if not known_level and not (entrypoint_tier and not level):
            error("UNKNOWN_VERIFICATION_LEVEL", claim, f"level {level!r} is not audited")
        if policy_for(claim).fail_closed and not status.startswith(FAIL_CLOSED_PREFIX):
            error("NOT_FAIL_CLOSED", claim, f"status {status!r} is not a WITHHELD_* code")

        # --- entrypoint -------------------------------------------------- #
        entry_text = claim.entrypoint or ""
        dotted = list(dict.fromkeys(_DOTTED_RE.findall(entry_text)))
        if level in UNIMPLEMENTED_LEVELS:
            if entry_text:
                error(
                    "ENTRYPOINT_FOR_UNIMPLEMENTED",
                    claim,
                    f"{level} entry still names entrypoint {entry_text!r}",
                )
        elif level in (TESTED_LEVELS | IMPLEMENTED_LEVELS) and not dotted:
            error(
                "ENTRYPOINT_UNPARSEABLE",
                claim,
                f"{level} entry names no tensorlbm.* dotted path ({entry_text!r})",
            )
        elif entrypoint_tier and entry_text and not dotted:
            # The entrypoint tier has no level to key off, but a non-empty
            # entrypoint string that names no ``tensorlbm.*`` path is still a
            # pointer nobody can follow.
            error(
                "ENTRYPOINT_UNPARSEABLE",
                claim,
                f"entrypoint names no tensorlbm.* dotted path ({entry_text!r})",
            )

        context_modules: list[str] = []
        for dotted_path in dotted:
            try:
                target = resolve_dotted(dotted_path)
            except EntrypointNotFound as exc:
                error("ENTRYPOINT_UNRESOLVED", claim, str(exc))
                continue
            # A *module* is a legitimate entrypoint: several contracts point at
            # the module that hosts the mechanism (``tensorlbm.amr_common``)
            # rather than at one callable inside it.
            if not callable(target) and not isinstance(target, ModuleType):
                error(
                    "ENTRYPOINT_NOT_CALLABLE",
                    claim,
                    f"{dotted_path} resolves to {type(target).__name__}",
                )
            context_modules.append(".".join(dotted_path.split(".")[:-1]))

        if entry_text and dotted:
            for segment in (s.strip() for s in entry_text.split("/")):
                if not segment or _DOTTED_RE.search(segment):
                    continue
                ident = _LEADING_IDENT_RE.match(segment)
                if ident is None:
                    continue
                name = ident.group(1)
                if not any(_has_attribute(mod, name) for mod in context_modules):
                    warn(
                        "ENTRYPOINT_SEGMENT_UNRESOLVED",
                        claim,
                        f"{segment!r}: {name!r} not found on {context_modules}",
                    )

        # --- test evidence ----------------------------------------------- #
        if entrypoint_tier:
            # The contract's schema has no evidence field: there is nothing to
            # resolve and nothing to demand.  Auditing only the entrypoint and
            # saying so beats inventing an evidence requirement.
            continue
        evidence = claim.test_evidence or ""
        refs, missing = resolve_test_evidence(evidence, resolved_root)
        if level in TESTED_LEVELS:
            if not evidence:
                error("EVIDENCE_MISSING", claim, f"{level} entry carries no evidence")
            elif missing:
                error(
                    "EVIDENCE_UNRESOLVED",
                    claim,
                    f"cited test file(s) not found under {resolved_root}: {sorted(set(missing))}",
                )
            elif not refs:
                error(
                    "EVIDENCE_UNPARSEABLE",
                    claim,
                    f"evidence names no test_*.py file ({evidence!r})",
                )
            for ref in refs:
                if ref.node and not re.search(
                    rf"def {re.escape(ref.node)}\s*\(", ref.file.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                ):
                    error(
                        "EVIDENCE_NODE_MISSING",
                        claim,
                        f"{ref.name}::{ref.node} no longer exists",
                    )
                if _count_tests(ref.file) == 0:
                    error(
                        "EVIDENCE_EMPTY",
                        claim,
                        f"{ref.name} contains no test functions",
                    )
            for lattice in sorted(
                key for key in claim.path if _LATTICE_RE.fullmatch(key)
            ):
                bodies = [
                    ref.file.read_text(encoding="utf-8", errors="ignore") for ref in refs
                ]
                if bodies and not any(lattice in body for body in bodies):
                    warn(
                        "LATTICE_NOT_IN_EVIDENCE",
                        claim,
                        f"claim is keyed by {lattice} but no cited test file mentions it",
                    )
        elif level in IMPLEMENTED_LEVELS:
            if evidence:
                error(
                    "EVIDENCE_LEVEL_MISMATCH",
                    claim,
                    f"{level} entry must not carry evidence ({evidence!r})",
                )
        elif level in UNIMPLEMENTED_LEVELS and _TESTFILE_RE.search(evidence):
            # A prose note ("no implementation for this lattice") is fine here;
            # naming a test file for something unimplemented is not.
            error(
                "EVIDENCE_FOR_UNIMPLEMENTED",
                claim,
                f"{level} entry cites test evidence ({evidence!r})",
            )

    report.issues = issues
    return report


def audit_contracts(
    contracts: Mapping[str, tuple[str, str]] | None = None,
    policies: Mapping[str, ContractPolicy] | None = None,
) -> DriftReport:
    """Collect every contract claim and audit it in one pass.

    A contract that yields *no* claims is reported as ``CONTRACT_EMPTY``: an
    audit that silently skips a contract is a false negative with the same
    consequence as drift (the evidence surface goes unchecked), so vacuity must
    fail loudly.  This check is what caught this auditor's own first blind spot
    (``SlidingMeshCapability`` names its fields ``implementation_status`` /
    ``verification_evidence``, so the duck-typing above had to be widened).
    """
    contracts = contracts or CONTRACTS
    policies = policies if policies is not None else CONTRACT_POLICIES
    claims = collect_claims(contracts)
    report = audit_claims(claims, policies=policies)
    report.policies = {
        name: {"fail_closed": policy.fail_closed, "reason": policy.reason}
        for name, policy in policies.items()
        if name in {claim.contract for claim in claims}
    }
    seen = {claim.contract for claim in claims}
    for name in contracts:
        if name not in seen:
            report.issues.append(
                Issue(
                    "error",
                    "CONTRACT_EMPTY",
                    name,
                    (),
                    "capability matrix yielded no claims — record schema not recognised",
                )
            )
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tensorlbm.capability_contract_drift",
        description="Audit capability-contract entrypoints and test evidence.",
    )
    parser.add_argument("--json", action="store_true", help="machine readable report")
    parser.add_argument(
        "--strict", action="store_true", help="exit nonzero on warnings as well"
    )
    args = parser.parse_args(argv)

    report = audit_contracts()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(
            f"capability contracts: {len(report.claims)} claims from "
            f"{len({c.contract for c in report.claims})} contracts "
            f"(tests root: {report.tests_root})"
        )
        for issue in report.issues:
            print(
                f"  [{issue.severity}] {issue.code} "
                f"{issue.contract}[{'/'.join(issue.path)}]: {issue.detail}"
            )
        print(
            f"  -> {len(report.errors)} error(s), {len(report.warnings)} warning(s); "
            f"{'OK' if report.ok else 'DRIFT'}"
        )
    if report.errors or (args.strict and report.warnings):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())