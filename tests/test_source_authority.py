from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_source_authority", ROOT / "tools/check_source_authority.py"
)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class SourceAuthorityTests(unittest.TestCase):
    def test_portable_benchmark_entrypoints_use_canonical_source(self) -> None:
        ledger = json.loads(
            (
                ROOT
                / "docs/architecture/migrations/portable-benchmark-source-entrypoints-2026-11-28.json"
            ).read_text(encoding="utf-8")
        )
        policy = CHECKER.load_policy()
        self.assertEqual(len(ledger["entries"]), 8)
        self.assertEqual(
            policy["historical_tree_oid"], ledger["source_authority"]["unchanged_frozen_tree"]
        )
        for row in ledger["entries"]:
            with self.subTest(path=row["new_path"]):
                path = ROOT / row["new_path"]
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(row["new_path"], policy["allowed_active_references"])
                self.assertNotIn(row["old_source_path"], source)
                depth = row["canonical_parent_depth"]
                self.assertEqual(path.resolve().parents[depth] / "src", ROOT / "src")
                self.assertIn(f'Path(__file__).resolve().parents[{depth}] / "src"', source)
        receipts = ledger["compatibility"]["actual_receipts"]
        self.assertEqual(
            {row["path"] for row in receipts}, {row["new_path"] for row in ledger["entries"]}
        )
        for receipt in receipts:
            self.assertEqual(receipt["new_isolated_help_exit"], 0)
            self.assertTrue(receipt["help_byte_identical"])
            self.assertEqual(receipt["actual_imported_package"], "src/tensorlbm/__init__.py")
            self.assertFalse(receipt["main_invoked"])
        self.assertEqual(ledger["science"]["current_acceptance"], "not_evaluated")
        self.assertFalse(ledger["science"]["algorithm_changed"])

    def test_split_path_forms_are_detected(self) -> None:
        checkout, source = CHECKER.TOKEN.split("/")
        examples = [
            f"path = repo / {checkout!r} / {source!r}",
            f"path = Path(repo, {checkout!r}, {source!r})",
            f"path = repo.joinpath({checkout!r}, {source!r})",
            f"path = os.path.join(repo, {checkout!r}, {source!r})",
        ]
        for line in examples:
            with self.subTest(line=line):
                self.assertEqual(CHECKER.matching_reference_lines(line, Path("run.py")), [line])
        split = f"path = os.path.join(\n    repo, {checkout!r},\n    {source!r}\n)"
        self.assertEqual(
            CHECKER.matching_reference_lines(split, Path("run.py")),
            split.splitlines()[1:3],
        )
        prose = repr(examples[0])
        self.assertEqual(CHECKER.matching_reference_lines(prose, Path("test_run.py")), [])

    def test_repository_matches_source_authority_policy(self) -> None:
        self.assertEqual(CHECKER.validate(CHECKER.load_policy()), [])

    def test_unregistered_lbm_file_is_rejected(self) -> None:
        policy = CHECKER.load_policy()
        path = "src/lbm/collision/d3q27.py"
        policy["legacy_import_shims"].pop(path)
        self.assertIn(
            f"unexpected file in the lbm compatibility package: {path}", CHECKER.validate(policy)
        )

    def test_changed_lbm_shim_is_rejected(self) -> None:
        policy = CHECKER.load_policy()
        path = "src/lbm/boundary/spherical_force.py"
        policy["legacy_import_shims"][path] = "0" * 64
        self.assertIn(f"lbm compatibility shim changed: {path}", CHECKER.validate(policy))

    def test_checker_does_not_register_its_own_policy_literal(self) -> None:
        policy = CHECKER.load_policy()
        self.assertNotIn(
            "tools/check_source_authority.py", CHECKER.collect_active_references(policy)
        )

    def test_new_active_nested_consumer_is_rejected(self) -> None:
        policy = CHECKER.load_policy()
        references = policy["allowed_active_references"]
        path = next(iter(references))
        references.pop(path)
        self.assertIn(f"new active nested-source reference: {path}", CHECKER.validate(policy))

    def test_reference_line_drift_is_rejected(self) -> None:
        policy = CHECKER.load_policy()
        path, reference = next(iter(policy["allowed_active_references"].items()))
        reference["matching_lines_sha256"] = "0" * 64
        self.assertTrue(
            any(
                "reference lines changed" in error and path in error
                for error in CHECKER.validate(policy)
            )
        )


if __name__ == "__main__":
    unittest.main()
