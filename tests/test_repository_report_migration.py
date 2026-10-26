"""Guard historical report relocation without importing a numerical runtime.

Hashes refer to the pre-migration checkout. Passing these checks preserves
provenance; it does not validate the scientific claims in the archived reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO / "docs/architecture/migrations/report-assets-2026-10-26.json"
NOTE_START = "<!-- repository-layout-note:start -->"
NOTE_END = "<!-- repository-layout-note:end -->"
NOTE = re.compile(r"\n" + re.escape(NOTE_START) + r"\n.*?" + re.escape(NOTE_END) + r"\n", re.S)


class RepositoryReportMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))

    def assert_fingerprint(self, path: Path, sha256: str, size: int):
        raw = path.read_bytes()
        self.assertEqual(len(raw), size, str(path))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), sha256, str(path))

    def test_report_bodies_unchanged_except_declared_links_and_note(self):
        for item in self.ledger["reports"]:
            with self.subTest(report=item["new_path"]):
                text = (REPO / item["new_path"]).read_text(encoding="utf-8")
                self.assertEqual(text.count(NOTE_START), 1)
                self.assertEqual(text.count(NOTE_END), 1)
                text, count = NOTE.subn("", text)
                self.assertEqual(count, 1)
                for old, new in item["link_rewrites"].items():
                    self.assertEqual(text.count(f"]({new})"), 1, new)
                    text = text.replace(f"]({new})", f"]({old})")
                raw = text.encode("utf-8")
                self.assertEqual(len(raw), item["original_bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["original_sha256"])
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_deduplicated_images_retain_original_bytes(self):
        for item in self.ledger["deduplicated_images"]:
            with self.subTest(image=item["old_path"]):
                self.assert_fingerprint(
                    REPO / item["canonical_path"], item["sha256"], item["bytes"]
                )
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_archived_evidence_and_couette_images_are_unchanged(self):
        for item in self.ledger["preserved_files"]:
            with self.subTest(path=item["path"]):
                self.assert_fingerprint(REPO / item["path"], item["sha256"], item["bytes"])

    def test_report_and_case_inline_local_links_resolve(self):
        # Deliberately scoped: this is not a repository-wide Markdown parser,
        # remote URL check, anchor check or a check of external /nfs provenance.
        paths = [item["new_path"] for item in self.ledger["reports"]] + [
            "benchmarks/verified/couette_3d/README.md",
            "benchmarks/pending/capillary_invasion_washburn/README.md",
        ]
        for name in paths:
            path = REPO / name
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)]+)\)", text):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                with self.subTest(document=name, target=target):
                    resolved = (path.parent / unquote(parsed.path)).resolve()
                    self.assertTrue(resolved.is_relative_to(REPO))
                    self.assertTrue(resolved.exists(), f"Missing link: {name} -> {target}")

    def test_docs_index_and_benchmark_backlinks_point_to_archived_reports(self):
        index = (REPO / "docs/README.md").read_text(encoding="utf-8")
        cases = ["verified/couette_3d", "pending/capillary_invasion_washburn"]
        for item, case in zip(self.ledger["reports"], cases, strict=True):
            name = Path(item["new_path"]).name
            with self.subTest(report=name):
                self.assertIn(f"](reports/{name})", index)
                case_readme = (REPO / "benchmarks" / case / "README.md").read_text(encoding="utf-8")
                self.assertIn(f"](../../../docs/reports/{name})", case_readme)

    def test_migrations_registered_without_reintroducing_root_exceptions(self):
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        paths = [row["old_path"] for row in rows]
        self.assertEqual(len(paths), len(set(paths)))
        manifest = {row["old_path"]: row for row in rows}
        policy = json.loads((REPO / "docs/architecture/repository_layout_policy.json").read_text())
        roots = set(policy["root_files"]) | set(policy["legacy_root_files"])
        for item in self.ledger["reports"] + self.ledger["deduplicated_images"]:
            old = item["old_path"]
            with self.subTest(old_path=old):
                self.assertNotIn(old, roots)
                expected = item.get("new_path", item.get("canonical_path"))
                self.assertEqual(manifest[old]["new_path"], expected)
                self.assertIn(
                    manifest[old]["status"],
                    {"migrated_pending_review", "deduplicated_pending_review", "accepted"},
                )


if __name__ == "__main__":
    unittest.main()
