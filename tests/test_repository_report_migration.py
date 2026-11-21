"""Guard historical report relocation without importing a numerical runtime.

Hashes refer to the pre-migration checkout. Passing these checks preserves
provenance; it does not validate the scientific claims in the archived reports.
"""

from __future__ import annotations

import ast

import csv
import hashlib
import json
import re
import subprocess
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
                fingerprint = item.get("current_fingerprint", {
                    "bytes": item["original_bytes"], "sha256": item["original_sha256"],
                })
                self.assertEqual(len(raw), fingerprint["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), fingerprint["sha256"])
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
                fingerprint = item.get("current_fingerprint", item)
                self.assert_fingerprint(
                    REPO / item["path"], fingerprint["sha256"], fingerprint["bytes"]
                )

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

    def test_m26816_archive_preserves_root_assets(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "m26816-root-retirement-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        for item in ledger["entries"]:
            destination = REPO / item["new_path"]
            with self.subTest(path=destination):
                self.assert_fingerprint(destination, item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], item["new_path"])
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_case_pool_entries_keep_compatibility_wrappers(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "case-pool-entrypoints-2026-11-21.json").read_text()
        )
        for item in ledger["entries"]:
            canonical = REPO / item["new_path"]
            with self.subTest(runner=canonical):
                wrapper = ast.parse((REPO / item["old_path"]).read_text(encoding="utf-8"))
                target = canonical.relative_to(REPO).with_suffix("")
                self.assertIn(
                    ".".join(target.parts),
                    [node.module for node in wrapper.body if isinstance(node, ast.ImportFrom)],
                )
                index = canonical.with_name("README.md")
                text = index.read_text(encoding="utf-8")
                self.assertIn(canonical.relative_to(REPO).as_posix(), text)
                for link in re.findall(r"\]\(([^)]+)\)", text):
                    self.assertTrue((index.parent / link).is_file(), link)

    def test_development_archive_pilot_preserves_linked_evidence(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "development-archive-first-batch.json").read_text()
        )
        campaign = REPO / "xval/XVAL-LBM-GRAPH-UNROLL-BPTT-LES-PARAMETER/README.md"
        text = campaign.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)]+)\)", text)
        resolved = {(campaign.parent / target).resolve() for target in targets}
        for target in resolved:
            self.assertTrue(target.is_file(), str(target))
        for item in ledger["entries"]:
            destination = REPO / item["new_path"]
            with self.subTest(path=destination):
                self.assert_fingerprint(destination, item["sha256"], item["size_bytes"])
                self.assertIn(destination, resolved)
                self.assertFalse((REPO / item["old_path"]).exists())

    def test_sod_figures_and_m26765_publisher_preserve_assets(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "sod-figures-and-m26765-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        report_path = REPO / "docs/report_m40.md"
        report = report_path.read_text(encoding="utf-8")
        for item in ledger["entries"]:
            destination = REPO / item["new_path"]
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(destination, item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], item["new_path"])
                self.assertFalse((REPO / item["old_path"]).exists())
                if destination.suffix == ".png":
                    target = destination.relative_to(report_path.parent).as_posix()
                    self.assertIn(f"]({target})", report)
        index = REPO / "audit/m26765/README.md"
        for target in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
            with self.subTest(link=target):
                self.assertTrue((index.parent / target).is_file())

    def test_audit_records_retain_bytes_and_sibling_evidence(self):
        ledger = json.loads((LEDGER_PATH.parent / "audit-records-2026-11-21.json").read_text())
        for move in ledger["entries"]:
            with self.subTest(path=move["new_path"]):
                self.assert_fingerprint(REPO / move["new_path"], move["sha256"], move["size_bytes"])
        report = REPO / "audit/m18313/audit_m18313.md"
        self.assertIn("numbers_from_artifact.json", report.read_text())
        self.assertTrue(report.with_name("numbers_from_artifact.json").is_file())

    def test_november_root_migration_destinations(self):
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            moves = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if "nov-2026 root migration" in row["verification"]
            }
        self.assertIn("bench_t4_medium.py", moves)
        self.assertIn("audit_m18221", moves)
        self.assertIn("verdict_m15215.sql", moves)
        for old, new in moves.items():
            with self.subTest(old=old, new=new):
                # Local runners may recreate ignored scratch directories.
                # Their tracked descendants, rather than filesystem presence,
                # are governed by the repository layout checker.
                self.assertTrue((REPO / new).exists())
                if (REPO / new).is_file():
                    self.assertFalse((REPO / old).is_file())

    def test_wave_one_archive_preserves_source_files_and_both_report_versions(self):
        ledger = json.loads((REPO / "docs/architecture/migrations/m26711-2026-11-21.json").read_text())
        self.assertEqual(len(ledger["entries"]), 23)
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                blob = (REPO / item["new_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(blob).hexdigest(), item["sha256"])
                self.assertEqual(len(blob), item["size_bytes"])
                self.assertEqual(
                    subprocess.check_output(
                        ["git", "hash-object", item["new_path"]],
                        cwd=REPO,
                        text=True,
                    ).strip(),
                    item["git_blob"],
                )
                index_entry = subprocess.check_output(
                    ["git", "ls-files", "-s", "--", item["new_path"]],
                    cwd=REPO,
                    text=True,
                ).split()
                self.assertEqual(index_entry[0], item["git_mode"])
                self.assertFalse((REPO / item["old_path"]).exists())
        for mission in ledger["report_missions"]:
            report = json.loads((REPO / f"audit/m{mission}/report.json").read_text())
            self.assertEqual(int(report["mission_id"]), mission)

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

    def test_legacy_audits_are_preserved_under_missions(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "legacy-audit-root-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        for item in ledger["entries"]:
            destination = item["new_path"]
            with self.subTest(path=destination):
                self.assert_fingerprint(REPO / destination, item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], destination)
                self.assertFalse((REPO / item["old_path"]).exists())
        for relative in ("audit/m17444/README.md", "audit/m18155/README.md"):
            index = REPO / relative
            for target in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
                with self.subTest(index=relative, link=target):
                    self.assertTrue((index.parent / target).is_file())

    def test_water_entry_report_layout_preserves_assets_and_index_links(self):
        ledger = json.loads(
            (LEDGER_PATH.parent / "water-entry-report-layout-2026-11-21.json").read_text()
        )
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {row["old_path"]: row["new_path"] for row in csv.DictReader(stream)}
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                destination = item["new_path"]
                self.assert_fingerprint(REPO / destination, item["sha256"], item["size_bytes"])
                self.assertEqual(manifest[item["old_path"]], destination)
                self.assertFalse((REPO / item["old_path"]).exists())
        index = REPO / "docs/reports/water_entry/README.md"
        for target in re.findall(r"\]\(([^)]+)\)", index.read_text(encoding="utf-8")):
            with self.subTest(link=target):
                self.assertTrue((index.parent / target).exists())


    def test_hidden_audit_root_preserves_replay_files(self):
        ledger_path = LEDGER_PATH.parent / "hidden-audit-root-2026-11-21.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with (REPO / "docs/architecture/migration_manifest.csv").open(encoding="utf-8") as stream:
            manifest = {
                row["old_path"]: row["new_path"]
                for row in csv.DictReader(stream)
                if row["verification"] == ledger_path.relative_to(REPO).as_posix()
            }
        self.assertEqual(manifest, {item["old_path"]: item["new_path"] for item in ledger["entries"]})
        for item in ledger["entries"]:
            with self.subTest(path=item["new_path"]):
                self.assert_fingerprint(REPO / item["new_path"], item["sha256"], item["size_bytes"])
                self.assertFalse((REPO / item["old_path"]).exists())
        audit_links = {
            "docs/board_audit/m23219/AUDIT-m23126.md":
                "../../../development/audit/m23219/repro/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/",
            "docs/board_audit_m15893/M15893_BOARD_AUDIT.md":
                "../../development/audit/m15893/repro/",
        }
        for name, target in audit_links.items():
            with self.subTest(audit=name):
                doc = REPO / name
                self.assertIn(f"]({target})", doc.read_text(encoding="utf-8"))
                self.assertTrue((doc.parent / target).is_dir())


if __name__ == "__main__":
    unittest.main()
